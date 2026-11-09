#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Replace presentation-cropped wrist insets with exact saved 224px policy OBS.

The overview pixels and rollout timing are retained from the existing video.
Only the two declared wrist ROIs are replaced.  Each saved 224x224 RGB policy
tensor is resized as a complete canvas to the historical 180x135 video slot, so
its real OpenPI black letterbox remains visible.

Two kinds of run are supported, because both can end up as grid inputs:

``rerender``
    The post-hoc renderer already republished ``<scene>_<run>.mp4`` and
    describes it in ``rerender/sim_summary.json``.  That summary is updated in
    place, exactly as before.

``generation``
    The rollout burned its own wrist insets during generation (``--policy-obs
    -inset``), so ``<run>/sim_summary.json`` is the generation record and must
    stay untouched.  The republished video is instead described by a new
    ``<run>/saved_policy_obs/`` sidecar, which the grid renderer reads as the
    summary of the *published* video.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import cv2

from robosuite_env.rollout.outputs import open_video_capture
from robosuite_env.rollout.replay_policy_insets import (
    ReplayPolicyInsetError,
    load_replay_policy_insets,
)
from robosuite_env.tools.make_rollout_grid_video import (
    SAVED_POLICY_OBS_DIRNAME,
)
from robosuite_env.tools.validate_rollout_video import (
    validate_saved_wrist_insets,
    validate_video,
    write_json_atomic,
)


RAW_DISPLAY_CONTRACT = "raw_policy_tensor_full_canvas_resize_v1"
# Wrist-inset sources this tool can republish. The overlay replaces the two
# declared ROIs wholesale, so a presentation crop of the policy tensor and a
# second live wrist render are both repairable. Only an undeclared source is
# refused, because then the ROI geometry cannot be trusted.
REWRITABLE_WRIST_SOURCES = ("saved", "policy_observation", "rendered")
RAW_LAYOUT = {
    "version": 3,
    "left_xywh": [4, 4, 180, 135],
    "right_xywh": [188, 4, 180, 135],
}


class RewriteError(RuntimeError):
    """An existing rollout cannot be safely rewritten."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_state(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "mode": int(stat.st_mode),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "state": _file_state(resolved),
    }


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RewriteError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RewriteError(f"{label} must be a JSON object: {path}")
    return value


def _load_state_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RewriteError(
                        f"state log line {line_number} is not an object: {path}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise RewriteError(f"cannot read state log {path}: {exc}") from exc
    if not rows:
        raise RewriteError(f"state log is empty: {path}")
    return rows


def _raw_transform() -> dict[str, Any]:
    return {
        "algorithm": RAW_DISPLAY_CONTRACT,
        "resize_mode": "raw224",
        "source_hw": [224, 224],
        "policy_hw": [224, 224],
        "policy_content_xywh": [0, 0, 224, 224],
        "output_hw": [135, 180],
        "output_content_xywh": [0, 0, 180, 135],
    }


def _raw_summary(summary: dict[str, Any]) -> dict[str, Any]:
    obs_gs_only = summary.get("obs_gs_only", False)
    if not isinstance(obs_gs_only, bool):
        raise RewriteError(
            f"saved policy OBS summary has invalid obs_gs_only={obs_gs_only!r}"
        )
    if obs_gs_only:
        raise RewriteError(
            "GS-only diagnostic observations cannot be published as saved "
            "policy OBS"
        )
    source = summary.get("video_wrist_inset_source")
    if source not in REWRITABLE_WRIST_SOURCES:
        raise RewriteError("rerender summary is not a saved-policy-OBS video")
    updated = copy.deepcopy(summary)
    # The republished video's wrist pixels are read back from the saved PNGs,
    # whatever produced them during generation. Keep the generation record so
    # the change of source stays auditable.
    updated["video_wrist_inset_source"] = "saved"
    updated["video_wrist_inset_source_generated"] = source
    updated["video_wrist_inset_layout"] = copy.deepcopy(RAW_LAYOUT)
    updated["video_wrist_display_contract"] = RAW_DISPLAY_CONTRACT
    updated["video_wrist_display_transforms"] = {
        side: _raw_transform() for side in ("L", "R")
    }
    updated["state_replay_policy_obs_display"] = "raw224"
    return updated


def _is_published_raw224(
    video_path: Path,
    summary: dict[str, Any],
    validation_path: Path,
) -> bool:
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        return False
    obs_gs_only = summary.get("obs_gs_only", False)
    if not isinstance(obs_gs_only, bool) or obs_gs_only:
        return False
    if summary.get("video_wrist_display_contract") != RAW_DISPLAY_CONTRACT:
        return False
    if summary.get("video_wrist_inset_layout") != RAW_LAYOUT:
        return False
    if not validation_path.is_file():
        return False
    try:
        validation = _load_json_object(validation_path, label="video validation")
    except RewriteError:
        return False
    exact = validation.get("saved_wrist_exact_match")
    return bool(
        validation.get("ok") is True
        and isinstance(exact, dict)
        and exact.get("ok") is True
        and exact.get("layout_version") == 3
    )


def _temporary_path(parent: Path, stem: str, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{stem}.", suffix=suffix, dir=parent
    )
    os.close(descriptor)
    return Path(name)


def _compose_raw224_video(
    source_video: Path,
    output_video: Path,
    raw_display_frames,
    *,
    sim_panel_x_offset: int,
) -> dict[str, Any]:
    capture = open_video_capture(source_video)
    if not capture.isOpened():
        raise RewriteError(f"OpenCV could not open video: {source_video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if not math.isfinite(fps) or fps <= 0.0:
        capture.release()
        raise RewriteError(f"video has invalid FPS {fps}: {source_video}")
    right_edge = sim_panel_x_offset + 188 + 180
    if width < right_edge or height < 139:
        capture.release()
        raise RewriteError(
            f"video {width}x{height} is too small for saved wrist ROIs"
        )

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RewriteError(f"OpenCV could not create video: {output_video}")

    expected = tuple(raw_display_frames)
    frame_count = 0
    bgr_cache: dict[int, tuple[Any, Any]] = {}
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_count >= len(expected):
                raise RewriteError(
                    f"video has more frames than saved OBS schedule ({len(expected)})"
                )
            pair = expected[frame_count]
            cache_key = id(pair)
            pair_bgr = bgr_cache.get(cache_key)
            if pair_bgr is None:
                pair_bgr = tuple(
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in pair
                )
                bgr_cache[cache_key] = pair_bgr
            left_x = sim_panel_x_offset + 4
            right_x = sim_panel_x_offset + 188
            frame_bgr[4:139, left_x : left_x + 180] = pair_bgr[0]
            frame_bgr[4:139, right_x : right_x + 180] = pair_bgr[1]
            writer.write(frame_bgr)
            frame_count += 1
    finally:
        capture.release()
        writer.release()
    if frame_count != len(expected):
        raise RewriteError(
            f"video decoded {frame_count} frames but saved OBS schedule has "
            f"{len(expected)}"
        )
    if not output_video.is_file() or output_video.stat().st_size <= 0:
        raise RewriteError(f"rewritten video is empty: {output_video}")
    return {
        "frames": frame_count,
        "fps": fps,
        "size_wh": [width, height],
        "unique_display_pairs": len(bgr_cache),
    }


def _resolve_publication(run_dir: Path) -> tuple[str, Path, Path]:
    """Pick which summary describes the published video for this run.

    A post-hoc rendered run already republished its video and owns
    ``rerender/sim_summary.json``. A run that rendered inline still has only
    its generation summary, which must not be rewritten, so the republished
    video is described by a ``saved_policy_obs/`` sidecar instead.
    """

    rerender_summary = run_dir / "rerender" / "sim_summary.json"
    if rerender_summary.is_file():
        return "rerender", run_dir / "rerender", rerender_summary
    return (
        "generation",
        run_dir / SAVED_POLICY_OBS_DIRNAME,
        run_dir / "sim_summary.json",
    )


def rewrite_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    scene = run_dir.parent.name
    video_path = run_dir / f"{scene}_{run_dir.name}.mp4"
    state_log = run_dir / "logs" / "sim_log.jsonl"
    policy_obs_dir = run_dir / "logs" / "policy_obs"
    mode, publication_dir, source_summary_path = _resolve_publication(run_dir)
    summary_path = publication_dir / "sim_summary.json"
    validation_path = publication_dir / "video_validation.json"
    for path, label in (
        (video_path, "video"),
        (source_summary_path, f"{mode} summary"),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RewriteError(f"missing {label}: {path}")

    published_summary_path = (
        summary_path if summary_path.is_file() else source_summary_path
    )
    published_summary = _load_json_object(
        published_summary_path, label=f"{mode} summary"
    )
    if _is_published_raw224(video_path, published_summary, validation_path):
        return {
            "ok": True,
            "skipped": True,
            "mode": mode,
            "run": run_dir.name,
            "video": str(video_path),
            "summary": str(summary_path),
            "contract": RAW_DISPLAY_CONTRACT,
        }
    summary = _load_json_object(source_summary_path, label=f"{mode} summary")
    if not summary.get("video_wrist_insets_enabled", True):
        # A clean or wrist-free video carries no OBS pixels at all, so it can
        # never misrepresent the policy input and needs no republish.
        return {
            "ok": True,
            "skipped": True,
            "reason": "wrist_insets_disabled",
            "mode": mode,
            "run": run_dir.name,
            "video": str(video_path),
            "summary": str(source_summary_path),
            "contract": None,
        }
    if not state_log.is_file() or state_log.stat().st_size <= 0:
        raise RewriteError(f"missing state log: {state_log}")
    publication_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_state_rows(state_log)
    try:
        schedule = load_replay_policy_insets(rows, policy_obs_dir)
    except ReplayPolicyInsetError as exc:
        raise RewriteError(f"invalid saved policy OBS schedule: {exc}") from exc
    rendered_frames = summary.get("rendered_frames")
    if (
        isinstance(rendered_frames, bool)
        or not isinstance(rendered_frames, int)
        or rendered_frames != len(schedule.raw_display_frames)
    ):
        raise RewriteError(
            f"{mode} summary frame count differs from saved OBS schedule: "
            f"{rendered_frames!r} != {len(schedule.raw_display_frames)}"
        )
    sim_offset = summary.get("video_sim_panel_x_offset")
    if isinstance(sim_offset, bool) or not isinstance(sim_offset, int) or sim_offset < 0:
        raise RewriteError(f"invalid video_sim_panel_x_offset={sim_offset!r}")

    base_video_sha256 = _sha256(video_path)
    candidate_video = _temporary_path(video_path.parent, video_path.stem, ".mp4")
    candidate_summary = _temporary_path(publication_dir, "sim_summary", ".json")
    try:
        composition = _compose_raw224_video(
            video_path,
            candidate_video,
            schedule.raw_display_frames,
            sim_panel_x_offset=sim_offset,
        )
        updated_summary = _raw_summary(summary)
        candidate_summary.write_text(
            json.dumps(updated_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validation = validate_video(candidate_video, candidate_summary)
        validation["saved_wrist_exact_match"] = validate_saved_wrist_insets(
            candidate_video,
            candidate_summary,
            schedule.raw_display_frames,
            observation_frames=schedule.observation_frames,
        )
        validation["raw224_policy_obs_overlay"] = {
            "contract": RAW_DISPLAY_CONTRACT,
            "base_video_sha256": base_video_sha256,
            **composition,
        }

        os.replace(candidate_video, video_path)
        write_json_atomic(summary_path, updated_summary)
        validation["video"] = str(video_path)
        validation["summary"] = str(summary_path)
        validation["file_size_bytes"] = video_path.stat().st_size
        validation["published_summary_sha256"] = _sha256(summary_path)
        if mode == "rerender":
            validation["rerender_summary_sha256"] = validation[
                "published_summary_sha256"
            ]
        write_json_atomic(validation_path, validation)

        provenance_path = publication_dir / "provenance.json"
        provenance = (
            _load_json_object(provenance_path, label=f"{mode} provenance")
            if provenance_path.is_file()
            else {
                "schema_version": 3,
                "kind": (
                    "fr3-state-log-video-rerender"
                    if mode == "rerender"
                    else "fr3-generated-video-saved-policy-obs-overlay"
                ),
            }
        )
        provenance["created_at"] = datetime.now(timezone.utc).isoformat()
        provenance["mode"] = (
            "state-log-replay+raw224-policy-obs-overlay"
            if mode == "rerender"
            else "generated-video+raw224-policy-obs-overlay"
        )
        provenance["postprocess"] = {
            "contract": RAW_DISPLAY_CONTRACT,
            "base_video_sha256": base_video_sha256,
            "source_summary": _file_record(source_summary_path),
            "state_log": _file_record(state_log),
            "tool": _file_record(Path(__file__)),
            "saved_wrist_schedule_sha256": validation[
                "saved_wrist_exact_match"
            ]["expected_schedule_sha256"],
        }
        provenance["output"] = {
            "decoded_frames": validation["decoded_frames"],
            "decoded_sha256": validation["decoded_sha256"],
            "video": _file_record(video_path),
            "published_summary": _file_record(summary_path),
            "validation": _file_record(validation_path),
        }
        write_json_atomic(provenance_path, provenance)
    finally:
        for temporary in (candidate_video, candidate_summary):
            if temporary.exists():
                temporary.unlink()

    return {
        "ok": True,
        "skipped": False,
        "mode": mode,
        "run": run_dir.name,
        "video": str(video_path),
        "summary": str(summary_path),
        "video_sha256": _sha256(video_path),
        "contract": RAW_DISPLAY_CONTRACT,
        **composition,
    }


def _selected_runs(run_root: Path, scene: str, runs_csv: str) -> list[Path]:
    scene_dir = run_root.expanduser().resolve() / scene
    if not scene_dir.is_dir():
        raise RewriteError(f"scene directory does not exist: {scene_dir}")
    if runs_csv:
        names = runs_csv.split(",")
        if any(re.fullmatch(r"run_\d+", name) is None for name in names):
            raise RewriteError("--runs must contain comma-separated run_<integer> names")
        if len(set(names)) != len(names):
            raise RewriteError("--runs contains duplicate entries")
        runs = [scene_dir / name for name in names]
    else:
        runs = sorted(scene_dir.glob("run_*"))
    missing = [path for path in runs if not path.is_dir()]
    if missing:
        raise RewriteError(f"run directories do not exist: {missing}")
    if not runs:
        raise RewriteError(f"no runs selected under {scene_dir}")
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--runs",
        default="",
        help="comma-separated run directory names; default rewrites every run",
    )
    arguments = parser.parse_args()
    try:
        runs = _selected_runs(arguments.run_root, arguments.scene, arguments.runs)
        reports = [rewrite_run(run_dir) for run_dir in runs]
    except (OSError, RewriteError, ValueError) as exc:
        print(f"[raw224-obs] FAILED {exc}", file=sys.stderr, flush=True)
        return 1
    report = {
        "ok": True,
        "contract": RAW_DISPLAY_CONTRACT,
        "runs": reports,
        "rewritten": sum(not item["skipped"] for item in reports),
        "skipped": sum(item["skipped"] for item in reports),
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
