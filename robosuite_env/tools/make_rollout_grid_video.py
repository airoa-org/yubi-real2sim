#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Render rollout videos through a required, fail-closed layout template."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Sequence

import cv2
import numpy as np

from robosuite_env.rollout.outputs import (
    _resolve_ffmpeg,
    _transcode_to_h264,
    open_video_capture,
)
from robosuite_env.rollout.replay_policy_insets import (
    ReplayPolicyInsetError,
    load_replay_policy_insets,
)
from robosuite_env.tools.validate_rollout_video import (
    VideoValidationError,
    validate_saved_wrist_insets,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_PATH = (
    REPO_ROOT
    / "config"
    / "render_templates"
    / "rollout_grid_clear_headers_4x3.json"
)


class TemplateError(ValueError):
    """The grid template or one of its required inputs is invalid."""


GRID_CONTENT_VALIDATION_THRESHOLDS = {
    "mean_abs_error_max": 8.0,
    "frame_mean_abs_error_max": 12.0,
    "coarse_block_mean_abs_error_max": 35.0,
    "fraction_abs_error_gt_32_max": 0.08,
}

# A grid tiles rollout videos that already carry burned-in wrist OBS insets.
# Those inset pixels are the evidence of what the policy actually saw, so they
# must be the exact saved 224x224 policy tensors read back from
# ``<run>/logs/policy_obs``. Re-rendering the wrist camera for presentation, or
# re-cropping the policy tensor to hide its OpenPI letterbox, produces pixels
# the policy never received and is therefore forbidden in a grid input.
SAVED_POLICY_OBS_DIRNAME = "saved_policy_obs"
SAVED_WRIST_DISPLAY_CONTRACT = "raw_policy_tensor_full_canvas_resize_v1"
SAVED_WRIST_INSET_LAYOUT_VERSION = 3
SAVED_WRIST_INSET_SOURCE = "saved"
GRID_SAVED_POLICY_OBS_CONTRACT = "grid_inputs_embed_exact_saved_policy_obs_v1"
REWRITE_TOOL_HINT = (
    "python -m robosuite_env.tools.rewrite_saved_policy_obs_video "
    "<run_root> --scene <scene>"
)


def _file_state(path: Path) -> dict[str, int]:
    observed = path.stat()
    if not stat.S_ISREG(observed.st_mode):
        raise RuntimeError(f"provenance input is not a regular file: {path}")
    return {
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "mode": int(observed.st_mode),
        "size": int(observed.st_size),
        "mtime_ns": int(observed.st_mtime_ns),
        "ctime_ns": int(observed.st_ctime_ns),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    """Hash one stable regular file and retain its filesystem identity."""

    resolved = path.expanduser().resolve()
    try:
        before = _file_state(resolved)
        digest = _sha256_file(resolved)
        after = _file_state(resolved)
    except OSError as exc:
        raise RuntimeError(f"could not fingerprint {resolved}: {exc}") from exc
    if before != after:
        raise RuntimeError(f"file changed while it was fingerprinted: {resolved}")
    return {"path": str(resolved), "sha256": digest, "state": after}


def _read_stable_file(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Read bytes that are cryptographically identical to the returned record."""

    resolved = path.expanduser().resolve()
    try:
        before = _file_state(resolved)
        raw = resolved.read_bytes()
        after = _file_state(resolved)
    except OSError as exc:
        raise RuntimeError(f"could not read {resolved}: {exc}") from exc
    if before != after:
        raise RuntimeError(f"file changed while it was read: {resolved}")
    return raw, {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "state": after,
    }


def _assert_record_current(record: dict[str, Any], *, label: str) -> None:
    """Fail if a recorded source was replaced or changed after capture."""

    current = _file_record(Path(record["path"]))
    if current["state"] != record["state"] or current["sha256"] != record["sha256"]:
        raise RuntimeError(f"{label} changed during grid rendering: {record['path']}")


def _exact_keys(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TemplateError(f"{path} must be a JSON object")
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise TemplateError(f"{path} has invalid keys: {' '.join(details)}")
    return value


def _positive_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TemplateError(f"{path} must be a positive integer")
    return int(value)


def _number(value: Any, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TemplateError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise TemplateError(f"{path} must be finite and >= {minimum}")
    return result


def _int_list(value: Any, length: int, path: str) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise TemplateError(f"{path} must contain exactly {length} integers")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TemplateError(f"{path}[{index}] must be an integer")
        result.append(int(item))
    return result


def _color(value: Any, path: str) -> list[int]:
    result = _int_list(value, 3, path)
    if any(channel < 0 or channel > 255 for channel in result):
        raise TemplateError(f"{path} channels must be in [0, 255]")
    return result


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TemplateError(f"{path} must be a non-empty string")
    return value


def _rectangles_overlap(a: Sequence[int], b: Sequence[int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def load_template(path: Path) -> dict[str, Any]:
    """Load and strictly validate the sole source of grid layout geometry."""

    resolved = path.expanduser().resolve()
    try:
        raw, template_record = _read_stable_file(resolved)
    except (OSError, RuntimeError) as exc:
        raise TemplateError(f"could not read grid template {resolved}: {exc}") from exc
    try:
        template = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TemplateError(f"grid template is not valid JSON: {exc}") from exc

    allowed_keys = {
        "schema_version",
        "id",
        "description",
        "video_glob",
        "expected_videos",
        "input_frame_wh",
        "fps_mode",
        "poster_frame_mode",
        "codec_preference",
        "geometry",
        "metrics",
        "style",
    }
    if "summary_content" in template:
        allowed_keys.add("summary_content")
    template = _exact_keys(template, allowed_keys, "template")
    if template["schema_version"] != 1:
        raise TemplateError("template.schema_version must be 1")
    summary_content = template.get("summary_content", "legacy")
    if summary_content not in ("legacy", "policy_eval"):
        raise TemplateError(
            "template.summary_content must be 'legacy' or 'policy_eval'"
        )
    template["summary_content"] = summary_content
    _nonempty_string(template["id"], "template.id")
    _nonempty_string(template["description"], "template.description")
    _nonempty_string(template["video_glob"], "template.video_glob")
    expected_videos = _positive_integer(
        template["expected_videos"], "template.expected_videos"
    )
    input_width, input_height = _int_list(
        template["input_frame_wh"], 2, "template.input_frame_wh"
    )
    if input_width <= 0 or input_height <= 0:
        raise TemplateError("template.input_frame_wh values must be positive")
    if template["fps_mode"] != "source":
        raise TemplateError("template.fps_mode must be 'source'")
    if template["poster_frame_mode"] != "middle":
        raise TemplateError("template.poster_frame_mode must be 'middle'")
    codecs = template["codec_preference"]
    if (
        not isinstance(codecs, list)
        or not codecs
        or any(not isinstance(codec, str) or len(codec) != 4 for codec in codecs)
    ):
        raise TemplateError(
            "template.codec_preference must contain one or more four-character codecs"
        )

    geometry = _exact_keys(
        template["geometry"],
        {
            "canvas_wh",
            "columns",
            "rows",
            "rollout_cells",
            "cell_wh",
            "header_height",
            "source_xywh_within_cell",
            "summary_xywh",
        },
        "template.geometry",
    )
    canvas_width, canvas_height = _int_list(
        geometry["canvas_wh"], 2, "template.geometry.canvas_wh"
    )
    cell_width, cell_height = _int_list(
        geometry["cell_wh"], 2, "template.geometry.cell_wh"
    )
    columns = _positive_integer(geometry["columns"], "template.geometry.columns")
    rows = _positive_integer(geometry["rows"], "template.geometry.rows")
    rollout_cells = _positive_integer(
        geometry["rollout_cells"], "template.geometry.rollout_cells"
    )
    header_height = _positive_integer(
        geometry["header_height"], "template.geometry.header_height"
    )
    source_rect = _int_list(
        geometry["source_xywh_within_cell"],
        4,
        "template.geometry.source_xywh_within_cell",
    )
    summary_rect = _int_list(
        geometry["summary_xywh"], 4, "template.geometry.summary_xywh"
    )
    if min(canvas_width, canvas_height, cell_width, cell_height) <= 0:
        raise TemplateError("canvas and cell dimensions must be positive")
    if canvas_width % 2 or canvas_height % 2:
        raise TemplateError("template canvas dimensions must be even for MP4")
    if canvas_width != columns * cell_width or canvas_height != rows * cell_height:
        raise TemplateError("template grid cells must exactly tile the canvas")
    if rollout_cells != expected_videos:
        raise TemplateError(
            "template.geometry.rollout_cells must equal template.expected_videos"
        )
    if rollout_cells >= columns * rows:
        raise TemplateError("template must reserve at least one non-rollout summary cell")
    sx, sy, sw, sh = source_rect
    if (
        sx < 0
        or sy < header_height
        or sw <= 0
        or sh <= 0
        or sx + sw > cell_width
        or sy + sh > cell_height
    ):
        raise TemplateError(
            "template source rectangle must fit below the dedicated cell header"
        )
    if sy != header_height:
        raise TemplateError(
            "template source rectangle must start exactly below the dedicated header"
        )
    qx, qy, qw, qh = summary_rect
    if (
        qx < 0
        or qy < 0
        or qw <= 0
        or qh <= 0
        or qx + qw > canvas_width
        or qy + qh > canvas_height
    ):
        raise TemplateError("template summary rectangle must fit inside the canvas")
    rollout_rects = []
    for index in range(rollout_cells):
        row, column = divmod(index, columns)
        rollout_rects.append(
            [column * cell_width, row * cell_height, cell_width, cell_height]
        )
    if any(_rectangles_overlap(rect, summary_rect) for rect in rollout_rects):
        raise TemplateError("template summary rectangle overlaps a rollout cell")
    remaining = columns * rows - rollout_cells
    if qw * qh != remaining * cell_width * cell_height:
        raise TemplateError(
            "template summary rectangle must exactly occupy all reserved grid cells"
        )

    metrics = _exact_keys(
        template["metrics"],
        {"success_field", "score_field"},
        "template.metrics",
    )
    _nonempty_string(metrics["success_field"], "template.metrics.success_field")
    _nonempty_string(metrics["score_field"], "template.metrics.score_field")

    style = _exact_keys(
        template["style"],
        {
            "canvas_bgr",
            "header_bgr",
            "header_text_bgr",
            "success_text_bgr",
            "summary_bgr",
            "summary_text_bgr",
            "header_font_scale",
            "header_font_thickness",
            "summary_title_font_scale",
            "summary_body_font_scale",
            "summary_font_thickness",
            "summary_footer",
        },
        "template.style",
    )
    for key in (
        "canvas_bgr",
        "header_bgr",
        "header_text_bgr",
        "success_text_bgr",
        "summary_bgr",
        "summary_text_bgr",
    ):
        _color(style[key], f"template.style.{key}")
    for key in (
        "header_font_scale",
        "summary_title_font_scale",
        "summary_body_font_scale",
    ):
        if _number(style[key], f"template.style.{key}", minimum=0.01) <= 0:
            raise TemplateError(f"template.style.{key} must be positive")
    for key in ("header_font_thickness", "summary_font_thickness"):
        _positive_integer(style[key], f"template.style.{key}")
    if summary_content == "policy_eval":
        # The policy_eval panel renders no footer line; the key stays for
        # schema stability but may be empty.
        if not isinstance(style["summary_footer"], str):
            raise TemplateError("template.style.summary_footer must be a string")
    else:
        _nonempty_string(style["summary_footer"], "template.style.summary_footer")

    template["_path"] = str(resolved)
    template["_sha256"] = hashlib.sha256(raw).hexdigest()
    template["_record"] = template_record
    return template


def _discover_videos(
    run_root: Path,
    video_glob: str,
    *,
    scene: str | None = None,
    runs: Sequence[str] | None = None,
) -> list[Path]:
    videos = [
        path
        for path in run_root.glob(video_glob)
        if path.name != "trajectory_analysis.mp4"
    ]
    if scene is not None:
        videos = [path for path in videos if path.parent.parent.name == scene]

    selected_runs = _normalize_run_filter(runs)
    if selected_runs is not None:
        selected_set = set(selected_runs)
        videos = [path for path in videos if path.parent.name in selected_set]
        matches: dict[str, list[Path]] = {name: [] for name in selected_runs}
        for path in videos:
            matches[path.parent.name].append(path)
        missing = [name for name in selected_runs if not matches[name]]
        ambiguous = {
            name: sorted(str(path) for path in paths)
            for name, paths in matches.items()
            if len(paths) > 1
        }
        if missing or ambiguous:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if ambiguous:
                details.append(f"ambiguous={ambiguous}")
            raise TemplateError(
                "--runs must resolve to exactly one rollout video per name: "
                + "; ".join(details)
            )

    def key(path: Path) -> tuple[str, int, str, str]:
        run_name = path.parent.name
        match = re.fullmatch(r"run_(\d+)", run_name)
        run_index = int(match.group(1)) if match is not None else sys.maxsize
        return (path.parent.parent.name, run_index, run_name, path.name)

    return sorted(videos, key=key)


def _normalize_run_filter(
    runs: Sequence[str] | str | None,
) -> tuple[str, ...] | None:
    """Return a validated exact run-name filter without silently deduplicating."""

    if runs is None:
        return None
    raw_items = runs.split(",") if isinstance(runs, str) else list(runs)
    items = [str(item).strip() for item in raw_items]
    if not items or any(not item for item in items):
        raise TemplateError("--runs must contain non-empty comma-separated names")
    invalid = [item for item in items if re.fullmatch(r"run_\d+", item) is None]
    if invalid:
        raise TemplateError(
            "--runs entries must match run_<integer>: " + ", ".join(invalid)
        )
    duplicates = sorted({item for item in items if items.count(item) > 1})
    if duplicates:
        raise TemplateError(
            "--runs contains duplicate entries: " + ", ".join(duplicates)
        )
    return tuple(items)


def _load_summary_with_record(
    video: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = video.parent / "sim_summary.json"
    if not path.is_file():
        raise TemplateError(f"missing rollout summary for {video}: {path}")
    try:
        raw, record = _read_stable_file(path)
        summary = json.loads(raw.decode("utf-8"))
    except (OSError, RuntimeError, UnicodeError, json.JSONDecodeError) as exc:
        raise TemplateError(f"could not read rollout summary {path}: {exc}") from exc
    if not isinstance(summary, dict):
        raise TemplateError(f"rollout summary must contain a JSON object: {path}")
    return summary, record


def _load_summary(video: Path) -> dict[str, Any]:
    summary, _record = _load_summary_with_record(video)
    return summary


def published_video_summary_path(run_dir: Path) -> Path:
    """Path of the summary that describes the *published* rollout video.

    A run can publish its canonical ``<scene>_<run>.mp4`` more than once. The
    post-hoc renderer republishes it and describes the result in
    ``rerender/sim_summary.json``; the saved-policy-OBS overlay republishes it
    and describes the result in ``saved_policy_obs/sim_summary.json``. The
    generation summary only describes the published video when neither of
    those republications happened.
    """

    for candidate in (
        run_dir / SAVED_POLICY_OBS_DIRNAME / "sim_summary.json",
        run_dir / "rerender" / "sim_summary.json",
    ):
        if candidate.is_file():
            return candidate
    return run_dir / "sim_summary.json"


def _capture_input_records(
    videos: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind every rollout video and the metadata used to label its cell."""

    summaries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for video in videos:
        summary, summary_record = _load_summary_with_record(video)
        record: dict[str, Any] = {
            "scene": video.parent.parent.name,
            "run": video.parent.name,
            "video": _file_record(video),
            "sim_summary": summary_record,
        }
        rerender_dir = video.parent / "rerender"
        saved_obs_dir = video.parent / SAVED_POLICY_OBS_DIRNAME
        optional_paths = {
            "rerender_summary": rerender_dir / "sim_summary.json",
            "rerender_validation": rerender_dir / "video_validation.json",
            "rerender_provenance": rerender_dir / "provenance.json",
            "saved_policy_obs_summary": saved_obs_dir / "sim_summary.json",
            "saved_policy_obs_validation": (
                saved_obs_dir / "video_validation.json"
            ),
            "saved_policy_obs_provenance": saved_obs_dir / "provenance.json",
        }
        for key, path in optional_paths.items():
            if path.exists():
                if not path.is_file():
                    raise RuntimeError(
                        f"optional provenance artifact is not a file: {path}"
                    )
                record[key] = _file_record(path)
        summaries.append(summary)
        records.append(record)
    return summaries, records


def _assert_render_sources_current(
    input_records: Sequence[dict[str, Any]],
    renderer_record: dict[str, Any],
    template_record: dict[str, Any],
) -> None:
    for item in input_records:
        prefix = f"{item['scene']}/{item['run']}"
        for key, value in item.items():
            if key in ("scene", "run"):
                continue
            _assert_record_current(value, label=f"{prefix} {key}")
    _assert_record_current(renderer_record, label="grid renderer code")
    _assert_record_current(template_record, label="grid template")


def _read_state_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TemplateError(
                        f"state-log line {number} is not a JSON object: {path}"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TemplateError(f"could not read state log {path}: {exc}") from exc
    if not rows:
        raise TemplateError(f"state log is empty: {path}")
    return rows


def validate_saved_policy_obs_inputs(
    videos: Sequence[Path],
) -> dict[str, Any]:
    """Fail closed unless every grid input embeds the exact saved policy OBS.

    A rollout grid is published as evidence of what the policy saw. Its wrist
    insets are therefore only admissible when they are the saved 224x224
    policy tensors themselves, scaled as a complete canvas so the real OpenPI
    letterbox stays visible. Runs that burned a re-rendered wrist camera, or a
    presentation crop of the policy tensor, are rejected here instead of being
    tiled into a grid that would misrepresent the policy input.
    """

    reports: list[dict[str, Any]] = []
    for video in videos:
        run_dir = video.parent
        label = f"{run_dir.parent.name}/{run_dir.name}"
        summary_path = published_video_summary_path(run_dir)
        if not summary_path.is_file():
            raise TemplateError(
                f"{label}: missing published-video summary {summary_path}"
            )
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TemplateError(
                f"{label}: could not read {summary_path}: {exc}"
            ) from exc
        if not isinstance(summary, dict):
            raise TemplateError(
                f"{label}: {summary_path} must contain a JSON object"
            )
        entry: dict[str, Any] = {
            "scene": run_dir.parent.name,
            "run": run_dir.name,
            "published_summary": str(summary_path),
        }
        if "video_wrist_insets_enabled" not in summary:
            raise TemplateError(
                f"{label}: {summary_path} does not declare "
                "video_wrist_insets_enabled, so the provenance of the wrist "
                "OBS burned into this grid input is unknown"
            )
        enabled = summary["video_wrist_insets_enabled"]
        if not isinstance(enabled, bool):
            raise TemplateError(
                f"{label}: video_wrist_insets_enabled must be a boolean"
            )
        source = summary.get("video_wrist_inset_source")
        if not enabled:
            if source not in (None, "off"):
                raise TemplateError(
                    f"{label}: wrist insets are disabled but "
                    f"video_wrist_inset_source is {source!r}"
                )
            entry.update(
                {
                    "wrist_inset_source": "off",
                    "layout_version": None,
                    "display_contract": None,
                    "saved_wrist_exact_match": None,
                }
            )
            reports.append(entry)
            continue

        layout = summary.get("video_wrist_inset_layout")
        layout_version = layout.get("version") if isinstance(layout, dict) else None
        contract = summary.get("video_wrist_display_contract")
        if (
            source != SAVED_WRIST_INSET_SOURCE
            or contract != SAVED_WRIST_DISPLAY_CONTRACT
            or layout_version != SAVED_WRIST_INSET_LAYOUT_VERSION
        ):
            raise TemplateError(
                f"{label}: a grid may only tile rollout videos whose wrist OBS "
                "insets are the exact saved 224px policy observation; "
                "re-rendering or presentation-cropping the OBS is forbidden. "
                f"{summary_path} declares source={source!r} "
                f"contract={contract!r} layout_version={layout_version!r}, "
                f"expected source={SAVED_WRIST_INSET_SOURCE!r} "
                f"contract={SAVED_WRIST_DISPLAY_CONTRACT!r} "
                f"layout_version={SAVED_WRIST_INSET_LAYOUT_VERSION}. "
                f"Publish the saved OBS first: {REWRITE_TOOL_HINT}"
            )

        state_log = run_dir / "logs" / "sim_log.jsonl"
        policy_obs_dir = run_dir / "logs" / "policy_obs"
        if not state_log.is_file():
            raise TemplateError(f"{label}: missing state log {state_log}")
        if not policy_obs_dir.is_dir():
            raise TemplateError(
                f"{label}: missing saved policy OBS directory {policy_obs_dir}"
            )
        try:
            schedule = load_replay_policy_insets(
                _read_state_rows(state_log), policy_obs_dir
            )
        except ReplayPolicyInsetError as exc:
            raise TemplateError(
                f"{label}: saved policy OBS schedule is incomplete: {exc}"
            ) from exc
        try:
            exact = validate_saved_wrist_insets(
                video,
                summary_path,
                schedule.raw_display_frames,
                observation_frames=schedule.observation_frames,
            )
        except VideoValidationError as exc:
            raise TemplateError(
                f"{label}: burned-in wrist OBS differs from the saved 224px "
                f"policy observation: {exc}"
            ) from exc
        except RuntimeError as exc:
            raise TemplateError(
                f"{label}: exact saved-wrist validation failed: {exc}"
            ) from exc
        entry.update(
            {
                "wrist_inset_source": source,
                "layout_version": layout_version,
                "display_contract": contract,
                "unique_saved_obs_pairs": int(schedule.unique_pairs),
                "saved_wrist_exact_match": exact,
            }
        )
        reports.append(entry)

    checked = [item for item in reports if item["saved_wrist_exact_match"]]
    return {
        "schema_version": 1,
        "contract": GRID_SAVED_POLICY_OBS_CONTRACT,
        "rerendered_wrist_obs_allowed": False,
        "checked_rollouts": len(reports),
        "exact_saved_obs_rollouts": len(checked),
        "saved_wrist_exact_match": bool(checked) and len(checked) == len(reports),
        "inputs": reports,
    }


def _metric_values(
    videos: list[Path],
    template: dict[str, Any],
    summaries: Sequence[dict[str, Any]] | None = None,
) -> tuple[list[bool], list[float | int], list[str]]:
    success_field = template["metrics"]["success_field"]
    score_field = template["metrics"]["score_field"]
    if summaries is None:
        summaries = [_load_summary(video) for video in videos]
    if len(summaries) != len(videos):
        raise RuntimeError("summary count does not match rollout video count")
    successes: list[bool] = []
    scores: list[float | int] = []
    scene_count = len({video.parent.parent.name for video in videos})
    labels = []
    for video, summary in zip(videos, summaries):
        if success_field not in summary:
            raise TemplateError(
                f"{video.parent}/sim_summary.json is missing {success_field!r}"
            )
        if score_field not in summary:
            raise TemplateError(
                f"{video.parent}/sim_summary.json is missing {score_field!r}"
            )
        raw_success = summary[success_field]
        if not isinstance(raw_success, (bool, int)) or int(raw_success) not in (0, 1):
            raise TemplateError(
                f"{success_field} must be boolean or 0/1 in {video.parent}"
            )
        raw_score = summary[score_field]
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise TemplateError(f"{score_field} must be numeric in {video.parent}")
        score = float(raw_score)
        if not math.isfinite(score):
            raise TemplateError(f"{score_field} must be finite in {video.parent}")
        normalized_score: float | int = int(score) if score.is_integer() else score
        successes.append(bool(raw_success))
        scores.append(normalized_score)
        run_label = video.parent.name
        if scene_count > 1:
            run_label = f"{video.parent.parent.name}/{run_label}"
        labels.append(
            f"{run_label} | {success_field}={int(bool(raw_success))} | "
            f"{score_field}={_format_number(normalized_score)}"
        )
    return successes, scores, labels


def _format_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{float(value):g}"


def _batch_server_checkpoint(run_root: Path) -> str | None:
    """Checkpoint path recorded by the policy batch server, if this run root
    was produced in batch-server mode (per-run summaries then carry no
    ``policy`` path of their own)."""

    ready_path = run_root / "policy_batch_server_ready.json"
    if not ready_path.is_file():
        return None
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TemplateError(
            f"could not read policy batch server identity {ready_path}: {exc}"
        ) from exc
    value = ready.get("metadata", {}).get("policy")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _policy_eval_fields(
    videos: Sequence[Path],
    summaries: Sequence[dict[str, Any]],
    run_root: Path,
) -> dict[str, str]:
    """Bind the one policy identity a policy_eval grid describes.

    Every rollout summary must carry the same checkpoint path, instruction,
    and success-criterion (task) name; mixing runs from different policies in
    one evaluation grid would silently misattribute the printed identity.
    """

    def _task_name(summary: dict[str, Any]) -> Any:
        criterion = summary.get("success_criterion")
        if isinstance(criterion, dict):
            return criterion.get("name")
        return None

    batch_checkpoint = _batch_server_checkpoint(run_root)

    def _checkpoint(summary: dict[str, Any]) -> Any:
        value = summary.get("policy")
        if isinstance(value, str) and value.strip():
            return value
        # Batch-server rollouts record the server address per run and the
        # checkpoint once at the run root.
        if summary.get("policy_batch_server"):
            return batch_checkpoint
        return value

    extractors = (
        ("checkpoint", _checkpoint),
        ("instruction", lambda summary: summary.get("instruction")),
        ("task", _task_name),
    )
    values: dict[str, str] = {}
    for field, extract in extractors:
        seen: list[str] = []
        for video, summary in zip(videos, summaries):
            value = extract(summary)
            if not isinstance(value, str) or not value.strip():
                raise TemplateError(
                    "policy_eval summary content requires a non-empty "
                    f"{field} in {video.parent}/sim_summary.json"
                )
            seen.append(value.strip())
        if len(set(seen)) != 1:
            raise TemplateError(
                f"policy_eval grid requires one shared {field}; "
                f"found {sorted(set(seen))}"
            )
        values[field] = seen[0]
    return values


def _strict_index(value: Any, *, label: str, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TemplateError(
            f"{label}: policy chunk record has invalid {field}={value!r}"
        )
    return int(value)


def _load_adopted_chunk_rows(
    run_dir: Path, summary: dict[str, Any], label: str
) -> list[dict[str, Any]] | None:
    """Read the adopted action chunks recorded for one rollout.

    ``None`` means the rollout recorded no chunk log at all, which is the case
    for non-policy rollouts. A summary that declares ``policy_chunks_jsonl``
    and then has no readable log is a fail-closed error: the panel must never
    silently print "not recorded" for a run that did record chunks.
    """

    path = run_dir / "logs" / "policy_chunks.jsonl"
    declared = summary.get("policy_chunks_jsonl")
    declares_log = isinstance(declared, str) and bool(declared.strip())
    if not path.is_file():
        if not declares_log:
            return None
        raise TemplateError(
            f"{label}: sim_summary.json declares policy_chunks_jsonl but "
            f"{path} does not exist"
        )
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TemplateError(
                        f"{label}: policy chunk line {number} is not an object"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TemplateError(f"{label}: could not read {path}: {exc}") from exc
    if not rows:
        raise TemplateError(f"{label}: policy chunk log is empty: {path}")
    return rows


def _adopted_chunk_timeline(
    rows: Sequence[dict[str, Any]], frames: int, label: str
) -> list[dict[str, int] | None]:
    """Map every rendered frame to the adopted action chunk executing on it.

    Each inference produces ``action_horizon`` steps and the rollout adopts
    only ``execution_start:execution_stop`` of them, anchored at the chunk's
    observation frame. Later chunks are applied last so a replan that preempts
    the previous chunk is what the frame is attributed to.
    """

    timeline: list[dict[str, int] | None] = [None] * frames
    for order, row in enumerate(rows, start=1):
        observation = _strict_index(
            row.get("observation_frame"), label=label, field="observation_frame"
        )
        start = _strict_index(
            row.get("execution_start"), label=label, field="execution_start"
        )
        stop = _strict_index(
            row.get("execution_stop"), label=label, field="execution_stop"
        )
        if stop <= start:
            raise TemplateError(
                f"{label}: policy chunk {order} has empty adopted slice "
                f"[{start}, {stop})"
            )
        chunk_index = _strict_index(
            row.get("chunk_index", order - 1), label=label, field="chunk_index"
        )
        global_start = observation + start
        global_stop = observation + stop
        for frame in range(max(0, global_start), min(frames, global_stop)):
            timeline[frame] = {
                "chunk_index": chunk_index,
                "adopted": order,
                "step": frame - global_start + 1,
                "adopted_steps": stop - start,
                "start": global_start,
                "stop": global_stop,
            }
    return timeline


def _format_range(values: Sequence[Any]) -> str:
    """Render one value, or the closed span the rollouts disagree over."""

    unique = sorted({value for value in values if value is not None})
    if not unique:
        return "-"
    if len(unique) == 1:
        return str(unique[0])
    return f"{unique[0]}~{unique[-1]}"


def _adopted_chunk_panel_line(
    timelines: Sequence[list[dict[str, int] | None] | None],
    totals: Sequence[int | None],
    horizons: Sequence[int | None],
    adopted_steps: Sequence[int | None],
    frame_index: int,
) -> str:
    """One panel line describing the adopted action chunks at this frame."""

    entries = [
        timeline[frame_index]
        for timeline in timelines
        if timeline is not None and frame_index < len(timeline)
    ]
    present = [entry for entry in entries if entry is not None]
    horizon_text = _format_range(horizons)
    adopted_text = _format_range(adopted_steps)
    configuration = f"ah={horizon_text} exec={adopted_text}"
    if not present:
        if not any(value is not None for value in totals):
            return "adopted action chunks: not recorded"
        return f"adopted action chunks -/{_format_range(totals)}  {configuration}"
    return (
        "adopted action chunks "
        f"{_format_range([item['adopted'] for item in present])}"
        f"/{_format_range(totals)}"
        f" | chunk {_format_range([item['chunk_index'] for item in present])}"
        f" step {_format_range([item['step'] for item in present])}"
        f"/{_format_range([item['adopted_steps'] for item in present])}"
        f" | frames {_format_range([item['start'] for item in present])}"
        f"-{_format_range([item['stop'] - 1 for item in present])}"
        f" | {configuration}"
    )


def _adopted_chunk_panel_state(
    videos: Sequence[Path],
    summaries: Sequence[dict[str, Any]],
    frames: int,
) -> dict[str, Any]:
    """Per-rollout adopted-chunk timelines plus the shared configuration."""

    timelines: list[list[dict[str, int] | None] | None] = []
    totals: list[int | None] = []
    horizons: list[int | None] = []
    adopted_steps: list[int | None] = []
    for video, summary in zip(videos, summaries):
        run_dir = video.parent
        label = f"{run_dir.parent.name}/{run_dir.name}"
        rows = _load_adopted_chunk_rows(run_dir, summary, label)
        if rows is None:
            timelines.append(None)
            totals.append(None)
        else:
            timelines.append(_adopted_chunk_timeline(rows, frames, label))
            totals.append(len(rows))
        for field, collected in (
            ("action_horizon", horizons),
            ("adopted_action_chunks", adopted_steps),
        ):
            value = summary.get(field)
            collected.append(
                None
                if isinstance(value, bool) or not isinstance(value, int)
                else int(value)
            )
    return {
        "timelines": timelines,
        "totals": totals,
        "horizons": horizons,
        "adopted_steps": adopted_steps,
    }


def _open_writer(
    path: Path,
    fps: float,
    size: tuple[int, int],
    codecs: Sequence[str],
) -> tuple[cv2.VideoWriter, str]:
    for codec in codecs:
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), fps, size
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"could not open an MP4 writer for {path}")


def _ascii(value: str, path: str) -> str:
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise TemplateError(
            f"{path} must currently use ASCII because the template font is OpenCV Hershey"
        ) from exc
    return value


def _put_text_fit(
    image: np.ndarray,
    text: str,
    *,
    origin: tuple[int, int],
    maximum_width: int,
    scale: float,
    color: Sequence[int],
    thickness: int,
    minimum_scale: float = 0.10,
) -> None:
    selected = float(scale)
    while selected > minimum_scale:
        (width, _height), _baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, selected, thickness
        )
        if width <= maximum_width:
            break
        selected *= 0.92
    selected = max(selected, minimum_scale)
    rendered_text = text
    while rendered_text:
        (width, _height), _baseline = cv2.getTextSize(
            rendered_text, cv2.FONT_HERSHEY_SIMPLEX, selected, thickness
        )
        if width <= maximum_width:
            break
        if rendered_text == "~":
            rendered_text = ""
            break
        stem = rendered_text[:-1] if rendered_text.endswith("~") else rendered_text
        stem = stem[:-1].rstrip()
        rendered_text = f"{stem}~" if stem else "~"
    cv2.putText(
        image,
        rendered_text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        selected,
        tuple(int(channel) for channel in color),
        int(thickness),
        cv2.LINE_AA,
    )


def _validate_grid_content(
    grid_path: Path,
    videos: Sequence[Path],
    *,
    expected_frames: int,
    expected_size: tuple[int, int],
    columns: int,
    cell_wh: tuple[int, int],
    source_xywh: tuple[int, int, int, int],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare every encoded grid cell with its decoded source frame.

    The comparison happens after grid encoding, so thresholds deliberately allow
    normal lossy MP4 error while still rejecting swaps, temporal drift, and
    localized layout corruption.
    """

    limits = dict(GRID_CONTENT_VALIDATION_THRESHOLDS)
    if thresholds is not None:
        limits.update(thresholds)
    grid_capture = open_video_capture(grid_path)
    source_captures = [open_video_capture(path) for path in videos]
    try:
        if not grid_capture.isOpened():
            raise RuntimeError(f"rendered grid video cannot be decoded: {grid_path}")
        failed = [
            str(path)
            for path, capture in zip(videos, source_captures)
            if not capture.isOpened()
        ]
        if failed:
            raise RuntimeError(
                f"grid content validation could not open source videos: {failed}"
            )
        width = int(round(grid_capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(grid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if (width, height) != expected_size:
            raise RuntimeError(
                "grid content validation found wrong dimensions: "
                f"{width}x{height}, expected {expected_size[0]}x{expected_size[1]}"
            )

        cell_width, cell_height = cell_wh
        sx, sy, sw, sh = source_xywh
        block_wh = (max(1, math.ceil(sw / 16)), max(1, math.ceil(sh / 16)))
        aggregate_sum = 0.0
        aggregate_count = 0
        aggregate_bad = 0
        maximum_frame_mae = 0.0
        maximum_block_mae = 0.0
        per_input = [
            {
                "path": str(path.resolve()),
                "sum_abs_error": 0.0,
                "sample_count": 0,
                "count_abs_error_gt_32": 0,
                "max_frame_mean_abs_error": 0.0,
                "max_coarse_block_mean_abs_error": 0.0,
            }
            for path in videos
        ]

        for frame_index in range(expected_frames):
            ok, grid_frame = grid_capture.read()
            if not ok:
                raise RuntimeError(
                    f"grid content validation could not decode frame {frame_index}"
                )
            if grid_frame.shape[:2] != (expected_size[1], expected_size[0]):
                raise RuntimeError(
                    f"grid frame {frame_index} has shape {grid_frame.shape}, "
                    f"expected {expected_size[1]}x{expected_size[0]}"
                )
            for index, (path, capture) in enumerate(zip(videos, source_captures)):
                source_ok, source_frame = capture.read()
                if not source_ok:
                    raise RuntimeError(
                        f"grid content validation could not decode frame {frame_index} "
                        f"from {path}"
                    )
                expected = cv2.resize(
                    source_frame, (sw, sh), interpolation=cv2.INTER_AREA
                )
                row, column = divmod(index, columns)
                cell_x = column * cell_width
                cell_y = row * cell_height
                actual = grid_frame[
                    cell_y + sy : cell_y + sy + sh,
                    cell_x + sx : cell_x + sx + sw,
                ]
                if actual.shape != expected.shape:
                    raise RuntimeError(
                        f"grid source ROI {index} has shape {actual.shape}, "
                        f"expected {expected.shape}"
                    )
                difference = np.abs(
                    actual.astype(np.int16) - expected.astype(np.int16)
                )
                frame_mae = float(difference.mean())
                block_mae = float(
                    cv2.resize(
                        difference.astype(np.float32),
                        block_wh,
                        interpolation=cv2.INTER_AREA,
                    ).max()
                )
                error_sum = float(difference.sum(dtype=np.int64))
                sample_count = int(difference.size)
                bad_count = int(np.count_nonzero(difference > 32))
                stats = per_input[index]
                stats["sum_abs_error"] += error_sum
                stats["sample_count"] += sample_count
                stats["count_abs_error_gt_32"] += bad_count
                stats["max_frame_mean_abs_error"] = max(
                    stats["max_frame_mean_abs_error"], frame_mae
                )
                stats["max_coarse_block_mean_abs_error"] = max(
                    stats["max_coarse_block_mean_abs_error"], block_mae
                )
                aggregate_sum += error_sum
                aggregate_count += sample_count
                aggregate_bad += bad_count
                maximum_frame_mae = max(maximum_frame_mae, frame_mae)
                maximum_block_mae = max(maximum_block_mae, block_mae)

        extra, _frame = grid_capture.read()
        if extra:
            raise RuntimeError(
                f"rendered grid has more than the expected {expected_frames} frames"
            )
        for path, capture in zip(videos, source_captures):
            extra, _frame = capture.read()
            if extra:
                raise RuntimeError(
                    f"source video has more than the expected {expected_frames} frames: {path}"
                )

        report_inputs = []
        for stats in per_input:
            count = int(stats.pop("sample_count"))
            bad = int(stats.pop("count_abs_error_gt_32"))
            error_sum = float(stats.pop("sum_abs_error"))
            report_inputs.append(
                {
                    **stats,
                    "mean_abs_error": error_sum / count,
                    "fraction_abs_error_gt_32": bad / count,
                }
            )
        report = {
            "schema_version": 1,
            "method": "decode_each_grid_and_source_frame_then_compare_resized_cell",
            "checked_frames": expected_frames,
            "checked_rollouts": len(videos),
            "source_xywh_within_cell": [sx, sy, sw, sh],
            "thresholds": limits,
            "metrics": {
                "mean_abs_error": aggregate_sum / aggregate_count,
                "max_frame_mean_abs_error": maximum_frame_mae,
                "max_coarse_block_mean_abs_error": maximum_block_mae,
                "fraction_abs_error_gt_32": aggregate_bad / aggregate_count,
            },
            "inputs": report_inputs,
        }
        violations = []
        all_metrics = [("aggregate", report["metrics"])] + [
            (f"input[{index}]", metrics)
            for index, metrics in enumerate(report_inputs)
        ]
        for label, metrics in all_metrics:
            for metric, threshold_key in (
                ("mean_abs_error", "mean_abs_error_max"),
                ("max_frame_mean_abs_error", "frame_mean_abs_error_max"),
                (
                    "max_coarse_block_mean_abs_error",
                    "coarse_block_mean_abs_error_max",
                ),
                ("fraction_abs_error_gt_32", "fraction_abs_error_gt_32_max"),
            ):
                if float(metrics[metric]) > float(limits[threshold_key]):
                    violations.append(
                        f"{label}.{metric}={metrics[metric]:.6g} > "
                        f"{limits[threshold_key]:.6g}"
                    )
        if violations:
            raise RuntimeError(
                "encoded grid content does not match its bound inputs: "
                + "; ".join(violations)
            )
        return report
    finally:
        grid_capture.release()
        for capture in source_captures:
            capture.release()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def render_from_template(
    *,
    run_root: Path,
    output: Path,
    template_path: Path,
    title: str = "",
    expected_frames: int = 0,
    scene: str | None = None,
    runs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Render one grid; layout and styling can only come from a template."""

    template = load_template(template_path)
    run_root = run_root.expanduser().resolve()
    output = output.expanduser().resolve()
    selected_runs = _normalize_run_filter(runs)
    videos = _discover_videos(
        run_root,
        template["video_glob"],
        scene=scene,
        runs=selected_runs,
    )
    if len(videos) != template["expected_videos"]:
        raise RuntimeError(
            f"template {template['id']!r} requires {template['expected_videos']} "
            f"rollout videos below {run_root}, found {len(videos)}"
        )
    if expected_frames < 0:
        raise ValueError("expected_frames must be non-negative")

    if output in {path.resolve() for path in videos}:
        raise RuntimeError("grid output path must not overwrite a rollout input video")
    renderer_record = _file_record(Path(__file__))
    summaries, input_records = _capture_input_records(videos)
    # Fail closed before any encoding work: a grid must never be published
    # from rollout videos whose wrist OBS was re-rendered for presentation.
    saved_policy_obs_validation = validate_saved_policy_obs_inputs(videos)
    successes, scores, labels = _metric_values(videos, template, summaries)
    policy_eval_fields = (
        _policy_eval_fields(videos, summaries, run_root)
        if template["summary_content"] == "policy_eval"
        else None
    )
    labels = [_ascii(label, "rollout label") for label in labels]
    if template["summary_content"] == "policy_eval":
        # The evaluation panel is titled by the experiment directory itself so
        # the grid is attributable without an operator-supplied title.
        display_title = _ascii(run_root.resolve().name, "title")
    else:
        display_title = _ascii(title or scene or run_root.name, "title")
    footer = _ascii(template["style"]["summary_footer"], "template summary footer")
    captures = [open_video_capture(path) for path in videos]
    temporary_video: Path | None = None
    temporary_poster: Path | None = None
    content_validation: dict[str, Any] | None = None
    output_video_record: dict[str, Any] | None = None
    output_poster_record: dict[str, Any] | None = None
    try:
        if not all(capture.isOpened() for capture in captures):
            failed = [
                str(path)
                for path, capture in zip(videos, captures)
                if not capture.isOpened()
            ]
            raise RuntimeError(f"could not open rollout videos: {failed}")
        widths = [int(round(item.get(cv2.CAP_PROP_FRAME_WIDTH))) for item in captures]
        heights = [int(round(item.get(cv2.CAP_PROP_FRAME_HEIGHT))) for item in captures]
        frame_counts = [int(round(item.get(cv2.CAP_PROP_FRAME_COUNT))) for item in captures]
        frame_rates = [float(item.get(cv2.CAP_PROP_FPS)) for item in captures]
        source_size = tuple(template["input_frame_wh"])
        if set(zip(widths, heights)) != {source_size}:
            raise RuntimeError(
                f"template {template['id']!r} requires source dimensions "
                f"{source_size[0]}x{source_size[1]}, found {list(zip(widths, heights))}"
            )
        if len(set(frame_counts)) != 1 or frame_counts[0] <= 0:
            raise RuntimeError(f"rollout video frame counts differ or are empty: {frame_counts}")
        if expected_frames and frame_counts[0] != expected_frames:
            raise RuntimeError(
                f"expected {expected_frames} frames per rollout, found {frame_counts[0]}"
            )
        if any(fps <= 0 for fps in frame_rates) or max(frame_rates) - min(frame_rates) > 0.05:
            raise RuntimeError(f"rollout video frame rates differ or are invalid: {frame_rates}")

        adopted_chunks = _adopted_chunk_panel_state(
            videos, summaries, frame_counts[0]
        )
        geometry = template["geometry"]
        style = template["style"]
        canvas_width, canvas_height = geometry["canvas_wh"]
        cell_width, cell_height = geometry["cell_wh"]
        columns = geometry["columns"]
        header_height = geometry["header_height"]
        sx, sy, sw, sh = geometry["source_xywh_within_cell"]
        qx, qy, qw, qh = geometry["summary_xywh"]
        output_size = (canvas_width, canvas_height)
        fps = frame_rates[0]
        poster_index = (frame_counts[0] - 1) // 2
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_video = output.with_name(
            f".{output.stem}.tmp-{os.getpid()}{output.suffix}"
        )
        poster_path = output.with_name(f"{output.stem}_poster.jpg")
        temporary_poster = poster_path.with_name(
            f".{poster_path.stem}.tmp-{os.getpid()}{poster_path.suffix}"
        )
        writer, codec = _open_writer(
            temporary_video,
            fps,
            output_size,
            template["codec_preference"],
        )
        written = 0
        try:
            for frame_index in range(frame_counts[0]):
                canvas = np.full(
                    (canvas_height, canvas_width, 3),
                    np.asarray(style["canvas_bgr"], dtype=np.uint8),
                    dtype=np.uint8,
                )
                rollout_pixels: list[tuple[int, int, np.ndarray]] = []
                for index, (capture, label, success) in enumerate(
                    zip(captures, labels, successes)
                ):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(
                            f"failed to read frame {frame_index} from {videos[index]}"
                        )
                    row, column = divmod(index, columns)
                    cell_x = column * cell_width
                    cell_y = row * cell_height
                    canvas[
                        cell_y : cell_y + header_height,
                        cell_x : cell_x + cell_width,
                    ] = style["header_bgr"]
                    _put_text_fit(
                        canvas,
                        label,
                        origin=(cell_x + 8, cell_y + min(header_height - 9, 25)),
                        maximum_width=cell_width - 16,
                        scale=style["header_font_scale"],
                        color=(
                            style["success_text_bgr"]
                            if success
                            else style["header_text_bgr"]
                        ),
                        thickness=style["header_font_thickness"],
                    )
                    rendered = cv2.resize(
                        frame, (sw, sh), interpolation=cv2.INTER_AREA
                    )
                    canvas[
                        cell_y + sy : cell_y + sy + sh,
                        cell_x + sx : cell_x + sx + sw,
                    ] = rendered
                    rollout_pixels.append((cell_x, cell_y, rendered))

                canvas[qy : qy + qh, qx : qx + qw] = style["summary_bgr"]
                text_x = qx + 28
                success_count = sum(successes)
                success_rate = success_count / len(successes)
                score_array = np.asarray(scores, dtype=float)
                score_values = ", ".join(_format_number(value) for value in scores)
                score_values_line = (
                    f"{template['metrics']['score_field']} values: {score_values}"
                    if len(scores) <= 20
                    else f"{template['metrics']['score_field']} values: n={len(scores)}"
                )
                adopted_chunk_line = _ascii(
                    _adopted_chunk_panel_line(
                        adopted_chunks["timelines"],
                        adopted_chunks["totals"],
                        adopted_chunks["horizons"],
                        adopted_chunks["adopted_steps"],
                        frame_index,
                    ),
                    "adopted action chunk line",
                )
                if policy_eval_fields is not None:
                    summary_lines = [
                        (display_title, style["summary_title_font_scale"]),
                        (
                            f"task: {policy_eval_fields['task']}   "
                            f"frame {frame_index + 1}/{frame_counts[0]}",
                            style["summary_body_font_scale"],
                        ),
                        (
                            f"instruction: {policy_eval_fields['instruction']}",
                            style["summary_body_font_scale"],
                        ),
                        (
                            f"checkpoint: {policy_eval_fields['checkpoint']}",
                            style["summary_body_font_scale"],
                        ),
                        (
                            f"success: {success_count}/{len(successes)} "
                            f"({success_rate * 100.0:.1f}%)   "
                            f"{template['metrics']['score_field']} "
                            f"mean={score_array.mean():.2f} "
                            f"min={_format_number(min(scores))} "
                            f"max={_format_number(max(scores))}",
                            style["summary_body_font_scale"],
                        ),
                        (adopted_chunk_line, style["summary_body_font_scale"]),
                    ]
                else:
                    summary_lines = [
                        (display_title, style["summary_title_font_scale"]),
                        (
                            f"frame {frame_index + 1}/{frame_counts[0]}",
                            style["summary_body_font_scale"],
                        ),
                        (
                            f"{template['metrics']['success_field']}: "
                            f"{success_count}/{len(successes)} ({success_rate * 100.0:.1f}%)",
                            style["summary_body_font_scale"],
                        ),
                        (
                            score_values_line,
                            style["summary_body_font_scale"],
                        ),
                        (
                            f"{template['metrics']['score_field']} mean={score_array.mean():.2f}  "
                            f"min={_format_number(min(scores))}  max={_format_number(max(scores))}",
                            style["summary_body_font_scale"],
                        ),
                        (adopted_chunk_line, style["summary_body_font_scale"]),
                        (footer, 0.43),
                    ]
                divisions = len(summary_lines) + 1
                for line_index, (text, scale) in enumerate(summary_lines):
                    _put_text_fit(
                        canvas,
                        text,
                        origin=(
                            text_x,
                            qy + int(round((line_index + 1) * qh / divisions)),
                        ),
                        maximum_width=qw - 56,
                        scale=scale,
                        color=style["summary_text_bgr"],
                        thickness=style["summary_font_thickness"],
                    )
                # Every label lives in a dedicated header strip or the reserved
                # summary panel. Prove it per frame instead of asserting it in
                # the manifest: the rollout pixels must still be byte-identical
                # to the resized source after all text has been drawn.
                for cell_x, cell_y, rendered in rollout_pixels:
                    drawn = canvas[
                        cell_y + sy : cell_y + sy + sh,
                        cell_x + sx : cell_x + sx + sw,
                    ]
                    if not np.array_equal(drawn, rendered):
                        raise RuntimeError(
                            "a grid label covered rollout pixels in cell at "
                            f"({cell_x}, {cell_y}) on frame {frame_index}"
                        )
                writer.write(canvas)
                if frame_index == poster_index:
                    if not cv2.imwrite(str(temporary_poster), canvas):
                        raise RuntimeError(
                            f"could not write poster image {temporary_poster}"
                        )
                written += 1
        finally:
            writer.release()
        # Rendering is finished. Release its decoders before ffmpeg and content
        # validation open the encoded grid and a fresh set of source readers.
        for capture in captures:
            capture.release()
        captures.clear()
        if written != frame_counts[0]:
            raise RuntimeError(f"wrote {written} grid frames, expected {frame_counts[0]}")
        if temporary_poster is None or not temporary_poster.is_file():
            raise RuntimeError("template poster frame was not rendered")
        # Publish H.264: OpenCV lacks an H.264 encoder, so the grid streams
        # into the fourcc intermediate above and the pinned ffmpeg re-encodes
        # it before validation, which then checks the published bytes.
        intermediate_codec = codec
        ffmpeg = _resolve_ffmpeg()
        if ffmpeg is not None:
            encoded_video = temporary_video.with_name(
                temporary_video.stem + ".h264" + temporary_video.suffix
            )
            try:
                _transcode_to_h264(str(temporary_video), str(encoded_video), ffmpeg)
                os.replace(encoded_video, temporary_video)
            except BaseException:
                if encoded_video.exists():
                    encoded_video.unlink()
                raise
            codec = "h264"
        content_validation = _validate_grid_content(
            temporary_video,
            videos,
            expected_frames=written,
            expected_size=output_size,
            columns=columns,
            cell_wh=(cell_width, cell_height),
            source_xywh=(sx, sy, sw, sh),
        )
        _assert_render_sources_current(
            input_records,
            renderer_record,
            template["_record"],
        )
        temporary_video_record = _file_record(temporary_video)
        temporary_poster_record = _file_record(temporary_poster)
        # Recheck after output hashing too: no known-good output is replaced
        # unless every bound source remained unchanged for the whole encode.
        _assert_render_sources_current(
            input_records,
            renderer_record,
            template["_record"],
        )
        os.replace(temporary_video, output)
        temporary_video = None
        os.replace(temporary_poster, poster_path)
        temporary_poster = None
        output_video_record = _file_record(output)
        output_poster_record = _file_record(poster_path)
        if output_video_record["sha256"] != temporary_video_record["sha256"]:
            raise RuntimeError("grid video changed while it was atomically installed")
        if output_poster_record["sha256"] != temporary_poster_record["sha256"]:
            raise RuntimeError("grid poster changed while it was atomically installed")

        rollout_rects = []
        for index in range(template["expected_videos"]):
            row, column = divmod(index, columns)
            rollout_rects.append(
                [column * cell_width, row * cell_height, cell_width, cell_height]
            )
        success_field = template["metrics"]["success_field"]
        score_field = template["metrics"]["score_field"]
        resolved_template_path = Path(template["_path"])
        try:
            manifest_template_path = str(resolved_template_path.relative_to(REPO_ROOT))
        except ValueError:
            manifest_template_path = str(resolved_template_path)
        manifest = {
            "run_root": str(run_root),
            "output": str(output),
            "poster": str(poster_path),
            "title": display_title,
            "scene": scene,
            "runs": list(selected_runs) if selected_runs is not None else None,
            "codec": codec,
            "intermediate_codec": intermediate_codec,
            "fps": fps,
            "frames": written,
            "width": canvas_width,
            "height": canvas_height,
            "inputs": [str(path.resolve()) for path in videos],
            "template": template["id"],
            "template_path": manifest_template_path,
            "template_sha256": template["_sha256"],
            "template_schema_version": template["schema_version"],
            "provenance": {
                "schema_version": 1,
                "inputs": input_records,
                "renderer": renderer_record,
                "template": template["_record"],
                "outputs": {
                    "video": output_video_record,
                    "poster": output_poster_record,
                },
            },
            "content_validation": content_validation,
            "saved_policy_obs_validation": saved_policy_obs_validation,
            "render_contract": {
                "layout_source": "required_json_template",
                "template_only": True,
                "labels_obscure_rollout_pixels": False,
                "labels_obscure_rollout_pixels_check": (
                    "per_frame_exact_cell_pixel_equality"
                ),
                "legacy_layout_cli_supported": False,
                "wrist_obs_source": "exact_saved_224px_policy_obs",
                "wrist_obs_rerendered": False,
            },
            "geometry": {
                "canvas_wh": [canvas_width, canvas_height],
                "columns": columns,
                "rows": geometry["rows"],
                "rollout_cells": template["expected_videos"],
                "cell_wh": [cell_width, cell_height],
                "header_height": header_height,
                "source_wh": [widths[0], heights[0]],
                "rendered_source_wh": [sw, sh],
                "source_padding_ltrb": [
                    sx,
                    sy - header_height,
                    cell_width - sx - sw,
                    cell_height - sy - sh,
                ],
                "source_xywh_within_cell": [sx, sy, sw, sh],
                "rollout_cell_xywh": rollout_rects,
                "summary_xywh": [qx, qy, qw, qh],
            },
            "metrics": {
                "official_success_field": success_field,
                "official_score_field": score_field,
                f"{success_field}_values": successes,
                f"{success_field}_count": int(sum(successes)),
                f"{success_field}_rate": float(sum(successes) / len(successes)),
                f"{score_field}_values": scores,
                f"{score_field}_mean": float(np.mean(scores)),
                f"{score_field}_min": min(scores),
                f"{score_field}_max": max(scores),
            },
        }
        _write_json_atomic(output.with_suffix(".json"), manifest)
        return manifest
    finally:
        for capture in captures:
            capture.release()
        for temporary in (temporary_video, temporary_poster):
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE_PATH,
        help=f"required layout template (default: {DEFAULT_TEMPLATE_PATH})",
    )
    parser.add_argument("--title", default="")
    parser.add_argument(
        "--scene",
        default=None,
        help="optional exact scene directory name; required for per-scene comparison grids",
    )
    parser.add_argument(
        "--runs",
        default="",
        help=(
            "optional comma-separated exact run directory names, for example "
            "run_01,run_03; every name must resolve to exactly one video"
        ),
    )
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=0,
        help="optional frame-count assertion; 0 accepts the common input count",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.expected_frames < 0:
        parser.error("--expected-frames must be non-negative")
    try:
        args.runs = _normalize_run_filter(args.runs) if args.runs else None
    except TemplateError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = render_from_template(
            run_root=args.run_root,
            output=args.output,
            template_path=args.template,
            title=args.title,
            expected_frames=args.expected_frames,
            scene=args.scene,
            runs=args.runs,
        )
    except (OSError, RuntimeError, TemplateError, ValueError) as exc:
        print(f"rollout grid render failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
