# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Rerender output validation and immutable provenance publication."""


from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robosuite_env.contracts.provenance import official_generation_files_unchanged
from robosuite_env.rollout.replay_policy_insets import (
    load_replay_policy_insets,
)
from robosuite_env.tools.validate_rollout_video import (
    VideoValidationError,
    validate_saved_wrist_insets,
    validate_video,
    write_json_atomic,
)

from robosuite_env.rerender.cadence import (
    _validate_frame_hz_replay_cadence,
    _validate_policy_action_replay_cadence,
)
from robosuite_env.rerender.common import (
    REPO_ROOT,
    RerenderError,
    SourceRun,
    _SHA256_RE,
)
from robosuite_env.rerender.provenance import (
    _assert_record_current,
    _assert_renderer_code_current,
    _file_record,
)
from robosuite_env.rerender.registration import (
    _assert_registration_matches,
    _capture_summary_gs_registration,
)
from robosuite_env.rerender.replay_argv import _without_option
from robosuite_env.rerender.sources import (
    _assert_batch_policy_lineage_current,
    prepare_source_run,
)


def validate_rerender_output(
    source: SourceRun | Path,
    video_path: Path,
    *,
    expected_wrist_insets: str | None = None,
) -> dict[str, Any]:
    """Validate and atomically publish the post-hoc render's quality report."""

    source_run = source if isinstance(source, SourceRun) else prepare_source_run(source)
    run_dir = source_run.run_dir
    if not video_path.is_absolute():
        video_path = REPO_ROOT / video_path
    summary_path = run_dir / "rerender" / "sim_summary.json"
    report_path = run_dir / "rerender" / "video_validation.json"
    try:
        replay_summary_record = _file_record(summary_path, include_state=True)
        result = validate_video(video_path, summary_path)
        with summary_path.open("r", encoding="utf-8") as handle:
            replay_summary = json.load(handle)
        _assert_record_current(
            replay_summary_record,
            label="rerender summary",
            verify_hash=True,
        )
        source_registration = source_run.source_records.get("gs_registration")
        if not isinstance(source_registration, dict):
            raise RerenderError("source has no fail-closed GS registration record")
        replay_registration = _capture_summary_gs_registration(
            source_run.argv,
            replay_summary,
            label=f"{run_dir}: rerender summary",
        )
        _assert_registration_matches(
            source_registration,
            replay_registration,
            label=f"{run_dir}: rerender summary",
            compare_effective=True,
        )
        result["gs_registration_exact_match"] = True
        reported_wrist_source = replay_summary.get("video_wrist_inset_source")
        if expected_wrist_insets is not None and reported_wrist_source != expected_wrist_insets:
            raise RuntimeError(
                "replay summary wrist source differs from requested mode: "
                f"{reported_wrist_source!r} != {expected_wrist_insets!r}"
            )
        if reported_wrist_source == "saved":
            schedule = load_replay_policy_insets(
                source_run.state_rows,
                run_dir / "logs" / "policy_obs",
            )
            layout = replay_summary.get("video_wrist_inset_layout")
            layout_version = layout.get("version") if isinstance(layout, dict) else None
            expected_wrist_frames = (
                schedule.raw_display_frames
                if layout_version == 3
                else schedule.display_frames
            )
            result["saved_wrist_exact_match"] = validate_saved_wrist_insets(
                video_path,
                summary_path,
                expected_wrist_frames,
                observation_frames=schedule.observation_frames,
            )
        cadence_validation = _validate_policy_action_replay_cadence(
            source_run,
            replay_summary,
            result,
        )
        if cadence_validation is not None:
            result["policy_action_playback_contract"] = cadence_validation
        frame_hz_validation = _validate_frame_hz_replay_cadence(
            source_run,
            replay_summary,
            result,
        )
        if frame_hz_validation is not None:
            result["frame_hz_playback_contract"] = frame_hz_validation
        _assert_record_current(
            replay_summary_record,
            label="rerender summary",
            verify_hash=True,
        )
        result["rerender_summary_sha256"] = replay_summary_record["sha256"]
    except Exception as exc:
        if isinstance(exc, VideoValidationError):
            failure = exc.result
        else:
            failure = {
                "ok": False,
                "video": str(video_path.resolve()),
                "summary": str(summary_path.resolve()),
                "error": f"{type(exc).__name__}: {exc}",
            }
        write_json_atomic(report_path, failure)
        raise RerenderError(
            f"{run_dir}: video validation failed: {exc}; see {report_path}"
        ) from exc
    try:
        rendered_frames = result["decoded_frames"]
        if (
            isinstance(rendered_frames, bool)
            or not isinstance(rendered_frames, int)
            or rendered_frames != source_run.expected_frames
        ):
            raise RerenderError(
                f"{run_dir}: incomplete replay rendered {rendered_frames} of "
                f"{source_run.expected_frames} logged frames"
            )
    except (IndexError, KeyError, TypeError, ValueError, RerenderError) as exc:
        failure = dict(result)
        failure["ok"] = False
        failure["error"] = str(exc)
        write_json_atomic(report_path, failure)
        if isinstance(exc, RerenderError):
            raise
        raise RerenderError(f"{run_dir}: replay completeness check failed: {exc}") from exc
    write_json_atomic(report_path, result)
    return result


def _sanitized_replay_argv(replay: list[str], secret: str) -> list[str]:
    sanitized, removed = _without_option(replay, "--gs-server-token")
    if removed != 1:
        raise RerenderError("rerender argv has no unique --gs-server-token binding")
    if secret and any(secret in token for token in sanitized):
        raise RerenderError("rerender token leaked into sanitized argv")
    return sanitized


def _source_with_generation_recheck(
    source: SourceRun, renderer_unchanged: bool
) -> dict[str, Any]:
    """The source records, the official generation check rechecked now.

    The check the renderer made before rendering is stored with
    ``unchanged_at_write``: whether the same checkout still held every file
    it compared, and the renderer code it identified, when the video was
    done. Completion reporting refuses the render when it is false.
    """

    records = dict(source.source_records)
    check = records.get("official_generation_check")
    if check is not None:
        records["official_generation_check"] = {
            **check,
            "unchanged_at_write": renderer_unchanged
            and official_generation_files_unchanged(check),
        }
    return records


def write_rerender_provenance(
    *,
    source: SourceRun,
    replay: list[str],
    gs_token: str,
    scene: dict[str, Any],
    video_path: Path,
    validation: dict[str, Any],
    view_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Atomically bind one validated derivative to its immutable inputs."""

    for key in ("generation_summary", "state_log", "run_command"):
        _assert_record_current(
            source.source_records[key],
            label=key.replace("_", " "),
            verify_hash=True,
        )
    _assert_batch_policy_lineage_current(
        source.source_records["batch_policy_lineage"]
    )
    if source.official:
        for key in ("run_manifest", "worker_contract"):
            _assert_record_current(
                source.source_records[key],
                label=key.replace("_", " "),
                verify_hash=True,
            )
    renderer_unchanged = _assert_renderer_code_current(source.renderer_code)
    _assert_record_current(scene["manifest"], label="scene manifest")
    _assert_record_current(scene["ply"], label="scene PLY")
    source_registration = source.source_records.get("gs_registration")
    if not isinstance(source_registration, dict):
        raise RerenderError("source has no fail-closed GS registration record")
    source_w2gs_file = source_registration.get("file")
    if source_w2gs_file is not None:
        if not isinstance(source_w2gs_file, dict):
            raise RerenderError("source GS registration file record is invalid")
        _assert_record_current(
            source_w2gs_file,
            label="source GS registration",
            verify_hash=True,
        )
    scene_registration = scene.get("w2gs")
    if not isinstance(scene_registration, dict):
        raise RerenderError("scene has no GS registration identity")
    scene_w2gs_file = scene_registration.get("file")
    if scene_w2gs_file is not None:
        if not isinstance(scene_w2gs_file, dict):
            raise RerenderError("scene GS registration file record is invalid")
        _assert_record_current(
            scene_w2gs_file,
            label="scene GS registration",
            verify_hash=True,
        )

    decoded_frames = validation.get("decoded_frames")
    decoded_sha256 = validation.get("decoded_sha256")
    if decoded_frames != source.expected_frames:
        raise RerenderError("validated video frame count changed before provenance")
    if not isinstance(decoded_sha256, str) or _SHA256_RE.fullmatch(
        decoded_sha256
    ) is None:
        raise RerenderError("video validation has no valid decoded SHA-256")
    replay_summary_sha256 = validation.get("rerender_summary_sha256")
    if not isinstance(replay_summary_sha256, str) or _SHA256_RE.fullmatch(
        replay_summary_sha256
    ) is None:
        raise RerenderError("video validation has no valid rerender summary SHA-256")

    video_path = video_path.resolve()
    replay_summary_path = source.run_dir / "rerender" / "sim_summary.json"
    validation_path = source.run_dir / "rerender" / "video_validation.json"
    replay_summary_record = _file_record(replay_summary_path, include_state=True)
    if replay_summary_record["sha256"] != replay_summary_sha256:
        raise RerenderError("rerender summary changed after video validation")
    video_record = _file_record(video_path, include_state=True)
    validation_record = _file_record(validation_path, include_state=True)
    _assert_record_current(
        replay_summary_record,
        label="rerender summary",
        verify_hash=True,
    )
    payload = {
        "schema_version": 5,
        "kind": "real2sim-rerender-provenance",
        "mode": "official" if source.official else "diagnostic",
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
        "source": _source_with_generation_recheck(source, renderer_unchanged),
        "renderer": {
            "sanitized_argv": _sanitized_replay_argv(replay, gs_token),
            "code": source.renderer_code,
            # None attests that the recorded third-person viewpoint was kept.
            "view_overrides": dict(view_overrides) if view_overrides else None,
        },
        "scene": scene,
        "output": {
            "video": video_record,
            "validation": validation_record,
            "rerender_summary": replay_summary_record,
            "decoded_frames": decoded_frames,
            "decoded_sha256": decoded_sha256,
        },
    }
    cadence_contract = validation.get("policy_action_playback_contract")
    if cadence_contract is not None:
        source_cadence = source.source_records.get(
            "policy_action_playback_contract"
        )
        if (
            not isinstance(source_cadence, dict)
            or not isinstance(cadence_contract, dict)
            or any(
                cadence_contract.get(key) != value
                for key, value in source_cadence.items()
            )
        ):
            raise RerenderError(
                "validated policy-action playback contract differs from source"
            )
        payload["policy_action_playback_contract"] = cadence_contract
    write_json_atomic(source.run_dir / "rerender" / "provenance.json", payload)
    return payload
