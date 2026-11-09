# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Saved-state replay cadence contracts bound to the source run's clock."""


from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from robosuite_env.rerender.common import (
    RerenderError,
    SourceRun,
    _last_option_value,
)
from robosuite_env.rerender.registration import (
    _require_summary_close,
    _required_finite_summary_number,
)


from robosuite_env.robots.profiles import robot_embodiment
from robosuite_env.rollout.timing import exact_state_replay_playback_clock


def _fr3_policy_action_playback_contract(
    argv: list[str], source_summary: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Validate a live FR3 action clock before converting it to saved replay.

    Legacy commands without a positive ``--policy-action-hz`` return ``None``
    unchanged.  A positive clock opts into the new contract and therefore has
    no permissive fallback: command, generation summary, controller schedule,
    and nominal presentation timing must all agree.
    """

    raw_action_hz = _last_option_value(argv, "--policy-action-hz")
    if raw_action_hz is None:
        return None
    try:
        action_hz = float(raw_action_hz)
    except ValueError as exc:
        raise RerenderError(
            f"recorded --policy-action-hz is not numeric: {raw_action_hz!r}"
        ) from exc
    if not math.isfinite(action_hz) or action_hz < 0.0:
        raise RerenderError(
            f"recorded --policy-action-hz must be finite and non-negative, "
            f"got {raw_action_hz!r}"
        )
    if action_hz == 0.0:
        return None
    output_fps = int(round(action_hz))
    if not math.isclose(action_hz, float(output_fps), rel_tol=0.0, abs_tol=1e-12):
        raise RerenderError(
            "recorded --policy-action-hz must be an integer for exact "
            "one-state-row-per-frame replay"
        )
    if source_summary is None:
        raise RerenderError(
            "recorded --policy-action-hz requires the source sim_summary.json "
            "to validate replay cadence"
        )
    if not isinstance(source_summary, dict):
        raise RerenderError(
            "recorded --policy-action-hz requires a JSON-object source summary"
        )
    if str(source_summary.get("robot", "")).strip().lower() != "fr3":
        raise RerenderError(
            "recorded --policy-action-hz replay requires source summary robot='fr3'"
        )
    if source_summary.get("physics") != "dynamic":
        raise RerenderError(
            "recorded --policy-action-hz replay requires source summary "
            "physics='dynamic'"
        )
    recorded_robot = (_last_option_value(argv, "--robot") or "fr3").lower()
    recorded_physics = _last_option_value(argv, "--physics") or "kinematic"
    try:
        recorded_embodiment = robot_embodiment(recorded_robot)
    except ValueError:
        recorded_embodiment = recorded_robot
    if recorded_embodiment != "fr3" or recorded_physics != "dynamic":
        raise RerenderError(
            "recorded --policy-action-hz command must be an FR3 dynamic rollout"
        )
    raw_frame_hz = _last_option_value(argv, "--frame-hz")
    if raw_frame_hz is not None:
        try:
            recorded_frame_hz = float(raw_frame_hz)
        except ValueError as exc:
            raise RerenderError(
                f"recorded --frame-hz is not numeric: {raw_frame_hz!r}"
            ) from exc
        if not math.isfinite(recorded_frame_hz) or recorded_frame_hz != 0.0:
            raise RerenderError(
                "recorded --policy-action-hz requires recorded --frame-hz to "
                "be absent or zero"
            )
    raw_repeats = _last_option_value(argv, "--dynamic-control-repeats")
    if raw_repeats is not None:
        try:
            recorded_repeats = int(raw_repeats)
        except ValueError as exc:
            raise RerenderError(
                "recorded --dynamic-control-repeats must be an integer"
            ) from exc
        if recorded_repeats != 0:
            raise RerenderError(
                "recorded --policy-action-hz requires "
                "--dynamic-control-repeats 0"
            )

    frames_raw = source_summary.get("frames")
    if (
        isinstance(frames_raw, bool)
        or not isinstance(frames_raw, int)
        or frames_raw <= 0
    ):
        raise RerenderError(
            f"source summary frames must be a positive integer, got {frames_raw!r}"
        )
    frames = int(frames_raw)
    raw_recorded_frames = _last_option_value(argv, "--frames")
    try:
        recorded_frames = int(raw_recorded_frames) if raw_recorded_frames else 0
    except ValueError as exc:
        raise RerenderError("recorded --frames must be an integer") from exc
    if recorded_frames != frames:
        raise RerenderError(
            f"recorded --frames={recorded_frames}, but source summary has "
            f"frames={frames}"
        )
    nominal_dt_s = 1.0 / action_hz
    nominal_duration_s = frames / action_hz

    for key in (
        "policy_action_hz_requested",
        "policy_action_hz_effective",
        "frame_hz_effective",
    ):
        _require_summary_close(source_summary, key, action_hz)
    for key in (
        "policy_action_dt_s",
        "policy_action_dt_nominal_s",
        "frame_dt_s",
        "playback_action_dt_s",
    ):
        _require_summary_close(source_summary, key, nominal_dt_s)

    summary_output_fps = source_summary.get("output_fps")
    if (
        isinstance(summary_output_fps, bool)
        or not isinstance(summary_output_fps, int)
        or summary_output_fps != output_fps
    ):
        raise RerenderError(
            f"source summary output_fps={summary_output_fps!r} disagrees with "
            f"--policy-action-hz {action_hz:g}"
        )
    dynamic_subframes = source_summary.get("dynamic_subframes")
    if isinstance(dynamic_subframes, bool) or dynamic_subframes != 1:
        raise RerenderError(
            "source summary dynamic_subframes must be 1 for one action per frame"
        )
    requested_repeats = source_summary.get("dynamic_control_repeats_requested")
    if isinstance(requested_repeats, bool) or requested_repeats != 0:
        raise RerenderError(
            "source summary dynamic_control_repeats_requested must be 0 for "
            "the phase scheduler"
        )
    if source_summary.get("fr3_dynamic_tick_schedule") != (
        "source_to_controller_cumulative_nearest_half_up"
    ):
        raise RerenderError(
            "source summary has no recognized FR3 policy-action tick schedule"
        )
    if (
        source_summary.get("fr3_dynamic_runtime_tick_count_matches_schedule")
        is not True
    ):
        raise RerenderError(
            "source summary does not attest that runtime FR3 ticks match the schedule"
        )

    control_hz = _required_finite_summary_number(
        source_summary, "dynamic_control_hz", positive=True
    )
    raw_recorded_control_hz = _last_option_value(argv, "--dynamic-control-hz")
    try:
        recorded_control_hz = float(raw_recorded_control_hz or 20.0)
    except ValueError as exc:
        raise RerenderError(
            "recorded --dynamic-control-hz must be numeric"
        ) from exc
    if (
        not math.isfinite(recorded_control_hz)
        or not math.isclose(
            recorded_control_hz, control_hz, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise RerenderError(
            f"recorded --dynamic-control-hz={raw_recorded_control_hz or 20.0}, "
            f"but source summary dynamic_control_hz={control_hz:g}"
        )
    if action_hz > control_hz:
        raise RerenderError(
            f"source policy action rate {action_hz:g}Hz exceeds controller "
            f"rate {control_hz:g}Hz"
        )
    tick_count_raw = source_summary.get("fr3_dynamic_control_tick_count_total")
    runtime_tick_count_raw = source_summary.get(
        "fr3_dynamic_runtime_control_tick_count_total"
    )
    if (
        isinstance(tick_count_raw, bool)
        or not isinstance(tick_count_raw, int)
        or tick_count_raw <= 0
        or runtime_tick_count_raw != tick_count_raw
    ):
        raise RerenderError(
            "source summary FR3 scheduled/runtime controller tick totals are invalid"
        )
    expected_tick_count = int(
        math.floor(frames * control_hz / action_hz + 0.5 + 1e-12)
    )
    if tick_count_raw != expected_tick_count:
        raise RerenderError(
            "source summary FR3 controller tick total disagrees with the "
            "recorded cumulative-nearest schedule: "
            f"{tick_count_raw} != {expected_tick_count}"
        )
    simulated_duration_s = tick_count_raw / control_hz
    _require_summary_close(
        source_summary,
        "fr3_dynamic_source_duration_s",
        nominal_duration_s,
        tolerance=1e-8,
    )
    _require_summary_close(
        source_summary,
        "fr3_dynamic_simulated_duration_s",
        simulated_duration_s,
        tolerance=1e-8,
    )
    _require_summary_close(
        source_summary,
        "fr3_dynamic_timing_error_s",
        simulated_duration_s - nominal_duration_s,
        tolerance=1e-8,
    )
    _require_summary_close(
        source_summary,
        "dynamic_action_sim_dt_s",
        simulated_duration_s / frames,
        tolerance=1e-8,
    )
    phase_bound_s = 0.5 / control_hz
    _require_summary_close(
        source_summary,
        "fr3_dynamic_boundary_phase_error_bound_s",
        phase_bound_s,
        tolerance=1e-8,
    )
    phase_max_abs_s = _required_finite_summary_number(
        source_summary, "fr3_dynamic_boundary_phase_error_s_max_abs"
    )
    if phase_max_abs_s < 0.0 or phase_max_abs_s > phase_bound_s + 1e-8:
        raise RerenderError(
            "source summary FR3 boundary phase-error magnitude is outside "
            f"[0, {phase_bound_s:g}]s: {phase_max_abs_s:g}s"
        )
    if abs(simulated_duration_s - nominal_duration_s) > phase_bound_s + 1e-8:
        raise RerenderError(
            "source summary FR3 total timing error exceeds the cumulative-"
            f"nearest boundary bound: {simulated_duration_s - nominal_duration_s:g}s"
        )

    return {
        "schema_version": 1,
        "kind": "fr3-policy-action-to-exact-state-replay",
        "source_policy_action_hz": action_hz,
        "source_policy_action_dt_nominal_s": nominal_dt_s,
        "source_frames": frames,
        "source_nominal_duration_s": nominal_duration_s,
        "source_control_hz": control_hz,
        "source_control_tick_count": tick_count_raw,
        "source_simulated_duration_s": simulated_duration_s,
        "source_boundary_phase_error_max_abs_s": phase_max_abs_s,
        "replay_frame_hz": action_hz,
        "replay_output_fps": output_fps,
        "replay_expected_frames": frames,
        "replay_expected_duration_s": nominal_duration_s,
        "mapping": "one saved state row to one encoded frame",
    }


def _require_decoded_timing(
    video_validation: dict[str, Any],
    *,
    expected_fps: int,
    expected_hz: float,
    expected_duration_s: float,
) -> dict[str, Any]:
    """Check the decoded MP4 clock against a source-derived expectation."""

    timing = video_validation.get("timing")
    if not isinstance(timing, dict):
        raise RerenderError("video validation has no decoded timing contract")
    for key, expected in (
        ("summary_output_fps", expected_fps),
        ("decoded_fps", expected_hz),
        ("expected_duration_s", expected_duration_s),
        ("decoded_duration_s", expected_duration_s),
    ):
        raw = timing.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RerenderError(f"video validation timing {key} must be numeric")
        observed = float(raw)
        # Keep this independent of tolerance fields in the generated report;
        # a provenance gate must not be weakened by mutable reported limits.
        tolerance = (
            0.05
            if "fps" in key
            else max(0.001, expected_duration_s * 0.002)
        )
        if not math.isfinite(observed) or not math.isclose(
            observed, expected, rel_tol=0.0, abs_tol=tolerance
        ):
            raise RerenderError(
                f"video validation timing {key}={raw!r}, expected {expected:g}"
            )
    return timing


def _validate_policy_action_replay_cadence(
    source: SourceRun,
    replay_summary: dict[str, Any],
    video_validation: dict[str, Any],
) -> dict[str, Any] | None:
    """Attest that a saved-state render preserves its source action clock."""

    contract = source.source_records.get("policy_action_playback_contract")
    if contract is None:
        return None
    if not isinstance(contract, dict):
        raise RerenderError("source policy-action playback contract is invalid")
    expected_hz = float(contract["replay_frame_hz"])
    expected_fps = int(contract["replay_output_fps"])
    expected_frames = int(contract["replay_expected_frames"])
    expected_duration_s = float(contract["replay_expected_duration_s"])
    expected_dt_s = 1.0 / expected_hz

    required_keys = (
        "robot",
        "physics",
        "frames",
        "rendered_frames",
        "state_log_replay",
        "replay_arm_source",
        "replay_object_source",
        "policy_action_hz_requested",
        "policy_action_hz_effective",
        "policy_action_dt_s",
        "policy_action_dt_nominal_s",
        "frame_hz_effective",
        "frame_dt_s",
        "output_fps",
        "playback_action_dt_s",
        "dynamic_subframes",
        "dynamic_action_sim_dt_s",
        "dynamic_control_repeats",
    )
    missing = [key for key in required_keys if key not in replay_summary]
    if missing:
        raise RerenderError(
            "rerender summary is missing policy-action cadence fields: "
            + ", ".join(missing)
        )
    expected_state_log = (source.run_dir / "logs" / "sim_log.jsonl").resolve()
    try:
        replay_state_log = Path(str(replay_summary["state_log_replay"])).resolve()
    except (OSError, TypeError, ValueError) as exc:
        raise RerenderError("rerender summary state_log_replay is invalid") from exc
    identity_mismatches = []
    if str(replay_summary["robot"]).lower() != "fr3":
        identity_mismatches.append("robot")
    if replay_summary["physics"] != "dynamic":
        identity_mismatches.append("physics")
    if replay_state_log != expected_state_log:
        identity_mismatches.append("state_log_replay")
    if replay_summary["replay_arm_source"] != "generation_state_log":
        identity_mismatches.append("replay_arm_source")
    if replay_summary["replay_object_source"] != "generation_state_log":
        identity_mismatches.append("replay_object_source")
    if identity_mismatches:
        raise RerenderError(
            "rerender summary is not the expected inert FR3 state replay: "
            + ", ".join(identity_mismatches)
        )

    for key in ("frames", "rendered_frames"):
        value = replay_summary[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value != expected_frames
        ):
            raise RerenderError(
                f"rerender summary {key}={value!r}, expected {expected_frames}"
            )
    output_fps = replay_summary["output_fps"]
    if (
        isinstance(output_fps, bool)
        or not isinstance(output_fps, int)
        or output_fps != expected_fps
    ):
        raise RerenderError(
            f"rerender summary output_fps={output_fps!r}, expected {expected_fps}"
        )
    if (
        isinstance(replay_summary["dynamic_subframes"], bool)
        or replay_summary["dynamic_subframes"] != 1
    ):
        raise RerenderError("rerender summary dynamic_subframes must be 1")
    if replay_summary["dynamic_action_sim_dt_s"] is not None:
        raise RerenderError(
            "rerender summary dynamic_action_sim_dt_s must be null for inert state replay"
        )
    if replay_summary["dynamic_control_repeats"] is not None:
        raise RerenderError(
            "rerender summary dynamic_control_repeats must be null for inert state replay"
        )
    if (
        isinstance(replay_summary["policy_action_hz_requested"], bool)
        or replay_summary["policy_action_hz_requested"] != 0.0
    ):
        raise RerenderError(
            "rerender retained live policy_action_hz_requested instead of converting it"
        )
    for key in (
        "policy_action_hz_effective",
        "policy_action_dt_s",
        "policy_action_dt_nominal_s",
    ):
        if replay_summary[key] is not None:
            raise RerenderError(f"rerender summary {key} must be null")
    for key, expected in (
        ("frame_hz_effective", expected_hz),
        ("frame_dt_s", expected_dt_s),
        ("playback_action_dt_s", expected_dt_s),
    ):
        raw = replay_summary[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RerenderError(f"rerender summary {key} must be numeric")
        observed = float(raw)
        if not math.isfinite(observed) or not math.isclose(
            observed, expected, rel_tol=0.0, abs_tol=1e-6
        ):
            raise RerenderError(
                f"rerender summary {key}={raw!r}, expected {expected:g}"
            )

    timing = _require_decoded_timing(
        video_validation,
        expected_fps=expected_fps,
        expected_hz=expected_hz,
        expected_duration_s=expected_duration_s,
    )

    return {
        **contract,
        "replay_summary": {
            "frame_hz_effective": float(replay_summary["frame_hz_effective"]),
            "frame_dt_s": float(replay_summary["frame_dt_s"]),
            "output_fps": output_fps,
            "frames": int(replay_summary["frames"]),
            "rendered_frames": int(replay_summary["rendered_frames"]),
            "playback_action_dt_s": float(
                replay_summary["playback_action_dt_s"]
            ),
            "state_log_replay": str(replay_state_log),
        },
        "decoded_video": {
            "fps": float(timing["decoded_fps"]),
            "frames": int(video_validation["decoded_frames"]),
            "duration_s": float(timing["decoded_duration_s"]),
        },
        "no_time_stretch": True,
    }


def _frame_hz_playback_contract(
    argv: list[str], source_summary: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Bind a recorded ``--frame-hz`` replay to the source run's own clock.

    The replay keeps the recorded ``--frame-hz`` and presents one saved state
    row per encoded frame at exactly that rate.  The generation summary
    recorded how long each of those rows lasted (``playback_action_dt_s``),
    so that value, not the rerender's own summary, is what the replay must
    reproduce.  Commands without a positive ``--frame-hz`` return ``None``
    unchanged; a positive ``--policy-action-hz`` has its own contract above,
    which requires ``--frame-hz`` to be absent or zero.
    """

    raw_frame_hz = _last_option_value(argv, "--frame-hz")
    if raw_frame_hz is None:
        return None
    try:
        frame_hz = float(raw_frame_hz)
    except ValueError as exc:
        raise RerenderError(
            f"recorded --frame-hz is not numeric: {raw_frame_hz!r}"
        ) from exc
    if not math.isfinite(frame_hz) or frame_hz < 0.0:
        raise RerenderError(
            f"recorded --frame-hz must be finite and non-negative, got {raw_frame_hz!r}"
        )
    if frame_hz == 0.0:
        return None
    try:
        replay_clock = exact_state_replay_playback_clock(frame_hz)
    except ValueError as exc:
        raise RerenderError(
            f"recorded --frame-hz={raw_frame_hz} cannot be replayed exactly: {exc}"
        ) from exc
    if not isinstance(source_summary, dict):
        raise RerenderError(
            "recorded --frame-hz requires the source sim_summary.json to "
            "validate replay timing"
        )
    frames = source_summary.get("frames")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
        raise RerenderError(
            f"source summary frames must be a positive integer, got {frames!r}"
        )
    source_dt_s = _required_finite_summary_number(
        source_summary, "playback_action_dt_s", positive=True
    )
    # Summaries round periods to 1e-6 s.
    if not math.isclose(
        source_dt_s, replay_clock.frame_dt_s, rel_tol=0.0, abs_tol=1e-6
    ):
        raise RerenderError(
            f"source summary playback_action_dt_s={source_dt_s:g}s, but the "
            f"replay of recorded --frame-hz {frame_hz:g} would show each saved "
            f"row for {replay_clock.frame_dt_s:g}s"
        )
    return {
        "schema_version": 1,
        "kind": "frame-hz-to-exact-state-replay",
        "source_frame_hz": frame_hz,
        "source_playback_dt_s": source_dt_s,
        "source_frames": frames,
        "replay_output_fps": replay_clock.output_fps,
        "replay_frame_dt_s": replay_clock.frame_dt_s,
        "replay_expected_duration_s": frames * source_dt_s,
        "mapping": "one saved state row to one encoded frame",
    }


def _validate_frame_hz_replay_cadence(
    source: SourceRun,
    replay_summary: dict[str, Any],
    video_validation: dict[str, Any],
) -> dict[str, Any] | None:
    """Attest that a ``--frame-hz`` replay keeps the source presentation clock.

    ``validate_video`` already compares the MP4 with the rerender summary;
    a replay that is wrong but self-consistent passes that.  This compares
    both with the source run instead.
    """

    if source.source_records.get("policy_action_playback_contract") is not None:
        return None
    contract = _frame_hz_playback_contract(source.argv, source.summary)
    if contract is None:
        return None
    expected_fps = int(contract["replay_output_fps"])
    expected_dt_s = float(contract["source_playback_dt_s"])
    expected_frames = int(contract["source_frames"])
    expected_duration_s = float(contract["replay_expected_duration_s"])

    for key in ("frames", "rendered_frames"):
        value = replay_summary.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value != expected_frames
        ):
            raise RerenderError(
                f"rerender summary {key}={value!r}, but the source run has "
                f"{expected_frames} frames"
            )
    output_fps = replay_summary.get("output_fps")
    if (
        isinstance(output_fps, bool)
        or not isinstance(output_fps, (int, float))
        or output_fps != expected_fps
    ):
        raise RerenderError(
            f"rerender summary output_fps={output_fps!r}, but the source run "
            f"presents {expected_fps} saved rows per second"
        )
    for key, expected in (
        ("playback_action_dt_s", expected_dt_s),
        ("frame_dt_s", expected_dt_s),
        ("frame_hz_effective", float(contract["source_frame_hz"])),
    ):
        raw = replay_summary.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RerenderError(f"rerender summary {key} must be numeric")
        if not math.isfinite(float(raw)) or not math.isclose(
            float(raw), expected, rel_tol=0.0, abs_tol=1e-6
        ):
            raise RerenderError(
                f"rerender summary {key}={raw!r}, but the source run implies "
                f"{expected:g}"
            )
    timing = _require_decoded_timing(
        video_validation,
        expected_fps=expected_fps,
        expected_hz=1.0 / expected_dt_s,
        expected_duration_s=expected_duration_s,
    )
    return {
        **contract,
        "replay_summary": {
            "frames": int(replay_summary["frames"]),
            "rendered_frames": int(replay_summary["rendered_frames"]),
            "output_fps": output_fps,
            "playback_action_dt_s": float(replay_summary["playback_action_dt_s"]),
            "frame_dt_s": float(replay_summary["frame_dt_s"]),
            "frame_hz_effective": float(replay_summary["frame_hz_effective"]),
        },
        "decoded_video": {
            "fps": float(timing["decoded_fps"]),
            "frames": int(video_validation["decoded_frames"]),
            "duration_s": float(timing["decoded_duration_s"]),
        },
        "no_time_stretch": True,
    }
