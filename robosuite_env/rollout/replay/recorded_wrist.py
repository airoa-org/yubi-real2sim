# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The wrist camera a recorded run used, for every path that replays its state log.

Rerender (``rerender.replay_argv``) and a direct
``python -m robosuite_env.sim --state-log-replay`` both resolve it here, so a replay renders the camera the run recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

from real2sim.wrist_profiles import FRANKA_PINHOLE_WRIST_RGB_PROFILE, WRIST_BACKENDS

#: Why a replay refuses a run whose recorded wrist backend this checkout does not render.
UNSUPPORTED_RECORDED_WRIST_BACKEND = (
    "this run did not record a supported wrist backend; it was recorded before "
    "this repository was published, and this repository cannot rerender it"
)


class RecordedWristError(ValueError):
    """The run's recorded wrist camera is unknown or contradicts the request."""


def log_source_summary(sim_log: Path) -> dict | None:
    """The ``sim_summary.json`` the worker writes beside its state log, if readable."""

    try:
        with (Path(sim_log).resolve().parent / "sim_summary.json").open(encoding="utf-8") as stream:
            summary = json.load(stream)
    except (OSError, ValueError):
        return None
    return summary if isinstance(summary, dict) else None


def recorded_value(source_summary: dict | None, sim_log: Path, key: str) -> str | None:
    """``key`` from the source summary, else from the first state-log policy observation."""

    if isinstance(source_summary, dict) and source_summary.get(key):
        return str(source_summary[key])
    try:
        with Path(sim_log).open(encoding="utf-8") as rows:
            for number, line in enumerate(rows, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RecordedWristError(f"{sim_log}:{number}: {exc}") from exc
                policy = row.get("policy") if isinstance(row, dict) else None
                obs_log = policy.get("obs_log") if isinstance(policy, dict) else None
                if isinstance(obs_log, dict) and obs_log.get(key):
                    return str(obs_log[key])
    except OSError as exc:
        raise RecordedWristError(f"cannot read the state log {sim_log}: {exc}") from exc
    return None


def recorded_wrist_profile(source_summary: dict | None, sim_log: Path) -> str:
    """The run's profile; a run that records none used ``franka-pinhole-wrist-rgb``, the only camera then."""

    key = "policy_yubi_render_profile"
    profile = recorded_value(source_summary, sim_log, key)
    if profile:
        return profile
    # Runs recorded before policy_wrist_intrinsics existed record null without a policy.
    # A summary with intrinsics but no profile names no camera.
    if isinstance(source_summary, dict) and "policy_wrist_intrinsics" in source_summary:
        raise RecordedWristError(f"the source summary records no {key}, so its wrist camera is unknown; re-record the run")
    return FRANKA_PINHOLE_WRIST_RGB_PROFILE


def recorded_wrist_backend(source_summary: dict | None, sim_log: Path) -> str:
    """The run's backend; a run that records none, or one this checkout does not render, is refused."""

    key = "policy_wrist_backend"
    backend = recorded_value(source_summary, sim_log, key)
    if not backend and isinstance(source_summary, dict) and key in source_summary:  # only unreleased development builds stored null
        raise RecordedWristError(f"the source summary records a null {key}, so its wrist camera is unknown; re-record the run")
    if backend not in WRIST_BACKENDS:
        raise RecordedWristError(UNSUPPORTED_RECORDED_WRIST_BACKEND)
    return backend


def recorded_gs_kb4_near_cull(source_summary: dict | None, sim_log: Path) -> str:
    """The run's KB4 near cull; a run that records none rendered with the fixed 0.2 GS-unit cull (``legacy``)."""

    key = "gs_kb4_near_cull"
    mode = recorded_value(source_summary, sim_log, key)
    if mode:
        return mode
    if isinstance(source_summary, dict) and key in source_summary:
        raise RecordedWristError(f"the source summary records a null {key}, so its 3DGS near cull is unknown; re-record the run")
    return "legacy"


#: (argument dest, flag, resolver) of every wrist option a state-log replay takes from the run.
RECORDED_WRIST_OPTIONS = (
    ("policy_yubi_render_profile", "--policy-yubi-render-profile", recorded_wrist_profile),
    ("policy_wrist_backend", "--policy-wrist-backend", recorded_wrist_backend),
    ("gs_kb4_near_cull", "--gs-kb4-near-cull", recorded_gs_kb4_near_cull),
)


def resolve_state_log_wrist_options(arguments) -> None:
    """Give a ``--state-log-replay`` the run's recorded wrist camera.

    An omitted flag takes the recorded value; an explicit flag must equal it.
    ``arguments._explicit_wrist_options`` names the flags given on the command line.
    """

    if not getattr(arguments, "state_log_replay", None):
        return
    sim_log = Path(arguments.state_log_replay)
    summary = log_source_summary(sim_log)
    explicit = getattr(arguments, "_explicit_wrist_options", frozenset())
    for dest, flag, resolve in RECORDED_WRIST_OPTIONS:
        recorded = resolve(summary, sim_log)
        if dest in explicit and getattr(arguments, dest) != recorded:
            raise RecordedWristError(
                f"{flag} {getattr(arguments, dest)} contradicts the replayed run, which recorded {recorded}; "
                "omit the flag to use the recorded value"
            )
        setattr(arguments, dest, recorded)
