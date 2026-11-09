# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Scene-input identity recorded in, and checked against, init states.

A deterministic init state freezes one run's settled object poses. Those poses
depend on data files as much as on source: the hand-authored layout or measured
pose contract that places the objects, the initial arm pose, and the base
calibration. The batch runner's cache name hashes source code; this module
names the data, so the name changes when a layout does.

Loading is the other half. An explicit ``--init-state-in`` bypasses the cache
name entirely, and the compiled-model signature does not cover object poses.
The identity therefore also records where the environment seated each object
at reset -- ``layout_seating_report``: every layout-placed object and the
cup/plate pair, its support, its authored z and the z it was seated at on the
compiled table. A load compares both the resolved inputs and that record with
the current run, so a state written against another layout, another seating
rule or another table is refused rather than restored over this run's start.

The seating report, not the ready-object registry, is the pose record on
purpose: the registry is rebound to post-drive poses for some tasks, and only
on the path that writes a state, so a writer and a loader would disagree.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

INIT_IDENTITY_KEY = "init_identity"
INIT_IDENTITY_SCHEMA = "real2sim.init_state_identity.v1"
#: A seated height is the same height when it agrees to a micrometre. Seating
#: the same model is exact; the slack only absorbs last-bit floating point
#: differences between hosts, three orders of magnitude below any seating,
#: table or layout change that matters.
SEATED_Z_TOLERANCE_M = 1e-6


class InitIdentityError(ValueError):
    """A run's scene inputs cannot be resolved, or a state does not match them."""


def _text(arguments: Any, name: str) -> str:
    return str(getattr(arguments, name, "") or "").strip()


def _file_record(raw: str | os.PathLike[str], role: str) -> dict[str, str]:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(raw))))
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise InitIdentityError(
            f"scene input {role} is unreadable: {path}: {exc}"
        ) from exc
    return {"path": str(path), "sha256": digest}


def _value_record(value: Any) -> dict[str, str]:
    return {"value": str(value)}


def _selected_layout(arguments: Any, task: str, robot: str) -> Path:
    """The container / box layout, through the selection every consumer uses.

    The worker flag, then the harness-only ``RS_*_LAYOUT_JSON`` override, then
    the task default (``tasks/layout_selection.py``): the file the environment
    loads, so a state is never reused across layouts.
    """

    from robosuite_env.tasks.layout_selection import (
        LAYOUT_PARAMETERS,
        LayoutError,
        select_task_layout,
    )

    try:
        return select_task_layout(
            task, _text(arguments, LAYOUT_PARAMETERS[task]), robot=robot
        ).path
    except LayoutError as exc:
        raise InitIdentityError(str(exc)) from exc


def resolve_scene_inputs(arguments: Any, *, robot: str) -> dict[str, dict[str, str]]:
    """Resolve the data files and selectors that place this run's objects.

    ``arguments`` is the worker's parsed argument namespace. Each task resolves
    its layout exactly as its environment does, including the ``RS_*``
    overrides those environments honour.
    """

    objects = _text(arguments, "objects")
    inputs: dict[str, dict[str, str]] = {}
    if objects.endswith(".json"):
        inputs["objects_positions"] = _file_record(objects, "objects_positions")
    elif objects == "cup_plate":
        # tasks/envs/cup_plate*.py place a named cup_plate run from these.
        for name in ("RS_CUP_XY", "RS_PLATE_XY"):
            if os.environ.get(name):
                inputs[f"env:{name}"] = _value_record(os.environ[name])
    elif objects == "box_pack":
        inputs["manual_layout"] = _file_record(
            _selected_layout(arguments, objects, robot), "manual_layout"
        )
        inputs["box_pack_task_index"] = _value_record(
            int(getattr(arguments, "box_pack_task_index", 0) or 0)
        )

    initial_arm_joints = _text(arguments, "initial_arm_joints_json")
    if initial_arm_joints:
        inputs["initial_arm_joints"] = _file_record(
            initial_arm_joints, "initial_arm_joints"
        )

    from robosuite_env.robots.base_placement import (
        BasePlacementError,
        default_base_placement_path,
    )

    base_placement = _text(arguments, "base_placement")
    if not base_placement:
        try:
            base_placement = os.fspath(default_base_placement_path(robot))
        except (BasePlacementError, ValueError) as exc:
            raise InitIdentityError(str(exc)) from exc
    inputs["base_placement"] = _file_record(base_placement, "base_placement")
    return inputs


def _inputs_sha256(inputs: Mapping[str, Mapping[str, Any]]) -> str:
    """Hash content and selector values only, so a checkout can move."""

    canonical = {
        role: record.get("sha256", record.get("value"))
        for role, record in inputs.items()
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def scene_input_identity(arguments: Any, *, robot: str) -> dict[str, Any]:
    """Return the resolved scene inputs and their location-independent digest."""

    inputs = resolve_scene_inputs(arguments, robot=robot)
    return {
        "schema": INIT_IDENTITY_SCHEMA,
        "inputs": inputs,
        "inputs_sha256": _inputs_sha256(inputs),
    }


def seated_objects(seating_report: Any) -> list[dict[str, Any]] | None:
    """Canonicalize the environment's layout seating report.

    ``seating_report`` is ``env.layout_seating_report`` (see
    ``CupPlateLiftGW._seat_layout_placements``); ``None`` means the
    environment seats nothing.
    """

    if seating_report is None:
        return None
    if isinstance(seating_report, (str, bytes)) or not isinstance(
        seating_report, Sequence
    ):
        raise InitIdentityError("the environment's layout seating report is not a list")
    rows: list[dict[str, Any]] = []
    for row in seating_report:
        if not isinstance(row, Mapping):
            raise InitIdentityError("a layout seating report row is not a mapping")
        try:
            record = {
                "object": str(row["object"]),
                "support": str(row["support"]),
                "z_before_m": float(row["z_before_m"]),
                "z_seated_m": float(row["z_seated_m"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise InitIdentityError(
                f"a layout seating report row is malformed: {exc!r}"
            ) from exc
        if not (
            math.isfinite(record["z_before_m"]) and math.isfinite(record["z_seated_m"])
        ):
            raise InitIdentityError(
                f"layout seating report for {record['object']!r} is not finite"
            )
        rows.append(record)
    return sorted(rows, key=lambda record: record["object"])


def init_state_identity(
    arguments: Any,
    *,
    robot: str,
    seating_report: Any,
) -> dict[str, Any]:
    """Return the full identity a written init state records."""

    identity = scene_input_identity(arguments, robot=robot)
    identity["seated_objects"] = seated_objects(seating_report)
    return identity


def _describe(record: Mapping[str, Any]) -> str:
    if "sha256" in record:
        return f"{record.get('path')} (sha256 {str(record['sha256'])[:12]})"
    return repr(record.get("value"))


def _seating_mismatches(recorded: Any, current: list[dict[str, Any]] | None) -> list[str]:
    if recorded is None or current is None:
        if recorded is None and current is None:
            return []
        written = "no seating" if recorded is None else f"{len(recorded)} seated objects"
        now = "no seating" if current is None else f"{len(current)} seated objects"
        return [f"seated objects: written with {written}, this run has {now}"]
    if not isinstance(recorded, list) or any(
        not isinstance(row, Mapping) or "object" not in row for row in recorded
    ):
        return ["its recorded seated objects are malformed"]
    before = {str(row["object"]): row for row in recorded}
    now = {row["object"]: row for row in current}
    failures: list[str] = []
    if set(before) != set(now):
        failures.append(
            f"seated objects differ: written with {sorted(before)}, this run "
            f"seats {sorted(now)}"
        )
    for name in sorted(set(before) & set(now)):
        written, current_row = before[name], now[name]
        if str(written.get("support")) != current_row["support"]:
            failures.append(
                f"{name} was seated on {written.get('support')!r}, this run "
                f"seats it on {current_row['support']!r}"
            )
        for key, label in (("z_seated_m", "seated"), ("z_before_m", "authored")):
            try:
                value = float(written.get(key))
            except (TypeError, ValueError):
                failures.append(f"{name} recorded {key} is malformed")
                continue
            if (
                not math.isfinite(value)
                or abs(value - current_row[key]) > SEATED_Z_TOLERANCE_M
            ):
                failures.append(
                    f"{name} {label} z differs by "
                    f"{(value - current_row[key]) * 1000.0:+.3f} mm (written "
                    f"{value:.6f} m, this run {current_row[key]:.6f} m)"
                )
    return failures


def identity_mismatches(
    recorded: Any,
    current: Mapping[str, Any],
) -> list[str]:
    """Explain every way a recorded identity differs from the current run."""

    if not isinstance(recorded, Mapping):
        return [
            "it records no scene-input identity (it was written before init "
            "states were bound to their layout and seated start), so nothing "
            "shows its objects start where this run places them"
        ]
    failures: list[str] = []
    if recorded.get("schema") != current["schema"]:
        failures.append(
            f"identity schema is {recorded.get('schema')!r}, expected "
            f"{current['schema']!r}"
        )
    recorded_inputs = recorded.get("inputs")
    if not isinstance(recorded_inputs, Mapping) or any(
        not isinstance(record, Mapping) for record in recorded_inputs.values()
    ):
        failures.append("its recorded scene inputs are malformed")
        recorded_inputs = {}
    elif recorded.get("inputs_sha256") != _inputs_sha256(recorded_inputs):
        failures.append("its recorded inputs_sha256 does not match its inputs")
    current_inputs = current["inputs"]
    for role in sorted(set(recorded_inputs) | set(current_inputs)):
        before = recorded_inputs.get(role)
        now = current_inputs.get(role)
        if before is None:
            failures.append(f"{role}: none when written, this run uses {_describe(now)}")
        elif now is None:
            failures.append(f"{role}: written with {_describe(before)}, this run uses none")
        elif before.get("sha256", before.get("value")) != now.get(
            "sha256", now.get("value")
        ):
            failures.append(
                f"{role}: written with {_describe(before)}, this run uses "
                f"{_describe(now)}"
            )
    if "seated_objects" not in recorded:
        failures.append("it records no seated objects")
    else:
        failures.extend(
            _seating_mismatches(recorded["seated_objects"], current["seated_objects"])
        )
    return failures


def validate_init_state_identity(
    payload: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    source: str,
) -> None:
    """Refuse a saved state whose scene identity is not the current run's."""

    failures = identity_mismatches(payload.get(INIT_IDENTITY_KEY), current)
    if failures:
        raise InitIdentityError(
            f"{source} does not belong to this run's scene: "
            + "; ".join(failures)
            + ". Restoring it would start the objects from another layout or "
            "seating. Delete it, or rebuild it for this run with --init-state-out "
            "--init-state-only."
        )


def _worker_arguments(argv: Sequence[str]) -> Any:
    from robosuite_env.rollout.cli import parse

    arguments = parse(list(argv))
    # Mirror robosuite_env/rollout/setup/args.py: a policy run on the named
    # cup_plate task loads the standard positions file instead.
    if (
        (arguments.policy or arguments.policy_batch_server)
        and arguments.objects == "cup_plate"
    ):
        from robosuite_env.paths import PACKAGE_ROOT

        standard = PACKAGE_ROOT / "obj_calib" / "scene_positions_sam3.json"
        if standard.is_file():
            arguments.objects = os.fspath(standard)
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    """Print the scene-input digest for one worker argument vector.

    Usage: ``python -m robosuite_env.rollout.state.init_identity [--json] --
    <worker arguments>``. Run from the repository root, as the worker is.
    """

    raw = list(sys.argv[1:] if argv is None else argv)
    if "--" not in raw:
        print(
            "usage: python -m robosuite_env.rollout.state.init_identity "
            "[--json] -- <worker arguments>",
            file=sys.stderr,
        )
        return 2
    split = raw.index("--")
    options, worker_argv = raw[:split], raw[split + 1 :]
    if set(options) - {"--json"}:
        print(f"unknown option(s): {sorted(set(options) - {'--json'})}", file=sys.stderr)
        return 2
    try:
        arguments = _worker_arguments(worker_argv)
        identity = scene_input_identity(arguments, robot=arguments.robot)
    except InitIdentityError as exc:
        print(f"[init-identity] {exc}", file=sys.stderr)
        return 2
    if "--json" in options:
        print(json.dumps(identity, indent=2, sort_keys=True))
    else:
        print(identity["inputs_sha256"], flush=True)
    return 0


__all__ = [
    "INIT_IDENTITY_KEY",
    "INIT_IDENTITY_SCHEMA",
    "InitIdentityError",
    "SEATED_Z_TOLERANCE_M",
    "identity_mismatches",
    "init_state_identity",
    "resolve_scene_inputs",
    "scene_input_identity",
    "seated_objects",
    "validate_init_state_identity",
]


if __name__ == "__main__":
    raise SystemExit(main())
