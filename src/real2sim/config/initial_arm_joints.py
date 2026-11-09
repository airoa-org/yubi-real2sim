# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Named registry of the checked-in FR3 initial arm-joint poses.

A profile selects one of these by name::

    simulation:
      parameters:
        initial_arm_joints: grip07

The name is resolved here, in the launcher, into the same absolute path the
historical ``simulation.parameters.initial_arm_joints_json`` produced, so the
worker argument vector stays ``--initial-arm-joints-json <path>`` either way.
Every consumer of that argument -- the schema-2 pose loader, the joint ready
gate, and the settled init-state contract -- is unaffected by
which of the two spellings a profile used.

Both spellings are accepted, but never together: a profile that sets a name and
a path at the same time is ambiguous and is rejected rather than resolved by
precedence.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path

from real2sim._bootstrap import REPO_ROOT
from real2sim.config._types import ConfigError


#: Registered pose name -> repository-relative JSON path.
#:
#: ``fr3/flat/registered_cup_calibrated``
#:     The calibrated cup_plate ready pose. schema_version 1, arms only, so a
#:     rollout that selects it settles the gripper from the model state.
#: ``fr3/flat/grip07``
#:     Captured on the real robot from the real-robot deployment configuration
#:     as three ROS ``joint_trajectory`` messages: ``fr3_joint1..7`` (left arm),
#:     ``right_fr3_joint1..7`` (right arm), and ``hand_left_motor_joint`` /
#:     ``hand_right_motor_joint`` (gripper).  schema_version 2, so it also
#:     declares the initial gripper angle in raw motor radians.
#: ``fr3/duo/home``
#:     The FR3 Duo shared-mount start pose, taught on the rig rather than
#:     configured, so the two arms hold different joint values.
#:     schema_version 2. ``robosuite_env/obj_calib/README_duo_home_pose.md``
#:     explains how it differs from the symmetric textbook pose.
#: ``fr3/duo/home_20260906``
#:     A Duo start pose read off the live arms, with both grippers at 0.7 rad.
#:     schema_version 2.
#: ``fr3/duo/init_high``
#:     A raised Duo start pose (both arms lifted well clear of the table,
#:     asymmetric per arm), supplied directly rather than taught or captured
#:     from the rig. schema_version 2, grippers at 0.7313 / 0.7466 rad.
#:
#: A registered pose is never changed in place: a recorded rollout resolves
#: its pose by name, so a new pose gets a new name.
#: Every pose lives at ``assets/initial_poses/<robot>/<rig>/<name>.json`` and
#: is registered under ``<robot>/<rig>/<name>``: ``fr3/duo`` and ``fr3/flat``.
#: The file's ``signature`` block names the same robot and rig and
#: the scenes the pose is used with, and the launcher refuses a profile whose
#: ``robot:`` belongs to another rig (``check_pose_signature``).
INITIAL_ARM_JOINT_POSES: dict[str, str] = {
    "fr3/duo/init_high": "assets/initial_poses/fr3/duo/init_high.json",
    "fr3/duo/home": "assets/initial_poses/fr3/duo/home.json",
    "fr3/duo/home_20260906": "assets/initial_poses/fr3/duo/home_20260906.json",
    "fr3/flat/grip07": "assets/initial_poses/fr3/flat/grip07.json",
    "fr3/flat/registered_cup_calibrated": "assets/initial_poses/fr3/flat/registered_cup_calibrated.json",
}

#: Alternative names that resolve to the same registered pose.
LEGACY_POSE_NAMES: dict[str, str] = {
    "duo_init_high": "fr3/duo/init_high",
    "grip07": "fr3/flat/grip07",
    "registered_cup_calibrated": "fr3/flat/registered_cup_calibrated",
    "fr3/franka_flat/grip07": "fr3/flat/grip07",
}

NAME_FIELD = "simulation.parameters.initial_arm_joints"
PATH_FIELD = "simulation.parameters.initial_arm_joints_json"


def initial_arm_joint_pose_names() -> tuple[str, ...]:
    """Return every registered pose name, in a stable order."""

    return tuple(sorted(INITIAL_ARM_JOINT_POSES))


def canonical_pose_name(name: str) -> str:
    """Return the ``<robot>/<rig>/<name>`` spelling of a pose name."""

    return LEGACY_POSE_NAMES.get(str(name), str(name))


def initial_arm_joint_pose_path(name: str) -> Path:
    """Resolve one registered pose name to its absolute repository path."""

    key = str(name)
    relative = INITIAL_ARM_JOINT_POSES.get(canonical_pose_name(key))
    if relative is None:
        suggestions = difflib.get_close_matches(
            key,
            (*initial_arm_joint_pose_names(), *LEGACY_POSE_NAMES),
            n=3,
            cutoff=0.5,
        )
        error = ConfigError(
            f"{NAME_FIELD} is not a registered initial arm joint pose: {key!r}",
            path=NAME_FIELD,
            allowed=list(initial_arm_joint_pose_names()),
            got=key,
        )
        if suggestions:
            error.suggestions = tuple(suggestions)
        raise error
    return Path(os.path.abspath(REPO_ROOT / relative))


def resolve_initial_arm_joint_pose(name: str | None, path: str | None) -> str | None:
    """Return the absolute pose path a profile selected, or ``None``.

    ``name`` is the registry spelling and ``path`` the already-resolved
    historical path spelling. Exactly one of them may be set.
    """

    if name is not None and path is not None:
        raise ConfigError(
            f"{NAME_FIELD} and {PATH_FIELD} both select an initial arm joint "
            "pose; set exactly one",
            path=NAME_FIELD,
            allowed=list(initial_arm_joint_pose_names()),
            got=[name, path],
        )
    if name is not None:
        return str(initial_arm_joint_pose_path(name))
    # A profile written before the poses moved still names the old path.
    from robosuite_env.robots.legacy_pose_paths import redirect_legacy_pose_path

    return redirect_legacy_pose_path(path)


def format_initial_arm_joint_pose_table() -> str:
    """Return the operator-facing ``name -> repository path`` listing."""

    names = initial_arm_joint_pose_names()
    width = max((len(name) for name in names), default=0)
    lines = [f"{NAME_FIELD} accepts:"]
    lines.extend(
        f"  {name.ljust(width)}  {INITIAL_ARM_JOINT_POSES[name]}" for name in names
    )
    lines.append("older names, still accepted:")
    old_width = max((len(name) for name in LEGACY_POSE_NAMES), default=0)
    lines.extend(
        f"  {old.ljust(old_width)}  -> {new}"
        for old, new in sorted(LEGACY_POSE_NAMES.items())
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "INITIAL_ARM_JOINT_POSES",
    "LEGACY_POSE_NAMES",
    "canonical_pose_name",
    "check_pose_signature",
    "read_pose_signature",
    "NAME_FIELD",
    "PATH_FIELD",
    "format_initial_arm_joint_pose_table",
    "initial_arm_joint_pose_names",
    "initial_arm_joint_pose_path",
    "resolve_initial_arm_joint_pose",
]


#: A profile's ``robot:`` and the robot / rig a pose signature names.
PROFILE_ROBOT_RIG: dict[str, tuple[str, str]] = {
    "fr3_duo": ("fr3", "duo"),
    "fr3": ("fr3", "flat"),
}


def read_pose_signature(path: str | os.PathLike) -> dict | None:
    """Return a pose file's ``signature`` block, or ``None`` if it has none.

    Files written before the block existed carry none and stay valid.
    """

    import json

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    signature = payload.get("signature") if isinstance(payload, dict) else None
    return signature if isinstance(signature, dict) else None


def check_pose_signature(path: str | None, robot: str) -> None:
    """Refuse a pose signed for another rig than the profile's ``robot:``."""

    if path is None:
        return
    signature = read_pose_signature(path)
    if signature is None:
        return
    # Spell the robot the way the profile registry does (case and spaces), so
    # a profile accepted as " FR3 " is checked as fr3, not skipped.
    from robosuite_env.robots.profiles import normalize_robot_name

    try:
        robot = normalize_robot_name(robot)
    except ValueError:
        return  # an unsupported robot is refused where the profile is read
    expected = PROFILE_ROBOT_RIG[robot]
    signed = (signature.get("robot"), signature.get("rig"))
    if signed != expected:
        raise ConfigError(
            f"{NAME_FIELD} selects a pose signed for {signed[0]}/{signed[1]} "
            f"(scene {signature.get('scene')!r}), but the profile's robot "
            f"{robot!r} is {expected[0]}/{expected[1]}",
            path=NAME_FIELD,
            allowed=["/".join(expected)],
            got="/".join(str(part) for part in signed),
        )
