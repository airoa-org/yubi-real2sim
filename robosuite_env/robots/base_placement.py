#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""The contract for where a rig's mounted roots stand on the table.

One file per rig layout, naming a pose for every base slot the rig declares.
The poses are table-top relative and quoted the way URDF does, ``Rz(y) @ Ry(p)
@ Rx(r)``; converting them into the order robosuite composes is the caller's
job and happens in exactly one place (``robosuite_env.tasks.scene``).

Validation is fail-closed on purpose. A calibration that is merely readable
still mis-places a robot in a scene that looks plausible from most camera
angles, and the run completes and reports a number. Every check here exists to
turn one of those silent outcomes into a refusal to start.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .profiles import get_robot_profile, normalize_robot_name


SCHEMA_VERSION = 2

#: Poses are expressed against the table-top origin the task builds its scene
#: around, not against the MuJoCo world origin.
REFERENCE_FRAME = "table"


#: The layout each rig stands in unless a run names another file. One rig, one
#: file, so one rig's calibration is never handed to another.
DEFAULT_BASE_PLACEMENTS = {
    "fr3": "fr3_base_from_blend.json",
    "fr3_duo": "fr3_base_duo.json",
}


class BasePlacementError(ValueError):
    """A calibration file cannot be trusted to place this rig."""


@dataclass(frozen=True)
class BasePose:
    """One mounted root, table-top relative, in URDF rpy order."""

    pos_m: np.ndarray
    rpy_rad: np.ndarray


@dataclass(frozen=True)
class RobotBasePlacement:
    """A validated layout for every base slot one rig declares."""

    robot: str
    poses: Mapping[str, BasePose]
    source_path: str
    source_sha256: str
    source: Mapping[str, Any]

    def pose(self, slot_name: str) -> BasePose:
        try:
            return self.poses[slot_name]
        except KeyError as exc:  # pragma: no cover - guarded at load time
            raise BasePlacementError(
                f"{self.source_path} has no pose for base slot {slot_name!r}"
            ) from exc


def _rpy_rad(entry: Mapping[str, Any], path: str, slot: str) -> np.ndarray:
    """Read one slot's orientation as an ``(r, p, y)`` radian triple.

    ``yaw_deg`` is the flat layout every pedestal scene uses; ``rpy_deg``
    carries the full triple a tilted shared mount needs. Requiring exactly one
    of them keeps a mount pose from being half-applied: a file that quotes a
    roll and a yaw under two different keys would otherwise silently drop one.
    """

    has_yaw = "yaw_deg" in entry
    has_rpy = "rpy_deg" in entry
    if has_yaw == has_rpy:
        raise BasePlacementError(
            f"{path} slot {slot!r} must set exactly one of yaw_deg or rpy_deg, "
            f"got {sorted(entry)}"
        )
    if has_yaw:
        yaw = entry["yaw_deg"]
        if not np.isfinite(float(yaw)):
            raise BasePlacementError(f"{path} slot {slot!r} yaw_deg is not finite")
        return np.array([0.0, 0.0, np.radians(float(yaw))])
    rpy_deg = np.asarray(entry["rpy_deg"], dtype=float)
    if rpy_deg.shape != (3,):
        raise BasePlacementError(
            f"{path} slot {slot!r} rpy_deg must hold three angles, "
            f"got shape {rpy_deg.shape}"
        )
    if not np.all(np.isfinite(rpy_deg)):
        raise BasePlacementError(f"{path} slot {slot!r} rpy_deg is not finite")
    return np.radians(rpy_deg)


def _pos_m(entry: Mapping[str, Any], path: str, slot: str) -> np.ndarray:
    if "pos" not in entry:
        raise BasePlacementError(f"{path} slot {slot!r} is missing pos")
    pos = np.asarray(entry["pos"], dtype=float)
    if pos.shape != (3,):
        raise BasePlacementError(
            f"{path} slot {slot!r} pos must hold three numbers, got shape {pos.shape}"
        )
    if not np.all(np.isfinite(pos)):
        raise BasePlacementError(f"{path} slot {slot!r} pos is not finite")
    return pos


def load_robot_base_placement(path, *, robot: str | None = None) -> RobotBasePlacement:
    """Read and validate one rig's base calibration.

    ``robot`` is the embodiment the caller is about to build. Passing it turns
    "this file belongs to another rig" into an error here rather than into two
    arms standing on a tilted mount that the scene does not have.
    """

    resolved = os.fspath(path)
    try:
        raw = open(resolved, "rb").read()
    except OSError as exc:
        raise BasePlacementError(f"base calibration is unreadable: {resolved}: {exc}") from exc
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BasePlacementError(f"base calibration is not valid JSON: {resolved}: {exc}") from exc
    if not isinstance(document, dict):
        raise BasePlacementError(f"base calibration must be an object: {resolved}")

    version = document.get("schema_version")
    if version != SCHEMA_VERSION:
        raise BasePlacementError(
            f"{resolved} must declare schema_version {SCHEMA_VERSION}, got {version!r}"
        )
    frame = document.get("frame", REFERENCE_FRAME)
    if frame != REFERENCE_FRAME:
        raise BasePlacementError(
            f"{resolved} frame must be {REFERENCE_FRAME!r}, got {frame!r}"
        )

    declared = document.get("robot")
    try:
        declared_robot = normalize_robot_name(declared)
    except (TypeError, ValueError) as exc:
        raise BasePlacementError(
            f"{resolved} declares an unknown robot: {declared!r}"
        ) from exc
    if robot is not None:
        wanted = normalize_robot_name(robot)
        if declared_robot != wanted:
            raise BasePlacementError(
                f"{resolved} is calibrated for {declared_robot!r}, not {wanted!r}"
            )

    base = document.get("base")
    if not isinstance(base, dict):
        raise BasePlacementError(f"{resolved} is missing a base object")

    profile = get_robot_profile(declared_robot)
    expected = {slot.name for slot in profile.base_slots}
    if set(base) != expected:
        raise BasePlacementError(
            f"{resolved} base must name exactly {sorted(expected)} for "
            f"{declared_robot!r}, got {sorted(base)}"
        )

    poses = {}
    for slot_name, entry in base.items():
        if not isinstance(entry, dict):
            raise BasePlacementError(f"{resolved} slot {slot_name!r} must be an object")
        poses[slot_name] = BasePose(
            pos_m=_pos_m(entry, resolved, slot_name),
            rpy_rad=_rpy_rad(entry, resolved, slot_name),
        )

    return RobotBasePlacement(
        robot=declared_robot,
        poses=poses,
        source_path=resolved,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source=document.get("source", {}),
    )


def default_base_placement_path(robot: str):
    """Return the calibration a rig uses when a run names none."""

    from ..paths import OBJECT_CALIBRATION_ROOT

    name = normalize_robot_name(robot)
    try:
        return OBJECT_CALIBRATION_ROOT / DEFAULT_BASE_PLACEMENTS[name]
    except KeyError as exc:  # pragma: no cover - registry and profiles agree
        raise BasePlacementError(f"no default base placement for {name!r}") from exc


def resolve_robot_base_placement(robot: str, explicit=None) -> RobotBasePlacement:
    """Load the calibration a run will place ``robot`` with."""

    path = explicit or default_base_placement_path(robot)
    return load_robot_base_placement(path, robot=robot)
