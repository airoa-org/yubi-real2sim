# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Validated JSON input for a dual-FR3 rollout ready pose."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
JOINT_ORDER = tuple(f"j{index}" for index in range(1, 8))
_PAYLOAD_KEYS = frozenset(
    {"schema_version", "units", "joint_order", "left_arm", "right_arm"}
)
# schema_version 2 adds exactly one optional block. A v1 file stays valid
# unchanged and keeps producing ``grip_by_arm=None``, which is what every
# profile that predates the block relies on.
_OPTIONAL_PAYLOAD_KEYS_BY_SCHEMA = {
    1: frozenset({"signature"}),
    2: frozenset({"gripper", "signature"}),
}
# ``signature`` names the robot and scene a pose belongs to
# (``{"robot", "scene", "scenes"}``). The launcher compares its robot with the
# profile's; the joint values are read exactly as before, and a file without
# the block stays valid.

# The real robot publishes these two motor angles on its gripper
# joint-trajectory topic.  They are the same two names
# LeRobot stores as ``observation.joint_states[:2]``.
GRIPPER_JOINT_ORDER = ("hand_left_motor_joint", "hand_right_motor_joint")
# robosuite_env/assets/gripper_calib_rad.json defines
# ``calibrated_left_angle_rad = raw_left_angle_rad + left_offset_rad`` and is
# applied by rotating the compiled finger *body frames*
# (``policy_coordinate: "raw"``), leaving ``qpos0``, the joint ranges and the
# actuator ctrlrange untouched.  Finger ``qpos`` therefore stays in the raw
# motor coordinate the real robot publishes, so a recorded motor angle is used
# verbatim.  Only ``raw`` is accepted; a file that claims another coordinate is
# rejected instead of being silently converted.
GRIPPER_COORDINATE = "raw"
_GRIPPER_KEYS = frozenset({"coordinate", "joint_order", "left", "right"})
# robosuite_env/rollout/build/gripper.py clamps every finger command to this
# closed interval; a declared initial angle outside it would be silently
# clipped, so it is rejected here instead.
GRIPPER_MIN_RAD = 0.0
GRIPPER_MAX_RAD = 0.7854


@dataclass(frozen=True)
class InitialArmJointPose:
    """One provenance-bound pair of left / right FR3 joint vectors."""

    source_path: str
    source_sha256: str
    q_by_arm: Mapping[str, np.ndarray]
    schema_version: int = SCHEMA_VERSION
    # ``None`` means "this file declares no initial gripper angle"; the rollout
    # then keeps its historical behaviour of settling from the model state.
    grip_by_arm: Mapping[str, float] | None = None


def _joint_vector(value: Any, name: str) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != 7
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise ValueError(f"{name} must contain exactly seven JSON numbers")
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _gripper_angle(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number")
    angle = float(value)
    if not np.isfinite(angle):
        raise ValueError(f"{name} must be finite")
    if angle < GRIPPER_MIN_RAD or angle > GRIPPER_MAX_RAD:
        raise ValueError(
            f"{name} must be within [{GRIPPER_MIN_RAD:g}, {GRIPPER_MAX_RAD:g}] rad, "
            f"got {angle:g}"
        )
    return angle


def _gripper_by_arm(value: Any) -> Mapping[str, float]:
    """Validate the optional schema-v2 gripper block in raw motor radians."""

    if not isinstance(value, dict):
        raise ValueError("gripper must be an object")
    unknown = sorted(set(value) - _GRIPPER_KEYS)
    missing = sorted(_GRIPPER_KEYS - set(value))
    if unknown or missing:
        detail = []
        if unknown:
            detail.append(f"gripper unknown keys={unknown}")
        if missing:
            detail.append(f"gripper missing keys={missing}")
        raise ValueError("; ".join(detail))
    if value["coordinate"] != GRIPPER_COORDINATE:
        raise ValueError(
            f"gripper.coordinate must be {GRIPPER_COORDINATE!r}; the checked-in "
            "YUBI calibration is baked into the finger body frames, so finger "
            "qpos stays in the raw motor coordinate"
        )
    if value["joint_order"] != list(GRIPPER_JOINT_ORDER):
        raise ValueError(f"gripper.joint_order must be {list(GRIPPER_JOINT_ORDER)}")
    return {
        "L": _gripper_angle(value["left"], "gripper.left"),
        "R": _gripper_angle(value["right"], "gripper.right"),
    }


def parse_initial_arm_joint_pose_payload(
    payload: Any,
    *,
    source_path: str,
    source_sha256: str,
) -> InitialArmJointPose:
    """Validate the strict JSON payload and map human sides to simulator keys."""

    if not isinstance(payload, dict):
        raise ValueError("initial arm joint pose JSON must be an object")
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in SUPPORTED_SCHEMA_VERSIONS
    ):
        raise ValueError(
            f"schema_version must be one of {list(SUPPORTED_SCHEMA_VERSIONS)}, "
            f"got {schema_version!r}"
        )
    optional_keys = _OPTIONAL_PAYLOAD_KEYS_BY_SCHEMA[schema_version]
    unknown = sorted(set(payload) - _PAYLOAD_KEYS - optional_keys)
    missing = sorted(_PAYLOAD_KEYS - set(payload))
    if unknown or missing:
        detail = []
        if unknown:
            detail.append(f"unknown keys={unknown}")
        if missing:
            detail.append(f"missing keys={missing}")
        raise ValueError("; ".join(detail))
    if payload["units"] != "rad":
        raise ValueError("units must be 'rad'")
    if payload["joint_order"] != list(JOINT_ORDER):
        raise ValueError(f"joint_order must be {list(JOINT_ORDER)}")
    gripper = payload.get("gripper")
    grip_by_arm = None if gripper is None else _gripper_by_arm(gripper)

    # sim.py owns robot0=R and robot1=L. Keep that mapping explicit here so a
    # human-facing left-then-right table cannot silently swap the two arms.
    return InitialArmJointPose(
        source_path=str(source_path),
        source_sha256=str(source_sha256),
        q_by_arm={
            "L": _joint_vector(payload["left_arm"], "left_arm"),
            "R": _joint_vector(payload["right_arm"], "right_arm"),
        },
        schema_version=int(schema_version),
        grip_by_arm=grip_by_arm,
    )


def resolve_initial_grip_targets(
    pose: InitialArmJointPose | None,
    *,
    mimicgen_init_episode: Any = None,
) -> dict[str, float] | None:
    """Return the declared raw initial gripper angles keyed by ``R``/``L``.

    ``None`` means the rollout keeps its historical behaviour of settling the
    gripper from the model state. MimicGen replays its own initial gripper
    angle, so a file that also declares one is refused rather than silently
    overwritten by whichever writer runs last.
    """

    if pose is None or pose.grip_by_arm is None:
        return None
    if mimicgen_init_episode:
        raise ValueError(
            "the initial arm joint pose declares a gripper angle and "
            "--mimicgen-init-episode replays its own; both claim the initial "
            "gripper state. Use a schema-1 initial arm joint file with MimicGen"
        )
    return {key: float(pose.grip_by_arm[key]) for key in ("R", "L")}


def load_initial_arm_joint_pose(path: str | Path) -> InitialArmJointPose:
    """Read and validate one initial-arm-joint JSON file."""

    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    return parse_initial_arm_joint_pose_payload(
        payload,
        source_path=str(source),
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def validate_joint_ranges(
    q_by_arm: Mapping[str, np.ndarray],
    ranges_by_arm: Mapping[str, np.ndarray],
) -> None:
    """Reject targets outside the compiled FR3 hard joint limits."""

    for key in ("R", "L"):
        q = np.asarray(q_by_arm[key], dtype=np.float64)
        ranges = np.asarray(ranges_by_arm[key], dtype=np.float64)
        if q.shape != (7,) or ranges.shape != (7, 2):
            raise ValueError(f"{key} joint target/range has an invalid shape")
        outside = np.flatnonzero((q < ranges[:, 0]) | (q > ranges[:, 1]))
        if outside.size:
            detail = ", ".join(
                f"j{index + 1}={q[index]:.6f} not in "
                f"[{ranges[index, 0]:.6f}, {ranges[index, 1]:.6f}]"
                for index in outside
            )
            raise ValueError(f"{key} initial joint target is outside hard limits: {detail}")


def joint_ready_gate_report(
    actual_by_arm: Mapping[str, np.ndarray],
    target_by_arm: Mapping[str, np.ndarray],
    *,
    tolerance_deg: float,
) -> dict[str, Any]:
    """Return a deterministic requested-vs-actual joint ready-pose report."""

    tolerance = float(tolerance_deg)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("joint ready tolerance must be finite and positive")
    report: dict[str, Any] = {
        "mode": "specified_joint_json",
        "tolerance_deg": tolerance,
        "passed": True,
        "arms": {},
    }
    for key in ("R", "L"):
        actual = np.asarray(actual_by_arm[key], dtype=np.float64)
        target = np.asarray(target_by_arm[key], dtype=np.float64)
        if actual.shape != (7,) or target.shape != (7,):
            raise ValueError(f"{key} actual/target joint vector must have shape (7,)")
        error_deg = np.degrees(np.abs(actual - target))
        max_index = int(np.argmax(error_deg))
        passed = bool(float(error_deg[max_index]) <= tolerance)
        report["arms"][key] = {
            "target_q_rad": [float(value) for value in target],
            "actual_q_rad": [float(value) for value in actual],
            "absolute_error_deg": [float(value) for value in error_deg],
            "max_error_deg": float(error_deg[max_index]),
            "max_error_joint": max_index + 1,
            "passed": passed,
        }
        report["passed"] = bool(report["passed"] and passed)
    return report
