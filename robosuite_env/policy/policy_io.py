# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Pure policy I/O helpers for ``sim.py``.

This module intentionally has no MuJoCo, ROS, or renderer dependencies.  It
captures the policy-facing conventions that are easy to regress inside the
large simulator entry point:

* Yubi action layout is ``left(7), right(7), gripL, gripR``.
* Pose deltas are XYZW quaternions and are integrated in the current canonical
  hand-root local frame.
* ``sim.py`` stores live target quaternions in the FR3/YUBI gripper frame, so
  observation and action application convert through the fixed
  canonical-hand-root <-> gripper-frame rotation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .rel2abs import apply_delta_local


ACTION_DIM = 16
LEFT_ACTION_SLICE = slice(0, 7)
RIGHT_ACTION_SLICE = slice(7, 14)
GRIP_L_INDEX = 14
GRIP_R_INDEX = 15
POLICY_IMAGE_SIZE = 224
# Match the real FR3 controller's per-message ``cmd_rel_pose`` safety limits.  The policy
# pose actions in this repository use the same local-frame SE(3) delta
# convention, so these are step limits rather than Cartesian speed limits.
FR3_POLICY_MAX_REL_POSE_TRANSLATION_M = 0.05
FR3_POLICY_MAX_REL_POSE_ROTATION_RAD = 0.2618
FR3_POLICY_ACTION_CLAMP_TELEMETRY_SCHEMA_VERSION = 1
POLICY_POSE_REPRESENTATIONS = ("delta", "anchored")
# Exact FR3/YUBI transform from the MuJoCo ``eef`` body to the real
# ``hand_root`` frame used by the policy.  The real URDF places hand_root at
# [0, 1.5, 51.0] mm under yubi_base while the simulated eef is at
# [0, 0, 103.4] mm.
FR3_EEF_TO_HAND_ROOT_M = (0.0, 0.0015, -0.0524)
FR3_GRIPPER_ROOT_TO_HAND_ROOT_M = (0.0, 0.0015, 0.0510)
_UNIT_QUAT_ATOL = 1e-4
_MIN_ACTION_QUAT_NORM = 1e-8


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = np.asarray(a, dtype=float)
    w1, x1, y1, z1 = np.asarray(b, dtype=float)
    return np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ],
        dtype=float,
    )


def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quat_wxyz_from_matrix(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return normalize_quat_wxyz(np.array([w, x, y, z], dtype=float))


def _validated_pose_xyzw(value: np.ndarray, *, name: str) -> np.ndarray:
    """Return one finite SE(3) pose after strict shape/unit-quaternion checks."""

    pose = np.asarray(value, dtype=float)
    if pose.shape != (7,):
        raise ValueError(f"{name} must have shape (7,), got {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must be finite")
    quat_norm = float(np.linalg.norm(pose[3:7]))
    if not np.isclose(quat_norm, 1.0, rtol=0.0, atol=_UNIT_QUAT_ATOL):
        raise ValueError(
            f"{name} quaternion must be unit length within {_UNIT_QUAT_ATOL:g}, "
            f"got norm {quat_norm:.9g}"
        )
    result = pose.copy()
    result[3:7] /= quat_norm
    return result


def _unit_quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert an already validated XYZW unit quaternion without epsilon bias."""

    x, y, z, w = np.asarray(quat_xyzw, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def compose_pose_xyzw(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compose two ``[x,y,z,qx,qy,qz,qw]`` SE(3) transforms."""

    lhs = _validated_pose_xyzw(first, name="first pose")
    rhs = _validated_pose_xyzw(second, name="second pose")
    position = lhs[:3] + _unit_quat_xyzw_to_matrix(lhs[3:7]) @ rhs[:3]
    quat = quat_mul_wxyz(xyzw_to_wxyz(lhs[3:7]), xyzw_to_wxyz(rhs[3:7]))
    quat /= np.linalg.norm(quat)
    quat_xyzw = wxyz_to_xyzw(quat)
    if quat_xyzw[3] < 0.0:
        quat_xyzw = -quat_xyzw
    return np.concatenate([position, quat_xyzw])


def invert_pose_xyzw(pose: np.ndarray) -> np.ndarray:
    """Invert one ``[x,y,z,qx,qy,qz,qw]`` SE(3) transform."""

    transform = _validated_pose_xyzw(pose, name="pose")
    quat_wxyz = quat_conj_wxyz(xyzw_to_wxyz(transform[3:7]))
    quat_xyzw = wxyz_to_xyzw(quat_wxyz)
    if quat_xyzw[3] < 0.0:
        quat_xyzw = -quat_xyzw
    position = -_unit_quat_xyzw_to_matrix(quat_xyzw) @ transform[:3]
    return np.concatenate([position, quat_xyzw])


def axis_rot_matrix(axis: int, deg: float) -> np.ndarray:
    h = np.radians(float(deg)) / 2.0
    q = np.array([np.cos(h), 0.0, 0.0, 0.0], dtype=float)
    q[1 + int(axis)] = np.sin(h)
    return quat_wxyz_to_matrix(q)


R_CANON_TO_FRANKA = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    dtype=float,
) @ axis_rot_matrix(2, 90.0)
Q_CANON_TO_FRANKA_WXYZ = quat_wxyz_from_matrix(R_CANON_TO_FRANKA)


def validate_action_chunk(chunk: np.ndarray) -> np.ndarray:
    arr = np.asarray(chunk)
    if arr.ndim != 2:
        raise ValueError(f"policy action chunk must be rank-2 [horizon, action_dim], got shape {arr.shape}")
    if arr.shape[1] < ACTION_DIM:
        raise ValueError(f"policy action chunk dim must be >= {ACTION_DIM}, got shape {arr.shape}")
    try:
        finite = bool(np.all(np.isfinite(arr)))
    except TypeError as exc:
        raise ValueError("policy action chunk must be numeric and finite") from exc
    if not finite:
        raise ValueError("policy action chunk must be finite")
    return arr


def validate_policy_pose_execution_contract(
    pose_repr: str,
    *,
    robot: str,
    policy_delta_anchor: str,
    rf_overlap: int,
    action_clamp: bool,
) -> str:
    """Validate simulator features whose math assumes per-step pose deltas.

    ``anchored`` means every pose row is relative to the one observation-time
    hand pose, rather than to the preceding row.  The RF action-conditioning
    path, live-TCP rebasing, and per-row FR3 delta clamp all consume per-step
    deltas, so combining any of them with anchored rows would silently
    execute a different trajectory.
    """

    representation = str(pose_repr)
    if representation not in POLICY_POSE_REPRESENTATIONS:
        raise ValueError(
            "policy pose representation must be one of "
            f"{POLICY_POSE_REPRESENTATIONS}, got {representation!r}"
        )
    if representation == "delta":
        return representation
    if str(robot) != "fr3":
        raise ValueError("anchored policy pose execution is currently supported only for FR3")
    if str(policy_delta_anchor) != "observation":
        raise ValueError(
            "anchored policy poses require policy_delta_anchor='observation'; "
            "live-TCP rebasing changes the fixed inference-time anchor"
        )
    if int(rf_overlap) != 0:
        raise ValueError(
            "anchored policy poses require rf_overlap=0 because RF conditioning and "
            "left-reference rebasing currently operate on per-step deltas"
        )
    if bool(action_clamp):
        raise ValueError(
            "anchored policy poses require --no-policy-action-clamp; the current FR3 "
            "clamp limits each row as a per-step delta, not as a fixed-anchor waypoint"
        )
    return representation


def validate_unclamped_fr3_policy_action_chunk(chunk: np.ndarray) -> np.ndarray:
    """Validate raw FR3 pose deltas without changing their magnitude or axis."""

    sequence = validate_action_chunk(chunk)
    for action_slice in (LEFT_ACTION_SLICE, RIGHT_ACTION_SLICE):
        quaternion_norm = np.linalg.norm(sequence[:, action_slice][:, 3:7], axis=1)
        if np.any(quaternion_norm <= _MIN_ACTION_QUAT_NORM):
            raise ValueError("unclamped FR3 policy delta quaternion must be non-zero")
    return sequence


def clamp_fr3_policy_action_chunk(
    chunk: np.ndarray,
    *,
    max_translation_m: float = FR3_POLICY_MAX_REL_POSE_TRANSLATION_M,
    max_rotation_rad: float = FR3_POLICY_MAX_REL_POSE_ROTATION_RAD,
) -> np.ndarray:
    """Clamp each arm's local SE(3) policy delta to Franka deploy limits.

    Translation is limited by its three-vector norm, not per axis. Rotation is
    normalized, converted to the shortest axis-angle representation, and
    limited while preserving its axis. Gripper and padded action dimensions are
    copied unchanged. The input array is never mutated.
    """

    sequence = validate_action_chunk(chunk)
    translation_limit = float(max_translation_m)
    rotation_limit = float(max_rotation_rad)
    if not np.isfinite(translation_limit) or translation_limit < 0.0:
        raise ValueError("FR3 policy translation clamp must be finite and non-negative")
    if not np.isfinite(rotation_limit) or not 0.0 <= rotation_limit <= np.pi:
        raise ValueError("FR3 policy rotation clamp must be finite and in [0, pi]")
    if not np.all(np.isfinite(sequence)):
        raise ValueError("FR3 policy action chunk must be finite before clamping")

    output_dtype = sequence.dtype if np.issubdtype(sequence.dtype, np.floating) else float
    result = np.array(sequence, dtype=output_dtype, copy=True)
    for action_slice in (LEFT_ACTION_SLICE, RIGHT_ACTION_SLICE):
        pose = result[:, action_slice]

        translation = np.asarray(pose[:, :3], dtype=float)
        translation_norm = np.linalg.norm(translation, axis=1)
        translation_scale = np.ones_like(translation_norm)
        limited_translation = translation_norm > translation_limit
        np.divide(
            translation_limit,
            translation_norm,
            out=translation_scale,
            where=limited_translation,
        )
        translation_scale[~limited_translation] = 1.0
        pose[:, :3] = translation * translation_scale[:, None]

        quaternion = np.asarray(pose[:, 3:7], dtype=float)
        quaternion_norm = np.linalg.norm(quaternion, axis=1)
        usable = quaternion_norm > _MIN_ACTION_QUAT_NORM
        normalized = np.zeros_like(quaternion)
        normalized[:, 3] = 1.0
        normalized[usable] = quaternion[usable] / quaternion_norm[usable, None]

        # q and -q encode the same rotation. Canonicalizing w >= 0 selects the
        # shortest rotation, matching Eigen::AngleAxis in the real FR3 controller.
        normalized[normalized[:, 3] < 0.0] *= -1.0
        vector_norm = np.linalg.norm(normalized[:, :3], axis=1)
        angle = 2.0 * np.arctan2(
            vector_norm,
            np.clip(normalized[:, 3], 0.0, 1.0),
        )
        limited_rotation = angle > rotation_limit
        if np.any(limited_rotation):
            axes = normalized[limited_rotation, :3] / vector_norm[
                limited_rotation, None
            ]
            half_angle = 0.5 * rotation_limit
            normalized[limited_rotation, :3] = axes * np.sin(half_angle)
            normalized[limited_rotation, 3] = np.cos(half_angle)
        pose[:, 3:7] = normalized

    return result


def _max_float_or_none(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    return float(np.max(values)) if values.size else None


def _fr3_policy_action_clamp_arm_telemetry(
    raw_pose: np.ndarray,
    clamped_pose: np.ndarray,
    *,
    translation_limit: float,
    rotation_limit: float,
) -> tuple[dict, np.ndarray]:
    """Describe clamp interventions for one arm without changing either input."""

    raw_pose = np.asarray(raw_pose, dtype=float)
    clamped_pose = np.asarray(clamped_pose, dtype=float)
    if raw_pose.shape != clamped_pose.shape or raw_pose.ndim != 2 or raw_pose.shape[1] != 7:
        raise ValueError(
            "FR3 clamp telemetry poses must have matching [horizon, 7] shapes, "
            f"got raw={raw_pose.shape}, clamped={clamped_pose.shape}"
        )

    translation_input_norm = np.linalg.norm(raw_pose[:, :3], axis=1)
    translation_output_norm = np.linalg.norm(clamped_pose[:, :3], axis=1)
    translation_clipped = translation_input_norm > translation_limit

    raw_quaternion = raw_pose[:, 3:7]
    raw_quaternion_norm = np.linalg.norm(raw_quaternion, axis=1)
    usable = raw_quaternion_norm > _MIN_ACTION_QUAT_NORM
    normalized = np.zeros_like(raw_quaternion)
    normalized[:, 3] = 1.0
    normalized[usable] = raw_quaternion[usable] / raw_quaternion_norm[usable, None]
    normalized[normalized[:, 3] < 0.0] *= -1.0
    raw_vector_norm = np.linalg.norm(normalized[:, :3], axis=1)
    rotation_input_angle = 2.0 * np.arctan2(
        raw_vector_norm,
        np.clip(normalized[:, 3], 0.0, 1.0),
    )
    rotation_clipped = rotation_input_angle > rotation_limit

    output_quaternion = clamped_pose[:, 3:7]
    output_quaternion_norm = np.linalg.norm(output_quaternion, axis=1)
    output_normalized = np.zeros_like(output_quaternion)
    output_normalized[:, 3] = 1.0
    output_usable = output_quaternion_norm > _MIN_ACTION_QUAT_NORM
    output_normalized[output_usable] = (
        output_quaternion[output_usable]
        / output_quaternion_norm[output_usable, None]
    )
    output_normalized[output_normalized[:, 3] < 0.0] *= -1.0
    output_vector_norm = np.linalg.norm(output_normalized[:, :3], axis=1)
    rotation_output_angle = 2.0 * np.arctan2(
        output_vector_norm,
        np.clip(output_normalized[:, 3], 0.0, 1.0),
    )

    row_clipped = translation_clipped | rotation_clipped
    translation_reduction = np.maximum(
        translation_input_norm - translation_output_norm,
        0.0,
    )
    rotation_reduction = np.maximum(
        rotation_input_angle - rotation_output_angle,
        0.0,
    )
    telemetry = {
        "rows_received": int(raw_pose.shape[0]),
        "rows_evaluated": int(raw_pose.shape[0]),
        "intervention_occurred": bool(np.any(row_clipped)),
        "rows_clipped": int(np.count_nonzero(row_clipped)),
        "translation_rows_clipped": int(np.count_nonzero(translation_clipped)),
        "rotation_rows_clipped": int(np.count_nonzero(rotation_clipped)),
        "zero_quaternion_rows_sanitized": int(np.count_nonzero(~usable)),
        "max_translation_input_norm_m": _max_float_or_none(translation_input_norm),
        "max_translation_output_norm_m": _max_float_or_none(translation_output_norm),
        "max_translation_clipped_amount_m": _max_float_or_none(
            translation_reduction[translation_clipped]
        ),
        "max_rotation_input_rad": _max_float_or_none(rotation_input_angle),
        "max_rotation_output_rad": _max_float_or_none(rotation_output_angle),
        "max_rotation_clipped_amount_rad": _max_float_or_none(
            rotation_reduction[rotation_clipped]
        ),
    }
    return telemetry, row_clipped


def clamp_fr3_policy_action_chunk_with_telemetry(
    chunk: np.ndarray,
    *,
    max_translation_m: float = FR3_POLICY_MAX_REL_POSE_TRANSLATION_M,
    max_rotation_rad: float = FR3_POLICY_MAX_REL_POSE_ROTATION_RAD,
) -> tuple[np.ndarray, dict]:
    """Clamp an FR3 policy chunk and report which limits actually intervened.

    ``rows_clipped`` counts policy time rows where either arm hit either limit.
    ``arm_rows_clipped`` counts arm/time pairs, while the translation and
    rotation counters count their own arm/time interventions.  These distinct
    counters prevent a row that hits multiple limits from being reported as
    several policy rows.  The returned action array is produced by
    :func:`clamp_fr3_policy_action_chunk`, keeping execution semantics on the
    established clamp path.
    """

    clamped = clamp_fr3_policy_action_chunk(
        chunk,
        max_translation_m=max_translation_m,
        max_rotation_rad=max_rotation_rad,
    )
    raw = validate_action_chunk(chunk)
    translation_limit = float(max_translation_m)
    rotation_limit = float(max_rotation_rad)
    arms = {}
    row_masks = []
    for arm, action_slice in (
        ("left", LEFT_ACTION_SLICE),
        ("right", RIGHT_ACTION_SLICE),
    ):
        arm_telemetry, row_mask = _fr3_policy_action_clamp_arm_telemetry(
            raw[:, action_slice],
            clamped[:, action_slice],
            translation_limit=translation_limit,
            rotation_limit=rotation_limit,
        )
        arms[arm] = arm_telemetry
        row_masks.append(row_mask)
    rows_clipped = (
        np.logical_or.reduce(row_masks)
        if row_masks
        else np.zeros(raw.shape[0], dtype=bool)
    )

    return clamped, {
        "schema_version": FR3_POLICY_ACTION_CLAMP_TELEMETRY_SCHEMA_VERSION,
        "enabled": True,
        "measurement": "applied",
        "scope": "inferred_chunk_before_execution_slicing",
        "limits": {
            "translation_m": translation_limit,
            "rotation_rad": rotation_limit,
        },
        "rows_received": int(raw.shape[0]),
        "rows_evaluated": int(raw.shape[0]),
        "intervention_occurred": bool(np.any(rows_clipped)),
        "rows_clipped": int(np.count_nonzero(rows_clipped)),
        "arm_rows_evaluated": int(2 * raw.shape[0]),
        "arm_rows_clipped": int(sum(arm["rows_clipped"] for arm in arms.values())),
        "translation_arm_rows_clipped": int(
            sum(arm["translation_rows_clipped"] for arm in arms.values())
        ),
        "rotation_arm_rows_clipped": int(
            sum(arm["rotation_rows_clipped"] for arm in arms.values())
        ),
        "zero_quaternion_arm_rows_sanitized": int(
            sum(arm["zero_quaternion_rows_sanitized"] for arm in arms.values())
        ),
        "arms": arms,
    }


def fr3_policy_action_clamp_not_applied_telemetry(chunk: np.ndarray) -> dict:
    """Describe an FR3 policy chunk for which the action clamp was disabled."""

    rows = int(validate_action_chunk(chunk).shape[0])
    arms = {
        arm: {
            "rows_received": rows,
            "rows_evaluated": 0,
            "intervention_occurred": False,
            "rows_clipped": 0,
            "translation_rows_clipped": 0,
            "rotation_rows_clipped": 0,
            "zero_quaternion_rows_sanitized": 0,
            "max_translation_input_norm_m": None,
            "max_translation_output_norm_m": None,
            "max_translation_clipped_amount_m": None,
            "max_rotation_input_rad": None,
            "max_rotation_output_rad": None,
            "max_rotation_clipped_amount_rad": None,
        }
        for arm in ("left", "right")
    }
    return {
        "schema_version": FR3_POLICY_ACTION_CLAMP_TELEMETRY_SCHEMA_VERSION,
        "enabled": False,
        "measurement": "not_applied",
        "scope": "inferred_chunk_before_execution_slicing",
        "limits": {
            "translation_m": FR3_POLICY_MAX_REL_POSE_TRANSLATION_M,
            "rotation_rad": FR3_POLICY_MAX_REL_POSE_ROTATION_RAD,
        },
        "rows_received": rows,
        "rows_evaluated": 0,
        "intervention_occurred": False,
        "rows_clipped": 0,
        "arm_rows_evaluated": 0,
        "arm_rows_clipped": 0,
        "translation_arm_rows_clipped": 0,
        "rotation_arm_rows_clipped": 0,
        "zero_quaternion_arm_rows_sanitized": 0,
        "arms": arms,
    }


def aggregate_fr3_policy_action_clamp_telemetry(
    chunk_telemetry: list[dict],
    *,
    enabled: bool,
) -> dict:
    """Aggregate per-chunk FR3 clamp records into one rollout summary."""

    expected_enabled = bool(enabled)
    expected_limits = {
        "translation_m": FR3_POLICY_MAX_REL_POSE_TRANSLATION_M,
        "rotation_rad": FR3_POLICY_MAX_REL_POSE_ROTATION_RAD,
    }
    for index, item in enumerate(chunk_telemetry):
        if not isinstance(item, dict):
            raise ValueError(f"FR3 clamp telemetry chunk {index} must be a mapping")
        if item.get("schema_version") != FR3_POLICY_ACTION_CLAMP_TELEMETRY_SCHEMA_VERSION:
            raise ValueError(f"FR3 clamp telemetry chunk {index} has an unsupported schema")
        if item.get("enabled") is not expected_enabled:
            raise ValueError(
                f"FR3 clamp telemetry chunk {index} enabled state disagrees with rollout"
            )
        if item.get("limits") != expected_limits:
            raise ValueError(
                f"FR3 clamp telemetry chunk {index} limits disagree with rollout limits"
            )

    def _sum(field: str) -> int:
        return int(sum(int(item[field]) for item in chunk_telemetry))

    def _max(records: list[dict], field: str) -> float | None:
        values = [record[field] for record in records if record[field] is not None]
        return max((float(value) for value in values), default=None)

    arms = {}
    for arm in ("left", "right"):
        records = [item["arms"][arm] for item in chunk_telemetry]
        arms[arm] = {
            "rows_received": int(sum(int(record["rows_received"]) for record in records)),
            "rows_evaluated": int(sum(int(record["rows_evaluated"]) for record in records)),
            "intervention_occurred": any(
                bool(record["intervention_occurred"]) for record in records
            ),
            "rows_clipped": int(sum(int(record["rows_clipped"]) for record in records)),
            "translation_rows_clipped": int(
                sum(int(record["translation_rows_clipped"]) for record in records)
            ),
            "rotation_rows_clipped": int(
                sum(int(record["rotation_rows_clipped"]) for record in records)
            ),
            "zero_quaternion_rows_sanitized": int(
                sum(int(record["zero_quaternion_rows_sanitized"]) for record in records)
            ),
            "max_translation_input_norm_m": _max(
                records, "max_translation_input_norm_m"
            ),
            "max_translation_output_norm_m": _max(
                records, "max_translation_output_norm_m"
            ),
            "max_translation_clipped_amount_m": _max(
                records, "max_translation_clipped_amount_m"
            ),
            "max_rotation_input_rad": _max(records, "max_rotation_input_rad"),
            "max_rotation_output_rad": _max(records, "max_rotation_output_rad"),
            "max_rotation_clipped_amount_rad": _max(
                records, "max_rotation_clipped_amount_rad"
            ),
        }

    return {
        "schema_version": FR3_POLICY_ACTION_CLAMP_TELEMETRY_SCHEMA_VERSION,
        "enabled": expected_enabled,
        "measurement": "applied" if expected_enabled else "not_applied",
        "scope": "all_inferred_primary_chunks_before_execution_slicing",
        "limits": expected_limits,
        "chunks_received": int(len(chunk_telemetry)),
        "chunks_evaluated": int(len(chunk_telemetry) if expected_enabled else 0),
        "chunks_with_clipping": int(
            sum(int(item["rows_clipped"]) > 0 for item in chunk_telemetry)
        ),
        "rows_received": _sum("rows_received"),
        "rows_evaluated": _sum("rows_evaluated"),
        "intervention_occurred": any(
            int(item["rows_clipped"]) > 0 for item in chunk_telemetry
        ),
        "rows_clipped": _sum("rows_clipped"),
        "arm_rows_evaluated": _sum("arm_rows_evaluated"),
        "arm_rows_clipped": _sum("arm_rows_clipped"),
        "translation_arm_rows_clipped": _sum("translation_arm_rows_clipped"),
        "rotation_arm_rows_clipped": _sum("rotation_arm_rows_clipped"),
        "zero_quaternion_arm_rows_sanitized": _sum(
            "zero_quaternion_arm_rows_sanitized"
        ),
        "arms": arms,
    }


def _validated_yubi_action_sequence(
    actions: np.ndarray,
    *,
    name: str,
    require_nonempty: bool = False,
) -> np.ndarray:
    sequence = validate_action_chunk(actions)
    if require_nonempty and sequence.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one action")
    if not np.all(np.isfinite(sequence)):
        raise ValueError(f"{name} must be finite")
    # Policy outputs are not guaranteed to land exactly on S^3.  Execution
    # normalizes each delta quaternion in ``rel2abs.apply_delta_local``, so the
    # frame propagation used for rebasing must follow that same contract.  A
    # zero quaternion, however, has no well-defined rotation and is rejected.
    for side, action_slice in (
        ("left", LEFT_ACTION_SLICE),
        ("right", RIGHT_ACTION_SLICE),
    ):
        quat_norms = np.linalg.norm(
            np.asarray(sequence[:, action_slice], dtype=float)[:, 3:7], axis=1
        )
        invalid = quat_norms < _MIN_ACTION_QUAT_NORM
        if np.any(invalid):
            index = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                f"{name} {side} quaternion at row {index} must have non-zero "
                f"length, got norm {quat_norms[index]:.9g}"
            )
    return sequence


def propagate_yubi_inter_hand_state(
    initial_h: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Propagate ``H = T_left^-1 T_right`` through a physical Yubi action prefix.

    Each action row contains body-frame increments ``A = delta_left`` and
    ``B = delta_right``.  Consequently the inter-hand state before the next
    action is ``H_next = A^-1 H B``.  Passing ``actions[:index]`` returns the
    inter-hand state immediately before action ``index``.
    """

    state = _validated_pose_xyzw(initial_h, name="initial inter-hand pose")
    sequence = _validated_yubi_action_sequence(actions, name="Yubi actions")
    for row in np.asarray(sequence, dtype=float):
        left_delta = row[LEFT_ACTION_SLICE].copy()
        right_delta = row[RIGHT_ACTION_SLICE].copy()
        # Match the rollout integrator: physical action quaternions are
        # normalized immediately before local-frame composition.
        left_delta[3:7] /= np.linalg.norm(left_delta[3:7])
        right_delta[3:7] /= np.linalg.norm(right_delta[3:7])
        state = compose_pose_xyzw(
            compose_pose_xyzw(invert_pose_xyzw(left_delta), state),
            right_delta,
        )
    return state


def left_reference_action_correction(
    source_h: np.ndarray,
    destination_h: np.ndarray,
) -> np.ndarray:
    """Return ``K = destination_h^-1 source_h`` for a left-reference bridge."""

    source = _validated_pose_xyzw(source_h, name="source inter-hand pose")
    destination = _validated_pose_xyzw(
        destination_h, name="destination inter-hand pose"
    )
    return compose_pose_xyzw(invert_pose_xyzw(destination), source)


def pose_left_multiply_affine_xyzw(
    correction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical ``M, b`` for left-multiplying a 7D pose by ``correction``.

    For ``r = [p, q_xyzw]`` and ``K = [t, q_K]``, this represents
    ``p' = t + R_K p`` and ``q' = q_K * q`` as ``r' = M @ r + b``.
    The quaternion operand is intentionally not normalized, so the same affine
    map can be conjugated into quantile-normalized RF latent coordinates.
    """

    transform = _validated_pose_xyzw(correction, name="left-multiply correction")
    rotation = _unit_quat_xyzw_to_matrix(transform[3:7])
    w, x, y, z = xyzw_to_wxyz(transform[3:7])
    quat_left_matrix_xyzw = np.array(
        [
            [w, -z, y, x],
            [z, w, -x, y],
            [-y, x, w, z],
            [-x, -y, -z, w],
        ],
        dtype=float,
    )
    matrix = np.zeros((7, 7), dtype=float)
    matrix[:3, :3] = rotation
    matrix[3:7, 3:7] = quat_left_matrix_xyzw
    bias = np.zeros(7, dtype=float)
    bias[:3] = transform[:3]
    return matrix, bias


def rebase_yubi_action_segment_left_reference(
    actions: np.ndarray,
    source_h: np.ndarray,
    destination_h: np.ndarray,
) -> np.ndarray:
    """Rebase a physical action segment while preserving its left-frame path.

    The segment's first left delta is unchanged.  Its first right delta ``B``
    becomes ``K B`` for ``K = destination_h^-1 source_h``; after that first
    row the destination inter-hand state equals the source state, so every
    later pose delta can remain unchanged.  Gripper and padded dimensions are
    always copied bit-for-bit.
    """

    sequence = _validated_yubi_action_sequence(
        actions,
        name="Yubi action segment",
        require_nonempty=True,
    )
    correction = left_reference_action_correction(source_h, destination_h)
    output_dtype = sequence.dtype if np.issubdtype(sequence.dtype, np.floating) else float
    result = np.array(sequence, dtype=output_dtype, copy=True)

    identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    if np.allclose(correction[:3], 0.0, rtol=0.0, atol=1e-12) and np.allclose(
        correction[3:7], identity_quat, rtol=0.0, atol=1e-12
    ):
        return result

    matrix, bias = pose_left_multiply_affine_xyzw(correction)
    result[0, RIGHT_ACTION_SLICE] = (
        matrix @ np.asarray(sequence[0, RIGHT_ACTION_SLICE], dtype=float) + bias
    )
    return result


def rf_conditioning_source_inter_hand(
    active_observation_h: np.ndarray,
    active_chunk: np.ndarray,
    tail_start: int,
    target_observation_h: np.ndarray,
    *,
    policy_delta_anchor: str,
) -> tuple[np.ndarray, str]:
    """Return the source boundary used to rebase an RF static prefix.

    Observation-anchored rollout pre-integrates the whole active chunk, so its
    tail starts at the planned state reached by ``active_chunk[:tail_start]``.
    Legacy live anchoring applies each local delta from the live TCP; the raw
    tail is already the command history beginning at the target/midpoint OBS
    and therefore needs an identity static rebase.
    """

    chunk = validate_action_chunk(active_chunk)
    start = int(tail_start)
    if start < 0 or start > chunk.shape[0]:
        raise ValueError(
            f"RF conditioning tail_start must be in [0, {chunk.shape[0]}], got {start}"
        )
    target = _validated_pose_xyzw(
        target_observation_h, name="target observation inter-hand pose"
    )
    anchor = str(policy_delta_anchor)
    if anchor == "live":
        return target, "target_observation_live"
    if anchor != "observation":
        raise ValueError(
            "policy_delta_anchor must be 'observation' or 'live', "
            f"got {policy_delta_anchor!r}"
        )
    source = propagate_yubi_inter_hand_state(
        _validated_pose_xyzw(
            active_observation_h, name="active observation inter-hand pose"
        ),
        chunk[:start],
    )
    return source, "active_observation_planned_boundary"


def validate_exec_horizon(chunk: np.ndarray, exec_steps: int) -> None:
    horizon = int(np.asarray(chunk).shape[0])
    steps = int(exec_steps)
    if steps <= 0:
        raise ValueError(f"--exec must be positive for policy rollout, got {steps}")
    if horizon < steps:
        raise ValueError(
            f"--exec ({steps}) exceeds policy action horizon ({horizon}); "
            "set --exec <= chunk.shape[0] or load the checkpoint with the correct --ah"
        )


def validate_rf_overlap(exec_steps: int, overlap_steps: int, action_horizon: int | None = None) -> int:
    """Validate lagged RF overlap settings and return the first executed index.

    With overlap enabled, the model conditions indices ``[0:overlap]`` on the
    historical tail and the simulator executes ``[overlap:overlap + exec]``.
    """

    steps = int(exec_steps)
    overlap = int(overlap_steps)
    if steps <= 0:
        raise ValueError(f"--exec must be positive for policy rollout, got {steps}")
    if overlap < 0:
        raise ValueError(f"--rf-overlap must be non-negative, got {overlap}")
    if overlap == 0:
        required_horizon = steps
    else:
        if overlap > steps:
            raise ValueError(
                f"--rf-overlap ({overlap}) must not exceed --exec ({steps})"
            )
        required_horizon = steps + overlap
    if action_horizon is not None and int(action_horizon) < required_horizon:
        raise ValueError(
            f"RF overlap requires action horizon >= exec + overlap "
            f"({steps} + {overlap} = {required_horizon}), got {int(action_horizon)}"
        )
    return overlap


def rf_overlap_scheduler_flags(
    block_step: int,
    *,
    exec_steps: int,
    overlap_steps: int,
) -> tuple[bool, bool]:
    """Return ``(infer_pending, activate_pending)`` for one rollout frame.

    The conditioned chunk is inferred after ``exec - overlap`` actions have
    completed, then held until the current ``exec``-action block completes.
    Legacy mode (``overlap=0``) returns no overlap scheduler events.
    """

    step = int(block_step)
    if step < 0:
        raise ValueError(f"RF overlap block step must be non-negative, got {step}")
    overlap = validate_rf_overlap(exec_steps, overlap_steps)
    if overlap == 0:
        return False, False
    steps = int(exec_steps)
    return step == steps - overlap, step >= steps


def rf_overlap_conditioning_tail(
    chunk: np.ndarray,
    *,
    execution_start: int,
    exec_steps: int,
    overlap_steps: int,
) -> np.ndarray:
    """Return the historical action tail aligned with the next chunk prefix."""

    arr = validate_action_chunk(chunk)
    overlap = int(overlap_steps)
    start = int(execution_start)
    steps = int(exec_steps)
    validate_rf_overlap(steps, overlap, arr.shape[0])
    if overlap == 0:
        return arr[:0].copy()
    end = start + steps
    tail_start = end - overlap
    if start < 0 or end > arr.shape[0]:
        raise ValueError(
            f"executed action slice [{start}:{end}] exceeds chunk horizon {arr.shape[0]}"
        )
    return arr[tail_start:end].copy()


def add_yubi_action_conditioning(obs: dict, action_prefix: np.ndarray) -> dict:
    """Attach a physical 16D Yubi action prefix for the normal input transform.

    ``YubiInputs`` will concatenate these fields, quantile-normalize them, and
    pad them to the model action dimension.  Keeping this conversion inside the
    policy transform prevents raw physical actions from entering the RF ODE.
    """

    prefix = validate_action_chunk(action_prefix)
    conditioned = dict(obs)
    conditioned["observation.pose.left_hand_root.relative"] = np.asarray(
        prefix[:, LEFT_ACTION_SLICE], dtype=np.float32
    )
    conditioned["observation.pose.right_hand_root.relative"] = np.asarray(
        prefix[:, RIGHT_ACTION_SLICE], dtype=np.float32
    )
    conditioned["action.joint_states"] = np.asarray(
        prefix[:, [GRIP_L_INDEX, GRIP_R_INDEX]], dtype=np.float32
    )
    return conditioned


# ---------------------------------------------------------------------------
# Relative-hand state quaternion hemisphere
# ---------------------------------------------------------------------------

#: Observation key of the left-to-right hand-root pose sent as policy state.
POLICY_STATE_REL_POSE_KEY = "observation.pose.left_hand_root_to_right_hand_root.absolute"
#: ``YubiInputs`` (robosuite_env/policy/yubi_openpi/yubi_policy.py) builds
#: ``state = concat(<POLICY_STATE_REL_POSE_KEY> (7), observation.joint_states (2))``
#: padded to the model action_dim, and ``relative_hand_state`` emits the pose
#: as ``[dx, dy, dz, qx, qy, qz, qw]`` (``wxyz_to_xyzw``). The scalar part of
#: the relative quaternion is therefore state index 6 of a 9D YUBI state.
POLICY_STATE_REL_QUAT_SLICE = slice(3, 7)
POLICY_STATE_REL_QW_INDEX = 6
POLICY_STATE_YUBI_DIM = 9
POLICY_STATE_QUAT_HEMISPHERES = ("positive", "negative")
POLICY_STATE_QUAT_HEMISPHERE_CHOICES = ("auto", *POLICY_STATE_QUAT_HEMISPHERES)
POLICY_STATE_QUAT_HEMISPHERE_RULE = (
    "norm_stats.state q99[qw] < 0 -> negative; q01[qw] > 0 -> positive; "
    "otherwise (mixed) -> positive"
)
_POLICY_STATE_QUAT_COMPONENT_BOUND = 1.0 + 1e-6


class PolicyStateQuatHemisphereError(ValueError):
    """The training-state quaternion hemisphere of a checkpoint is undeterminable."""


def classify_policy_state_qw(qw_q01: float, qw_q99: float) -> tuple[str, str]:
    """Return ``(classification, hemisphere)`` for training-state qw quantiles.

    A mixed-sign training state has no sign rule that matches it better than
    another, so it keeps the historical ``positive`` rule.
    """

    if qw_q99 < 0.0:
        return "negative", "negative"
    if qw_q01 > 0.0:
        return "positive", "positive"
    return "mixed", "positive"


def _norm_stats_quantiles(
    state: Any, name: str, path: Path
) -> list[float]:
    values = state.get(name) if isinstance(state, dict) else None
    if not isinstance(values, list) or not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in values
    ):
        raise PolicyStateQuatHemisphereError(
            f"policy norm stats {path} norm_stats.state.{name} must be a list "
            "of finite numbers"
        )
    return [float(value) for value in values]


def policy_state_quat_hemisphere_from_norm_stats(
    norm_stats_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Resolve the state quaternion hemisphere from one ``norm_stats.json``.

    The qw index is confirmed from the statistics rather than assumed: the
    state must be the 9D YUBI layout (every dimension past 9 is zero padding)
    and dimensions 3..6 must be unit-quaternion components. Anything else is
    refused as an ambiguous qw index.
    """

    path = Path(norm_stats_path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PolicyStateQuatHemisphereError(
            f"cannot read policy norm stats {path}: {exc}"
        ) from exc
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyStateQuatHemisphereError(
            f"policy norm stats {path} are not valid JSON: {exc}"
        ) from exc
    norm_stats = document.get("norm_stats") if isinstance(document, dict) else None
    state = norm_stats.get("state") if isinstance(norm_stats, dict) else None
    if not isinstance(state, dict):
        raise PolicyStateQuatHemisphereError(
            f"policy norm stats {path} have no norm_stats.state statistics"
        )
    q01 = _norm_stats_quantiles(state, "q01", path)
    q99 = _norm_stats_quantiles(state, "q99", path)
    if len(q01) != len(q99) or len(q01) < POLICY_STATE_YUBI_DIM:
        raise PolicyStateQuatHemisphereError(
            f"policy norm stats {path} state quantiles have lengths "
            f"{len(q01)}/{len(q99)}; the {POLICY_STATE_YUBI_DIM}D YUBI state "
            "layout, and so the qw index, cannot be confirmed"
        )
    extra_dims = [
        index
        for index in range(POLICY_STATE_YUBI_DIM, len(q01))
        if q01[index] != 0.0 or q99[index] != 0.0
    ]
    if extra_dims:
        raise PolicyStateQuatHemisphereError(
            f"policy norm stats {path} state has non-padding dimensions "
            f"{extra_dims} beyond the {POLICY_STATE_YUBI_DIM}D YUBI layout "
            "[relative pose xyzw (7), joints (2)]; the qw index is ambiguous"
        )
    for index in range(POLICY_STATE_REL_QUAT_SLICE.start, POLICY_STATE_REL_QUAT_SLICE.stop):
        if (
            q01[index] > q99[index]
            or abs(q01[index]) > _POLICY_STATE_QUAT_COMPONENT_BOUND
            or abs(q99[index]) > _POLICY_STATE_QUAT_COMPONENT_BOUND
        ):
            raise PolicyStateQuatHemisphereError(
                f"policy norm stats {path} state dimension {index} "
                f"(q01={q01[index]}, q99={q99[index]}) is not a unit-quaternion "
                "component; the qw index is ambiguous"
            )
    qw_q01 = q01[POLICY_STATE_REL_QW_INDEX]
    qw_q99 = q99[POLICY_STATE_REL_QW_INDEX]
    classification, hemisphere = classify_policy_state_qw(qw_q01, qw_q99)
    return {
        "hemisphere": hemisphere,
        "classification": classification,
        "rule": POLICY_STATE_QUAT_HEMISPHERE_RULE,
        "norm_stats_path": str(path),
        "norm_stats_sha256": hashlib.sha256(raw).hexdigest(),
        "qw_index": POLICY_STATE_REL_QW_INDEX,
        "qw_q01": qw_q01,
        "qw_q99": qw_q99,
    }


class PolicyStateQuatHemisphereAbsent(PolicyStateQuatHemisphereError):
    """The checkpoint carries no norm stats at all (no assets tree or no file).

    Distinct from stats that exist but are unreadable or ambiguous: plan
    resolution records an absent source as unresolved, and the launch preflight
    refuses it unless the checkpoint artifact is a placeholder test fixture.
    Present-but-broken stats always fail.
    """


def orient_policy_state_quat_wxyz(
    qr_wxyz: np.ndarray,
    quat_hemisphere: str,
    previous_quat_xyzw: np.ndarray | None = None,
) -> np.ndarray:
    """Choose the sign of the relative-hand state quaternion sent to the policy.

    q and -q are the same rotation, but the policy consumes the raw components,
    so the sign must match the checkpoint's training state. The training state
    is the raw ``conj(qL) * qR`` of the dataset, never sign-normalised, and no
    training episode changes sign over time.

    ``positive`` is the historical fixed rule (qw >= 0, added for
    determinism). It stays byte-for-byte unchanged, including near qw ~= 0
    where it flips discontinuously, because every checkpoint validated so far
    ran under it: the cup checkpoint was trained on qw > 0, and the box checkpoints were trained on mixed signs, for which no sign rule
    fits better. Adding continuity here would change their observations.

    ``negative`` serves checkpoints trained only on qw <= 0 (training data:
    117/117 episodes negative). The first observation of a rollout takes
    qw <= 0; every later observation keeps the sign of the previous sent
    quaternion (flip when dot(q, q_prev) < 0). A fixed qw <= 0 rule would jump
    mid-episode whenever the simulated qw crosses zero, which the training
    data never does.
    """

    q = np.asarray(qr_wxyz)
    if quat_hemisphere == "positive":
        if previous_quat_xyzw is not None:
            raise ValueError(
                "the positive state quaternion hemisphere is a fixed rule and "
                "takes no previous quaternion"
            )
        if q[0] < 0.0:
            q = -q
        return q
    if quat_hemisphere == "negative":
        if previous_quat_xyzw is None:
            return -q if q[0] > 0.0 else q
        previous_wxyz = xyzw_to_wxyz(np.asarray(previous_quat_xyzw, dtype=float))
        return -q if float(np.dot(q, previous_wxyz)) < 0.0 else q
    raise ValueError(
        "state quaternion hemisphere must be one of "
        f"{POLICY_STATE_QUAT_HEMISPHERES}, got {quat_hemisphere!r}"
    )


def relative_hand_state(
    left_pos: np.ndarray,
    left_gripper_quat_wxyz: np.ndarray,
    right_pos: np.ndarray,
    right_gripper_quat_wxyz: np.ndarray,
    *,
    q_franka_to_canon_wxyz: np.ndarray | None = None,
    quat_hemisphere: str = "positive",
    previous_quat_xyzw: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``left_hand_root -> right_hand_root`` pose in policy coordinates.

    The quaternion sign follows ``quat_hemisphere``; see
    ``orient_policy_state_quat_wxyz``. ``previous_quat_xyzw`` is the previous
    observation's sent quaternion and is used only by ``negative``.
    """

    q_fc = quat_conj_wxyz(Q_CANON_TO_FRANKA_WXYZ) if q_franka_to_canon_wxyz is None else q_franka_to_canon_wxyz
    qc_l = normalize_quat_wxyz(quat_mul_wxyz(left_gripper_quat_wxyz, q_fc))
    qc_r = normalize_quat_wxyz(quat_mul_wxyz(right_gripper_quat_wxyz, q_fc))

    dp = quat_wxyz_to_matrix(qc_l).T @ (np.asarray(right_pos, dtype=float) - np.asarray(left_pos, dtype=float))
    qr = normalize_quat_wxyz(quat_mul_wxyz(quat_conj_wxyz(qc_l), qc_r))
    qr = orient_policy_state_quat_wxyz(qr, quat_hemisphere, previous_quat_xyzw)
    return np.concatenate([dp, wxyz_to_xyzw(qr)]).astype(np.float32)


class PolicyStateQuatEncoder:
    """Per-rollout relative-hand state encoder that owns sign continuity.

    Create one per rollout (policy client). ``reset`` starts a new episode: the
    next observation chooses its hemisphere afresh.
    """

    def __init__(
        self,
        quat_hemisphere: str,
        *,
        q_franka_to_canon_wxyz: np.ndarray | None = None,
    ) -> None:
        if quat_hemisphere not in POLICY_STATE_QUAT_HEMISPHERES:
            raise ValueError(
                "state quaternion hemisphere must be one of "
                f"{POLICY_STATE_QUAT_HEMISPHERES}, got {quat_hemisphere!r}"
            )
        self.quat_hemisphere = quat_hemisphere
        self._q_franka_to_canon_wxyz = q_franka_to_canon_wxyz
        self._previous_quat_xyzw: np.ndarray | None = None
        self._observations = 0
        self._episode_resets = 0
        self._sign_jumps = 0
        self._opposite_hemisphere = 0
        self._qw_min: float | None = None
        self._qw_max: float | None = None

    def reset(self) -> None:
        self._previous_quat_xyzw = None
        self._episode_resets += 1

    def __call__(
        self,
        left_pos: np.ndarray,
        left_gripper_quat_wxyz: np.ndarray,
        right_pos: np.ndarray,
        right_gripper_quat_wxyz: np.ndarray,
    ) -> np.ndarray:
        previous = self._previous_quat_xyzw
        state = relative_hand_state(
            left_pos,
            left_gripper_quat_wxyz,
            right_pos,
            right_gripper_quat_wxyz,
            q_franka_to_canon_wxyz=self._q_franka_to_canon_wxyz,
            quat_hemisphere=self.quat_hemisphere,
            previous_quat_xyzw=previous if self.quat_hemisphere == "negative" else None,
        )
        quat = np.asarray(state[POLICY_STATE_REL_QUAT_SLICE], dtype=float)
        if previous is not None and float(np.dot(quat, previous)) < 0.0:
            self._sign_jumps += 1
        qw = float(quat[-1])
        if (self.quat_hemisphere == "positive" and qw < 0.0) or (
            self.quat_hemisphere == "negative" and qw > 0.0
        ):
            self._opposite_hemisphere += 1
        self._qw_min = qw if self._qw_min is None else min(self._qw_min, qw)
        self._qw_max = qw if self._qw_max is None else max(self._qw_max, qw)
        self._observations += 1
        self._previous_quat_xyzw = quat.copy()
        return state

    def summary(self) -> dict[str, Any]:
        return {
            "quat_hemisphere": self.quat_hemisphere,
            "observations": int(self._observations),
            "episode_resets": int(self._episode_resets),
            "sent_qw_min": self._qw_min,
            "sent_qw_max": self._qw_max,
            "sent_qw_opposite_hemisphere_observations": int(self._opposite_hemisphere),
            "sign_jumps_between_consecutive_observations": int(self._sign_jumps),
        }


def apply_policy_delta(
    current_pos: np.ndarray,
    current_gripper_quat_wxyz: np.ndarray,
    rel7_xyzw: np.ndarray,
    *,
    q_franka_to_canon_wxyz: np.ndarray | None = None,
    q_canon_to_franka_wxyz: np.ndarray | None = None,
    world_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one policy 7D relative action to a live gripper-frame pose.

    ``current_pos`` is the policy TCP / hand-root point in world coordinates.
    The delta translation is rotated by the current canonical hand-root
    orientation, exactly as ``rel2abs.apply_delta_local`` does for replay.
    """

    q_fc = quat_conj_wxyz(Q_CANON_TO_FRANKA_WXYZ) if q_franka_to_canon_wxyz is None else q_franka_to_canon_wxyz
    q_cf = Q_CANON_TO_FRANKA_WXYZ if q_canon_to_franka_wxyz is None else q_canon_to_franka_wxyz
    q_canon = normalize_quat_wxyz(quat_mul_wxyz(current_gripper_quat_wxyz, q_fc))
    p2, q2_xyzw = apply_delta_local(current_pos, wxyz_to_xyzw(q_canon), rel7_xyzw)
    q2_canon = normalize_quat_wxyz(xyzw_to_wxyz(q2_xyzw))
    q2_gripper = normalize_quat_wxyz(quat_mul_wxyz(q2_canon, q_cf))
    if world_offset is not None:
        p2 = np.asarray(p2, dtype=float) + np.asarray(world_offset, dtype=float)
    return np.asarray(p2, dtype=float), q2_gripper


def integrate_policy_delta_trajectory(
    anchor_pos: np.ndarray,
    anchor_gripper_quat_wxyz: np.ndarray,
    relative_actions_xyzw: np.ndarray,
    *,
    q_franka_to_canon_wxyz: np.ndarray | None = None,
    q_canon_to_franka_wxyz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a policy pose-delta sequence from one observation-time pose.

    Row ``i`` of the returned arrays is the absolute target after applying
    ``relative_actions_xyzw[i]``.  Keeping the integration anchor independent
    of the live plant state prevents controller tracking error from deleting
    the unachieved part of each relative action.
    """

    rel = np.asarray(relative_actions_xyzw, dtype=float)
    if rel.ndim != 2 or rel.shape[1] != 7:
        raise ValueError(f"policy pose deltas must have shape [horizon, 7], got {rel.shape}")
    anchor_p = np.asarray(anchor_pos, dtype=float)
    anchor_q = np.asarray(anchor_gripper_quat_wxyz, dtype=float)
    if anchor_p.shape != (3,):
        raise ValueError(f"policy trajectory anchor position must have shape (3,), got {anchor_p.shape}")
    if anchor_q.shape != (4,):
        raise ValueError(f"policy trajectory anchor quaternion must have shape (4,), got {anchor_q.shape}")
    if not np.all(np.isfinite(rel)) or not np.all(np.isfinite(anchor_p)) or not np.all(np.isfinite(anchor_q)):
        raise ValueError("policy trajectory anchor and deltas must be finite")

    positions = np.empty((rel.shape[0], 3), dtype=float)
    quaternions = np.empty((rel.shape[0], 4), dtype=float)
    pos = anchor_p.copy()
    quat = normalize_quat_wxyz(anchor_q)
    for index, delta in enumerate(rel):
        pos, quat = apply_policy_delta(
            pos,
            quat,
            delta,
            q_franka_to_canon_wxyz=q_franka_to_canon_wxyz,
            q_canon_to_franka_wxyz=q_canon_to_franka_wxyz,
        )
        positions[index] = pos
        quaternions[index] = quat
    return positions, quaternions


def compose_policy_anchored_trajectory(
    anchor_pos: np.ndarray,
    anchor_gripper_quat_wxyz: np.ndarray,
    anchored_actions_xyzw: np.ndarray,
    *,
    q_franka_to_canon_wxyz: np.ndarray | None = None,
    q_canon_to_franka_wxyz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose every fixed-anchor policy waypoint with one observation pose.

    Row ``i`` is interpreted as the complete transform from the pose captured
    for inference to waypoint ``i``.  Rows are therefore applied independently
    to the unchanged anchor; they are deliberately not integrated with one
    another.
    """

    anchored = np.asarray(anchored_actions_xyzw, dtype=float)
    if anchored.ndim != 2 or anchored.shape[1] != 7:
        raise ValueError(
            "anchored policy pose waypoints must have shape [horizon, 7], "
            f"got {anchored.shape}"
        )
    anchor_p = np.asarray(anchor_pos, dtype=float)
    anchor_q = np.asarray(anchor_gripper_quat_wxyz, dtype=float)
    if anchor_p.shape != (3,):
        raise ValueError(
            f"anchored policy trajectory position must have shape (3,), got {anchor_p.shape}"
        )
    if anchor_q.shape != (4,):
        raise ValueError(
            "anchored policy trajectory quaternion must have shape (4,), "
            f"got {anchor_q.shape}"
        )
    if (
        not np.all(np.isfinite(anchored))
        or not np.all(np.isfinite(anchor_p))
        or not np.all(np.isfinite(anchor_q))
    ):
        raise ValueError("anchored policy trajectory anchor and waypoints must be finite")
    if float(np.linalg.norm(anchor_q)) <= _MIN_ACTION_QUAT_NORM:
        raise ValueError("anchored policy trajectory anchor quaternion must be non-zero")
    if anchored.shape[0] and np.any(
        np.linalg.norm(anchored[:, 3:7], axis=1) <= _MIN_ACTION_QUAT_NORM
    ):
        raise ValueError("anchored policy waypoint quaternion must be non-zero")

    positions = np.empty((anchored.shape[0], 3), dtype=float)
    quaternions = np.empty((anchored.shape[0], 4), dtype=float)
    normalized_anchor_q = normalize_quat_wxyz(anchor_q)
    for index, waypoint in enumerate(anchored):
        pos, quat = apply_policy_delta(
            anchor_p,
            normalized_anchor_q,
            waypoint,
            q_franka_to_canon_wxyz=q_franka_to_canon_wxyz,
            q_canon_to_franka_wxyz=q_canon_to_franka_wxyz,
        )
        positions[index] = pos
        quaternions[index] = quat
    return positions, quaternions


@dataclass(frozen=True)
class PolicyImageDisplayTransform:
    """Immutable geometry record for one policy-image display conversion.

    The policy image itself is never modified.  ``policy_content_xywh`` records
    the exact rectangle occupied by the resized source image inside the policy
    tensor, while ``output_content_xywh`` records its aspect-preserving display
    rectangle.  Rectangles use ``(x, y, width, height)`` coordinates.
    """

    algorithm: str
    resize_mode: str
    source_hw: tuple[int, int]
    policy_hw: tuple[int, int]
    policy_content_xywh: tuple[int, int, int, int]
    output_hw: tuple[int, int]
    output_content_xywh: tuple[int, int, int, int]


def _strict_image_hw(value: object, *, name: str) -> tuple[int, int]:
    """Validate an image ``(height, width)`` pair without lossy coercion."""

    try:
        dimensions = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must be a (height, width) pair") from exc
    if len(dimensions) != 2:
        raise ValueError(f"{name} must be a (height, width) pair")
    if any(
        isinstance(dimension, (bool, np.bool_))
        or not isinstance(dimension, (int, np.integer))
        for dimension in dimensions
    ):
        raise ValueError(f"{name} dimensions must be integers")
    height, width = (int(dimension) for dimension in dimensions)
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} dimensions must be positive, got {(height, width)}")
    return height, width


def openpi_letterbox_content_xywh(
    source_hw: tuple[int, int],
    policy_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return OpenPI's exact resized-image rectangle inside a padded image.

    This mirrors ``openpi_client.image_tools._resize_with_pad_pil``: choose the
    larger source-to-target ratio, truncate both resized dimensions with
    ``int``, then truncate half of each remaining dimension for the top/left
    padding.  Geometry comes exclusively from the declared source raster; no
    border pixels are inspected.
    """

    source_height, source_width = _strict_image_hw(source_hw, name="source_hw")
    policy_height, policy_width = _strict_image_hw(policy_hw, name="policy_hw")
    ratio = max(
        source_width / policy_width,
        source_height / policy_height,
    )
    resized_height = int(source_height / ratio)
    resized_width = int(source_width / ratio)
    if resized_height <= 0 or resized_width <= 0:
        raise ValueError(
            "declared source aspect ratio produces an empty OpenPI letterbox "
            f"content rectangle: source_hw={(source_height, source_width)}, "
            f"policy_hw={(policy_height, policy_width)}"
        )
    pad_y = max(0, int((policy_height - resized_height) / 2))
    pad_x = max(0, int((policy_width - resized_width) / 2))
    if (
        pad_x + resized_width > policy_width
        or pad_y + resized_height > policy_height
    ):
        raise ValueError("computed OpenPI letterbox rectangle exceeds the policy image")
    return pad_x, pad_y, resized_width, resized_height


def _aspect_fit_xywh(
    source_hw: tuple[int, int],
    output_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    source_height, source_width = source_hw
    output_height, output_width = output_hw
    scale = min(output_width / source_width, output_height / source_height)
    fitted_width = min(output_width, max(1, int(round(source_width * scale))))
    fitted_height = min(output_height, max(1, int(round(source_height * scale))))
    offset_x = (output_width - fitted_width) // 2
    offset_y = (output_height - fitted_height) // 2
    return offset_x, offset_y, fitted_width, fitted_height


def prepare_policy_wrist_display_image(
    image: np.ndarray,
    *,
    resize_mode: str,
    source_hw: tuple[int, int],
    output_hw: tuple[int, int] = (135, 180),
) -> tuple[np.ndarray, PolicyImageDisplayTransform]:
    """Make an aspect-correct display inset from an unchanged policy image.

    ``pad`` removes only the deterministic OpenPI letterbox rectangle derived
    from ``source_hw``.  It deliberately does not search for black borders,
    because valid camera content can itself be black.  ``stretch`` consumes the
    whole legacy policy image, then restores the declared source aspect ratio.
    Both paths aspect-fit into a black output canvas without flipping, rotating,
    or changing RGB channel order.
    """

    policy_image = np.asarray(image)
    if policy_image.dtype != np.uint8:
        raise ValueError(f"policy display image must be uint8, got {policy_image.dtype}")
    if policy_image.ndim != 3 or policy_image.shape[2] != 3:
        raise ValueError(
            "policy display image must have shape (height, width, 3), "
            f"got {policy_image.shape}"
        )
    declared_source_hw = _strict_image_hw(source_hw, name="source_hw")
    declared_output_hw = _strict_image_hw(output_hw, name="output_hw")
    policy_hw = (int(policy_image.shape[0]), int(policy_image.shape[1]))

    if resize_mode == "pad":
        policy_content_xywh = openpi_letterbox_content_xywh(
            declared_source_hw,
            policy_hw,
        )
        algorithm = "openpi_letterbox_unpad_aspect_fit_v1"
    elif resize_mode == "stretch":
        policy_content_xywh = (0, 0, policy_hw[1], policy_hw[0])
        algorithm = "stretch_restore_declared_aspect_fit_v1"
    else:
        raise ValueError(
            f"unknown policy image resize mode {resize_mode!r}; expected 'pad' or 'stretch'"
        )

    input_x, input_y, input_width, input_height = policy_content_xywh
    content = policy_image[
        input_y : input_y + input_height,
        input_x : input_x + input_width,
    ]
    if content.shape != (input_height, input_width, 3):
        raise ValueError(
            "policy content crop does not match its declared geometry: "
            f"expected {(input_height, input_width, 3)}, got {content.shape}"
        )

    output_content_xywh = _aspect_fit_xywh(declared_source_hw, declared_output_hw)
    output_x, output_y, output_width, output_height = output_content_xywh
    import cv2 as cv

    interpolation = (
        cv.INTER_AREA
        if output_width <= input_width and output_height <= input_height
        else cv.INTER_LINEAR
    )
    resized = cv.resize(
        content,
        (output_width, output_height),
        interpolation=interpolation,
    )
    output = np.zeros((*declared_output_hw, 3), dtype=np.uint8)
    output[
        output_y : output_y + output_height,
        output_x : output_x + output_width,
    ] = resized
    transform = PolicyImageDisplayTransform(
        algorithm=algorithm,
        resize_mode=resize_mode,
        source_hw=declared_source_hw,
        policy_hw=policy_hw,
        policy_content_xywh=policy_content_xywh,
        output_hw=declared_output_hw,
        output_content_xywh=output_content_xywh,
    )
    return output, transform


def resize_policy_image(
    image: np.ndarray,
    mode: str,
    *,
    size: int = POLICY_IMAGE_SIZE,
    pad_fn: Callable[[np.ndarray, int, int], np.ndarray] | None = None,
) -> np.ndarray:
    """Resize a wrist image for policy input.

    ``pad`` uses OpenPI's ``resize_with_pad`` geometry.  ``stretch`` preserves
    the legacy ``cv.resize`` behavior and exists only for backward-compatible
    reruns.
    """

    arr = np.asarray(image)
    if mode == "pad":
        if pad_fn is None:
            from openpi_client import image_tools

            pad_fn = image_tools.resize_with_pad
        return np.asarray(pad_fn(arr, size, size))
    if mode == "stretch":
        import cv2 as cv

        return cv.resize(arr, (size, size), interpolation=cv.INTER_AREA)
    raise ValueError(f"unknown policy image resize mode {mode!r}; expected 'pad' or 'stretch'")


def adjust_image_contrast(
    image: np.ndarray,
    factor: float,
    *,
    midpoint: float = 127.5,
) -> np.ndarray:
    """Adjust uint8 RGB contrast around a fixed mid-gray point."""

    value = float(factor)
    center = float(midpoint)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("contrast factor must be finite and non-negative")
    if not np.isfinite(center):
        raise ValueError("contrast midpoint must be finite")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"contrast image must be uint8, got {arr.dtype}")
    if value == 1.0:
        return arr.copy()
    adjusted = (arr.astype(np.float32) - center) * value + center
    return np.clip(np.rint(adjusted), 0.0, 255.0).astype(np.uint8)


def preprocess_sim_policy_wrist_image(
    image: np.ndarray,
    *,
    contrast_factor: float,
    resize_mode: str,
    size: int = POLICY_IMAGE_SIZE,
    pad_fn: Callable[[np.ndarray, int, int], np.ndarray] | None = None,
) -> np.ndarray:
    """Prepare a simulated wrist image without altering letterbox pixels.

    Contrast is intentionally applied to the rendered image before resizing.
    In ``pad`` mode this keeps the padding created by ``resize_with_pad`` at
    its exact RGB fill value instead of turning black padding gray.
    """

    adjusted = adjust_image_contrast(image, contrast_factor)
    return resize_policy_image(
        adjusted,
        resize_mode,
        size=size,
        pad_fn=pad_fn,
    )
