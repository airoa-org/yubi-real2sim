#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Validated YUBI zero-angle calibration for the simulation geometry."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence
import xml.etree.ElementTree as ET

import numpy as np


CALIBRATION_DEFINITION = (
    "calibrated_left_angle_rad = raw_left_angle_rad + left_offset_rad"
)
REFERENCE_JOINTS = ("finger_joint_left", "finger_joint_right")


class GripperCalibrationError(ValueError):
    """Raised when calibration input or the target model violates its contract."""


@dataclass(frozen=True)
class GripperCalibration:
    path: Path
    sha256: str
    schema_version: int
    unit: str
    left_offset_rad: float
    right_offset_rad: float
    reference_xml_zero_rad: float
    definition: str
    reference_xml: str
    reference_joints: tuple[str, str]
    source_blend: str

def _finite_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool):
        raise GripperCalibrationError(f"{key} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GripperCalibrationError(f"{key} must be a finite number") from exc
    if not math.isfinite(number):
        raise GripperCalibrationError(f"{key} must be a finite number")
    return number


def load_gripper_calibration(
    path: str | Path,
    *,
    expected_reference_xml_name: str = "yubi_left.xml",
) -> GripperCalibration:
    """Load the checked calibration schema and bind it to the left-hand MJCF."""

    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise GripperCalibrationError(
            f"gripper calibration file is not readable: {source}"
        ) from exc
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GripperCalibrationError(
            f"gripper calibration is not valid JSON: {source}"
        ) from exc
    if not isinstance(payload, dict):
        raise GripperCalibrationError("gripper calibration root must be an object")
    if payload.get("schema_version") != 1:
        raise GripperCalibrationError("only gripper calibration schema_version 1 is supported")
    if payload.get("unit") != "rad":
        raise GripperCalibrationError("gripper calibration unit must be 'rad'")
    if payload.get("definition") != CALIBRATION_DEFINITION:
        raise GripperCalibrationError(
            f"gripper calibration definition must be {CALIBRATION_DEFINITION!r}"
        )

    left_offset = _finite_number(payload, "left_offset_rad")
    left_alias = _finite_number(payload, "left")
    if not math.isclose(left_offset, left_alias, rel_tol=0.0, abs_tol=1e-12):
        raise GripperCalibrationError(
            "left and left_offset_rad must encode the same calibration"
        )
    right_offset = _finite_number(payload, "right_offset_rad")
    right_alias = _finite_number(payload, "right")
    if not math.isclose(right_offset, right_alias, rel_tol=0.0, abs_tol=1e-12):
        raise GripperCalibrationError(
            "right and right_offset_rad must encode the same calibration"
        )
    reference_zero = _finite_number(payload, "reference_xml_zero_rad")
    if not math.isclose(reference_zero, 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise GripperCalibrationError(
            "reference_xml_zero_rad must be 0 for body-quaternion calibration"
        )
    joints = payload.get("reference_joints")
    if not isinstance(joints, list) or tuple(joints) != REFERENCE_JOINTS:
        raise GripperCalibrationError(
            f"reference_joints must be {list(REFERENCE_JOINTS)!r}"
        )
    reference_xml = payload.get("reference_xml")
    if not isinstance(reference_xml, str) or not reference_xml:
        raise GripperCalibrationError("reference_xml must be a non-empty path")
    if Path(reference_xml).name != expected_reference_xml_name:
        raise GripperCalibrationError(
            "calibration reference_xml does not match the runtime left-hand asset: "
            f"{Path(reference_xml).name!r} != {expected_reference_xml_name!r}"
        )
    source_blend = payload.get("source_blend", "")
    if not isinstance(source_blend, str):
        raise GripperCalibrationError("source_blend must be a string")

    return GripperCalibration(
        path=source,
        sha256=hashlib.sha256(raw).hexdigest(),
        schema_version=1,
        unit="rad",
        left_offset_rad=left_offset,
        right_offset_rad=right_offset,
        reference_xml_zero_rad=reference_zero,
        definition=CALIBRATION_DEFINITION,
        reference_xml=reference_xml,
        reference_joints=REFERENCE_JOINTS,
        source_blend=source_blend,
    )


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )


def _axis_angle_quat_wxyz(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise GripperCalibrationError("finger joint axis must have non-zero finite norm")
    half = 0.5 * float(angle_rad)
    return np.concatenate(([math.cos(half)], axis / norm * math.sin(half)))


def _reference_joint_frames(
    runtime_xml: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    try:
        root = ET.fromstring(runtime_xml.read_bytes())
    except (OSError, ET.ParseError) as exc:
        raise GripperCalibrationError(
            f"runtime left-hand MJCF is not readable XML: {runtime_xml}"
        ) from exc
    frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for body in root.iter("body"):
        body_quat = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
        if body_quat.shape != (4,) or not np.all(np.isfinite(body_quat)):
            raise GripperCalibrationError(
                f"invalid body quaternion in runtime left-hand MJCF: {body.get('name')}"
            )
        body_quat /= np.linalg.norm(body_quat)
        for joint in body.findall("joint"):
            joint_name = joint.get("name", "")
            if joint_name not in REFERENCE_JOINTS:
                continue
            axis = np.fromstring(joint.get("axis", "0 0 1"), sep=" ")
            if axis.shape != (3,) or not np.all(np.isfinite(axis)):
                raise GripperCalibrationError(
                    f"invalid joint axis in runtime left-hand MJCF: {joint_name}"
                )
            axis /= np.linalg.norm(axis)
            frames[joint_name] = (body_quat.copy(), axis)
    if set(frames) != set(REFERENCE_JOINTS):
        raise GripperCalibrationError(
            "runtime left-hand MJCF does not contain both calibration joints"
        )
    return frames


def _same_quaternion(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        np.allclose(left, right, rtol=0.0, atol=1e-10)
        or np.allclose(left, -right, rtol=0.0, atol=1e-10)
    )


def _apply_gripper_calibration(
    model: Any,
    *,
    compiled_joint_ids: Sequence[int],
    compiled_joint_names: Sequence[str],
    calibration: GripperCalibration,
    runtime_reference_xml: str | Path,
    logical_hand: str,
    offset_rad: float,
    expected_reference_xml_name: str,
) -> dict[str, Any]:
    """Bake one hand's offset into finger body frames without changing raw qpos."""

    if len(compiled_joint_ids) != 2 or len(compiled_joint_names) != 2:
        raise GripperCalibrationError(
            f"logical-{logical_hand} calibration requires exactly two joints"
        )
    if len(set(int(value) for value in compiled_joint_ids)) != 2:
        raise GripperCalibrationError(
            f"logical-{logical_hand} finger joint IDs must be distinct"
        )
    runtime_xml = Path(runtime_reference_xml).expanduser().resolve()
    if runtime_xml.name != expected_reference_xml_name:
        raise GripperCalibrationError(
            f"runtime {logical_hand}-hand MJCF does not match calibration reference asset"
        )
    try:
        runtime_xml_bytes = runtime_xml.read_bytes()
        runtime_xml_sha256 = hashlib.sha256(runtime_xml_bytes).hexdigest()
    except OSError as exc:
        raise GripperCalibrationError(
            f"runtime left-hand MJCF is not readable: {runtime_xml}"
        ) from exc
    reference_frames = _reference_joint_frames(runtime_xml)

    qpos0_before = np.asarray(model.qpos0, dtype=float).copy()
    ranges_before = np.asarray(model.jnt_range, dtype=float).copy()
    ctrlrange_before = np.asarray(model.actuator_ctrlrange, dtype=float).copy()
    body_ids: set[int] = set()
    joint_reports: list[dict[str, Any]] = []
    for reference_name, raw_joint_id, compiled_name in zip(
        calibration.reference_joints, compiled_joint_ids, compiled_joint_names
    ):
        joint_id = int(raw_joint_id)
        if joint_id < 0 or joint_id >= int(model.njnt):
            raise GripperCalibrationError(f"invalid compiled joint ID: {joint_id}")
        if not str(compiled_name).endswith(reference_name):
            raise GripperCalibrationError(
                f"compiled joint {compiled_name!r} does not match {reference_name!r}"
            )
        body_id = int(model.jnt_bodyid[joint_id])
        if body_id in body_ids:
            raise GripperCalibrationError("finger joints must belong to distinct bodies")
        body_ids.add(body_id)
        axis = np.asarray(model.jnt_axis[joint_id], dtype=float).copy()
        old_quat = np.asarray(model.body_quat[body_id], dtype=float).copy()
        old_norm = float(np.linalg.norm(old_quat))
        if not math.isfinite(old_norm) or old_norm <= 1e-12:
            raise GripperCalibrationError(
                f"finger body quaternion for {compiled_name!r} is invalid"
            )
        old_quat /= old_norm
        reference_quat, reference_axis = reference_frames[reference_name]
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12 or not np.allclose(
            axis / axis_norm, reference_axis, rtol=0.0, atol=1e-10
        ):
            raise GripperCalibrationError(
                f"compiled joint axis for {compiled_name!r} does not match the reference MJCF"
            )
        if not _same_quaternion(old_quat, reference_quat):
            raise GripperCalibrationError(
                f"compiled finger body for {compiled_name!r} is not at the "
                "reference MJCF frame; calibration may already be applied"
            )
        offset_quat = _axis_angle_quat_wxyz(axis, float(offset_rad))
        new_quat = _quat_multiply_wxyz(old_quat, offset_quat)
        new_quat /= np.linalg.norm(new_quat)
        model.body_quat[body_id] = new_quat
        joint_reports.append(
            {
                "reference_joint": reference_name,
                "compiled_joint": str(compiled_name),
                "joint_id": joint_id,
                "body_id": body_id,
                "local_axis": axis.tolist(),
                "body_quat_before_wxyz": old_quat.tolist(),
                "body_quat_after_wxyz": new_quat.tolist(),
            }
        )

    if not np.array_equal(np.asarray(model.qpos0), qpos0_before):
        raise GripperCalibrationError("calibration unexpectedly changed model.qpos0")
    if not np.array_equal(np.asarray(model.jnt_range), ranges_before):
        raise GripperCalibrationError("calibration unexpectedly changed joint ranges")
    if not np.array_equal(np.asarray(model.actuator_ctrlrange), ctrlrange_before):
        raise GripperCalibrationError("calibration unexpectedly changed actuator ctrlrange")

    return {
        "enabled": True,
        "method": "compiled_body_quat_local_axis_offset",
        "logical_hand": logical_hand,
        "policy_coordinate": "raw",
        "path": str(calibration.path),
        "sha256": calibration.sha256,
        "schema_version": calibration.schema_version,
        "unit": calibration.unit,
        "definition": calibration.definition,
        "offset_rad": float(offset_rad),
        "left_offset_rad": calibration.left_offset_rad,
        "right_offset_rad": calibration.right_offset_rad,
        "reference_xml_zero_rad": calibration.reference_xml_zero_rad,
        "reference_xml": calibration.reference_xml,
        "runtime_reference_xml": str(runtime_xml),
        "runtime_reference_xml_sha256": runtime_xml_sha256,
        "source_blend": calibration.source_blend,
        "joints": joint_reports,
        "invariants": {
            "qpos0_unchanged": True,
            "joint_ranges_unchanged": True,
            "actuator_ctrlrange_unchanged": True,
            "right_hand_unchanged": logical_hand == "L",
            "left_hand_unchanged": logical_hand == "R",
        },
    }


def apply_left_gripper_calibration(
    model: Any,
    *,
    compiled_joint_ids: Sequence[int],
    compiled_joint_names: Sequence[str],
    calibration: GripperCalibration,
    runtime_reference_xml: str | Path,
) -> dict[str, Any]:
    """Bake the left offset into finger body frames without changing raw qpos."""

    return _apply_gripper_calibration(
        model,
        compiled_joint_ids=compiled_joint_ids,
        compiled_joint_names=compiled_joint_names,
        calibration=calibration,
        runtime_reference_xml=runtime_reference_xml,
        logical_hand="L",
        offset_rad=calibration.left_offset_rad,
        expected_reference_xml_name=Path(calibration.reference_xml).name,
    )


def apply_right_gripper_calibration(
    model: Any,
    *,
    compiled_joint_ids: Sequence[int],
    compiled_joint_names: Sequence[str],
    calibration: GripperCalibration,
    runtime_reference_xml: str | Path,
) -> dict[str, Any]:
    """Bake the same checked-in offset into the right YUBI without raw qpos changes."""

    return _apply_gripper_calibration(
        model,
        compiled_joint_ids=compiled_joint_ids,
        compiled_joint_names=compiled_joint_names,
        calibration=calibration,
        runtime_reference_xml=runtime_reference_xml,
        logical_hand="R",
        offset_rad=calibration.right_offset_rad,
        expected_reference_xml_name="yubi_right.xml",
    )


def apply_bilateral_gripper_calibration(
    model: Any,
    *,
    right_joint_ids: Sequence[int],
    right_joint_names: Sequence[str],
    left_joint_ids: Sequence[int],
    left_joint_names: Sequence[str],
    calibration: GripperCalibration,
    right_runtime_reference_xml: str | Path,
    left_runtime_reference_xml: str | Path,
) -> dict[str, Any]:
    """Apply the checked calibration to both logical YUBI hands.

    Rollout/rendering entry points should use this function instead of calling
    a one-hand helper.  It deliberately requires both joint sets and returns
    one provenance record containing both hand reports, so a caller cannot
    silently calibrate only one side.
    """

    right_report = apply_right_gripper_calibration(
        model,
        compiled_joint_ids=right_joint_ids,
        compiled_joint_names=right_joint_names,
        calibration=calibration,
        runtime_reference_xml=right_runtime_reference_xml,
    )
    left_report = apply_left_gripper_calibration(
        model,
        compiled_joint_ids=left_joint_ids,
        compiled_joint_names=left_joint_names,
        calibration=calibration,
        runtime_reference_xml=left_runtime_reference_xml,
    )
    return {
        "enabled": True,
        "method": "compiled_body_quat_local_axis_offset_bilateral",
        "logical_hand": "R+L",
        "policy_coordinate": "raw",
        "path": str(calibration.path),
        "sha256": calibration.sha256,
        "schema_version": calibration.schema_version,
        "unit": calibration.unit,
        "definition": calibration.definition,
        "offset_rad": calibration.left_offset_rad,
        "left_offset_rad": calibration.left_offset_rad,
        "right_offset_rad": calibration.right_offset_rad,
        "reference_xml_zero_rad": calibration.reference_xml_zero_rad,
        "reference_xml": calibration.reference_xml,
        "source_blend": calibration.source_blend,
        "hands": {"R": right_report, "L": left_report},
        "joints": right_report["joints"] + left_report["joints"],
        "runtime_reference_xml_right": right_report["runtime_reference_xml"],
        "runtime_reference_xml_sha256_right": right_report[
            "runtime_reference_xml_sha256"
        ],
        "runtime_reference_xml_left": left_report["runtime_reference_xml"],
        "runtime_reference_xml_sha256_left": left_report[
            "runtime_reference_xml_sha256"
        ],
        "invariants": {
            "qpos0_unchanged": True,
            "joint_ranges_unchanged": True,
            "actuator_ctrlrange_unchanged": True,
            "right_hand_unchanged": False,
            "left_hand_unchanged": False,
        },
    }
