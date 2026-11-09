# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Runtime adapter from scratch-state FR3 IK to live MuJoCo qpos pinning."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import mujoco
import numpy as np

from .absolute_ik import AbsoluteFR3IK


FR3_NULLSPACE_MODES = ("home", "off")
"""Null-space posture modes for the ``joint_ik`` arm controller.

``home`` projects a canonical-home posture objective through the Jacobian
null space with ``--joint-ik-nullspace-gain``; ``off`` solves the pure
whole-pose IK (effective gain 0) so the two can be compared with one switch.
"""


def effective_nullspace_gain(mode: str, configured_gain: float) -> float:
    """Return the null-space gain the IK solver actually receives."""

    if mode not in FR3_NULLSPACE_MODES:
        raise ValueError(
            f"fr3 nullspace mode must be one of {FR3_NULLSPACE_MODES}, got {mode!r}"
        )
    return 0.0 if mode == "off" else float(configured_gain)


def controller_summary_name(
    physics: str, controller: str, nullspace: str = "home"
) -> str:
    """Stable run-summary name for the selected FR3 arm backend."""

    if nullspace not in FR3_NULLSPACE_MODES:
        raise ValueError(
            f"fr3 nullspace mode must be one of {FR3_NULLSPACE_MODES}, got {nullspace!r}"
        )
    if controller == "cartesian_impedance":
        if physics != "dynamic":
            raise ValueError("Cartesian impedance requires dynamic physics")
        return "fr3_cartesian_impedance_torque"
    if controller != "joint_ik":
        raise ValueError(f"FR3 controller {controller!r} has no control-path summary")
    if physics == "kinematic":
        return (
            "fr3_mujoco_home_nullspace_ik"
            if nullspace == "home"
            else "fr3_mujoco_ik"
        )
    return (
        "fr3_absolute_home_nullspace_joint_ik"
        if nullspace == "home"
        else "fr3_absolute_joint_ik"
    )


def controller_path_summary(physics: str, controller: str) -> str:
    """Human-readable control pipeline recorded in each run summary."""

    if controller == "cartesian_impedance":
        if physics != "dynamic":
            raise ValueError("Cartesian impedance requires dynamic physics")
        return "deploy_cmd_twist_to_persistent_equilibrium_to_inertia_aware_impedance_torque"
    if controller != "joint_ik":
        raise ValueError(f"FR3 controller {controller!r} has no control-path summary")
    if physics == "kinematic":
        return "mujoco_home_nullspace_ik_to_direct_qpos_pin"
    return "absolute_whole_pose_ik_to_stateful_jerk_limited_joint_position_pd"


def solve_and_pin_nullspace_ik(
    solver: AbsoluteFR3IK,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    arm_spec: Mapping[str, Any],
    target_position: Any,
    target_quaternion_wxyz: Any,
    error_fn: Callable[[str, Any, Any], tuple[float, float]],
) -> tuple[np.ndarray, tuple[float, float], bool]:
    """Solve from the live arm state, pin the result, and report TCP error."""

    position = np.asarray(target_position, dtype=np.float64)
    quaternion = np.asarray(target_quaternion_wxyz, dtype=np.float64)
    qpos_indices = np.asarray(arm_spec["qi"], dtype=int)
    seed = np.asarray(data.qpos[qpos_indices], dtype=np.float64).copy()
    result = solver.solve(
        side,
        position,
        quaternion,
        seed_q=seed,
        continuity_q=seed,
    )
    q = np.asarray(result.q, dtype=np.float64).copy()
    data.qpos[qpos_indices] = q
    mujoco.mj_forward(model, data)
    return q, error_fn(side, position, quaternion), bool(result.converged)


__all__ = [
    "AbsoluteFR3IK",
    "FR3_NULLSPACE_MODES",
    "controller_path_summary",
    "controller_summary_name",
    "effective_nullspace_gain",
    "solve_and_pin_nullspace_ik",
]
