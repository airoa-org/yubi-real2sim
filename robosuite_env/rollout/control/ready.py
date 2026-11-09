# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Ready-pose gates and setup-time grasp control helpers."""

import numpy as np

from robosuite_env.robots.fr3.initial_pose import joint_ready_gate_report
from robosuite_env.rollout.control.pose_math import policy_tcp_error

def _joint_ik_ready_gate_report(state):
    if state._INITIAL_ARM_JOINT_Q is not None:
        report=joint_ready_gate_report(
            {
                key:np.asarray(
                    state.d.qpos[state.ARM[key]["qi"]],dtype=np.float64
                ).copy()
                for key in ("R","L")
            },
            state._INITIAL_ARM_JOINT_Q,
            tolerance_deg=float(state.A.initial_arm_joints_tol_deg),
        )
        if not report["passed"]:
            failures=[
                f"{key}=j{state['max_error_joint']} "
                f"{state['max_error_deg']:.6f}deg"
                for key,state in report["arms"].items()
                if not state["passed"]
            ]
            raise RuntimeError(
                "specified initial arm joint gate failed "
                f"(limit {float(state.A.initial_arm_joints_tol_deg):g}deg): "
                + ", ".join(failures)
            )
        return report
    report={}
    failures=[]
    for key in ("R","L"):
        position_m,orientation_rad=policy_tcp_error(state,key,*state.READY[key])
        position_mm=float(position_m)*1000.0
        orientation_deg=float(np.degrees(orientation_rad))
        report[key]={
            "position_error_mm":position_mm,
            "orientation_error_deg":orientation_deg,
        }
        if (
            position_mm>float(state.A.joint_ik_ready_position_tol_mm)
            or orientation_deg>float(state.A.joint_ik_ready_orientation_tol_deg)
        ):
            failures.append(
                f"{key}={position_mm:.3f}mm/{orientation_deg:.3f}deg"
            )
    if failures:
        raise RuntimeError(
            "joint_ik ready pose gate failed "
            f"(limits {float(state.A.joint_ik_ready_position_tol_mm):g}mm/"
            f"{float(state.A.joint_ik_ready_orientation_tol_deg):g}deg): "
            + ", ".join(failures)
        )
    return report


__all__ = [
    "_joint_ik_ready_gate_report",
]
