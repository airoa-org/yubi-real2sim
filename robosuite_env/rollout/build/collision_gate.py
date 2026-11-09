# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""FR3 arm-to-desk collision-gate setup and trajectory previews."""

from __future__ import annotations

import mujoco
import numpy as np

from robosuite_env.robots.fr3.collision_gate import ArmDeskCollisionGate


def configure_arm_desk_collision_gate(state) -> None:
    """Build the joint-IK desk gate from compiled collision geometry."""

    state._ARM_DESK_COLLISION_GATE = None
    state._ARM_LINK_COLLISION_GEOMS = {"R": set(), "L": set()}
    state._DESK_COLLISION_GATE_GEOMS = set()
    if not (
        state.FR3_STRATEGY.uses_joint_ik
        and float(state.A.joint_ik_desk_collision_clearance_mm) > 0.0
    ):
        return

    for key, robot_index in (("R", 0), ("L", 1)):
        expected_link_bodies = {
            f"robot{robot_index}_link{link_index}"
            for link_index in range(1, 8)
        }
        observed_link_bodies = set()
        for geom_id in range(state.m.ngeom):
            body_name = (
                mujoco.mj_id2name(
                    state.m,
                    mujoco.mjtObj.mjOBJ_BODY,
                    state.m.geom_bodyid[geom_id],
                )
                or ""
            )
            if body_name not in expected_link_bodies:
                continue
            if int(state.m.geom_group[geom_id]) != 0 or not (
                int(state.m.geom_contype[geom_id])
                or int(state.m.geom_conaffinity[geom_id])
            ):
                continue
            state._ARM_LINK_COLLISION_GEOMS[key].add(int(geom_id))
            observed_link_bodies.add(body_name)
        missing_links = sorted(expected_link_bodies - observed_link_bodies)
        if missing_links:
            raise RuntimeError(
                "joint_ik desk collision gate found no collision geom for "
                f"{key} arm bodies: {missing_links}"
            )

    state._DESK_COLLISION_GATE_GEOMS = {
        int(geom_id)
        for geom_id in state._SUPPORT_GEOM_IDS
        if int(state.m.geom_contype[geom_id])
        or int(state.m.geom_conaffinity[geom_id])
    }
    if not state._DESK_COLLISION_GATE_GEOMS:
        raise RuntimeError(
            "joint_ik desk collision gate found no active desk collision geoms"
        )
    state._ARM_DESK_COLLISION_GATE = ArmDeskCollisionGate(
        state.m,
        arm_qpos_indices={key: state.ARM[key]["qi"] for key in ("R", "L")},
        arm_geom_ids={
            key: sorted(state._ARM_LINK_COLLISION_GEOMS[key])
            for key in ("R", "L")
        },
        desk_geom_ids=sorted(state._DESK_COLLISION_GATE_GEOMS),
        clearance_m=(
            float(state.A.joint_ik_desk_collision_clearance_mm) / 1000.0
        ),
    )
    print(
        "[sim] joint_ik arm/desk collision gate: "
        f"clearance={float(state.A.joint_ik_desk_collision_clearance_mm):g}mm "
        f"arm_geoms=R{len(state._ARM_LINK_COLLISION_GEOMS['R'])}/"
        f"L{len(state._ARM_LINK_COLLISION_GEOMS['L'])} "
        f"desk_geoms={len(state._DESK_COLLISION_GATE_GEOMS)}; "
        "gripper/object contacts excluded",
        flush=True,
    )


def preview_joint_trajectory_clearance(state, key, target):
    trajectory = state._JOINT_IK_TRAJECTORY[key]
    preview = trajectory.preview_target(
        target,
        sample_period_s=1.0 / float(state.env.control_freq),
        horizon_s=None,
    )
    checks = (
        state._ARM_DESK_COLLISION_GATE.check_path(
            key,
            preview.positions,
            base_qpos=np.asarray(state.d.qpos, dtype=np.float64),
        )
        if (
            state._ARM_DESK_COLLISION_GATE is not None
            and preview.update.accepted
            and len(preview.positions)
        )
        else ()
    )
    return preview, checks


__all__ = [
    "configure_arm_desk_collision_gate",
    "preview_joint_trajectory_clearance",
]
