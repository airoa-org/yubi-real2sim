# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Physics substeps and policy gripper-intent latching."""

import mujoco

from robosuite_env.rollout.build.controllers import _pin_arm_q
from robosuite_env.rollout.build.gripper import (
    apply_yubi_torque_substep as _apply_yubi_torque_substep,
    set_grip_command as _set_grip_command,
)
from robosuite_env.rollout.control.grasp_intent import latch_policy_gripper_intents
from robosuite_env.rollout.control.rule_grasp import (
    _finger_actual,
    _rule_grasp_update,
)


def _latch_policy_grip_intents(state, grip_cmds):
    return latch_policy_gripper_intents(
        grip_cmds,
        {key: _finger_actual(state, key) for key in grip_cmds},
        state.A.rule_grasp_policy_eps_rad,
    )


def _kinematic_physics_substeps(
    state, q_targets, grip_cmds, frame=None, grip_intents=None
):
    if state.YUBI is None:
        mujoco.mj_forward(state.m, state.d)
        if state.A.grasp_mode == "rule":
            _rule_grasp_update(
                state, grip_cmds, frame=frame, grip_intents=grip_intents
            )
        return
    if state.DYNAMIC_YUBI_HOOK is not None:
        # ``--ik-replay`` pins the recorded arm joints and advances MuJoCo
        # directly, without env.step().  Use the same dynamic CbPC hook here so
        # replay still receives one current-control torque update per physics
        # tick and exposes the same cadence counters as dynamic policy rollout.
        state.DYNAMIC_YUBI_HOOK.set_targets(grip_cmds, t=float(state.d.time))
    else:
        for key, gv in grip_cmds.items():
            _set_grip_command(state, key, gv)
    n = max(1, int(state.A.substeps))
    for si in range(n):
        state.d.qfrc_applied[:] = 0.0
        for key, q7 in q_targets.items():
            _pin_arm_q(state, key, q7)
        if state.DYNAMIC_YUBI_HOOK is not None:
            state.DYNAMIC_YUBI_HOOK.apply(t=float(state.d.time))
        else:
            _apply_yubi_torque_substep(state)
        mujoco.mj_step(state.m, state.d)
        state._SIM_TIME += float(state.m.opt.timestep)
        if state.A.grasp_mode == "rule":
            _rule_grasp_update(
                state,
                grip_cmds,
                frame=frame,
                substep=si,
                grip_intents=grip_intents,
            )
        for key, q7 in q_targets.items():
            _pin_arm_q(state, key, q7)
    mujoco.mj_forward(state.m, state.d)


__all__ = ["_kinematic_physics_substeps", "_latch_policy_grip_intents"]
