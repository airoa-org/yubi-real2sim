# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""YUBI and gripper actuator helpers."""

from __future__ import annotations

import mujoco
import numpy as np

from robosuite_env.rollout.control.grasp_intent import YUBI_JAW_MAX_RAD
from robosuite_env.rollout.control.rule_grasp import _finger_actual


def finger_vel(state, key):
    return float(
        0.5
        * (
            state.d.qvel[state.ARM[key]["fd"][0]]
            + state.d.qvel[state.ARM[key]["fd"][1]]
        )
    )


def grip_state_obs(state):
    return np.array(
        [_finger_actual(state, "L"), _finger_actual(state, "R")], np.float32
    )


def disable_finger_position_actuators(state):
    for key in ("R", "L"):
        for aa in state.ARM[key]["fa"]:
            if aa >= 0:
                state.m.actuator_gainprm[aa, :] = 0.0
                state.m.actuator_biasprm[aa, :] = 0.0
                state.d.ctrl[aa] = 0.0
    print(
        "[sim] disabled built-in finger position actuators for CbPC qfrc control",
        flush=True,
    )


def setup_yubi_cbpc(state):
    if state.A.grip_ctrl != "cbpc":
        return
    if state.A.physics == "dynamic" and not state.A.dynamic_grip_cbpc:
        return
    from robosuite_env.robots.yubi.control import (
        BimanualYubiController,
        YubiConfig,
        needs_calibration,
    )

    cfg = YubiConfig.load(state.A.grip_cfg)
    cfg.action_mode = "user_angle_rad"
    cfg.controller.torque_scale = float(state.A.torque_scale)
    for side_cfg in (cfg.left, cfg.right):
        side_cfg.user_to_joint_scale = 1.0
        side_cfg.max_open_user_rad = YUBI_JAW_MAX_RAD
        side_cfg.joint_max_rad = YUBI_JAW_MAX_RAD
    miss = needs_calibration(cfg)
    if miss:
        raise RuntimeError(
            f"YUBI CbPC config is not runnable; missing {miss}: {state.A.grip_cfg}"
        )
    state.YUBI = BimanualYubiController(cfg, float(state.m.opt.timestep))
    disable_finger_position_actuators(state)
    print(
        f"[sim] YUBI CbPC active cfg={state.A.grip_cfg} "
        f"dt={state.m.opt.timestep:.4f} "
        f"torque_scale={cfg.controller.torque_scale}",
        flush=True,
    )


def reset_yubi_from_current(state):
    if state.YUBI is None:
        return
    state.YUBI.left.reset(_finger_actual(state, "L"))
    state.YUBI.right.reset(_finger_actual(state, "R"))
    print(
        f"[sim] YUBI reset from model q: L={_finger_actual(state, 'L'):.4f} "
        f"R={_finger_actual(state, 'R'):.4f}",
        flush=True,
    )


def force_finger_state(state, gL, gR):
    for key, gv in (("L", gL), ("R", gR)):
        gv = float(np.clip(gv, 0.0, YUBI_JAW_MAX_RAD))
        for fa in state.ARM[key]["fj"]:
            state.d.qpos[fa] = gv
        for fd in state.ARM[key]["fd"]:
            state.d.qvel[fd] = 0.0
    mujoco.mj_forward(state.m, state.d)
    reset_yubi_from_current(state)


def set_grip_command(state, key, gv, t=None):
    gv = float(np.clip(gv, 0.0, YUBI_JAW_MAX_RAD))
    if state.YUBI is not None:
        command_time = state._SIM_TIME if t is None else float(t)
        getattr(state.YUBI, "right" if key == "R" else "left").set_action(
            gv, command_time
        )
    else:
        for aa in state.ARM[key]["fa"]:
            if aa >= 0:
                state.d.ctrl[aa] = gv


def apply_yubi_torque_substep(state):
    if state.YUBI is None:
        return
    for key, side in (("R", "right"), ("L", "left")):
        h = getattr(state.YUBI, side)
        tau = float(
            h.compute_torque(
                state._SIM_TIME, _finger_actual(state, key), finger_vel(state, key)
            )
        )
        for fd in state.ARM[key]["fd"]:
            state.d.qfrc_applied[fd] += 0.5 * tau


def yubi_telemetry(state):
    if state.YUBI is None:
        return None
    out = {}
    for key, side in (("R", "right"), ("L", "left")):
        te = getattr(state.YUBI, side).telemetry
        out[key] = {
            "tau_cmd": round(float(te.get("tau_cmd", 0.0)), 5),
            "sat": bool(te.get("saturated", False)),
            "sat_ratio": round(float(te.get("sat_ratio", 0.0)), 3),
            "err": round(float(te.get("e", 0.0)), 5),
        }
    return out


def grip_actuator_debug(state, grip_cmds=None):
    out = {}
    for key in ("R", "L"):
        qpos = [float(state.d.qpos[j]) for j in state.ARM[key]["fj"]]
        qvel = [float(state.d.qvel[j]) for j in state.ARM[key]["fd"]]
        ctrls = []
        forces = []
        ctrlranges = []
        forceranges = []
        for aa in state.ARM[key]["fa"]:
            if aa < 0:
                ctrls.append(None)
                forces.append(None)
                ctrlranges.append(None)
                forceranges.append(None)
                continue
            ctrls.append(float(state.d.ctrl[aa]))
            forces.append(float(state.d.actuator_force[aa]))
            ctrlranges.append([float(x) for x in state.m.actuator_ctrlrange[aa]])
            forceranges.append([float(x) for x in state.m.actuator_forcerange[aa]])
        rec = {
            "qpos": [round(x, 5) for x in qpos],
            "qvel": [round(x, 5) for x in qvel],
            "mean_qpos": round(float(np.mean(qpos)), 5),
            "mean_qvel": round(float(np.mean(qvel)), 5),
            "ctrl": [round(x, 5) if x is not None else None for x in ctrls],
            "actuator_force": [
                round(x, 5) if x is not None else None for x in forces
            ],
            "ctrlrange": ctrlranges,
            "forcerange": forceranges,
        }
        if grip_cmds is not None and key in grip_cmds:
            target = float(grip_cmds[key])
            rec["cmd"] = round(target, 5)
            rec["cmd_minus_mean_qpos"] = round(
                target - float(np.mean(qpos)), 5
            )
        out[key] = rec
    return out


__all__ = [
    "apply_yubi_torque_substep",
    "disable_finger_position_actuators",
    "finger_vel",
    "force_finger_state",
    "grip_actuator_debug",
    "grip_state_obs",
    "reset_yubi_from_current",
    "set_grip_command",
    "setup_yubi_cbpc",
    "yubi_telemetry",
]
