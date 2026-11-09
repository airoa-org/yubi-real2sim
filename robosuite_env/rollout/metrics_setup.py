# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Frame metrics, success evaluation, and dynamic telemetry."""

from __future__ import annotations

import numpy as np

from robosuite_env.rollout.build.geoms import gripper_cup_contacts
from robosuite_env.rollout.build.gripper import finger_vel
from robosuite_env.rollout.control.rule_grasp import (
    _finger_actual as finger_actual,
    _finger_angles as finger_angles,
)
from robosuite_env.rollout.state.render_state import encode_render_state_row

def frame_debug_metrics(state):
    metrics=state._FRAME_METRICS.collect(state.TGT)
    metrics.update(
        encode_render_state_row(
            np.asarray(state.d.qpos,dtype=np.float64),
            state._RENDER_STATE_CONTRACT,
        )
    )
    return metrics

def tick_success_criterion(state, *, sample_id=None):
    """Advance a stateful success criterion by exactly one evaluation sample."""
    if state._SUCCESS_CRITERION is None:
        return None
    tick = getattr(state._SUCCESS_CRITERION, "tick", None)
    if tick is None:
        return None
    return tick(sample_id=sample_id)

def add_success_metrics(state, rec):
    if state._SUCCESS_CRITERION is not None:
        if state.A.physics != "dynamic" or state.A.state_log_replay or state.A.ik_replay:
            tick_success_criterion(
                state, sample_id=("output_frame", int(rec.get("frame", -1)))
            )
        rec["success"] = state._SUCCESS_CRITERION.evaluate()
    return rec

def _record_dynamic_sample(state,
    repeat,
    joint_command_sample,
    gripper_microsteps,
):
    from robosuite_env.rollout.render.capture import capture_render_sample
    if repeat not in state._frame_telemetry_reps:
        if repeat in state._frame_capture_reps:
            capture_render_sample(state,state.t)
        return
    sample={
        "repeat":int(repeat),
        "sim_time_s":round(float(state.d.time),6),
        "render_sample":bool(repeat in state._frame_render_reps),
        "servo_tick_start":int(repeat),
        "servo_tick_count":1,
        "grip_actual":{},
        "grip_actual_fingers":{},
        "grip_velocity":{},
        "actuator_ctrl":{},
        "actuator_force":{},
        "actuator_force_limit":{},
    }
    for key in ("R","L"):
        sample["grip_actual"][key]=round(finger_actual(state,key),6)
        sample["grip_actual_fingers"][key]={
            side:round(value,6)
            for side,value in finger_angles(state,key).items()
        }
        sample["grip_velocity"][key]=round(finger_vel(state,key),6)
        sample["actuator_ctrl"][key]=[
            round(float(state.d.ctrl[index]),6)
            for index in state.ARM[key]["fa"] if index>=0
        ]
        sample["actuator_force"][key]=[
            round(float(state.d.actuator_force[index]),6)
            for index in state.ARM[key]["fa"] if index>=0
        ]
        sample["actuator_force_limit"][key]=[
            round(float(np.max(np.abs(state.m.actuator_forcerange[index]))),6)
            for index in state.ARM[key]["fa"] if index>=0
        ]
    if state.A.debug_grasp_contact:
        sample["grasp_contact"]=gripper_cup_contacts(state,state.A.grasp_contact_tol_mm)
    if state.DYNAMIC_YUBI_HOOK is not None:
        sample["gripper_control_microsteps"]=gripper_microsteps
        sample["gripper_control_tick_count_total"]=int(
            state.DYNAMIC_YUBI_HOOK.tick_count
        )
        if state.A.grip_ctrl=="cbpc":
            sample["cbpc_microsteps"]=gripper_microsteps
            sample["cbpc_tick_count_total"]=int(state.DYNAMIC_YUBI_HOOK.tick_count)
    if joint_command_sample is not None:
        sample["arm_command"]=joint_command_sample
        for key in ("R","L"):
            q_measured=np.asarray(state.d.qpos[state.ARM[key]["qi"]],dtype=np.float64)
            q_command=np.asarray(state._JOINT_IK_COMMAND[key],dtype=np.float64)
            sample["arm_command"][key].update({
                "q_measured":[round(float(value),6) for value in q_measured],
                "tracking_error_max_rad":round(
                    float(np.max(np.abs(q_command-q_measured))),6
                ),
            })
    state._dynamic_grip_samples.append(sample)
    if repeat in state._frame_capture_reps:
        capture_render_sample(state,state.t)

__all__ = [
    "_record_dynamic_sample",
    "add_success_metrics",
    "frame_debug_metrics",
    "tick_success_criterion",
]
