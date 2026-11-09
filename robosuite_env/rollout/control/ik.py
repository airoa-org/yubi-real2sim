# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""FR3 IK, Cartesian command, and dynamic action helpers."""

from __future__ import annotations

import functools

import mujoco
import numpy as np

from robosuite_env.robots.fr3.runtime_ik import solve_and_pin_nullspace_ik
from robosuite_env.rollout.control.joint_trajectory import StatefulJerkLimitedTrajectory
from robosuite_env.rollout.control.pose_math import policy_tcp_error


def ik_solve(state,key,tp,tq):
    return solve_and_pin_nullspace_ik(state._ABSOLUTE_JOINT_IK,state.m,state.d,key,state.ARM[key],tp,tq,functools.partial(policy_tcp_error,state))

def _dynamic_env_action(state,arm_commands, grip_commands):
    """Build the two-robot action without assuming a controller's dimensions."""

    per_robot=[]
    for key,robot_index in (("R",0),("L",1)):
        grip=float(np.clip((float(grip_commands[key])/0.7854)*2.0-1.0,-1.0,1.0))
        per_robot.append(
            state.env.robots[robot_index].create_action_vector(
                {
                    "right":np.asarray(arm_commands[key],dtype=np.float64),
                    "right_gripper":np.asarray([grip],dtype=np.float64),
                }
            )
        )
    action=np.concatenate(per_robot)
    if action.shape != (state.env.action_dim,):
        raise RuntimeError(
            f"dynamic action builder produced {action.shape}, expected {(state.env.action_dim,)}"
        )
    return action

def _new_joint_ik_trajectory(state,key,position,velocity=None,acceleration=None):
    return StatefulJerkLimitedTrajectory(
        np.asarray(position,dtype=np.float64),
        velocity=(
            None
            if velocity is None
            else np.asarray(velocity,dtype=np.float64)
        ),
        acceleration=(
            None
            if acceleration is None
            else np.asarray(acceleration,dtype=np.float64)
        ),
        delta_time_s=1.0/float(state.env.control_freq),
        max_velocity_rad_s=float(state.A.joint_ik_max_velocity),
        max_acceleration_rad_s2=float(state.A.joint_ik_max_acceleration),
        max_jerk_rad_s3=float(state.A.joint_ik_max_jerk),
        position_lower=np.asarray(
            state.ARM[key]["jr"][:,0],dtype=np.float64
        ),
        position_upper=np.asarray(
            state.ARM[key]["jr"][:,1],dtype=np.float64
        ),
    )

def _guard_ik_result(state,key,tp,tq,q7,res,conv,prev):
    dqv=np.abs(q7-prev)*180.0/np.pi if prev is not None else np.zeros(7)
    raw_jump=bool(state.A.jump_deg>0 and prev is not None and float(dqv.max())>state.A.jump_deg)
    meta={"raw_jump":raw_jump,"held":False,"retries":0,
          "raw_dqmax_deg":float(dqv.max()),"raw_jump_joint":(int(dqv.argmax())+1 if prev is not None else 0)}
    if raw_jump and state.A.ik_jump_reject!="off":
        if float(dqv.max())>state.A.jump_deg:
            q7=prev.copy()
            state.d.qpos[state.ARM[key]["qi"]]=q7; mujoco.mj_forward(state.m,state.d)
            res=policy_tcp_error(state,key,tp,tq); conv=False; dqv=np.zeros(7); meta["held"]=True
        else:
            state.d.qpos[state.ARM[key]["qi"]]=q7; mujoco.mj_forward(state.m,state.d)
    jump=bool(state.A.jump_deg>0 and prev is not None and float(dqv.max())>state.A.jump_deg)
    return q7,res,conv,dqv,jump,meta


def solve_ik_guarded(state,key,tp,tq,prev):
    q7,res,conv=ik_solve(state,key,tp,tq)
    return _guard_ik_result(state,key,tp,tq,q7,res,conv,prev)


__all__ = [
    "_dynamic_env_action",
    "_new_joint_ik_trajectory",
    "ik_solve",
    "solve_ik_guarded",
]
