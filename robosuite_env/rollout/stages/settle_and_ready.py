# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Object settling and validated ready-pose stage."""

import json

import mujoco
import numpy as np

from robosuite_env.rollout.build.gripper import (
    force_finger_state as _force_finger_state,
    reset_yubi_from_current as _reset_yubi_from_current,
)
from robosuite_env.rollout.control.ik import (
    _dynamic_env_action,
    ik_solve,
)
from robosuite_env.rollout.control.joint_trajectory import bounded_minimum_jerk_segment
from robosuite_env.rollout.control.pose_math import qmul
from robosuite_env.rollout.control.ready import _joint_ik_ready_gate_report
from robosuite_env.rollout.grasp.script import GraspTestScript
from robosuite_env.rollout.observe.deploy_anchor import load_policy_deploy_ready
from robosuite_env.rollout.setup.frames import _qmat
from robosuite_env.rollout.state.cache import validate_saved_settled_init_payload
from robosuite_env.rollout.state.init_state import (
    _load_sim_init_state,
    _validate_loaded_object_state_matches_payload,
    _write_sim_init_state,
)
from robosuite_env.rollout.state.mimicgen_init import _apply_mimicgen_training_init
from robosuite_env.rollout.state.settle import (
    _restore_ready_objects_stationary,
    _settle_ready_objects,
    _settled_init_state_contract,
    _validate_settled_objects,
)

def _target_collision_geoms(state):
    """The target body's own collision geoms; fails closed when it has none."""

    body_id=int(state.m.jnt_bodyid[int(state._TASK_OBJECT_BINDINGS["cup"]["joint_id"])])
    geoms=[
        geom for geom in range(int(state.m.ngeom))
        if int(state.m.geom_bodyid[geom])==body_id
        and (int(state.m.geom_contype[geom]) or int(state.m.geom_conaffinity[geom]))
    ]
    if not geoms:
        raise SystemExit(
            "[sim] grasp-test --grasp-hand both: the target body has no collision geom"
        )
    return geoms


def _target_half_extent_x(state):
    """Half the target body's world x extent, from its own collision geoms.

    The two-handed probe puts one TCP at each end of the object, so the ends
    have to be where the collision hull ends. ``geom_rbound`` is a bounding
    sphere and overshoots a flat object badly -- 139 mm against the gingham
    container's own 121 mm -- which puts both hands off the object and onto the
    desk. Each geom's own AABB, rotated into the world, is the extent.
    """
    xs=[]
    for geom in _target_collision_geoms(state):
        centre=np.asarray(state.d.geom_xpos[geom],dtype=float)
        rotation=np.asarray(state.d.geom_xmat[geom],dtype=float).reshape(3,3)
        local_centre=np.asarray(state.m.geom_aabb[geom][:3],dtype=float)
        half=np.asarray(state.m.geom_aabb[geom][3:],dtype=float)
        world_centre=centre+rotation@local_centre
        reach=float(np.abs(rotation[0])@half)
        xs.extend((world_centre[0]-reach,world_centre[0]+reach))
    return 0.5*(max(xs)-min(xs))


def _two_hand_grasp_sides(state):
    """Which hand stands on -x and which on +x, read from the compiled rig.

    Duo's robot0 (R) is based on -x and robot1 (L) on +x; the flat FR3
    calibration is the other way round. Reading the hands' own positions keeps
    the probe from crossing the arms on a rig it was not written against.
    """
    positions={hand:float(state.d.xpos[int(state.ARM[hand]["eef"])][0]) for hand in ("R","L")}
    if positions["R"]<=positions["L"]:
        return {"R":-1.0,"L":+1.0}
    return {"R":+1.0,"L":-1.0}


def _load_grasp_pose_json(path):
    """Read and check the per-hand grasp poses the probe is asked to use."""

    try:
        with open(path) as handle:
            poses=json.load(handle)
    except OSError as exc:
        raise SystemExit(f"[sim] --grasp-pose-json cannot be read: {exc}") from exc
    except ValueError as exc:
        raise SystemExit(f"[sim] --grasp-pose-json is not valid JSON: {exc}") from exc
    checked={}
    for hand in ("R","L"):
        entry=poses.get(hand)
        if not isinstance(entry,dict):
            raise SystemExit(f"[sim] --grasp-pose-json is missing hand {hand!r}")
        position=np.asarray(entry.get("pos_rel_m",()),dtype=float)
        quaternion=np.asarray(entry.get("quat_wxyz",()),dtype=float)
        if position.shape!=(3,) or not np.all(np.isfinite(position)):
            raise SystemExit(
                f"[sim] --grasp-pose-json hand {hand!r}: pos_rel_m must be three finite numbers"
            )
        if quaternion.shape!=(4,) or not np.all(np.isfinite(quaternion)):
            raise SystemExit(
                f"[sim] --grasp-pose-json hand {hand!r}: quat_wxyz must be four finite numbers"
            )
        norm=float(np.linalg.norm(quaternion))
        if norm<1e-9:
            raise SystemExit(
                f"[sim] --grasp-pose-json hand {hand!r}: quat_wxyz has no length"
            )
        checked[hand]={"pos_rel_m":position,"quat_wxyz":quaternion/norm}
    return checked


def _configure_two_hand_grasp(state):
    """Place both hands' scripted poses around the target's current pose.

    Called once before the ready drive, and again after settling and any
    ``--init-state-in``, because those move the target and every pose here is
    measured from it.
    """
    target=np.asarray(state.cup,dtype=float)
    by_hand={}
    if state.A.grasp_pose_json:
        _target_collision_geoms(state)
        poses=_load_grasp_pose_json(state.A.grasp_pose_json)
        for hand,entry in poses.items():
            grasp=target+entry["pos_rel_m"]
            by_hand[hand]={
                "pre":grasp+np.array([0.0,0.0,state.A.pregrasp_z-state.A.grasp_z]),
                "grasp":grasp,
                "top":grasp+np.array([0.0,0.0,state.A.lift_h]),
                "quat":entry["quat_wxyz"].copy(),
            }
        span=None
    else:
        span=(
            float(state.A.grasp_span) if state.A.grasp_span>0.0
            else _target_half_extent_x(state)
        )
        gx=float(target[0])+state.A.grasp_dx
        gy=float(target[1])+state.A.grasp_dy
        for hand,sign in _two_hand_grasp_sides(state).items():
            hx=gx+sign*span
            by_hand[hand]={
                "pre":np.array([hx,gy,state.A.pregrasp_z]),
                "grasp":np.array([hx,gy,state.A.grasp_z]),
                "top":np.array([hx,gy,state.A.grasp_z+state.A.lift_h]),
                "quat":state._GQ.copy(),
            }
    state._GRASP_SPAN=span
    state._P_BY_HAND=by_hand
    # The single-hand fields the startup print and the summary read.
    state._P_PRE=by_hand["R"]["pre"]
    state._P_GRASP=by_hand["R"]["grasp"]
    state._P_TOP=by_hand["R"]["top"]
    state._P_PARK=by_hand["R"]["top"].copy()
    if state._PB is not None:
        state.grasp_scripts={
            hand:GraspTestScript(
                state,state._PB,poses["quat"],poses["pre"],poses["grasp"],poses["top"]
            )
            for hand,poses in by_hand.items()
        }
    return {hand:(poses["pre"].copy(),poses["quat"].copy())
            for hand,poses in by_hand.items()}


def _ready_pin_is_planar(state):
    """Whether the ready drive holds only the target's planar pose.

    Cup P&P keeps its historical full-pose pin unchanged. For
    box-pack a full-pose pin is wrong: re-imposing the stored pose every tick
    lifts a seated object off its contact.
    """
    return bool(state._BOX_PACK_TASK)

def _twist_about_world_z(quat_wxyz):
    """Heading of a wxyz quaternion: its twist angle about world z.

    Unlike the heading of a body axis, this stays defined when the object
    lies on its side.
    """
    return 2.0*float(np.arctan2(quat_wxyz[3],quat_wxyz[0]))

def _hold_planar_pose(qpos,qvel,qpos_address,dof_address,stored_qpos):
    """Hold one free joint's x, y and heading at ``stored_qpos``.

    Height, roll and pitch and their rates are left to gravity and contact, so
    the object comes to rest on whatever supports it. A free joint's angular
    velocity is in the body frame; only its world-z component is removed.
    """
    a=int(qpos_address); v=int(dof_address)
    stored=np.asarray(stored_qpos,dtype=np.float64)
    quat=np.asarray(qpos[a+3:a+7],dtype=np.float64).copy()
    half=0.5*(_twist_about_world_z(stored[3:7])-_twist_about_world_z(quat))
    held=np.zeros(4)
    mujoco.mju_mulQuat(held,np.array([np.cos(half),0.0,0.0,np.sin(half)]),quat)
    held/=np.linalg.norm(held)
    rotation=np.zeros(9)
    mujoco.mju_quat2Mat(rotation,held)
    rotation=rotation.reshape(3,3)
    omega_world=rotation@np.asarray(qvel[v+3:v+6],dtype=np.float64)
    omega_world[2]=0.0
    qpos[a:a+2]=stored[:2]
    qpos[a+3:a+7]=held
    qvel[v:v+2]=0.0
    qvel[v+3:v+6]=rotation.T@omega_world

def _pin_ready_target(state):
    """Hold the task target against the moving arms for one ready-drive step."""
    if _ready_pin_is_planar(state):
        _hold_planar_pose(
            state.d.qpos,state.d.qvel,state.CUPa,state.cdof,
            state._READY_OBJECT_QPOS[state._TASK_OBJECT_TARGET][2],
        )
        return
    state.d.qpos[state.CUPa:state.CUPa+3]=state.cup; state.d.qpos[state.CUPa+3:state.CUPa+7]=state._READY_PRIMARY_QUAT; state.d.qvel[state.cdof:state.cdof+6]=0.

def _keep_ready_objects_where_they_landed(state):
    """Point the stationary restore at the objects' post-drive poses.

    Under a planar hold, gravity and contact place the objects during the
    drive: a container or tape settles into its contact.
    Restoring the pre-drive pose would lift them back off their support, or
    into the air, so the registry takes the landed pose and the restore only
    zeroes velocities.
    """
    state._READY_OBJECT_QPOS={
        name:(qadr,dadr,np.asarray(state.d.qpos[qadr:qadr+7],dtype=float).copy())
        for name,(qadr,dadr,_stored) in state._READY_OBJECT_QPOS.items()
    }

def run(state):
    """Settle task objects and drive both arms to the validated ready pose."""
    state._joint_ik_ready_gate_report = _joint_ik_ready_gate_report
    READY={"R":(state.cup+np.array([0.,0.,0.18]),np.array([0.,-0.259,0.966,0.])/np.linalg.norm([0.,-0.259,0.966,0.])),
           "L":(state.cup+np.array([-0.30,0.,0.20]),np.array([0.,0.259,0.966,0.])/np.linalg.norm([0.,0.259,0.966,0.]))}
    state.READY = READY
    if state.A.grasp_test:
        def _grasp_quat_topdown():
            q_top = np.array([0.0, 1.0, 0.0, 0.0])
            if abs(state.A.grasp_yaw) > 1e-9:
                h = np.radians(state.A.grasp_yaw) / 2.0
                q_yaw = np.array([np.cos(h), 0.0, 0.0, np.sin(h)])
                q = qmul(q_yaw, q_top)
            else:
                q = q_top
            return q / (np.linalg.norm(q) + 1e-12)
        def _grasp_quat_side(azim_deg):
            phi=np.radians(azim_deg)
            zc=np.array([np.cos(phi),np.sin(phi),0.0])   # approach root+Z horizontal
            yc=np.array([0.0,0.0,1.0])                    # swing root+Y up
            xc=np.cross(yc,zc)                            # close root+X horizontal perp
            R=np.column_stack([xc,yc,zc]); q=_qmat(R)
            return q/(np.linalg.norm(q)+1e-12), zc
        GH = state.A.grasp_hand
        state.GH = GH
        GH_OTHER = None if GH == "both" else ("L" if state.GH == "R" else "R")
        state.GH_OTHER = GH_OTHER
        _gx = float(state.cup[0]) + state.A.grasp_dx
        state._gx = _gx
        _gy = float(state.cup[1]) + state.A.grasp_dy
        state._gy = _gy
        if GH == "both":
            # One TCP at each end of the target, both driven through the same
            # phases: the two-handed lift the container task asks for, without a
            # policy.
            if state.A.approach != "topdown":
                raise SystemExit(
                    "[sim] grasp-test --grasp-hand both supports --approach topdown only"
                )
            _GQ = _grasp_quat_topdown()
            state._GQ = _GQ
            state._PB = None
            _configure_two_hand_grasp(state)
            _P_PRE = state._P_PRE
            _P_GRASP = state._P_GRASP
            _P_TOP = state._P_TOP
            _P_PARK = state._P_PARK
        elif state.A.approach=="side":
            _GQ,_AV=_grasp_quat_side(state.A.approach_azim)
            state._GQ = _GQ
            state._AV = _AV
            _P_GRASP=np.array([_gx-state.A.grasp_reach*_AV[0], _gy-state.A.grasp_reach*_AV[1], state.A.grasp_z])
            state._P_GRASP = _P_GRASP
            _P_PRE=_P_GRASP-state.A.side_standoff*_AV
            state._P_PRE = _P_PRE
            _P_TOP=_P_GRASP+np.array([0.0,0.0,state.A.lift_h])
            state._P_TOP = _P_TOP
            _P_PARK=np.array([_gx, _gy, state.A.grasp_z+state.A.lift_h+0.10])
            state._P_PARK = _P_PARK
        else:
            _GQ = _grasp_quat_topdown()
            state._GQ = _GQ
            _P_PRE = np.array([_gx, _gy, state.A.pregrasp_z])
            state._P_PRE = _P_PRE
            _P_GRASP = np.array([_gx, _gy, state.A.grasp_z])
            state._P_GRASP = _P_GRASP
            _P_TOP = np.array([_gx, _gy, state.A.grasp_z + state.A.lift_h])
            state._P_TOP = _P_TOP
            _P_PARK = np.array([_gx - 0.35, _gy, state.A.grasp_z + state.A.lift_h + 0.05])
            state._P_PARK = _P_PARK
        _PH = [("settle",state.A.ph_settle),("descend",state.A.ph_descend),("close",state.A.ph_close),
               ("lift",state.A.ph_lift),("hold",state.A.ph_hold),("open",state.A.ph_open),("drop",state.A.ph_drop)]
        state._PH = _PH
        _PB = {}
        state._PB = _PB
        _acc = 0
        state._acc = _acc
        for _name,_len in _PH:
            _PB[_name] = (_acc, _acc + int(_len)); _acc += int(_len)
            state._acc = _acc
        GRASP_N = _acc
        state.GRASP_N = GRASP_N
        state.grasp_script = GraspTestScript(
            state, _PB, _GQ, _P_PRE, _P_GRASP, _P_TOP
        )
        if GH == "both":
            READY = _configure_two_hand_grasp(state)
        else:
            state.grasp_scripts = {GH: state.grasp_script}
            READY = {state.GH: (_P_PRE.copy(), _GQ.copy()),
                     state.GH_OTHER: (_P_PARK.copy(), _GQ.copy())}
        state.READY = READY
    state.POLICY_INIT_GRIP=None
    state.ANCHOR_TABLE_ORIGIN=None
    state.POLICY_ANCHOR_EFFECTIVE=("ready" if state.A.policy_anchor=="ready" else None)
    state.POLICY_ANCHOR_EPISODE=None
    if state.POLICY_SOURCE and state.A.policy_anchor=="deploy":
        _deploy_ready=load_policy_deploy_ready(state)
        state._deploy_ready = _deploy_ready
        if _deploy_ready is not None: READY=_deploy_ready; state.READY = READY
    if state.POLICY_ANCHOR_EFFECTIVE is None:
        state.POLICY_ANCHOR_EFFECTIVE="not_applicable"
    _DYNAMIC_READY_DRIVEN=False
    state._DYNAMIC_READY_DRIVEN = _DYNAMIC_READY_DRIVEN
    _DYNAMIC_READY_STEPS=None
    state._DYNAMIC_READY_STEPS = _DYNAMIC_READY_STEPS
    _READY_MOVE_STEPS=None
    state._READY_MOVE_STEPS = _READY_MOVE_STEPS
    _READY_HOLD_STEPS=None
    state._READY_HOLD_STEPS = _READY_HOLD_STEPS
    _READY_GATE_REPORT=None
    state._READY_GATE_REPORT = _READY_GATE_REPORT
    if state.A.ik_replay or state.A.state_log_replay:
        mujoco.mj_forward(state.m,state.d)   # stored joints are set per-frame in the loop; no ready drive is needed
    elif state.A.physics=="kinematic":
        for k in ("R","L"): ik_solve(state,k,*state.READY[k])
        mujoco.mj_forward(state.m,state.d)
    else:
        _DYNAMIC_READY_DRIVEN=True
        state._DYNAMIC_READY_DRIVEN = _DYNAMIC_READY_DRIVEN
        # The historical ready drive commands both hands fully open. A
        # schema-2 initial-arm-joint file that declares a gripper angle
        # replaces only that command, exactly like it replaces only the arm
        # ready goals; a schema-1 file leaves the vector byte-identical.
        _READY_GRIP_CMD=(
            dict(state._INITIAL_GRIP_Q)
            if state._INITIAL_GRIP_Q is not None
            else {"R":0.7854,"L":0.7854}
        )
        state._READY_GRIP_CMD = _READY_GRIP_CMD
        if state.DYNAMIC_YUBI_HOOK is not None:
            state.DYNAMIC_YUBI_HOOK.set_targets(dict(_READY_GRIP_CMD),t=float(state.d.time))
        if state.EMBODIMENT=="fr3":
            if float(state.A.dynamic_ready_sec) <= 0:
                raise SystemExit("[sim] --dynamic-ready-sec must be > 0")
            _DYNAMIC_READY_STEPS=max(1,int(round(float(state.A.dynamic_ready_sec)*float(state.env.control_freq))))
            state._DYNAMIC_READY_STEPS = _DYNAMIC_READY_STEPS
            print(
                f"[sim] dynamic ready drive: {state._DYNAMIC_READY_STEPS} controller steps "
                f"({float(state._DYNAMIC_READY_STEPS)/float(state.env.control_freq):.3f}s)",
                flush=True,
            )
            _READY_JOINT_IK={}
            state._READY_JOINT_IK = _READY_JOINT_IK
            _READY_JOINT_SEGMENTS={}
            state._READY_JOINT_SEGMENTS = _READY_JOINT_SEGMENTS
            if state.FR3_STRATEGY.supports_joint_ready:
                _READY_HOLD_STEPS=min(
                    state._DYNAMIC_READY_STEPS-1,
                    int(round(
                        min(2.0,float(state.A.dynamic_ready_sec)*0.25)
                        *float(state.env.control_freq)
                    )),
                )
                state._READY_HOLD_STEPS = _READY_HOLD_STEPS
                _READY_MOVE_STEPS=state._DYNAMIC_READY_STEPS-state._READY_HOLD_STEPS
                state._READY_MOVE_STEPS = _READY_MOVE_STEPS
                _ready_move_duration=float(state._READY_MOVE_STEPS)/float(state.env.control_freq)
                state._ready_move_duration = _ready_move_duration
                print(
                    f"[sim] joint_ik ready trajectory: "
                    f"{_ready_move_duration:.3f}s move + "
                    f"{float(state._READY_HOLD_STEPS)/float(state.env.control_freq):.3f}s hold",
                    flush=True,
                )
                for key in ("R","L"):
                    _ready_seed=np.asarray(state.d.qpos[state.ARM[key]["qi"]],dtype=np.float64).copy()
                    state._ready_seed = _ready_seed
                    if state._INITIAL_ARM_JOINT_Q is not None:
                        _ready_goal=state._INITIAL_ARM_JOINT_Q[key].copy()
                        state._ready_goal = _ready_goal
                        _ready_result=None
                        state._ready_result = _ready_result
                    else:
                        _ready_result=state._ABSOLUTE_JOINT_IK.solve(
                            key,*state.READY[key],seed_q=_ready_seed,continuity_q=_ready_seed
                        )
                        state._ready_result = _ready_result
                        _ready_goal=_ready_result.q
                        state._ready_goal = _ready_goal
                    state._READY_JOINT_IK[key]=_ready_result
                    state._READY_JOINT_SEGMENTS[key]=bounded_minimum_jerk_segment(
                        _ready_seed,
                        _ready_goal,
                        steps=state._READY_MOVE_STEPS,
                        duration_s=_ready_move_duration,
                        max_velocity_rad_s=float(state.A.joint_ik_max_velocity),
                        max_acceleration_rad_s2=float(state.A.joint_ik_max_acceleration),
                    )
                    if _ready_result is None:
                        print(
                            f"[sim] joint_ik ready JSON goal {key}: "
                            f"q={np.round(_ready_goal,6).tolist()} "
                            f"segment_scale={state._READY_JOINT_SEGMENTS[key].scale:.4f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[sim] joint_ik ready solve {key}: converged={_ready_result.converged} "
                            f"iters={_ready_result.iterations} pos={_ready_result.pos_error_mm:.3f}mm "
                            f"ori={_ready_result.ori_error_deg:.3f}deg "
                            f"limit_margin={_ready_result.min_limit_margin_rad:.4f}rad "
                            f"segment_scale={state._READY_JOINT_SEGMENTS[key].scale:.4f}",
                            flush=True,
                        )
            for _ready_step in range(state._DYNAMIC_READY_STEPS):
                if state.FR3_STRATEGY.supports_joint_ready:
                    _ready_index=min(_ready_step,state._READY_MOVE_STEPS-1)
                    state._ready_index = _ready_index
                    act=_dynamic_env_action(state,
                        {
                            key:state._READY_JOINT_SEGMENTS[key].commands[_ready_index]
                            for key in ("R","L")
                        },
                        dict(state._READY_GRIP_CMD),
                    )
                _pin_ready_target(state)
                state.env.step(act)
            if state.FR3_STRATEGY.supports_joint_ready:
                _READY_GATE_REPORT=_joint_ik_ready_gate_report(state)
                state._READY_GATE_REPORT = _READY_GATE_REPORT
                if state._INITIAL_ARM_JOINT_Q is not None:
                    for key,_gate_state in state._READY_GATE_REPORT["arms"].items():
                        print(
                            f"[sim] specified joint ready gate {key}: "
                            f"max_error={_gate_state['max_error_deg']:.6f}deg "
                            f"at j{_gate_state['max_error_joint']}",
                            flush=True,
                        )
                else:
                    for key,_gate_state in state._READY_GATE_REPORT.items():
                        print(
                            f"[sim] joint_ik ready gate {key}: "
                            f"{_gate_state['position_error_mm']:.3f}mm/"
                            f"{_gate_state['orientation_error_deg']:.3f}deg",
                            flush=True,
                        )
    if state.A.grasp_test:
        _force_finger_state(state,state.A.open_grip, state.A.open_grip)
    elif state.POLICY_SOURCE and state.POLICY_INIT_GRIP is not None:
        _force_finger_state(state,state.POLICY_INIT_GRIP["L"],state.POLICY_INIT_GRIP["R"])
    elif state._INITIAL_GRIP_Q is not None:
        # Snap both hands onto the declared raw motor angle so the settle and
        # the first policy observation start from it exactly, instead of from
        # wherever the ready-drive controller happened to converge.
        _force_finger_state(state,state._INITIAL_GRIP_Q["L"],state._INITIAL_GRIP_Q["R"])
        print(
            "[sim] initial gripper applied (raw motor rad): "
            f"L={state._INITIAL_GRIP_Q['L']:.6f} R={state._INITIAL_GRIP_Q['R']:.6f}",
            flush=True,
        )
    else:
        _reset_yubi_from_current(state)
    # The dynamic ready-drive is an arm-only setup operation. Its duration-scaled
    # env.step() calls must not become part of the task's initial cup / plate state. Explicit
    # saved states and MimicGen provide their own authoritative object states and
    # are intentionally left untouched.
    OBJECT_SETTLE_REPORT=None
    state.OBJECT_SETTLE_REPORT = OBJECT_SETTLE_REPORT
    if state._DYNAMIC_READY_DRIVEN and not state.A.init_state_in and not state.A.mimicgen_init_episode:
        if _ready_pin_is_planar(state):
            _keep_ready_objects_where_they_landed(state)
        _restore_ready_objects_stationary(state)
        if float(state.A.dynamic_object_settle_max_sec)>0.0:
            OBJECT_SETTLE_REPORT=_settle_ready_objects(state)
            state.OBJECT_SETTLE_REPORT = OBJECT_SETTLE_REPORT
    _LOADED_INIT_STATE=None
    state._LOADED_INIT_STATE = _LOADED_INIT_STATE
    if state.A.init_state_in:
        _LOADED_INIT_STATE=_load_sim_init_state(state,state.A.init_state_in)
        state._LOADED_INIT_STATE = _LOADED_INIT_STATE
        if (
            float(state.A.dynamic_object_settle_max_sec)>0.0
            and not state.A.mimicgen_init_episode
        ):
            try:
                validate_saved_settled_init_payload(
                    state._LOADED_INIT_STATE,_settled_init_state_contract(state)
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"init-state settled provenance gate failed: {exc}"
                ) from exc
            _loaded_validation=_validate_settled_objects(state,"init_state")
            state._loaded_validation = _loaded_validation
            _validate_loaded_object_state_matches_payload(
                state,state._LOADED_INIT_STATE,_loaded_validation["objects"]
            )
            OBJECT_SETTLE_REPORT={
                **_loaded_validation,
                "saved_report":state._LOADED_INIT_STATE.get("object_settle"),
            }
            state.OBJECT_SETTLE_REPORT = OBJECT_SETTLE_REPORT
    MIMICGEN_INIT_REPORT=None
    state.MIMICGEN_INIT_REPORT = MIMICGEN_INIT_REPORT
    if state.A.mimicgen_init_episode:
        if not state.A.mimicgen_init_state_log_root:
            raise SystemExit("[sim] --mimicgen-init-episode requires --mimicgen-init-state-log-root")
        MIMICGEN_INIT_REPORT=_apply_mimicgen_training_init(state,
            state.A.mimicgen_init_episode,state.A.mimicgen_init_state_log_root
        )
        state.MIMICGEN_INIT_REPORT = MIMICGEN_INIT_REPORT
    elif state.A.mimicgen_init_state_log_root:
        raise SystemExit("[sim] --mimicgen-init-state-log-root requires --mimicgen-init-episode")
    state.cup=np.asarray(state.d.qpos[state.CUPa:state.CUPa+3],dtype=np.float64).copy()
    if state.A.grasp_test and state.GH == "both":
        # Settling and --init-state-in may have moved the target since the
        # poses above were placed around it, and every scripted pose here is
        # measured from the target, so place them again on the settled pose.
        _configure_two_hand_grasp(state)
    if state.A.init_state_out:
        _write_sim_init_state(state,
            state.A.init_state_out,
            "post-ready",
            require_settled_contract=bool(
                float(state.A.dynamic_object_settle_max_sec)>0.0
            ),
        )
        if state.A.init_state_only:
            raise SystemExit(0)
    elif state.A.init_state_only:
        raise SystemExit("[sim] --init-state-only requires --init-state-out")
    if state.DYNAMIC_YUBI_HOOK is not None:
        state.DYNAMIC_YUBI_HOOK.reset_timing(clear_samples=True,reset_counts=True)

    _JOINT_IK_COMMAND={}
    state._JOINT_IK_COMMAND = _JOINT_IK_COMMAND
    _JOINT_IK_COMMAND_VELOCITY={}
    state._JOINT_IK_COMMAND_VELOCITY = _JOINT_IK_COMMAND_VELOCITY
    _JOINT_IK_COMMAND_ACCELERATION={}
    state._JOINT_IK_COMMAND_ACCELERATION = _JOINT_IK_COMMAND_ACCELERATION
    _JOINT_IK_TRAJECTORY={}
    state._JOINT_IK_TRAJECTORY = _JOINT_IK_TRAJECTORY
    _JOINT_IK_LAST_RESULT={}
    state._JOINT_IK_LAST_RESULT = _JOINT_IK_LAST_RESULT
    _JOINT_IK_LAST_TARGET_UPDATE={}
    state._JOINT_IK_LAST_TARGET_UPDATE = _JOINT_IK_LAST_TARGET_UPDATE
    _JOINT_IK_LAST_SAMPLE={}
    state._JOINT_IK_LAST_SAMPLE = _JOINT_IK_LAST_SAMPLE
    _JOINT_IK_LAST_SAFETY_DECISION={}
    state._JOINT_IK_LAST_SAFETY_DECISION = _JOINT_IK_LAST_SAFETY_DECISION
    state._JOINT_IK_SOLVE_COUNT=0
    state._JOINT_IK_FAIL_COUNT=0
    state._JOINT_IK_TARGET_REJECT_COUNT=0
    _JOINT_IK_TARGET_ATTEMPT_COUNT={"R":0,"L":0}
    state._JOINT_IK_TARGET_ATTEMPT_COUNT = _JOINT_IK_TARGET_ATTEMPT_COUNT
    _JOINT_IK_TARGET_REJECT_REASONS={"R":{},"L":{}}
    state._JOINT_IK_TARGET_REJECT_REASONS = _JOINT_IK_TARGET_REJECT_REASONS
    _JOINT_IK_SAFETY_STATS={
        _key:{
            "full_target":0,
            "ik_nonconverged_hold":0,
            "collision_endpoint":0,
            "collision_clipped":0,
            "collision_hold":0,
            "trajectory_preview_collision":0,
            "trajectory_collision_brake":0,
            "trajectory_collision_hold":0,
            "ruckig_reject":0,
            "scratch_collision_checks":0,
            "trajectory_preview_checks":0,
            "ignored_zero_distance_pairs":0,
            "initial_min_clearance_mm":float("inf"),
            "raw_min_clearance_mm":float("inf"),
            "selected_min_clearance_mm":float("inf"),
            "trajectory_preview_min_clearance_mm":float("inf"),
        }
        for _key in ("R","L")
    }
    state._JOINT_IK_SAFETY_STATS = _JOINT_IK_SAFETY_STATS
    _JOINT_IK_INITIAL_SAFETY={}
    state._JOINT_IK_INITIAL_SAFETY = _JOINT_IK_INITIAL_SAFETY


__all__ = ["run"]
