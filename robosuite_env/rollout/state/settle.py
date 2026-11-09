# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Settled-object state and readiness contracts."""

from __future__ import annotations

import mujoco
import numpy as np

from robosuite_env.rollout.state.object_settle import (
    ConsecutiveQuietWindow,
    FreeJointMotion,
    QuietThresholds,
    resolve_transitive_support,
)
from robosuite_env.rollout.control.rule_grasp import (
    _assert_rule_grasp_pristine,
    _finger_actual,
)
from robosuite_env.rollout.control.ik import (
    _dynamic_env_action as dynamic_env_action,
)
from robosuite_env.rollout.state.cache import SETTLED_INIT_STATE_SCHEMA
from robosuite_env.rollout.state.fingerprint import physical_model_fingerprint
from robosuite_env.rollout.setup.frames import _quat_distance_deg


def initialize_ready_object_registry(state) -> None:
    """Capture task-object poses and static support geometry before ready drive."""

    state.cup = state.d.qpos[state.CUPa : state.CUPa + 3].copy()
    state._READY_OBJECT_QPOS = {
        name: (
            int(binding["qpos_address"]),
            int(binding["dof_address"]),
            np.asarray(
                state.d.qpos[
                    int(binding["qpos_address"]) : int(binding["qpos_address"])
                    + 7
                ],
                float,
            ).copy(),
        )
        for name, binding in state._TASK_OBJECT_BINDINGS.items()
    }
    state._READY_PRIMARY_QUAT = (
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    )
    state._OBJECT_ROOT_BODY_IDS = {
        name: int(state.m.jnt_bodyid[int(binding["joint_id"])])
        for name, binding in state._TASK_OBJECT_BINDINGS.items()
    }
    state._SUPPORT_GEOM_IDS = {
        int(geom_id)
        for geom_id in range(state.m.ngeom)
        if (
            (
                mujoco.mj_id2name(
                    state.m, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )
                or ""
            )
            == "table_collision"
            or (
                mujoco.mj_id2name(
                    state.m, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )
                or ""
            ).startswith("gw_desk_coll_")
        )
    }


def initialize_settle_contact_registry(state) -> None:
    """Populate settle thresholds and the geom-to-object ownership map."""

    state._OBJECT_SETTLE_SUPPORT_FORCE_N = 1e-4
    state._OBJECT_SETTLE_SUPPORT_NORMAL_Z = 0.5
    state._OBJECT_GEOM_IDS = {
        name: {
            int(geom_id)
            for geom_id in range(state.m.ngeom)
            if _body_in_subtree(
                state, int(state.m.geom_bodyid[geom_id]), root_id
            )
        }
        for name, root_id in state._OBJECT_ROOT_BODY_IDS.items()
    }
    state._OBJECT_GEOM_OWNER = {
        geom_id: name
        for name, geom_ids in state._OBJECT_GEOM_IDS.items()
        for geom_id in geom_ids
    }

def _object_free_joint_report(state):
    """Return the exact free-joint pose / velocity saved for task objects."""
    out={}
    for name,binding in state._TASK_OBJECT_BINDINGS.items():
        qadr=int(binding["qpos_address"])
        dadr=int(binding["dof_address"])
        qvel=np.asarray(state.d.qvel[dadr:dadr+6],float).copy()
        qacc=np.asarray(state.d.qacc[dadr:dadr+6],float).copy()
        motion=FreeJointMotion.from_vectors(qvel,qacc)
        out[name]={
            "qpos":[float(x) for x in np.asarray(state.d.qpos[qadr:qadr+7],float)],
            "qvel":[float(x) for x in qvel],
            "qacc":[float(x) for x in qacc],
            "linear_speed_m_s":motion.linear_speed_m_s,
            "angular_speed_rad_s":motion.angular_speed_rad_s,
            "linear_acceleration_m_s2":motion.linear_acceleration_m_s2,
            "angular_acceleration_rad_s2":motion.angular_acceleration_rad_s2,
            "stationary":bool(
                motion.linear_speed_m_s<=1e-12
                and motion.angular_speed_rad_s<=1e-12
            ),
        }
    return out

def _restore_ready_objects_stationary(state):
    """Undo object motion caused solely by the dynamic arm ready-drive."""
    for _name,(_qadr,_dadr,_qpos) in state._READY_OBJECT_QPOS.items():
        state.d.qpos[_qadr:_qadr+7]=_qpos
        state.d.qvel[_dadr:_dadr+6]=0.0
        state.d.qacc[_dadr:_dadr+6]=0.0
        state.d.qacc_warmstart[_dadr:_dadr+6]=0.0
    mujoco.mj_forward(state.m,state.d)
    for _name,(_qadr,_dadr,_qpos) in state._READY_OBJECT_QPOS.items():
        # Do not persist a warm-start acceleration from the discarded ready-drive.
        state.d.qacc_warmstart[_dadr:_dadr+6]=0.0
    report=_object_free_joint_report(state)
    moving=[name for name,state in report.items() if not state["stationary"]]
    if moving:
        raise RuntimeError(f"ready object reset left non-stationary free joints: {moving}")
    poses=", ".join(
        f"{name}={np.round(state.d.qpos[qadr:qadr+3],4).tolist()}"
        for name,(qadr,_dadr,_qpos) in state._READY_OBJECT_QPOS.items()
    )
    print(f"[sim] dynamic ready object reset: {poses}; free-joint velocities stationary",flush=True)
    return report

def _body_in_subtree(state,body_id,root_id):
    body_id=int(body_id); root_id=int(root_id)
    while body_id>0:
        if body_id==root_id:
            return True
        body_id=int(state.m.body_parentid[body_id])
    return body_id==root_id

def _object_support_contact_report(state):
    """Return each object's load-bearing path to the static desk.

    Task objects can legitimately finish stacked.  An object is therefore
    supported when it either has a vertical load-bearing static-desk contact
    itself or belongs to a vertical object-contact component that reaches such
    an object.  A quiet but floating component has no desk root and is
    rejected.
    """
    direct={name:[] for name in state._READY_OBJECT_QPOS}
    object_contacts={name:[] for name in state._READY_OBJECT_QPOS}
    adjacency={name:set() for name in state._READY_OBJECT_QPOS}
    force6=np.zeros(6,dtype=np.float64)
    for contact_index in range(int(state.d.ncon)):
        contact=state.d.contact[contact_index]
        geom1=int(contact.geom1); geom2=int(contact.geom2)
        owner1=state._OBJECT_GEOM_OWNER.get(geom1)
        owner2=state._OBJECT_GEOM_OWNER.get(geom2)
        static1=geom1 in state._SUPPORT_GEOM_IDS
        static2=geom2 in state._SUPPORT_GEOM_IDS
        if not (
            (owner1 is not None and (static2 or owner2 is not None))
            or (owner2 is not None and static1)
        ):
            continue
        normal=np.asarray(contact.frame[:3],dtype=np.float64)
        mujoco.mj_contactForce(state.m,state.d,contact_index,force6)
        normal_force=float(force6[0])
        vertical=abs(float(normal[2]))
        if (
            vertical<state._OBJECT_SETTLE_SUPPORT_NORMAL_Z
            or normal_force<state._OBJECT_SETTLE_SUPPORT_FORCE_N
        ):
            continue
        common={
            "contact_index":int(contact_index),
            "normal_force_n":normal_force,
            "normal_z_abs":vertical,
        }
        if owner1 is not None and static2:
            direct[owner1].append({
                **common,
                "support_geom":(
                    mujoco.mj_id2name(
                        state.m,mujoco.mjtObj.mjOBJ_GEOM,geom2
                    ) or f"geom{geom2}"
                ),
            })
        elif owner2 is not None and static1:
            direct[owner2].append({
                **common,
                "support_geom":(
                    mujoco.mj_id2name(
                        state.m,mujoco.mjtObj.mjOBJ_GEOM,geom1
                    ) or f"geom{geom1}"
                ),
            })
        elif owner1 is not None and owner2 is not None and owner1!=owner2:
            adjacency[owner1].add(owner2)
            adjacency[owner2].add(owner1)
            object_contacts[owner1].append({**common,"other_object":owner2})
            object_contacts[owner2].append({**common,"other_object":owner1})

    resolved=resolve_transitive_support(
        state._TASK_OBJECT_NAMES,
        [name for name,contacts in direct.items() if contacts],
        adjacency,
    )
    report={}
    for name in state._TASK_OBJECT_NAMES:
        report[name]={
            "supported":bool(resolved[name]["supported"]),
            "direct_supported":bool(direct[name]),
            "support_path":list(resolved[name]["support_path"]),
            "contacts":direct[name],
            "object_contacts":object_contacts[name],
        }
    return report

def _object_settle_thresholds(state):
    return QuietThresholds(
        linear_speed_m_s=float(state.A.dynamic_object_settle_linear_speed_mm_s)/1000.0,
        angular_speed_rad_s=float(state.A.dynamic_object_settle_angular_speed_rad_s),
        linear_acceleration_m_s2=float(
            state.A.dynamic_object_settle_linear_acceleration_m_s2
        ),
        angular_acceleration_rad_s2=float(
            state.A.dynamic_object_settle_angular_acceleration_rad_s2
        ),
    )

def _object_settle_threshold_report(state, thresholds=None):
    thresholds=thresholds or _object_settle_thresholds(state)
    return {
        "linear_speed_m_s":thresholds.linear_speed_m_s,
        "angular_speed_rad_s":thresholds.angular_speed_rad_s,
        "linear_acceleration_m_s2":thresholds.linear_acceleration_m_s2,
        "angular_acceleration_rad_s2":thresholds.angular_acceleration_rad_s2,
        "instantaneous_acceleration_gate_enabled":False,
        "instantaneous_acceleration_metrics":"telemetry_only",
        "position_drift_mm":float(
            state.A.dynamic_object_settle_position_drift_mm
        ),
        "orientation_drift_deg":float(
            state.A.dynamic_object_settle_orientation_drift_deg
        ),
        "support_force_n":state._OBJECT_SETTLE_SUPPORT_FORCE_N,
        "support_normal_z_abs":state._OBJECT_SETTLE_SUPPORT_NORMAL_Z,
    }

def _settled_init_state_contract(state):
    return {
        "schema":SETTLED_INIT_STATE_SCHEMA,
        "expected_objects":list(state._TASK_OBJECT_NAMES),
        "model_physics_sha256":physical_model_fingerprint(state),
        "controller":"joint_ik",
        "control_hz":float(state.env.control_freq),
        "mujoco_timestep_s":float(state.m.opt.timestep),
        "min_duration_s":float(state.A.dynamic_object_settle_min_sec),
        "stable_duration_s":float(state.A.dynamic_object_settle_stable_sec),
        "stable_samples_required":max(
            1,
            int(np.ceil(
                float(state.A.dynamic_object_settle_stable_sec)
                * float(state.env.control_freq)
            )),
        ),
        "thresholds":_object_settle_threshold_report(state),
        "ready_position_tolerance_mm":float(
            state.A.joint_ik_ready_position_tol_mm
        ),
        "ready_orientation_tolerance_deg":float(
            state.A.joint_ik_ready_orientation_tol_deg
        ),
        "initial_arm_joints_json":(
            state._INITIAL_ARM_JOINT_POSE.source_path
            if state._INITIAL_ARM_JOINT_POSE is not None else None
        ),
        "initial_arm_joints_sha256":(
            state._INITIAL_ARM_JOINT_POSE.source_sha256
            if state._INITIAL_ARM_JOINT_POSE is not None else None
        ),
        "initial_arm_joint_targets_rad":(
            {
                key:[float(value) for value in state._INITIAL_ARM_JOINT_Q[key]]
                for key in ("R","L")
            }
            if state._INITIAL_ARM_JOINT_Q is not None else None
        ),
        "initial_arm_joints_tolerance_deg":float(
            state.A.initial_arm_joints_tol_deg
        ),
    }

def _object_motion_from_report(report):
    return {
        name:FreeJointMotion(
            linear_speed_m_s=float(state["linear_speed_m_s"]),
            angular_speed_rad_s=float(state["angular_speed_rad_s"]),
            linear_acceleration_m_s2=float(
                state["linear_acceleration_m_s2"]
            ),
            angular_acceleration_rad_s2=float(
                state["angular_acceleration_rad_s2"]
            ),
        )
        for name,state in report.items()
    }

def _object_pose_window_ok(state, anchor,current):
    position_limit=float(state.A.dynamic_object_settle_position_drift_mm)
    orientation_limit=float(state.A.dynamic_object_settle_orientation_drift_deg)
    for name,state in current.items():
        if name not in anchor:
            return False
        q0=np.asarray(anchor[name]["qpos"],dtype=np.float64)
        q1=np.asarray(state["qpos"],dtype=np.float64)
        if float(np.linalg.norm(q1[:3]-q0[:3])*1000.0)>position_limit:
            return False
        if _quat_distance_deg(q1[3:7],q0[3:7])>orientation_limit:
            return False
    return True

def _validate_settled_objects(state,source):
    """Read-only gate used after loading a cached post-settle state."""
    _assert_rule_grasp_pristine(state,source)
    object_report=_object_free_joint_report(state)
    expected=set(state._TASK_OBJECT_NAMES)
    if set(object_report)!=expected:
        raise RuntimeError(
            f"object settle requires exactly {sorted(expected)}, got "
            f"{sorted(object_report)}"
        )
    motions=_object_motion_from_report(object_report)
    thresholds=_object_settle_thresholds(state)
    support=_object_support_contact_report(state)
    failures=[]
    for name,motion in motions.items():
        if not thresholds.accepts(
            motion,
            include_acceleration=False,
        ):
            failures.append(
                f"{name}:v={motion.linear_speed_m_s:.3e}m/s,"
                f"w={motion.angular_speed_rad_s:.3e}rad/s,"
                f"a={motion.linear_acceleration_m_s2:.3e}m/s2,"
                f"alpha={motion.angular_acceleration_rad_s2:.3e}rad/s2"
            )
        if not support.get(name,{}).get("supported",False):
            failures.append(f"{name}:no vertical load-bearing desk contact")
    if failures:
        raise RuntimeError(
            f"object settle read-only gate failed after {source}: "
            + "; ".join(failures)
        )
    ready_gate=_settled_ready_gate(state)
    return {
        "source":str(source),
        "validated":True,
        "objects":object_report,
        "support":support,
        "ready_gate":ready_gate,
    }

def _settled_ready_gate(state):
    """Ready-pose gate re-checked alongside a settled object state.

    FR3 drives both arms to a validated ready pose before the objects settle,
    and this re-checks that pose.
    """
    return (
        state._joint_ik_ready_gate_report(state)
        if state.FR3_STRATEGY.supports_joint_ready else None
    )

def _object_settle_hold_action(state,grip_hold):
    """Action that holds both arms still while the task objects settle.

    FR3 holds the last command of its validated ready trajectory, one action
    vector per FR3.
    """
    arm_hold={
        key:np.asarray(
            state._READY_JOINT_SEGMENTS[key].commands[-1],dtype=np.float64
        )
        for key in ("R","L")
    }
    return dynamic_env_action(state,arm_hold,grip_hold)

def _settle_ready_objects(state):
    """Settle restored task objects while holding the arm / hand pose."""
    _assert_rule_grasp_pristine(state,"object_settle_start")
    if set(state._READY_OBJECT_QPOS)!=set(state._TASK_OBJECT_NAMES):
        raise RuntimeError(
            "dynamic object settle registry mismatch; "
            f"expected={list(state._TASK_OBJECT_NAMES)} "
            f"got={sorted(state._READY_OBJECT_QPOS)}"
        )
    control_hz=float(state.env.control_freq)
    min_steps=max(
        0,int(np.ceil(float(state.A.dynamic_object_settle_min_sec)*control_hz))
    )
    max_steps=max(
        1,int(np.ceil(float(state.A.dynamic_object_settle_max_sec)*control_hz))
    )
    stable_steps=max(
        1,int(np.ceil(float(state.A.dynamic_object_settle_stable_sec)*control_hz))
    )
    thresholds=_object_settle_thresholds(state)
    gate=ConsecutiveQuietWindow(
        thresholds,
        stable_steps,
        include_acceleration=False,
    )
    grip_hold={"R":_finger_actual(state,"R"),"L":_finger_actual(state,"L")}
    if state.DYNAMIC_YUBI_HOOK is not None:
        state.DYNAMIC_YUBI_HOOK.set_targets(grip_hold,t=float(state.d.time))
    action=_object_settle_hold_action(state,grip_hold)
    raw_report=_object_free_joint_report(state)
    quiet_anchor=None
    final_report=raw_report
    final_support=_object_support_contact_report(state)
    passed=False
    for settle_step in range(max_steps):
        state.env.step(action)
        final_report=_object_free_joint_report(state)
        motions=_object_motion_from_report(final_report)
        final_support=_object_support_contact_report(state)
        motion_quiet=all(
            thresholds.accepts(value,include_acceleration=False)
            for value in motions.values()
        )
        support_quiet=all(
            final_support.get(name,{}).get("supported",False)
            for name in motions
        )
        if motion_quiet and support_quiet:
            if quiet_anchor is None:
                quiet_anchor={
                    name:{"qpos":list(state["qpos"])}
                    for name,state in final_report.items()
                }
            drift_quiet=_object_pose_window_ok(state, quiet_anchor,final_report)
        else:
            quiet_anchor=None
            drift_quiet=False
        window_passed=gate.observe(
            motions,
            extra_quiet=bool(support_quiet and drift_quiet),
        )
        if gate.consecutive_samples==0:
            quiet_anchor=None
        if window_passed and settle_step+1>=min_steps:
            passed=True
            used_steps=settle_step+1
            break
    else:
        used_steps=max_steps
    if not passed:
        detail=", ".join(
            f"{name}:v={state['linear_speed_m_s']:.3e},"
            f"w={state['angular_speed_rad_s']:.3e},"
            f"a={state['linear_acceleration_m_s2']:.3e},"
            f"alpha={state['angular_acceleration_rad_s2']:.3e},"
            f"support={final_support.get(name,{}).get('supported',False)}"
            for name,state in final_report.items()
        )
        raise RuntimeError(
            "dynamic object settle timed out after "
            f"{float(used_steps)/control_hz:.3f}s: {detail}"
        )
    _assert_rule_grasp_pristine(state,"object_settle_end")
    quiet_final_report=final_report
    # Publish a genuinely static initial condition.  The quiet gate accepts a
    # small solver velocity, but carrying that residual into every rollout
    # would make frame zero depend on the first integration step.  Positions
    # and orientations stay untouched; only free-joint motion history is
    # canonicalized before the exact integration state is serialized.
    for binding in state._TASK_OBJECT_BINDINGS.values():
        dadr=int(binding["dof_address"])
        state.d.qvel[dadr:dadr+6]=0.0
        state.d.qacc_warmstart[dadr:dadr+6]=0.0
    # Canonicalize derived contact / acceleration fields before reporting.
    # This is the exact state that the init-state serializer will capture.
    mujoco.mj_forward(state.m,state.d)
    canonical=_validate_settled_objects(state,"physics_settle_final")
    final_report=canonical["objects"]
    final_support=canonical["support"]
    ready_gate=canonical["ready_gate"]
    state.cup=np.asarray(state.d.qpos[state.CUPa:state.CUPa+3],dtype=np.float64).copy()
    report={
        "source":"physics_settle",
        "validated":True,
        "steps":int(used_steps),
        "duration_s":float(used_steps)/control_hz,
        "min_duration_s":float(state.A.dynamic_object_settle_min_sec),
        "max_duration_s":float(state.A.dynamic_object_settle_max_sec),
        "stable_duration_s":float(state.A.dynamic_object_settle_stable_sec),
        "stable_samples":int(gate.consecutive_samples),
        "thresholds":_object_settle_threshold_report(state, thresholds),
        "pre_settle_objects":raw_report,
        "raw_objects":quiet_final_report,
        "objects":final_report,
        "support":final_support,
        "ready_gate":ready_gate,
        "rule_grasp_unchanged":True,
    }
    poses=", ".join(
        f"{name}={np.round(np.asarray(state['qpos'][:3]),5).tolist()}"
        for name,state in final_report.items()
    )
    print(
        f"[sim] dynamic object settle PASS: {used_steps} steps "
        f"({float(used_steps)/control_hz:.3f}s) {poses}",
        flush=True,
    )
    return report

__all__ = [
    "_body_in_subtree",
    "_object_free_joint_report",
    "_object_motion_from_report",
    "_object_pose_window_ok",
    "_object_settle_hold_action",
    "_object_settle_threshold_report",
    "_object_settle_thresholds",
    "_object_support_contact_report",
    "_quat_distance_deg",
    "_restore_ready_objects_stationary",
    "_settle_ready_objects",
    "_settled_init_state_contract",
    "_settled_ready_gate",
    "_validate_settled_objects",
    "initialize_ready_object_registry",
    "initialize_settle_contact_registry",
]
