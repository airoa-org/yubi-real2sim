# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Exact init-state serialization, restore, and provenance gates."""

from __future__ import annotations

import json
import os

import mujoco
import numpy as np

from robosuite_env.robots.profiles import normalize_robot_name
from robosuite_env.rollout.control.pose_math import policy_tcp_pose
from robosuite_env.rollout.control.rule_grasp import _finger_actual as finger_actual
from robosuite_env.rollout.state.cache import (
    atomic_write_json,
    validate_saved_settled_init_payload,
)
from robosuite_env.rollout.state.init_identity import (
    INIT_IDENTITY_KEY,
    InitIdentityError,
    init_state_identity,
    validate_init_state_identity,
)
from robosuite_env.rollout.state.mujoco import (
    INTEGRATION_STATE_KEY,
    capture_integration_state,
    forward_preserving_warmstart,
    has_integration_state,
    restore_integration_or_legacy_state,
)
from robosuite_env.rollout.state.settle import (
    _object_free_joint_report as object_free_joint_report,
    _settled_init_state_contract as settled_init_state_contract,
    _validate_settled_objects as validate_settled_objects,
)
from robosuite_env.rollout.build.gripper import reset_yubi_from_current

def _validate_calibration_provenance(state, payload, source):
    if not state.GRIPPER_CALIBRATION_REPORT.get("enabled"):
        return
    state_report=payload.get("gripper_calibration")
    if not isinstance(state_report, dict):
        raise SystemExit(
            f"[sim] {source} has no calibrated gripper provenance"
        )
    mismatches=[
        key for key in state._CALIBRATION_PROVENANCE_KEYS
        if state_report.get(key)!=state.GRIPPER_CALIBRATION_REPORT.get(key)
    ]
    if mismatches:
        raise SystemExit(
            f"[sim] {source} gripper calibration provenance mismatch: "
            + ", ".join(mismatches)
        )

def _current_init_identity(state):
    """Resolve this run's scene inputs and where its environment seated objects."""
    try:
        return init_state_identity(
            state.A,
            robot=state.ROBOT,
            seating_report=getattr(state.env,"layout_seating_report",None),
        )
    except InitIdentityError as exc:
        raise SystemExit(
            f"[sim] init-state scene identity is unavailable: {exc}"
        ) from exc

def _sim_state_payload(state,label,include_settled_contract=False):
    mujoco.mj_forward(state.m,state.d)
    def arr(x):
        return [float(v) for v in np.asarray(x, float).ravel()]
    rule_grasp=None
    if state.RULE_GRASP is not None:
        rule_grasp={
            "active":bool(state.RULE_GRASP.get("active",False)),
            "hand":state.RULE_GRASP.get("hand"),
            "hold_mode":"weld_equality" if state.A.grasp_mode=="rule" else None,
            "weld_eq_id":state.RULE_GRASP.get("weld_eq_id"),
            "events":state.RULE_GRASP.get("events",[]),
        }
    payload={
        "version":5,
        "label":str(label),
        "robot":str(state.ROBOT),
        "model_signature":state.MODEL_SIGNATURE,
        "gripper_calibration":state.GRIPPER_CALIBRATION_REPORT,
        "physics":str(state.A.physics),
        "policy_anchor":str(state.A.policy_anchor),
        "deterministic_seed":int(state.A.deterministic_seed),
        "nq":int(state.m.nq),
        "nv":int(state.m.nv),
        "nu":int(state.m.nu),
        "na":int(state.m.na),
        "nmocap":int(state.m.nmocap),
        "time":float(state.d.time),
        "qpos":arr(state.d.qpos),
        "qvel":arr(state.d.qvel),
        "qacc":arr(state.d.qacc),
        "qacc_warmstart":arr(state.d.qacc_warmstart),
        "ctrl":arr(state.d.ctrl),
        "qfrc_applied":arr(state.d.qfrc_applied),
        "xfrc_applied":arr(state.d.xfrc_applied),
        "act":arr(state.d.act) if state.m.na else [],
        "mocap_pos":arr(state.d.mocap_pos) if state.m.nmocap else [],
        "mocap_quat":arr(state.d.mocap_quat) if state.m.nmocap else [],
        INTEGRATION_STATE_KEY:capture_integration_state(mujoco,state.m,state.d),
        "arms":{},
        "objects":object_free_joint_report(state),
        "rule_grasp":rule_grasp,
        "pose_provenance":{
            "definition":(
                "post-ready state after policy-anchor IK and before replay action 0"
                if str(label)=="post-ready"
                else str(label)
            ),
            "startup_pose_source":f"{state.ROBOT} model init_qpos",
            "policy_anchor_requested":str(state.A.policy_anchor),
            "policy_anchor_effective":str(state.POLICY_ANCHOR_EFFECTIVE),
            "policy_anchor_episode":state.POLICY_ANCHOR_EPISODE,
            "policy_body_targets":{
                key:{
                    "body":mujoco.mj_id2name(
                        state.m,mujoco.mjtObj.mjOBJ_BODY,state.ARM[key]["policy"]
                    ),
                    "pos_m":arr(state.READY[key][0]),
                    "quat_wxyz":arr(state.READY[key][1]),
                }
                for key in ("R","L")
            },
        },
    }
    for key in ("R","L"):
        p,q=policy_tcp_pose(state,key)
        payload["arms"][key]={
            "joint_names":list(state.ARM[key]["joint_names"]),
            "q":[float(x) for x in state.d.qpos[state.ARM[key]["qi"]]],
            "policy_body":mujoco.mj_id2name(
                state.m,mujoco.mjtObj.mjOBJ_BODY,state.ARM[key]["policy"]
            ),
            "tcp_pos":arr(p),
            "tcp_quat":arr(q),
            "finger_actual":float(finger_actual(state,key)),
        }
    mimicgen_report=state.MIMICGEN_INIT_REPORT
    if mimicgen_report is not None:
        payload["mimicgen_init"]=mimicgen_report
    object_settle_report=state.OBJECT_SETTLE_REPORT
    if object_settle_report is not None:
        payload["object_settle"]=object_settle_report
        if include_settled_contract:
            payload["object_settle_contract"]=settled_init_state_contract(state)
    # Bind the saved object state to the layout inputs and seated start it was
    # built from; _load_sim_init_state refuses it for any other scene.
    payload[INIT_IDENTITY_KEY]=_current_init_identity(state)
    return payload

def _write_sim_init_state(state,path,label,require_settled_contract=False):
    path=os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    os.makedirs(os.path.dirname(path) or ".",exist_ok=True)
    payload=_sim_state_payload(state,
        label,include_settled_contract=bool(require_settled_contract)
    )
    if require_settled_contract:
        saved_report=payload.get("object_settle")
        if (
            isinstance(saved_report,dict)
            and saved_report.get("source")=="init_state"
        ):
            saved_report=saved_report.get("saved_report")
        if not isinstance(saved_report,dict):
            raise RuntimeError(
                "settled init-state output has no authoritative physics report"
            )
        canonical=validate_settled_objects(state,"init_state_write")
        saved_report=dict(saved_report)
        saved_report.update({
            "objects":canonical["objects"],
            "support":canonical["support"],
            "ready_gate":canonical["ready_gate"],
            "saved_state_validation":canonical,
        })
        payload["objects"]=canonical["objects"]
        payload["object_settle"]=saved_report
        try:
            validate_saved_settled_init_payload(
                payload,settled_init_state_contract(state)
            )
        except ValueError as exc:
            raise RuntimeError(
                f"settled init-state contract validation failed before write: {exc}"
            ) from exc
    atomic_write_json(path,payload,mode=0o600)
    motion=", ".join(
        f"{name}:stationary={state['stationary']} "
        f"linear={state['linear_speed_m_s']:.3e}m/s angular={state['angular_speed_rad_s']:.3e}rad/s"
        for name,state in payload["objects"].items()
    )
    print(f"[sim] init-state out -> {path}; object velocities [{motion}]",flush=True)

def _canonicalize_robosuite_controllers_from_current(state):
    refreshed=0
    for robot in getattr(state.env,"robots",()):
        composite=getattr(robot,"composite_controller",None)
        if composite is None:
            continue
        composite.update_state()
        for controller in composite.part_controllers.values():
            controller.update(force=True)
            refreshed+=1
        composite.reset()
    return refreshed

def _load_sim_init_state(state,path):
    path=os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    with open(path) as f:
        payload=json.load(f)
    try:
        state_robot=normalize_robot_name(payload.get("robot",""))
    except ValueError as exc:
        raise SystemExit(f"[sim] init-state {path} has invalid/missing robot: {payload.get('robot')!r}") from exc
    if state_robot != state.ROBOT:
        raise SystemExit(f"[sim] init-state {path} robot mismatch: got {state_robot}, expected {state.ROBOT}")
    state_signature=payload.get("model_signature")
    if state_signature is not None and state_signature != state.MODEL_SIGNATURE:
        raise SystemExit(
            f"[sim] init-state {path} model signature mismatch: "
            f"got {state_signature}, expected {state.MODEL_SIGNATURE}"
        )
    if state_signature is None:
        print(f"[sim] warning: legacy init-state {path} has no model signature; dimension checks only",flush=True)
    # The model signature does not cover object poses, and an explicit
    # --init-state-in never passes through the batch cache name. Refuse a
    # state whose layout inputs or seated start are not this run's.
    try:
        validate_init_state_identity(
            payload,_current_init_identity(state),source=f"init-state {path}"
        )
    except InitIdentityError as exc:
        raise SystemExit(f"[sim] {exc}") from exc
    _validate_calibration_provenance(state, payload, f"init-state {path}")
    for name,expected in (("nq",state.m.nq),("nv",state.m.nv),("nu",state.m.nu),("na",state.m.na),("nmocap",state.m.nmocap)):
        got=int(payload.get(name,-1))
        if got!=int(expected):
            raise SystemExit(f"[sim] init-state {path} {name} mismatch: got {got}, expected {int(expected)}")
    if not has_integration_state(payload):
        state.d.time=float(payload.get("time",0.0))
        state.d.qpos[:]=np.asarray(payload["qpos"],float).reshape(state.m.nq)
        state.d.qvel[:]=np.asarray(payload["qvel"],float).reshape(state.m.nv)
        if "qacc" in payload:
            state.d.qacc[:]=np.asarray(payload["qacc"],float).reshape(state.m.nv)
        state.d.ctrl[:]=np.asarray(payload.get("ctrl",[]),float).reshape(state.m.nu)
        if state.m.na and "act" in payload:
            state.d.act[:]=np.asarray(payload["act"],float).reshape(state.m.na)
        if state.m.nmocap and payload.get("mocap_pos"):
            state.d.mocap_pos[:]=np.asarray(payload["mocap_pos"],float).reshape(state.m.nmocap,3)
        if state.m.nmocap and payload.get("mocap_quat"):
            state.d.mocap_quat[:]=np.asarray(payload["mocap_quat"],float).reshape(state.m.nmocap,4)
    try:
        state_meta=restore_integration_or_legacy_state(mujoco,state.m,state.d,payload)
    except ValueError as exc:
        raise SystemExit(f"[sim] invalid init-state {path}: {exc}") from exc
    forward_preserving_warmstart(mujoco,state.m,state.d,state_meta["warmstart"])
    refreshed=_canonicalize_robosuite_controllers_from_current(state)
    state.d.qacc_warmstart[:]=state_meta["warmstart"]
    reset_yubi_from_current(state)
    print(f"[sim] init-state in <- {path} source={state_meta['source']} "
          f"spec={state_meta['spec']} size={state_meta['size']} controllers={refreshed}",flush=True)
    return payload

def _validate_loaded_object_state_matches_payload(state,payload,current_objects):
    """Bind provenance metadata to the integration state MuJoCo restored."""
    saved=payload.get("objects")
    expected=set(state._TASK_OBJECT_NAMES)
    if not isinstance(saved,dict) or set(saved)!=expected:
        raise RuntimeError(
            "saved object state does not match the contracted task-object set"
        )
    failures=[]
    for name in state._TASK_OBJECT_NAMES:
        current=current_objects.get(name,{})
        for field,size in (("qpos",7),("qvel",6),("qacc",6)):
            saved_value=np.asarray(saved[name].get(field,[]),dtype=np.float64)
            current_value=np.asarray(current.get(field,[]),dtype=np.float64)
            if (
                saved_value.shape!=(size,)
                or current_value.shape!=(size,)
                or not np.array_equal(saved_value,current_value)
            ):
                failures.append(f"{name}.{field}")
    if failures:
        raise RuntimeError(
            "restored MuJoCo integration state differs from saved object "
            f"metadata: {failures}"
        )

__all__ = [
    "_canonicalize_robosuite_controllers_from_current",
    "_load_sim_init_state",
    "_sim_state_payload",
    "_validate_calibration_provenance",
    "_validate_loaded_object_state_matches_payload",
    "_write_sim_init_state",
]
