# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""MimicGen frame-0 initialization."""

from __future__ import annotations

import json
import os

import mujoco
import numpy as np

from robosuite_env.rollout.control.pose_math import policy_tcp_pose, qmul
from robosuite_env.rollout.control.rule_grasp import _finger_actual as finger_actual
from robosuite_env.rollout.state.init_state import (
    _canonicalize_robosuite_controllers_from_current as canonicalize_robosuite_controllers_from_current,
)
from robosuite_env.rollout.setup.frames import (
    _quat_distance_deg as quat_distance_deg,
    xy2wx_,
)
from robosuite_env.rollout.build.gripper import reset_yubi_from_current

def _apply_mimicgen_training_init(state,episode_path,state_log_root):
    """Restore the exact physical frame-0 configuration used by MimicGen FT."""
    episode_path=os.path.abspath(os.path.expanduser(os.path.expandvars(episode_path)))
    state_log_root=os.path.abspath(os.path.expanduser(os.path.expandvars(state_log_root)))
    if not os.path.isfile(episode_path):
        raise SystemExit(f"[sim] MimicGen init episode not found: {episode_path}")
    if not os.path.isdir(state_log_root):
        raise SystemExit(f"[sim] MimicGen init state-log root not found: {state_log_root}")
    import pandas as _pd
    frame=_pd.read_parquet(episode_path).iloc[0]
    required=(
        "source_sample_id",
        "observation.pose.right_hand_root.absolute",
        "observation.pose.left_hand_root.absolute",
        "observation.joint_states",
        "observation.object.cup_bottom",
        "observation.object.cup_center",
        "observation.object.cup_quat",
        "observation.object.plate_center",
        "observation.object.plate_quat",
    )
    missing=[key for key in required if key not in frame.index]
    if missing:
        raise SystemExit(f"[sim] MimicGen init episode missing frame-0 fields: {missing}")
    sample_id=str(frame["source_sample_id"])
    state_log=os.path.join(state_log_root,sample_id,"logs","sim_log.jsonl")
    if not os.path.isfile(state_log):
        raise SystemExit(f"[sim] MimicGen generation state log not found: {state_log}")
    source_frame=None
    with open(state_log) as stream:
        for line in stream:
            if line.strip():
                source_frame=json.loads(line)
                break
    if source_frame is None or int(source_frame.get("frame",-1))!=0:
        raise SystemExit(f"[sim] MimicGen generation state log has no frame 0: {state_log}")

    arm_targets={}
    for key,feature in (("R","observation.pose.right_hand_root.absolute"),("L","observation.pose.left_hand_root.absolute")):
        raw=np.asarray(frame[feature],float)
        q=np.asarray(source_frame.get("arms",{}).get(key,{}).get("q",[]),float)
        if raw.shape!=(7,) or q.shape!=(7,) or not np.all(np.isfinite(raw)) or not np.all(np.isfinite(q)):
            raise SystemExit(f"[sim] invalid MimicGen frame-0 {key} pose/joints in {episode_path} / {state_log}")
        qcanon=xy2wx_(raw[3:7]); qcanon/=np.linalg.norm(qcanon)+1e-12
        qgrip=qmul(qcanon,state.Q_CANON_TO_ROBOT); qgrip/=np.linalg.norm(qgrip)+1e-12
        arm_targets[key]={"q":q,"tcp_pos":raw[:3],"tcp_quat":qgrip}
    grip_state=np.asarray(frame["observation.joint_states"],float).reshape(-1)
    # Replay bundles may append generation-time arm q after the canonical
    # [left_grip, right_grip] observation.  Exact initialization only consumes
    # those first two YUBI values.
    if grip_state.size < 2:
        raise SystemExit(
            f"[sim] invalid MimicGen frame-0 grip in {episode_path}: shape={grip_state.shape}"
        )
    grip=grip_state[:2]
    # The cup free-joint origin is the bottom_site.  MimicGen stores both that
    # physical qpos (cup_bottom) and the geometric center used to synthesize
    # trajectories (cup_center, 38.171 mm above the origin).  Training images
    # were rendered by assigning cup_bottom to the free joint, so using the
    # center here would visibly float the cup above the table.
    cup_bottom=np.asarray(frame["observation.object.cup_bottom"],float)
    cup_center=np.asarray(frame["observation.object.cup_center"],float)
    cup_quat=np.asarray(frame["observation.object.cup_quat"],float)
    plate_pos=np.asarray(frame["observation.object.plate_center"],float)
    plate_quat=np.asarray(frame["observation.object.plate_quat"],float)
    for name,value,shape in (
        ("grip",grip,(2,)),("cup_bottom",cup_bottom,(3,)),("cup_center",cup_center,(3,)),
        ("cup_quat",cup_quat,(4,)),
        ("plate_center",plate_pos,(3,)),("plate_quat",plate_quat,(4,)),
    ):
        if value.shape!=shape or not np.all(np.isfinite(value)):
            raise SystemExit(f"[sim] invalid MimicGen frame-0 {name} in {episode_path}: shape={value.shape}")
    cup_quat=cup_quat/(np.linalg.norm(cup_quat)+1e-12)
    plate_quat=plate_quat/(np.linalg.norm(plate_quat)+1e-12)

    state.d.time=0.0
    state.d.qvel[:]=0.0; state.d.qacc[:]=0.0; state.d.qacc_warmstart[:]=0.0
    state.d.ctrl[:]=0.0; state.d.qfrc_applied[:]=0.0; state.d.xfrc_applied[:]=0.0
    if state.m.na: state.d.act[:]=0.0
    for key in ("R","L"):
        state.d.qpos[state.ARM[key]["qi"]]=arm_targets[key]["q"]
    # LeRobot joint-state order is [left_grip, right_grip].
    for key,value in (("L",grip[0]),("R",grip[1])):
        for qadr in state.ARM[key]["fj"]: state.d.qpos[qadr]=float(value)
    state.d.qpos[state.CUPa:state.CUPa+3]=cup_bottom; state.d.qpos[state.CUPa+3:state.CUPa+7]=cup_quat
    if state.PLATEa is None:
        raise SystemExit("[sim] MimicGen exact init requires the plate free joint")
    state.d.qpos[state.PLATEa:state.PLATEa+3]=plate_pos; state.d.qpos[state.PLATEa+3:state.PLATEa+7]=plate_quat
    mujoco.mj_forward(state.m,state.d)
    refreshed=canonicalize_robosuite_controllers_from_current(state)
    state.d.qacc_warmstart[:]=0.0
    reset_yubi_from_current(state)
    state.POLICY_INIT_GRIP={"L":float(grip[0]),"R":float(grip[1])}
    state.cup=state.d.qpos[state.CUPa:state.CUPa+3].copy()

    arms_report={}
    position_errors=[]; orientation_errors=[]
    for key in ("R","L"):
        actual_pos,actual_quat=policy_tcp_pose(state,key)
        pos_err=float(np.linalg.norm(actual_pos-arm_targets[key]["tcp_pos"])*1000.0)
        ori_err=quat_distance_deg(actual_quat,arm_targets[key]["tcp_quat"])
        position_errors.append(pos_err); orientation_errors.append(ori_err)
        arms_report[key]={
            "tcp_position_error_mm":pos_err,
            "tcp_orientation_error_deg":ori_err,
            "q":[float(x) for x in state.d.qpos[state.ARM[key]["qi"]]],
            "finger_actual":float(finger_actual(state,key)),
            "finger_target":float(state.POLICY_INIT_GRIP[key]),
        }
    cup_position_error=float(np.linalg.norm(state.d.qpos[state.CUPa:state.CUPa+3]-cup_bottom)*1000.0)
    plate_position_error=float(np.linalg.norm(state.d.qpos[state.PLATEa:state.PLATEa+3]-plate_pos)*1000.0)
    cup_orientation_error=quat_distance_deg(state.d.qpos[state.CUPa+3:state.CUPa+7],cup_quat)
    plate_orientation_error=quat_distance_deg(state.d.qpos[state.PLATEa+3:state.PLATEa+7],plate_quat)
    position_errors.extend((cup_position_error,plate_position_error))
    orientation_errors.extend((cup_orientation_error,plate_orientation_error))
    max_position=max(position_errors); max_orientation=max(orientation_errors)
    if max_position>float(state.A.mimicgen_init_position_tol_mm) or max_orientation>float(state.A.mimicgen_init_orientation_tol_deg):
        raise SystemExit(
            f"[sim] MimicGen exact init mismatch sample={sample_id}: max_position={max_position:.6f} mm "
            f"(tol={state.A.mimicgen_init_position_tol_mm}) max_orientation={max_orientation:.6f} deg "
            f"(tol={state.A.mimicgen_init_orientation_tol_deg})"
        )
    report={
        "sample_id":sample_id,"episode":episode_path,"generation_state_log":state_log,
        "controllers_refreshed":int(refreshed),"arms":arms_report,
        "cup_position_error_mm":cup_position_error,"plate_position_error_mm":plate_position_error,
        "cup_orientation_error_deg":cup_orientation_error,"plate_orientation_error_deg":plate_orientation_error,
        "max_position_error_mm":max_position,"max_orientation_error_deg":max_orientation,
        "position_tolerance_mm":float(state.A.mimicgen_init_position_tol_mm),
        "orientation_tolerance_deg":float(state.A.mimicgen_init_orientation_tol_deg),
        "cup_bottom":[float(x) for x in cup_bottom],"cup_center":[float(x) for x in cup_center],
        "plate_center":[float(x) for x in plate_pos],
        "grip":{"L":float(grip[0]),"R":float(grip[1])},
    }
    print(f"[sim] MimicGen exact init sample={sample_id} max_pos={max_position:.6f}mm "
          f"max_ori={max_orientation:.6f}deg controllers={refreshed}",flush=True)
    return report

__all__ = ["_apply_mimicgen_training_init"]
