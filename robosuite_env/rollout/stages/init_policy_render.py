# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Policy/replay source initialization and renderer setup stage."""

import functools
import json
import os
import socket
import time

import cv2 as cv
import mujoco
import numpy as np

from robosuite_env.policy import policy_io as pio
from robosuite_env.rollout.build.gripper import force_finger_state as _force_finger_state
from robosuite_env.rollout.control.ik import _new_joint_ik_trajectory
from robosuite_env.rollout.control.pose_math import eef_pose, policy_tcp_pose
from robosuite_env.rollout.control.ready import _joint_ik_ready_gate_report
from robosuite_env.rollout.metrics import FrameMetricsCollector
from robosuite_env.rollout.observe.deploy_anchor import _anchor_table_origin
from robosuite_env.rollout.observe.policy_client import _PolicyBinding
from robosuite_env.rollout.observe.wrist import setup_policy_wrist
from robosuite_env.rollout.outputs import ArraySpool, EmptyArraySpool, open_video_capture
from robosuite_env.rollout.policy_chunks import PolicyPlanConfig
from robosuite_env.rollout.render.overview import setup_overview_render
from robosuite_env.rollout.render.video_wrist import setup_video_wrist
from robosuite_env.rollout.replay.sources import (
    ReplayActionSource,
    load_replay,
    load_state_log_replay,
)
from robosuite_env.rollout.setup.frames import ee_tgt
from robosuite_env.rollout.state.settle import _object_free_joint_report
from robosuite_env.rollout.control.rule_grasp import _finger_actual
from robosuite_env.tasks.success import make_success_criterion

def success_options(state):
    """Options for make_success_criterion."""
    return {
        "contact_tolerance_m": float(state.A.success_contact_tol_mm) / 1000.0,
        "contact_penetration_tolerance_m": float(state.A.success_contact_penetration_tol_mm) / 1000.0,
        "place_contact_penetration_tolerance_m": float(state.A.success_place_contact_penetration_tol_mm) / 1000.0,
        "plane_gap_tolerance_m": float(state.A.success_plane_gap_tol_mm) / 1000.0,
        "plane_penetration_tolerance_m": float(state.A.success_plane_penetration_tol_mm) / 1000.0,
        "bottom_contact_tolerance_m": float(state.A.success_bottom_contact_tol_mm) / 1000.0,
        "footprint_slack_m": float(state.A.success_footprint_slack_mm) / 1000.0,
        "upright_tolerance_deg": float(state.A.success_upright_tol_deg),
        "cup_above_plate_test": str(
            getattr(state.A, "cup_above_plate_test", "middle") or "middle"
        ),
        "score_condition": state.A.score_condition,
        "geometry_mode": state.A.success_mode,
        "stable_sec": float(state.A.success_stable_sec),
        "evaluation_hz": 50.0,
        "linear_speed_m_s": float(state.A.success_linear_speed_mm_s) / 1000.0,
        "angular_speed_rad_s": float(state.A.success_angular_speed_rad_s),
        "table_z_m": float(state.env.table_offset[2]),
        "table_offset_m": [float(value) for value in state.env.table_offset],
        "robot_geom_ids": sorted(state.ALLARM),
        # The prompt fixes these; the env exported them when it built.
        "box_pack_object": os.environ.get("RS_BOX_PACK_OBJECT", "tape"),
        "box_pack_hand": os.environ.get("RS_BOX_PACK_HAND", "right"),
        "box_pack_direction": os.environ.get(
            "RS_BOX_PACK_DIRECTION", "into_box"
        ),
    }

def record_locked_openpi(state):
    """Refuse an unlocked OpenPI in this worker and record what it verified.

    Uses the shared guard (``ensure_locked_openpi``) in its ``CHEAP`` mode:
    the imported file is one ``RECORD`` lists, and ``direct_url.json`` names the
    locked commit. The launcher ran the ``FULL`` check once for the run. The
    records, marked ``checked_in: "rollout worker"``, land in
    ``state.LOCKED_IMPORTS``, which ``sim_summary.json`` reports, so the run
    states the OpenPI this process imported and the check it ran here.
    """

    from robosuite_env.policy.policy_batch_infer import ensure_locked_openpi

    records = [
        {**record, "checked_in": "rollout worker"}
        for record in ensure_locked_openpi(state.REPO)
    ]
    state.LOCKED_IMPORTS = list(getattr(state, "LOCKED_IMPORTS", None) or []) + records
    return records


def run(state):
    """Initialise policy source, renderers, spools, and per-run logging."""
    if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.FR3_STRATEGY.uses_joint_ik:
        for _key in ("R","L"):
            _initial_joint_command=np.asarray(
                state.d.qpos[state.ARM[_key]["qi"]],dtype=np.float64
            ).copy()
            state._initial_joint_command = _initial_joint_command
            state._JOINT_IK_TRAJECTORY[_key]=_new_joint_ik_trajectory(state,
                _key,
                _initial_joint_command
            )
            state._JOINT_IK_COMMAND[_key]=_initial_joint_command
            state._JOINT_IK_COMMAND_VELOCITY[_key]=(
                state._JOINT_IK_TRAJECTORY[_key].velocity.copy()
            )
            state._JOINT_IK_COMMAND_ACCELERATION[_key]=(
                state._JOINT_IK_TRAJECTORY[_key].acceleration.copy()
            )
            state._JOINT_IK_LAST_TARGET_UPDATE[_key]=(
                state._JOINT_IK_TRAJECTORY[_key].last_target_update
            )
            if state._ARM_DESK_COLLISION_GATE is not None:
                _initial_safety=state._ARM_DESK_COLLISION_GATE.check(
                    _key,
                    _initial_joint_command,
                    base_qpos=np.asarray(state.d.qpos,dtype=np.float64),
                )
                state._initial_safety = _initial_safety
                state._JOINT_IK_INITIAL_SAFETY[_key]={
                    "safe":bool(_initial_safety.safe),
                    "minimum_clearance_mm":round(
                        float(_initial_safety.minimum_distance_m)*1000.0,6
                    ),
                    "closest_arm_geom":_initial_safety.arm_geom_name,
                    "closest_desk_geom":_initial_safety.desk_geom_name,
                    "collision_contact_pairs":int(
                        _initial_safety.collision_contact_pairs
                    ),
                    "ignored_zero_distance_pairs":int(
                        _initial_safety.ignored_zero_distance_pairs
                    ),
                }
                if not _initial_safety.safe:
                    raise RuntimeError(
                        "joint_ik initial command is below arm/desk clearance "
                        f"for {_key}: "
                        f"{_initial_safety.minimum_distance_m*1000.0:.6f}mm "
                        f"{_initial_safety.arm_geom_name}/"
                        f"{_initial_safety.desk_geom_name}"
                    )
                state._JOINT_IK_SAFETY_STATS[_key][
                    "initial_min_clearance_mm"
                ]=float(_initial_safety.minimum_distance_m)*1000.0
    state.TGT={"R":ee_tgt(state,"R"),"L":ee_tgt(state,"L")}
    from robosuite_env.robots.fr3.impedance import install_fr3_impedance
    install_fr3_impedance(state)

    _READY_EE={k:[np.asarray(state.TGT[k][0]).copy(),np.asarray(state.TGT[k][1]).copy()] for k in("R","L")}   # ready anchor for absolute replay
    state._READY_EE = _READY_EE
    if state.A.state_log_replay:
        print(
            "[sim] ready for exact state-log replay; each saved full qpos is "
            "applied before its frame is captured",
            flush=True,
        )
    else:
        _ready_object_label="cup"
        print(
            f"[sim] ready. {_ready_object_label}={np.round(state.cup,3).tolist()} "
            f"R_eef={np.round(state.d.xpos[state.ARM['R']['eef']],3).tolist()}",
            flush=True,
        )
    if state.A.objects in {"box_pack"}:
        if state.A.physics == "dynamic" and not np.isclose(
            float(state.env.control_freq), 50.0, rtol=0.0, atol=1e-9
        ):
            raise SystemExit(
                f"[sim] {state.A.objects} stability contract requires exactly "
                "50 Hz dynamic "
                f"control ticks, got {float(state.env.control_freq):g} Hz"
            )
    _SUCCESS_OPTIONS = success_options(state)
    state._SUCCESS_OPTIONS = _SUCCESS_OPTIONS
    try:
        _SUCCESS_CRITERION = make_success_criterion(
            state.A.success_criterion, mujoco=mujoco, model=state.m, data=state.d, options=_SUCCESS_OPTIONS
        )
        state._SUCCESS_CRITERION = _SUCCESS_CRITERION
    except ValueError as exc:
        if str(state.A.success_criterion).strip().lower() == "auto" and "could not resolve" in str(exc):
            _SUCCESS_CRITERION = None
            state._SUCCESS_CRITERION = _SUCCESS_CRITERION
            print(f"[sim] success criterion auto disabled: {exc}", flush=True)
        else:
            raise SystemExit(f"[sim] {exc}") from exc
    if _SUCCESS_CRITERION is not None:
        print(f"[sim] success criterion: {_SUCCESS_CRITERION.describe()}", flush=True)
    _FRAME_METRICS=FrameMetricsCollector(
        mujoco=mujoco,
        model=state.m,
        data=state.d,
        arm=state.ARM,
        cup_qpos_address=state.CUPa,
        plate_qpos_address=state.PLATEa,
        render_profile_name=state.A.policy_yubi_render_profile,
        eef_pose=functools.partial(eef_pose,state),
        policy_tcp_pose=functools.partial(policy_tcp_pose,state),
    )
    state._FRAME_METRICS = _FRAME_METRICS

    if state.A.anchor_dump:
        if not (state.POLICY_SOURCE and state.A.policy_anchor=="deploy"):
            raise SystemExit("[sim] --anchor-dump requires --policy-anchor deploy with --policy")
        if state.POLICY_INIT_GRIP is not None:
            _force_finger_state(state,state.POLICY_INIT_GRIP["L"],state.POLICY_INIT_GRIP["R"])
        _anchor_dir=os.path.abspath(state.A.anchor_dump)
        state._anchor_dir = _anchor_dir
        os.makedirs(_anchor_dir,exist_ok=True)
        _anchor_payload={
            "table_origin":[float(x) for x in (state.ANCHOR_TABLE_ORIGIN if state.ANCHOR_TABLE_ORIGIN is not None else _anchor_table_origin(state))],
            "grip_init":state.POLICY_INIT_GRIP if state.POLICY_INIT_GRIP is not None else {"L":_finger_actual(state,"L"),"R":_finger_actual(state,"R")},
        }
        state._anchor_payload = _anchor_payload
        _cup_init=np.asarray(state.d.qpos[state.CUPa:state.CUPa+3],dtype=np.float64).copy()
        state._cup_init = _cup_init
        for _key in ("R","L"):
            _p,_q=policy_tcp_pose(state,_key)
            state._p = _p
            state._q = _q
            _anchor_payload[_key]={
                "pos":[float(x) for x in _p],
                "quat":[float(x) for x in _q],
                "joint_q_rad":[
                    float(x) for x in np.asarray(
                        state.d.qpos[state.ARM[_key]["qi"]],dtype=np.float64
                    )
                ],
            }
        _anchor_payload["initial_arm_joint_pose"]=(
            {
                "source_path":state._INITIAL_ARM_JOINT_POSE.source_path,
                "source_sha256":state._INITIAL_ARM_JOINT_POSE.source_sha256,
                "targets_rad":{
                    key:[float(x) for x in state._INITIAL_ARM_JOINT_Q[key]]
                    for key in ("R","L")
                },
                "ready_gate":_joint_ik_ready_gate_report(state),
            }
            if state._INITIAL_ARM_JOINT_POSE is not None else None
        )
        _anchor_payload["d_R_tcp_cup_mm"]=float(np.linalg.norm(np.asarray(_anchor_payload["R"]["pos"],float)-_cup_init)*1000.0)
        _anchor_payload["objects"]=_object_free_joint_report(state)
        _anchor_json=os.path.join(_anchor_dir,"anchor.json")
        state._anchor_json = _anchor_json
        json.dump(_anchor_payload,open(_anchor_json,"w"),indent=2)
        if not state.A.no_render:
            with state.render_lock():
                _ren0=mujoco.Renderer(state.m,480,640)
                state._ren0 = _ren0
                _cam0=mujoco.MjvCamera(); _cam0.type=mujoco.mjtCamera.mjCAMERA_FREE
                state._cam0 = _cam0
                _view_raw=str(state.A.view_lookat or "").replace(","," ").strip()
                state._view_raw = _view_raw
                _view_lookat=(
                    np.asarray([float(x) for x in _view_raw.split()],dtype=float)
                    if _view_raw else np.asarray([0.0,0.0,float(state.A.table_z)+0.15],dtype=float)
                )
                state._view_lookat = _view_lookat
                if _view_lookat.shape != (3,):
                    raise SystemExit(
                        f"[sim] --view-lookat expects three numbers, got: {state.A.view_lookat!r}"
                    )
                _cam0.azimuth=float(state.A.view_azimuth)  # noqa: V101  # mujoco.MjvCamera field
                _cam0.elevation=float(state.A.view_elevation)  # noqa: V101  # mujoco.MjvCamera field
                _cam0.distance=float(state.A.view_distance)
                _cam0.lookat[:]=_view_lookat
                _vopt0=mujoco.MjvOption()
                state._vopt0 = _vopt0
                _vopt0.geomgroup[:]=0
                _vopt0.geomgroup[1]=1
                _ren0.update_scene(
                    state.d,
                    camera=_cam0,
                    scene_option=_vopt0,
                )
                if state.A.show_origin:
                    _scene0=_ren0.scene
                    state._scene0 = _scene0
                    def _anchor_origin_geom(_type,_size,_pos,_rgba):
                        if _scene0.ngeom >= _scene0.maxgeom:
                            return
                        mujoco.mjv_initGeom(
                            _scene0.geoms[_scene0.ngeom],_type,
                            np.asarray(_size,np.float64),np.asarray(_pos,np.float64),
                            np.eye(3).flatten(),np.asarray(_rgba,np.float32),
                        )
                        _scene0.ngeom += 1
                    _anchor_origin_geom(mujoco.mjtGeom.mjGEOM_SPHERE,[0.03,0,0],[0,0,0],[1,1,0,1])
                    _anchor_origin_geom(mujoco.mjtGeom.mjGEOM_BOX,[0.06,0.0025,0.0025],[0.06,0,0],[1,0,0,1])
                    _anchor_origin_geom(mujoco.mjtGeom.mjGEOM_BOX,[0.0025,0.06,0.0025],[0,0.06,0],[0,1,0,1])
                    _anchor_origin_geom(mujoco.mjtGeom.mjGEOM_BOX,[0.0025,0.0025,0.06],[0,0,0.06],[0.3,0.3,1,1])
                _img0=_ren0.render().copy()
                state._img0 = _img0
                cv.imwrite(os.path.join(_anchor_dir,"anchor_frame0.png"),cv.cvtColor(_img0,cv.COLOR_RGB2BGR))
                _ren0.close()
        print(f"[sim] anchor dump -> {_anchor_json}",flush=True)
        raise SystemExit(0)

    # ---------------- source iterator ----------------
    state.POLICY=bool(state.POLICY_SOURCE)
    _STATE_REPLAY_POLICY_INSETS=None
    state._STATE_REPLAY_POLICY_INSETS = _STATE_REPLAY_POLICY_INSETS
    state._STATE_RENDER_VALIDATION=None
    state._STATE_RENDER_QPOS=None
    state._STATE_RENDER_ROWS_VALIDATED=0
    state._STATE_RENDER_ROWS_APPLIED=0
    state._STATE_RENDER_MAX_APPLY_ERROR=0.0
    state._STATE_REPLAY_TARGET_BASELINE_SOURCE=None
    if state.A.grasp_test:
        state.N = state.GRASP_N
        _grasp_where=(
            " ".join(f"{hand}={poses['grasp'].round(4).tolist()}"
                     for hand,poses in state._P_BY_HAND.items())
            + (f" span={state._GRASP_SPAN:.4f}" if state._GRASP_SPAN is not None else " span=per-hand-json")
            if state.GH=="both" else state._P_GRASP.round(4).tolist()
        )
        print(f"[sim] grasp-test: hand={state.GH} N={state.N} phases={state._PB} grasp={_grasp_where} "
              f"top={state._P_TOP.round(4).tolist()} yaw={state.A.grasp_yaw}", flush=True)
    elif state.A.state_log_replay:
        load_state_log_replay(state)
        _STATE_REPLAY_POLICY_INSETS = state._STATE_REPLAY_POLICY_INSETS
    elif state.A.replay:
        relR,relL,grip,state.N,absR,absL,REPLAY_OBJ=load_replay(state.A.replay,state.A.frames)
        state.relR = relR
        state.relL = relL
        state.grip = grip
        state.absR = absR
        state.absL = absL
        state.REPLAY_OBJ = REPLAY_OBJ
        if state.A.replay_object_source == "parquet":
            missing=[k for k in ("observation.object.cup_bottom","observation.object.plate_center") if k not in REPLAY_OBJ]
            state.missing = missing
            if missing:
                raise SystemExit(f"[sim] --replay-object-source parquet missing columns in --replay: {missing}")
            print(f"[sim] replay object source: parquet columns={sorted(REPLAY_OBJ)}", flush=True)
        else:
            REPLAY_OBJ={}
            state.REPLAY_OBJ = REPLAY_OBJ
        if len(grip): _force_finger_state(state,grip[0,0],grip[0,1])
        # ABSOLUTE-ANCHORED replay: reproduce the recorded displacement exactly (relative-delta integration would be
        # world-additive on LOCAL-frame deltas -> wrong height/shape). absolute frame is z-up == sim world
        # up to a translation, so pos[t]=ready+(a[t]-a[0]); orientation = recorded rot-from-frame0 onto ready.
        _A0R=absR[0].copy(); _A0L=absL[0].copy()
        state._A0R = _A0R
        state._A0L = _A0L
        state._DATA={"R":[absR[0,:3].copy(),absR[0,3:7].copy()],"L":[absL[0,:3].copy(),absL[0,3:7].copy()]}  # data-frame pose (xyzw), for delta/policy integration
        state.step_action = ReplayActionSource(relR, relL, grip)
    elif state.A.ik_replay:
        import pandas as _pd
        _JS=np.stack(_pd.read_parquet(state.A.ik_replay)["observation.joint_states"].values).astype(float)
        state._JS = _JS
        if _JS.shape[1]<16: raise SystemExit(f"[sim] --ik-replay expects joint_states dim>=16 (grip2+R7+L7); got {_JS.shape}. Use an added_ik parquet.")
        state.IKGRIP=_JS[:,:2]; state.IK_JR=_JS[:,2:9]; state.IK_JL=_JS[:,9:16]; state.N=len(_JS) if state.A.frames<=0 else min(state.A.frames,len(_JS))
        if len(state.IKGRIP): _force_finger_state(state,state.IKGRIP[0,0],state.IKGRIP[0,1])
        print(f"[sim] ik-replay: {os.path.basename(state.A.ik_replay)} frames={len(_JS)} joint_states_dim={_JS.shape[1]} (drive FR3 with stored IK, no solve)",flush=True)
    else:
        # ---------------- policy (openpi pi0.5) rollout ----------------
        state.N=state.A.frames if state.A.frames>0 else 150
        # Convert the controlled robot body back to the dataset / policy hand-root
        # frame for observations, using FR3's fixed rotation.
        _Q_FC=state.Q_ROBOT_TO_CANON.copy()
        state._Q_FC = _Q_FC
        _DROT_Q=_Q_FC.copy()
        state._DROT_Q = _DROT_Q
        GI_L,GI_R=pio.GRIP_L_INDEX,pio.GRIP_R_INDEX   # yubi action layout: [left pose(7), right pose(7), finger(2)]=[gripL, gripR]
        state.GI_L = GI_L
        state.GI_R = GI_R
        # The same check the launcher preflight runs: refuse an OpenPI that is
        # not the installed, locked distribution before a policy is loaded or
        # a batch server is contacted (this process still resizes with
        # openpi_client).
        record_locked_openpi(state)
        if state.A.policy_batch_server:
            from robosuite_env.policy.policy_batch_server import BatchPolicyClient
            _timeout = None if float(state.A.policy_batch_timeout_sec) <= 0.0 else float(state.A.policy_batch_timeout_sec)
            state._timeout = _timeout
            _cid = state.A.policy_batch_client_id or f"{socket.gethostname()}:{os.getpid()}:{os.path.basename(state.LOGDIR)}"
            state._cid = _cid
            print(f"[sim] connecting policy batch server={state.A.policy_batch_server} client_id={_cid}", flush=True)
            state.policy=BatchPolicyClient(state.A.policy_batch_server, client_id=_cid, timeout_sec=_timeout)
            print(f"[sim] policy batch server connected metadata={state.policy.metadata}", flush=True)
            # The server names the checkpoint it loaded, which is the only way
            # a batch worker learns it: --policy goes to the server, and this
            # process is given an address. Where the table knows that
            # checkpoint's wording, hold the prompt to it -- the launcher's
            # preflight has already done so, and this catches a launch that
            # never ran one (a direct scripts/run_policy_batch.sh call, which
            # takes the checkpoint and the instruction as separate arguments).
            # Where the table does not know it, the table's answer would be a
            # guess, and refusing on a guess kills a correct run minutes in.
            if state.A.objects == "box_pack":
                from robosuite_env.tasks.box_pack import (
                    box_pack_checkpoint_is_registered,
                    require_box_pack_instruction,
                )

                _served = (state.policy.metadata or {}).get("policy")
                if box_pack_checkpoint_is_registered(_served):
                    require_box_pack_instruction(
                        state.A.instruction,
                        int(state.A.box_pack_task_index),
                        checkpoint=_served,
                    )
                else:
                    print(
                        "[sim] box pack prompt not held to the served checkpoint "
                        f"{_served!r}: no wording is registered for it",
                        flush=True,
                    )
        else:
            from robosuite_env.policy import policy_batch_infer as _policy_backend
            print(
                f"[sim] loading policy ckpt={state.A.policy} "
                f"config={state.A.config}/{state.A.config_src} ah={state.A.ah or 'config'} ...",
                flush=True,
            )
            state.policy=_policy_backend.load_openpi_policy(
                state.A.policy,
                config=state.A.config,
                config_src=state.A.config_src,
                action_horizon=state.A.ah,
                # A chained rollout overrides obs["prompt"] on every inference,
                # so the loader default must still be a trained primitive rather
                # than the untrained high-level label.
                instruction=state.A.instruction,
                repo_root=state.REPO,
            )
            print(
                "[sim] policy loaded "
                f"metadata={_policy_backend.policy_load_metadata(state.policy)}",
                flush=True,
            )
        if state.A.rf_overlap:
            from robosuite_env.policy import policy_batch_infer as _policy_backend
            state.policy_backend = _policy_backend
        else:
            state.policy_backend = None
        _grip={"R":0.0,"L":0.0}
        state._grip = _grip
        if state.POLICY_INIT_GRIP is not None: _grip.update(state.POLICY_INIT_GRIP)
        state._pstate={
            "block_step":0,
            "action_start":0,
            "chunk":None,
            "chunk_index":-1,
            "infer_count":0,
            "obs_count":0,
            "active_obs_meta":{},
            "pending_obs":None,
            "pending_obs_meta":None,
            "pending_inferred":None,
            "last_infer_meta":None,
            "last_prefix_error":None,
            "last_bridge_error":None,
            "last_bridge_rebased_prev_distance":None,
            "active_dynamic_prev_weight":False,
            "active_prev_weight_schedule":None,
            "conditioning_rebase_meta":None,
            "target_trajectory":None,
        }
        _policy_binding = _PolicyBinding(state, _Q_FC)
        state.ee_now = _policy_binding.ee_now
        state.rel_state = _policy_binding.rel_state
        state.pol_apply = _policy_binding.pol_apply
    print(f"[sim] source frames N={state.N}"+("  (POLICY rollout)" if state.POLICY else ""),flush=True)

    setup_overview_render(state)
    setup_video_wrist(state)
    setup_policy_wrist(state)
    realcap=None
    state.realcap = realcap
    if state.A.real_video and os.path.isfile(state.A.real_video):
        realcap=open_video_capture(state.A.real_video)
        if not realcap.isOpened():
            raise RuntimeError(f"OpenCV could not open real video: {state.A.real_video}")
        print(f"[sim] real video: {state.A.real_video} ({int(realcap.get(7))}f)",flush=True)
        state.realcap = realcap

    # ============================ PASS 1: policy wrists + state capture ============================
    # The isolated path stores render-safe state only, then writes the actual
    # overview pixels after wrist teardown.
    _FRAME_SPOOL_DIR=(
        os.environ.get("REAL2SIM_FRAME_SPOOL_DIR")
        or os.path.join(state.LOGDIR,".frame-spool")
    )
    state._FRAME_SPOOL_DIR = _FRAME_SPOOL_DIR
    _FRAME_SPOOL_SAMPLES=(0 if state.A.no_render else int(state.N)*int(state.DYNAMIC_SUBFRAMES))
    state._FRAME_SPOOL_SAMPLES = _FRAME_SPOOL_SAMPLES
    _OVERVIEW_SPOOL_RESERVE=_FRAME_SPOOL_SAMPLES*(480*640*3+1024)
    state._OVERVIEW_SPOOL_RESERVE = _OVERVIEW_SPOOL_RESERVE
    _WRIST_SPOOL_RESERVE=(
        _FRAME_SPOOL_SAMPLES*(2*135*180*3+1024)
        if not state.A.no_render and not state.A.clean_video and not state.A.no_wrist else 0
    )
    state._WRIST_SPOOL_RESERVE = _WRIST_SPOOL_RESERVE
    if state.A.no_render:
        # A no-video rollout must not depend on spool filesystem type or free
        # capacity.
        _OVERVIEW_VIDEO_FRAMES=EmptyArraySpool()
        state._OVERVIEW_VIDEO_FRAMES = _OVERVIEW_VIDEO_FRAMES
        _WRIST_VIDEO_FRAMES=EmptyArraySpool()
        state._WRIST_VIDEO_FRAMES = _WRIST_VIDEO_FRAMES
    else:
        _OVERVIEW_VIDEO_FRAMES=ArraySpool(
            directory=_FRAME_SPOOL_DIR,reserve_bytes=_OVERVIEW_SPOOL_RESERVE
        )
        state._OVERVIEW_VIDEO_FRAMES = _OVERVIEW_VIDEO_FRAMES
        _WRIST_VIDEO_FRAMES=ArraySpool(
            directory=_FRAME_SPOOL_DIR,reserve_bytes=_WRIST_SPOOL_RESERVE
        )
        state._WRIST_VIDEO_FRAMES = _WRIST_VIDEO_FRAMES
    render_source_frames=[]
    state.render_source_frames = render_source_frames
    state._OVERVIEW_STATE_FRAMES=[]
    state._CAPTURE_INDEX=0
    _tloop=time.time()
    state._tloop = _tloop
    state.WALL_STOPPED=False
    logf=open(os.path.join(state.LOGDIR,"sim_log.jsonl"),"w") if (state.A.physics in ("kinematic","dynamic") or state.A.grasp_test) else None
    state.logf = logf
    _SIM_LOG_ENABLED=logf is not None
    state._SIM_LOG_ENABLED = _SIM_LOG_ENABLED
    state._RECS=[]
    state.maxpen=0.0; state.npen=0; state.nfail=0
    _QPREV={"R":None,"L":None}; state.njump=0   # IK jump detector: previous per-arm joint solution + jump count
    state._QPREV = _QPREV

    state._POLICY_CHUNK_RECORDS=[]
    _POLICY_PLAN_CONFIG=PolicyPlanConfig(
        execution_steps=state.A.exec,
        overlap_steps=state.A.rf_overlap,
        sampler_configured=state.A.rf_overlap_sampler,
        collapse_reason=None,
        prev_weight_configured=None,
        step_shift_configured=state.A.rf_overlap_step_shift,
        sample_num_steps=None,
        policy_delta_anchor=state.A.policy_delta_anchor,
        policy_pose_repr=state.A.policy_pose_repr,
    )
    state._POLICY_PLAN_CONFIG = _POLICY_PLAN_CONFIG


__all__ = ["run", "success_options"]
