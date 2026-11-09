# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Pass-1 simulation, control, telemetry, and capture loop."""

import json
import os
import time

import mujoco
import numpy as np

from robosuite_env.policy import policy_io as pio
from robosuite_env.rollout.control.pose_math import (
    _rlist,
    policy_tcp_error,
    qconj,
    qmul,
)
from robosuite_env.rollout.control.rule_grasp import (
    _finger_actual,
    _finger_angles,
    _rule_grasp_status,
    _rule_grasp_update,
)
from robosuite_env.rollout.timing import (
    nearest_control_ticks_for_source_action,
    resolve_dynamic_timing,
)
from real2sim.start_state_guard import check_start_state


from robosuite_env.rollout.build.collision_gate import (
    preview_joint_trajectory_clearance as _preview_joint_trajectory_clearance,
)
from robosuite_env.rollout.build.controllers import _pin_arm_q
from robosuite_env.rollout.build.geoms import (
    gripper_cup_contacts,
    penetrations,
)
from robosuite_env.rollout.build.gripper import (
    grip_actuator_debug as _grip_actuator_debug,
    set_grip_command as _set_grip_command,
    yubi_telemetry as _yubi_telemetry,
)
from robosuite_env.rollout.control.ik import (
    _dynamic_env_action,
    solve_ik_guarded,
)
from robosuite_env.rollout.control.substeps import (
    _kinematic_physics_substeps,
    _latch_policy_grip_intents,
)
from robosuite_env.rollout.metrics_setup import (
    _record_dynamic_sample,
    add_success_metrics,
    frame_debug_metrics,
    tick_success_criterion,
)
from robosuite_env.rollout.observe.policy_client import (
    _activate_policy_chunk,
    _capture_next_policy_observation,
    _infer_policy_chunk,
    _prepare_rebased_conditioning_tail,
)
from robosuite_env.rollout.render.capture import (
    capture_render_sample as _capture_render_sample,
)
from robosuite_env.rollout.replay.sources import (
    _force_replay_object_poses,
    replay_integrate_relative_delta,
)
from robosuite_env.rollout.setup.frames import xy2wx
from robosuite_env.rollout.state.dumps import (
    maybe_dump_record_state as _maybe_dump_record_state,
)

def _check_start_state(state):
    """Look at the pose the episode is about to start from, once, before step 0.

    This is the last moment the start state is still the start state. Logging
    the start pose catches a misplaced object before step 0.
    """
    state._start_state_report = None
    mode = str(getattr(state.A, "start_state_guard", "warn") or "warn").lower()
    if mode == "off":
        return
    # A replay renders recorded states; it does not start an episode. Its
    # first frame is the logged state after the run's own settle, where
    # packed parts may legitimately rest a few millimetres into each other,
    # so the guard (which the run already passed at its real start) would
    # refuse a valid run's video.
    if any(getattr(state.A, name, None) for name in ("state_log_replay", "ik_replay", "replay")):
        return
    tol_mm = float(getattr(state.A, "start_state_penetration_tol_mm", 2.0))
    report = check_start_state(
        state.m,
        state.d,
        penetration_tol_m=tol_mm / 1000.0,
        # The arms touch themselves and their own pads by construction. A pair
        # is only excused when *both* sides are the robot, so a link driven
        # into a task object is still reported.
        ignore_body_prefixes=("robot0", "robot1", "gripper0", "gripper1"),
    )
    state._start_state_report = report
    if not report.failed:
        print(f"[sim] {report.describe()}", flush=True)
        return
    message = "[sim] " + report.describe()
    if mode == "fail":
        raise SystemExit(message)
    print(message, flush=True)


def _success_exit_after_frame(state, frame):
    """Frame the success was first seen, once the grace window has elapsed.

    Once the criterion reports success, simulate a short
    grace window and then stop, rather than running out the whole frame
    budget. The grace frames are ordinary frames -- stepped, logged, scored --
    so a success that does not survive them is still visible in the log, and
    the run's official field still comes from the last frame.

    Returns ``None`` while the rollout should keep going, or the frame at
    which success was first seen once it is time to stop.
    """

    grace = int(getattr(state.A, "success_exit_grace_frames", 0) or 0)
    if grace <= 0 or state._SUCCESS_CRITERION is None:
        return None
    first = getattr(state, "_SUCCESS_FIRST_FRAME", None)
    if first is None:
        record = state._RECS[-1] if getattr(state, "_RECS", None) else None
        success = record.get("success") if isinstance(record, dict) else None
        if isinstance(success, dict):
            achieved = bool(success.get("success"))
        elif success is None:
            # This frame took a path that records no success block; the
            # criterion is still live, so ask it directly rather than
            # treating a missing record as a failure.
            achieved = bool(state._SUCCESS_CRITERION.evaluate().get("success"))
        else:
            achieved = bool(success)
        if not achieved:
            return None
        state._SUCCESS_FIRST_FRAME = int(frame)
        return None

    return int(first) if frame - int(first) >= grace else None


def run(state):
    """Run the main per-frame simulation/control/render loop."""

    _check_start_state(state)
    state._FR3_DYNAMIC_CONTROL_TICKS_PER_ACTION=[]
    state._FR3_DYNAMIC_CONTROL_TICKS_PER_ACTION=state._FR3_DYNAMIC_CONTROL_TICKS_PER_ACTION
    state._FR3_DYNAMIC_ENV_STEP_COUNT=0
    state._FR3_DYNAMIC_ENV_STEP_COUNT=state._FR3_DYNAMIC_ENV_STEP_COUNT
    for state.t in range(state.N):
        pol_rec=None
        _frame_control_repeats=state.DYNAMIC_CONTROL_REPEATS
        _frame_servo_repeats=state.DYNAMIC_CONTROL_REPEATS
        state._frame_telemetry_reps=set()
        state._frame_render_reps=set()
        state._frame_capture_reps=set()
        if state.A.physics=="dynamic" and not state.A.state_log_replay:
            if state._FR3_DYNAMIC_SOURCE_HZ is not None:
                _frame_control_repeats=nearest_control_ticks_for_source_action(
                    int(state.t),
                    float(state._FR3_DYNAMIC_SOURCE_HZ),
                    control_frequency_hz=float(state.A.dynamic_control_hz),
                )
                _frame_servo_repeats=_frame_control_repeats
                state._FR3_DYNAMIC_CONTROL_TICKS_PER_ACTION.append(
                    int(_frame_control_repeats)
                )
            _frame_timing=resolve_dynamic_timing(
                state.A.exec,_frame_control_repeats,state.DYNAMIC_SUBFRAMES
            )
            state._frame_telemetry_reps=set(range(_frame_control_repeats))
            state._frame_render_reps=set(_frame_timing.render_repeats)
            if state.DYNAMIC_SUBFRAMES>1:
                state._frame_capture_reps=set(state._frame_render_reps)
        if state.A.state_log_replay:
            src=state.STATE_ROWS[state.t]
            _exact_render_qpos=(
                state._STATE_RENDER_QPOS[state.t].copy()
                if state._STATE_RENDER_QPOS is not None else None
            )
            if _exact_render_qpos is not None:
                if _exact_render_qpos.shape!=(int(state.m.nq),):
                    raise RuntimeError(
                        "validated render-state qpos changed shape before replay: "
                        f"frame={state.t} shape={_exact_render_qpos.shape} nq={int(state.m.nq)}"
                    )
                state.d.qpos[:]=_exact_render_qpos
                state.d.qvel[:]=0.0
                state.d.qacc[:]=0.0
                state.d.qacc_warmstart[:]=0.0
            if state._STATE_REPLAY_POLICY_INSETS is not None:
                saved_left,saved_right=state._STATE_REPLAY_POLICY_INSETS.frames[state.t]
                if state.A.state_replay_policy_obs_display=="raw224":
                    display_left,display_right=(
                        state._STATE_REPLAY_POLICY_INSETS.raw_display_frames[state.t]
                    )
                else:
                    display_left,display_right=(
                        state._STATE_REPLAY_POLICY_INSETS.display_frames[state.t]
                    )
                transform_left,transform_right=state._STATE_REPLAY_POLICY_INSETS.display_transforms[state.t]
                state.POLICY_OBS_RAW["L"]=saved_left
                state.POLICY_OBS_RAW["R"]=saved_right
                state.POLICY_OBS_DISPLAY["L"]=display_left
                state.POLICY_OBS_DISPLAY["R"]=display_right
                state.POLICY_OBS_DISPLAY_TRANSFORM["L"]=transform_left
                state.POLICY_OBS_DISPLAY_TRANSFORM["R"]=transform_right
            pol_rec=src.get("policy") if isinstance(src.get("policy"),dict) else None
            if pol_rec is not None:
                for key in ("R","L"):
                    tw=pol_rec.get(f"target_world_{key}")
                    tq=pol_rec.get(f"target_quat_{key}")
                    if tw is not None:
                        state.TGT[key][0]=np.asarray(tw,float)
                    if tq is not None:
                        state.TGT[key][1]=np.asarray(tq,float)
            if _exact_render_qpos is None:
                for key in ("R","L"):
                    q7=np.asarray(((src.get("arms") or {}).get(key) or {}).get("q",[]),float)
                    if q7.size == 7:
                        state.d.qpos[state.ARM[key]["qi"]]=q7
                        state.d.qvel[state.ARM[key]["dof"]]=0.0
                cpos=src.get("cup_pos")
                if cpos is not None and len(cpos) >= 3:
                    state.d.qpos[state.CUPa:state.CUPa+3]=np.asarray(cpos[:3],float)
                    if state.CUPa+7 <= len(state.d.qpos):
                        cquat=src.get("cup_quat")
                        state.d.qpos[state.CUPa+3:state.CUPa+7]=(np.asarray(cquat[:4],float)
                                                if cquat is not None and len(cquat) >= 4
                                                else [1.0,0.0,0.0,0.0])
                    state.d.qvel[state.cdof:state.cdof+6]=0.0
                ppos=src.get("plate_pos")
                if state.PLATEa is not None and ppos is not None and len(ppos) >= 3:
                    state.d.qpos[state.PLATEa:state.PLATEa+3]=np.asarray(ppos[:3],float)
                    if state.PLATEa+7 <= len(state.d.qpos):
                        pquat=src.get("plate_quat")
                        state.d.qpos[state.PLATEa+3:state.PLATEa+7]=(np.asarray(pquat[:4],float)
                                                    if pquat is not None and len(pquat) >= 4
                                                    else [1.0,0.0,0.0,0.0])
                    if state.PLATEd is not None:
                        state.d.qvel[state.PLATEd:state.PLATEd+6]=0.0
            grip_cmds={}
            for key in ("R","L"):
                gv=((src.get("grip_actual") or {}).get(key, (src.get("grip") or {}).get(key, _finger_actual(state,key))))
                if _exact_render_qpos is None:
                    for fj in state.ARM[key]["fj"]:
                        state.d.qpos[fj]=float(gv)
                    for fd in state.ARM[key]["fd"]:
                        state.d.qvel[fd]=0.0
                cmd=(src.get("grip") or {}).get(key, gv)
                grip_cmds[key]=float(cmd)
                _set_grip_command(state,key,cmd)
            mujoco.mj_forward(state.m,state.d)
            if state.A.grasp_mode=="rule" and _exact_render_qpos is None:
                _rule_grasp_update(state,grip_cmds, frame=state.t)
            rec={"frame":int(state.t),
                 "arms":src.get("arms",{}),
                 "grip":{"R":round(float(grip_cmds.get("R",_finger_actual(state,"R"))),5),
                         "L":round(float(grip_cmds.get("L",_finger_actual(state,"L"))),5)},
                 "grip_actual":{"R":round(_finger_actual(state,"R"),5),"L":round(_finger_actual(state,"L"),5)}}
            if pol_rec is not None:
                rec["policy"]=pol_rec
            if state.A.debug_grip_actuator:
                rec["grip_actuator"]=_grip_actuator_debug(state,grip_cmds)
            if state.A.debug_grasp_contact:
                rec["grasp_contact"]=gripper_cup_contacts(state,state.A.grasp_contact_tol_mm)
            if state.A.grasp_mode=="rule":
                rec["rule_grasp"]=(
                    src.get("rule_grasp",{})
                    if _exact_render_qpos is not None
                    and isinstance(src.get("rule_grasp"),dict)
                    else _rule_grasp_status(state)
                )
            if _exact_render_qpos is not None:
                _apply_error=float(
                    np.max(np.abs(np.asarray(state.d.qpos)-_exact_render_qpos))
                )
                state._STATE_RENDER_MAX_APPLY_ERROR=max(
                    state._STATE_RENDER_MAX_APPLY_ERROR,_apply_error
                )
                if _apply_error>1e-12:
                    raise RuntimeError(
                        "exact render-state replay qpos drifted before capture: "
                        f"frame={state.t} max_abs_error={_apply_error:.17g}"
                    )
                state._STATE_RENDER_ROWS_APPLIED+=1
            pen=penetrations(state); fmax=max([p["depth_mm"] for p in pen],default=0.0)
            rec.update(frame_debug_metrics(state))
            add_success_metrics(state,rec)
            rec["penetration"]=pen; rec["max_pen_mm"]=round(fmax,3)
            _maybe_dump_record_state(state,rec)
            state.maxpen=max(state.maxpen,fmax); state.npen+=1 if pen else 0
            if state.logf: state.logf.write(json.dumps(rec)+"\n")
            state._RECS.append(rec)
            _capture_render_sample(state,state.t)
            if state.t%25==0: print(f"[sim] f{state.t}/{state.N} (state-log-replay)",flush=True)
            continue
        if state.A.ik_replay:                       # drive FR3 directly with the stored IK joints (no online solve)
            rec={"frame":int(state.t),"arms":{}}
            q_targets={}
            grip_cmds={
                "R":float(np.clip(state.IKGRIP[state.t,1],0,0.7854)),
                "L":float(np.clip(state.IKGRIP[state.t,0],0,0.7854)),
            }
            grip_intents=(_latch_policy_grip_intents(state,grip_cmds)
                          if state.A.grasp_mode=="rule" else None)
            for key in ("R","L"):
                q7=np.asarray(state.IK_JR[state.t] if key=="R" else state.IK_JL[state.t],float)
                _pin_arm_q(state,key,q7)
                q_targets[key]=q7.copy()
                # grip names: [0]=left finger, [1]=right finger
                if state.YUBI is None:
                    for fa in state.ARM[key]["fj"]: state.d.qpos[fa]=grip_cmds[key]
                    _set_grip_command(state,key,grip_cmds[key])
                prev=state._QPREV[key]; dqv=np.abs(q7-prev)*180.0/np.pi if prev is not None else np.zeros(7)
                dqmax=float(dqv.max()); jump=bool(state.A.jump_deg>0 and prev is not None and dqmax>state.A.jump_deg); state._QPREV[key]=q7.copy()
                rec["arms"][key]={"q":[round(float(x),5) for x in q7],"res_pos_mm":0.0,"res_ori_deg":0.0,"converged":True,
                                  "dqmax_deg":round(dqmax,2),"jump_joint":(int(dqv.argmax())+1 if prev is not None else 0),"jump":jump}
                if jump: state.njump+=1; print(f"[sim] WARNING: ik-replay JUMP {key} f{state.t}: joint{int(dqv.argmax())+1} Δ={dqmax:.1f}deg",flush=True)
            _kinematic_physics_substeps(state,q_targets,grip_cmds,frame=state.t,
                                        grip_intents=grip_intents)
            rec["grip"]={"R":round(float(grip_cmds["R"]),5),"L":round(float(grip_cmds["L"]),5)}
            rec["grip_actual"]={"R":round(_finger_actual(state,"R"),5),"L":round(_finger_actual(state,"L"),5)}
            if state.YUBI is not None: rec["yubi"]=_yubi_telemetry(state)
            if state.A.debug_grip_actuator: rec["grip_actuator"]=_grip_actuator_debug(state,grip_cmds)
            if state.A.debug_grasp_contact: rec["grasp_contact"]=gripper_cup_contacts(state,state.A.grasp_contact_tol_mm)
            if state.A.grasp_mode=="rule": rec["rule_grasp"]=_rule_grasp_status(state)
            pen=penetrations(state); fmax=max([p["depth_mm"] for p in pen],default=0.0)
            rec.update(frame_debug_metrics(state))
            add_success_metrics(state,rec)
            rec["penetration"]=pen; rec["max_pen_mm"]=round(fmax,3); state.maxpen=max(state.maxpen,fmax); state.npen+=1 if pen else 0
            _maybe_dump_record_state(state,rec)
            if state.logf: state.logf.write(json.dumps(rec)+"\n")
            state._RECS.append(rec)
            _capture_render_sample(state,state.t)
            if state.t%25==0: print(f"[sim] f{state.t}/{state.N} (ik-replay)",flush=True)
            continue
        if state.POLICY:
            st=state._pstate
            inferred_this_frame=False; activated_this_frame=False; captured_overlap_obs=False
            reuse_capture=None
            overlap_infer_due,overlap_activate_due=pio.rf_overlap_scheduler_flags(
                st["block_step"],exec_steps=state.A.exec,overlap_steps=state.A.rf_overlap,
            )
            if st["chunk"] is None:
                obs,obs_meta=_capture_next_policy_observation(state,state.t)
                _activate_policy_chunk(state,_infer_policy_chunk(state,obs,obs_meta,state.t))
                inferred_this_frame=True; activated_this_frame=True
                reuse_capture=(obs,obs_meta)
            elif state.A.rf_overlap and overlap_activate_due:
                inferred=st.get("pending_inferred")
                if inferred is None:
                    raise RuntimeError("RF overlap block ended without a precomputed midpoint inference")
                st["pending_inferred"]=None
                st["pending_obs"]=None; st["pending_obs_meta"]=None
                _activate_policy_chunk(state,inferred)
                activated_this_frame=True
            elif st["block_step"]>=state.A.exec:
                obs,obs_meta=_capture_next_policy_observation(state,state.t)
                _activate_policy_chunk(state,_infer_policy_chunk(state,obs,obs_meta,state.t))
                inferred_this_frame=True
            # Activation resets block_step to zero. Re-evaluate capture/inference
            # flags so overlap==exec can cache the new boundary OBS immediately.
            if state.A.rf_overlap:
                overlap_infer_due,_=pio.rf_overlap_scheduler_flags(
                    st["block_step"],exec_steps=state.A.exec,overlap_steps=state.A.rf_overlap,
                )
            if state.A.rf_overlap and overlap_infer_due:
                if st.get("pending_inferred") is not None:
                    raise RuntimeError("RF overlap midpoint reached with an inference already pending")
                # With full overlap, the initial inference and midpoint inference
                # are both due on frame zero. Reuse the immutable first capture so
                # wrist EGL is never invoked twice without the compatibility
                # overview render that follows each simulation sample.
                if reuse_capture is not None:
                    st["pending_obs"],st["pending_obs_meta"]=reuse_capture
                else:
                    st["pending_obs"],st["pending_obs_meta"]=_capture_next_policy_observation(state,state.t)
                conditioning_prefix,conditioning_rebase_meta=(
                    _prepare_rebased_conditioning_tail(state,st["pending_obs"])
                )
                st["pending_inferred"]=_infer_policy_chunk(
                    state,st["pending_obs"],st["pending_obs_meta"],state.t,conditioning_prefix,
                    conditioning_rebase_meta
                )
                inferred_this_frame=True
                captured_overlap_obs=True

            execution_step=int(st["block_step"])
            action_index=int(st["action_start"]+execution_step)
            a=st["chunk"][action_index]
            active_obs_meta=st.get("active_obs_meta") or {}
            pol_rec={"chunk_step":action_index,
                     "execution_step":execution_step,
                     "chunk_index":int(st["chunk_index"]),
                     "replan_boundary":bool(execution_step==0 and st["chunk_index"]>0),
                     "inferred_this_frame":bool(inferred_this_frame),
                     "activated_this_frame":bool(activated_this_frame),
                     "overlap_obs_captured":bool(captured_overlap_obs),
                     "rf_overlap":int(state.A.rf_overlap),
                     "rf_overlap_sampler":state.A.rf_overlap_sampler,
                     "rf_overlap_sampler_configured":state.A.rf_overlap_sampler,
                     "rf_overlap_sampler_effective":"single",
                     "rf_overlap_collapse_reason":None,
                     "rf_overlap_prev_weight":None,
                     "rf_overlap_prev_weight_configured":None,
                     "rf_overlap_prev_weight_effective":None,
                     "rf_overlap_prev_weight_dynamic":False,
                     "rf_overlap_prev_weight_schedule":None,
                     "rf_overlap_step_shift":int(state.A.rf_overlap_step_shift),
                     "rf_overlap_step_shift_configured":int(state.A.rf_overlap_step_shift),
                     "rf_overlap_step_shift_effective":0,
                     "rf_overlap_sample_num_steps":None,
                     "active_rf_overlap_sampler":st.get("active_sampler","single"),
                     "active_rf_overlap_prev_weight":st.get("active_prev_weight"),
                     "active_rf_overlap_prev_weight_dynamic":bool(
                         st.get("active_dynamic_prev_weight", False)
                     ),
                     "active_rf_overlap_prev_weight_schedule":st.get(
                         "active_prev_weight_schedule"
                     ),
                     "active_rf_overlap_step_shift":0,
                     "action_execution_slice":[int(st["action_start"]),int(st["action_start"]+state.A.exec)],
                     "image_source":state.A.policy_image_source,
                     "image_index":active_obs_meta.get("image_index"),
                     "image_raw_index":active_obs_meta.get("image_raw_index"),
                     "pose_source":state.A.policy_pose_source,
                     "pose_index":active_obs_meta.get("pose_index"),
                     "observation_frame":active_obs_meta.get("observation_frame"),
                     "inference_frame":st.get("active_inference_frame"),
                     "obs_log":st.get("active_obs_log"),
                     "infer_meta":st.get("policy_result_meta"),
                     "conditioned_chunk":bool(st.get("active_conditioned",False)),
                     "policy_delta_anchor":state.A.policy_delta_anchor,
                     "policy_pose_repr":state.A.policy_pose_repr,
                     "policy_state_quat_hemisphere":getattr(state.A,"policy_state_quat_hemisphere",None),
                     "target_anchor_frame":active_obs_meta.get("observation_frame"),
                     "prefix_roundtrip_max_abs":st.get("last_prefix_error"),
                     "bridge_roundtrip_max_abs":st.get("last_bridge_error"),
                     "bridge_rebased_prev_distance_max_abs":st.get(
                         "last_bridge_rebased_prev_distance"
                     ),
                     "conditioning_left_reference_rebase":st.get("conditioning_rebase_meta"),
                     "raw_grip_L":round(float(a[state.GI_L]),5),"raw_grip_R":round(float(a[state.GI_R]),5),
                     "dpos_L":_rlist(a[0:3]),"dpos_R":_rlist(a[7:10]),
                     "dquat_L":_rlist(a[3:7]),"dquat_R":_rlist(a[10:14])}
            if st.get("pending_obs_meta") is not None:
                pol_rec["pending_observation_frame"]=int(st["pending_obs_meta"]["observation_frame"])
            if st.get("pending_inferred") is not None:
                pol_rec["pending_chunk_index"]=int(st["pending_inferred"]["chunk_index"])
                pol_rec["pending_inference_frame"]=int(st["pending_inferred"]["inference_frame"])
            for side,off,gi in (("R",7,state.GI_R),("L",0,state.GI_L)):   # yubi action: pose L/R, then gripL/gripR
                if state.FR3_IMPEDANCE is not None:
                    state._grip[side]=float(np.clip(a[gi],0,0.7854))
                    continue
                if side=="L" and state.A.policy_freeze_left:
                    state.TGT[side][0]=state._READY_EE[side][0].copy(); state.TGT[side][1]=state._READY_EE[side][1].copy()
                    state._grip[side]=float(np.clip(a[gi],0,0.7854))
                    continue
                if state.A.policy_delta_anchor=="observation":
                    trajectory=st["target_trajectory"][side]
                    tp=trajectory["position"][action_index]
                    tq=trajectory["quaternion"][action_index]
                else:
                    p,q=state.ee_now(side); tp,tq=state.pol_apply(p,q,a,off)
                state.TGT[side][0]=np.asarray(tp,float)+state.T; state.TGT[side][1]=np.asarray(tq,float)
                state._grip[side]=float(np.clip(a[gi],0,0.7854))
                pol_rec[f"target_world_{side}"]=_rlist(state.TGT[side][0])
                pol_rec[f"target_quat_{side}"]=_rlist(state.TGT[side][1])   # world target quaternion (wxyz)
            pol_rec["applied_grip_L"]=round(float(state._grip["L"]),5); pol_rec["applied_grip_R"]=round(float(state._grip["R"]),5)
            pol_rec["relative_integrator"]=(
                "policy_io.compose_policy_anchored_trajectory:observation_chunk"
                if state.A.policy_pose_repr=="anchored"
                else (
                    "rel2abs.apply_delta_local:observation_chunk"
                    if state.A.policy_delta_anchor=="observation"
                    else "rel2abs.apply_delta_local:live_tcp"
                )
            )
            if state.FR3_IMPEDANCE is not None:
                state.FR3_IMPEDANCE.set_policy_actions(
                    a, state.FRAME_HZ_EFFECTIVE, freeze_left=state.A.policy_freeze_left)
                pol_rec["relative_integrator"] = "deploy_cmd_twist:persistent_equilibrium"
            if state.A.policy_freeze_left: pol_rec["freeze_left"]=True
            gR=state._grip["R"]; gL=state._grip["L"]
            st["block_step"]+=1
        elif state.A.grasp_test and state.GH == "both":
            _grips = {}
            for _hand, _script in state.grasp_scripts.items():
                p, q, g, ph = _script(state.t)
                state.TGT[_hand][0] = np.asarray(p, float) + state.T
                state.TGT[_hand][1] = np.asarray(q, float)
                _grips[_hand] = float(g)
            gR, gL = _grips["R"], _grips["L"]
            pol_rec = {"grasp_phase": ph, "grip_cmd": round(gR,5),
                       "grip_cmd_by_hand": {k: round(v,5) for k,v in _grips.items()}}
        elif state.A.grasp_test:
            p, q, g, ph = state.grasp_script(state.t)
            state.TGT[state.GH][0] = np.asarray(p, float) + state.T
            state.TGT[state.GH][1] = np.asarray(q, float)
            state.TGT[state.GH_OTHER][0] = state._P_PARK + state.T
            state.TGT[state.GH_OTHER][1] = state._GQ
            if state.GH == "R": gR, gL = g, state.A.close_grip
            else:         gR, gL = state.A.close_grip, g
            pol_rec = {"grasp_phase": ph, "grip_cmd": round(g,5)}
        else:
            dR,dL,gR,gL=state.step_action(state.t)
            # FR3 replay may place absolute poses or anchor the recorded
            # trajectory at the simulator ready pose.
            if os.environ.get("REPLAY_DELTA") or state.A.abs_frame:
                if state.t>0:
                    state._DATA["R"]=list(replay_integrate_relative_delta(state._DATA["R"][0],state._DATA["R"][1],dR))
                    state._DATA["L"]=list(replay_integrate_relative_delta(state._DATA["L"][0],state._DATA["L"][1],dL))
                srcR=np.r_[state._DATA["R"][0],state._DATA["R"][1]]; srcL=np.r_[state._DATA["L"][0],state._DATA["L"][1]]
            else:                                # legacy ready-anchored path: recorded ABSOLUTE pose
                srcR=state.absR[state.t]; srcL=state.absL[state.t]
            for side,a,a0 in (("R",srcR,state._A0R),("L",srcL,state._A0L)):
                if state.A.abs_frame:                  # EE pose = absolute frame-0 + integrated deltas (pos & rot both absolute)
                    state.TGT[side][0]=np.asarray(a[:3],float)
                    q=xy2wx(a[3:7]); q/=np.linalg.norm(q)+1e-9
                    state.TGT[side][1]=qmul(q,state.Q_CANON_TO_FRANKA)
                else:                            # legacy: anchor onto sim ready pose, apply only the recorded delta-from-frame0
                    state.TGT[side][0]=state._READY_EE[side][0]+(a[:3]-a0[:3])
                    dq=qmul(xy2wx(a[3:7]),qconj(xy2wx(a0[3:7]))); dq/=np.linalg.norm(dq)+1e-9
                    state.TGT[side][1]=qmul(dq,state._READY_EE[side][1])
                state.TGT[side][0]=state.TGT[side][0]+state.T      # --world-offset: add ONLY to the EE target position (world); base/desk/cup/GS FIXED -> IK re-solves
        if state.A.physics=="kinematic":
            rec={"frame":int(state.t),"arms":{},"grip":{"R":round(float(gR),5),"L":round(float(gL),5)}}
            if pol_rec is not None: rec["policy"]=pol_rec
            grip_cmds={"R":float(np.clip(gR,0,0.7854)),"L":float(np.clip(gL,0,0.7854))}
            grip_intents=(_latch_policy_grip_intents(state,grip_cmds)
                          if state.A.grasp_mode=="rule" else None)
            q_targets={}
            for key,gv in (("R",gR),("L",gL)):
                prev=state._QPREV[key]
                q7,res,conv,dqv,jump,jmeta=solve_ik_guarded(state,key,np.asarray(state.TGT[key][0]),np.asarray(state.TGT[key][1]),prev)
                _pin_arm_q(state,key,q7)
                q_targets[key]=q7.copy()
                if state.YUBI is None:
                    for fa in state.ARM[key]["fj"]: state.d.qpos[fa]=grip_cmds[key]
                    _set_grip_command(state,key,grip_cmds[key])
                # --- IK jump detection: max per-joint change vs previous frame (discontinuity / flip) ---
                dqmax=float(dqv.max())
                state._QPREV[key]=q7.copy()
                rec["arms"][key]={"q":[round(float(x),5) for x in q7],"res_pos_mm":round(res[0]*1000,3),
                                  "res_ori_deg":round(np.degrees(res[1]),3),"converged":bool(conv),
                                  "dqmax_deg":round(dqmax,2),"jump_joint":(int(dqv.argmax())+1 if prev is not None else 0),
                                  "jump":jump,"raw_jump":bool(jmeta["raw_jump"]),"held":bool(jmeta["held"]),
                                  "raw_dqmax_deg":round(float(jmeta["raw_dqmax_deg"]),2),"raw_jump_joint":int(jmeta["raw_jump_joint"]),
                                  "jump_retries":int(jmeta["retries"])}
                state.nfail+=0 if conv else 1
                if jump:
                    state.njump+=1
                    print(f"[sim] WARNING: IK JUMP {key} f{state.t}: joint{int(dqv.argmax())+1} Δ={dqmax:.1f}deg (>{state.A.jump_deg}deg)",flush=True)
                elif jmeta["held"]:
                    print(f"[sim] held IK jump {key} f{state.t}: kept previous q after {jmeta['retries']} retry(s)",flush=True)
            _kinematic_physics_substeps(state,q_targets,grip_cmds,frame=state.t,
                                        grip_intents=grip_intents)
            forced_replay_objects = _force_replay_object_poses(state,state.t)
            if forced_replay_objects:
                rec["replay_object_source"] = "parquet"
                if state.A.grasp_mode == "rule":
                    _rule_grasp_update(state,grip_cmds, frame=state.t,
                                       grip_intents=grip_intents)
            rec["grip_actual"]={"R":round(_finger_actual(state,"R"),5),"L":round(_finger_actual(state,"L"),5)}
            if state.YUBI is not None: rec["yubi"]=_yubi_telemetry(state)
            if state.A.debug_grip_actuator: rec["grip_actuator"]=_grip_actuator_debug(state,grip_cmds)
            if state.A.debug_grasp_contact: rec["grasp_contact"]=gripper_cup_contacts(state,state.A.grasp_contact_tol_mm)
            if state.A.grasp_mode=="rule": rec["rule_grasp"]=_rule_grasp_status(state)
            pen=penetrations(state); fmax=max([p["depth_mm"] for p in pen],default=0.0)
            rec.update(frame_debug_metrics(state))
            add_success_metrics(state,rec)
            rec["penetration"]=pen; rec["max_pen_mm"]=round(fmax,3); state.maxpen=max(state.maxpen,fmax); state.npen+=1 if pen else 0
            _maybe_dump_record_state(state,rec)
            state.logf.write(json.dumps(rec)+"\n"); state._RECS.append(rec)
        else:
            state._dynamic_grip_samples=[]
            _impedance_frame_samples = []
            grip_cmds={"R":float(np.clip(gR,0,0.7854)),
                       "L":float(np.clip(gL,0,0.7854))}
            grip_intents=(_latch_policy_grip_intents(state,grip_cmds)
                          if state.A.grasp_mode=="rule" else None)
            if state.DYNAMIC_YUBI_HOOK is not None:
                state.DYNAMIC_YUBI_HOOK.set_targets(grip_cmds,t=float(state.d.time))
            if state.EMBODIMENT=="fr3":
                if state.FR3_STRATEGY.uses_joint_ik:
                    for _key in ("R","L"):
                        _joint_trajectory=state._JOINT_IK_TRAJECTORY[_key]
                        _ik_target_position=np.asarray(
                            state.TGT[_key][0],dtype=np.float64
                        )
                        _ik_target_quaternion=np.asarray(
                            state.TGT[_key][1],dtype=np.float64
                        )
                        _joint_seed=_joint_trajectory.position.copy()
                        _joint_result=state._ABSOLUTE_JOINT_IK.solve(
                            _key,
                            _ik_target_position,
                            _ik_target_quaternion,
                            seed_q=_joint_seed,
                            continuity_q=_joint_seed,
                        )
                        state._JOINT_IK_SOLVE_COUNT+=1
                        if not _joint_result.converged:
                            state._JOINT_IK_FAIL_COUNT+=1
                            state.nfail+=1
                        _joint_goal=np.asarray(_joint_result.q,dtype=np.float64)
                        _joint_goal_finite=bool(np.all(np.isfinite(_joint_goal)))
                        if not _joint_goal_finite:
                            _joint_goal=_joint_seed.copy()
                            state._JOINT_IK_FAIL_COUNT+=1
                            state.nfail+=1
                        _joint_goal=np.clip(
                            _joint_goal,
                            np.asarray(state.ARM[_key]["jr"][:,0],dtype=np.float64),
                            np.asarray(state.ARM[_key]["jr"][:,1],dtype=np.float64),
                        )
                        _safety_decision={
                            "decision":"full_target",
                            "workspace_decision":"ungated",
                            "fail_hold_enabled":bool(state.A.joint_ik_fail_hold),
                            "collision_gate_enabled":bool(
                                state._ARM_DESK_COLLISION_GATE is not None
                            ),
                            "raw_q":[
                                round(float(_value),6) for _value in _joint_goal
                            ],
                            "selected_q":[
                                round(float(_value),6) for _value in _joint_goal
                            ],
                            "continued_active_target_q":None,
                            "selected_fraction":1.0,
                            "workspace_filter_reason":None,
                            "raw_min_desk_clearance_mm":None,
                            "selected_min_desk_clearance_mm":None,
                            "trajectory_preview_min_desk_clearance_mm":None,
                            "trajectory_preview_duration_s":None,
                            "trajectory_preview_samples":0,
                            "trajectory_preview_first_unsafe_index":None,
                            "trajectory_preview_first_unsafe_clearance_mm":None,
                            "trajectory_preview_first_unsafe_arm_geom":None,
                            "trajectory_preview_first_unsafe_desk_geom":None,
                            "closest_arm_geom":None,
                            "closest_desk_geom":None,
                            "selected_closest_arm_geom":None,
                            "selected_closest_desk_geom":None,
                            "blocked_fraction":None,
                            "blocked_arm_geom":None,
                            "blocked_desk_geom":None,
                            "ignored_zero_distance_pairs":0,
                            "scratch_collision_checks":0,
                        }
                        if not _joint_goal_finite:
                            _target_update=_joint_trajectory.reject_target(
                                _joint_goal,
                                reason="ik_nonfinite_hold",
                            )
                            _safety_decision["decision"]="ik_nonfinite_hold"
                            _safety_decision["workspace_decision"]="not_evaluated"
                            state._JOINT_IK_SAFETY_STATS[_key][
                                "ik_nonconverged_hold"
                            ]+=1
                        elif (
                            bool(state.A.joint_ik_fail_hold)
                            and not _joint_result.converged
                        ):
                            _ik_failure_reason=(
                                _joint_result.fallback_reason or "solver_failure"
                            )
                            _target_update=_joint_trajectory.reject_target(
                                _joint_goal,
                                reason=f"ik_nonconverged:{_ik_failure_reason}",
                            )
                            _safety_decision["decision"]="ik_nonconverged_hold"
                            _safety_decision["workspace_decision"]="not_evaluated"
                            state._JOINT_IK_SAFETY_STATS[_key][
                                "ik_nonconverged_hold"
                            ]+=1
                        elif state._ARM_DESK_COLLISION_GATE is not None:
                            _safe_anchor=np.asarray(
                                _joint_trajectory.position,dtype=np.float64
                            )
                            _maximum_target_delta=float(np.max(
                                np.abs(_joint_goal-_safe_anchor),initial=0.0
                            ))
                            _path_samples=max(
                                1,
                                int(np.ceil(
                                    _maximum_target_delta/np.radians(2.0)
                                )),
                            )
                            _collision_filter=state._ARM_DESK_COLLISION_GATE.filter_target(
                                _key,
                                _joint_goal,
                                _safe_anchor,
                                base_qpos=np.asarray(state.d.qpos,dtype=np.float64),
                                path_samples=_path_samples,
                                bisection_iterations=8,
                                retreat_rad=float(np.radians(0.25)),
                            )
                            _raw_clearance_mm=(
                                float(
                                    _collision_filter
                                    .requested_check.minimum_distance_m
                                )*1000.0
                            )
                            _selected_clearance_mm=(
                                float(
                                    _collision_filter
                                    .selected_check.minimum_distance_m
                                )*1000.0
                            )
                            _safety_decision.update({
                                "workspace_decision":(
                                    "collision_clipped"
                                    if _collision_filter.clipped else
                                    "full_target"
                                    if _collision_filter.accepted else
                                    "collision_hold"
                                ),
                                "workspace_filter_reason":(
                                    _collision_filter.reason
                                ),
                                "selected_q":[
                                    round(float(_value),6)
                                    for _value in _collision_filter.selected_target
                                ],
                                "selected_fraction":round(
                                    float(_collision_filter.selected_fraction),6
                                ),
                                "raw_min_desk_clearance_mm":round(
                                    _raw_clearance_mm,6
                                ),
                                "selected_min_desk_clearance_mm":round(
                                    _selected_clearance_mm,6
                                ),
                                "closest_arm_geom":(
                                    _collision_filter
                                    .requested_check.arm_geom_name
                                ),
                                "closest_desk_geom":(
                                    _collision_filter
                                    .requested_check.desk_geom_name
                                ),
                                "selected_closest_arm_geom":(
                                    _collision_filter
                                    .selected_check.arm_geom_name
                                ),
                                "selected_closest_desk_geom":(
                                    _collision_filter
                                    .selected_check.desk_geom_name
                                ),
                                "blocked_fraction":(
                                    round(
                                        float(_collision_filter.blocked_fraction),6
                                    )
                                    if _collision_filter.blocked_fraction is not None
                                    else None
                                ),
                                "blocked_arm_geom":(
                                    _collision_filter.blocked_check.arm_geom_name
                                    if _collision_filter.blocked_check is not None
                                    else None
                                ),
                                "blocked_desk_geom":(
                                    _collision_filter.blocked_check.desk_geom_name
                                    if _collision_filter.blocked_check is not None
                                    else None
                                ),
                                "ignored_zero_distance_pairs":int(
                                    _collision_filter
                                    .requested_check.ignored_zero_distance_pairs
                                    +_collision_filter
                                    .selected_check.ignored_zero_distance_pairs
                                ),
                                "scratch_collision_checks":int(
                                    _collision_filter.collision_checks
                                ),
                            })
                            _safety_stats=state._JOINT_IK_SAFETY_STATS[_key]
                            _safety_stats["scratch_collision_checks"]+=int(
                                _collision_filter.collision_checks
                            )
                            _safety_stats["ignored_zero_distance_pairs"]+=int(
                                _safety_decision["ignored_zero_distance_pairs"]
                            )
                            _safety_stats["raw_min_clearance_mm"]=min(
                                float(_safety_stats["raw_min_clearance_mm"]),
                                _raw_clearance_mm,
                            )
                            _safety_stats["selected_min_clearance_mm"]=min(
                                float(_safety_stats["selected_min_clearance_mm"]),
                                _selected_clearance_mm,
                            )
                            if not _collision_filter.requested_check.safe:
                                _safety_stats["collision_endpoint"]+=1
                            if _collision_filter.accepted:
                                _trajectory_preview,_preview_checks=(
                                    _preview_joint_trajectory_clearance(state,
                                        _key,
                                        _collision_filter.selected_target,
                                    )
                                )
                                _preview_check_count=len(_preview_checks)
                                _safety_stats["trajectory_preview_checks"]+=int(
                                    _preview_check_count
                                )
                                _safety_stats["scratch_collision_checks"]+=int(
                                    _preview_check_count
                                )
                                _safety_decision[
                                    "trajectory_preview_samples"
                                ]=int(_preview_check_count)
                                _safety_decision[
                                    "trajectory_preview_duration_s"
                                ]=(
                                    round(
                                        float(
                                            _trajectory_preview.update
                                            .trajectory_duration_s
                                        ),
                                        6,
                                    )
                                    if (
                                        _trajectory_preview.update
                                        .trajectory_duration_s is not None
                                    )
                                    else None
                                )
                                _safety_decision["scratch_collision_checks"]+=int(
                                    _preview_check_count
                                )
                                _preview_min_check=(
                                    min(
                                        _preview_checks,
                                        key=lambda _check:(
                                            _check.minimum_distance_m
                                        ),
                                    )
                                    if _preview_checks else None
                                )
                                if _preview_min_check is not None:
                                    _preview_min_clearance_mm=(
                                        float(
                                            _preview_min_check.minimum_distance_m
                                        )*1000.0
                                    )
                                    _safety_decision[
                                        "trajectory_preview_min_desk_clearance_mm"
                                    ]=round(_preview_min_clearance_mm,6)
                                    _safety_stats[
                                        "trajectory_preview_min_clearance_mm"
                                    ]=min(
                                        float(_safety_stats[
                                            "trajectory_preview_min_clearance_mm"
                                        ]),
                                        _preview_min_clearance_mm,
                                    )
                                    _preview_ignored_zero_pairs=sum(
                                        int(
                                            _check
                                            .ignored_zero_distance_pairs
                                        )
                                        for _check in _preview_checks
                                    )
                                    _safety_decision[
                                        "ignored_zero_distance_pairs"
                                    ]+=_preview_ignored_zero_pairs
                                    _safety_stats[
                                        "ignored_zero_distance_pairs"
                                    ]+=_preview_ignored_zero_pairs
                                _first_unsafe_preview=next(
                                    (
                                        (_index,_check)
                                        for _index,_check in enumerate(
                                            _preview_checks
                                        )
                                        if not _check.safe
                                    ),
                                    None,
                                )
                                if (
                                    _trajectory_preview.update.accepted
                                    and _first_unsafe_preview is None
                                ):
                                    _target_update=_joint_trajectory.set_target(
                                        _collision_filter.selected_target
                                    )
                                elif not _trajectory_preview.update.accepted:
                                    _target_update=_joint_trajectory.set_target(
                                        _collision_filter.selected_target
                                    )
                                else:
                                    _unsafe_index,_unsafe_check=(
                                        _first_unsafe_preview
                                    )
                                    _safety_stats[
                                        "trajectory_preview_collision"
                                    ]+=1
                                    _safety_decision[
                                        "trajectory_preview_first_unsafe_index"
                                    ]=int(_unsafe_index)
                                    _safety_decision[
                                        "trajectory_preview_first_unsafe_clearance_mm"
                                    ]=round(
                                        float(_unsafe_check.minimum_distance_m)
                                        *1000.0,
                                        6,
                                    )
                                    _safety_decision[
                                        "trajectory_preview_first_unsafe_arm_geom"
                                    ]=_unsafe_check.arm_geom_name
                                    _safety_decision[
                                        "trajectory_preview_first_unsafe_desk_geom"
                                    ]=_unsafe_check.desk_geom_name
                                    # Every accepted active trajectory has already
                                    # passed a full-duration clearance preview. Keep
                                    # its validated remainder exactly as-is. A new
                                    # Ruckig target at the current position is not a
                                    # true hold when velocity is non-zero: it can
                                    # overshoot into the desk clearance shell before
                                    # returning to the same position.
                                    _target_update=_joint_trajectory.reject_target(
                                        _collision_filter.selected_target,
                                        reason="trajectory_preview_collision_hold",
                                    )
                                    _safety_decision["decision"]=(
                                        "trajectory_collision_hold"
                                    )
                                    _safety_decision["continued_active_target_q"]=[
                                        round(float(_value),6)
                                        for _value in _joint_trajectory.active_target
                                    ]
                                    _safety_stats[
                                        "trajectory_collision_hold"
                                    ]+=1
                                if _collision_filter.clipped:
                                    if (
                                        _safety_decision["decision"]
                                        =="full_target"
                                    ):
                                        _safety_decision["decision"]=(
                                            "collision_clipped"
                                        )
                            else:
                                _target_update=_joint_trajectory.reject_target(
                                    _joint_goal,
                                    reason=(
                                        "arm_desk_collision:"
                                        f"{_collision_filter.reason}"
                                    ),
                                )
                                _safety_stats["collision_hold"]+=1
                                _safety_decision["decision"]="collision_hold"
                        else:
                            _target_update=_joint_trajectory.set_target(_joint_goal)
                        state._JOINT_IK_TARGET_ATTEMPT_COUNT[_key]+=1
                        if _target_update.accepted:
                            _accepted_decision=str(_safety_decision["decision"])
                            if _accepted_decision=="full_target":
                                state._JOINT_IK_SAFETY_STATS[_key]["full_target"]+=1
                            elif _accepted_decision=="collision_clipped":
                                state._JOINT_IK_SAFETY_STATS[_key][
                                    "collision_clipped"
                                ]+=1
                        else:
                            if not str(_safety_decision["decision"]).endswith("hold"):
                                _safety_decision["decision"]="ruckig_reject"
                                state._JOINT_IK_SAFETY_STATS[_key]["ruckig_reject"]+=1
                            state._JOINT_IK_TARGET_REJECT_COUNT+=1
                            _reject_reason=(
                                _target_update.reason or "unspecified_rejection"
                            )
                            state._JOINT_IK_TARGET_REJECT_REASONS[_key][_reject_reason]=(
                                state._JOINT_IK_TARGET_REJECT_REASONS[_key].get(
                                    _reject_reason,0
                                )+1
                            )
                            if (
                                state._JOINT_IK_TARGET_REJECT_COUNT<=10
                                or state._JOINT_IK_TARGET_REJECT_COUNT%100==0
                            ):
                                print(
                                    f"[sim] joint_ik safe-target reject {_key} f{state.t}: "
                                    f"{_target_update.reason}; continuing previous "
                                    "validated jerk-limited trajectory",
                                    flush=True,
                                )
                        state._JOINT_IK_LAST_RESULT[_key]=_joint_result
                        state._JOINT_IK_LAST_TARGET_UPDATE[_key]=_target_update
                        state._JOINT_IK_LAST_SAFETY_DECISION[_key]=_safety_decision
                for _rep in range(_frame_control_repeats):
                    _joint_command_sample=None
                    if state.FR3_STRATEGY.uses_joint_ik:
                        _joint_commands={}
                        _joint_command_sample={}
                        for _key in ("R","L"):
                            _trajectory_sample=(
                                state._JOINT_IK_TRAJECTORY[_key].step()
                            )
                            _joint_command=_trajectory_sample.position
                            state._JOINT_IK_LAST_SAMPLE[_key]=_trajectory_sample
                            _joint_commands[_key]=_joint_command
                            _previous_command=state._JOINT_IK_COMMAND[_key]
                            _discrete_command_velocity=(
                                (_joint_command-_previous_command)
                                * float(state.env.control_freq)
                            )
                            _previous_acceleration=(
                                state._JOINT_IK_COMMAND_ACCELERATION[_key]
                            )
                            _command_velocity=_trajectory_sample.velocity
                            _command_acceleration=_trajectory_sample.acceleration
                            _command_jerk=_trajectory_sample.jerk
                            _average_command_jerk=(
                                (_command_acceleration-_previous_acceleration)
                                * float(state.env.control_freq)
                            )
                            state._JOINT_IK_COMMAND[_key]=_joint_command.copy()
                            state._JOINT_IK_COMMAND_VELOCITY[_key]=_command_velocity.copy()
                            state._JOINT_IK_COMMAND_ACCELERATION[_key]=_command_acceleration.copy()
                            _target_update=state._JOINT_IK_LAST_TARGET_UPDATE[_key]
                            _joint_command_sample[_key]={
                                "q_command":[round(float(x),6) for x in _joint_command],
                                "q_velocity":[
                                    round(float(x),6) for x in _command_velocity
                                ],
                                "q_acceleration":[
                                    round(float(x),6) for x in _command_acceleration
                                ],
                                "q_jerk":[
                                    round(float(x),6) for x in _command_jerk
                                ],
                                "velocity_max_rad_s":round(
                                    float(np.max(np.abs(_command_velocity))),6
                                ),
                                "acceleration_max_rad_s2":round(
                                    float(np.max(np.abs(_command_acceleration))),6
                                ),
                                "jerk_max_rad_s3":round(
                                    float(np.max(np.abs(_command_jerk))),6
                                ),
                                "average_jerk_max_rad_s3":round(
                                    float(np.max(np.abs(_average_command_jerk))),6
                                ),
                                "discrete_position_step_velocity_max_rad_s":round(
                                    float(np.max(np.abs(
                                        _discrete_command_velocity
                                    ))),6
                                ),
                                "target_accepted":bool(_target_update.accepted),
                                "target_rejection_reason":_target_update.reason,
                                "target_safety":dict(
                                    state._JOINT_IK_LAST_SAFETY_DECISION[_key]
                                ),
                                "requested_target":[
                                    round(float(x),6)
                                    for x in _target_update.requested_target
                                ],
                                "active_target":[
                                    round(float(x),6)
                                    for x in _trajectory_sample.active_target
                                ],
                                "requested_trajectory_duration_s":(
                                    round(
                                        float(
                                            _target_update.trajectory_duration_s
                                        ),6
                                    )
                                    if _target_update.trajectory_duration_s
                                    is not None else None
                                ),
                                "requested_trajectory_result_code":(
                                    int(_target_update.result_code)
                                    if _target_update.result_code is not None
                                    else None
                                ),
                                "requested_trajectory_position_min":(
                                    _rlist(
                                        _target_update.predicted_position_min,6
                                    )
                                    if _target_update.predicted_position_min
                                    is not None else None
                                ),
                                "requested_trajectory_position_max":(
                                    _rlist(
                                        _target_update.predicted_position_max,6
                                    )
                                    if _target_update.predicted_position_max
                                    is not None else None
                                ),
                                "active_trajectory_duration_s":round(
                                    float(_trajectory_sample.trajectory_duration_s),6
                                ),
                                "active_trajectory_new_calculation":bool(
                                    _trajectory_sample.new_calculation
                                ),
                                "active_trajectory_finished":bool(
                                    _trajectory_sample.finished
                                ),
                                "active_trajectory_result_code":int(
                                    _trajectory_sample.result_code
                                ),
                                "active_trajectory_position_min":[
                                    round(float(x),6)
                                    for x in _trajectory_sample.predicted_position_min
                                ],
                                "active_trajectory_position_max":[
                                    round(float(x),6)
                                    for x in _trajectory_sample.predicted_position_max
                                ],
                            }
                        act=_dynamic_env_action(state,_joint_commands,grip_cmds)
                    else:
                        # Arm outputs are overwritten by the physics-rate impedance hook.
                        act=_dynamic_env_action(state,
                            {k: state.d.qpos[state.ARM[k]["qi"]].copy() for k in ("R", "L")}, grip_cmds)
                    state.env.step(act)
                    if state.FR3_IMPEDANCE is not None:
                        _impedance_frame_samples.extend(state.FR3_IMPEDANCE.drain_samples())
                    tick_success_criterion(state)
                    if state._FR3_DYNAMIC_SOURCE_HZ is not None:
                        state._FR3_DYNAMIC_ENV_STEP_COUNT+=1
                    _cbpc_microsteps=(
                        state.DYNAMIC_YUBI_HOOK.drain_samples() if state.DYNAMIC_YUBI_HOOK is not None else []
                    )
                    if state.A.grasp_mode=="rule":
                        _rule_grasp_update(state,grip_cmds, frame=state.t, substep=_rep,
                                           grip_intents=grip_intents)
                    _record_dynamic_sample(state,
                        _rep,
                        _joint_command_sample,
                        _cbpc_microsteps,
                    )
            forced_replay_objects = _force_replay_object_poses(state,state.t)
            mujoco.mj_forward(state.m,state.d)
            if state.logf:
                _dynamic_arms={}
                for key in ("R","L"):
                    q7=state.d.qpos[state.ARM[key]["qi"]].copy()
                    prev=state._QPREV[key]
                    dqv=(np.abs(q7-prev)*180.0/np.pi
                         if prev is not None else np.zeros(7))
                    dqmax=float(dqv.max())
                    jump=bool(state.A.jump_deg>0 and prev is not None and dqmax>state.A.jump_deg)
                    state._QPREV[key]=q7.copy()
                    res=policy_tcp_error(state,
                        key,np.asarray(state.TGT[key][0],float),np.asarray(state.TGT[key][1],float)
                    )
                    joint_vel=np.asarray(state.d.qvel[state.ARM[key]["dof"]],float)
                    joint_acc=np.asarray(state.d.qacc[state.ARM[key]["dof"]],float)
                    _dynamic_arms[key]={
                        "q":[round(float(x),5) for x in q7],
                        "res_pos_mm":round(float(res[0])*1000.0,3),
                        "res_ori_deg":round(float(np.degrees(res[1])),3),
                        "converged":bool(res[0]<=0.005 and res[1]<=np.radians(5.0)),
                        "dqmax_deg":round(dqmax,2),
                        "jump_joint":(int(dqv.argmax())+1 if prev is not None else 0),
                        "jump":jump,
                        "raw_jump":jump,
                        "held":False,
                        "raw_dqmax_deg":round(dqmax,2),
                        "raw_jump_joint":(int(dqv.argmax())+1 if prev is not None else 0),
                        "jump_retries":0,
                        "joint_velocity_rad_s":[round(float(x),5) for x in joint_vel],
                        "joint_acceleration_rad_s2":[round(float(x),5) for x in joint_acc],
                        "joint_velocity_max_rad_s":round(float(np.max(np.abs(joint_vel))),5),
                        "joint_acceleration_max_rad_s2":round(float(np.max(np.abs(joint_acc))),5),
                    }
                    # The FR3 joint_ik telemetry is only produced by the FR3
                    # branch above.
                    if state.EMBODIMENT=="fr3" and state.FR3_STRATEGY.uses_joint_ik:
                        _ik_result=state._JOINT_IK_LAST_RESULT[key]
                        _target_update=state._JOINT_IK_LAST_TARGET_UPDATE[key]
                        _trajectory_sample=state._JOINT_IK_LAST_SAMPLE[key]
                        _condition=float(_ik_result.condition_number)
                        _dynamic_arms[key]["absolute_ik"]={
                            "converged":bool(_ik_result.converged),
                            "iterations":int(_ik_result.iterations),
                            "position_error_mm":round(float(_ik_result.pos_error_mm),4),
                            "orientation_error_deg":round(float(_ik_result.ori_error_deg),4),
                            "min_limit_margin_rad":round(
                                float(_ik_result.min_limit_margin_rad),6
                            ),
                            "condition_number":(
                                round(_condition,4) if np.isfinite(_condition) else None
                            ),
                            "fallback_reason":_ik_result.fallback_reason,
                            "q_goal":[round(float(x),6) for x in _ik_result.q],
                        }
                        _dynamic_arms[key]["joint_command"]={
                            "q":[round(float(x),6) for x in state._JOINT_IK_COMMAND[key]],
                            "q_velocity":[
                                round(float(x),6)
                                for x in state._JOINT_IK_COMMAND_VELOCITY[key]
                            ],
                            "q_acceleration":[
                                round(float(x),6)
                                for x in state._JOINT_IK_COMMAND_ACCELERATION[key]
                            ],
                            "requested_target":[
                                round(float(x),6)
                                for x in _target_update.requested_target
                            ],
                            "active_target":[
                                round(float(x),6)
                                for x in _trajectory_sample.active_target
                            ],
                            "target_accepted":bool(_target_update.accepted),
                            "target_rejection_reason":_target_update.reason,
                            "target_safety":dict(
                                state._JOINT_IK_LAST_SAFETY_DECISION[key]
                            ),
                            "requested_trajectory_duration_s":(
                                round(
                                    float(
                                        _target_update.trajectory_duration_s
                                    ),6
                                )
                                if _target_update.trajectory_duration_s
                                is not None else None
                            ),
                            "requested_trajectory_result_code":(
                                int(_target_update.result_code)
                                if _target_update.result_code is not None
                                else None
                            ),
                            "requested_trajectory_position_min":(
                                _rlist(
                                    _target_update.predicted_position_min,6
                                )
                                if _target_update.predicted_position_min
                                is not None else None
                            ),
                            "requested_trajectory_position_max":(
                                _rlist(
                                    _target_update.predicted_position_max,6
                                )
                                if _target_update.predicted_position_max
                                is not None else None
                            ),
                            "active_trajectory_duration_s":round(
                                float(_trajectory_sample.trajectory_duration_s),6
                            ),
                            "active_trajectory_finished":bool(
                                _trajectory_sample.finished
                            ),
                            "active_trajectory_position_min":[
                                round(float(x),6)
                                for x in _trajectory_sample.predicted_position_min
                            ],
                            "active_trajectory_position_max":[
                                round(float(x),6)
                                for x in _trajectory_sample.predicted_position_max
                            ],
                            "tracking_error_max_rad":round(
                                float(np.max(np.abs(
                                    state._JOINT_IK_COMMAND[key]-q7
                                ))),6
                            ),
                        }
                rec={"frame":int(state.t),
                     "arms":_dynamic_arms,
                     "grip":{"R":round(float(gR),5),"L":round(float(gL),5)}}
                rec["source_action_control_ticks"]=int(_frame_control_repeats)
                if pol_rec is not None:
                    rec["policy"]=pol_rec
                if forced_replay_objects:
                    rec["replay_object_source"]="parquet"
                rec["grip_actual"]={"R":round(_finger_actual(state,"R"),5),"L":round(_finger_actual(state,"L"),5)}
                rec["grip_actual_fingers"]={key:{side:round(value,6)
                                                    for side,value in _finger_angles(state,key).items()}
                                                for key in ("R","L")}
                rec["dynamic_grip_samples"]=state._dynamic_grip_samples
                if state.YUBI is not None:
                    rec["yubi"]=_yubi_telemetry(state)
                if state.A.debug_grip_actuator:
                    rec["grip_actuator"]=_grip_actuator_debug(state,grip_cmds)
                if state.A.debug_grasp_contact:
                    rec["grasp_contact"]=gripper_cup_contacts(state,state.A.grasp_contact_tol_mm)
                if state.A.grasp_mode=="rule":
                    rec["rule_grasp"]=_rule_grasp_status(state)
                pen=penetrations(state); fmax=max([p["depth_mm"] for p in pen],default=0.0)
                rec.update(frame_debug_metrics(state))
                add_success_metrics(state,rec)
                rec["penetration"]=pen; rec["max_pen_mm"]=round(fmax,3)
                _maybe_dump_record_state(state,rec)
                state.maxpen=max(state.maxpen,fmax); state.npen+=1 if pen else 0
                log_rec=({**rec,"fr3_impedance_microsteps":_impedance_frame_samples}
                         if _impedance_frame_samples else rec)
                state.logf.write(json.dumps(log_rec)+"\n"); state._RECS.append(rec)
        if not (state.A.physics=="dynamic" and state.DYNAMIC_SUBFRAMES>1):
            _capture_render_sample(state,state.t)
        if state.t%25==0: print(f"[sim] f{state.t}/{state.N} ({'no-render simulation' if state.A.no_render else 'pass1'})",flush=True)
        _success_stop=_success_exit_after_frame(state,state.t)
        if _success_stop is not None:
            state.SUCCESS_STOPPED=True
            state.N=state.t+1               # truncate: pass2/video/summary all read N; lists are appended
            print(
                f"[sim] success first seen at frame {_success_stop}: stopping at frame "
                f"{state.t} after {state.A.success_exit_grace_frames} grace frames "
                f"({state.N} frames done)",
                flush=True,
            )
            break
        if state.A.max_wall_sec>0 and (time.time()-state._tloop)>state.A.max_wall_sec:
            state.WALL_STOPPED=True; state.N=state.t+1               # truncate: pass2/video/summary all read N; lists are appended
            print(f"[sim] --max-wall-sec {state.A.max_wall_sec:g}s reached at frame {state.t}: stopping rollout ({state.N} frames done)",flush=True)
            break

    if state.A.policy_batch_server and state.policy is not None:
        # Reaching the end of pass 1 is a clean terminal state. Retire this
        # client's strict-batch slot before the remaining workers infer again;
        # exceptions never reach this path and surface to the server as EOF.
        state.policy.close()

    # ============================ PASS 2: isolated overview EGL, then stream ============================
    # Render every deferred overview while one GPU lifecycle lock remains held,
    # then stream composition and MP4 encoding from disk outside that lock.
    state._render_count=(
        0
        if state.A.no_render
        else (
            len(state._OVERVIEW_STATE_FRAMES)
            if state._DEFER_OVERVIEW_RENDER
            else len(state._OVERVIEW_VIDEO_FRAMES)
        )
    )
    state._VIDEO_NEEDS_WRISTS=bool(not state.A.no_render and not state.A.clean_video and not state.A.no_wrist)


__all__ = ["run"]
