# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Policy client binding, observation logging, and chunk lifecycle."""

import numpy as np

from robosuite_env.robots.fr3.controllers import fr3_uses_cmd_twist_torque
from robosuite_env.policy import policy_io as pio
from robosuite_env.rollout.control.pose_math import policy_tcp_pose
from robosuite_env.rollout.observe.wrist import (
    capture_policy_observation as _capture_policy_observation,
)
from robosuite_env.rollout.observe.obs_log import (
    save_policy_obs_before_infer as _save_policy_obs_before_infer,
)
from robosuite_env.rollout.policy_chunks import (
    build_policy_chunk_result,
    build_policy_plan_record,
)


def policy_state_quat_hemisphere_metadata(state):
    """Return ``(source, metadata)`` of the policy whose norm stats this worker uses."""

    if state.A.policy_batch_server:
        return "policy_batch_server", dict(getattr(state.policy, "metadata", None) or {})
    from robosuite_env.policy.policy_batch_infer import policy_load_metadata

    return "loaded_checkpoint", policy_load_metadata(state.policy)


def verify_policy_state_quat_hemisphere(requested, source, metadata):
    """Refuse a worker whose state quaternion hemisphere disagrees with the policy's.

    The launcher resolves the hemisphere from ``policy.checkpoint``; the policy
    (batch server or in-process load) resolves it again from the norm stats it
    actually loaded and publishes the result. A worker that would send the other
    sign, or a server that publishes no resolution, is refused.
    """

    if requested not in pio.POLICY_STATE_QUAT_HEMISPHERES:
        raise RuntimeError(
            "--policy-state-quat-hemisphere must be one of "
            f"{pio.POLICY_STATE_QUAT_HEMISPHERES}, got {requested!r}"
        )
    published = metadata.get("policy_state_quat_hemisphere")
    resolution = metadata.get("policy_state_quat_hemisphere_resolution")
    if (
        published not in pio.POLICY_STATE_QUAT_HEMISPHERES
        or not isinstance(resolution, dict)
        or resolution.get("hemisphere") != published
    ):
        raise RuntimeError(
            f"{source} did not publish a valid policy_state_quat_hemisphere "
            f"(got {published!r}); it predates the hemisphere contract or could "
            "not resolve its norm stats. Refusing to run."
        )
    if published != requested:
        raise RuntimeError(
            "state quaternion hemisphere mismatch: this worker was started with "
            f"--policy-state-quat-hemisphere {requested} but the {source} norm "
            f"stats {resolution.get('norm_stats_path')} (sha256 "
            f"{resolution.get('norm_stats_sha256')}, qw q01 "
            f"{resolution.get('qw_q01')}, q99 {resolution.get('qw_q99')}) resolve "
            f"to {published}. Refusing to run."
        )
    record = {
        "hemisphere": requested,
        "verified": True,
        "verified_against": source,
    }
    for key in (
        "classification",
        "rule",
        "norm_stats_path",
        "norm_stats_sha256",
        "qw_index",
        "qw_q01",
        "qw_q99",
    ):
        record[key] = resolution.get(key)
    return record


class _PolicyBinding:
    def __init__(self, state, q_franka_to_canon):
        self._state = state
        self._q_franka_to_canon = q_franka_to_canon
        hemisphere = state.A.policy_state_quat_hemisphere
        record = verify_policy_state_quat_hemisphere(
            hemisphere, *policy_state_quat_hemisphere_metadata(state)
        )
        # A LeRobot pose source replays the dataset quaternion with its own
        # sign; the hemisphere applies only to simulated hand poses.
        record["applied_to_observation"] = state.A.policy_pose_source != "lerobot"
        # One binding, and so one encoder, per rollout: sign continuity starts
        # at this rollout's first observation.
        self._state_quat = pio.PolicyStateQuatEncoder(
            hemisphere, q_franka_to_canon_wxyz=q_franka_to_canon
        )
        state.POLICY_STATE_QUAT_HEMISPHERE = record
        state.POLICY_STATE_QUAT_ENCODER = self._state_quat

    def ee_now(self, key):
        return policy_tcp_pose(self._state, key)

    def rel_state(self):
        pL, qL = self.ee_now("L")
        pR, qR = self.ee_now("R")
        return self._state_quat(pL, qL, pR, qR)

    def pol_apply(self, p, q, a, off):
        return pio.apply_policy_delta(
            p,
            q,
            a[off:off + 7],
            q_franka_to_canon_wxyz=self._q_franka_to_canon,
            q_canon_to_franka_wxyz=self._state.Q_CANON_TO_ROBOT,
        )

def _policy_display_transform_record(transform):
    if transform is None:
        return None
    return {
        "algorithm":transform.algorithm,
        "resize_mode":transform.resize_mode,
        "source_hw":list(transform.source_hw),
        "policy_hw":list(transform.policy_hw),
        "policy_content_xywh":list(transform.policy_content_xywh),
        "output_hw":list(transform.output_hw),
        "output_content_xywh":list(transform.output_content_xywh),
    }

def _raw_policy_display_transform_record():
    return {
        "algorithm":"raw_policy_tensor_full_canvas_resize_v1",
        "resize_mode":"raw224",
        "source_hw":[224,224],
        "policy_hw":[224,224],
        "policy_content_xywh":[0,0,224,224],
        "output_hw":[135,180],
        "output_content_xywh":[0,0,180,135],
    }

def _capture_next_policy_observation(state,frame):
    """Capture one distinct OBS and advance the OBS index used by LeRobot mode."""
    st=state._pstate
    obs,meta=_capture_policy_observation(state,frame,st["obs_count"])
    st["obs_count"]+=1
    return obs,meta

def _build_policy_target_trajectory(state, chunk,obs_meta):
    # cmd_twist quaternion xyz are velocity coefficients, so they need no pose anchor.
    if fr3_uses_cmd_twist_torque(getattr(state.A, "fr3_controller", "joint_ik")):
        return None
    if (
        state.A.policy_pose_repr=="delta" and state.A.policy_delta_anchor!="observation"
    ):
        return None
    target_trajectory={}
    for side,action_slice in (("L",pio.LEFT_ACTION_SLICE),("R",pio.RIGHT_ACTION_SLICE)):
        anchor=obs_meta["execution_anchor"][side]
        trajectory_fn=(
            pio.compose_policy_anchored_trajectory
            if state.A.policy_pose_repr=="anchored"
            else pio.integrate_policy_delta_trajectory
        )
        positions,quaternions=trajectory_fn(
            np.asarray(anchor["position"],float),
            np.asarray(anchor["gripper_quat_wxyz"],float),
            np.asarray(chunk[:,action_slice],float),
            q_franka_to_canon_wxyz=state._Q_FC,
            q_canon_to_franka_wxyz=state.Q_CANON_TO_ROBOT,
        )
        target_trajectory[side]={"position":positions,"quaternion":quaternions}
    return target_trajectory

def _relative_pose_error(source,target):
    source=np.asarray(source,float); target=np.asarray(target,float)
    position_m=float(np.linalg.norm(source[:3]-target[:3]))
    qa=source[3:7]/max(float(np.linalg.norm(source[3:7])),1e-12)
    qb=target[3:7]/max(float(np.linalg.norm(target[3:7])),1e-12)
    orientation_deg=float(np.degrees(2.0*np.arccos(np.clip(abs(float(np.dot(qa,qb))),0.0,1.0))))
    return position_m,orientation_deg

def _prepare_rebased_conditioning_tail(state,target_obs):
    """Retarget the old action tail to a new OBS in the left-hand reference frame."""
    st=state._pstate
    if st.get("chunk") is None or not st.get("active_obs_meta"):
        raise RuntimeError("RF left-reference rebase requires an active policy chunk")
    raw_tail=pio.rf_overlap_conditioning_tail(
        st["chunk"],execution_start=st["action_start"],
        exec_steps=state.A.exec,overlap_steps=state.A.rf_overlap,
    )
    tail_start=int(st["action_start"]+state.A.exec-state.A.rf_overlap)
    destination_h=np.asarray(
        target_obs["observation.pose.left_hand_root_to_right_hand_root.absolute"],float
    )
    source_obs_h=np.asarray(st["active_obs_meta"].get("policy_left_to_right"),float)
    if source_obs_h.shape!=(7,):
        raise RuntimeError(
            "active policy OBS is missing a 7D left-to-right pose for RF rebase"
        )
    source_h,source_mode=pio.rf_conditioning_source_inter_hand(
        source_obs_h,np.asarray(st["chunk"],float),tail_start,destination_h,
        policy_delta_anchor=state.A.policy_delta_anchor,
    )
    rebased=pio.rebase_yubi_action_segment_left_reference(
        raw_tail,source_h,destination_h
    )
    correction=pio.left_reference_action_correction(source_h,destination_h)
    source_next=pio.propagate_yubi_inter_hand_state(source_h,raw_tail[:1])
    destination_next=pio.propagate_yubi_inter_hand_state(destination_h,rebased[:1])
    position_error_m,orientation_error_deg=_relative_pose_error(
        source_next,destination_next
    )
    meta={
        "basis":"left_hand_root",
        "source_mode":source_mode,
        "tail_start":tail_start,
        "source_boundary_left_to_right":np.asarray(source_h,float).tolist(),
        "destination_left_to_right":destination_h.tolist(),
        "right_first_action_correction":np.asarray(correction,float).tolist(),
        "raw_to_rebased_max_abs":float(np.max(np.abs(
            np.asarray(raw_tail[:,:pio.ACTION_DIM],float)
            -np.asarray(rebased[:,:pio.ACTION_DIM],float)
        ))),
        "next_inter_hand_position_error_m":position_error_m,
        "next_inter_hand_orientation_error_deg":orientation_error_deg,
    }
    return rebased,meta

def _prepare_policy_action_chunk(state, actions):
    chunk=np.asarray(actions)
    if state.EMBODIMENT=="fr3":
        if fr3_uses_cmd_twist_torque(getattr(state.A, "fr3_controller", "joint_ik")):
            validated=pio.validate_action_chunk(chunk)
            return validated,pio.fr3_policy_action_clamp_not_applied_telemetry(validated)
        if state.A.policy_action_clamp:
            return pio.clamp_fr3_policy_action_chunk_with_telemetry(chunk)
        validated=pio.validate_unclamped_fr3_policy_action_chunk(chunk)
        return validated,pio.fr3_policy_action_clamp_not_applied_telemetry(validated)
    return pio.validate_action_chunk(chunk),None

def _infer_policy_chunk(
    state,obs,obs_meta,inference_frame,conditioning_prefix=None,conditioning_rebase_meta=None
):
    """Infer one chunk, optionally hard-conditioning its normalized RF prefix."""
    st=state._pstate
    observation_frame=int(obs_meta["observation_frame"])
    obs_log=_save_policy_obs_before_infer(state,
        observation_frame,st["infer_count"],obs,infer_frame=inference_frame,
        obs_meta=obs_meta,conditioning_prefix=conditioning_prefix,
        conditioning_rebase_meta=conditioning_rebase_meta,
    )
    model_obs=(pio.add_yubi_action_conditioning(obs,conditioning_prefix)
               if conditioning_prefix is not None else obs)
    if conditioning_prefix is not None and not state.A.policy_batch_server:
        _policy_result=state.policy_backend.infer_many(state.policy,[model_obs])[0]
    else:
        _policy_result=state.policy.infer(model_obs)
    chunk,action_clamp_telemetry=_prepare_policy_action_chunk(state,
        _policy_result["actions"]
    )
    pio.validate_exec_horizon(chunk,state.A.exec)
    pio.validate_rf_overlap(state.A.exec,state.A.rf_overlap,chunk.shape[0])
    target_trajectory=_build_policy_target_trajectory(state, chunk,obs_meta)
    result_meta={k:v for k,v in _policy_result.items() if k!="actions"}
    prefix_error=None
    if conditioning_prefix is not None:
        prefix=np.asarray(conditioning_prefix,float)
        prefix_error=float(np.max(np.abs(np.asarray(chunk[:len(prefix),:pio.ACTION_DIM],float)-prefix[:,:pio.ACTION_DIM])))
    chunk_index=int(st["chunk_index"]+1)
    execution_start=int(state.A.rf_overlap if conditioning_prefix is not None else 0)
    state_quat_hemisphere=getattr(state,"POLICY_STATE_QUAT_HEMISPHERE",None) or {}
    plan_extra={}
    if action_clamp_telemetry is not None:
        plan_extra["policy_action_clamp_telemetry"]=action_clamp_telemetry
    if state_quat_hemisphere:
        plan_extra["policy_state_quat_hemisphere"]={
            key:state_quat_hemisphere.get(key)
            for key in ("hemisphere","verified","norm_stats_sha256","qw_q01","qw_q99")
        }
    plan_record=build_policy_plan_record(
        state._POLICY_PLAN_CONFIG,
        chunk_index=chunk_index,
        observation_frame=observation_frame,
        inference_frame=inference_frame,
        execution_start=execution_start,
        sampler="single",
        prev_weight=None,
        dynamic_prev_weight=False,
        prev_weight_schedule=None,
        step_shift=0,
        conditioned=conditioning_prefix is not None,
        execution_anchor=obs_meta["execution_anchor"],
        prefix_error=prefix_error,
        conditioning_prefix=conditioning_prefix,
        conditioning_rebase_meta=conditioning_rebase_meta,
        chunk=chunk,
        extra=plan_extra or None,
    )
    state._POLICY_CHUNK_RECORDS.append(plan_record)
    st["infer_count"]+=1
    if chunk_index==0:
        print(f"[obs0] rel_pose={np.round(obs['observation.pose.left_hand_root_to_right_hand_root.absolute'],4).tolist()} joint={obs['observation.joint_states'].tolist()} image_source={state.A.policy_image_source} image_index={obs_meta['image_index']} pose_source={state.A.policy_pose_source} pose_index={obs_meta['pose_index']} chunk_shape={chunk.shape} a0R_dpos={np.round(chunk[0][7:10],4).tolist()} a0L_dpos={np.round(chunk[0][0:3],4).tolist()} gripL={float(chunk[0][state.GI_L]):.3f} gripR={float(chunk[0][state.GI_R]):.3f} state_quat_hemisphere={state_quat_hemisphere.get('hemisphere')} verified={state_quat_hemisphere.get('verified')} norm_stats_sha256={state_quat_hemisphere.get('norm_stats_sha256')} qw_q01={state_quat_hemisphere.get('qw_q01')} qw_q99={state_quat_hemisphere.get('qw_q99')}",flush=True)
    elif conditioning_prefix is not None:
        print(f"[rf-overlap] chunk={chunk_index} obs_f={observation_frame} infer_f={inference_frame} prefix={len(conditioning_prefix)} execute=[{state.A.rf_overlap}:{state.A.rf_overlap+state.A.exec}] roundtrip_max={prefix_error:.3e}",flush=True)
    return build_policy_chunk_result(
        chunk=chunk,
        chunk_index=chunk_index,
        obs_meta=obs_meta,
        obs_log=obs_log,
        policy_result_meta=result_meta,
        prefix_error=prefix_error,
        bridge_error=None,
        bridge_rebased_prev_distance=None,
        prev_weight=None,
        execution_start=execution_start,
        sampler="single",
        conditioned=conditioning_prefix is not None,
        inference_frame=inference_frame,
        conditioning_rebase_meta=conditioning_rebase_meta,
        target_trajectory=target_trajectory,
        extra={"policy_prompt":str(obs["prompt"])},
    )

def _activate_policy_chunk(state,inferred):
    st=state._pstate
    st["chunk"]=inferred["chunk"]
    st["chunk_index"]=inferred["chunk_index"]
    st["active_obs_meta"]=inferred["obs_meta"]
    st["active_obs_log"]=inferred["obs_log"]
    st["policy_result_meta"]=inferred["policy_result_meta"]
    st["last_prefix_error"]=inferred["prefix_error"]
    st["last_bridge_error"]=inferred["bridge_error"]
    st["last_bridge_rebased_prev_distance"]=inferred.get("bridge_rebased_prev_distance")
    st["active_dynamic_prev_weight"]=inferred.get("dynamic_prev_weight", False)
    st["active_prev_weight_schedule"]=inferred.get("prev_weight_schedule")
    st["conditioning_rebase_meta"]=inferred.get("conditioning_rebase_meta")
    st["active_conditioned"]=inferred["conditioned"]
    st["active_sampler"]=inferred["sampler"]
    st["active_prev_weight"]=inferred.get("prev_weight")
    st["active_inference_frame"]=inferred["inference_frame"]
    st["target_trajectory"]=inferred["target_trajectory"]
    st["action_start"]=int(inferred["execution_start"])
    st["block_step"]=0


__all__ = [
    "_PolicyBinding",
    "_activate_policy_chunk",
    "_capture_next_policy_observation",
    "_infer_policy_chunk",
    "_policy_display_transform_record",
    "_prepare_rebased_conditioning_tail",
    "_raw_policy_display_transform_record",
]
