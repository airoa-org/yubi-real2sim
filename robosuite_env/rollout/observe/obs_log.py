# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Persist the exact policy observations used for inference."""

import json
import os

import cv2 as cv
import numpy as np



def save_policy_obs_before_infer(
    state,
    t,
    infer_count,
    obs,
    *,
    infer_frame=None,
    obs_meta=None,
    conditioning_prefix=None,
    conditioning_rebase_meta=None,
):
    """Persist the exact observation images immediately before policy.infer."""
    if not state.POLICY:
        return None
    targets = [state.POLICY_OBS_LOGDIR]
    if state.A.dump_obs and os.path.abspath(state.A.dump_obs) != os.path.abspath(
        state.POLICY_OBS_LOGDIR
    ):
        targets.append(state.A.dump_obs)
    obs_meta = obs_meta or {}
    rec = {
        "frame": int(t),
        "observation_frame": int(t),
        "inference_frame": int(t if infer_frame is None else infer_frame),
        "infer_count": int(infer_count),
        "image_source": state.A.policy_image_source,
        "image_index": obs_meta.get("image_index"),
        "image_raw_index": obs_meta.get("image_raw_index"),
        "image_source_shapes": obs_meta.get("image_source_shapes"),
        "pose_source": state.A.policy_pose_source,
        "pose_index": obs_meta.get("pose_index"),
        "execution_anchor": obs_meta.get("execution_anchor"),
        "policy_image_resize": state.A.policy_image_resize,
        "policy_yubi_render_profile": state.A.policy_yubi_render_profile,
        "policy_wrist_backend": state.A.policy_wrist_backend,
        "gs_kb4_near_cull": state.A.gs_kb4_near_cull,
        "policy_wrist_contrast": float(state.A.policy_wrist_contrast),
        "policy_wrist_preprocess_order": state._POLICY_WRIST_PREPROCESS_ORDER,
        "prompt": state.A.instruction,
        "rf_overlap": int(state.A.rf_overlap),
        "rf_overlap_sampler": state.A.rf_overlap_sampler,
        "rf_overlap_sampler_configured": state.A.rf_overlap_sampler,
        "rf_overlap_sampler_effective": "single",
        "rf_overlap_collapse_reason": None,
        "rf_overlap_prev_weight": None,
        "rf_overlap_prev_weight_configured": None,
        "rf_overlap_prev_weight_effective": None,
        "rf_overlap_prev_weight_dynamic": False,
        "rf_overlap_prev_weight_schedule": None,
        "rf_overlap_step_shift": int(state.A.rf_overlap_step_shift),
        "rf_overlap_step_shift_configured": int(state.A.rf_overlap_step_shift),
        "rf_overlap_step_shift_effective": 0,
        "rf_overlap_sample_num_steps": None,
        "paths": {},
    }
    if conditioning_prefix is not None:
        rec["conditioning_prefix"] = [
            [float(x) for x in row]
            for row in np.asarray(conditioning_prefix, float)
        ]
    if conditioning_rebase_meta is not None:
        rec["conditioning_left_reference_rebase"] = dict(
            conditioning_rebase_meta
        )
    for outdir in targets:
        os.makedirs(outdir, exist_ok=True)
        prefix = (
            "primary"
            if os.path.abspath(outdir) == os.path.abspath(state.POLICY_OBS_LOGDIR)
            else "dump_obs"
        )
        file_tag = f"_f{t:04d}"
        for key, side in (
            ("observation.image.left", "L"),
            ("observation.image.right", "R"),
        ):
            img = np.asarray(obs[key])
            path = os.path.join(outdir, f"obs_{side}{file_tag}.png")
            cv.imwrite(path, cv.cvtColor(img, cv.COLOR_RGB2BGR))
            rec["paths"][f"{prefix}_{side}"] = path
        if prefix == "primary":
            meta_path = os.path.join(outdir, f"obs{file_tag}.json")
            meta = dict(rec)
            meta["image_shapes"] = {
                "L": list(np.asarray(obs["observation.image.left"]).shape),
                "R": list(np.asarray(obs["observation.image.right"]).shape),
            }
            meta["rel_pose"] = [
                float(x)
                for x in np.asarray(
                    obs[
                        "observation.pose.left_hand_root_to_right_hand_root.absolute"
                    ]
                ).ravel()
            ]
            meta["joint_states"] = [
                float(x)
                for x in np.asarray(obs["observation.joint_states"]).ravel()
            ]
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            rec["paths"]["primary_meta"] = meta_path
    state.POLICY_OBS_LOGGED.append(rec)
    return rec


__all__ = ["save_policy_obs_before_infer"]
