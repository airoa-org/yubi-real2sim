# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Policy wrist setup and observation capture."""

from __future__ import annotations

import os

import numpy as np

from robosuite_env.policy import policy_io as pio
from robosuite_env.rollout.build.gripper import grip_state_obs
from robosuite_env.rollout.render.video_wrist import WristPairFactory
from robosuite_env.rollout.resource_lifecycle import close_renderers


def policy_img_resize(state, img):
    """Resize a wrist image to the policy's 224x224 input contract."""

    return pio.resize_policy_image(img, state.A.policy_image_resize)


def validate_ephemeral_policy_capture(state, capture, side) -> None:
    """Fail closed before EGL teardown if GS data is not detached."""

    if not state._POLICY_WRIST_EPHEMERAL or state._ogs is None:
        return
    names = getattr(capture, "batch_face_names", None)
    payloads = getattr(capture, "batch_payloads", None)
    batch_render = getattr(capture, "batch_render", None)
    if (
        not callable(batch_render)
        or names is None
        or payloads is None
        or len(names) != len(payloads)
        or not names
    ):
        raise RuntimeError(
            f"ephemeral policy wrist {side} did not capture a complete "
            "immutable GS batch before EGL teardown"
        )


def setup_policy_wrist(state) -> None:
    """Load optional real observations and configure policy wrist EGL owners."""

    state.renderers.wfR_obs = state.renderers.wfL_obs = None
    state._POLICY_WRIST_RENDERER_STATS = {
        "create_count": 0,
        "close_count": 0,
        "live_count": 0,
        "peak_live_count": 0,
    }
    state.POLICY_LEROBOT_IMAGES = None
    state.POLICY_LEROBOT_POSES = None
    state.POLICY_OBS_DISPLAY = {"L": None, "R": None}
    state.POLICY_OBS_RAW = {"L": None, "R": None}
    state.POLICY_OBS_DISPLAY_TRANSFORM = {"L": None, "R": None}
    state.POLICY_OBS_LOGDIR = (
        os.path.join(state.LOGDIR, "policy_obs") if state.POLICY else ""
    )
    state.POLICY_OBS_LOGGED = []

    if state.POLICY and (
        state.A.policy_pose_source == "lerobot"
        or state.A.policy_image_source == "lerobot"
    ):
        from robosuite_env.rollout.sources.lerobot import (
            load_lerobot_policy_images,
            load_lerobot_policy_poses,
        )

    if state.POLICY and state.A.policy_pose_source == "lerobot":
        state.POLICY_LEROBOT_POSES = load_lerobot_policy_poses(
            state.A.policy_image_episode
        )
        print(
            "[sim] policy obs poses ready (LeRobot real proprioception, "
            f"index={state.A.policy_image_index}, "
            f"{'loop' if state.A.policy_image_loop else 'clamp'})",
            flush=True,
        )
    if state.POLICY:
        if state.A.policy_image_source == "lerobot":
            state.POLICY_LEROBOT_IMAGES = load_lerobot_policy_images(
                state.A.policy_image_episode
            )
            print(
                "[sim] policy obs images ready (LeRobot real wrist, "
                f"index={state.A.policy_image_index}, "
                f"{'loop' if state.A.policy_image_loop else 'clamp'})",
                flush=True,
            )
        else:
            from robosuite_env.rendering.wrist_fisheye import (
                WristFisheye as _WF,
            )

            state._ogs = (
                state.gsc
                if (state.gsc is not None and not state.A.no_obs_gs)
                else None
            )
            state._rpre, state._lpre = (
                state.PROFILE.wrist_prefixes["R"],
                state.PROFILE.wrist_prefixes["L"],
            )
            state._black_obs = bool(
                state.A.black_background and state._ogs is None
            )
            state._POLICY_WRIST_EPHEMERAL = bool(state._RENDER_LOCK_PATH)
            if state._POLICY_WRIST_EPHEMERAL and state._ogs is not None:
                state._batch_rgb = callable(
                    getattr(state._ogs, "render_payloads", None)
                )
                state._batch_rgbd = callable(
                    getattr(state._ogs, "render_payloads_with_depth", None)
                )
                state._required_batch_available = (
                    state._batch_rgb
                    if state.A.gs_composite_mode == "foreground"
                    else state._batch_rgbd
                )
                if not callable(
                    getattr(state._ogs, "camera_payload", None)
                ) or not state._required_batch_available:
                    raise RuntimeError(
                        "a render-locked policy wrist requires an immutable "
                        "GS batch payload API for composite mode "
                        f"{state.A.gs_composite_mode!r}"
                    )

            state._new_policy_wrist_pair = WristPairFactory(
                state,
                _WF,
                state._rpre,
                state._lpre,
                state._ogs,
                state._black_obs,
                state.A.policy_yubi_render_profile,
                state.A.gs_composite_mode,
                state._POLICY_WRIST_RENDERER_STATS,
                state._POLICY_WRIST_EPHEMERAL,
                "wfR_obs",
                "policy",
                gs_only=state.A.obs_gs_only,
            )

            if not state._POLICY_WRIST_EPHEMERAL:
                with state.render_lock():
                    (
                        state.renderers.wfR_obs,
                        state.renderers.wfL_obs,
                    ) = state._new_policy_wrist_pair()
                print(
                    "[sim] policy obs wrist renderers ready "
                    f"({'3DGS background' if state._ogs is not None else 'clean, no GS'}, "
                    f"profile={state.A.policy_yubi_render_profile}, "
                    "lifecycle=persistent)",
                    flush=True,
                )
            else:
                print(
                    "[sim] policy obs wrist renderer ready on demand "
                    f"({'3DGS background' if state._ogs is not None else 'clean, no GS'}, "
                    f"profile={state.A.policy_yubi_render_profile}, "
                    "lifecycle=per-observation locked create/capture/close)",
                    flush=True,
                )


def capture_policy_observation(state, frame, infer_index):
    """Render/copy an observation for RF-overlap inference."""

    lr_raw = (
        int(infer_index if state.A.policy_image_index == "infer" else frame)
        if (
            state.A.policy_image_source == "lerobot"
            or state.A.policy_pose_source == "lerobot"
        )
        else None
    )
    if state.A.policy_image_source == "lerobot":
        nimg = len(state.POLICY_LEROBOT_IMAGES["L"])
        img_i = (
            lr_raw % nimg
            if state.A.policy_image_loop
            else max(0, min(lr_raw, nimg - 1))
        )
        source_L = np.asarray(state.POLICY_LEROBOT_IMAGES["L"][img_i])
        source_R = np.asarray(state.POLICY_LEROBOT_IMAGES["R"][img_i])
        oL = policy_img_resize(state, source_L)
        oR = policy_img_resize(state, source_R)
        image_index = int(img_i)
        image_raw_index = int(lr_raw)
    else:
        # Both MuJoCo wrists complete before either CUDA-backed GS request.
        obs_wfR = state.renderers.wfR_obs
        obs_wfL = state.renderers.wfL_obs
        with state.render_lock():
            if state._POLICY_WRIST_EPHEMERAL:
                live_policy_names = state.renderers.live_names(
                    ("wfR_obs", "wfL_obs", "wfR", "wfL", "ren")
                )
                if (
                    live_policy_names
                    or state._POLICY_WRIST_RENDERER_STATS["live_count"] != 0
                ):
                    raise RuntimeError(
                        "ephemeral policy wrist requires zero live MuJoCo EGL "
                        f"owners, found names={live_policy_names} live="
                        f"{state._POLICY_WRIST_RENDERER_STATS['live_count']}"
                    )
                obs_wfR, obs_wfL = state._new_policy_wrist_pair()
                state.renderers.wfR_obs = obs_wfR
                state.renderers.wfL_obs = obs_wfL
            try:
                mujoco_L = obs_wfL.capture_mujoco_stage()
                validate_ephemeral_policy_capture(state, mujoco_L, "L")
                mujoco_R = obs_wfR.capture_mujoco_stage()
                validate_ephemeral_policy_capture(state, mujoco_R, "R")
            finally:
                if state._POLICY_WRIST_EPHEMERAL:
                    policy_wrist_close_errors = close_renderers(
                        state,
                        ("wfL_obs", "wfR_obs"),
                    )
                    if not policy_wrist_close_errors:
                        state._POLICY_WRIST_RENDERER_STATS["close_count"] += 1
                        state._POLICY_WRIST_RENDERER_STATS["live_count"] -= 1
                    if policy_wrist_close_errors:
                        raise RuntimeError(
                            "ephemeral policy wrist EGL cleanup failed: "
                            + "; ".join(policy_wrist_close_errors)
                        )
        # GS completion stays outside the physical-GPU lock.
        pending_L = obs_wfL.complete_gs_stage(mujoco_L)
        pending_R = obs_wfR.complete_gs_stage(mujoco_R)
        source_L = obs_wfL.finalize_cpu_stage(pending_L)
        source_R = obs_wfR.finalize_cpu_stage(pending_R)
        for side, source, renderer in (
            ("L", source_L, obs_wfL),
            ("R", source_R, obs_wfR),
        ):
            expected_shape = (
                renderer.source_height,
                renderer.source_width,
                3,
            )
            if np.asarray(source).shape != expected_shape:
                raise RuntimeError(
                    f"sim policy wrist {side} source shape changed: "
                    f"{np.asarray(source).shape} != {expected_shape}"
                )
        oL = pio.preprocess_sim_policy_wrist_image(
            source_L,
            contrast_factor=state.A.policy_wrist_contrast,
            resize_mode=state.A.policy_image_resize,
        )
        oR = pio.preprocess_sim_policy_wrist_image(
            source_R,
            contrast_factor=state.A.policy_wrist_contrast,
            resize_mode=state.A.policy_image_resize,
        )
        image_index = None
        image_raw_index = None

    source_shapes = {
        "L": list(np.asarray(source_L).shape),
        "R": list(np.asarray(source_R).shape),
    }
    for side, policy_image in (("L", oL), ("R", oR)):
        display, transform = pio.prepare_policy_wrist_display_image(
            np.asarray(policy_image),
            resize_mode=state.A.policy_image_resize,
            source_hw=tuple(source_shapes[side][:2]),
            output_hw=(135, 180),
        )
        state.POLICY_OBS_RAW[side] = np.asarray(policy_image).copy()
        state.POLICY_OBS_DISPLAY[side] = display
        state.POLICY_OBS_DISPLAY_TRANSFORM[side] = transform

    if state.A.policy_pose_source == "lerobot":
        npose = len(state.POLICY_LEROBOT_POSES["rel"])
        pose_i = (
            lr_raw % npose
            if state.A.policy_image_loop
            else max(0, min(lr_raw, npose - 1))
        )
        rel_obs = state.POLICY_LEROBOT_POSES["rel"][pose_i]
        js_obs = state.POLICY_LEROBOT_POSES["joint"][pose_i]
        pose_index = int(pose_i)
    else:
        rel_obs = state.rel_state()
        js_obs = grip_state_obs(state)
        pose_index = None

    obs = {
        "observation.image.left": np.asarray(oL).copy(),
        "observation.image.right": np.asarray(oR).copy(),
        "observation.pose.left_hand_root_to_right_hand_root.absolute": np.asarray(
            rel_obs
        ).copy(),
        "observation.joint_states": np.asarray(js_obs).copy(),
        "prompt": state.A.instruction,
    }
    execution_anchor = {}
    for side in ("L", "R"):
        anchor_pos, anchor_quat = state.ee_now(side)
        execution_anchor[side] = {
            "position": [float(x) for x in np.asarray(anchor_pos, float)],
            "gripper_quat_wxyz": [
                float(x) for x in np.asarray(anchor_quat, float)
            ],
        }
    meta = {
        "observation_frame": int(frame),
        "image_index": image_index,
        "image_raw_index": image_raw_index,
        "pose_index": pose_index,
        "image_source_shapes": source_shapes,
        "policy_left_to_right": np.asarray(rel_obs, float).tolist(),
        "execution_anchor": execution_anchor,
    }
    return obs, meta


__all__ = [
    "capture_policy_observation",
    "policy_img_resize",
    "setup_policy_wrist",
    "validate_ephemeral_policy_capture",
]
