# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Per-frame overview-state and wrist capture."""

from __future__ import annotations

import numpy as np

from robosuite_env.policy import policy_io as pio
from robosuite_env.rollout.render.overview import (
    add_origin,
    apply_black_background,
    overview_render_data,
)
from robosuite_env.rollout.render.video_wrist import (
    validate_ephemeral_video_wrist_capture,
)
from robosuite_env.rollout.resource_lifecycle import close_renderers


def capture_render_sample(state, source_t) -> None:
    """Record the current MuJoCo state as one output video frame."""

    overview_video_frame = None
    video_wrist_images = pending_wrist_images = None
    pending_wrist_renderers = None
    if not state.A.no_render:
        if state._DEFER_OVERVIEW_RENDER:
            state._OVERVIEW_STATE_FRAMES.append(
                {
                    "qpos": np.asarray(
                        state.d.qpos, dtype=np.float64
                    ).copy(),
                    "time": float(state.d.time),
                    "mocap_pos": np.asarray(
                        state.d.mocap_pos, dtype=np.float64
                    ).copy(),
                    "mocap_quat": np.asarray(
                        state.d.mocap_quat, dtype=np.float64
                    ).copy(),
                }
            )
        else:
            if state.renderers.ren is None:
                raise RuntimeError("inline overview renderer is unavailable")
            with state.render_lock():
                state.renderers.ren.update_scene(
                    overview_render_data(state),
                    camera=state.ov,
                    scene_option=state.vopt,
                )
                apply_black_background(state, state.renderers.ren.scene)
                add_origin(state, state.renderers.ren.scene)
                overview_video_frame = state.renderers.ren.render().copy()
        state.render_source_frames.append(int(source_t))

        if state._WILL_CREATE_VIDEO_WRIST_RENDERER:
            # Capture both EGL renderers before starting either CUDA-backed GS
            # request. The outer lock keeps the pair atomic across workers.
            capture_wfR = state.renderers.wfR
            capture_wfL = state.renderers.wfL
            with state.render_lock():
                if state._VIDEO_WRIST_EPHEMERAL:
                    live_video_names = state.renderers.live_names(
                        ("wfR", "wfL", "wfR_obs", "wfL_obs", "ren")
                    )
                    if (
                        live_video_names
                        or state._VIDEO_WRIST_RENDERER_STATS["live_count"] != 0
                    ):
                        raise RuntimeError(
                            "ephemeral video wrist requires zero live MuJoCo "
                            f"EGL owners, found names={live_video_names} live="
                            f"{state._VIDEO_WRIST_RENDERER_STATS['live_count']}"
                        )
                    capture_wfR, capture_wfL = state._new_video_wrist_pair()
                    state.renderers.wfR = capture_wfR
                    state.renderers.wfL = capture_wfL
                try:
                    mujoco_wrist_captures = (
                        capture_wfL.capture_mujoco_stage(),
                        capture_wfR.capture_mujoco_stage(),
                    )
                    validate_ephemeral_video_wrist_capture(
                        state, mujoco_wrist_captures[0], "L"
                    )
                    validate_ephemeral_video_wrist_capture(
                        state, mujoco_wrist_captures[1], "R"
                    )
                finally:
                    if state._VIDEO_WRIST_EPHEMERAL:
                        video_wrist_close_errors = close_renderers(
                            state,
                            ("wfL", "wfR"),
                        )
                        if not video_wrist_close_errors:
                            state._VIDEO_WRIST_RENDERER_STATS[
                                "close_count"
                            ] += 1
                            state._VIDEO_WRIST_RENDERER_STATS[
                                "live_count"
                            ] -= 1
                        if video_wrist_close_errors:
                            raise RuntimeError(
                                "ephemeral video wrist EGL cleanup failed: "
                                + "; ".join(video_wrist_close_errors)
                            )
            pending_wrist_images = (
                capture_wfL.complete_gs_stage(mujoco_wrist_captures[0]),
                capture_wfR.complete_gs_stage(mujoco_wrist_captures[1]),
            )
            pending_wrist_renderers = (capture_wfL, capture_wfR)

        if not state.A.no_wrist and not state.A.clean_video:
            if state._POLICY_INSET_INPUT:
                if any(
                    state.POLICY_OBS_DISPLAY[side] is None
                    for side in ("L", "R")
                ):
                    raise RuntimeError(
                        "policy wrist display was not initialized before "
                        "video capture"
                    )
                if any(
                    state.POLICY_OBS_RAW[side] is None
                    for side in ("L", "R")
                ):
                    raise RuntimeError(
                        "raw policy wrist image was not initialized before "
                        "video capture"
                    )
                wl = state.POLICY_OBS_DISPLAY["L"].copy()
                wr = state.POLICY_OBS_DISPLAY["R"].copy()
            else:
                pending_wfL, pending_wfR = pending_wrist_renderers
                wl = pending_wfL.finalize_cpu_stage(pending_wrist_images[0])
                wr = pending_wfR.finalize_cpu_stage(pending_wrist_images[1])
            if not state.A.clean_video:
                video_wrist_images = (wl, wr)

    # Disk I/O and CPU wrist projection stay outside the shared GPU lock.
    if overview_video_frame is not None:
        state._OVERVIEW_VIDEO_FRAMES.append(overview_video_frame)
    if video_wrist_images is not None:
        if state._POLICY_INSET_INPUT:
            display_pair = tuple(
                np.asarray(image) for image in video_wrist_images
            )
            for side, image in zip(("L", "R"), display_pair):
                if image.dtype != np.uint8 or image.shape != (135, 180, 3):
                    raise RuntimeError(
                        f"policy wrist display {side} must be uint8 "
                        f"(135,180,3), got {image.dtype} {image.shape}"
                    )
        else:
            display_pair = tuple(
                pio.prepare_policy_wrist_display_image(
                    np.asarray(image),
                    resize_mode="stretch",
                    source_hw=tuple(np.asarray(image).shape[:2]),
                    output_hw=(135, 180),
                )[0]
                for image in video_wrist_images
            )
        state._WRIST_VIDEO_FRAMES.append(np.stack(display_pair))
    state._CAPTURE_INDEX += 1


__all__ = ["capture_render_sample"]
