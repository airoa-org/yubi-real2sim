# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Deferred overview rendering and final video composition."""

from __future__ import annotations

import cv2 as cv
import mujoco
import numpy as np

from robosuite_env.rollout.render.overview import (
    add_origin,
    apply_black_background,
    overview_render_data,
)
from robosuite_env.rollout.resource_lifecycle import close_renderers


def iter_overview_video_frames(state):
    """Yield overview RGB frames without retaining rendered pixels in RAM."""

    if not state._DEFER_OVERVIEW_RENDER:
        yield from state._OVERVIEW_VIDEO_FRAMES
        return

    # Close every old wrist/overview owner before the isolated context exists.
    rendered_count = 0
    with state.render_lock():
        close_errors = close_renderers(
            state,
            ("wfL_obs", "wfR_obs", "wfL", "wfR", "ren"),
        )
        if close_errors:
            raise RuntimeError(
                "deferred wrist EGL cleanup failed before isolated overview "
                "pass: "
                + "; ".join(close_errors)
            )
        overview_renderer = mujoco.Renderer(state.m, 480, 640)
        state.renderers.ren = overview_renderer
        try:
            for snapshot in state._OVERVIEW_STATE_FRAMES:
                overview_renderer.update_scene(
                    overview_render_data(state, snapshot),
                    camera=state.ov,
                    scene_option=state.vopt,
                )
                apply_black_background(state, overview_renderer.scene)
                add_origin(state, overview_renderer.scene)
                frame = overview_renderer.render().copy()
                if state.GS_DEPTH is not None and state.GSBG is not None:
                    depth, foreground = (
                        state.gs_api.render_depth_and_foreground_mask(
                            overview_renderer
                        )
                    )
                elif (
                    state.GSBG is not None
                    and state.A.gs_composite_mode == "foreground"
                ):
                    foreground = state.gs_api.render_foreground_mask(
                        overview_renderer
                    )
                if state.GS_DEPTH is not None and state.GSBG is not None:
                    frame = state.gs_api.composite_background_depth(
                        frame,
                        depth,
                        foreground,
                        state.GSBG,
                        state.GS_DEPTH,
                    )
                elif (
                    state.GSBG is not None
                    and state.A.gs_composite_mode == "foreground"
                ):
                    frame = state.gs_api.composite_background_mask(
                        frame,
                        foreground,
                        state.GSBG,
                    )
                state._OVERVIEW_VIDEO_FRAMES.append(frame)
                rendered_count += 1
        finally:
            overview_close_errors = close_renderers(
                state,
                ("ren",),
            )
            if overview_close_errors:
                raise RuntimeError(
                    "deferred overview EGL cleanup failed after rendering: "
                    + "; ".join(overview_close_errors)
                )
    print(
        f"[sim] isolated overview EGL pass complete: {rendered_count} frames "
        "after wrist teardown",
        flush=True,
    )
    yield from state._OVERVIEW_VIDEO_FRAMES


def iter_output_video_frames(state):
    """Compose and yield one final RGB video frame at a time."""

    if len(state.render_source_frames) != state._render_count:
        raise RuntimeError(
            "video source/overview frame count mismatch: "
            f"source={len(state.render_source_frames)} "
            f"overview={state._render_count}"
        )
    expected_wrist_count = (
        state._render_count if state._VIDEO_NEEDS_WRISTS else 0
    )
    if len(state._WRIST_VIDEO_FRAMES) != expected_wrist_count:
        raise RuntimeError(
            "video wrist/overview frame count mismatch: "
            f"wrist={len(state._WRIST_VIDEO_FRAMES)} "
            f"overview={state._render_count}"
        )
    wrist_frames = iter(state._WRIST_VIDEO_FRAMES)
    yielded_count = 0
    for t, ovf in enumerate(iter_overview_video_frames(state)):
        if t >= state._render_count:
            raise RuntimeError(
                "overview renderer produced more frames than captured states"
            )
        source_t = state.render_source_frames[t]
        if (
            state.gsc is not None
            and state.SKY is not None
            and state.GSBG is not None
            and state.GS_DEPTH is None
            and not (
                state._DEFER_OVERVIEW_RENDER
                and state.A.gs_composite_mode == "foreground"
            )
        ):
            foreground = (
                np.abs(ovf.astype(np.int16) - state.SKY.astype(np.int16)).sum(2)
                > 14
            )
            out = state.GSBG.copy()
            out[foreground] = ovf[foreground]
            ovf = out
        if state._VIDEO_NEEDS_WRISTS:
            wrist_pair = next(wrist_frames)
            ovf[4:139, 4:184] = wrist_pair[0]
            ovf[4:139, 188:368] = wrist_pair[1]
            cv.putText(
                ovf,
                "wristL",
                (6, 151),
                cv.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
            cv.putText(
                ovf,
                "wristR",
                (190, 151),
                cv.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
        sim = ovf
        if not state.A.clean_video:
            cv.putText(
                sim,
                "SIM replay",
                (8, sim.shape[0] - 12),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cols = []
            if state.realcap is not None:
                state.realcap.set(
                    cv.CAP_PROP_POS_FRAMES,
                    min(source_t, int(state.realcap.get(7)) - 1),
                )
                ok, real_frame = state.realcap.read()
                real_panel = (
                    cv.cvtColor(
                        cv.resize(
                            real_frame,
                            (sim.shape[1], sim.shape[0]),
                        ),
                        cv.COLOR_BGR2RGB,
                    )
                    if ok
                    else np.full_like(sim, 20)
                )
                cv.putText(
                    real_panel,
                    "REAL data (center cam)",
                    (8, 24),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
                cols.append(real_panel)
            cols.append(sim)
            sim = np.hstack(cols)
        yielded_count += 1
        yield sim
    if yielded_count != state._render_count:
        raise RuntimeError(
            "overview renderer produced fewer frames than captured states: "
            f"rendered={yielded_count} expected={state._render_count}"
        )


__all__ = ["iter_output_video_frames", "iter_overview_video_frames"]
