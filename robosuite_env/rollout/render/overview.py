# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Overview render planning and pass-one EGL setup."""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import mujoco
import numpy as np

from robosuite_env.rollout.resource_lifecycle import close_renderers


@dataclass(frozen=True)
class OverviewRenderPlan:
    locked_policy_obs_video_insets: bool
    policy_inset_input: bool
    will_create_video_wrist_renderer: bool
    will_create_policy_wrist_renderer: bool
    will_create_any_wrist_renderer: bool
    video_wrist_ephemeral: bool
    defer_overview_render: bool


def resolve_overview_render_plan(state) -> OverviewRenderPlan:
    """Resolve renderer ownership flags without creating an EGL context."""

    locked_policy_obs_video_insets = bool(
        state.POLICY
        and state._RENDER_LOCK_PATH
        and state.A.policy_image_source != "lerobot"
        and not state.A.no_render
        and not state.A.clean_video
        and not state.A.no_wrist
    )
    policy_inset_input = bool(
        (
            state.POLICY
            and (
                state.A.policy_image_source == "lerobot"
                or state.A.policy_obs_inset
            )
        )
        or state._STATE_REPLAY_POLICY_INSETS is not None
        or locked_policy_obs_video_insets
    )
    will_create_video_wrist_renderer = bool(
        not state.A.no_wrist
        and not state.A.no_render
        and not state.A.clean_video
        and not policy_inset_input
    )
    will_create_policy_wrist_renderer = bool(
        state.POLICY and state.A.policy_image_source != "lerobot"
    )
    will_create_any_wrist_renderer = bool(
        will_create_video_wrist_renderer
        or will_create_policy_wrist_renderer
    )
    video_wrist_ephemeral = bool(
        state._RENDER_LOCK_PATH and will_create_video_wrist_renderer
    )
    defer_overview_render = bool(
        state.A.defer_overview_render
        or state.A.policy_batch_server
        or (state._RENDER_LOCK_PATH and not state.A.no_render)
        or (will_create_any_wrist_renderer and not state.A.no_render)
        or (
            state.gsc is not None
            and state.A.gs_composite_mode in ("depth", "foreground")
        )
    )
    return OverviewRenderPlan(
        locked_policy_obs_video_insets=locked_policy_obs_video_insets,
        policy_inset_input=policy_inset_input,
        will_create_video_wrist_renderer=will_create_video_wrist_renderer,
        will_create_policy_wrist_renderer=will_create_policy_wrist_renderer,
        will_create_any_wrist_renderer=will_create_any_wrist_renderer,
        video_wrist_ephemeral=video_wrist_ephemeral,
        defer_overview_render=defer_overview_render,
    )


def overview_render_data(state, snapshot=None):
    """Build a render-only snapshot without dynamic solver state."""

    if state._OVERVIEW_RENDER_DATA is None:
        return state.d
    render_data = state._OVERVIEW_RENDER_DATA
    source_qpos = state.d.qpos if snapshot is None else snapshot["qpos"]
    source_time = float(
        state.d.time if snapshot is None else snapshot["time"]
    )
    render_data.qpos[:] = source_qpos
    render_data.qvel[:] = 0.0
    render_data.qacc[:] = 0.0
    render_data.ctrl[:] = 0.0
    render_data.time = source_time
    if state.m.nmocap:
        render_data.mocap_pos[:] = (
            state.d.mocap_pos
            if snapshot is None
            else snapshot["mocap_pos"]
        )
        render_data.mocap_quat[:] = (
            state.d.mocap_quat
            if snapshot is None
            else snapshot["mocap_quat"]
        )
    mujoco.mj_forward(state.m, render_data)
    return render_data


def parse_view_lookat(state, raw):
    raw = str(raw or "").replace(",", " ").strip()
    if not raw:
        return np.asarray([0.0, 0.0, float(state.A.table_z) + 0.15], float)
    vals = [float(x) for x in raw.split()]
    if len(vals) != 3:
        raise SystemExit(
            f"[sim] --view-lookat expects three numbers, got: {raw!r}"
        )
    return np.asarray(vals, float)


def add_origin(state, scene) -> None:
    if not state.A.show_origin:
        return

    def add_geom(geom_type, size, pos, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            geom_type,
            np.asarray(size, np.float64),
            np.asarray(pos, np.float64),
            np.eye(3).flatten(),
            np.asarray(rgba, np.float32),
        )
        scene.ngeom += 1

    add_geom(
        mujoco.mjtGeom.mjGEOM_SPHERE,
        [0.03, 0, 0],
        [0, 0, 0],
        [1, 1, 0, 1],
    )
    add_geom(
        mujoco.mjtGeom.mjGEOM_BOX,
        [0.06, 0.0025, 0.0025],
        [0.06, 0, 0],
        [1, 0, 0, 1],
    )
    add_geom(
        mujoco.mjtGeom.mjGEOM_BOX,
        [0.0025, 0.06, 0.0025],
        [0, 0.06, 0],
        [0, 1, 0, 1],
    )
    add_geom(
        mujoco.mjtGeom.mjGEOM_BOX,
        [0.0025, 0.0025, 0.06],
        [0, 0, 0.06],
        [0.3, 0.3, 1, 1],
    )


def apply_black_background(state, scene) -> None:
    if not state.A.black_background:
        return
    for flag in ("mjRND_SKYBOX", "mjRND_HAZE", "mjRND_FOG"):
        if hasattr(mujoco.mjtRndFlag, flag):
            scene.flags[int(getattr(mujoco.mjtRndFlag, flag))] = 0


def setup_overview_render(state) -> None:
    """Prefetch GS and release pass-one EGL before wrist construction."""

    plan = resolve_overview_render_plan(state)
    state._LOCKED_POLICY_OBS_VIDEO_INSETS = (
        plan.locked_policy_obs_video_insets
    )
    state._POLICY_INSET_INPUT = plan.policy_inset_input
    state._WILL_CREATE_VIDEO_WRIST_RENDERER = (
        plan.will_create_video_wrist_renderer
    )
    state._WILL_CREATE_POLICY_WRIST_RENDERER = (
        plan.will_create_policy_wrist_renderer
    )
    state._WILL_CREATE_ANY_WRIST_RENDERER = (
        plan.will_create_any_wrist_renderer
    )
    state._VIDEO_WRIST_EPHEMERAL = plan.video_wrist_ephemeral
    state._DEFER_OVERVIEW_RENDER = plan.defer_overview_render
    state._PASS1_OVERVIEW_RENDERER_RELEASED = False
    if state.A.policy_batch_server and not state.A.defer_overview_render:
        print(
            "[sim] policy batch server active: forcing isolated overview EGL pass",
            flush=True,
        )
    if (
        state._LOCKED_POLICY_OBS_VIDEO_INSETS
        and not state.A.policy_obs_inset
    ):
        print(
            "[sim] render lock active: using the saved policy observation for "
            "video wrist insets to avoid a second live EGL owner",
            flush=True,
        )

    state.renderers.ren = None
    if not state.A.no_render and not state._DEFER_OVERVIEW_RENDER:
        with state.render_lock():
            state.renderers.ren = mujoco.Renderer(state.m, 480, 640)
    elif state.POLICY and state.A.no_render:
        print(
            "[sim] no-render policy rollout: overview EGL renderer omitted",
            flush=True,
        )

    state._OVERVIEW_RENDER_DATA = (
        mujoco.MjData(state.m) if not state.A.no_render else None
    )
    state._VIEW_LOOKAT = parse_view_lookat(state, state.A.view_lookat)
    state.ov = mujoco.MjvCamera()
    state.ov.type = mujoco.mjtCamera.mjCAMERA_FREE
    state.ov.azimuth = float(state.A.view_azimuth)  # noqa: V101  # mujoco.MjvCamera field
    state.ov.elevation = float(state.A.view_elevation)  # noqa: V101  # mujoco.MjvCamera field
    state.ov.distance = float(state.A.view_distance)
    state.ov.lookat[:] = state._VIEW_LOOKAT
    state.vopt = mujoco.MjvOption()
    state.vopt.geomgroup[:] = 0
    state.vopt.geomgroup[1] = 1
    state.OVFOV = float(state.m.vis.global_.fovy)
    state._BGOPT = mujoco.MjvOption()
    state._BGOPT.geomgroup[:] = 0

    # GS prefetch is deliberately before every wrist renderer.
    state.SKY = state.GSBG = state.GS_DEPTH = None
    state.GS_OVERVIEW_ATTEMPTS = 0
    if state.gsc is not None and not state.A.no_render:
        with state.render_lock():
            if state.renderers.ren is None:
                state.renderers.ren = mujoco.Renderer(state.m, 480, 640)
            try:
                state.renderers.ren.update_scene(
                    overview_render_data(state),
                    camera=state.ov,
                    scene_option=state._BGOPT,
                )
                state.SKY = state.renderers.ren.render().copy()
                state._gs_overview_payload = state.gsc.camera_payload(
                    state.renderers.ren.scene,
                    640,
                    480,
                    fovy_deg=state.OVFOV,
                )
            finally:
                if state._DEFER_OVERVIEW_RENDER:
                    # Destroy EGL before the later CUDA-backed HTTP wait.
                    state._overview_close_errors = close_renderers(
                        state,
                        ("ren",),
                    )
                    if state._overview_close_errors:
                        raise RuntimeError(
                            "failed to release overview EGL renderer before "
                            "GS prefetch: "
                            + "; ".join(state._overview_close_errors)
                        )
                    state._PASS1_OVERVIEW_RENDERER_RELEASED = True

        if state.A.gs_composite_mode == "depth":
            state._gs_overview = state.gsc.render_payloads_with_depth(
                [state._gs_overview_payload]
            )[0]
            state.GSBG = state._gs_overview.rgb
            state.GS_DEPTH = state._gs_overview.depth
        else:
            state.GSBG = state.gsc.render_payloads(
                [state._gs_overview_payload]
            )[0]
            state.GS_DEPTH = None
        state.GS_OVERVIEW_ATTEMPTS = int(state.gsc.last_render_attempts)
        if state.GSBG is not None and state.GSBG.shape[:2] != state.SKY.shape[:2]:
            state.GSBG = cv.resize(
                state.GSBG,
                (state.SKY.shape[1], state.SKY.shape[0]),
            )
        if (
            state.GS_DEPTH is not None
            and state.GS_DEPTH.shape[:2] != state.SKY.shape[:2]
        ):
            state.GS_DEPTH = cv.resize(
                state.GS_DEPTH,
                (state.SKY.shape[1], state.SKY.shape[0]),
                interpolation=cv.INTER_NEAREST,
            )
        # Preserve the historical redundant assignment after prefetch.
        if state.GS_DEPTH is not None or (
            state.GSBG is not None
            and state.A.gs_composite_mode == "foreground"
        ):
            state._DEFER_OVERVIEW_RENDER = True
        print(
            "[sim] GS backdrop+sky rendered once "
            f"(before wrist renderers; attempts={state.GS_OVERVIEW_ATTEMPTS}; "
            f"composite={state.A.gs_composite_mode})",
            flush=True,
        )

    if state._DEFER_OVERVIEW_RENDER and state.renderers.ren is not None:
        state._overview_close_errors = close_renderers(
            state,
            ("ren",),
        )
        if state._overview_close_errors:
            raise RuntimeError(
                "failed to release pass-1 overview EGL renderer before "
                "wrist setup: "
                + "; ".join(state._overview_close_errors)
            )
        state._PASS1_OVERVIEW_RENDERER_RELEASED = True
    if state._PASS1_OVERVIEW_RENDERER_RELEASED:
        print(
            "[sim] overview output deferred; pass-1 EGL renderer released "
            "before wrist setup",
            flush=True,
        )


__all__ = [
    "OverviewRenderPlan",
    "add_origin",
    "apply_black_background",
    "overview_render_data",
    "parse_view_lookat",
    "resolve_overview_render_plan",
    "setup_overview_render",
]
