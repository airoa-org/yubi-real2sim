# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Lazy 3DGS client construction and world-frame alignment."""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import numpy as np


def load_gs_background_api():
    """Load the repository-local 3DGS API only when a GS scene is active."""

    from scene.gs_background_client import (
        GSBackgroundClient,
        compose_world_to_gs,
        composite_background_depth,
        composite_background_mask,
        render_depth_and_foreground_mask,
        render_foreground_mask,
    )

    return SimpleNamespace(
        GSBackgroundClient=GSBackgroundClient,
        compose_world_to_gs=compose_world_to_gs,
        composite_background_depth=composite_background_depth,
        composite_background_mask=composite_background_mask,
        render_depth_and_foreground_mask=render_depth_and_foreground_mask,
        render_foreground_mask=render_foreground_mask,
    )


def setup_gs_client(state) -> None:
    """Create the fixed-background client before any wrist EGL renderer."""

    state.gsc = None
    state.gs_api = None
    state.GS_RAW_MAT_W2GS = None
    state.GS_EFFECTIVE_MAT_W2GS = None
    if state.GS_URL:
        state.gs_api = load_gs_background_api()
        if state.A.gs_render_timeout_sec <= 0.0:
            raise SystemExit("[sim] --gs-render-timeout-sec must be > 0")
        if state.A.gs_render_attempts <= 0:
            raise SystemExit("[sim] --gs-render-attempts must be > 0")
        if state.A.gs_render_retry_delay_sec < 0.0:
            raise SystemExit("[sim] --gs-render-retry-delay-sec must be >= 0")
        state.gsc = state.gs_api.GSBackgroundClient(
            url=state.GS_URL,
            ready_token=state.A.gs_server_token,
            fovy_deg=90.0,
            timeout_sec=float(state.A.gs_render_timeout_sec),
            max_attempts=int(state.A.gs_render_attempts),
            strict=True,
            retry_delay_sec=float(state.A.gs_render_retry_delay_sec),
            color_gain_rgb=np.asarray(state.A.gs_color_gain_rgb, float),
            color_gamma_rgb=np.asarray(state.A.gs_color_gamma_rgb, float),
            kb4_near_cull=state.A.gs_kb4_near_cull,
        )  # GS backdrop is FIXED (--world-offset moves only the EE target)
        if state.A.gs_rot_R is not None:
            state.gsc.rot_R = np.asarray(state.A.gs_rot_R, float)
            state.gsc.rot_piv = np.asarray(
                state.A.gs_pivot
                if state.A.gs_pivot is not None
                else [0.0, 0.0, state.A.gs_pivz],
                float,
            )
        elif state.A.gs_rot:
            state.ax = state.A.gs_rot[0]
            state.dg = math.radians(float(state.A.gs_rot[1:]))
            c = math.cos(state.dg)
            s = math.sin(state.dg)
            state.gsc.rot_R = {
                "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
                "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
                "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
            }[state.ax]
            state.gsc.rot_piv = np.asarray(
                state.A.gs_pivot
                if state.A.gs_pivot is not None
                else [0.0, 0.0, state.A.gs_pivz],
                float,
            )
        if state.A.gs_mat_w2gs is not None:
            state.GS_RAW_MAT_W2GS = np.asarray(state.A.gs_mat_w2gs, float)
            if state.A.gs_w2gs:
                print(
                    "[sim] GS hand-alignment loaded: validated scene contract "
                    f"file {state.A.gs_w2gs}",
                    flush=True,
                )
            else:
                print(
                    "[sim] GS hand-alignment loaded: inline matrix from "
                    "background definition",
                    flush=True,
                )
        elif state.A.gs_w2gs:
            if os.path.isfile(state.A.gs_w2gs):
                state.GS_RAW_MAT_W2GS = np.load(state.A.gs_w2gs)
                print(
                    f"[sim] GS hand-alignment loaded: {state.A.gs_w2gs}",
                    flush=True,
                )
            else:
                print(
                    "[sim] warning: GS alignment not found, using "
                    f"rotation/identity: {state.A.gs_w2gs}",
                    flush=True,
                )
        state._gs_calibration_world_offset = np.asarray(
            state.A.gs_calibration_world_offset, float
        )
        if state.GS_RAW_MAT_W2GS is None and np.any(
            state._gs_calibration_world_offset != 0.0
        ):
            raise SystemExit(
                "[sim] --gs-calibration-world-offset requires a valid "
                "world->GS matrix"
            )
        if state.GS_RAW_MAT_W2GS is not None:
            try:
                state.GS_EFFECTIVE_MAT_W2GS = state.gs_api.compose_world_to_gs(
                    state.GS_RAW_MAT_W2GS,
                    state._gs_calibration_world_offset,
                )
            except ValueError as exc:
                raise SystemExit(
                    f"[sim] invalid GS world alignment: {exc}"
                ) from exc
            state.gsc.mat_w2gs = state.GS_EFFECTIVE_MAT_W2GS
            print(
                "[sim] GS calibration-world offset applied before world->GS: "
                f"{state._gs_calibration_world_offset.tolist()}",
                flush=True,
            )

    # Wrist renderers are created later. CUDA-backed GS interaction can corrupt
    # a wrist EGL renderer that already exists, so this assignment stays here.
    state.renderers.wfR = state.renderers.wfL = None


__all__ = ["load_gs_background_api", "setup_gs_client"]
