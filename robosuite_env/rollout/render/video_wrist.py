# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Video wrist renderer construction with explicit EGL ownership."""

from __future__ import annotations

from robosuite_env.rendering.rgb_color_calibration import (
    DEFAULT_RGB_COLOR_CALIBRATION,
)
from robosuite_env.rollout.resource_lifecycle import (
    close_renderer_with_retry,
)


class WristPairFactory:
    """Create an R/L pair sharing the right renderer's EGL context."""

    def __init__(
        self,
        state,
        renderer_cls,
        right_prefix,
        left_prefix,
        gs_client,
        black_background,
        render_profile,
        composite_mode,
        stats,
        ephemeral,
        owner_slot,
        label,
        gs_only=False,
    ):
        self._state = state
        self._renderer_cls = renderer_cls
        self._right_prefix = right_prefix
        self._left_prefix = left_prefix
        self._gs_client = gs_client
        self._black_background = black_background
        self._render_profile = render_profile
        self._composite_mode = composite_mode
        self._stats = stats
        self._ephemeral = ephemeral
        self._owner_slot = owner_slot
        self._label = label
        self._gs_only = bool(gs_only)

    def __call__(self):
        right = None
        try:
            right = self._renderer_cls(
                self._state.env,
                prefix=self._right_prefix,
                gs_client=self._gs_client,
                black_background=self._black_background,
                render_lock=self._state.render_lock,
                side="right",
                render_profile=self._render_profile,
                camera_side="right",
                rgb_color_calibration_path=DEFAULT_RGB_COLOR_CALIBRATION,
                composite_mode=self._composite_mode,
                gs_only=self._gs_only,
                backend=self._state.A.policy_wrist_backend,
            )
            if self._ephemeral:
                self._stats["create_count"] += 1
                self._stats["live_count"] += 1
                self._stats["peak_live_count"] = max(
                    self._stats["peak_live_count"], self._stats["live_count"]
                )
            left = self._renderer_cls(
                self._state.env,
                prefix=self._left_prefix,
                gs_client=self._gs_client,
                black_background=self._black_background,
                render_lock=self._state.render_lock,
                side="left",
                render_profile=self._render_profile,
                camera_side="left",
                rgb_color_calibration_path=DEFAULT_RGB_COLOR_CALIBRATION,
                composite_mode=self._composite_mode,
                gs_only=self._gs_only,
                renderer=right.renderer,
                backend=self._state.A.policy_wrist_backend,
                source_renderer=getattr(right, "_source_renderer", None),
            )
        except BaseException as exc:
            if right is not None:
                close_error = close_renderer_with_retry(right)
                if self._ephemeral and close_error is None:
                    self._stats["close_count"] += 1
                    self._stats["live_count"] -= 1
                if close_error is not None:
                    setattr(self._state.renderers, self._owner_slot, right)
                    if hasattr(exc, "add_note"):
                        exc.add_note(
                            f"{self._label} wrist owner cleanup failed after "
                            f"constructor failure: {close_error}"
                        )
            raise
        return right, left


def validate_ephemeral_video_wrist_capture(state, capture, side) -> None:
    """Require a detached GS payload before an ephemeral EGL teardown."""

    if not state._VIDEO_WRIST_EPHEMERAL or state.wgs is None:
        return
    names = getattr(capture, "batch_face_names", None)
    payloads = getattr(capture, "batch_payloads", None)
    if (
        not callable(getattr(capture, "batch_render", None))
        or names is None
        or payloads is None
        or len(names) != len(payloads)
        or not names
    ):
        raise RuntimeError(
            f"ephemeral video wrist {side} did not capture a complete "
            "immutable GS batch before EGL teardown"
        )


def setup_video_wrist(state) -> None:
    """Create persistent wrists or install the per-frame pair factory."""

    state._VIDEO_WRIST_RENDERER_STATS = {
        "create_count": 0,
        "close_count": 0,
        "live_count": 0,
        "peak_live_count": 0,
    }
    if state._WILL_CREATE_VIDEO_WRIST_RENDERER:
        from robosuite_env.rendering.wrist_fisheye import WristFisheye

        # In POLICY mode the display insets match policy OBS unless --no-obs-gs;
        # --wrist-gs still forces GS for replay video.
        state.wgs = (
            state.gsc
            if (state.A.wrist_gs or (state.POLICY and not state.A.no_obs_gs))
            else None
        )
        state._rpre, state._lpre = (
            state.PROFILE.wrist_prefixes["R"],
            state.PROFILE.wrist_prefixes["L"],
        )
        state._black_wrist = bool(
            state.A.black_background and state.wgs is None
        )
        state._wrist_profile = state.A.policy_yubi_render_profile
        if state._VIDEO_WRIST_EPHEMERAL and state.wgs is not None:
            state._video_batch_rgb = callable(
                getattr(state.wgs, "render_payloads", None)
            )
            state._video_batch_rgbd = callable(
                getattr(state.wgs, "render_payloads_with_depth", None)
            )
            state._video_batch_available = (
                state._video_batch_rgb
                if state.A.gs_composite_mode == "foreground"
                else state._video_batch_rgbd
            )
            if not callable(
                getattr(state.wgs, "camera_payload", None)
            ) or not state._video_batch_available:
                raise RuntimeError(
                    "a render-locked video wrist requires an immutable GS "
                    "batch payload API for composite mode "
                    f"{state.A.gs_composite_mode!r}"
                )

        state._new_video_wrist_pair = WristPairFactory(
            state,
            WristFisheye,
            state._rpre,
            state._lpre,
            state.wgs,
            state._black_wrist,
            state._wrist_profile,
            state.A.gs_composite_mode,
            state._VIDEO_WRIST_RENDERER_STATS,
            state._VIDEO_WRIST_EPHEMERAL,
            "wfR",
            "video",
        )

        if not state._VIDEO_WRIST_EPHEMERAL:
            with state.render_lock():
                (
                    state.renderers.wfR,
                    state.renderers.wfL,
                ) = state._new_video_wrist_pair()
        else:
            print(
                "[sim] video wrist renderer ready on demand "
                "(lifecycle=per-frame locked create/capture/close)",
                flush=True,
            )


__all__ = [
    "WristPairFactory",
    "setup_video_wrist",
    "validate_ephemeral_video_wrist_capture",
]
