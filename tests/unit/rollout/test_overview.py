# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Tests for rollout overview rendering decisions."""

from tests._helpers.worker_ast import *  # noqa: F403

def test_overview_render_plan_truth_table() -> None:
    def resolve(**overrides):
        values = {
            "POLICY": False,
            "_RENDER_LOCK_PATH": "",
            "_STATE_REPLAY_POLICY_INSETS": None,
            "gsc": None,
        }
        args = {
            "policy_image_source": "sim",
            "no_render": False,
            "clean_video": False,
            "no_wrist": True,
            "policy_obs_inset": False,
            "defer_overview_render": False,
            "policy_batch_server": "",
            "gs_composite_mode": "depth",
        }
        for key, value in overrides.items():
            (args if key in args else values)[key] = value
        values["A"] = SimpleNamespace(**args)
        return resolve_overview_render_plan(SimpleNamespace(**values))

    inline = resolve()
    assert not inline.defer_overview_render
    assert not inline.will_create_any_wrist_renderer

    video = resolve(no_wrist=False)
    assert video.will_create_video_wrist_renderer
    assert video.will_create_any_wrist_renderer
    assert video.defer_overview_render

    locked_policy = resolve(
        POLICY=True,
        _RENDER_LOCK_PATH="/tmp/render.lock",
        no_wrist=False,
    )
    assert locked_policy.locked_policy_obs_video_insets
    assert locked_policy.policy_inset_input
    assert not locked_policy.will_create_video_wrist_renderer
    assert locked_policy.will_create_policy_wrist_renderer
    assert locked_policy.defer_overview_render

    saved_insets = resolve(
        _STATE_REPLAY_POLICY_INSETS=object(),
        no_wrist=False,
    )
    assert saved_insets.policy_inset_input
    assert not saved_insets.will_create_video_wrist_renderer

    gs_depth = resolve(gsc=object())
    assert gs_depth.defer_overview_render
