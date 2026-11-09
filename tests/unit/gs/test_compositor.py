# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pytest

from robosuite_env.rollout.preflight import _apply_gs_definition
from scene.gs_background_client import (
    composite_background_depth,
    composite_background_mask,
    foreground_mask_from_depth,
    render_depth,
    render_depth_and_foreground_mask,
    render_foreground_mask,
)


class GSBackgroundCompositorTest(unittest.TestCase):
    @staticmethod
    def _fake_renderer(depth, *, segmentation_error=False, depth_error=False):
        class Renderer:
            def __init__(self):
                self.scene = types.SimpleNamespace(
                    flags=np.array([7, 0, 0, 9], dtype=np.uint8)
                )
                self._depth_rendering = False
                self._segmentation_rendering = False
                self._model = types.SimpleNamespace(
                    stat=types.SimpleNamespace(extent=2.0),
                    vis=types.SimpleNamespace(
                        map=types.SimpleNamespace(znear=0.005, zfar=50.0)
                    ),
                )
                self.render_calls = 0

            def enable_depth_rendering(self):
                self._segmentation_rendering = False
                self._depth_rendering = True

            def disable_depth_rendering(self):
                self._depth_rendering = False

            def enable_segmentation_rendering(self):
                self._segmentation_rendering = True
                self._depth_rendering = False

            def disable_segmentation_rendering(self):
                self._segmentation_rendering = False

            def render(self):
                self.render_calls += 1
                # Match MuJoCo 3.8.1: render() mutates these flags before a
                # depth / segmentation readback and does not restore them if
                # conversion raises.
                self.scene.flags[1:3] = 1
                if self._segmentation_rendering:
                    if segmentation_error:
                        raise IndexError("invalid encoded segmentation id")
                    result = np.full((*np.asarray(depth).shape, 2), -1, np.int32)
                    result[..., 0] = 3
                    result[..., 1] = 5
                    return result
                if self._depth_rendering:
                    if depth_error:
                        raise RuntimeError("depth readback failed")
                    return np.asarray(depth, dtype=np.float32)
                return np.zeros((*np.asarray(depth).shape, 3), dtype=np.uint8)

        return Renderer()

    def test_depth_capture_restores_modes_and_scene_flags_after_exception(self):
        renderer = self._fake_renderer([[1.0]], depth_error=True)
        renderer._segmentation_rendering = True
        original_flags = renderer.scene.flags.copy()

        with self.assertRaisesRegex(RuntimeError, "depth readback failed"):
            render_depth(renderer)

        self.assertFalse(renderer._depth_rendering)
        self.assertTrue(renderer._segmentation_rendering)
        np.testing.assert_array_equal(renderer.scene.flags, original_flags)

    def test_segmentation_failure_restores_flags_before_depth_fallback(self):
        renderer = self._fake_renderer(
            np.full((2, 3), 2.0, dtype=np.float32),
            segmentation_error=True,
        )
        original_flags = renderer.scene.flags.copy()

        foreground = render_foreground_mask(renderer)

        np.testing.assert_array_equal(
            foreground, np.ones((2, 3), dtype=bool)
        )
        np.testing.assert_array_equal(renderer.scene.flags, original_flags)
        self.assertFalse(renderer._depth_rendering)
        self.assertFalse(renderer._segmentation_rendering)
        self.assertEqual(renderer.render_calls, 2)

    def test_raw_idcolor_mask_bypasses_public_remap_and_restores_state(self):
        class Renderer:
            def __init__(self):
                self.scene = types.SimpleNamespace(
                    flags=np.array([7, 0, 0, 9], dtype=np.uint8)
                )
                self._scene = self.scene
                self._rect = object()
                self._mjr_context = object()
                self._height = 2
                self._width = 3
                self._gl_context = None
                self._depth_rendering = False
                self._segmentation_rendering = False
                self.render_calls = 0

            def enable_depth_rendering(self):
                self._segmentation_rendering = False
                self._depth_rendering = True

            def disable_depth_rendering(self):
                self._depth_rendering = False

            def enable_segmentation_rendering(self):
                self._segmentation_rendering = True
                self._depth_rendering = False

            def disable_segmentation_rendering(self):
                self._segmentation_rendering = False

            def render(self):
                self.render_calls += 1
                raise AssertionError(
                    "raw ID-color occupancy must bypass public segmentation remap"
                )

        renderer = Renderer()
        original_flags = renderer.scene.flags.copy()
        raw_id_colors = np.array(
            [
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                [[0, 0, 1], [255, 255, 255], [0, 0, 0]],
            ],
            dtype=np.uint8,
        )

        def fake_mjr_render(rect, scene, context):
            self.assertIs(rect, renderer._rect)
            self.assertIs(scene, renderer._scene)
            self.assertIs(context, renderer._mjr_context)
            self.assertTrue(renderer._segmentation_rendering)
            self.assertEqual(int(scene.flags[1]), 1)
            self.assertEqual(int(scene.flags[2]), 1)

        def fake_mjr_read_pixels(rgb, depth, rect, context):
            self.assertIsNone(depth)
            self.assertIs(rect, renderer._rect)
            self.assertIs(context, renderer._mjr_context)
            rgb[:] = raw_id_colors

        fake_mujoco = types.SimpleNamespace(
            mjtRndFlag=types.SimpleNamespace(
                mjRND_SEGMENT=1,
                mjRND_IDCOLOR=2,
            ),
            mjr_render=mock.Mock(side_effect=fake_mjr_render),
            mjr_readPixels=mock.Mock(side_effect=fake_mjr_read_pixels),
        )
        with mock.patch.dict(sys.modules, {"mujoco": fake_mujoco}):
            foreground = render_foreground_mask(renderer)

        np.testing.assert_array_equal(
            foreground,
            np.array(
                [[False, True, True], [True, True, False]], dtype=bool
            ),
        )
        fake_mujoco.mjr_render.assert_called_once()
        fake_mujoco.mjr_readPixels.assert_called_once()
        self.assertEqual(renderer.render_calls, 0)
        self.assertFalse(renderer._depth_rendering)
        self.assertFalse(renderer._segmentation_rendering)
        np.testing.assert_array_equal(renderer.scene.flags, original_flags)

    def test_combined_depth_mask_uses_depth_plus_segmentation(self):
        renderer = self._fake_renderer([[2.0, 100.0, np.nan, 0.0]])
        original_flags = renderer.scene.flags.copy()

        depth, foreground = render_depth_and_foreground_mask(renderer)

        np.testing.assert_allclose(
            depth, [[2.0, 100.0, np.nan, 0.0]], equal_nan=True
        )
        np.testing.assert_array_equal(
            foreground, [[True, True, True, True]]
        )
        self.assertEqual(renderer.render_calls, 2)
        np.testing.assert_array_equal(renderer.scene.flags, original_flags)

    def test_constant_depth_without_model_is_not_all_background(self):
        foreground = foreground_mask_from_depth(
            np.full((2, 3), 2.0, dtype=np.float32), far_clip=None
        )
        np.testing.assert_array_equal(
            foreground, np.ones((2, 3), dtype=bool)
        )
        rgb = np.full((2, 3, 3), 11, dtype=np.uint8)
        background = np.full((2, 3, 3), 22, dtype=np.uint8)

        output = composite_background_mask(rgb, foreground, background)

        np.testing.assert_array_equal(output, rgb)

    def test_reverse_z_clear_depth_with_large_clip_ratio_is_background(self):
        renderer = self._fake_renderer(
            [[99.99341, 99.0]], segmentation_error=True
        )

        _depth, foreground = render_depth_and_foreground_mask(renderer)

        np.testing.assert_array_equal(foreground, [[False, True]])

    def test_depth_compositor_fails_safe_for_invalid_mujoco_depth(self):
        mujoco_rgb = np.full((1, 5, 3), 11, dtype=np.uint8)
        gs_rgb = np.full((1, 5, 3), 22, dtype=np.uint8)
        output = composite_background_depth(
            mujoco_rgb,
            np.array([[1.0, np.nan, 0.0, -1.0, 3.0]], dtype=np.float32),
            np.ones((1, 5), dtype=bool),
            gs_rgb,
            np.full((1, 5), 2.0, dtype=np.float32),
        )

        np.testing.assert_array_equal(
            output[0, :, 0], np.array([11, 11, 11, 11, 22], dtype=np.uint8)
        )
        mujoco_rgb = np.full((1, 4, 3), 11, dtype=np.uint8)
        gs_rgb = np.full((1, 4, 3), 22, dtype=np.uint8)
        output = composite_background_depth(
            mujoco_rgb,
            np.full((1, 4), 5.0, dtype=np.float32),
            np.ones((1, 4), dtype=bool),
            gs_rgb,
            np.array([[np.inf, np.nan, 0.0, -1.0]], dtype=np.float32),
        )

        np.testing.assert_array_equal(output, mujoco_rgb)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scene_definition_selects_safe_foreground_composite(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["sim.py"])
    args = SimpleNamespace(
        gs="",
        gs_background_source="",
        gs_background_name="",
        gs_composite_mode="depth",
        gs_port=8097,
        gs_rot="",
        gs_rot_R=None,
        gs_pivot=None,
        gs_pivz=0.0,
        gs_mat_w2gs=None,
        gs_w2gs="",
    )

    _apply_gs_definition(
        args,
        {
            "name": "task-object-ghost-scene",
            "ply_path": "scene.ply",
            "composite_mode": "foreground_mask",
        },
        str(tmp_path),
        "test-scene",
    )

    assert args.gs_composite_mode == "foreground"


def test_explicit_composite_cli_overrides_scene_definition(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["sim.py", "--gs-composite-mode", "depth"],
    )
    args = SimpleNamespace(
        gs="",
        gs_background_source="",
        gs_background_name="",
        gs_composite_mode="depth",
        gs_port=8097,
        gs_rot="",
        gs_rot_R=None,
        gs_pivot=None,
        gs_pivz=0.0,
        gs_mat_w2gs=None,
        gs_w2gs="",
    )

    _apply_gs_definition(
        args,
        {"ply_path": "scene.ply", "composite_mode": "foreground"},
        str(tmp_path),
        "test-scene",
    )

    assert args.gs_composite_mode == "depth"


def test_scene_definition_pins_world_to_gs_sha256(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["sim.py"])
    alignment = tmp_path / "mujoco_to_gs.npy"
    np.save(alignment, np.eye(4, dtype=np.float64))
    args = SimpleNamespace(
        gs="",
        gs_background_source="",
        gs_background_name="",
        gs_composite_mode="depth",
        gs_port=8097,
        gs_rot="",
        gs_rot_R=None,
        gs_pivot=None,
        gs_pivz=0.0,
        gs_mat_w2gs=None,
        gs_w2gs="",
    )

    _apply_gs_definition(
        args,
        {
            "ply_path": "scene.ply",
            "w2gs": alignment.name,
            "w2gs_sha256": _sha256(alignment),
        },
        str(tmp_path),
        "test-scene",
    )

    assert args.gs_w2gs == str(alignment)


def test_scene_definition_rejects_world_to_gs_sha256_mismatch(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(sys, "argv", ["sim.py"])
    alignment = tmp_path / "mujoco_to_gs.npy"
    np.save(alignment, np.eye(4, dtype=np.float64))
    args = SimpleNamespace(
        gs="",
        gs_background_source="",
        gs_background_name="",
        gs_composite_mode="depth",
        gs_port=8097,
        gs_rot="",
        gs_rot_R=None,
        gs_pivot=None,
        gs_pivz=0.0,
        gs_mat_w2gs=None,
        gs_w2gs="",
    )

    with pytest.raises(SystemExit, match="world->GS SHA-256 mismatch"):
        _apply_gs_definition(
            args,
            {
                "ply_path": "scene.ply",
                "w2gs": alignment.name,
                "w2gs_sha256": "0" * 64,
            },
            str(tmp_path),
            "test-scene",
        )


def test_real_compositor_all_false_mask_is_pure_gs() -> None:
    mujoco_rgb = np.full((2, 3, 3), 11, dtype=np.uint8)
    gs_rgb = np.full((2, 3, 3), 29, dtype=np.uint8)
    output = composite_background_depth(
        mujoco_rgb,
        np.full((2, 3), 0.5, dtype=np.float32),
        np.zeros((2, 3), dtype=bool),
        gs_rgb,
        np.full((2, 3), 1.0, dtype=np.float32),
    )

    np.testing.assert_array_equal(output, gs_rgb)


def test_real_compositor_nearer_mujoco_is_pure_mujoco() -> None:
    mujoco_rgb = np.full((2, 3, 3), 11, dtype=np.uint8)
    gs_rgb = np.full((2, 3, 3), 29, dtype=np.uint8)
    output = composite_background_depth(
        mujoco_rgb,
        np.full((2, 3), 0.5, dtype=np.float32),
        np.ones((2, 3), dtype=bool),
        gs_rgb,
        np.full((2, 3), 1.0, dtype=np.float32),
    )

    np.testing.assert_array_equal(output, mujoco_rgb)


def test_real_compositor_two_mm_tie_break() -> None:
    mujoco_rgb = np.full((1, 2, 3), 11, dtype=np.uint8)
    gs_rgb = np.full((1, 2, 3), 29, dtype=np.uint8)
    output = composite_background_depth(
        mujoco_rgb,
        np.array([[1.0019, 1.0021]], dtype=np.float64),
        np.ones((1, 2), dtype=bool),
        gs_rgb,
        np.ones((1, 2), dtype=np.float64),
    )

    np.testing.assert_array_equal(output[0, 0], mujoco_rgb[0, 0])
    np.testing.assert_array_equal(output[0, 1], gs_rgb[0, 1])


def test_real_compositor_missing_gs_depth_makes_mujoco_win_mask() -> None:
    mujoco_rgb = np.full((1, 3, 3), 11, dtype=np.uint8)
    gs_rgb = np.full((1, 3, 3), 29, dtype=np.uint8)
    foreground = np.array([[True, False, True]])
    output = composite_background_depth(
        mujoco_rgb,
        np.full((1, 3), 999.0, dtype=np.float32),
        foreground,
        gs_rgb,
        None,
    )

    np.testing.assert_array_equal(output[0, 0], mujoco_rgb[0, 0])
    np.testing.assert_array_equal(output[0, 1], gs_rgb[0, 1])
    np.testing.assert_array_equal(output[0, 2], mujoco_rgb[0, 2])
