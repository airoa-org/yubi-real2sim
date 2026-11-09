# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import ast
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from robosuite_env.rendering.wrist_fisheye import (
    WristFisheye,
    WristFisheyePending,
    _CapturedWristFace,
    _center_scale_crop,
)
from robosuite_env.rendering import fisheye_projection
from robosuite_env.rendering.fisheye_model import FisheyeIntrinsics
from robosuite_env.rollout.observe.wrist import setup_policy_wrist
from robosuite_env.rollout.render.video_wrist import setup_video_wrist


REPO_ROOT = Path(__file__).resolve().parents[3]


class _FakeRenderer:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _FailOnceRenderer(_FakeRenderer):
    def close(self):
        super().close()
        if self.close_calls == 1:
            raise RuntimeError("transient close failure")


class _RecordingRenderer:
    """MuJoCo renderer stand-in that records scene updates and refuses reuse after ``close``."""

    def __init__(self, events):
        self.events = events
        self.scene = SimpleNamespace(nlight=0, camera=None)
        self.closed = False

    def update_scene(self, _data, camera, scene_option):
        del scene_option
        if self.closed:
            raise AssertionError("closed renderer was reused")
        self.scene.camera = camera
        self.events.append(f"update:{camera}")

    def close(self):
        self.closed = True


class _SourceRenderer:
    """Direct KB4 source renderer stand-in with a 3x2 raster."""

    camera = FisheyeIntrinsics(3, 2, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0)
    theta_max = 1.0

    def __init__(self, events):
        self.events = events

    def render(self, scene):
        if scene.camera is None:
            raise AssertionError("the source camera must be selected first")
        self.events.append("source")
        return SimpleNamespace(
            rgb=np.zeros((2, 3, 3), dtype=np.uint8),
            segid=np.zeros((2, 3), dtype=np.int32),
            distance=np.ones((2, 3), dtype=np.float32),
        )


def _radial_result(value, depth):
    return SimpleNamespace(
        rgb=np.full((2, 3, 3), value, dtype=np.uint8),
        depth=np.full((2, 3), depth, dtype=np.float32),
        depth_convention="radial",
    )


def _direct_wrist(events, gs_client, render_lock):
    """A direct wrist with test doubles for MuJoCo, the source renderer and the GS client."""

    wrist = WristFisheye.__new__(WristFisheye)
    wrist.env = mock.Mock()
    wrist.env.sim.data._data = object()
    wrist.m = SimpleNamespace()
    wrist.face_cam = {"front": "front_cam"}
    wrist.black_background = False
    wrist.renderer = _RecordingRenderer(events)
    wrist._owns_renderer = False
    wrist._source_renderer = _SourceRenderer(events)
    wrist._owns_source_renderer = False
    wrist.gs = gs_client
    wrist._render_lock = render_lock
    wrist._profile_camera_positions = {}
    wrist._profile_excluded_geom_ids = ()
    wrist.source_height, wrist.source_width = 2, 3
    wrist.valid = np.ones((2, 3), dtype=bool)
    wrist.camera_model = _SourceRenderer.camera
    wrist.rgb_color_calibration = None
    return wrist


class WristFisheyeTest(unittest.TestCase):
    def test_policy_wrist_factory_receives_obs_gs_only(self):
        gs_client = SimpleNamespace(
            camera_payload=lambda *_args, **_kwargs: {},
            render_payloads_with_depth=lambda _payloads: (),
        )
        state = SimpleNamespace(
            A=SimpleNamespace(
                policy_pose_source="sim",
                policy_image_source="sim",
                no_obs_gs=False,
                obs_gs_only=True,
                black_background=False,
                policy_yubi_render_profile="policy-v1",
                gs_composite_mode="depth",
            ),
            POLICY=True,
            LOGDIR="logs",
            gsc=gs_client,
            PROFILE=SimpleNamespace(
                wrist_prefixes={"R": "right", "L": "left"}
            ),
            _RENDER_LOCK_PATH="render.lock",
            renderers=SimpleNamespace(),
            render_lock=mock.Mock(),
        )

        with mock.patch(
            "robosuite_env.rollout.observe.wrist.WristPairFactory"
        ) as factory:
            setup_policy_wrist(state)

        self.assertTrue(state._POLICY_WRIST_EPHEMERAL)
        self.assertIs(state._ogs, gs_client)
        self.assertTrue(factory.call_args.kwargs["gs_only"])

    def test_video_wrists_render_the_selected_policy_profile(self):
        state = SimpleNamespace(
            A=SimpleNamespace(
                wrist_gs=False,
                no_obs_gs=False,
                black_background=False,
                policy_yubi_render_profile="elp-mean-wrist-rgb",
                gs_composite_mode="depth",
            ),
            POLICY=True,
            gsc=None,
            PROFILE=SimpleNamespace(
                wrist_prefixes={"R": "right", "L": "left"}
            ),
            _WILL_CREATE_VIDEO_WRIST_RENDERER=True,
            _VIDEO_WRIST_EPHEMERAL=True,
        )

        with mock.patch(
            "robosuite_env.rollout.render.video_wrist.WristPairFactory"
        ) as factory:
            setup_video_wrist(state)

        self.assertEqual(state._wrist_profile, "elp-mean-wrist-rgb")
        self.assertEqual(factory.call_args.args[6], "elp-mean-wrist-rgb")

    def test_gs_only_forces_false_mask_into_depth_compositor(self):
        mujoco_rgb = np.full((2, 3, 3), 17, dtype=np.uint8)
        gs_rgb = np.full((2, 3, 3), 83, dtype=np.uint8)
        seen_masks = []

        def composite_depth(
            _mujoco_rgb,
            _mujoco_depth,
            foreground_mask,
            background,
            _background_depth,
        ):
            seen_masks.append(np.asarray(foreground_mask).copy())
            return background

        wrist = WristFisheye.__new__(WristFisheye)
        wrist.gs_only = True
        wrist._composite_depth = composite_depth
        wrist._composite = mock.Mock(
            side_effect=AssertionError("depth compositor must be used")
        )
        wrist.valid = np.ones((2, 3), dtype=bool)
        wrist.camera_model = object()
        wrist.rgb_color_calibration = None
        pending = WristFisheyePending(
            faces=(
                _CapturedWristFace(
                    name="source",
                    rgb=mujoco_rgb,
                    foreground_mask=np.ones((2, 3), dtype=bool),
                    background=gs_rgb,
                    mujoco_depth=np.ones((2, 3), dtype=np.float32),
                    background_depth=np.full((2, 3), 2.0, dtype=np.float32),
                ),
            )
        )

        image = wrist.finalize_cpu_stage(pending)

        np.testing.assert_array_equal(image, gs_rgb)
        self.assertEqual(len(seen_masks), 1)
        self.assertEqual(seen_masks[0].dtype, np.dtype(bool))
        self.assertFalse(np.any(seen_masks[0]))

    def test_projection_calibration_matches_training_simulator(self):
        source = (
            REPO_ROOT
            / "tests"
            / "fixtures"
            / "franka_yubi_sim"
            / "build_scene.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        values = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.startswith("CALIB_"):
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except (TypeError, ValueError):
                    continue
        expected = {
            "CALIB_IMAGE_WIDTH": fisheye_projection.OUT_W,
            "CALIB_IMAGE_HEIGHT": fisheye_projection.OUT_H,
            "CALIB_FX": fisheye_projection.FX,
            "CALIB_FY": fisheye_projection.FY,
            "CALIB_CX": fisheye_projection.CX,
            "CALIB_CY": fisheye_projection.CY,
            "CALIB_K1": fisheye_projection.K1,
            "CALIB_K2": fisheye_projection.K2,
            "CALIB_K3": fisheye_projection.K3,
            "CALIB_K4": fisheye_projection.K4,
            "CALIB_FULL_FOV_DEG": fisheye_projection.FULL_FOV_DEG,
        }
        self.assertEqual({key: values[key] for key in expected}, expected)

    def test_horizontal_edge_crop_places_optical_axis_at_requested_fraction(self):
        image = np.arange(5, dtype=np.uint8)[None, :, None]
        image = np.repeat(np.repeat(image, 3, axis=0), 3, axis=2)
        scaled = cv2.resize(image, (10, 3), interpolation=cv2.INTER_LINEAR)

        result = _center_scale_crop(
            image,
            2.0,
            1.0,
            crop_x_center_fraction=0.4,
        )

        # scaled optical axis x=5 lands at output x=2 (40% of width), so the
        # five-pixel crop starts at scaled x=3.
        np.testing.assert_array_equal(result, scaled[:, 3:8])

    def test_close_is_idempotent(self):
        for owns_renderer, expected_close_calls in ((True, 1), (False, 0)):
            with self.subTest(owns_renderer=owns_renderer):
                renderer = _FakeRenderer()
                wrist = WristFisheye.__new__(WristFisheye)
                wrist.renderer = renderer
                wrist._owns_renderer = owns_renderer

                wrist.close()
                wrist.close()

                self.assertIsNone(wrist.renderer)
                self.assertEqual(renderer.close_calls, expected_close_calls)

    def test_close_retains_renderer_until_release_succeeds(self):
        renderer = _FailOnceRenderer()
        wrist = WristFisheye.__new__(WristFisheye)
        wrist.renderer = renderer
        wrist._owns_renderer = True

        with self.assertRaisesRegex(RuntimeError, "transient close failure"):
            wrist.close()
        self.assertIs(wrist.renderer, renderer)

        wrist.close()
        self.assertIsNone(wrist.renderer)
        self.assertEqual(renderer.close_calls, 2)

    def test_injected_renderer_is_reused_without_constructing_another_context(self):
        profile = SimpleNamespace(
            name="shared-renderer-test",
            source_fov_deg=178.0,
            source_image_hw=(2, 3),
            base_face_size_178=2,
            camera=object(),
        )
        model = SimpleNamespace(
            ngeom=0,
            vis=SimpleNamespace(
                global_=SimpleNamespace(offwidth=1, offheight=1)
            ),
        )
        env = SimpleNamespace(
            sim=SimpleNamespace(model=SimpleNamespace(_model=model))
        )
        renderer_factory = mock.Mock(
            side_effect=AssertionError(
                "an injected renderer must not allocate another EGL context"
            )
        )
        fake_mujoco = SimpleNamespace(
            Renderer=renderer_factory,
            mjtObj=SimpleNamespace(mjOBJ_CAMERA=1, mjOBJ_GEOM=2),
            mj_name2id=lambda _model, _obj_type, _name: 0,
        )
        injected = _FakeRenderer()
        source_camera = object()
        shared_source = SimpleNamespace(camera=source_camera, valid=np.ones((2, 3), dtype=bool))

        with (
            mock.patch(
                "robosuite_env.rendering.wrist_fisheye.mujoco", fake_mujoco
            ),
            mock.patch(
                "robosuite_env.rendering.wrist_fisheye.load_rgb_color_calibration",
                return_value=None,
            ),
            mock.patch(
                "robosuite_env.rendering.wrist_fisheye.get_wrist_render_profile",
                return_value=profile,
            ),
            mock.patch(
                "robosuite_env.rendering.wrist_fisheye.wrist_source_intrinsics",
                return_value=source_camera,
            ),
            mock.patch.object(
                WristFisheye,
                "_configure_versioned_wrist_profile",
                return_value=None,
            ),
        ):
            wrist = WristFisheye(
                env,
                prefix="L",
                render_profile="shared-renderer-test",
                camera_side="left",
                renderer=injected,
                backend="direct_kb4",
                source_renderer=shared_source,
            )

        renderer_factory.assert_not_called()
        self.assertIs(wrist.renderer, injected)
        self.assertFalse(wrist._owns_renderer)
        self.assertIs(wrist._source_renderer, shared_source)
        self.assertFalse(wrist._owns_source_renderer)

    def test_gs_camera_snapshot_is_taken_under_the_lock_and_sent_after_it(self):
        events = []
        lock_active = [False]

        @contextmanager
        def render_transaction():
            self.assertFalse(lock_active[0])
            lock_active[0] = True
            events.append("lock_enter")
            try:
                yield
            finally:
                events.append("lock_exit")
                lock_active[0] = False

        class _BatchGsClient:
            def camera_payload(self, scene, width, height, camera_settings):
                self.assert_locked = lock_active[0]
                events.append(f"snapshot:{scene.camera}")
                return {"camera": scene.camera, "width": width, "height": height, **camera_settings}

            def render_payloads_with_depth(self, payloads):
                self.assert_request_unlocked = not lock_active[0]
                events.append("depth_batch:" + ",".join(payload["camera"] for payload in payloads))
                return [_radial_result(1, 2.0) for _payload in payloads]

        wrist = _direct_wrist(events, _BatchGsClient(), render_transaction)
        wrist._composite = lambda *_args: self.fail(
            "mask compositor must not be used when both depths are available"
        )
        wrist._composite_depth = (
            lambda _rgb, _mujoco_depth, _mask, background, _gs_depth: background
        )

        image = wrist.render()

        np.testing.assert_array_equal(image, np.ones((2, 3, 3), dtype=np.uint8))
        self.assertTrue(wrist.gs.assert_locked)
        self.assertTrue(wrist.gs.assert_request_unlocked)
        self.assertEqual(
            events,
            ["lock_enter", "update:front_cam", "source", "snapshot:front_cam", "lock_exit", "depth_batch:front_cam"],
        )

    def test_immutable_batch_completion_does_not_reuse_closed_renderer(self):
        events = []

        class _BatchGsClient:
            def camera_payload(self, scene, width, height, camera_settings):
                return {"camera": str(scene.camera), "width": width, "height": height, **camera_settings}

            def render_payloads_with_depth(self, payloads):
                self.payload_cameras = [item["camera"] for item in payloads]
                self.renderer_was_closed = wrist.renderer is None
                return [_radial_result(1, 2.0) for _payload in payloads]

        wrist = _direct_wrist(events, _BatchGsClient(), contextmanager(lambda: (yield)))
        renderer = wrist.renderer
        wrist._owns_renderer = True
        wrist._composite = lambda *_args: self.fail(
            "depth compositor should be selected"
        )
        wrist._composite_depth = (
            lambda _rgb, _mujoco_depth, _mask, background, _gs_depth: background
        )

        capture = wrist.capture_mujoco_stage()
        renderer.scene.camera = "mutated_after_capture"
        wrist.close()
        pending = wrist.complete_gs_stage(capture)
        image = wrist.finalize_cpu_stage(pending)

        self.assertTrue(renderer.closed)
        self.assertTrue(wrist.gs.renderer_was_closed)
        self.assertEqual(wrist.gs.payload_cameras, ["front_cam"])
        np.testing.assert_array_equal(image, np.ones((2, 3, 3), dtype=np.uint8))

    def test_shared_model_state_is_snapshotted_after_lock_acquisition(self):
        events = []

        class _Model:
            cam_pos = np.asarray([[99.0, 99.0, 99.0]], dtype=np.float64)
            geom_group = np.asarray([9], dtype=np.int32)

        model = _Model()

        @contextmanager
        def render_transaction():
            # Simulate a previous transaction restoring the real model state
            # immediately before this transaction acquires ownership.
            model.cam_pos[0] = [1.0, 2.0, 3.0]
            model.geom_group[0] = 1
            events.append("lock_enter")
            try:
                yield
            finally:
                events.append("lock_exit")

        wrist = _direct_wrist(events, None, render_transaction)
        wrist.m = model
        wrist._profile_camera_positions = {0: np.asarray([4.0, 5.0, 6.0])}
        wrist._profile_excluded_geom_ids = (0,)

        with mock.patch("robosuite_env.rendering.wrist_fisheye.mujoco.mj_camlight") as refresh:
            wrist.render()
        self.assertEqual(refresh.call_count, 2)

        np.testing.assert_array_equal(model.cam_pos[0], [1.0, 2.0, 3.0])
        self.assertEqual(int(model.geom_group[0]), 1)
        self.assertEqual(events, ["lock_enter", "update:front_cam", "source", "lock_exit"])

if __name__ == "__main__":
    unittest.main()


from dataclasses import FrozenInstanceError
import unittest

from robosuite_env.rendering.wrist_render_profiles import (
    FRANKA_PINHOLE_WRIST_RGB_PROFILE,
    get_wrist_render_profile,
)


class WristRenderProfilesTest(unittest.TestCase):
    def test_selected_franka_profile_remains_available(self):
        profile = get_wrist_render_profile(FRANKA_PINHOLE_WRIST_RGB_PROFILE)
        self.assertEqual(profile.contract_version, 2)
        self.assertEqual(profile.source_fov_deg, 200.0)
        self.assertEqual(profile.source_image_hw, (539, 719))
        self.assertEqual(profile.base_face_size_178, 640)
        self.assertEqual(profile.camera_backward_m, 0.010)

    def test_right_wrist_uses_selected_asymmetric_horizontal_crop(self):
        left_crop = get_wrist_render_profile(FRANKA_PINHOLE_WRIST_RGB_PROFILE).crop("left")
        self.assertEqual(left_crop.left_edge_fov_deg, 160.0)
        self.assertEqual(left_crop.right_edge_fov_deg, 160.0)
        self.assertEqual(left_crop.top_edge_fov_deg, 150.0)
        self.assertEqual(left_crop.bottom_edge_fov_deg, 160.0)
        self.assertEqual(left_crop.horizontal_optical_axis_fraction, 0.5)

        crop = get_wrist_render_profile(FRANKA_PINHOLE_WRIST_RGB_PROFILE).crop("right")
        self.assertEqual(crop.left_edge_fov_deg, 155.0)
        self.assertEqual(crop.right_edge_fov_deg, 165.0)
        self.assertEqual(crop.top_edge_fov_deg, 155.0)
        self.assertEqual(crop.bottom_edge_fov_deg, 155.0)
        self.assertEqual(crop.effective_horizontal_fov_deg, 160.0)
        self.assertEqual(crop.effective_vertical_fov_deg, 155.0)
        self.assertEqual(crop.horizontal_optical_axis_fraction, 0.484375)
        self.assertEqual(crop.vertical_optical_axis_fraction, 0.5)

    def test_profile_is_immutable_and_old_names_fail(self):
        profile = get_wrist_render_profile(FRANKA_PINHOLE_WRIST_RGB_PROFILE)
        with self.assertRaises(FrozenInstanceError):
            profile.source_fov_deg = 178.0
        for old_name in (
            "legacy",
            "yubi-white-wrist-20260729",
            "franka-pinhole-wrist-rgb-v1",
        ):
            with self.subTest(old_name=old_name):
                with self.assertRaisesRegex(ValueError, "only supported profile"):
                    get_wrist_render_profile(old_name)


if __name__ == "__main__":
    unittest.main()


import unittest


from robosuite_env.rendering.rgb_color_calibration import (
    apply_rgb_color_calibration,
    load_rgb_color_calibration,
)


class RGBColorCalibrationTest(unittest.TestCase):
    def test_checked_in_asset_is_shared_and_black_preserving(self):
        calibration = load_rgb_color_calibration()
        self.assertEqual(calibration.space, "srgb_uint8")
        image = np.asarray(
            [[[0, 0, 0], [100, 100, 100], [255, 255, 255]]],
            dtype=np.uint8,
        )
        corrected = apply_rgb_color_calibration(image, calibration)
        np.testing.assert_array_equal(corrected[0, 0], [0, 0, 0])
        self.assertGreater(int(corrected[0, 1, 1]), int(image[0, 1, 1]))
        self.assertGreater(int(corrected[0, 1, 2]), int(image[0, 1, 2]))
        np.testing.assert_array_equal(corrected[0, 2], [255, 255, 255])


if __name__ == "__main__":
    unittest.main()
