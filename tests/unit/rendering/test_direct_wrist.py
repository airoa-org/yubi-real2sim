# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from robosuite_env.rendering.wrist_fisheye import (
    WristFisheye,
    WristFisheyePending,
    _CapturedWristFace,
)


SCENE = """
<mujoco>
  <visual><global offwidth="720" offheight="720"/><quality offsamples="0"/></visual>
  <worldbody>
    <light directional="true" castshadow="false"/>
    <geom name="object" type="box" pos="0 0 0" size=".2 .2 .05" group="1" rgba="1 .2 .1 1"/>
    <geom name="geometry42_visual" type="sphere" pos="0 0 .9" size=".03" group="1"/>
    <camera name="test_wrist_fisheye_front" pos="0 0 1"/>
    <camera name="other_wrist_fisheye_front" pos=".1 0 1"/>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize("exception", (False, True))
def test_profile_transaction_refreshes_and_restores_real_camera_cache(mujoco, exception):
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "test_wrist_fisheye_front")
    scene = mujoco.MjvScene(model, maxgeom=16)
    owner = SimpleNamespace(_model=model, _scene=scene, _scene_option=mujoco.MjvOption())
    original = data.cam_xpos.copy()
    positions = model.cam_pos.copy()
    groups = model.geom_group.copy()
    qpos, qvel, time = data.qpos.copy(), data.qvel.copy(), data.time
    wrist = WristFisheye.__new__(WristFisheye)
    wrist.m = model
    wrist.env = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(_data=data)))
    wrist._render_lock = nullcontext
    wrist._profile_camera_positions = {camera: positions[camera] + [0, 0, .010]}
    wrist._profile_excluded_geom_ids = (1,)

    with pytest.raises(RuntimeError, match="test exception") if exception else nullcontext():
        with wrist._profile_transaction():
            mujoco.Renderer.update_scene(owner, data, camera=camera)
            observed = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
            np.testing.assert_allclose(observed.pos, original[camera] + [0, 0, .010], atol=1e-6)
            np.testing.assert_allclose(data.cam_xpos[camera], original[camera] + [0, 0, .010], atol=1e-12)
            assert model.geom_group[1] == 5
            if exception:
                raise RuntimeError("test exception")

    np.testing.assert_array_equal(model.cam_pos, positions)
    np.testing.assert_array_equal(model.geom_group, groups)
    np.testing.assert_array_equal(data.cam_xpos, original)
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    assert data.time == time


@pytest.mark.parametrize("mode", ("depth", "foreground"))
def test_direct_composition_keeps_shared_rgb_winner(mode):
    from scene.gs_background_client import composite_background_depth, composite_background_mask

    wrist = WristFisheye.__new__(WristFisheye)
    wrist.backend = "direct_kb4"
    wrist.camera_model = object()
    wrist.valid = np.array([[True, True], [True, False]])
    wrist.rgb_color_calibration = None
    wrist._composite = composite_background_mask
    wrist.composite_mode = mode
    wrist._composite_depth = composite_background_depth if mode == "depth" else None
    foreground = np.array([[True, True], [False, False]])
    mj_depth = np.array([[1, 3], [np.inf, np.nan]], np.float32)
    gs_depth = np.array([[2, 2], [4, np.nan]], np.float32)
    pending = WristFisheyePending((_CapturedWristFace(
        "source", np.full((2, 2, 3), 80, np.uint8), foreground,
        np.full((2, 2, 3), 160, np.uint8), mj_depth, gs_depth,
    ),))
    rgb = wrist.finalize_cpu_stage(pending)
    np.testing.assert_array_equal(rgb[..., 0], [[80, 160 if mode == "depth" else 80], [160, 0]])


def test_direct_backend_requires_model_negotiation_before_creating_gl():
    with pytest.raises(RuntimeError, match="camera-model negotiation"):
        WristFisheye(None, backend="direct_kb4", gs_client=object())


def test_unknown_backend_is_rejected_before_creating_gl():
    with pytest.raises(ValueError, match="unknown wrist backend"):
        WristFisheye(None, backend="unexpected")


def test_direct_capture_calls_real_renderer_once_and_restores_pose(gpu_egl, monkeypatch):
    mujoco = gpu_egl
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = SimpleNamespace(sim=SimpleNamespace(
        model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data),
    ))
    original = data.cam_xpos.copy()
    wrist = WristFisheye(
        env, prefix="test", camera_side="left", backend="direct_kb4",
        render_profile="elp-mean-wrist-rgb", black_background=True,
    )
    calls = []
    actual_render = wrist._source_renderer.render

    def counted(scene):
        calls.append(np.asarray(scene.camera[0].pos).copy())
        return actual_render(scene)

    monkeypatch.setattr(wrist._source_renderer, "render", counted)
    try:
        capture = wrist.capture_mujoco_stage()
        assert len(calls) == 1
        assert tuple(wrist.face_cam) == ("front",)
        assert len(capture.faces) == 1 and capture.faces[0].name == "source"
        source = capture.faces[0]
        assert source.rgb.shape == (480, 640, 3)
        assert source.foreground_mask.any()
        assert np.isfinite(source.mujoco_depth[source.foreground_mask]).all()
        assert np.isnan(source.mujoco_depth[~wrist.valid]).all()
        assert not source.rgb.flags.writeable
        np.testing.assert_array_equal(data.cam_xpos, original)
        pending = wrist.complete_gs_stage(capture)
        assert np.isclose(source.mujoco_depth[242, 320], .96, atol=1e-3)
        assert wrist.finalize_cpu_stage(pending).shape == (480, 640, 3)
    finally:
        wrist.close()
        wrist.close()


def test_policy_pair_shares_direct_resources_and_captures_both_views(gpu_egl):
    from robosuite_env.rollout.render.video_wrist import WristPairFactory

    mujoco = gpu_egl
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = SimpleNamespace(sim=SimpleNamespace(
        model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data),
    ))
    state = SimpleNamespace(env=env, render_lock=nullcontext,
                            A=SimpleNamespace(policy_wrist_backend="direct_kb4"))
    factory = WristPairFactory(
        state, WristFisheye, "test", "other", None, True,
        "elp-mean-wrist-rgb", "depth", {}, False, "wfR_obs", "policy",
    )
    right, left = factory()
    try:
        assert right.backend == left.backend == "direct_kb4"
        assert right.renderer is left.renderer
        assert right._source_renderer is left._source_renderer
        captured = [w.capture_mujoco_stage() for w in (left, right)]
        assert not np.array_equal(captured[0].faces[0].rgb, captured[1].faces[0].rgb)
        left.close()
        pending = right.complete_gs_stage(captured[1])
        assert right.finalize_cpu_stage(pending).shape == (480, 640, 3)
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("feature", ("skin", "flex"))
def test_direct_rejects_unrepresented_scene_geometry_before_drawing(feature):
    from robosuite_env.rendering.mujoco_kb4 import SceneRenderer

    scene = SimpleNamespace(flags=np.zeros(64), stereo=0, enabletransform=0,
                            framewidth=0, nskin=0, nflex=0)
    setattr(scene, "n" + feature, 1)
    renderer = SceneRenderer.__new__(SceneRenderer)
    with pytest.raises(NotImplementedError, match=feature):
        renderer.render(scene)
