# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from scene.gs_background_client import GSBackgroundClient, GSRenderResult
from scene.gs_background_protocol import (
    CAPABILITIES_HEADER,
    KB4_CAPABILITY,
    KB4_FIELDS,
    KB4_NEAR_CAPABILITY,
    KB4_THETA_MARGIN,
    LEGACY_KB4_NEAR_GS_UNITS,
    RGBD_BATCH_CONTENT_TYPE,
    kb4_valid_mask,
    pack_rgbd_batch,
    validate_camera_settings,
    validate_kb4_near,
)
from scene.gs_background_server import _encode_depth_png, _encode_jpeg, _native_kb4_supported
from robosuite_env.rendering.wrist_render_profiles import FisheyeIntrinsics, camera_rays


def settings():
    return dict(camera_model="kb4", fx=2.0, fy=2.0, cx=4.0, cy=4.0,
                k1=0.0, k2=0.0, k3=0.0, k4=0.0, theta_max=math.radians(100))


def payload():
    return dict(width=9, height=9, pos=[0, 0, 0], forward=[0, 0, 1], up=[0, -1, 0],
                znear=0.1, zfar=10.0, camera_settings=settings(), kb4_near=0.4,
                _real2sim_depth_scale_to_meters=0.25)


BOTH = f"{KB4_CAPABILITY},{KB4_NEAR_CAPABILITY}"


class Response:
    def __init__(self, body=b"{}", headers=None):
        self.body = body
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


@pytest.mark.parametrize("value", [None, [], "kb4", {}, {**settings(), "camera_model": "pinhole"},
                                    {**settings(), "image_width": 1}])
def test_camera_settings_reject_non_contract_objects(value):
    with pytest.raises(ValueError):
        validate_camera_settings(value, 9, 9)


@pytest.mark.parametrize("key,value", [("fx", 0), ("fy", -1), ("fx", True), ("cx", "4"),
                                      ("k1", math.inf), ("k2", math.nan), ("theta_max", 0),
                                      ("theta_max", math.pi), ("theta_max", math.pi - KB4_THETA_MARGIN),
                                      ("k1", -1)])
def test_camera_settings_reject_invalid_numbers_and_noninvertible_model(key, value):
    with pytest.raises(ValueError):
        validate_camera_settings({**settings(), key: value}, 9, 9)


def test_theta_max_margin_matches_the_native_rasterizer():
    config = (Path(__file__).resolve().parents[3] / "third_party/GaussianWrapping/submodules"
              / "diff-gaussian-rasterization_ours/cuda_rasterizer/config.h")
    if not config.exists():
        pytest.skip("the GaussianWrapping submodule is not checked out")
    match = re.search(r"KB4_THETA_MARGIN\s*=\s*([0-9.]+)f;", config.read_text())
    assert match is not None and float(match.group(1)) == KB4_THETA_MARGIN
    validate_camera_settings({**settings(), "theta_max": math.pi - KB4_THETA_MARGIN - 1e-3}, 9, 9)


@pytest.mark.parametrize("width,height", [(True, 9), (9.5, 9), ("9", 9), (9, 0), (9, -1)])
def test_camera_settings_reject_invalid_raster(width, height):
    with pytest.raises(ValueError):
        validate_camera_settings(settings(), width, height)


def test_shared_mask_retains_negative_optical_z():
    value = settings()
    value["fx"] = value["fy"] = 3 / math.radians(95)
    camera = FisheyeIntrinsics(9, 9, **{key: value[key] for key in KB4_FIELDS[:-1]})
    rays, expected = camera_rays(camera, value["theta_max"])
    actual = kb4_valid_mask(9, 9, value)
    np.testing.assert_array_equal(actual, expected)
    assert actual[4, 7] and rays[4, 7, 2] < 0
    assert not actual.flags.writeable


def test_depth_decode_is_radial_meters_once_and_restores_invalid_circle():
    client = GSBackgroundClient("http://unused", color_gain_rgb=(2, 1, 1))
    camera = payload()
    rgb = np.full((9, 9, 3), 80, np.uint8)
    depth = np.full((9, 9), 4, np.float32)
    depth[4, 4] = 0
    result = client._decode_render_result(_encode_jpeg(rgb, 90), _encode_depth_png(depth, 0.1, 10), camera)
    valid = kb4_valid_mask(9, 9, camera["camera_settings"])
    assert result.depth_convention == "radial"
    np.testing.assert_allclose(result.depth[valid & np.isfinite(result.depth)], 1, atol=2e-5)
    assert np.isinf(result.depth[4, 4])
    assert np.isnan(result.depth[~valid]).all()
    assert not result.rgb[~valid].any()
    assert GSRenderResult(rgb, depth).depth_convention == "optical_axis"


def test_direct_depth_missing_never_accepts_rgb_only():
    client = GSBackgroundClient("http://unused")
    with pytest.raises(RuntimeError, match="missing radial depth"):
        client._decode_render_result(_encode_jpeg(np.zeros((9, 9, 3), np.uint8), 80), None, payload())


def test_required_camera_fails_on_old_server_even_when_batch_disabled():
    client = GSBackgroundClient("http://unused", prefer_batch=False)
    with mock.patch("urllib.request.urlopen", return_value=Response()):
        with pytest.raises(RuntimeError, match=KB4_CAPABILITY):
            client.require_camera_model("kb4")


def test_direct_request_checks_reply_capability_after_server_restart():
    client = GSBackgroundClient("http://unused", prefer_batch=False)
    responses = [Response(headers={CAPABILITIES_HEADER: BOTH}),
                 Response(_encode_jpeg(np.zeros((9, 9, 3), np.uint8), 80))]
    with mock.patch("urllib.request.urlopen", side_effect=responses):
        with pytest.raises(RuntimeError, match="does not confirm"):
            client.render_payloads_with_depth([payload()])


def test_direct_single_transport_preserves_settings_and_scale_metadata():
    camera = payload()
    jpeg = _encode_jpeg(np.full((9, 9, 3), 80, np.uint8), 90)
    depth = _encode_depth_png(np.full((9, 9), 4, np.float32), 0.1, 10)
    responses = [Response(headers={CAPABILITIES_HEADER: BOTH}),
                 Response(pack_rgbd_batch([(jpeg, depth)]),
                          {CAPABILITIES_HEADER: BOTH, "Content-Type": RGBD_BATCH_CONTENT_TYPE})]
    client = GSBackgroundClient("http://unused", prefer_batch=False)
    with mock.patch("urllib.request.urlopen", side_effect=responses) as opening:
        [result] = client.render_payloads_with_depth([camera])
    request = json.loads(opening.call_args_list[1].args[0].data)
    assert request["camera_settings"] == settings()
    assert request["kb4_near"] == 0.4
    assert "_real2sim_depth_scale_to_meters" not in request
    assert camera["_real2sim_depth_scale_to_meters"] == 0.25
    assert result.depth_convention == "radial"


def test_native_capability_requires_all_three_interfaces():
    def renderer(*args, camera_settings=None):
        pass

    binding = SimpleNamespace(__doc__="arg25: Optional[Annotated[list[float], FixedSize(10)]]")
    native = SimpleNamespace(GaussianRasterizationSettings=SimpleNamespace(_fields=("camera_model", *KB4_FIELDS, "kb4_near")),
                             _C=SimpleNamespace(rasterize_gaussians=binding))
    assert _native_kb4_supported(renderer, native)
    assert not _native_kb4_supported(lambda *args: None, native)
    binding.__doc__ = "old native binding"
    assert not _native_kb4_supported(renderer, native)
    binding.__doc__ = "arg25: Optional[Annotated[list[float], FixedSize(10)]]"
    native.GaussianRasterizationSettings._fields = ()
    assert not _native_kb4_supported(renderer, native)


def test_native_capability_refuses_the_unpatched_rasterizer_with_its_fixed_near_cull(capsys):
    # GaussianWrapping cd8cb18, before kb4_near: nine KB4 values and no kb4_near setting.
    def renderer(*args, camera_settings=None):
        pass

    binding = SimpleNamespace(__doc__="arg25: Optional[Annotated[list[float], FixedSize(9)]]")
    native = SimpleNamespace(GaussianRasterizationSettings=SimpleNamespace(_fields=("camera_model", *KB4_FIELDS)),
                             _C=SimpleNamespace(rasterize_gaussians=binding))
    assert not _native_kb4_supported(renderer, native)
    assert "lacks kb4_near" in capsys.readouterr().out
    native.GaussianRasterizationSettings._fields = ("camera_model", *KB4_FIELDS, "kb4_near")
    assert not _native_kb4_supported(renderer, native)  # patched Python package over an unpatched native module


def test_required_camera_fails_on_a_server_without_the_near_cull_capability():
    client = GSBackgroundClient("http://unused", prefer_batch=False)
    with mock.patch("urllib.request.urlopen", return_value=Response(headers={CAPABILITIES_HEADER: KB4_CAPABILITY})):
        with pytest.raises(RuntimeError, match=KB4_NEAR_CAPABILITY):
            client.require_camera_model("kb4")


def test_kb4_request_without_or_with_an_invalid_near_cull_is_refused_before_sending():
    camera = payload()
    del camera["kb4_near"]
    with pytest.raises(ValueError, match="kb4_near"):
        GSBackgroundClient._wire_camera_payload(camera, want_depth=True)
    for bad in (0.0, -1.0, math.nan, math.inf, True, "0.2"):
        with pytest.raises(ValueError, match="kb4_near"):
            GSBackgroundClient._wire_camera_payload({**payload(), "kb4_near": bad}, want_depth=True)
        with pytest.raises(ValueError):
            validate_kb4_near(bad)


@pytest.mark.parametrize("world_to_gs", [1.0, 1.19395, 2.839645])
def test_near_cull_is_the_metric_znear_by_default_and_0p2_gs_units_for_legacy(world_to_gs):
    mat = np.diag([world_to_gs, world_to_gs, world_to_gs, 1.0])
    default = GSBackgroundClient("http://unused", mat_w2gs=mat)
    assert default.kb4_near_cull == "znear"
    assert default.kb4_near_gs_units(1.0 / world_to_gs) == pytest.approx(0.01 * world_to_gs)
    assert default.kb4_near_meters() == pytest.approx(0.01)
    legacy = GSBackgroundClient("http://unused", mat_w2gs=mat, kb4_near_cull="legacy")
    assert legacy.kb4_near_gs_units(1.0 / world_to_gs) == LEGACY_KB4_NEAR_GS_UNITS
    assert legacy.kb4_near_meters() == pytest.approx(0.2 / world_to_gs)
    with pytest.raises(ValueError, match="kb4_near_cull"):
        GSBackgroundClient("http://unused", kb4_near_cull="off").kb4_near_gs_units(1.0)


def test_native_capability_requires_the_wrist_model_in_the_server_interpreter(monkeypatch, capsys):
    def renderer(*args, camera_settings=None):
        pass

    binding = SimpleNamespace(__doc__="arg25: Optional[Annotated[list[float], FixedSize(10)]]")
    native = SimpleNamespace(GaussianRasterizationSettings=SimpleNamespace(_fields=("camera_model", *KB4_FIELDS, "kb4_near")),
                             _C=SimpleNamespace(rasterize_gaussians=binding))
    monkeypatch.setitem(sys.modules, "robosuite_env.rendering.wrist_render_profiles", None)
    assert not _native_kb4_supported(renderer, native)
    assert "install this checkout in the server interpreter" in capsys.readouterr().out
    old = SimpleNamespace(FisheyeIntrinsics=lambda *args, **kwargs: SimpleNamespace())  # importable, no min_distort_slope
    monkeypatch.setitem(sys.modules, "robosuite_env.rendering.wrist_render_profiles", old)
    assert not _native_kb4_supported(renderer, native)
    assert "min_distort_slope" in capsys.readouterr().out


@pytest.mark.parametrize("mode,expected_gs", [("znear", 0.01 * 1.19395), ("legacy", 0.2)])
def test_camera_payload_carries_the_near_cull_for_kb4_only(monkeypatch, mode, expected_gs):
    import mujoco

    def camera_in_model(pos, forward, up, scene):
        pos[:] = (0.1, 0.2, 0.3); forward[:] = (0.0, 0.0, -1.0); up[:] = (0.0, 1.0, 0.0)

    monkeypatch.setattr(mujoco, "mjv_cameraInModel", camera_in_model)
    client = GSBackgroundClient("http://unused", mat_w2gs=np.diag([1.19395] * 3 + [1.0]), kb4_near_cull=mode)
    kb4 = client.camera_payload(None, 9, 9, camera_settings=settings())
    assert kb4["kb4_near"] == pytest.approx(expected_gs)
    assert "kb4_near" not in client.camera_payload(None, 9, 9)  # the pinhole camera keeps its fixed cull
    assert GSBackgroundClient._wire_camera_payload(kb4, want_depth=True)["kb4_near"] == pytest.approx(expected_gs)
