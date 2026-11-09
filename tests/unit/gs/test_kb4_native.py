# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Real HTTP/owner/native KB4 coverage; run with GS_KB4_NATIVE_REPO set."""

import contextlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from scene import gs_background_server
from scene.gs_background_client import GSBackgroundClient
from scene.gs_background_protocol import LEGACY_KB4_NEAR_GS_UNITS, kb4_valid_mask
from robosuite_env.rendering.wrist_render_profiles import (
    ELP_MEAN_WRIST_RGB_PROFILE,
    FisheyeIntrinsics,
    camera_rays,
    get_wrist_render_profile,
)


_DISTANCE = 2.0
_SIGMA = 0.07
_OPACITY = 0.99
_ANGLES = (0.0, math.radians(95))
_Y_ANGLE = math.radians(45)  # off-axis in y, to catch swapped fx/fy or cx/cy


def _median_surface_depth(angle, ray_angle):
    # T=1-opacity*exp(-r²/(2*sigma²)) reaches 0.5 on this sphere.
    radius = _SIGMA * math.sqrt(2 * math.log(2 * _OPACITY))
    offset = angle - ray_angle
    return (_DISTANCE * math.cos(offset) -
            math.sqrt(radius**2 - (_DISTANCE * math.sin(offset))**2))


def _write_gaussians(path, centres, sigma):
    names = ["x", "y", "z", *[f"f_dc_{i}" for i in range(3)],
             *[f"f_rest_{i}" for i in range(45)], "opacity",
             *[f"scale_{i}" for i in range(3)], *[f"rot_{i}" for i in range(4)]]
    rows = [[*centre, *([0] * 48), math.log(_OPACITY / (1 - _OPACITY)),
             *([math.log(sigma)] * 3), 1, 0, 0, 0] for centre in centres]
    path.write_text(f"ply\nformat ascii 1.0\nelement vertex {len(rows)}\n" +
                    "".join(f"property float {name}\n" for name in names) +
                    "end_header\n" + "\n".join(" ".join(map(str, row)) for row in rows) + "\n")


def _write_two_gaussians(path):
    centres = [(_DISTANCE * math.sin(angle), 0, _DISTANCE * math.cos(angle)) for angle in _ANGLES]
    centres.append((0, _DISTANCE * math.sin(_Y_ANGLE), _DISTANCE * math.cos(_Y_ANGLE)))
    _write_gaussians(path, centres, _SIGMA)


@contextlib.contextmanager
def _native_server(tmp_path, ply):
    """Start the real GS server on ``ply``; yield its URL and log path."""

    gw = os.environ.get("GS_KB4_NATIVE_REPO")
    if not gw:
        pytest.skip("GS_KB4_NATIVE_REPO must identify the explicit KB4 development source")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "native-server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(gs_background_server.__file__).resolve()),
             "--gw-repo", gw, "--ply", str(ply), "--port", str(port),
             "--jpeg-quality", "90"], stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                assert process.poll() is None, log_path.read_text()
                try:
                    with urllib.request.urlopen(url + "/health", timeout=0.2) as response:
                        assert json.load(response)["status"] == "ok"
                    break
                except (urllib.error.URLError, TimeoutError):
                    assert time.monotonic() < deadline, log_path.read_text()
                    time.sleep(0.05)
            yield url, log_path
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


_ELP = get_wrist_render_profile(ELP_MEAN_WRIST_RGB_PROFILE).camera


@pytest.mark.gpu
@pytest.mark.parametrize(
    "k", ((0.0, 0.0, 0.0, 0.0), (_ELP.k1, _ELP.k2, _ELP.k3, _ELP.k4)), ids=("equidistant", "elp-mean")
)
def test_real_native_kb4_http_radial_depth_beyond_optical_plane(tmp_path, k):
    ply = tmp_path / "two-gaussians.ply"
    _write_two_gaussians(ply)
    with _native_server(tmp_path, ply) as (url, log_path):
        client = GSBackgroundClient(url, timeout_sec=10, strict=True)
        client.require_camera_model("kb4")
        settings = dict(camera_model="kb4", fx=32.0, fy=32.0, cx=64.0, cy=64.0,
                        k1=k[0], k2=k[1], k3=k[2], k4=k[3], theta_max=math.radians(100))
        scale = 0.25
        camera = dict(width=128, height=128, pos=[0, 0, 0], forward=[0, 0, 1],
                      up=[0, -1, 0], fovy_deg=90, znear=0.1, zfar=10,
                      _real2sim_depth_scale_to_meters=scale, camera_settings=settings,
                      kb4_near=client.kb4_near_gs_units(scale))
        [result] = client.render_payloads_with_depth([camera])
        valid = kb4_valid_mask(128, 128, settings)
        assert result.depth_convention == "radial"
        assert result.rgb.shape == (128, 128, 3)
        assert result.depth.shape == (128, 128)
        assert np.isnan(result.depth[~valid]).all()
        assert not result.rgb[~valid].any()
        assert np.isinf(result.depth[64, 90])
        quantization_error = (camera["zfar"] - camera["znear"]) / (2 * 65534)
        # Native config.h brackets 0.8 units with five 8-way refinements.
        refinement_error = 0.8 / 8**5
        model = FisheyeIntrinsics(128, 128, **{key: settings[key] for key in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")})
        rays, _ = camera_rays(model, settings["theta_max"])
        for angle in _ANGLES:
            column = round(settings["cx"] + settings["fx"] * model.distort(angle))
            ray_angle = math.acos(float(rays[64, column, 2]))
            # The tangent-plane splat omits the second-order range shift.
            projection_error = _DISTANCE * (angle - ray_angle)**2
            tolerance = scale * (quantization_error + refinement_error + projection_error)
            assert result.rgb[64, column].mean() > 50
            assert result.depth[64, column] == pytest.approx(
                scale * _median_surface_depth(angle, ray_angle), abs=tolerance)
        pinhole = {key: value for key, value in camera.items() if key not in ("camera_settings", "kb4_near")}
        pinhole.update(width=129, height=129)  # Pinhole's principal point is (size-1)/2.
        [legacy] = client.render_payloads_with_depth([pinhole])
        assert legacy.depth_convention == "optical_axis"
        tolerance = scale * (quantization_error + refinement_error)
        assert legacy.depth[64, 64] == pytest.approx(
            scale * _median_surface_depth(0.0, 0.0), abs=tolerance)
        assert legacy.depth[64, 64] == pytest.approx(result.depth[64, 64], abs=2 * tolerance)

        skewed = dict(settings, fx=30.0, fy=36.0, cx=66.0, cy=58.0)
        [result] = client.render_payloads_with_depth([dict(camera, camera_settings=skewed)])
        [[expected]], _ = cv2.fisheye.projectPoints(
            np.array([[[0, math.sin(_Y_ANGLE), math.cos(_Y_ANGLE)]]], dtype=np.float64),
            np.zeros(3), np.zeros(3),
            np.array([[skewed["fx"], 0, skewed["cx"]], [0, skewed["fy"], skewed["cy"]], [0, 0, 1]]),
            np.array(k, dtype=np.float64),
        )
        lower = result.rgb.sum(axis=2).astype(np.int64)
        lower[:72] = 0  # leaves only the +y Gaussian
        row, column = np.unravel_index(lower.argmax(), lower.shape)
        assert lower[row, column] > 150
        assert abs(column - expected[0]) <= 1 and abs(row - expected[1]) <= 1, (row, column, expected)
        assert "KB4 development dependency native=" in log_path.read_text()


_NEAR_DISTANCE = 0.03  # [m] between znear (0.01 m) and the legacy cull (0.2 GS units) at both scales
_NEAR_SIGMA = 0.002  # [m]


@pytest.mark.gpu
@pytest.mark.parametrize("scale", (1.0, 0.25), ids=("gs-unit-1m", "gs-unit-0.25m"))
def test_real_native_kb4_near_cull_keeps_a_gaussian_inside_the_legacy_distance(tmp_path, scale):
    # A GS unit is ``scale`` metres. The Gaussian lies 0.03 m in front of the camera: inside
    # 0.2 GS units at either scale and beyond the client's 0.01 m znear.
    distance = _NEAR_DISTANCE / scale
    assert 0.01 / scale < distance < LEGACY_KB4_NEAR_GS_UNITS
    ply = tmp_path / "near-gaussian.ply"
    _write_gaussians(ply, [(0, 0, distance)], _NEAR_SIGMA / scale)
    settings = dict(camera_model="kb4", fx=32.0, fy=32.0, cx=64.0, cy=64.0,
                    k1=0.0, k2=0.0, k3=0.0, k4=0.0, theta_max=math.radians(100))
    with _native_server(tmp_path, ply) as (url, _):
        results = {}
        for mode in ("znear", "legacy"):
            client = GSBackgroundClient(url, timeout_sec=10, strict=True, kb4_near_cull=mode)
            client.require_camera_model("kb4")
            camera = dict(width=128, height=128, pos=[0, 0, 0], forward=[0, 0, 1],
                          up=[0, -1, 0], fovy_deg=90, znear=client.znear / scale, zfar=client.zfar / scale,
                          _real2sim_depth_scale_to_meters=scale, camera_settings=settings,
                          kb4_near=client.kb4_near_gs_units(scale))
            [results[mode]] = client.render_payloads_with_depth([camera])
        assert results["znear"].rgb[64, 64].mean() > 50
        assert results["znear"].depth[64, 64] == pytest.approx(_NEAR_DISTANCE, abs=3 * _NEAR_SIGMA)
        assert not results["legacy"].rgb[60:69, 60:69].any()
        assert np.isinf(results["legacy"].depth[64, 64])
