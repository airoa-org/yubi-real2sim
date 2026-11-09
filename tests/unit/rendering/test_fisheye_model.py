# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Independent KB4 projection/inversion oracles."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from robosuite_env.rendering.fisheye_model import FisheyeIntrinsics
from robosuite_env.rendering.wrist_render_profiles import (
    ELP_MEAN_WRIST_RGB_PROFILE,
    FRANKA_PINHOLE_WRIST_RGB_PROFILE,
    wrist_image_intrinsics,
)


@pytest.mark.parametrize("profile", (FRANKA_PINHOLE_WRIST_RGB_PROFILE, ELP_MEAN_WRIST_RGB_PROFILE))
@pytest.mark.parametrize("side", ("left", "right"))
def test_nonzero_kb4_projection_and_inverse_match_opencv(profile, side):
    camera = wrist_image_intrinsics(profile, side).resized(320, 240)
    theta = np.radians([0.0, 10.0, 25.0, 45.0, 65.0, 80.0])
    phi = np.radians([0.0, 37.0, 105.0, 195.0, 270.0, 333.0])
    rays = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
    matrix = np.array([[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]])
    coefficients = np.array([camera.k1, camera.k2, camera.k3, camera.k4])
    assert np.all(coefficients != 0.0)
    pixels, _ = cv2.fisheye.projectPoints(rays[:, None, :], np.zeros(3), np.zeros(3), matrix, coefficients)
    np.testing.assert_allclose(camera.project_points(rays), pixels[:, 0, :], atol=1e-11, rtol=0)

    planar = cv2.fisheye.undistortPoints(
        pixels, matrix, coefficients, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
    )[:, 0, :]
    expected = np.column_stack([planar, np.ones(len(planar))])
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    recovered, valid = camera.unproject_points(pixels[:, 0, :], np.radians(100.0))
    assert valid.all()
    np.testing.assert_allclose(recovered, expected, atol=1e-11, rtol=0)


def test_the_bracketed_inverse_recovers_rays_when_newton_alone_cycles():
    a, b, lift = 0.5 ** -2, 1.5 ** -2, 1.0e-6
    coefficients = (
        -2.0 * (a + b) / 3.0,
        (a * a + 4.0 * a * b + b * b) / 5.0,
        -2.0 * a * b * (a + b) / 7.0,
        a * a * b * b / 9.0,
    )
    camera = FisheyeIntrinsics(41, 41, 20.0, 20.0, 20.0, 20.0, *(k / (1.0 + lift) for k in coefficients))
    theta = np.array([0.3, 0.65, 1.05, 1.35])
    rays = np.column_stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)])
    recovered, valid = camera.unproject_points(camera.project_points(rays), 1.5)
    assert valid.all()
    np.testing.assert_allclose(recovered, rays, atol=1e-10)


def test_arbitrary_pixel_inverse_checks_monotonicity_and_supports_scalar_pixels():
    camera = wrist_image_intrinsics(ELP_MEAN_WRIST_RGB_PROFILE, "right")
    ray, valid = camera.unproject_points([camera.cx, camera.cy], np.radians(100.0))
    np.testing.assert_array_equal(ray, [0.0, 0.0, 1.0])
    assert valid
    folded = replace(camera, k1=-0.5)
    with pytest.raises(ValueError, match="not monotonic"):
        folded.unproject_points([camera.cx, camera.cy], np.radians(100.0))
