# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The shared KB4 wrist camera for the single-pass renderers."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from robosuite_env.rendering import fisheye_projection as fp
from robosuite_env.rendering.wrist_render_profiles import (
    FRANKA_PINHOLE_WRIST_RGB_PROFILE,
    ELP_MEAN_WRIST_RGB_PROFILE,
    FisheyeIntrinsics,
    camera_rays,
    get_wrist_render_profile,
    local_projection_jacobian,
    project_points,
    projection_jacobian,
    ray_tangent_basis,
    wrist_source_intrinsics,
)

THETA_MAX = math.radians(100.0)


def _points():
    """OpenCV-frame points from the axis out to 100 degrees, at 2 cm to 2 m."""

    points = []
    for theta_deg in (0.0, 1.0e-6, 0.5, 10.0, 45.0, 75.0, 89.9, 90.0, 95.0, 100.0):
        for phi_deg in (0.0, 37.0, 90.0, 161.0, 250.0, 333.0):
            for distance in (0.02, 0.3, 2.0):
                theta, phi = math.radians(theta_deg), math.radians(phi_deg)
                direction = [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]
                points.append(distance * np.asarray(direction))
    return np.asarray(points)


def test_the_source_camera_is_fx_to_k4_centred_on_the_wider_raster():
    source = wrist_source_intrinsics(FRANKA_PINHOLE_WRIST_RGB_PROFILE)
    assert source == FisheyeIntrinsics(
        719, 539, fp.FX, fp.FY, fp.CX + 39.5, fp.CY + 29.5, fp.K1, fp.K2, fp.K3, fp.K4
    )
    assert (source.height, source.width) == get_wrist_render_profile(FRANKA_PINHOLE_WRIST_RGB_PROFILE).source_image_hw
    assert fp.source_intrinsics(fp.OUT_W, fp.OUT_H).cx == fp.CX
    elp = get_wrist_render_profile(ELP_MEAN_WRIST_RGB_PROFILE).camera
    assert wrist_source_intrinsics(ELP_MEAN_WRIST_RGB_PROFILE) == elp


@pytest.mark.parametrize("profile", (FRANKA_PINHOLE_WRIST_RGB_PROFILE, ELP_MEAN_WRIST_RGB_PROFILE))
def test_camera_rays_project_back_onto_their_pixel_centres(profile):
    camera = wrist_source_intrinsics(profile)
    rays, valid = camera_rays(camera, THETA_MAX)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1.0, atol=1.0e-6)
    ys, xs = np.nonzero(valid)
    theta = np.arccos(np.clip(rays[ys, xs, 2].astype(np.float64), -1.0, 1.0))
    assert theta.max() > math.radians(99.5)
    error = np.hypot(*(project_points(camera, rays[ys, xs]) - np.column_stack([xs, ys])).T)
    assert error.max() < 1.0e-3, f"max {error.max():.2e} px"


def test_invalid_pixels_get_the_ray_at_theta_max():
    rays, valid = camera_rays(wrist_source_intrinsics(FRANKA_PINHOLE_WRIST_RGB_PROFILE), THETA_MAX)
    assert not valid.all()
    np.testing.assert_allclose(rays[~valid][:, 2], math.cos(THETA_MAX), atol=1.0e-6)


def test_a_model_that_folds_back_between_slope_samples_is_refused():
    folded = replace(
        wrist_source_intrinsics(FRANKA_PINHOLE_WRIST_RGB_PROFILE),
        k1=-0.46675501614553866,
        k2=0.09803706034615614,
        k3=0.0,
        k4=0.0,
    )
    assert folded.distort_slope(np.linspace(0.0, THETA_MAX, 1024)).min() > 0.0
    with pytest.raises(ValueError, match="not monotonic"):
        camera_rays(folded, THETA_MAX)
    vertex = 1.0 - (3.0 * folded.k1) ** 2 / (20.0 * folded.k2)
    assert folded.min_distort_slope(THETA_MAX) == pytest.approx(vertex, abs=1.0e-12)


def _touching_model(lift):
    """The model whose slope ``((1 - theta^2)^4 + lift) / (1 + lift)`` is smallest at 1 rad."""

    k1, k2, k3, k4 = (k / (1.0 + lift) for k in (-4.0 / 3.0, 6.0 / 5.0, -4.0 / 7.0, 1.0 / 9.0))
    return replace(wrist_source_intrinsics(FRANKA_PINHOLE_WRIST_RGB_PROFILE), k1=k1, k2=k2, k3=k3, k4=k4)


def test_a_slope_that_touches_zero_is_refused():
    touching = _touching_model(0.0)
    assert touching.distort_slope(1.0) == 0.0
    with pytest.raises(ValueError, match="not monotonic"):
        camera_rays(touching, THETA_MAX)


def test_a_slope_just_above_zero_is_accepted():
    lifted = _touching_model(1.0e-12)
    assert lifted.min_distort_slope(THETA_MAX) == pytest.approx(1.0e-12, abs=1.0e-13)
    camera_rays(lifted, THETA_MAX)


def test_camera_rays_invert_a_model_whose_slope_has_two_shallow_minima():
    """Slope ``(((1 - (theta / 0.5)^2) (1 - (theta / 1.5)^2))^2 + lift) / (1 + lift)``, where Newton alone cycles."""

    a, b, lift = 0.5 ** -2, 1.5 ** -2, 1.0e-6
    k = (-2.0 * (a + b) / 3.0, (a * a + 4.0 * a * b + b * b) / 5.0, -2.0 * a * b * (a + b) / 7.0, a * a * b * b / 9.0)
    camera = FisheyeIntrinsics(41, 41, 20.0, 20.0, 20.0, 20.0, *(value / (1.0 + lift) for value in k))
    assert camera.min_distort_slope(1.5) > 0.0
    rays, valid = camera_rays(camera, 1.5)
    ys, xs = np.nonzero(valid)
    error = np.hypot(*(project_points(camera, rays[ys, xs]) - np.column_stack([xs, ys])).T)
    assert error.max() < 1.0e-3, f"max {error.max():.2e} px"


def test_min_distort_slope_is_the_minimum_over_the_whole_interval():
    camera = wrist_source_intrinsics(FRANKA_PINHOLE_WRIST_RGB_PROFILE)
    rng = np.random.default_rng(0)
    samples = np.linspace(0.0, 1.0, 400_001)
    interior = 0
    for _ in range(100):
        k1, k2, k3, k4 = rng.uniform([-0.5, -0.1, -0.05, -0.01], [0.1, 0.1, 0.05, 0.01])
        theta_max = math.radians(rng.uniform(60.0, 110.0))
        model = replace(camera, k1=k1, k2=k2, k3=k3, k4=k4)
        slope = model.distort_slope(samples * theta_max)
        interior += 0 < slope.argmin() < samples.size - 1
        assert -1.0e-12 <= slope.min() - model.min_distort_slope(theta_max) < 1.0e-9
    assert interior > 20


def test_project_points_keeps_the_full_angle_beyond_90_degrees():
    camera = wrist_source_intrinsics(ELP_MEAN_WRIST_RGB_PROFILE)
    theta = np.radians([80.0, 90.0, 95.0, 100.0])
    pixels = project_points(camera, np.column_stack([np.sin(theta), np.zeros(4), np.cos(theta)]))
    np.testing.assert_allclose(pixels[:, 0] - camera.cx, camera.fx * camera.distort(theta), rtol=1.0e-12)
    np.testing.assert_allclose(pixels[:, 1], camera.cy)


@pytest.mark.parametrize("profile", (FRANKA_PINHOLE_WRIST_RGB_PROFILE, ELP_MEAN_WRIST_RGB_PROFILE))
def test_projection_jacobian_matches_central_differences(profile):
    camera = wrist_source_intrinsics(profile)
    points = _points()
    jacobian = projection_jacobian(camera, points)
    assert jacobian.shape == (len(points), 2, 3)
    numeric = np.empty_like(jacobian)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = 1.0e-7
        h = step * np.linalg.norm(points, axis=1, keepdims=True)
        difference = project_points(camera, points + h) - project_points(camera, points - h)
        numeric[:, :, axis] = difference / (2.0 * h[:, [axis]])
    scale = np.abs(jacobian).max(axis=(1, 2), keepdims=True)
    np.testing.assert_allclose(jacobian / scale, numeric / scale, atol=1.0e-6)


def test_projection_jacobian_on_the_axis_is_the_pinhole_one():
    camera = wrist_source_intrinsics(FRANKA_PINHOLE_WRIST_RGB_PROFILE)
    np.testing.assert_allclose(
        projection_jacobian(camera, [0.0, 0.0, 0.5]),
        [[camera.fx / 0.5, 0.0, 0.0], [0.0, camera.fy / 0.5, 0.0]],
        rtol=1.0e-12,
    )


def test_ray_tangent_basis_is_e_theta_and_e_phi():
    points = _points()
    basis = ray_tangent_basis(points)
    ray = points / np.linalg.norm(points, axis=1, keepdims=True)
    frame = np.concatenate([basis, ray[:, :, None]], axis=2)
    np.testing.assert_allclose(np.swapaxes(frame, 1, 2) @ frame, np.broadcast_to(np.eye(3), frame.shape), atol=1.0e-12)
    np.testing.assert_allclose(np.linalg.det(frame), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(basis[:, 2, 0], -np.hypot(ray[:, 0], ray[:, 1]), atol=1.0e-12)
    np.testing.assert_array_equal(basis[:, 2, 1], 0.0)
    np.testing.assert_array_equal(ray_tangent_basis([0.0, 0.0, 0.7]), [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])


@pytest.mark.parametrize("profile", (FRANKA_PINHOLE_WRIST_RGB_PROFILE, ELP_MEAN_WRIST_RGB_PROFILE))
def test_local_projection_jacobian_is_the_derivative_in_the_ray_frame(profile):
    camera = wrist_source_intrinsics(profile)
    points = _points()
    basis = ray_tangent_basis(points)
    jacobian = local_projection_jacobian(camera, points)
    assert jacobian.shape == (len(points), 2, 2)
    numeric = np.empty_like(jacobian)
    for axis in range(2):
        h = 1.0e-7 * np.linalg.norm(points, axis=1, keepdims=True) * basis[:, :, axis]
        numeric[:, :, axis] = (project_points(camera, points + h) - project_points(camera, points - h)) / 2.0e-7
    scale = np.abs(jacobian).max(axis=(1, 2), keepdims=True)
    np.testing.assert_allclose(jacobian / scale, numeric / scale, atol=1.0e-6)


def test_the_local_jacobian_stays_regular_at_90_degrees():
    camera = wrist_source_intrinsics(FRANKA_PINHOLE_WRIST_RGB_PROFILE)
    point = [0.5, 0.0, 0.0]
    np.testing.assert_allclose(
        local_projection_jacobian(camera, point),
        np.diag([camera.fx * camera.distort_slope(math.pi / 2), camera.fy * camera.distort(math.pi / 2)]),
        rtol=1.0e-12,
    )
    assert np.linalg.matrix_rank(projection_jacobian(camera, point)[:, :2]) == 1
