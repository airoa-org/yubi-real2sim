# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The written wrist image follows ``wrist_image_intrinsics``, not ``FX``..``K4``.

Gaussian dots are drawn at known rays of the profile's KB4 source camera and finalized by the
production direct wrist path, which applies the profile crop. Their centroids are compared with
``project_points``, which is ``cv2.fisheye.projectPoints`` in front of the camera.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from robosuite_env.rendering import fisheye_projection as fp
from robosuite_env.rendering.wrist_fisheye import WristFisheye, _center_scale_crop
from robosuite_env.rendering.wrist_render_profiles import (
    FRANKA_PINHOLE_WRIST_RGB_PROFILE,
    ELP_MEAN_WRIST_RGB_PROFILE,
    FisheyeIntrinsics,
    camera_rays,
    get_wrist_render_profile,
    project_points,
    wrist_image_intrinsics,
    wrist_source_intrinsics,
)

SIDES = ("left", "right")
PROFILES = (FRANKA_PINHOLE_WRIST_RGB_PROFILE, ELP_MEAN_WRIST_RGB_PROFILE)
DOT_SIGMA_PX = 2.0
BORDER_PX = 14
# The bilinear resize of the profile crop moves a dot centroid by a few hundredths of a pixel.
DOT_TOLERANCE_PX = 0.35


def _rays():
    """Unit rays (x right, y up, z forward) on staggered rings out to 85 degrees."""

    rays = [(0.0, 0.0, 1.0)]
    for theta_deg in (20.0, 30.0, 40.0, 50.0, 60.0, 75.0, 85.0):
        theta = math.radians(theta_deg)
        for step in range(12):
            phi = math.radians(30.0 * step + theta_deg)
            rays.append((math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)))
    return np.asarray(rays)


def _opencv(rays):
    """Rays of the fisheye frame (+y up) in the OpenCV frame of ``project_points`` (+y down)."""

    return np.asarray(rays, dtype=float) * (1.0, -1.0, 1.0)


def _cv2_project(model, points):
    pixels, _ = cv2.fisheye.projectPoints(
        np.asarray(points, dtype=float).reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        np.array([[model.fx, 0.0, model.cx], [0.0, model.fy, model.cy], [0.0, 0.0, 1.0]]),
        np.array([model.k1, model.k2, model.k3, model.k4]),
    )
    return pixels.reshape(-1, 2)


def _source_rays(profile):
    """Ray (+y up) and validity of every pixel of the profile's KB4 source raster."""

    theta_max = math.radians(get_wrist_render_profile(profile).source_fov_deg / 2.0)
    rays, valid = camera_rays(wrist_source_intrinsics(profile), theta_max)
    return rays * np.float32([1.0, -1.0, 1.0]), valid


def _dot_source(profile, rays):
    """The source raster with a Gaussian dot of ``DOT_SIGMA_PX`` where the source camera puts every ray."""

    camera = wrist_source_intrinsics(profile)
    level = np.zeros((camera.height, camera.width))
    reach = int(math.ceil(6.0 * DOT_SIGMA_PX))
    for px, py in project_points(camera, _opencv(rays)):
        x0, x1 = max(0, int(px) - reach), min(camera.width, int(px) + reach + 1)
        y0, y1 = max(0, int(py) - reach), min(camera.height, int(py) + reach + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        y, x = np.mgrid[y0:y1, x0:x1]
        dot = np.exp(-0.5 * ((x - px) ** 2 + (y - py) ** 2) / DOT_SIGMA_PX**2)
        level[y0:y1, x0:x1] = np.maximum(level[y0:y1, x0:x1], dot)
    return np.repeat(np.rint(255.0 * level).astype(np.uint8)[..., None], 3, axis=2)


def _crop_arguments(profile, side):
    crop = profile.crop(side)
    return (
        profile.source_fov_deg / crop.effective_horizontal_fov_deg,
        profile.source_fov_deg / crop.effective_vertical_fov_deg,
    ), {
        "crop_x_center_fraction": crop.horizontal_optical_axis_fraction,
        "crop_y_center_fraction": crop.vertical_optical_axis_fraction,
    }


def _dot_centroids(image):
    """Intensity-weighted centroid of every dot that the image border does not cut."""

    level = image[..., 0].astype(float)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (level > 20.0).astype(np.uint8), connectivity=8
    )
    height, width = level.shape
    centroids = []
    for label in range(1, count):
        x, y, w, h, _ = stats[label]
        if x == 0 or y == 0 or x + w == width or y + h == height:
            continue
        ys, xs = np.nonzero(labels == label)
        weight = level[ys, xs]
        centroids.append((np.dot(xs, weight) / weight.sum(), np.dot(ys, weight) / weight.sum()))
    return np.asarray(centroids)


def _matched_centroids(model, rays, image):
    """Each ray the model puts inside the image, its prediction, and its dot."""

    expected = project_points(model, _opencv(rays))
    inside = (
        (expected[:, 0] > BORDER_PX)
        & (expected[:, 0] < model.width - 1 - BORDER_PX)
        & (expected[:, 1] > BORDER_PX)
        & (expected[:, 1] < model.height - 1 - BORDER_PX)
    )
    measured = _dot_centroids(image)
    distance = np.linalg.norm(expected[inside][:, None, :] - measured[None, :, :], axis=-1)
    nearest = distance.argmin(axis=1)
    assert len(set(nearest.tolist())) == len(nearest), "two rays matched one dot"
    return inside, expected[inside], measured[nearest]


def _written_rays(side):
    """Ray (+y up) and validity of every pixel that the legacy profile writes for ``side``."""

    profile = get_wrist_render_profile(FRANKA_PINHOLE_WRIST_RGB_PROFILE)
    rays, valid = _source_rays(profile.name)
    scales, fractions = _crop_arguments(profile, side)
    written = _center_scale_crop(np.asarray(rays, dtype=np.float32), *scales, **fractions)
    return written, _center_scale_crop(valid.astype(np.float32), *scales, **fractions) == 1.0


def _assert_rays_project_back(model, rays, valid, tolerance_px):
    """Every valid ray lands on its pixel, also beyond 90 degrees, where ``cv2.fisheye`` folds it back."""

    ys, xs = np.nonzero(valid)
    pixels = np.column_stack([xs, ys])
    points = _opencv(rays[ys, xs])
    error = np.linalg.norm(project_points(model, points) - pixels, axis=1)
    assert error.max() < tolerance_px, f"max {error.max():.2e} px"
    behind = points[:, 2] < 0.0
    assert behind.sum() > 1000
    folded = np.linalg.norm(_cv2_project(model, points[behind]) - pixels[behind], axis=1)
    assert folded.min() > 500.0, f"cv2.fisheye puts a ray beyond 90 degrees {folded.min():.1f} px off"


def test_project_points_is_cv2_fisheye_in_front_of_the_camera():
    model = wrist_image_intrinsics(FRANKA_PINHOLE_WRIST_RGB_PROFILE, "left")
    points = _opencv(_rays())
    assert (points[:, 2] > 0.0).all()
    np.testing.assert_allclose(project_points(model, points), _cv2_project(model, points), atol=1e-6)


@pytest.mark.parametrize("side", SIDES)
def test_every_written_pixel_ray_projects_back_onto_that_pixel(side):
    rays, valid = _written_rays(side)
    assert valid.mean() > 0.9
    model = wrist_image_intrinsics(FRANKA_PINHOLE_WRIST_RGB_PROFILE, side)
    _assert_rays_project_back(model, rays, valid, 0.01)


def test_the_legacy_profile_intrinsics_are_pinned():
    for side, (fx, fy, cx, cy), (fx640, fy640, cx640, cy640) in (
        ("left", (300.76, 309.82, 365.78, 251.98), (267.71, 275.91, 325.53, 224.34)),
        ("right", (300.76, 309.82, 354.78, 260.98), (267.71, 275.91, 315.74, 232.36)),
    ):
        model = wrist_image_intrinsics(FRANKA_PINHOLE_WRIST_RGB_PROFILE, side)
        assert (model.width, model.height) == (719, 539)
        np.testing.assert_allclose((model.fx, model.fy, model.cx, model.cy), (fx, fy, cx, cy), atol=0.005)
        assert (model.k1, model.k2, model.k3, model.k4) == (fp.K1, fp.K2, fp.K3, fp.K4)
        vga = model.resized(640, 480)
        np.testing.assert_allclose((vga.fx, vga.fy, vga.cx, vga.cy), (fx640, fy640, cx640, cy640), atol=0.005)
        assert vga.as_dict()["model"] == "kb4"


def test_wrist_image_intrinsics_refuses_an_unknown_side():
    with pytest.raises(ValueError, match="wrist side"):
        wrist_image_intrinsics(FRANKA_PINHOLE_WRIST_RGB_PROFILE, "center")


def test_the_elp_mean_profile_writes_one_uncropped_camera_model():
    profile = get_wrist_render_profile(ELP_MEAN_WRIST_RGB_PROFILE)
    assert profile.contract_version == 3
    assert profile.source_image_hw == (480, 640)
    for side in SIDES:
        assert wrist_image_intrinsics(ELP_MEAN_WRIST_RGB_PROFILE, side) == profile.camera
        with pytest.raises(ValueError, match="uncropped"):
            profile.crop(side)
    with pytest.raises(ValueError, match="wrist side"):
        wrist_image_intrinsics(ELP_MEAN_WRIST_RGB_PROFILE, "center")


def test_the_camera_profile_rays_project_back_onto_their_pixels():
    camera = get_wrist_render_profile(ELP_MEAN_WRIST_RGB_PROFILE).camera
    rays, valid = _source_rays(ELP_MEAN_WRIST_RGB_PROFILE)
    assert rays.shape == (camera.height, camera.width, 3)
    _assert_rays_project_back(camera, rays, valid, 1.0e-3)

    # The 200 degree view ends on a circle, so the corners stay black.
    edge = math.radians(100.0)
    radius = project_points(camera, [math.sin(edge), 0.0, math.cos(edge)])[0] - camera.cx
    y, x = np.mgrid[0 : camera.height, 0 : camera.width]
    np.testing.assert_array_equal(valid, np.hypot(x - camera.cx, y - camera.cy) <= radius)
    assert not valid[0, 0] and not valid[-1, -1] and valid[camera.height // 2].all()


def test_a_fisheye_model_that_folds_back_is_refused():
    folded = replace(get_wrist_render_profile(ELP_MEAN_WRIST_RGB_PROFILE).camera, k1=-0.5)
    with pytest.raises(ValueError, match="not monotonic"):
        camera_rays(folded, math.radians(100.0))


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("side", SIDES)
def test_the_constructed_renderer_writes_that_model(profile, side):
    # importorskip, not the mujoco fixture (which marks a test heavy and keeps it out of CI's main step):
    # the wrist only finalizes a given source raster on the CPU and opens no GL context.
    pytest.importorskip("mujoco")
    from robosuite_env.rendering.wrist_fisheye import WristFisheyePending, _CapturedWristFace
    from tests._helpers.wrist_scene import build_wrist_test_env

    rays = _rays()
    _, valid = _source_rays(profile)
    wrist = WristFisheye(
        build_wrist_test_env(),
        prefix=f"gripper0_{side}",
        side=side,
        render_profile=profile,
        camera_side=side,
        renderer=object(),
        backend="direct_kb4",
        source_renderer=SimpleNamespace(camera=wrist_source_intrinsics(profile), valid=valid),
    )
    wrist.rgb_color_calibration = None
    image = wrist.finalize_cpu_stage(
        WristFisheyePending(
            faces=(_CapturedWristFace(name="source", rgb=_dot_source(profile, rays), foreground_mask=None, background=None),)
        )
    )
    model = wrist_image_intrinsics(profile, side)
    assert image.shape[:2] == (model.height, model.width)
    inside, expected, measured = _matched_centroids(model, rays, image)
    error = np.linalg.norm(measured - expected, axis=1)
    assert inside.sum() >= 30
    assert error.max() < DOT_TOLERANCE_PX, f"{side}: max {error.max():.3f} px, mean {error.mean():.3f} px"
    if profile == FRANKA_PINHOLE_WRIST_RGB_PROFILE:
        # FX..K4 scaled to the written raster: a pairing the written image does not follow.
        constants = FisheyeIntrinsics(
            fp.OUT_W, fp.OUT_H, fp.FX, fp.FY, fp.CX, fp.CY, fp.K1, fp.K2, fp.K3, fp.K4
        ).resized(model.width, model.height)
        wrong = np.linalg.norm(project_points(constants, _opencv(rays[inside])) - measured, axis=1)
        assert wrong.max() > 20.0, f"{side}: FX..K4 is off by at most {wrong.max():.1f} px"
