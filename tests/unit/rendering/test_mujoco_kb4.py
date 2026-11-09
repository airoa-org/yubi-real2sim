# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""SceneRenderer must reproduce ``mjr_render`` of the same pinhole camera."""

import numpy as np
import pytest

from real2sim.wrist_profiles import (
    DEFAULT_POLICY_WRIST_RENDER_PROFILE, ELP_MEAN_WRIST_RGB_PROFILE, FRANKA_PINHOLE_WRIST_RGB_PROFILE,
    WRIST_RENDER_PROFILE_NAMES)

SCENE = """
<mujoco>
  <asset>
    <texture name="checker" type="2d" builtin="checker" rgb1="1 0 0" rgb2="0 0 1" width="64" height="64"/>
    <texture name="cube" type="cube" builtin="gradient" rgb1="1 1 0" rgb2="0 1 1" width="32" height="32"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="32" height="192"/>
    <material name="texgen" texture="checker" texrepeat="4 4" specular=".8" shininess=".6"/>
    <material name="uniform" texture="checker" texrepeat="2 3" texuniform="true" specular=".3" emission=".1"/>
    <material name="cube" texture="cube" specular=".5"/>
    <mesh name="tetra" vertex="0 0 0  .2 0 0  0 .2 0  0 0 .2"/>
    <mesh name="pyramid" vertex="0 0 0  .3 0 0  .3 .3 0  0 .3 0  .15 .15 .2"
          face="0 2 1  0 3 2  0 1 4  1 2 4  2 3 4  3 0 4" texcoord="0 0  1 0  1 1  0 1  .5 .5"/>
  </asset>
  <worldbody>
    <light pos="0 -1 3" dir="0 .3 -1" directional="true" castshadow="false"/>
    <light pos=".5 -1 1.5" dir="-.3 .6 -1" cutoff="40" exponent="5" castshadow="false" diffuse=".6 .5 .4"/>
    <light pos="0 -.9 .1" dir="0 0 1" cutoff="180" exponent="20" castshadow="false" diffuse=".3 .3 .3"/>
    <geom type="plane" size="2 2 .1" pos="0 0 -.3" material="texgen"/>
    <geom type="box" size=".3 .25 .2" material="texgen"/>
    <geom type="box" size=".1 .1 .1" pos="-.3 -.45 .1" material="uniform"/>
    <geom type="sphere" size=".2" pos=".6 0 0" rgba="0 1 0 1"/>
    <geom type="ellipsoid" size=".1 .15 .2" pos=".6 -.4 0" material="cube"/>
    <geom type="sphere" size=".15" pos=".2 -.5 .2" rgba="1 0 1 .5"/>
    <geom type="capsule" size=".1 .2" pos="-.6 0 0" rgba="1 1 0 1"/>
    <geom type="cylinder" size=".1 .2" pos="0 0 .6" rgba="0 1 1 1"/>
    <geom type="mesh" mesh="tetra" pos="-.4 -.4 .3" rgba=".8 .5 .2 1"/>
    <geom type="mesh" mesh="pyramid" pos=".1 -.6 -.25" euler="20 0 0" material="texgen"/>
    <camera name="side" pos="0 -2.2 .6" xyaxes="1 0 0 0 .25 1" fovy="90"/>
    <camera name="close" pos=".4 -.9 .5" xyaxes=".8 .6 0 -.2 .3 1" fovy="60"/>
    <camera name="top" pos="0 0 1.6" xyaxes="1 0 0 0 1 0" fovy="90"/>
  </worldbody>
</mujoco>
"""


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return np.inf if mse == 0 else 10 * np.log10(255.0**2 / mse)


def _mujoco_segid(mujoco, renderer):
    """``mjr_render`` segmentation as an index into ``scene.geoms`` (-1: none)."""

    flags = renderer.scene.flags
    saved = flags.copy()
    flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1
    try:
        raw = renderer.render().astype(np.int64)
    finally:
        flags[:] = saved
    return (raw[..., 0] + 256 * raw[..., 1] + 65536 * raw[..., 2] - 1).astype(np.int32)


@pytest.fixture()
def scene(gpu_egl):
    from robosuite_env.rendering.mujoco_kb4 import SceneRenderer

    mujoco = gpu_egl
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 240, 320)
    ours = SceneRenderer(model, 320, 240, renderer)
    yield mujoco, model, data, renderer, ours
    ours.close()
    renderer.close()


def _ray_distance(mujoco, model, data, scene, rows, cols, height, width):
    """``mj_multiRay`` through pixel centres: (model geom id, distance)."""

    cam = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
    forward, up = np.asarray(cam.forward), np.asarray(cam.up)
    focal = 0.5 * height / cam.frustum_top * cam.frustum_near
    x = (cols + 0.5 - 0.5 * width) / focal
    y = (0.5 * height - rows - 0.5) / focal
    vec = forward + x[:, None] * np.cross(forward, up) + y[:, None] * up
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    geomid = np.zeros(len(vec), np.int32)
    dist = np.zeros(len(vec))
    mujoco.mj_multiRay(model, data, np.asarray(cam.pos, np.float64), vec.ravel(), None, 1, -1,
                       geomid, dist, None, len(vec), 1e6)
    return geomid, dist


@pytest.mark.parametrize("camera", ["side", "close", "top"])
@pytest.mark.parametrize("skybox", [0, 1])
def test_matches_mjr_render(scene, camera, skybox):
    mujoco, model, data, renderer, ours = scene
    renderer.update_scene(data, camera=camera)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = skybox
    rgb = renderer.render().copy()
    segid = _mujoco_segid(mujoco, renderer)

    out = ours.render(renderer.scene)

    assert _psnr(rgb, out.rgb) >= 40.0
    occupied, ours_occupied = segid >= 0, out.segid >= 0
    assert (occupied & ours_occupied).sum() / (occupied | ours_occupied).sum() >= 0.999
    assert (segid[occupied] == out.segid[occupied]).mean() >= 0.999
    assert np.isinf(out.distance[~ours_occupied]).all()
    assert (np.isfinite(out.distance[ours_occupied]) & (out.distance[ours_occupied] > 0)).all()
    rows, cols = (a.ravel() for a in np.mgrid[0:240, 0:320])
    geomid, dist = _ray_distance(mujoco, model, data, renderer.scene, rows, cols, 240, 320)
    objid = np.array([renderer.scene.geoms[i].objid for i in range(renderer.scene.ngeom)] + [-1])
    same = (geomid >= 0) & (geomid == objid[out.segid[rows, cols]])
    assert same.sum() >= 0.99 * (geomid >= 0).sum()
    error = np.abs(dist - out.distance[rows, cols])[same]
    assert np.median(error) < 1e-3
    assert np.quantile(error, 0.99) < 1e-2  # tessellated spheres: 6 mm at P99 from "close"


@pytest.mark.parametrize("kb4", [False, True])
def test_leaves_mjr_render_unchanged(scene, kb4):
    mujoco, _, data, renderer, ours = scene
    renderer.update_scene(data, camera="close")
    before = renderer.render().copy()
    for _ in range(3):
        ours.render(renderer.scene, *(_wrist_camera() if kb4 else ()))
    after = renderer.render()
    assert np.abs(before.astype(int) - after.astype(int)).max() <= 1


_GL_OBJECTS = {"glCreateProgram": "glIsProgram", "glCreateShader": "glIsShader", "glGenBuffers": "glIsBuffer",
               "glGenFramebuffers": "glIsFramebuffer", "glGenRenderbuffers": "glIsRenderbuffer",
               "glGenTextures": "glIsTexture", "glGenVertexArrays": "glIsVertexArray"}


@pytest.mark.parametrize("failure", ["width", "compile", "link", "framebuffer"])
def test_failed_init_deletes_gl_objects(gpu_egl, monkeypatch, failure):
    from OpenGL import GL

    from robosuite_env.rendering import mujoco_kb4

    mujoco = gpu_egl
    model = mujoco.MjModel.from_xml_string(SCENE)
    renderer = mujoco.Renderer(model, 240, 320)
    created = []

    def record(gen, is_name):
        def wrapper(*args):
            names = gen(*args)
            created.extend((is_name, int(n)) for n in np.atleast_1d(names))
            return names
        return wrapper

    for gen, is_name in _GL_OBJECTS.items():
        monkeypatch.setattr(GL, gen, record(getattr(GL, gen), is_name))
    if failure == "compile":
        monkeypatch.setattr(mujoco_kb4, "_FRAGMENT_SHADER", "#version 460\nvoid main() { undeclared = 1; }")
    elif failure == "link":  # the fragment shader reads inputs that this vertex shader does not write
        monkeypatch.setattr(mujoco_kb4, "_VERTEX_SHADER", "#version 460\nvoid main() { gl_Position = vec4(0); }")
    elif failure == "framebuffer":
        status, calls = GL.glCheckFramebufferStatus, iter(range(3))
        monkeypatch.setattr(GL, "glCheckFramebufferStatus", lambda target: status(target) if next(calls) < 2 else 0)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            mujoco_kb4.SceneRenderer(model, 0 if failure == "width" else 320, 240, renderer)
        monkeypatch.undo()
        renderer._gl_context.make_current()
        assert [(is_name, n) for is_name, n in created if getattr(GL, is_name)(n)] == []
    finally:
        renderer.close()


@pytest.mark.parametrize("feature", ["shadow", "reflective", "segment", "framewidth", "spot cutoff"])
def test_rejects_unsupported_scene_features(scene, feature):
    mujoco, model, data, renderer, ours = scene
    if feature == "reflective":
        model.mat_reflectance[0] = 0.5
    renderer.update_scene(data, camera="side")
    if feature == "shadow":
        renderer.scene.lights[0].castshadow = 1
    elif feature == "segment":
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
    elif feature == "framewidth":
        renderer.scene.framewidth = 2
    elif feature == "spot cutoff":
        renderer.scene.lights[2].cutoff = 120
    with pytest.raises(NotImplementedError, match=feature):
        ours.render(renderer.scene)


# A camera among the geoms, so that the plane, the large box and the pyramid reach 90 to 100 degrees off its
# axis, and six axis-aligned 90-degree pinhole views at the same place as an independent colour reference.
_AMONG = np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]])  # fisheye frame (x right, y up, z forward) to world
#: (right, up, forward) of each reference pinhole view in the fisheye frame.
_REFERENCE_VIEW_AXES = {
    "front": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "right": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
    "left": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
    "up": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "down": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    "back": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
}
_REFERENCE_VIEW_AXES = {name: tuple(np.array(axis) for axis in axes) for name, axes in _REFERENCE_VIEW_AXES.items()}


def _reference_cameras():
    return "".join(
        f'<camera name="among_{name}" pos=".1 -.75 .1" fovy="90" xyaxes="{" ".join(map(str, _AMONG @ right))} '
        f'{" ".join(map(str, _AMONG @ up))}"/>' for name, (right, up, _) in _REFERENCE_VIEW_AXES.items())


def _kb4_scene_xml():
    among = f'<camera name="among" pos=".1 -.75 .1" xyaxes="0 1 0 0 0 1"/>{_reference_cameras()}'
    # The default 28x16 sphere mesh is 4 to 6 mm off mj_ray's exact sphere at P95, which would dominate the check.
    buffer = '<visual><global offwidth="1024" offheight="1024"/><quality numslices="200" numstacks="120"/></visual>'
    return SCENE.replace("<worldbody>", f"{buffer}<worldbody>").replace("</worldbody>", f"{among}</worldbody>")


_TESSELLATED = ("sphere", "ellipsoid", "capsule", "cylinder")  # mj_ray hits the exact shape


@pytest.fixture()
def kb4_scene(gpu_egl):
    from robosuite_env.rendering.mujoco_kb4 import SceneRenderer

    mujoco = gpu_egl
    model = mujoco.MjModel.from_xml_string(_kb4_scene_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 240, 320)
    ours = SceneRenderer(model, 320, 240, renderer)
    yield mujoco, model, data, renderer, ours
    ours.close()
    renderer.close()


def _wrist_camera(profile=DEFAULT_POLICY_WRIST_RENDER_PROFILE):
    from robosuite_env.rendering.wrist_render_profiles import get_wrist_render_profile, wrist_source_intrinsics

    profile = get_wrist_render_profile(profile)
    return wrist_source_intrinsics(profile.name), np.radians(profile.source_fov_deg / 2)


def _fisheye_ray_distance(mujoco, model, data, scene, rays):
    """``mj_multiRay`` along OpenCV-frame rays of the scene camera: (model geom id, distance)."""

    cam = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
    forward, up = np.asarray(cam.forward), np.asarray(cam.up)
    vec = rays @ np.stack([np.cross(forward, up), -up, forward])
    geomid = np.zeros(len(vec), np.int32)
    dist = np.zeros(len(vec))
    mujoco.mj_multiRay(model, data, np.asarray(cam.pos, np.float64), np.ascontiguousarray(vec).ravel(), None, 1, -1,
                       geomid, dist, None, len(vec), 1e6)
    return geomid, dist


@pytest.mark.parametrize("profile", WRIST_RENDER_PROFILE_NAMES)
@pytest.mark.parametrize("camera", ["side", "close", "top", "among"])
def test_kb4_matches_mj_ray(kb4_scene, camera, profile):
    from robosuite_env.rendering.wrist_render_profiles import camera_rays

    mujoco, model, data, renderer, ours = kb4_scene
    fisheye, theta_max = _wrist_camera(profile)
    renderer.update_scene(data, camera=camera)
    out = ours.render(renderer.scene, fisheye, theta_max)

    rays, valid = camera_rays(fisheye, theta_max)
    geomid, dist = _fisheye_ray_distance(mujoco, model, data, renderer.scene, rays[valid].astype(np.float64))
    objid = np.array([renderer.scene.geoms[i].objid for i in range(renderer.scene.ngeom)] + [-1])
    ours_geom = objid[out.segid[valid]]
    hit, ours_hit = geomid >= 0, ours_geom >= 0
    assert (hit & ours_hit).sum() / (hit | ours_hit).sum() >= 0.998
    same = hit & (geomid == ours_geom)
    assert same.sum() >= 0.99 * hit.sum()
    error = np.abs(dist - out.distance[valid])[same]
    assert np.quantile(error, 0.95) <= 2e-3
    # Planes, boxes and meshes are exact, so only rounding remains. A crack shows as an empty pixel
    # whose ray hits a geom and whose 8 neighbours are occupied.
    flat = same & ~np.isin(model.geom_type[np.maximum(geomid, 0)],
                           [getattr(mujoco.mjtGeom, f"mjGEOM_{name.upper()}") for name in _TESSELLATED])
    assert np.quantile(np.abs(dist - out.distance[valid])[flat], 0.99) < 1e-5
    assert np.abs(dist - out.distance[valid])[flat].max() < 1e-3  # the plane grid's zero-width edge strips draw nothing
    occupied = np.zeros(valid.shape, bool)
    occupied[valid] = ours_hit
    padded = np.pad(occupied, 1)
    neighbours = sum(np.roll(np.roll(padded, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1))[1:-1, 1:-1]
    rays_hit = np.zeros(valid.shape, bool)
    rays_hit[valid] = hit
    assert not (rays_hit & ~occupied & (neighbours == 8)).any()


def _aim_headlight_down(scene):
    for i in range(scene.nlight):  # as WristFisheye._fix_headlight, so that the reference views share one lighting
        if scene.lights[i].headlight:
            scene.lights[i].dir[:] = (0.0, 0.0, -1.0)


def test_kb4_colour_matches_six_pinhole_views(kb4_scene):
    """Away from outlines, the KB4 colour must equal six ``mjr_render`` pinhole views sampled at the same rays."""

    import cv2 as cv

    from robosuite_env.rendering.wrist_render_profiles import camera_rays

    mujoco, model, data, renderer, ours = kb4_scene
    fisheye, theta_max = _wrist_camera()
    renderer.update_scene(data, camera="among")
    _aim_headlight_down(renderer.scene)
    out = ours.render(renderer.scene, fisheye, theta_max)

    rays, valid = camera_rays(fisheye, theta_max)
    rays = rays.astype(np.float64) * (1, -1, 1)
    reference = np.zeros_like(out.rgb)
    faces = mujoco.Renderer(model, 1024, 1024)
    try:
        for name, (right, up, forward) in _REFERENCE_VIEW_AXES.items():
            faces.update_scene(data, camera=f"among_{name}")
            _aim_headlight_down(faces.scene)
            depth = np.maximum(rays @ forward, 1e-8)
            sx, sy = rays @ right / depth, rays @ up / depth
            on = valid & (rays @ forward > 0) & (np.abs(sx) <= 1) & (np.abs(sy) <= 1)
            x, y = ((sx + 1) * 512 - 0.5).astype(np.float32), ((1 - sy) * 512 - 0.5).astype(np.float32)
            reference[on] = cv.remap(faces.render(), x, y, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)[on]
    finally:
        faces.close()
    interior = cv.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    psnr = _psnr(reference[interior], out.rgb[interior])
    assert psnr >= 38.0, psnr  # 41.5 dB


_SPHERES = """
<mujoco>
  <asset><material name="glow" emission="1"/></asset>
  <worldbody>
    <camera name="eye" pos="0 0 0" xyaxes="1 0 0 0 1 0"/>
    {spheres}
  </worldbody>
</mujoco>
"""


# (theta, phi) [deg] past 60 degrees that fit in each source raster, and blobs stay 50 px apart. The 719x539 raster
# fits only near-horizontal blobs beyond 80 degrees; the 640x480 ELP raster fits only near-diagonal blobs beyond 70.
_FAR_DIRECTIONS = {
    FRANKA_PINHOLE_WRIST_RGB_PROFILE: [(80, 0), (80, 180), (90, 15), (90, 195), (95, 345), (95, 165), (98.5, 35),
                                       (98.5, 215)],
    ELP_MEAN_WRIST_RGB_PROFILE: [(70, 0), (95, 325), (95, 145), (98.5, 35), (98.5, 215)],
}


@pytest.mark.parametrize("profile", WRIST_RENDER_PROFILE_NAMES)
@pytest.mark.parametrize("supersample", [1, 1.5, 2])
def test_wrist_source_reprojects_sphere_centres(gpu_egl, supersample, profile):
    """An emissive sphere's intensity centroid must land on ``project_points`` of its centre."""

    from robosuite_env.rendering.mujoco_kb4 import WristSourceRenderer
    from robosuite_env.rendering.wrist_render_profiles import project_points

    mujoco = gpu_egl
    distance, radius = 0.5, 0.5 * np.sin(np.radians(1.0))
    directions = [(0, 0), (30, 0), (30, 120), (30, 240), (60, 60), (60, 180), (60, 300)] + _FAR_DIRECTIONS[profile]
    centres = np.array([distance * np.array([np.sin(np.radians(t)) * np.cos(np.radians(p)),
                                             np.sin(np.radians(t)) * np.sin(np.radians(p)), -np.cos(np.radians(t))])
                        for t, p in directions])
    model = mujoco.MjModel.from_xml_string(_SPHERES.format(spheres="".join(
        f'<geom type="sphere" size="{radius}" pos="{x} {y} {z}" material="glow"/>' for x, y, z in centres)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 240, 320)
    wrist = WristSourceRenderer(model, renderer, profile, supersample=supersample)
    try:
        renderer.update_scene(data, camera="eye")
        out = wrist.render(renderer.scene)
    finally:
        wrist.close()
        renderer.close()

    assert out.rgb.shape == (wrist.camera.height, wrist.camera.width, 3)
    assert not out.rgb[~wrist.valid].any() and (out.segid[~wrist.valid] == -1).all()
    assert np.isnan(out.distance[~wrist.valid]).all()
    expected = project_points(wrist.camera, centres * np.array([1.0, -1.0, -1.0]))
    intensity = out.rgb.astype(np.float64).mean(axis=2)
    rows, cols = np.mgrid[0:intensity.shape[0], 0:intensity.shape[1]]
    errors = []
    for x, y in expected:
        window = (np.abs(cols - x) <= 12) & (np.abs(rows - y) <= 12)
        weight = intensity * window
        support = weight.sum()
        assert np.isfinite(support) and support > 0, (x, y, support)
        errors.append(np.hypot((weight * cols).sum() / support - x, (weight * rows).sum() / support - y))
    assert np.isfinite(errors).all(), errors
    assert max(errors) <= 0.25, np.round(errors, 3)
    assert (np.isfinite(out.distance) == (out.segid >= 0)).all()


def test_kb4_needs_theta_max_and_a_monotonic_lens(kb4_scene):
    from dataclasses import replace

    mujoco, model, data, renderer, ours = kb4_scene
    fisheye, theta_max = _wrist_camera()
    renderer.update_scene(data, camera="close")
    with pytest.raises(ValueError, match="theta_max"):
        ours.render(renderer.scene, fisheye)
    folded = replace(fisheye, k1=-0.46675501614553866, k2=0.09803706034615614, k3=0.0, k4=0.0)
    with pytest.raises(ValueError, match="monotonic"):
        ours.render(renderer.scene, folded, theta_max)
