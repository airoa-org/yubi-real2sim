#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Production wrist observations from one direct KB4 source camera per wrist."""
import contextlib
from dataclasses import dataclass
import os
os.environ.setdefault("MUJOCO_GL", "egl")
_RENDERING_DIR = os.path.dirname(os.path.abspath(__file__))
_DIR = os.path.dirname(_RENDERING_DIR)
import numpy as np
import cv2 as cv
try:
    import mujoco
except ModuleNotFoundError:  # CPU-only contract tests exercise pure helpers.
    mujoco = None
from real2sim.wrist_profiles import WRIST_BACKENDS

from .fisheye_projection import (
    FULL_FOV_DEG,
    CX as CX,
    CY as CY,
    FX as FX,
    FY as FY,
    K1 as K1,
    K2 as K2,
    K3 as K3,
    K4 as K4,
    OUT_H,
    OUT_W,
)
from .rgb_color_calibration import (
    DEFAULT_RGB_COLOR_CALIBRATION,
    apply_rgb_color_calibration,
    load_rgb_color_calibration,
)
from .wrist_render_profiles import (
    DEFAULT_POLICY_WRIST_RENDER_PROFILE,
    crop_window,
    get_wrist_render_profile,
    wrist_source_intrinsics,
)

if mujoco is not None:
    _VIS_ONLY = mujoco.MjvOption()
    _VIS_ONLY.geomgroup[:] = 0
    _VIS_ONLY.geomgroup[1] = 1  # visual meshes only (hide collision = robosuite blue)
else:
    _VIS_ONLY = None


def _require_mujoco():
    if mujoco is None:
        raise RuntimeError(
            "MuJoCo is required to construct a wrist renderer; install the "
            "locked runtime environment"
        )
    return mujoco

# `geometry42_visual` is the physical YUBI camera housing/lens mesh.  It is
# part of the gripper visual model, but must not be rendered by any wrist
# camera because it is the camera's own body.
_WRIST_CAMERA_EXCLUDED_GEOM_SUFFIXES = ("geometry42_visual",)


@dataclass(frozen=True)
class _CapturedWristFace:
    """One face detached from the mutable GPU rendering transaction."""

    name: str
    rgb: np.ndarray
    foreground_mask: np.ndarray | None
    background: np.ndarray | None
    mujoco_depth: np.ndarray | None = None
    background_depth: np.ndarray | None = None


@dataclass(frozen=True)
class WristFisheyePending:
    """Immutable container passed from GPU capture to CPU finalization.

    The arrays are exposed through read-only views.  This keeps the captured
    frame stable while another render transaction reuses the MuJoCo renderer.
    """

    faces: tuple[_CapturedWristFace, ...]




@dataclass(frozen=True)
class WristFisheyeMujocoCapture:
    """MuJoCo pixels plus the detached GS batch request."""

    faces: tuple[_CapturedWristFace, ...]
    batch_face_names: tuple[str, ...] | None
    batch_payloads: tuple[dict, ...] | None
    batch_render: object | None


def _readonly_view(array):
    if array is None:
        return None
    view = np.asarray(array).view()
    view.setflags(write=False)
    return view


def _split_gs_render_result(result):
    """Return ``(rgb, depth)`` for depth-aware and legacy GS clients."""

    if result is None:
        return None, None
    if hasattr(result, "rgb"):
        return result.rgb, getattr(result, "depth", None)
    if isinstance(result, dict) and "rgb" in result:
        return result["rgb"], result.get("depth")
    if (
        not isinstance(result, np.ndarray)
        and isinstance(result, (tuple, list))
        and len(result) == 2
    ):
        return result[0], result[1]
    return result, None


def _quat_matrix(quat):
    output = np.empty(9, dtype=np.float64)
    _require_mujoco().mju_quat2Mat(
        output, np.ascontiguousarray(quat, dtype=np.float64)
    )
    return output.reshape(3, 3)


def _center_scale_crop(
    image,
    scale_x,
    scale_y,
    *,
    crop_x_center_fraction=0.5,
    crop_y_center_fraction=0.5,
    interpolation=None,
):
    """Apply an image-space FOV crop, optionally shifted on either axis.

    ``interpolation`` defaults to bilinear; depth maps pass nearest so the
    crop never blends a foreground and a background distance.

    ``crop_x_center_fraction`` is the optical-axis location in the output
    image measured from the left edge.
    ``crop_y_center_fraction`` is the optical-axis location in the output
    image.  The default keeps the historical centered crop.  A value below
    0.5 places the optical axis above the image center, which is useful when
    the requested upper and lower FOV edges are asymmetric.
    """

    import cv2 as cv

    array = np.asarray(image)
    height, width = array.shape[:2]
    if not 0.0 < float(crop_x_center_fraction) < 1.0:
        raise ValueError("crop_x_center_fraction must be between 0 and 1")
    if not 0.0 < float(crop_y_center_fraction) < 1.0:
        raise ValueError("crop_y_center_fraction must be between 0 and 1")
    # The source optical axis is at the center of the scaled raster.  Choose
    # the crop origin so that it lands at the requested output fraction.
    scaled_width, x0 = crop_window(width, scale_x, crop_x_center_fraction)
    scaled_height, y0 = crop_window(height, scale_y, crop_y_center_fraction)
    scaled = cv.resize(
        array,
        (scaled_width, scaled_height),
        interpolation=cv.INTER_LINEAR if interpolation is None else interpolation,
    )
    return np.ascontiguousarray(scaled[y0 : y0 + height, x0 : x0 + width])


class WristFisheye:
    """Render and composite one wrist through detached MuJoCo, GS and CPU stages."""
    def __init__(
        self,
        env,
        prefix="gripper0_right",
        gs_client=None,
        black_background=False,
        render_lock=None,
        side=None,
        *,
        render_profile=DEFAULT_POLICY_WRIST_RENDER_PROFILE,
        camera_side=None,
        rgb_color_calibration_path=None,
        composite_mode="depth",
        gs_only=False,
        renderer=None,
        backend,
        source_renderer=None,
    ):
        _require_mujoco()
        if backend not in WRIST_BACKENDS:
            raise ValueError(f"unknown wrist backend {backend!r}; expected {WRIST_BACKENDS}")
        self.backend = backend
        self._source_renderer = None
        self._owns_source_renderer = False
        if gs_client is not None:
            if not callable(getattr(gs_client, "require_camera_model", None)):
                raise RuntimeError("direct_kb4 requires a GS client with camera-model negotiation")
            gs_client.require_camera_model("kb4")
        self.env = env
        self.m = env.sim.model._model
        self.side = side
        self.rgb_color_calibration = load_rgb_color_calibration(
            DEFAULT_RGB_COLOR_CALIBRATION
            if rgb_color_calibration_path is None
            else rgb_color_calibration_path
        )
        self.gs = gs_client
        self.composite_mode = str(composite_mode).strip().lower()
        if self.composite_mode not in {"depth", "foreground"}:
            raise ValueError("composite_mode must be 'depth' or 'foreground'")
        self.gs_only = bool(gs_only)
        self.black_background = bool(black_background)
        self._render_lock = render_lock or contextlib.nullcontext
        self.profile = get_wrist_render_profile(render_profile)
        self.render_profile = self.profile.name
        self.camera_model = self.profile.camera
        self.camera_side = camera_side
        self.source_fov_deg = float(self.profile.source_fov_deg)
        self.source_height, self.source_width = self.profile.source_image_hw
        # The MuJoCo renderer keeps its historical square size: 640 px at 178 degrees,
        # scaled to the source view.
        self.face_size = max(1, int(round(
            int(self.profile.base_face_size_178) * (self.source_fov_deg / float(FULL_FOV_DEG))
        )))
        if self.camera_side not in {"left", "right"}:
            raise ValueError(
                "camera_side must be 'left' or 'right' for the calibrated "
                "YUBI wrist render profile"
            )
        self.face_cam = {"front": f"{prefix}_wrist_fisheye_front"}
        self.face_cam_ids = {}
        for f, nm in self.face_cam.items():
            camera_id = mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_CAMERA, nm
            )
            if camera_id < 0:
                raise RuntimeError(f"missing wrist camera {nm}")
            self.face_cam_ids[f] = int(camera_id)
        self._profile_camera_positions = {}
        self._configure_versioned_wrist_profile()
        # The shared scene XML historically allocated only a 640x480 offscreen
        # buffer. MuJoCo reads these dimensions when each Renderer is constructed.
        self.m.vis.global_.offwidth = max(
            int(self.m.vis.global_.offwidth), int(self.face_size)
        )
        self.m.vis.global_.offheight = max(
            int(self.m.vis.global_.offheight), int(self.face_size)
        )
        if renderer is None:
            self.renderer = mujoco.Renderer(
                self.m, self.face_size, self.face_size
            )
            self._owns_renderer = True
        else:
            # A left/right pair is always rendered sequentially under the
            # same GPU lock, so it can safely share one EGL context. This
            # reduces the pair's context count; cross-process context lifetime
            # exclusion remains the caller's responsibility.
            self.renderer = renderer
            self._owns_renderer = False
        try:
            if source_renderer is None:
                from .mujoco_kb4 import WristSourceRenderer

                self._source_renderer = WristSourceRenderer(
                    self.m, self.renderer, self.render_profile
                )
                self._owns_source_renderer = True
            else:
                if source_renderer.camera != wrist_source_intrinsics(self.render_profile):
                    raise ValueError("shared direct wrist renderer has different source intrinsics")
                self._source_renderer = source_renderer
            self.valid = self._source_renderer.valid
        except BaseException:
            self.close()
            raise
        if gs_client is not None:
            from scene.gs_background_client import (
                composite_background_depth,
                composite_background_mask,
            )

            self._composite = composite_background_mask
            self._composite_depth = (
                composite_background_depth if self.composite_mode == "depth" else None
            )

    def _configure_versioned_wrist_profile(self):
        """Apply the production camera setback, exclusion, and crop contract."""

        front_id = self.face_cam_ids["front"]
        front_quat = np.asarray(self.m.cam_quat[front_id], dtype=np.float64)
        forward_local = _quat_matrix(front_quat) @ np.asarray(
            [0.0, 0.0, -1.0], dtype=np.float64
        )
        for camera_id in self.face_cam_ids.values():
            self._profile_camera_positions[camera_id] = (
                np.asarray(self.m.cam_pos[camera_id], dtype=np.float64)
                - forward_local * self.profile.camera_backward_m
            )

        self._profile_excluded_geom_ids = tuple(
            geom_id
            for geom_id in range(self.m.ngeom)
            if (
                (
                    mujoco.mj_id2name(
                        self.m, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    )
                    or ""
                ).endswith(_WRIST_CAMERA_EXCLUDED_GEOM_SUFFIXES)
            )
        )

        self._profile_scale_x = self._profile_scale_y = 1.0
        self._profile_horizontal_center_fraction = 0.5
        self._profile_vertical_center_fraction = 0.5
        if self.camera_model is None:
            crop = self.profile.crop(self.camera_side)
            self._profile_scale_x = self.source_fov_deg / crop.effective_horizontal_fov_deg
            self._profile_scale_y = self.source_fov_deg / crop.effective_vertical_fov_deg
            self._profile_horizontal_center_fraction = crop.horizontal_optical_axis_fraction
            self._profile_vertical_center_fraction = crop.vertical_optical_axis_fraction

    @staticmethod
    def _fix_headlight(scn):
        # The MuJoCo headlight follows the camera's forward axis. Re-aim it to one fixed world
        # direction (straight down) so the wrist lighting does not depend on the view. The source
        # camera keeps this so its images keep the established lighting.
        for i in range(scn.nlight):
            L = scn.lights[i]
            if L.headlight:
                L.dir[:] = (0.0, 0.0, -1.0)

    @contextlib.contextmanager
    def _profile_transaction(self):
        with self._render_lock():
            data = self.env.sim.data._data
            positions = {
                camera_id: self.m.cam_pos[camera_id].copy()
                for camera_id in getattr(self, "_profile_camera_positions", {})
            }
            groups = {
                geom_id: int(self.m.geom_group[geom_id])
                for geom_id in getattr(self, "_profile_excluded_geom_ids", ())
            }
            try:
                for camera_id, position in getattr(self, "_profile_camera_positions", {}).items():
                    self.m.cam_pos[camera_id] = position
                for geom_id in groups:
                    self.m.geom_group[geom_id] = 5
                if positions:
                    mujoco.mj_camlight(self.m, data)  # update_scene reads cached data.cam_xpos.
                yield data
            finally:
                for geom_id, group in groups.items():
                    self.m.geom_group[geom_id] = group
                for camera_id, position in positions.items():
                    self.m.cam_pos[camera_id] = position
                if positions:
                    mujoco.mj_camlight(self.m, data)

    def _update_wrist_scene(self, data, camera):
        self.renderer.update_scene(data, camera=camera, scene_option=_VIS_ONLY)
        self._fix_headlight(self.renderer.scene)
        if self.black_background:
            for name in ("mjRND_SKYBOX", "mjRND_HAZE", "mjRND_FOG"):
                if hasattr(mujoco.mjtRndFlag, name):
                    self.renderer.scene.flags[int(getattr(mujoco.mjtRndFlag, name))] = 0

    def capture_mujoco_stage(self):
        """Capture the immutable MuJoCo source raster and the GS camera request.

        MuJoCo rendering and camera snapshots share one transaction. The
        method performs no network/CUDA request. Callers with two wrists can
        therefore capture both EGL renderers before starting any 3DGS work,
        avoiding the corrupting MuJoCo -> GS -> MuJoCo sequence.
        """

        batch_render = None
        payloads = None
        with self._profile_transaction() as data:
            self._update_wrist_scene(data, self.face_cam["front"])
            source = self._source_renderer.render(self.renderer.scene)
            if self.gs is not None:
                camera = self._source_renderer.camera
                settings = camera.as_dict()
                settings.pop("model")
                settings.pop("width")
                settings.pop("height")
                settings.update(camera_model="kb4", theta_max=self._source_renderer.theta_max)
                payloads = (self.gs.camera_payload(
                    self.renderer.scene, camera.width, camera.height,
                    camera_settings=settings,
                ),)
                batch_render = self.gs.render_payloads_with_depth
        return WristFisheyeMujocoCapture(
            faces=(_CapturedWristFace(
                name="source", rgb=_readonly_view(source.rgb.copy()),
                foreground_mask=_readonly_view(source.segid >= 0), background=None,
                mujoco_depth=_readonly_view(source.distance.copy()),
            ),),
            batch_face_names=("source",) if payloads is not None else None,
            batch_payloads=payloads, batch_render=batch_render,
        )

    def complete_gs_stage(self, capture):
        """Run deferred 3DGS work and attach its detached CPU results."""

        if not isinstance(capture, WristFisheyeMujocoCapture):
            raise TypeError("capture must be a WristFisheyeMujocoCapture")
        backgrounds = {}
        background_depths = {}

        if capture.batch_render is not None:
            rendered = capture.batch_render(capture.batch_payloads)
            if len(rendered) != len(capture.batch_face_names):
                raise RuntimeError(
                    "3DGS wrist batch returned "
                    f"{len(rendered)} faces for "
                    f"{len(capture.batch_face_names)} cameras"
                )
            shape = (self.source_height, self.source_width)
            for f, result in zip(capture.batch_face_names, rendered):
                background, background_depth = _split_gs_render_result(result)
                if background is None or np.asarray(background).shape != (*shape, 3):
                    raise RuntimeError("direct KB4 GS result has no matching source RGB raster")
                if getattr(result, "depth_convention", None) != "radial":
                    raise RuntimeError("direct KB4 GS result must declare radial depth")
                if background_depth is None or np.asarray(background_depth).shape != shape:
                    raise RuntimeError("direct KB4 GS result has no matching source depth raster")
                background_depth = np.where(self.valid, background_depth, np.nan)
                background = np.where(self.valid[..., None], background, 0).astype(np.uint8)
                backgrounds[f] = background
                background_depths[f] = background_depth

        return WristFisheyePending(
            faces=tuple(
                _CapturedWristFace(
                    name=face.name,
                    rgb=face.rgb,
                    mujoco_depth=face.mujoco_depth,
                    foreground_mask=face.foreground_mask,
                    background=_readonly_view(backgrounds.get(face.name)),
                    background_depth=_readonly_view(
                        background_depths.get(face.name)
                    ),
                )
                for face in capture.faces
            )
        )

    def capture_gpu_stage(self):
        """Compatibility wrapper for one wrist's EGL then 3DGS stages."""

        return self.complete_gs_stage(self.capture_mujoco_stage())

    def finalize_cpu_stage(self, pending):
        """Composite and project a captured frame without taking the GPU lock."""

        if not isinstance(pending, WristFisheyePending):
            raise TypeError("pending must be a WristFisheyePending")
        faces = {}
        for face in pending.faces:
            image = face.rgb
            if face.background is not None:
                foreground_mask = face.foreground_mask
                if getattr(self, "gs_only", False):
                    foreground_mask = np.zeros(
                        np.asarray(image).shape[:2], dtype=bool
                    )
                composite_depth = getattr(self, "_composite_depth", None)
                if (
                    callable(composite_depth)
                    and face.mujoco_depth is not None
                    and face.background_depth is not None
                ):
                    image = composite_depth(
                        image,
                        face.mujoco_depth,
                        foreground_mask,
                        face.background,
                        face.background_depth,
                    )
                else:
                    image = self._composite(
                        image, foreground_mask, face.background
                    )
            faces[face.name] = image
        if tuple(faces) != ("source",):
            raise ValueError("direct KB4 capture must contain exactly one source raster")
        image = np.where(self.valid[..., None], faces["source"], 0).astype(np.uint8)
        if getattr(self, "camera_model", None) is None:
            image = _center_scale_crop(
                image,
                self._profile_scale_x,
                self._profile_scale_y,
                crop_x_center_fraction=self._profile_horizontal_center_fraction,
                crop_y_center_fraction=self._profile_vertical_center_fraction,
            )
        calibration = getattr(self, "rgb_color_calibration", None)
        if calibration is None:
            return image
        return apply_rgb_color_calibration(image, calibration)

    def render(self):
        """Compatibility wrapper around the split GPU and CPU stages."""

        return self.finalize_cpu_stage(self.capture_gpu_stage())

    def close(self):
        """Release an owned MuJoCo renderer, or detach a shared handle."""
        source = getattr(self, "_source_renderer", None)
        if source is not None:
            if self._owns_source_renderer:
                source.close()
            self._source_renderer = None
        renderer = getattr(self, "renderer", None)
        if renderer is not None:
            if getattr(self, "_owns_renderer", True):
                renderer.close()
                # Keep the handle when close raises so the centralized cleanup
                # can retry it while the EGL display is still valid.
            self.renderer = None


if __name__ == "__main__":
    import cv2 as cv
    from ..robots.yubi import model as _yubi_model  # noqa: F401
    from ..tasks.envs.cup_plate import CupPlateLift
    from robosuite.controllers import load_composite_controller_config
    cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
    env = CupPlateLift(robots="Panda", controller_configs=cfg, gripper_types="YubiGripper",
                       has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, control_freq=20)
    env.reset()
    for _ in range(40):
        env.step(np.array([0, 0, -0.4, 0, 0, 0, 0.0]))   # descend toward the cup
    wf = WristFisheye(env, backend="direct_kb4", side="right", camera_side="right")
    fish = wf.render()
    out = os.path.join(_DIR, "wrist_fisheye.png")
    cv.imwrite(out, cv.cvtColor(fish, cv.COLOR_RGB2BGR))
    wf.close()
    env.close()
    print(f"[wrist-fisheye] wrote {out}  shape={fish.shape}  (OUT={OUT_W}x{OUT_H})")
