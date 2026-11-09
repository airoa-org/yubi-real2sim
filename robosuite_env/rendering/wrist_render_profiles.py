# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The production wrist-render contracts.

``elp-mean-wrist-rgb`` is the default; ``franka-pinhole-wrist-rgb`` remains selectable.
Earlier development variants are not kept.
Runtime and diagnostic callers share these immutable profiles so a rollout
cannot silently select an older camera, crop, resolution, or optical-center
convention.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from real2sim.wrist_profiles import (
    DEFAULT_POLICY_WRIST_RENDER_PROFILE,
    ELP_MEAN_WRIST_RGB_PROFILE,
    FRANKA_PINHOLE_WRIST_RGB_PROFILE,
    WRIST_RENDER_PROFILE_NAMES as WRIST_RENDER_PROFILE_NAMES,  # public re-export
)

from .fisheye_model import (
    FisheyeIntrinsics as FisheyeIntrinsics,
    camera_rays as camera_rays,
    local_projection_jacobian as local_projection_jacobian,
    project_points as project_points,
    projection_jacobian as projection_jacobian,
    ray_tangent_basis as ray_tangent_basis,
)
from .fisheye_projection import FULL_FOV_DEG, OUT_H, OUT_W, source_intrinsics


@dataclass(frozen=True)
class WristCrop:
    """FOV guide followed by each edge of one wrist image."""

    left_edge_fov_deg: float
    right_edge_fov_deg: float
    top_edge_fov_deg: float
    bottom_edge_fov_deg: float

    @property
    def effective_horizontal_fov_deg(self) -> float:
        return (self.left_edge_fov_deg + self.right_edge_fov_deg) * 0.5

    @property
    def effective_vertical_fov_deg(self) -> float:
        return (self.top_edge_fov_deg + self.bottom_edge_fov_deg) * 0.5

    @property
    def horizontal_optical_axis_fraction(self) -> float:
        return self.left_edge_fov_deg / (
            self.left_edge_fov_deg + self.right_edge_fov_deg
        )

    @property
    def vertical_optical_axis_fraction(self) -> float:
        return self.top_edge_fov_deg / (
            self.top_edge_fov_deg + self.bottom_edge_fov_deg
        )


@dataclass(frozen=True)
class WristRenderProfile:
    """Complete immutable defaults for production wrist rendering.

    A profile either crops ``FX``..``K4`` rendered at ``source_fov_deg``
    (``left_wrist`` / ``right_wrist``), or writes its ``camera`` model directly;
    then pixels whose ray lies beyond ``source_fov_deg / 2`` stay black.
    """

    name: str
    contract_version: int
    source_fov_deg: float
    left_wrist: WristCrop | None
    right_wrist: WristCrop | None
    base_face_size_178: int
    camera_backward_m: float
    provenance: str
    camera: FisheyeIntrinsics | None = None

    @property
    def source_image_hw(self) -> tuple[int, int]:
        """Return the exact pre-policy wrist raster declared by this profile.

        This is part of the camera contract, not something callers may infer
        from black pixels in a letterboxed policy image.
        """

        if self.camera is not None:
            return self.camera.height, self.camera.width
        scale = float(self.source_fov_deg) / float(FULL_FOV_DEG)
        return (
            max(1, int(round(float(OUT_H) * scale))),
            max(1, int(round(float(OUT_W) * scale))),
        )

    def crop(self, side: str) -> WristCrop:
        if side not in ("left", "right"):
            raise ValueError(f"wrist side must be 'left' or 'right', got {side!r}")
        if self.camera is not None:
            raise ValueError(f"wrist render profile {self.name!r} writes its camera model uncropped")
        return self.left_wrist if side == "left" else self.right_wrist


_PROFILE = WristRenderProfile(
    name=FRANKA_PINHOLE_WRIST_RGB_PROFILE,
    contract_version=2,
    source_fov_deg=200.0,
    left_wrist=WristCrop(
        left_edge_fov_deg=160.0,
        right_edge_fov_deg=160.0,
        top_edge_fov_deg=150.0,
        bottom_edge_fov_deg=160.0,
    ),
    right_wrist=WristCrop(
        left_edge_fov_deg=155.0,
        right_edge_fov_deg=165.0,
        top_edge_fov_deg=155.0,
        bottom_edge_fov_deg=155.0,
    ),
    base_face_size_178=640,
    camera_backward_m=0.010,
    provenance=(
        "initial-pose real-vs-sim selection; right crop uses "
        "left/right edge guides 155/165 degrees and top/bottom 155/155; "
        "left remains 160/160 and 150/160"
    ),
)
# Written image (wrist_image_intrinsics), 719x539 and in brackets 640x480; k1..k4 = K1..K4:
#   left   fx 300.76  fy 309.82  cx 365.78  cy 251.98   (267.71  275.91  325.53  224.34)
#   right  fx 300.76  fy 309.82  cx 354.78  cy 260.98   (267.71  275.91  315.74  232.36)

_ELP_MEAN_PROFILE = WristRenderProfile(
    name=ELP_MEAN_WRIST_RGB_PROFILE,
    contract_version=3,
    source_fov_deg=200.0,
    left_wrist=None,
    right_wrist=None,
    base_face_size_178=640,
    camera_backward_m=0.010,
    provenance=(
        "mean KB4 board calibration at 640x480 of 9 ELP "
        "wrist units (USBCAM027/070/097/117/129/147, cam000/040/050); k2..k4 are "
        "the self-calibration population priors; one model for both wrists"
    ),
    camera=FisheyeIntrinsics(
        width=640,
        height=480,
        fx=270.47,
        fy=270.47,
        cx=320.5,
        cy=242.5,
        k1=-0.0426,
        k2=-0.002615,
        k3=-0.0013373,
        k4=0.00012556,
    ),
)

_PROFILES = {profile.name: profile for profile in (_PROFILE, _ELP_MEAN_PROFILE)}


def get_wrist_render_profile(name: str = DEFAULT_POLICY_WRIST_RENDER_PROFILE) -> WristRenderProfile:
    """Return one of the supported wrist profiles."""

    try:
        return _PROFILES[str(name)]
    except KeyError:
        raise ValueError(
            f"unknown wrist render profile {name!r}; the only supported profiles "
            f"are {', '.join(map(repr, _PROFILES))}"
        ) from None


def crop_window(length: int, scale: float, center_fraction: float) -> tuple[int, int]:
    """Scaled length and crop origin of one axis of the profile crop."""

    scaled = max(length, int(round(length * float(scale))))
    origin = int(round(scaled * 0.5 - length * float(center_fraction)))
    return scaled, max(0, min(origin, scaled - length))


def wrist_source_intrinsics(profile: str) -> FisheyeIntrinsics:
    """KB4 model of the raster the renderer draws, before the crop of ``wrist_image_intrinsics``."""

    prof = get_wrist_render_profile(profile)
    if prof.camera is not None:
        return prof.camera
    height, width = prof.source_image_hw
    return source_intrinsics(width, height)


def wrist_image_geometry(profile: str, side: str):
    """Source model, resized dimensions and crop origin of one written wrist image."""

    prof = get_wrist_render_profile(profile)
    if side not in ("left", "right"):
        raise ValueError(f"wrist side must be 'left' or 'right', got {side!r}")
    height, width = prof.source_image_hw
    if prof.camera is not None:
        return prof.camera, width, height, 0, 0
    crop = prof.crop(side)
    scaled_width, x0 = crop_window(
        width, prof.source_fov_deg / crop.effective_horizontal_fov_deg,
        crop.horizontal_optical_axis_fraction,
    )
    scaled_height, y0 = crop_window(
        height, prof.source_fov_deg / crop.effective_vertical_fov_deg,
        crop.vertical_optical_axis_fraction,
    )
    return wrist_source_intrinsics(profile), scaled_width, scaled_height, x0, y0


def wrist_image_intrinsics(profile: str, side: str) -> FisheyeIntrinsics:
    """KB4 model of the written image, including the profile's resize and crop."""

    source, scaled_width, scaled_height, x0, y0 = wrist_image_geometry(profile, side)
    scaled = source.resized(scaled_width, scaled_height)
    return replace(
        scaled, width=source.width, height=source.height, cx=scaled.cx - x0, cy=scaled.cy - y0
    )
