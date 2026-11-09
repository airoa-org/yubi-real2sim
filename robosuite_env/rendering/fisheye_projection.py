# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light calibrated wrist fisheye constants.

``franka-pinhole-wrist-rgb`` renders its source raster with this calibration
(``source_intrinsics``) before the profile crop. Keeping it free of MuJoCo lets
CPU-only contract CI read the camera model.
"""

from __future__ import annotations

from .fisheye_model import FisheyeIntrinsics


# Mean ELP USB fisheye calibration used by the original training simulator.
# These constants intentionally match the reference values in
# tests/fixtures/franka_yubi_sim/build_scene.py, which
# tests/unit/rendering/test_wrist_render.py compares against this module.
OUT_W = 640
OUT_H = 480
FX = 240.54
FY = 240.28
CX = 324.92
CY = 233.28
K1 = -0.0446
K2 = 0.0009
K3 = -0.0039
K4 = 0.0007
FULL_FOV_DEG = 178.0


def source_intrinsics(width=OUT_W, height=OUT_H):
    return FisheyeIntrinsics(
        width, height, FX, FY, CX + (width - OUT_W) * 0.5,
        CY + (height - OUT_H) * 0.5, K1, K2, K3, K4,
    )


__all__ = [
    "CX",
    "CY",
    "FULL_FOV_DEG",
    "FX",
    "FY",
    "K1",
    "K2",
    "K3",
    "K4",
    "OUT_H",
    "OUT_W",
]
