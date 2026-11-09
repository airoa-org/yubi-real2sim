# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Repository and runtime resource paths shared by robosuite modules."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
ASSETS_ROOT = PACKAGE_ROOT / "assets"
CONTROLLERS_ROOT = PACKAGE_ROOT / "controllers"
OBJECT_CALIBRATION_ROOT = PACKAGE_ROOT / "obj_calib"


__all__ = [
    "ASSETS_ROOT",
    "CONTROLLERS_ROOT",
    "OBJECT_CALIBRATION_ROOT",
    "PACKAGE_ROOT",
    "REPOSITORY_ROOT",
]
