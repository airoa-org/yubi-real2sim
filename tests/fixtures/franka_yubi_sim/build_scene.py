# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Wrist-camera calibration constants of the original training simulator.

``robosuite_env/rendering/fisheye_projection.py`` copies these values, and
``tests/unit/rendering/test_wrist_render.py`` reads them from this file to
check that the copies match.
"""

# Mean ELP USB fisheye calibration of ten "HD USB Camera" units, 640x480 VGA,
# OpenCV cv2.fisheye model.
CALIB_IMAGE_WIDTH = 640
CALIB_IMAGE_HEIGHT = 480
CALIB_FX = 240.54
CALIB_FY = 240.28
CALIB_CX = 324.92
CALIB_CY = 233.28
CALIB_K1 = -0.0446
CALIB_K2 = 0.0009
CALIB_K3 = -0.0039
CALIB_K4 = 0.0007
CALIB_RECT_HFOV_DEG = 135.5
CALIB_RECT_VFOV_DEG = 123.34
CALIB_FULL_FOV_DEG = 178.0
