# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Wrist profile names shared by the launcher and renderer."""

FRANKA_PINHOLE_WRIST_RGB_PROFILE = "franka-pinhole-wrist-rgb"
ELP_MEAN_WRIST_RGB_PROFILE = "elp-mean-wrist-rgb"
DEFAULT_POLICY_WRIST_RENDER_PROFILE = ELP_MEAN_WRIST_RGB_PROFILE
WRIST_RENDER_PROFILE_NAMES = (
    FRANKA_PINHOLE_WRIST_RGB_PROFILE,
    ELP_MEAN_WRIST_RGB_PROFILE,
)
WRIST_BACKENDS = ("direct_kb4",)
