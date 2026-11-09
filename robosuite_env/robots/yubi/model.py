#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Native right- and left-hand YUBI robosuite gripper models."""
import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers import register_gripper

from ...paths import ASSETS_ROOT

YUBI_XML = str(ASSETS_ROOT / "yubi_right.xml")
YUBI_XML_LEFT = str(ASSETS_ROOT / "yubi_left.xml")


class _YubiGripperBase(GripperModel):
    """Two revolute fingers (axis +/-Y, range 0..0.7854 rad; 0=closed, 0.785=open). One action drives both."""

    xml_path = YUBI_XML
    collision_counts = (14, 6, 19, 5)

    def __init__(self, idn=0):
        super().__init__(self.xml_path, idn=idn)

    def format_action(self, action):  # noqa: V105  # CompositeController calls it
        # DIRECT position passthrough: action in [-1,1] -> both fingers same target (controller scales to
        # the joint range). The policy grip output is an ABSOLUTE finger angle, so we must NOT accumulate
        # (the Panda-style velocity accumulator made grip a velocity command -> never tracked the target).
        assert len(action) == self.dof
        self.current_action = np.array([action[0], action[0]], dtype=float)
        return self.current_action

    @property  # noqa: V106  # Robot.reset reads it
    def init_qpos(self):
        return np.array([0.0, 0.0])

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 1

    @property  # noqa: V106  # MujocoXMLModel.important_geoms reads it
    def _important_geoms(self):
        left_count, left_pad_count, right_count, right_pad_count = self.collision_counts
        left_collision = [f"finger_left_collision_h{i:03d}" for i in range(left_count)]
        left_pad_collision = [
            f"finger_left_pad_collision_h{i:03d}" for i in range(left_pad_count)
        ]
        right_collision = [f"finger_right_collision_h{i:03d}" for i in range(right_count)]
        right_pad_collision = [
            f"finger_right_pad_collision_h{i:03d}" for i in range(right_pad_count)
        ]
        return {
            "left_finger": left_collision + left_pad_collision,
            "right_finger": right_collision + right_pad_collision,
            "left_fingerpad": left_pad_collision,
            "right_fingerpad": right_pad_collision,
        }


@register_gripper  # noqa: V102  # robosuite gripper registry
class YubiGripper(_YubiGripperBase):
    """Right-hand YUBI geometry."""


@register_gripper  # noqa: V102  # robosuite gripper registry
class YubiGripperLeft(_YubiGripperBase):
    """Mirrored left-hand YUBI geometry used by the logical-left policy arm."""

    xml_path = YUBI_XML_LEFT
    collision_counts = (20, 6, 13, 6)
