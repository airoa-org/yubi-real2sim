# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Box packing on the calibrated FR3 rig and its registered 3DGS desk."""

from __future__ import annotations

from .box_pack import configure_box_pack_task
from .cup_plate_registered import CupPlateLiftGW


class BoxPackLiftGW(CupPlateLiftGW):
    """Cup/plate geometry and calibration, box-packing objects instead."""

    def _configure_task_objects(self):
        configure_box_pack_task(self)

    def _weld_placeholder_bodies(self):
        # The box and the tape, not the prompt's target: all twelve prompts
        # compile the same four objects, so this pair keeps the model
        # fingerprint identical across them.
        return (
            self.box_pack_logical_objects["box"].root_body,
            self.box_pack_logical_objects["tape"].root_body,
        )
