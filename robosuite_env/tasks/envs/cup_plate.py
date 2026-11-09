#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Cup + plate, both as native robosuite objects in a robosuite manipulation Environment.
(Lift base; the manipulation target stays `self.cube` = cup so Lift's refs/obs/reward keep working,
plate is added to the task scene.)"""
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

from ..objects import CupObject, PlateObject
from ..equalities import add_rule_grasp_weld_equalities
from ..scene import place_robots_at_stock_offsets


class CupPlateLift(Lift):
    """Lift with our cup as the target object and our plate placed on the table."""
    def _load_model(self):
        super(Lift, self)._load_model()
        place_robots_at_stock_offsets(
            self.robots, table_length=self.table_full_size[0]
        )
        arena = TableArena(table_full_size=self.table_full_size,
                           table_friction=self.table_friction, table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        self.cube = CupObject("cube")     # cup = manipulation target (keep attr name for Lift)
        self.plate = PlateObject("plate")
        # cup/plate table-frame XY (override via env). Default is the canonical run layout:
        # cup on the right-hand (robot0, -y) side, plate on the left (+y), same x-depth, ~0.20 m
        # apart so they do NOT overlap (plate r~0.09 + cup r~0.04 << 0.20).
        import os as _os
        _cx, _cy = [float(v) for v in _os.environ.get("RS_CUP_XY", "0.0,-0.10").split(",")]
        _px, _py = [float(v) for v in _os.environ.get("RS_PLATE_XY", "0.0,0.10").split(",")]
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(UniformRandomSampler(
            name="PlateSampler", mujoco_objects=self.plate, x_range=[_px, _px], y_range=[_py, _py],
            rotation=0.0, ensure_object_boundary_in_range=False, ensure_valid_placement=False,
            reference_pos=self.table_offset, z_offset=0.001, rng=self.rng))
        self.placement_initializer.append_sampler(UniformRandomSampler(
            name="CupSampler", mujoco_objects=self.cube, x_range=[_cx, _cx], y_range=[_cy, _cy],
            rotation=0.0, ensure_object_boundary_in_range=False, ensure_valid_placement=False,
            reference_pos=self.table_offset, z_offset=0.02, rng=self.rng))
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=[self.cube, self.plate],
        )
        add_rule_grasp_weld_equalities(
            self.model.root,
            placeholder_bodies=(self.cube.root_body, self.plate.root_body),
        )
