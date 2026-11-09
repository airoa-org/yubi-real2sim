# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Calibrated object placement shared by task environments."""

from copy import copy

import numpy as np
from robosuite.utils.placement_samplers import ObjectPositionSampler


class ExactObjectPoseSampler(ObjectPositionSampler):
    """Return calibrated full 6-DoF poses without yaw-only resampling."""

    def __init__(self, placements, *, reference_pos, name="ExactObjectPoseSampler"):
        # The name is a key, not a label: SequentialCompositeSampler stores its
        # children in a dict keyed by it, so two of these appended to the same
        # composite under one name silently leave only the second, and whatever
        # the first was meant to place never moves and stays at the world origin.
        objects = [entry[1] for entry in placements]
        super().__init__(
            name=name,
            mujoco_objects=objects,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
            reference_pos=np.asarray(reference_pos, dtype=float),
        )
        self._placements = tuple(placements)

    def reset(self):
        # These are calibrated scene anchors, not a mutable random sampler.
        return None

    def sample(self, fixtures=None, reference=None, on_top=True):  # noqa: V107  # SequentialCompositeSampler.sample passes on_top
        if reference is not None:
            raise ValueError("exact poses do not accept a relative reference")
        placed = {} if fixtures is None else copy(fixtures)
        base = np.asarray(self.reference_pos, dtype=float)
        for logical_name, obj, position_m, quaternion_wxyz in self._placements:
            if obj.name in placed:
                raise ValueError(
                    f"layout object {logical_name!r} / {obj.name!r} was placed twice"
                )
            position = base + np.asarray(position_m, dtype=float)
            quaternion = np.asarray(quaternion_wxyz, dtype=float)
            placed[obj.name] = (tuple(position), quaternion.copy(), obj)
        return placed
