# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Two exact-pose samplers in one composite must both survive.

The composite indexes children by name. Distinct selectable names ensure
that adding a second sampler does not overwrite the first object's placement.
"""

from __future__ import annotations

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robosuite_env.tasks.envs.placement import ExactObjectPoseSampler


class _Object:
    """The little of a robosuite object that the samplers touch."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.joints = [f"{name}_joint0"]

    @property
    def bottom_offset(self):
        return np.zeros(3)

    @property
    def horizontal_radius(self):
        return 0.0


def _sampler(names, *, sampler_name=None):
    placements = [
        (name, _Object(name), [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]) for name in names
    ]
    kwargs = {"reference_pos": np.zeros(3)}
    if sampler_name is not None:
        kwargs["name"] = sampler_name
    return ExactObjectPoseSampler(placements, **kwargs)


def test_two_samplers_under_one_name_lose_the_first():
    """The failure itself, so the fix below is measured against something."""

    composite = SequentialCompositeSampler(name="ObjectSampler")
    composite.append_sampler(_sampler(["container"]))
    composite.append_sampler(_sampler(["part_a", "part_b"]))

    assert len(composite.samplers) == 1
    placed = {
        obj.name for sampler in composite.samplers.values() for obj in sampler.mujoco_objects
    }
    assert "container" not in placed


def test_distinct_names_keep_both_samplers():
    composite = SequentialCompositeSampler(name="ObjectSampler")
    composite.append_sampler(_sampler(["container"], sampler_name="ContainerSampler"))
    composite.append_sampler(
        _sampler(["part_a", "part_b"], sampler_name="PartsSampler")
    )

    assert len(composite.samplers) == 2
    placed = {
        obj.name for sampler in composite.samplers.values() for obj in sampler.mujoco_objects
    }
    assert placed == {"container", "part_a", "part_b"}


def test_the_default_name_is_unchanged_for_single_sampler_callers():
    """Recorded, container and box each append one; their argv must not shift."""

    sampler = _sampler(["cup"])
    assert sampler.name == "ExactObjectPoseSampler"


def test_the_composite_passes_on_top_to_each_child():
    """SequentialCompositeSampler.sample calls each child with on_top=, so sample() keeps it."""

    composite = SequentialCompositeSampler(name="ObjectSampler")
    composite.append_sampler(_sampler(["cup"]))
    placed = composite.sample()
    assert set(placed) == {"cup"}
