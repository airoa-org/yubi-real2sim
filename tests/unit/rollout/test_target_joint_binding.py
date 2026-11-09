# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Bind cup and box targets through their own compiled free-joint names.

The env.cube attribute identifies the target, but its object name can be tape
or pen. An absent hardcoded joint name returns -1 and indexes another joint.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("robosuite")

from robosuite.models.arenas import EmptyArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask

from tests._helpers.asset_trees import requires_repository_meshes


def _joints(names_to_assets):
    objects = [
        MujocoXMLObject(
            f"robosuite_env/assets/{asset}.xml",
            name=name,
            joints=[dict(type="free")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        for name, asset in names_to_assets
    ]
    model = ManipulationTask(
        mujoco_arena=EmptyArena(), mujoco_robots=[], mujoco_objects=objects
    ).get_model(mode="mujoco")
    return model, [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)
    ]




@pytest.mark.assets
@requires_repository_meshes("cup", "plate", "box", "tape", "pen", "lan")
def test_the_target_joint_is_named_after_the_object_not_the_attribute():
    """Resolving from the object's name works for every task's target."""


    for target, asset_set in (
        # cup tasks: the object really is called "cube", so this is a no-op
        ("cube", [("cube", "cup"), ("plate", "plate")]),
        # box-pack: the target is whichever object the prompt names
        ("tape", [("box", "box"), ("tape", "tape"), ("pen", "pen"), ("lan", "lan")]),
        ("pen", [("box", "box"), ("tape", "tape"), ("pen", "pen"), ("lan", "lan")]),
    ):
        model, names = _joints(asset_set)
        joint = f"{target}_joint0"
        assert joint in names, f"{joint} missing from {names}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        assert jid >= 0
        # The address the ready drive pins must belong to the target itself.
        assert int(model.jnt_qposadr[jid]) == int(model.jnt_qposadr[names.index(joint)])
