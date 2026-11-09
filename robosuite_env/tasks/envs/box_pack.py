# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Environment wiring for the twelve box-packing prompts."""

from __future__ import annotations

import os

from ..box_pack import (
    BOX_PACK_OBJECT_NAMES,
    BOX_PACK_RECEPTACLE,
    box_pack_layout_path,
)
from ..layout_selection import (
    load_selected_layout,
    require_layout_rig,
    require_layout_scene,
)
from .placement import ExactObjectPoseSampler


def configure_box_pack_task(env) -> None:
    """Attach the box and the three packable objects at the authored poses."""

    from ..objects import OBJECT_ASSET_REGISTRY

    missing = sorted(set(BOX_PACK_OBJECT_NAMES) - set(OBJECT_ASSET_REGISTRY))
    if missing:
        raise ValueError(
            f"box pack object asset registry is incomplete: missing={missing}"
        )

    # For a list-valued profile the batch runner first selects this run_NN's
    # path and passes that scalar as --box-pack-layout-json. The worker exports
    # the selected profile path under RS_BOX_PACK_LAYOUT_JSON, so the shared
    # selection order resolves it here. A layout placed for another scene than
    # this env's is refused.
    selection = box_pack_layout_path()
    layout, _payload, record = load_selected_layout(
        selection, expected_objects=BOX_PACK_OBJECT_NAMES
    )
    require_layout_scene(
        layout, running_scene=getattr(env, "gw_scene_tag", None), selection=selection
    )
    # A layout that names its rigs (source.robots) is refused on any other.
    require_layout_rig(
        layout,
        robot=getattr(getattr(env, "base_placement", None), "robot", None),
        selection=selection,
    )
    # The worker exports the prompt's object from --box-pack-task-index.
    target = os.environ.get("RS_BOX_PACK_OBJECT", "tape").strip().lower()
    if target not in BOX_PACK_OBJECT_NAMES or target == BOX_PACK_RECEPTACLE:
        raise ValueError(
            "RS_BOX_PACK_OBJECT must name one packable object "
            f"{[n for n in BOX_PACK_OBJECT_NAMES if n != BOX_PACK_RECEPTACLE]}; "
            f"got {target!r}"
        )

    env.box_pack_logical_objects = {
        name: OBJECT_ASSET_REGISTRY[name](name=name)
        for name in BOX_PACK_OBJECT_NAMES
    }
    # The prompt's object is what the policy grips, so it takes the ``cube``
    # target role; the distractors and the box stay on the table with it.
    env.cube = env.box_pack_logical_objects[target]
    env.box = env.box_pack_logical_objects[BOX_PACK_RECEPTACLE]
    # Same as the container: the shared _load_model expects a plate slot.
    env.plate = None
    env.extra_task_objects = [
        obj for name, obj in env.box_pack_logical_objects.items() if name != target
    ]
    # The file and the digest of the bytes just validated, for provenance.
    env.task_layout_record = record
    env.box_pack_target_object = target

    placements = [
        (
            name,
            env.box_pack_logical_objects[name],
            layout["objects"][name]["position_m"],
            layout["objects"][name]["quaternion_wxyz"],
        )
        for name in BOX_PACK_OBJECT_NAMES
    ]
    env.placement_initializer = ExactObjectPoseSampler(
        placements, reference_pos=env.table_offset
    )
