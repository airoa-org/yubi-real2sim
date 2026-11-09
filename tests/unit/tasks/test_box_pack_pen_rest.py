# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Whether the box-packing layouts author the pen resting on one of its facets.

The pen's two collision hulls are not round: resting on a flat table it has a
set of discrete rest facets about its long axis (body y), and an orientation
between them rolls onto one once physics runs. Body z up (the identity
orientation) is not a facet: through ``env.reset()`` it rolls 20 deg in 3 s
and 45.6 deg in 15 s on the Duo table. The Duo task-default layout authors
the pen at identity, so its case is expected to fail (strict xfail): the policy
sees the pen after it has rolled.

Here the pen is compiled the way the environment compiles it (the registered
``PenObject`` in robosuite's world, whose option block the box_pack env keeps),
placed in each committed layout's authored orientation on a flat slab with the
registered table's contact parameters, seated with the production
``seat_lift``, and left alone for 15 s under the implicitfast integrator every
box_pack profile selects. A rest state barely moves; the identity orientation
turns by tens of degrees.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("robosuite")

from robosuite_env.tasks.layouts import convex_hull_halfspaces, seat_lift  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
OBJ_CALIB = REPO_ROOT / "robosuite_env" / "obj_calib"
PEN_HULL_MESHES = tuple(
    REPO_ROOT / "robosuite_env" / "assets" / "trellis_office" / "coll" / name
    for name in ("pen_000.obj", "pen_001.obj")
)
#: The Duo task-default layout authors the pen at identity, off a rest facet.
DUO_PEN_OFF_FACET = pytest.mark.xfail(
    strict=True,
    reason="the Duo layout authors the pen at identity, off a rest facet: it rolls 45.6 deg onto "
    "a facet during the 12 s ready drive and is static from frame 0, so the policy sees it rolled",
)
BOX_PACK_LAYOUTS = (
    pytest.param(
        OBJ_CALIB / "layout_box_pack_duo_manual.json",
        id="duo",
        marks=DUO_PEN_OFF_FACET,
    ),
)

SETTLE_S = 15.0
MAX_ROTATION_DEG = 2.0
#: The Duo desk's collision slab (duo_table_plane.obj): 1.24 m x 1.045 m x 50 mm,
#: compiled with MuJoCo's default contact parameters like every registered
#: table part. Its top is at z = 0 here; seating puts the pen on it.
TABLE_HALF_EXTENT_M = (0.62, 0.5225, 0.025)

pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in PEN_HULL_MESHES),
    reason="the pen collision meshes (gitignored trellis_office/coll) are not installed",
)


def _pen_on_a_flat_table():
    from robosuite.models.world import MujocoWorldBase

    from robosuite_env.tasks.objects import PenObject

    world = MujocoWorldBase()
    pen = PenObject("pen")
    world.worldbody.append(pen.get_obj())
    world.merge_assets(pen)
    sx, sy, sz = TABLE_HALF_EXTENT_M
    corners = " ".join(
        f"{x} {y} {z}" for x in (-sx, sx) for y in (-sy, sy) for z in (-2 * sz, 0.0)
    )
    ET.SubElement(world.asset, "mesh", name="table_slab", vertex=corners)
    ET.SubElement(world.worldbody, "geom", name="table_slab", type="mesh", mesh="table_slab")
    # Visual-only geoms carry no mass (the world compiles inertia from group 0
    # only) and no contact; dropping them skips a 700k-vertex render mesh.
    for parent in world.root.iter():
        for child in list(parent):
            if (
                child.tag == "geom"
                and child.get("contype") == "0"
                and child.get("conaffinity") == "0"
            ):
                parent.remove(child)
    used_meshes = {geom.get("mesh") for geom in world.root.iter("geom")}
    for child in list(world.asset):
        if child.tag in ("texture", "material") or (
            child.tag == "mesh" and child.get("name") not in used_meshes
        ):
            world.asset.remove(child)
    model = mujoco.MjModel.from_xml_string(world.get_xml())
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return model


def _world_vertices(model, data, geom_id):
    mesh_id = int(model.geom_dataid[geom_id])
    start = int(model.mesh_vertadr[mesh_id])
    local = np.asarray(
        model.mesh_vert[start : start + int(model.mesh_vertnum[mesh_id])], dtype=float
    )
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    return local @ rotation.T + np.asarray(data.geom_xpos[geom_id], dtype=float)


def _rotation_deg(first, second):
    return math.degrees(2.0 * math.acos(min(1.0, abs(float(np.dot(first, second))))))


@pytest.mark.parametrize("layout_path", BOX_PACK_LAYOUTS)
def test_the_authored_pen_orientation_is_a_rest_state(layout_path):
    quaternion = json.loads(layout_path.read_text(encoding="utf-8"))["objects"]["pen"][
        "quaternion_wxyz"
    ]
    model = _pen_on_a_flat_table()
    data = mujoco.MjData(model)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pen_joint0")
    body = int(model.jnt_bodyid[joint])
    qpos = int(model.jnt_qposadr[joint])
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_slab")
    pen_geoms = [
        geom
        for geom in range(model.ngeom)
        if int(model.geom_bodyid[geom]) == body
        and (model.geom_contype[geom] or model.geom_conaffinity[geom])
    ]
    assert len(pen_geoms) == 2

    data.qpos[qpos : qpos + 3] = 0.0
    data.qpos[qpos + 3 : qpos + 7] = quaternion
    mujoco.mj_kinematics(model, data)
    lift = seat_lift(
        [convex_hull_halfspaces(_world_vertices(model, data, g)) for g in pen_geoms],
        [convex_hull_halfspaces(_world_vertices(model, data, table))],
    )
    data.qpos[qpos + 2] = lift
    mujoco.mj_forward(model, data)
    start = np.asarray(data.qpos[qpos : qpos + 7], dtype=float).copy()

    worst_rotation = worst_translation = 0.0
    for _ in range(int(round(SETTLE_S / model.opt.timestep))):
        mujoco.mj_step(model, data)
        worst_rotation = max(
            worst_rotation, _rotation_deg(start[3:], data.qpos[qpos + 3 : qpos + 7])
        )
        worst_translation = max(
            worst_translation,
            float(np.linalg.norm(data.qpos[qpos : qpos + 3] - start[:3])),
        )

    assert worst_rotation < MAX_ROTATION_DEG, (
        f"{layout_path.name}: the pen authored at quaternion_wxyz={quaternion} turned "
        f"{worst_rotation:.2f} deg (moved {worst_translation * 1000.0:.2f} mm) within "
        f"{SETTLE_S:g} s on a flat table; author it on a rest facet"
    )
