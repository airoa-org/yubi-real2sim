# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The table surface the success criteria measure heights from.

These run on a compiled MuJoCo model rather than a fake, because the failure
this guards against lives in the compiled geometry: MuJoCo recentres mesh
vertices and moves the difference into ``geom_pos`` / ``geom_quat``, a table
part may hang off a static child body, and a multi-part CoACD table has no
single top. A CoACD-decomposed table can be 25 such parts whose highest vertex sits
5.5 mm above where most objects stand, so "the highest table vertex" is not
the table.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from robosuite_env.tasks.success._geometry import (  # noqa: E402
    TableDatum,
    TableSurface,
    table_collision_geom_ids,
)


SLAB_TOP_Z_M = 0.034

_XML = """
<mujoco>
  <compiler angle="degree"/>
  <asset>
    <!-- 1.0 x 0.8 m slab, 30 mm thick, top face at local z = 0 -->
    <mesh name="slab" vertex="-0.5 -0.4 -0.03  0.5 -0.4 -0.03  0.5 0.4 -0.03  -0.5 0.4 -0.03
                              -0.5 -0.4 0  0.5 -0.4 0  0.5 0.4 0  -0.5 0.4 0"/>
    <!-- a wedge whose top rises 10 mm along its 0.2 m local x -->
    <mesh name="wedge" vertex="-0.1 -0.05 -0.02  0.1 -0.05 -0.02  0.1 0.05 -0.02  -0.1 0.05 -0.02
                               -0.1 -0.05 0  0.1 -0.05 0.01  0.1 0.05 0.01  -0.1 0.05 0"/>
  </asset>
  <worldbody>
    <geom name="gw_desk_coll_0" type="mesh" mesh="slab" pos="0 0 0.034" group="3"/>
    <!-- A raised part on a static, rotated child body: world x in [0.40, 0.50],
         world y in [-0.1, 0.1], top rising from 39.5 mm to 49.5 mm along +y. -->
    <body name="lip_mount" pos="0.45 0 0.034" euler="0 0 90">
      <geom name="gw_desk_coll_1" type="mesh" mesh="wedge" pos="0 0 0.0055" group="3"/>
    </body>
    <body name="drawer" pos="-0.3 0 0">
      <joint name="drawer_slide" type="slide" axis="1 0 0"/>
      <geom name="gw_desk_coll_2" type="box" size="0.05 0.05 0.01"/>
    </body>
    <body name="cup" pos="0 0 0.2">
      <freejoint/>
      <geom name="cup_collision" type="box" size="0.02 0.02 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def table():
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = table_collision_geom_ids(model)
    return model, data, ids, TableSurface.from_model(model, ids)


def _name(model, geom_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)


def test_the_table_is_found_by_name_and_a_moving_part_is_not_measured(table):
    model, _, ids, surface = table

    assert {_name(model, g) for g in ids} == {
        "gw_desk_coll_0",
        "gw_desk_coll_1",
        "gw_desk_coll_2",
    }
    assert surface.geoms == ("gw_desk_coll_0", "gw_desk_coll_1")
    assert surface.unmeasured == ("gw_desk_coll_2",)


def test_the_surface_under_a_point_is_not_the_highest_table_vertex(table):
    _, _, _, surface = table

    assert surface.top_z_m == pytest.approx(0.0495, abs=1e-9)
    assert surface.height_at(0.0, 0.0) == pytest.approx(SLAB_TOP_Z_M, abs=1e-9)
    # Over the overlapping raised part, the higher of the two tops.
    assert surface.height_at(0.45, 0.0) == pytest.approx(0.0445, abs=1e-9)
    assert surface.height_at(2.0, 0.0) is None


def test_the_surface_matches_mujoco_ray_casts_everywhere(table):
    model, data, _, surface = table
    table_group_only = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
    hit = np.zeros(1, dtype=np.int32)

    checked = 0
    for x in np.arange(-0.575, 0.6, 0.05):
        for y in np.arange(-0.475, 0.5, 0.05):
            start = np.array([x, y, 1.0])
            down = np.array([0.0, 0.0, -1.0])
            try:
                distance = mujoco.mj_ray(
                    model, data, start, down, table_group_only, 1, -1, hit
                )
            except TypeError:
                distance = mujoco.mj_ray(
                    model, data, start, down, table_group_only, 1, -1, hit,
                    np.zeros(3),
                )
            expected = None if distance < 0 else 1.0 - distance
            measured = surface.height_at(float(x), float(y))
            assert (measured is None) == (expected is None), (x, y)
            if expected is not None:
                assert measured == pytest.approx(expected, abs=1e-9), (x, y)
                checked += 1
    assert checked > 300






def test_the_episode_datum_is_the_surface_under_the_footprint(table):
    model, _, _, _ = table
    datum = TableDatum(model=model, options={"table_z_m": 0.0}, owner="test")

    z_m = datum.resolve(
        (0.0, 0.0, 0.05),
        np.eye(3),
        (-0.08, -0.12, 0.0),
        (0.08, 0.12, 0.024),
        subject="container",
    )

    assert z_m == pytest.approx(SLAB_TOP_Z_M, abs=1e-9)
    described = datum.describe()
    assert described["source"] == "model:table_collision_surface_under_container_footprint"
    assert described["table_collision"]["top_vertex_z_m"] == pytest.approx(0.0495)
    assert described["reference_plane_z_m"] == 0.0


def test_a_footprint_off_the_table_falls_back_to_the_top_and_says_so(table):
    model, _, _, _ = table
    datum = TableDatum(model=model, options={}, owner="test")

    z_m = datum.resolve(
        (3.0, 3.0, 0.05), np.eye(3), (-0.05, -0.05, 0.0), (0.05, 0.05, 0.02),
        subject="box",
    )

    assert z_m == pytest.approx(0.0495, abs=1e-9)
    assert "not over the table" in datum.describe()["source"]
