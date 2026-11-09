# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Layout objects start resting on the surface that supports them.

A hand-authored layout writes z against a flat table plane at z = 0. The
compiled table is elsewhere -- the Duo collision top is +34 mm, and a CoACD table can
be 25 uneven hulls with legs and an apron under the top -- so the registered
environment seats every exactly placed object on the real support geometry
instead of trusting the typed number.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from robosuite_env.tasks.layouts import (  # noqa: E402
    LayoutError,
    convex_hull_halfspaces,
    hull_surface_z,
    seat_lift,
)

#: 0.1 micrometre: MuJoCo stores mesh vertices as float32 and the exact
#: overlap solve carries HiGHS's feasibility tolerance.
EXACT = 1e-7


def _box(lo, hi):
    return np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
        dtype=float,
    )


def _hull(lo, hi):
    return convex_hull_halfspaces(_box(lo, hi))


# --- geometry -----------------------------------------------------------------

DUO_TABLE = [_hull((-0.625, -0.895, -0.016), (0.615, 0.150, 0.034))]


def test_an_object_buried_in_the_table_is_lifted_exactly_onto_its_top():
    pen = [_hull((0.126, -0.603, 0.0), (0.144, -0.457, 0.0176))]

    assert seat_lift(pen, DUO_TABLE) == pytest.approx(0.034, abs=EXACT)


def test_a_floating_object_is_lowered_onto_the_table():
    tape = [_hull((-0.045, -0.585, 0.200), (0.035, -0.505, 0.252))]

    assert seat_lift(tape, DUO_TABLE) == pytest.approx(0.034 - 0.200, abs=EXACT)


def test_an_object_off_the_table_has_no_seat():
    stray = [_hull((2.0, 2.0, 0.0), (2.1, 2.1, 0.05))]

    assert seat_lift(stray, DUO_TABLE) is None


# A CoACD-like table: the top slab is not the highest hull (a raised lip at
# the far edge is), and a leg and an apron sit under it.
SLAB = _hull((-0.70, -0.10, -0.030), (0.70, 0.50, 0.000))
LIP = _hull((-0.70, 0.50, -0.030), (0.70, 0.55, 0.020))
LEG = _hull((-0.70, -0.10, -0.700), (-0.62, -0.02, -0.030))
APRON = _hull((-0.70, -0.10, -0.120), (0.70, -0.08, -0.030))
SCENE_TABLE = [LEG, APRON, SLAB, LIP]


def test_the_top_surface_is_found_among_legs_aprons_and_edge_hulls():
    naive_top = max(float(vertices[:, 2].max()) for _, vertices in SCENE_TABLE)
    cup_mid_table = [_hull((0.00, 0.20, 0.00), (0.07, 0.27, 0.077))]
    cup_over_leg = [_hull((-0.69, -0.09, 0.00), (-0.63, -0.03, 0.077))]

    assert naive_top == pytest.approx(0.020)
    assert seat_lift(cup_mid_table, SCENE_TABLE) == pytest.approx(0.0, abs=EXACT)
    assert seat_lift(cup_over_leg, SCENE_TABLE) == pytest.approx(0.0, abs=EXACT)
    xy = np.array([[0.0, 0.2], [0.0, 0.52], [-0.66, -0.06], [3.0, 3.0]])
    np.testing.assert_allclose(
        hull_surface_z(xy, SCENE_TABLE, upper=True), [0.0, 0.020, 0.0, -np.inf]
    )


def test_a_narrow_edge_hull_crossing_under_a_wide_base_lifts_it():
    # No vertex of the plate lies over the lip and no vertex of the lip lies
    # under the plate: only the edges cross. Vertex sampling alone says 0.
    plate_on_lip = [_hull((0.00, 0.47, 0.00), (0.16, 0.63, 0.017))]

    assert seat_lift(plate_on_lip, SCENE_TABLE) == pytest.approx(0.020, abs=EXACT)


def test_an_uneven_top_seats_on_its_highest_point_under_the_footprint():
    tilted = convex_hull_halfspaces(
        np.array(
            [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.05, 0.01 * x)]
        )
    )
    block = [_hull((0.10, -0.05, 0.0), (0.20, 0.05, 0.03))]

    assert seat_lift(block, [tilted]) == pytest.approx(0.002, abs=EXACT)


def test_a_table_peak_between_the_object_vertices_still_lifts_it():
    # The bump's apex lies under the middle of a wide flat base that has
    # vertices only at its corners.
    bump = convex_hull_halfspaces(
        np.array(
            [[-0.02, -0.02, -0.01], [0.02, -0.02, -0.01], [-0.02, 0.02, -0.01],
             [0.02, 0.02, -0.01], [0.0, 0.0, 0.010]]
        )
    )
    flat = _hull((-0.30, -0.30, -0.05), (0.30, 0.30, 0.0))
    base = [_hull((-0.10, -0.10, 0.0), (0.10, 0.10, 0.02))]

    assert seat_lift(base, [flat, bump]) == pytest.approx(0.010, abs=EXACT)


def test_a_degenerate_hull_is_rejected():
    with pytest.raises(LayoutError, match="degenerate"):
        convex_hull_halfspaces(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float))


# --- MuJoCo: the registered environment's seating ------------------------------

TABLE_TOP = 0.034
BOX_FLOOR = 0.006
# The runtime object assets' contact parameters, so the settle below behaves
# like theirs rather than like MuJoCo's softer defaults.
CONTACT = 'solref="0.005 1" solimp="0.95 0.99 0.001"'


def _v(points):
    return " ".join(f"{value:.9g}" for value in np.asarray(points, float).ravel())


def _model_xml(*, pen_xy=(0.135, -0.53), tape_xy=(0.40, -0.30)):
    table = _box((-0.625, -0.895, -0.016), (0.615, 0.150, TABLE_TOP))
    pen_a = _box((-0.009, -0.073, 0.0), (0.009, 0.0, 0.0176))
    pen_b = _box((-0.009, 0.0, 0.0), (0.009, 0.073, 0.0176))
    floor = _box((-0.08, -0.12, 0.0), (0.08, 0.12, BOX_FLOOR))
    rim = _box((-0.08, 0.11, 0.0), (0.08, 0.12, 0.0242))
    tape = _box((-0.025, -0.025, 0.0), (0.025, 0.025, 0.0226))
    return f"""
<mujoco>
  <option timestep="0.002" integrator="implicitfast"/>
  <asset>
    <mesh name="table" vertex="{_v(table)}"/>
    <mesh name="pen_a" vertex="{_v(pen_a)}"/>
    <mesh name="pen_b" vertex="{_v(pen_b)}"/>
    <mesh name="floor" vertex="{_v(floor)}"/>
    <mesh name="rim" vertex="{_v(rim)}"/>
    <mesh name="tape" vertex="{_v(tape)}"/>
  </asset>
  <worldbody>
    <geom name="gw_desk_coll_0" type="mesh" mesh="table"/>
    <body name="pen" pos="{pen_xy[0]} {pen_xy[1]} 0.0" quat="0.9238795 0 0 0.3826834">
      <freejoint name="pen_joint0"/>
      <geom name="pen_c0" type="mesh" mesh="pen_a" density="900" {CONTACT}/>
      <geom name="pen_c1" type="mesh" mesh="pen_b" density="900" {CONTACT}/>
      <geom name="pen_vis" type="mesh" mesh="pen_a" contype="0" conaffinity="0" group="1"/>
    </body>
    <body name="fixture_box" pos="-0.005 -0.775 0.0" quat="0.7071068 0 0 0.7071068">
      <freejoint name="fixture_box_joint0"/>
      <geom name="fixture_box_c0" type="mesh" mesh="floor" density="300" {CONTACT}/>
      <geom name="fixture_box_c1" type="mesh" mesh="rim" density="300" {CONTACT}/>
    </body>
    <body name="tape" pos="{tape_xy[0]} {tape_xy[1]} 0.0242">
      <freejoint name="tape_joint0"/>
      <geom name="tape_c0" type="mesh" mesh="tape" density="200" {CONTACT}/>
    </body>
  </worldbody>
</mujoco>"""


NAMES = ("pen", "fixture_box", "tape")


def _compiled(**kwargs):
    model = mujoco.MjModel.from_xml_string(_model_xml(**kwargs))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _geom(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _objects(model):
    out = {}
    for name in NAMES:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint0")
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        geoms = tuple(
            g for g in range(model.ngeom)
            if model.geom_bodyid[g] == body and (model.geom_contype[g] or model.geom_conaffinity[g])
        )
        out[name] = (int(model.jnt_qposadr[joint]), geoms)
    return out


def _lowest_z(model, data, name):
    from robosuite_env.tasks.envs.cup_plate_registered import _geom_world_vertices

    _, geoms = _objects(model)[name]
    return min(float(_geom_world_vertices(mujoco, model, data, g)[:, 2].min()) for g in geoms)


def _worst_penetration(model, data, geom_ids):
    geom_ids = set(geom_ids)
    depths = [
        -float(data.contact[i].dist)
        for i in range(data.ncon)
        if int(data.contact[i].geom1) in geom_ids or int(data.contact[i].geom2) in geom_ids
    ]
    return max(depths, default=0.0)


def _seat(model, data):
    from robosuite_env.tasks.envs.cup_plate_registered import seat_objects_on_table

    return seat_objects_on_table(
        mujoco,
        model,
        data,
        _objects(model),
        table_geom_ids=(_geom(model, "gw_desk_coll_0"),),
    )


def test_a_layout_object_buried_in_the_table_starts_resting_on_it():
    model, data = _compiled()
    pen_geoms = _objects(model)["pen"][1]
    assert _worst_penetration(model, data, pen_geoms) > 0.010  # the starting penetration
    poses_before = {name: data.qpos[q : q + 7].copy() for name, (q, _) in _objects(model).items()}

    report = _seat(model, data)
    mujoco.mj_forward(model, data)

    assert [row["object"] for row in report] == list(NAMES)
    assert _lowest_z(model, data, "pen") == pytest.approx(TABLE_TOP, abs=EXACT)
    assert _lowest_z(model, data, "fixture_box") == pytest.approx(TABLE_TOP, abs=EXACT)
    # Each object occupies a separate footprint and rests directly on the table.
    assert _lowest_z(model, data, "tape") == pytest.approx(TABLE_TOP, abs=EXACT)
    for name, (q, _) in _objects(model).items():
        np.testing.assert_array_equal(data.qpos[q : q + 2], poses_before[name][:2])
        np.testing.assert_array_equal(data.qpos[q + 3 : q + 7], poses_before[name][3:])

    all_geoms = [g for _, geoms in _objects(model).values() for g in geoms]
    assert _worst_penetration(model, data, all_geoms) < 1e-6
    start_z = {name: float(data.qpos[q + 2]) for name, (q, _) in _objects(model).items()}
    for _ in range(250):
        mujoco.mj_step(model, data)
    for name, (q, _) in _objects(model).items():
        assert abs(float(data.qpos[q + 2]) - start_z[name]) < 5e-4, name
    assert _worst_penetration(model, data, all_geoms) < 1e-3


def test_a_layout_object_off_the_table_is_rejected():
    model, data = _compiled(pen_xy=(3.0, 3.0))

    with pytest.raises(LayoutError, match="not over the table collision"):
        _seat(model, data)






class _PlacedObject:
    def __init__(self, name):
        self.name = name
        self.joints = [f"{name}_joint0"]
        self.root_body = name


def _fake_env(model, data):
    from robosuite.utils.placement_samplers import (
        SequentialCompositeSampler,
        UniformRandomSampler,
    )

    from robosuite_env.tasks.envs.placement import ExactObjectPoseSampler

    exact = ExactObjectPoseSampler(
        [(name, _PlacedObject(name), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)) for name in ("fixture_box", "tape")],
        reference_pos=(0.0, 0.0, 0.0),
    )
    # The pen stands in for a randomly sampled placement (cup/plate, other
    # parts): it must be left exactly where its sampler put it.
    dropped = UniformRandomSampler(
        name="DroppedSampler", mujoco_objects=[_PlacedObject("pen")],
        x_range=[0.0, 0.0], y_range=[0.0, 0.0],
    )
    composite = SequentialCompositeSampler(name="ObjectSampler")
    composite.append_sampler(exact)
    composite.append_sampler(dropped)
    return SimpleNamespace(
        placement_initializer=composite,
        table_rest_objects=(),
        sim=SimpleNamespace(
            model=SimpleNamespace(_model=model),
            data=SimpleNamespace(_data=data),
            forward=lambda: mujoco.mj_forward(model, data),
        ),
    )


def test_the_registered_env_seats_exact_placements_and_leaves_sampled_ones():
    from robosuite_env.tasks.envs.cup_plate_registered import CupPlateLiftGW

    model, data = _compiled()
    pen_q = _objects(model)["pen"][0]
    pen_z = float(data.qpos[pen_q + 2])
    env = _fake_env(model, data)

    CupPlateLiftGW._seat_layout_placements(env)

    assert [row["object"] for row in env.layout_seating_report] == ["fixture_box", "tape"]
    assert _lowest_z(model, data, "fixture_box") == pytest.approx(TABLE_TOP, abs=EXACT)
    assert _lowest_z(model, data, "tape") == pytest.approx(TABLE_TOP, abs=EXACT)
    assert float(data.qpos[pen_q + 2]) == pen_z


def test_a_sampler_placed_object_declared_to_rest_on_the_table_is_seated():
    # Cup/plate: XY from a sampler, z offsets that assumed a table at table_z.
    from robosuite_env.tasks.envs.cup_plate_registered import CupPlateLiftGW

    model, data = _compiled()
    assert _lowest_z(model, data, "pen") < TABLE_TOP - 0.010  # starts buried
    env = _fake_env(model, data)
    env.table_rest_objects = (_PlacedObject("pen"), None)

    CupPlateLiftGW._seat_layout_placements(env)

    rows = [row["object"] for row in env.layout_seating_report]
    assert sorted(rows) == ["fixture_box", "pen", "tape"]
    assert _lowest_z(model, data, "pen") == pytest.approx(TABLE_TOP, abs=EXACT)


def test_cup_plate_declares_both_objects_as_resting_on_the_table():
    import inspect

    from robosuite_env.tasks.envs.cup_plate_registered import CupPlateLiftGW

    source = inspect.getsource(CupPlateLiftGW._configure_task_objects)
    assert "self.table_rest_objects = (self.plate, self.cube)" in source
    assert CupPlateLiftGW.table_rest_objects == ()


@pytest.mark.parametrize(("deterministic", "expected"), [(False, ["sampled", "seated"]), (True, ["sampled"])])
def test_reset_seats_only_placements_it_sampled(monkeypatch, deterministic, expected):
    from robosuite_env.tasks.envs.cup_plate import CupPlateLift
    from robosuite_env.tasks.envs.cup_plate_registered import CupPlateLiftGW

    calls = []
    monkeypatch.setattr(CupPlateLift, "_reset_internal", lambda self: calls.append("sampled"))
    monkeypatch.setattr(CupPlateLiftGW, "_seat_layout_placements", lambda self: calls.append("seated"))
    env = object.__new__(CupPlateLiftGW)
    env.deterministic_reset = deterministic

    env._reset_internal()

    assert calls == expected


# --- a support object: real hull pairs from a CoACD table ---------------------------


def _real_pair(key):
    import base64
    import json
    import os

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
        "fixtures", "layout_seating_hull_pairs.json",
    )
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)["pairs"][key]

    def array(name):
        entry = record[name]
        raw = base64.b64decode(entry["base64"])
        return np.frombuffer(raw, dtype=entry["dtype"]).reshape(entry["shape"]).copy()

    scalars = {name: float(value) for name, value in record.items() if name.endswith("_m")}
    return (array("object_eq"), array("object_v")), (array("support_eq"), array("support_v")), scalars


@pytest.mark.parametrize("key", ["container_disjoint", "pen_disjoint"])
def test_real_pairs_highs_cannot_classify_have_no_overlap(key):
    # Real hulls: the calibrated support object's p_c6 against table part 16,
    # 62 mm apart, and the box layout's pen against a table part 0.3 mm away.
    # HiGHS's simplex stopped at model status kUnknown (15) on both infeasible
    # programs; seating must read that as no overlap, not raise it out of
    # env.reset().
    from robosuite_env.tasks.layouts import _max_vertical_overlap

    object_hull, table_hull, scalars = _real_pair(key)
    assert scalars["footprint_overlap_width_m"] < 0.0

    assert _max_vertical_overlap(object_hull, table_hull) is None
    assert seat_lift([object_hull], [table_hull]) is None


def test_a_real_overlapping_container_table_pair_keeps_its_exact_overlap():
    from robosuite_env.tasks.layouts import _max_vertical_overlap

    container_hull, table_hull, scalars = _real_pair("container_overlap")

    assert _max_vertical_overlap(container_hull, table_hull) == pytest.approx(
        scalars["max_vertical_overlap_m"], abs=EXACT
    )


def test_touching_footprints_share_no_support():
    from robosuite_env.tasks.layouts import _max_vertical_overlap

    block = _hull((0.0, 0.0, 0.0), (0.1, 0.1, 0.05))
    neighbour = _hull((0.1, 0.0, -0.05), (0.2, 0.1, 0.02))  # shares only x = 0.1

    assert _max_vertical_overlap(block, neighbour) is None


@pytest.mark.parametrize("status", [2, 4])
def test_a_solver_failure_on_overlapping_footprints_raises_instead_of_guessing(monkeypatch, status):
    # Footprints that share area make the program feasible, so neither
    # "infeasible" nor "unknown" may be read as "no overlap".
    import scipy.optimize

    from robosuite_env.tasks.layouts import _max_vertical_overlap

    methods = []

    def failing(*args, method=None, **kwargs):
        methods.append(method)
        return SimpleNamespace(status=status, message=f"forced ({method})", fun=None)

    monkeypatch.setattr(scipy.optimize, "linprog", failing)
    plate = _hull((0.0, 0.0, 0.0), (0.1, 0.1, 0.02))
    table = _hull((-0.5, -0.5, -0.05), (0.5, 0.5, 0.01))

    with pytest.raises(LayoutError, match="did not solve"):
        _max_vertical_overlap(plate, table)
    assert methods == ["highs", "highs-ipm", "highs-ds"]
