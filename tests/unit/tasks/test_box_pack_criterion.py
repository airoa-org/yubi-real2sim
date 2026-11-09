# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The box-pack success criterion.

It encodes a rule that is easy to state and easy to get subtly wrong, so the
cases here are the ones the operator's wording turns on: the hand that moved
the object is the hand the prompt named; the box leaving the table ends the
episode's chances for good.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from robosuite_env.tasks.success import (
    BoxPackCriterion,
    make_success_criterion,
)
from tests._helpers.asset_trees import requires_repository_meshes


class _FakeModel:
    """Just enough MjModel for the two criteria: bodies, geoms and meshes."""

    def __init__(
        self,
        bodies,
        geom_names,
        half_extent_xy,
        top_z,
        parents=None,
        geom_specs=None,
    ):
        self._bodies = list(bodies)
        self.nbody = len(self._bodies)
        # Body 0 stands in for the world, so a parent of 0 ends the walk.
        self.body_parentid = np.asarray(
            parents if parents is not None else [0] * self.nbody, dtype=int
        )
        if geom_specs is None:
            self._geom_names = list(geom_names)
            self.ngeom = len(self._geom_names)
            # One geom per body, in body order, all group 0 and all sharing a
            # mesh whose footprint is the caller's half-extent.
            self.geom_bodyid = np.arange(self.ngeom)
            self.geom_dataid = np.zeros(self.ngeom, dtype=int)
            self.mesh_vertadr = np.array([0])
            self.mesh_vertnum = np.array([4])
            hx, hy = half_extent_xy
            self.mesh_vert = np.array(
                [
                    [-hx, -hy, 0.0],
                    [hx, -hy, 0.0],
                    [hx, hy, top_z],
                    [-hx, hy, top_z],
                ],
                dtype=float,
            )
        else:
            # A box needs separate floor and wall hulls.  Every spec is
            # (geom name, owning body name, low corner, high corner).
            self._geom_names = [spec[0] for spec in geom_specs]
            self.ngeom = len(self._geom_names)
            body_ids = {name: index for index, name in enumerate(self._bodies)}
            self.geom_bodyid = np.asarray(
                [body_ids[spec[1]] for spec in geom_specs], dtype=int
            )
            self.geom_dataid = np.arange(self.ngeom, dtype=int)
            self.mesh_vertadr = np.arange(self.ngeom, dtype=int) * 8
            self.mesh_vertnum = np.full(self.ngeom, 8, dtype=int)
            vertices = []
            for _, _, low, high in geom_specs:
                x0, y0, z0 = low
                x1, y1, z1 = high
                vertices.extend(
                    (x, y, z)
                    for x in (x0, x1)
                    for y in (y0, y1)
                    for z in (z0, z1)
                )
            self.mesh_vert = np.asarray(vertices, dtype=float)

        self.geom_group = np.zeros(self.ngeom, dtype=int)
        # Each geom sits at its body origin with identity rotation, so the
        # placed hull extents equal the mesh extents.
        self.geom_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (self.ngeom, 1))
        self.geom_pos = np.zeros((self.ngeom, 3))

    def body(self, index):
        return types.SimpleNamespace(name=self._bodies[index])

    def geom(self, index):
        return types.SimpleNamespace(name=self._geom_names[index])


class _FakeData:
    def __init__(self, positions):
        self._names = list(positions)
        self.xpos = np.array([positions[n] for n in self._names], dtype=float)
        self.xmat = np.tile(np.eye(3).ravel(), (len(self._names), 1))
        self.ncon = 0
        self.contact = []

    def move(self, model, name, position):
        for index in range(model.nbody):
            if model.body(index).name == name:
                self.xpos[index] = np.asarray(position, dtype=float)
                return
        raise KeyError(name)


# The compiled Duo rig is two single-arm FR3s, so the arm index -- not the
# words "left" and "right" -- is what separates the grippers. Every YUBI
# gripper carries a body named finger_left and one named finger_right, and its
# subtree root is gripper<N>_right_gripper. These are the real names, dumped
# from a compiled two-FR3 model, so the tests can catch one gripper being
# counted as two.
GRIPPER_INDEX_BY_SIDE = {"right": 0, "left": 1}


def _gripper_bodies():
    """(body names, parent indices) for both grippers, after the objects."""

    names, parents = [], []
    for index in GRIPPER_INDEX_BY_SIDE.values():
        root = len(names)
        names.append(f"gripper{index}_right_gripper")
        parents.append(0)
        for finger in ("finger_left", "finger_right"):
            names.append(f"gripper{index}_{finger}")
            parents.append(root)
    return names, parents


def _finger_geoms(model, side):
    index = GRIPPER_INDEX_BY_SIDE[side]
    return [
        model._geom_names.index(f"gripper{index}_{finger}_g")
        for finger in ("finger_left", "finger_right")
    ]


def _box_world():
    hands, hand_parents = _gripper_bodies()
    objects = ["world", "box", "tape", "pen", "lan"]
    bodies = [*objects, *hands]
    parents = [0] * len(objects) + [
        parent + len(objects) if parent else 0 for parent in hand_parents
    ]
    ordinary_hull = ((-0.01, -0.01, 0.0), (0.01, 0.01, 0.02))
    geom_specs = [
        (f"{body}_g", body, *ordinary_hull) for body in bodies if body != "box"
    ]
    # An intentionally off-centre cavity, matching the real box's topology.
    # The floor ends below the configurable 50 mm cut; every wall begins at
    # the floor and reaches the same derived rim height.
    geom_specs.extend(
        [
            ("box_floor_g", "box", (-0.13, -0.15, 0.0), (0.13, 0.02, 0.026)),
            (
                "box_left_wall_g",
                "box",
                (-0.13, -0.15, 0.02),
                (-0.12, 0.02, 0.143),
            ),
            (
                "box_right_wall_g",
                "box",
                (0.12, -0.15, 0.02),
                (0.13, 0.02, 0.143),
            ),
            (
                "box_back_wall_g",
                "box",
                (-0.13, -0.15, 0.02),
                (0.13, -0.14, 0.143),
            ),
            (
                "box_front_wall_g",
                "box",
                (-0.13, 0.01, 0.02),
                (0.13, 0.02, 0.143),
            ),
            # This folded-back lid is taller than the walls but begins at the
            # rim, so it must be classified as a flap, not cavity height.
            (
                "box_flap_g",
                "box",
                (-0.13, 0.02, 0.143),
                (0.13, 0.04, 0.295),
            ),
        ]
    )
    model = _FakeModel(
        bodies,
        [],
        half_extent_xy=(0.105, 0.0575),
        top_z=0.23,
        parents=parents,
        geom_specs=geom_specs,
    )
    positions = {
        "world": (0.0, 0.0, 0.0),
        "box": (0.0, -0.82, 0.0),
        "tape": (-0.005, -0.545, 0.0),
        "pen": (0.135, -0.53, 0.0),
        "lan": (-0.175, -0.535, 0.0),
    }
    for name in hands:
        side = -0.2 if name.startswith("gripper1_") else 0.2
        positions[name] = (side, -0.6, 0.3)
    return model, _FakeData(positions)


def _grip(model, data, side, object_geom):
    """Close the named gripper's fingers on an object."""
    b = model._geom_names.index(object_geom)
    data.contact = [
        types.SimpleNamespace(geom1=b, geom2=geom)
        for geom in _finger_geoms(model, side)
    ]
    data.ncon = len(data.contact)


def _rest_on_box_floor(model, data, object_geom):
    """Contact between the object and the box, i.e. it is down on the floor.

    The fake box has separate floor and wall hulls, like the real asset.
    """
    a = model._geom_names.index("box_floor_g")
    b = model._geom_names.index(object_geom)
    data.contact = [types.SimpleNamespace(geom1=b, geom2=a)]
    data.ncon = 1


def _box_criterion(model, data, **options):
    return BoxPackCriterion(
        mujoco=None,
        model=model,
        data=data,
        options={
            "table_z_m": 0.0,
            "box_pack_object": "tape",
            "box_pack_hand": "right",
            "box_pack_direction": "into_box",
            **options,
        },
    )


def test_packing_with_the_named_hand_succeeds():
    model, data = _box_world()
    criterion = _box_criterion(model, data)

    # Lifted and held by the right gripper.
    data.move(model, "tape", (0.0, -0.7, 0.20))
    _grip(model, data, "right", "tape_g")
    criterion.evaluate()
    # Set down on the box floor.
    data.move(model, "tape", (0.0, -0.82, 0.05))
    _rest_on_box_floor(model, data, "tape_g")
    result = criterion.evaluate()

    assert result["conditions"]["object_on_box_floor"] is True
    assert result["conditions"]["object_in_box"] is True
    assert result["conditions"]["moved_by_named_hand"] is True
    assert result["success"] is True
    assert result["strict_success"] is True


def test_packing_with_the_other_hand_is_success_but_not_strict_success():
    model, data = _box_world()
    criterion = _box_criterion(model, data)

    data.move(model, "tape", (0.0, -0.7, 0.20))
    _grip(model, data, "left", "tape_g")
    criterion.evaluate()
    data.contact, data.ncon = [], 0
    data.move(model, "tape", (0.0, -0.82, 0.05))
    result = criterion.evaluate()

    assert result["conditions"]["object_in_box"] is True
    assert result["conditions"]["object_on_box_floor"] is False
    assert result["conditions"]["moved_by_named_hand"] is False
    assert result["conditions"]["moved_by_other_hand"] is True
    assert result["success"] is True
    assert result["strict_success"] is False


def test_an_object_above_the_wall_rim_is_not_in_the_cavity():
    model, data = _box_world()
    criterion = _box_criterion(model, data)

    # The flap reaches 295 mm, but the enclosing walls end at 143 mm.
    data.move(model, "tape", (0.0, -0.885, 0.16))
    result = criterion.evaluate()

    assert result["conditions"]["object_in_box"] is False
    assert result["success"] is False


def test_an_object_beside_a_wall_is_not_in_the_cavity():
    model, data = _box_world()
    criterion = _box_criterion(model, data)

    # Local x=123 mm is inside the box's outer AABB (130 mm, less 5 mm margin)
    # but embedded beside the 120 mm inner face of the right wall.
    data.move(model, "tape", (0.123, -0.885, 0.05))
    result = criterion.evaluate()

    assert result["conditions"]["object_in_box"] is False
    assert result["success"] is False


def test_the_cavity_bounds_and_floor_contact_are_reported_separately():
    model, data = _box_world()
    criterion = _box_criterion(model, data)

    described = criterion.describe()["box_cavity_local_bounds_m"]
    assert described["min"] == pytest.approx([-0.12, -0.14, 0.026])
    assert described["max"] == pytest.approx([0.12, 0.01, 0.143])

    data.move(model, "tape", (0.0, -0.885, 0.05))
    no_contact = criterion.evaluate()
    assert no_contact["success"] is True
    assert no_contact["conditions"]["object_on_box_floor"] is False
    assert no_contact["strict_success"] is False

    _grip(model, data, "right", "tape_g")
    held_without_floor_contact = criterion.evaluate()
    assert held_without_floor_contact["conditions"]["moved_by_named_hand"] is True
    assert held_without_floor_contact["strict_success"] is False

    _rest_on_box_floor(model, data, "tape_g")
    on_floor = criterion.evaluate()
    assert on_floor["conditions"]["object_on_box_floor"] is True
    assert on_floor["strict_success"] is True


def test_the_cavity_moves_with_a_translated_box():
    model, data = _box_world()
    criterion = _box_criterion(model, data)

    box_position = np.array([0.30, -0.55, 0.0])
    local_inside = np.array([0.05, -0.06, 0.05])
    data.move(model, "box", box_position)
    data.move(model, "tape", box_position + local_inside)

    result = criterion.evaluate()

    assert result["conditions"]["object_in_box"] is True
    assert result["metrics"]["in_box_frame_mm"] == pytest.approx(
        [50.0, -60.0, 50.0]
    )
    # The box's own pose is reported in the world, so a pushed box can be
    # followed without inverting the object's box-frame coordinates.
    assert result["metrics"]["box_world_mm"] == pytest.approx([300.0, -550.0, 0.0])
    assert result["metrics"]["box_yaw_deg"] == pytest.approx(0.0)
    assert result["metrics"]["box_tilt_deg"] == pytest.approx(0.0)


def test_the_cavity_moves_with_a_rotated_box():
    model, data = _box_world()
    criterion = _box_criterion(model, data)

    box_id = model._bodies.index("box")
    box_position = np.asarray(data.xpos[box_id], dtype=float)
    quarter_turn = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    local_inside = np.array([0.10, -0.12, 0.05])
    data.xmat[box_id] = quarter_turn.ravel()
    data.move(model, "tape", box_position + quarter_turn @ local_inside)

    result = criterion.evaluate()

    assert result["conditions"]["object_in_box"] is True
    assert result["metrics"]["in_box_frame_mm"] == pytest.approx(
        [100.0, -120.0, 50.0]
    )
    # A quarter turn about z shows up as yaw alone; the box still stands.
    assert result["metrics"]["box_yaw_deg"] == pytest.approx(90.0)
    assert result["metrics"]["box_tilt_deg"] == pytest.approx(0.0)


def test_a_tipping_box_reports_its_tilt():
    # The failure this is for: a box pushed over the table edge tips before it
    # falls, and tilt is what rises first. Yaw alone would read a tipping box
    # as standing still.
    model, data = _box_world()
    criterion = _box_criterion(model, data)

    box_id = model._bodies.index("box")
    angle = math.radians(30.0)
    tipped_about_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)],
        ]
    )
    data.xmat[box_id] = tipped_about_x.ravel()

    metrics = criterion.evaluate()["metrics"]

    assert metrics["box_tilt_deg"] == pytest.approx(30.0)
    assert metrics["box_yaw_deg"] == pytest.approx(0.0)


def test_a_contact_while_the_object_still_rests_on_the_table_is_not_a_grip():
    model, data = _box_world()
    criterion = _box_criterion(model, data)

    # Touching, but the tape has not left the surface.
    _grip(model, data, "right", "tape_g")
    criterion.evaluate()

    assert criterion._gripped_by["right"] is False


def test_knocking_the_box_off_is_recorded_but_does_not_decide_criterion_e():
    # Criterion (e): box conditions are recorded but do not gate the verdict.
    # The box latch is recorded, and so is the verdict that includes it.
    model, data = _box_world()
    criterion = _box_criterion(model, data)
    data.move(model, "tape", (0.0, -0.7, 0.20))
    _grip(model, data, "right", "tape_g")
    criterion.evaluate()
    data.contact, data.ncon = [], 0

    data.move(model, "box", (0.0, -0.95, -0.40))
    assert criterion.evaluate()["conditions"]["box_on_table"] is False

    data.move(model, "box", (0.0, -0.82, 0.0))
    data.move(model, "tape", (0.0, -0.82, 0.05))
    recovered = criterion.evaluate()

    assert recovered["conditions"]["object_in_box"] is True
    assert recovered["conditions"]["box_on_table"] is False
    assert recovered["conditions"]["contained_and_box_on_table"] is False
    assert recovered["success"] is True


def test_an_object_still_held_inside_the_box_is_not_yet_a_success():
    model, data = _box_world()
    criterion = _box_criterion(model, data)
    data.move(model, "tape", (0.0, -0.82, 0.05))
    _grip(model, data, "right", "tape_g")
    held = criterion.evaluate()
    assert held["conditions"]["object_in_box"] is True
    assert held["conditions"]["object_held"] is True
    assert held["success"] is False


def test_the_first_release_inside_the_box_is_latched():
    model, data = _box_world()
    criterion = _box_criterion(model, data)
    data.move(model, "tape", (0.0, -0.82, 0.05))
    _grip(model, data, "right", "tape_g")
    criterion.evaluate()
    data.contact, data.ncon = [], 0
    released = criterion.evaluate()
    assert released["success"] is True
    assert released["metrics"]["first_release_in_box_output_frame"] == 1

    # Later frames are not looked at: taking it out again does not undo it.
    data.move(model, "tape", (0.0, -0.4, 0.02))
    later = criterion.evaluate()
    assert later["conditions"]["object_in_box"] is False
    assert later["success"] is True


def test_the_out_of_box_direction_wants_the_object_on_the_table():
    model, data = _box_world()
    criterion = _box_criterion(model, data, box_pack_direction="out_of_box")

    data.move(model, "tape", (0.0, -0.82, 0.05))
    _grip(model, data, "right", "tape_g")
    inside = criterion.evaluate()
    assert inside["success"] is False, "still in the box"

    data.contact, data.ncon = [], 0
    data.move(model, "tape", (-0.30, -0.60, 0.0))
    outside = criterion.evaluate()

    assert outside["conditions"]["object_on_table"] is True
    assert outside["success"] is True


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"box_pack_hand": "both"}, "hand must be one of"),
        ({"box_pack_direction": "sideways"}, "direction must be one of"),
    ],
)
def test_an_unreadable_prompt_binding_is_refused(options, message):
    model, data = _box_world()
    with pytest.raises(ValueError, match=message):
        _box_criterion(model, data, **options)


@pytest.mark.parametrize(
    ("hand", "direction"),
    [("R", "in"), ("right", "into_box"), ("L", "out"), ("left", "out_of_box")],
)
def test_the_contract_spelling_and_the_english_one_both_bind(hand, direction):
    """box_pack.py hands out "R"/"in"; a config may write them out in full."""

    model, data = _box_world()
    criterion = _box_criterion(
        model, data, box_pack_hand=hand, box_pack_direction=direction
    )

    assert criterion.hand == ("right" if hand.lower() in ("r", "right") else "left")
    assert criterion.direction == (
        "into_box" if direction in ("in", "into_box") else "out_of_box"
    )


@pytest.mark.assets
@requires_repository_meshes("box")
def test_real_box_collision_hulls_derive_the_measured_cavity(mujoco, repo_root):
    from robosuite_env.tasks.success.box_pack import _derive_box_cavity

    model = mujoco.MjModel.from_xml_path(
        str(repo_root / "robosuite_env" / "assets" / "box.xml")
    )
    body_id = model.body("object").id

    bounds, floor_geoms = _derive_box_cavity(
        model, body_id, floor_cut_m=0.05
    )

    # The box is scaled in z only, to 11/14 of its captured height (232.0 mm).
    # The cavity keeps its plan and only loses depth: floor 20.2 mm, wall top
    # 112.6 mm, depth 92.4 mm. A 2.5 mm sampling sweep of the unscaled box
    # found x -117.5..117.5 and y -127.5..10.0 at z=35 mm; the plan is
    # unchanged by a z-only scale, and the analytic slice sits closer to the
    # floor, so allow one sampling cell plus the slight wall slope. These
    # figures are derived from the hulls the same way the criterion derives
    # them, so the numbers move with the asset rather than asserting a height
    # of their own.
    assert bounds[0] == pytest.approx([-0.1187, -0.1262, 0.0202], abs=0.004)
    assert bounds[1] == pytest.approx([0.1184, 0.0092, 0.1126], abs=0.004)
    assert len(floor_geoms) == 9
    # The flattening must not have touched the footprint.
    assert bounds[1, 0] - bounds[0, 0] == pytest.approx(0.2371, abs=0.003)
    assert bounds[1, 1] - bounds[0, 1] == pytest.approx(0.1353, abs=0.003)


def test_the_cavity_is_measured_just_above_the_floor():
    from robosuite_env.tasks.success.box_pack import _derive_box_cavity

    floor_top, wall_top, lean = 0.01, 0.11, 0.5
    half = {0: 0.10, 1: 0.06}  # inner half-width at the floor, per axis

    def cuboid(x0, x1, y0, y1, z0, z1):
        return [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]

    def wall(axis, sign):
        # Inner face leans inward by `lean` per unit height; outer face is vertical.
        verts = []
        for z in (floor_top, wall_top):
            inner = sign * (half[axis] - lean * (z - floor_top))
            for a in (inner, sign * 0.13):
                for b in (-0.13, 0.13):
                    verts.append((a, b, z) if axis == 0 else (b, a, z))
        return verts

    hulls = [cuboid(-0.13, 0.13, -0.13, 0.13, 0.0, floor_top)] + [
        wall(axis, sign) for axis in (0, 1) for sign in (-1, 1)
    ]
    n = len(hulls)
    model = types.SimpleNamespace(
        ngeom=n,
        geom_bodyid=np.ones(n, dtype=int),
        geom_group=np.zeros(n, dtype=int),
        geom_dataid=np.arange(n),
        mesh_vertadr=np.arange(n) * 8,
        mesh_vertnum=np.full(n, 8),
        mesh_vert=np.asarray(sum(hulls, []), dtype=float),
        geom_quat=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        geom_pos=np.zeros((n, 3)),
    )

    bounds, _ = _derive_box_cavity(model, 1, floor_cut_m=0.02)

    # The opening 1 % of the wall height above the floor, not at the floor
    # seam and not higher up the slope.
    slice_z = floor_top + 0.01 * (wall_top - floor_top)
    opening = {a: h - lean * (slice_z - floor_top) for a, h in half.items()}
    assert bounds[0] == pytest.approx([-opening[0], -opening[1], floor_top], abs=1e-9)
    assert bounds[1] == pytest.approx([opening[0], opening[1], wall_top], abs=1e-9)


def test_the_box_criterion_is_reachable_through_the_registry():
    model, data = _box_world()
    box = make_success_criterion(
        "object_in_box",
        mujoco=None,
        model=model,
        data=data,
        options={"table_z_m": 0.0, "box_pack_object": "pen"},
    )
    assert isinstance(box, BoxPackCriterion)
