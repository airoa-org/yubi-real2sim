# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Which geoms the arms are allowed to pick up.

``GRASPABLE_GEOMS`` is every geom with a free joint somewhere up its body chain
-- the property the weld itself requires -- minus the arms.

It is not ``CUP_GEOMS``. That set is built by ``classify``, which calls a geom
"cup" only when its body or geom name contains "cup" or "cube". Every task
assigns its target to the ``env.cube`` *attribute*, but the compiled body keeps
the object's own name (``fixture_box_main``, ``tape_main``,
``distractor_main``), and ``classify`` reads the compiled name, so a name test
would miss the container, the parts, the box objects and the distractors.
"""

from __future__ import annotations

import math
import types

import pytest

mujoco = pytest.importorskip("mujoco")

from robosuite_env.rollout.build.geoms import initialize_geometry_sets
from tests._helpers.asset_trees import requires_repository_meshes


ARM_PREFIXES = {
    "R": {"body_prefixes": ("robot0_", "gripper0_")},
    "L": {"body_prefixes": ("robot1_", "gripper1_")},
}


def _model(xml):
    return mujoco.MjModel.from_xml_string(xml)


def _state(model):
    return types.SimpleNamespace(
        m=model, d=mujoco.MjData(model), ARM={k: dict(v) for k, v in ARM_PREFIXES.items()}
    )


SCENE = """
<mujoco>
  <worldbody>
    <body name="table"><geom name="table_g" type="box" size="1 1 .1"/></body>
    <body name="robot0_link1">
      <joint name="r0" type="hinge"/>
      <geom name="robot0_link1_g" type="box" size=".05 .05 .05"/>
      <body name="gripper0_finger_left">
        <geom name="gripper0_finger_left_pad_collision" type="box" size=".01 .01 .01"/>
      </body>
      <body name="gripper0_finger_right">
        <geom name="gripper0_finger_right_pad_collision" type="box" size=".01 .01 .01"/>
      </body>
    </body>
    <body name="robot1_link1">
      <joint name="r1" type="hinge"/>
      <geom name="robot1_link1_g" type="box" size=".05 .05 .05"/>
      <body name="gripper1_finger_left">
        <geom name="gripper1_finger_left_pad_collision" type="box" size=".01 .01 .01"/>
      </body>
      <body name="gripper1_finger_right">
        <geom name="gripper1_finger_right_pad_collision" type="box" size=".01 .01 .01"/>
      </body>
    </body>
    <body name="cube_main" pos="0 0 1">
      <freejoint/><geom name="cube_g" type="box" size=".02 .02 .02"/>
    </body>
    <body name="fixture_box_main" pos="0 .3 1">
      <freejoint/><geom name="fixture_box_g" type="box" size=".1 .06 .01"/>
      <body name="fixture_box_lip"><geom name="fixture_box_lip_g" type="box" size=".1 .01 .02"/></body>
    </body>
    <body name="tape_main" pos="0 .6 1">
      <freejoint/><geom name="tape_g" type="cylinder" size=".03 .01"/>
    </body>
    <body name="distractor_main" pos="0 .9 1">
      <freejoint/><geom name="distractor_g" type="box" size=".01 .01 .01"/>
    </body>
    <body name="bolted_fixture" pos="0 1.2 1">
      <geom name="bolted_fixture_g" type="box" size=".05 .05 .05"/>
    </body>
  </worldbody>
</mujoco>
"""


def _bodies(model, geoms):
    return sorted({model.body(int(model.geom_bodyid[g])).name for g in geoms})


def test_every_movable_object_is_graspable_whatever_it_is_called():
    model = _model(SCENE)
    state = _state(model)

    initialize_geometry_sets(state)

    assert _bodies(model, state.GRASPABLE_GEOMS) == [
        "cube_main",
        "distractor_main",
        "fixture_box_lip",
        "fixture_box_main",
        "tape_main",
    ]


def test_cup_geoms_is_not_the_graspable_set():
    """CUP_GEOMS is not a set of graspable objects."""

    model = _model(SCENE)
    state = _state(model)

    initialize_geometry_sets(state)

    assert _bodies(model, state.CUP_GEOMS) == ["cube_main"]
    missed = state.GRASPABLE_GEOMS - state.CUP_GEOMS
    assert _bodies(model, missed) == [
        "distractor_main",
        "fixture_box_lip",
        "fixture_box_main",
        "tape_main",
    ]


def test_the_arms_are_not_graspable_by_either_arm():
    model = _model(SCENE)
    state = _state(model)

    initialize_geometry_sets(state)

    assert not (state.GRASPABLE_GEOMS & state.ALLARM)


def test_a_body_bolted_to_the_world_is_not_graspable():
    """The weld refuses a body no free joint can move, so it is not a candidate."""

    model = _model(SCENE)
    state = _state(model)

    initialize_geometry_sets(state)

    assert "bolted_fixture" not in _bodies(model, state.GRASPABLE_GEOMS)
    assert "table" not in _bodies(model, state.GRASPABLE_GEOMS)


def test_a_child_of_a_free_body_inherits_its_graspability():
    """The container's lip has no joint of its own but rides the container's free joint."""

    model = _model(SCENE)
    state = _state(model)

    initialize_geometry_sets(state)

    lip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixture_box_lip_g")
    assert lip in state.GRASPABLE_GEOMS


def _pinch(state, model, object_geom, monkeypatch, dist=-0.0005, force=4.0,
           normals=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))):
    """Put both of the right gripper's inner pads on one object geom.

    ``MjData.contact`` cannot be assigned, so the contact list is swapped for
    a stand-in. The stand-in has to carry everything the selector reads, which
    is now more than the geoms: a contact frame, the friction the solver would
    have used, and a force. A contact pressing with no force is not a grip and
    is not counted, so stubbing the force to zero would make every pinch here
    read as a hand merely resting against the object.

    The default is a real pinch: the two pads face each other. Note the geom
    order differs between the two contacts on purpose -- the object is geom1 in
    one and geom2 in the other, exactly as MuJoCo emits them -- so this also
    covers the sign handling that normalises the two cases.
    """

    from robosuite_env.rollout.control import rule_grasp

    pads = sorted(state.INNER_PAD_SIDE_GEOMS["R"]["left"]) + sorted(
        state.INNER_PAD_SIDE_GEOMS["R"]["right"]
    )
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, object_geom)
    friction = (1.0, 1.0, 0.005, 1e-4, 1e-4)

    def _frame(normal):
        return tuple(normal) + (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    contacts = [
        types.SimpleNamespace(geom1=gid, geom2=pads[0], dist=dist,
                              frame=_frame(normals[0]), friction=friction),
        types.SimpleNamespace(geom1=pads[-1], geom2=gid, dist=dist,
                              frame=_frame(normals[1]), friction=friction),
    ]
    state.d = types.SimpleNamespace(ncon=len(contacts), contact=contacts)
    state.A = types.SimpleNamespace(grasp_contact_tol_mm=3.0)

    def _force(m, d, i, out):
        out[0] = force

    monkeypatch.setattr(rule_grasp.mujoco, "mj_contactForce", _force)
    return contacts


@pytest.mark.parametrize(
    "geom,body",
    [
        ("cube_g", "cube_main"),
        ("fixture_box_g", "fixture_box_main"),
        ("tape_g", "tape_main"),
        ("distractor_g", "distractor_main"),
    ],
)
def test_the_contact_selector_accepts_every_movable_object(geom, body, monkeypatch):
    """The behavioural half: what rule grasp actually reads each step.

    A rule_grasp.py that read CUP_GEOMS would fail this for every object
    except the one named "cube", which the set-shape tests above cannot catch
    on their own.
    """

    from robosuite_env.rollout.control import rule_grasp

    model = _model(SCENE)
    state = _state(model)
    initialize_geometry_sets(state)
    _pinch(state, model, geom, monkeypatch)

    sides = rule_grasp._rule_grasp_contact_sides(state, "R")

    assert sides["both"] is True, f"{body} was not seen as pinched"
    assert sides["same_object"] is True
    assert sides["object_body"] == body
    assert sides["opposed"] is True, f"{body} pinched squarely and was refused"
    assert sides["normal_dot"] == pytest.approx(-1.0)


def test_two_pads_on_one_face_are_not_a_grasp(monkeypatch):
    """Both pads on one face of the tape are touching, not holding.

    "Both pads, same object" is not enough: jaws closing on a single face
    press into a surface that cannot push back. The normals have to oppose.
    """

    from robosuite_env.rollout.control import rule_grasp

    model = _model(SCENE)
    state = _state(model)
    initialize_geometry_sets(state)
    # Both normals end up pointing the same way into the object. The raw signs
    # differ because the geom order differs between the two contacts.
    _pinch(state, model, "tape_g", monkeypatch,
           normals=((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))

    sides = rule_grasp._rule_grasp_contact_sides(state, "R")

    assert sides["both"] is True, "the test needs both pads touching"
    assert sides["normal_dot"] == pytest.approx(1.0)
    assert sides["opposed"] is False


def test_pads_resting_without_pressing_are_not_a_grasp(monkeypatch):
    """No force, no grip -- and no vote on which way the normals point.

    Each contact is weighted by its normal force with no floor under the
    weight; a floor would turn a hand with its pads either side of an object it
    has not yet squeezed into an unweighted average of directions with a
    respectable magnitude, read as a firm pinch. Waiting a solver tick for real
    force costs nothing.
    """

    from robosuite_env.rollout.control import rule_grasp

    model = _model(SCENE)
    state = _state(model)
    initialize_geometry_sets(state)
    _pinch(state, model, "tape_g", monkeypatch, force=0.0)

    sides = rule_grasp._rule_grasp_contact_sides(state, "R")

    assert sides["both"] is True
    assert sides["pads_pressing"] is False
    assert sides["opposed"] is False


def test_a_contact_beyond_the_tolerance_is_not_a_pinch(monkeypatch):
    """The tolerance still applies; widening the candidate set did not drop it."""

    from robosuite_env.rollout.control import rule_grasp

    model = _model(SCENE)
    state = _state(model)
    initialize_geometry_sets(state)
    _pinch(state, model, "fixture_box_g", monkeypatch, dist=0.010)  # 10 mm, tol is 3 mm

    sides = rule_grasp._rule_grasp_contact_sides(state, "R")

    assert sides["both"] is False


def test_the_other_arms_pads_do_not_count_as_this_arms_grip(monkeypatch):
    from robosuite_env.rollout.control import rule_grasp

    model = _model(SCENE)
    state = _state(model)
    initialize_geometry_sets(state)
    _pinch(state, model, "fixture_box_g", monkeypatch)

    sides = rule_grasp._rule_grasp_contact_sides(state, "L")

    assert sides["both"] is False, "arm L must not be credited with arm R's pads"


@pytest.mark.assets
@requires_repository_meshes("fr3", "yubi_right", "cup", "tape")
def test_the_compiled_duo_rig_names_match_what_this_file_assumes():
    """Pin the real compiled names, so the assumed names cannot drift from them."""

    pytest.importorskip("robosuite")
    from robosuite.models.arenas import EmptyArena
    from robosuite.models.grippers import gripper_factory
    from robosuite.models.objects import MujocoXMLObject
    from robosuite.models.tasks import ManipulationTask

    from robosuite_env.robots.fr3.model import FR3

    robots = []
    for idn in (0, 1):
        robot = FR3(idn=idn)
        robot.add_gripper(gripper_factory("YubiGripper", idn=idn))
        robots.append(robot)
    objects = [
        MujocoXMLObject(
            f"robosuite_env/assets/{asset}.xml",
            name=name,
            joints=[dict(type="free")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        for name, asset in (("cube", "cup"), ("tape", "tape"))
    ]
    model = ManipulationTask(
        mujoco_arena=EmptyArena(), mujoco_robots=robots, mujoco_objects=objects
    ).get_model(mode="mujoco")
    state = _state(model)

    initialize_geometry_sets(state)

    assert _bodies(model, state.CUP_GEOMS) == ["cube_main"]
    assert _bodies(model, state.GRASPABLE_GEOMS) == [
        "cube_main",
        "tape_main",
    ]
    for key in ("R", "L"):
        assert state.INNER_PAD_SIDE_GEOMS[key]["left"], f"{key} lost its left pad geoms"
        assert state.INNER_PAD_SIDE_GEOMS[key]["right"], f"{key} lost its right pad geoms"
    assert not (state.GRASPABLE_GEOMS & state.ALLARM)


def test_the_opposition_bound_follows_the_pads_friction():
    """The line is derived, not chosen -- and from the cone the solver uses.

    Two contacts can produce equal and opposite forces only if their friction
    cones between them span opposite directions, which puts the boundary at
    ``-cos(atan(mu_L) + atan(mu_R))``. But MuJoCo's default cone is pyramidal,
    and a pyramid only holds ``mu/sqrt(2)`` along its diagonals: taking the raw
    mu would trust a grip the solver cannot balance. At the mu=1 these profiles
    set, the honest coefficient is 0.707 and the bound lands near -1/3.
    """

    from robosuite_env.rollout.control.rule_grasp import (
        RULE_GRASP_MAX_NORMAL_DOT,
        RULE_GRASP_NORMAL_DOT_MARGIN,
        _opposition_bound,
    )

    margin = RULE_GRASP_NORMAL_DOT_MARGIN

    def bound(mu, **kw):
        return _opposition_bound({"left": mu, "right": mu}, **kw)

    def analytic(mu):
        return -math.cos(2.0 * math.atan(mu)) - margin

    # A circular cone is the textbook case and the code still matches it.
    assert bound(1.0, pyramidal=False) == pytest.approx(-margin)
    assert bound(0.5, pyramidal=False) == pytest.approx(analytic(0.5))

    # The pyramid is what the models actually compile with, and it is stricter.
    assert bound(1.0) == pytest.approx(analytic(1.0 / math.sqrt(2.0)))
    assert bound(1.0) < bound(1.0, pyramidal=False)

    # Slippery pads need a squarer grip than sticky ones.
    assert bound(0.2) < bound(1.0)
    # Very sticky pads do not buy their way past "pointing the same way".
    assert bound(5.0) == pytest.approx(RULE_GRASP_MAX_NORMAL_DOT - margin)
    # Nothing measured: fall back to the ceiling rather than accepting anything.
    assert _opposition_bound({}) == pytest.approx(RULE_GRASP_MAX_NORMAL_DOT - margin)


def test_the_cone_type_is_read_from_the_model_not_assumed():
    """mjCONE_PYRAMIDAL is 0 and is MuJoCo's default; elliptic is opt-in."""

    from types import SimpleNamespace

    from robosuite_env.rollout.control.rule_grasp import _cone_is_pyramidal

    assert _cone_is_pyramidal(SimpleNamespace(opt=SimpleNamespace(
        cone=int(mujoco.mjtCone.mjCONE_PYRAMIDAL)))) is True
    assert _cone_is_pyramidal(SimpleNamespace(opt=SimpleNamespace(
        cone=int(mujoco.mjtCone.mjCONE_ELLIPTIC)))) is False
    # A model built fresh has not opted out.
    model = _model(SCENE)
    assert _cone_is_pyramidal(model) is True
