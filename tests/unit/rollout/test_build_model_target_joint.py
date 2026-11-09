# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The rollout worker resolves the task target and receptacle from the objects.

The container and box-pack envs install their target under its logical name
(``fixture_box``, ``tape``, ``pen``, ``lan``), not a fixed MJCF name such as
``cube_joint0``. ``rollout/build/model.py`` resolves the target joint from the
object's name, and a missing name raises: ``mj_name2id`` returns -1 for it, and
numpy would read -1 as the model's *last* joint.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace

import mujoco
import pytest

from robosuite_env.rollout.build import model as build_model
from tests._helpers.worker_ast import BUILD_MODEL_TREE

_JOINT = mujoco.mjtObj.mjOBJ_JOINT


def _state(free_objects, *, target, plate=None, target_joints=None):
    """A compiled model with one arm hinge first and ``free_objects`` after it.

    Objects are compiled in the given order, the way robosuite appends task
    objects after the robots, so the last name owns the model's last joint.
    """

    bodies = "".join(
        f'<body name="{name}_main" pos="{0.2 * index} 0 0.1">'
        f'<freejoint name="{name}_joint0"/><geom size="0.02"/></body>'
        for index, name in enumerate(free_objects)
    )
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        '<body name="robot0_link1"><joint name="robot0_joint1"/><geom size="0.01"/></body>'
        f"{bodies}</worldbody></mujoco>"
    )
    joints = (f"{target}_joint0",) if target_joints is None else tuple(target_joints)
    env = SimpleNamespace(
        cube=None if target is None and target_joints is None else SimpleNamespace(joints=joints),
        plate=None if plate is None else SimpleNamespace(joints=(f"{plate}_joint0",)),
    )
    return SimpleNamespace(m=model, env=env, ROBOT="fr3")


def _address(state, joint_name):
    joint_id = mujoco.mj_name2id(state.m, _JOINT, joint_name)
    assert joint_id >= 0, joint_name
    return int(state.m.jnt_qposadr[joint_id]), int(state.m.jnt_dofadr[joint_id])


def test_container_target_resolves_to_the_container_not_the_models_last_joint():
    state = _state(
        ("fixture_box", "part_a", "part_b"), target="fixture_box"
    )
    # The premise: no cube_joint0, and -1 would index the last part.
    assert mujoco.mj_name2id(state.m, _JOINT, "cube_joint0") == -1
    last_joint_qpos = int(state.m.jnt_qposadr[-1])
    assert last_joint_qpos == _address(state, "part_b_joint0")[0]

    target, plate = build_model._primary_object_bindings(state)

    assert target["joint_name"] == "fixture_box_joint0"
    assert (target["qpos_address"], target["dof_address"]) == _address(state, "fixture_box_joint0")
    assert target["qpos_address"] != last_joint_qpos
    assert plate is None


@pytest.mark.parametrize("prompt_object", ("tape", "pen", "lan"))
def test_box_pack_target_resolves_to_the_prompts_object(prompt_object):
    # box_pack compiles (target, *the other three in registry order).
    others = tuple(name for name in ("box", "tape", "pen", "lan") if name != prompt_object)
    state = _state((prompt_object, *others), target=prompt_object)

    target, plate = build_model._primary_object_bindings(state)

    assert (target["qpos_address"], target["dof_address"]) == _address(
        state, f"{prompt_object}_joint0"
    )
    assert target["qpos_address"] != int(state.m.jnt_qposadr[-1])
    assert plate is None


@pytest.mark.parametrize(
    ("free_objects", "plate_name"),
    (
        (("cube", "plate"), "plate"),  # cup P&P
        (("cube", "distractor_two", "receptacle"), None),  # no plate
    ),
)
def test_cube_named_tasks_resolve_exactly_what_the_fixed_names_did(free_objects, plate_name):
    state = _state(free_objects, target="cube", plate=plate_name)

    target, plate = build_model._primary_object_bindings(state)

    assert (target["qpos_address"], target["dof_address"]) == _address(state, "cube_joint0")
    if plate_name is None:
        assert plate is None
    else:
        assert (plate["qpos_address"], plate["dof_address"]) == _address(
            state, f"{plate_name}_joint0"
        )


@pytest.mark.parametrize(
    ("target", "target_joints", "message"),
    (
        # A target whose joint the model does not have must never become -1.
        (None, ("cube_joint0",), "absent from the compiled model"),
        (None, (), "exactly one joint"),
        (None, ("fixture_box_joint0", "part_b_joint0"), "exactly one joint"),
        (None, ("robot0_joint1",), "is not free"),
    ),
)
def test_an_unresolvable_target_raises_instead_of_wrapping(target, target_joints, message):
    state = _state(
        ("fixture_box", "part_b"), target=target, target_joints=target_joints
    )

    with pytest.raises(RuntimeError, match=message):
        build_model._primary_object_bindings(state)


def test_a_task_env_without_a_target_raises():
    state = _state(("fixture_box", "part_b"), target="fixture_box")
    state.env = SimpleNamespace(plate=None)

    with pytest.raises(RuntimeError, match="env.cube"):
        build_model._primary_object_bindings(state)


def test_an_unresolvable_receptacle_raises_instead_of_silently_dropping():
    state = _state(("cube", "part_b"), target="cube", plate="plate")

    with pytest.raises(RuntimeError, match="absent from the compiled model"):
        build_model._primary_object_bindings(state)


_NAME_LOOKUPS = {"jid", "bid", "aid", "mj_name2id"}
# The lookup wrappers themselves, and the helpers that reject a negative id
# before it can reach an array index.
_GUARDED_LOOKUP_CALLERS = {"jid", "bid", "aid", "_free_task_object_binding"}


def _call_name(node):
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def test_build_model_looks_names_up_only_through_fail_closed_helpers():
    # A bare jid()/bid()/aid() result must not be used as an index:
    # a missing name returns -1 and numpy reads it as "the last one". So no
    # stage code calls a lookup directly; every lookup goes through
    # _required_id (passed as ``lookup``) or a helper that rejects -1.
    unguarded = []
    for function in BUILD_MODEL_TREE.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and _call_name(node) in _NAME_LOOKUPS
                and function.name not in _GUARDED_LOOKUP_CALLERS
            ):
                unguarded.append((function.name, node.lineno, ast.unparse(node)))
    assert unguarded == []


def test_run_resolves_the_target_and_receptacle_through_the_object_helper():
    run = next(
        node
        for node in BUILD_MODEL_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    calls = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Call) and _call_name(node) == "_primary_object_bindings"
    ]
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("objects", "env_values", "expected"),
    (
        ("cup_plate", {"plate": SimpleNamespace(name="plate")}, {"target": "cup", "receptacle": "plate"}),
        (
            "box_pack",
            {
                "box_pack_target_object": "pen",
                "box": SimpleNamespace(name="box"),
            },
            {"target": "pen", "receptacle": "box"},
        ),
    ),
)
def test_task_binding_names_the_resolved_task_objects(objects, env_values, expected):
    state = SimpleNamespace(
        A=SimpleNamespace(objects=objects),
        env=SimpleNamespace(**env_values),
    )

    assert build_model._task_binding(state) == expected


def test_task_binding_refuses_a_missing_task_receptacle():
    state = SimpleNamespace(
        A=SimpleNamespace(objects="box_pack"),
        env=SimpleNamespace(box_pack_target_object="pen", box=None),
    )

    with pytest.raises(RuntimeError, match="logical target/receptacle"):
        build_model._task_binding(state)
