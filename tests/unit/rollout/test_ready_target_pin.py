# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The ready drive's hold on the task target (rollout/stages/settle_and_ready.py).

Cup P&P and recorded keep the historical full-pose pin. Box-pack
holds only x, y and heading while the arms drive to the ready pose, and start
the policy from where the target came to rest: a full-pose pin re-imposes a
stored pose that is not a rest pose, which lifts seated objects off their
contact every tick or holds a dropped object in the air.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from robosuite_env.rollout.stages import settle_and_ready as sar
from robosuite_env.rollout.state.settle import (
    _restore_ready_objects_stationary,
    initialize_ready_object_registry,
)
from tests._helpers.worker_ast import PACKAGE_ROOT

_TASK_FLAGS = ("_BOX_PACK_TASK",)


def _flags(planar_flag=None):
    return {name: name == planar_flag for name in _TASK_FLAGS}


def _quat(axis, degrees):
    out = np.zeros(4)
    mujoco.mju_axisAngle2Quat(out, np.asarray(axis, float), np.radians(degrees))
    return out


def _mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def _rotation(quat):
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(quat, float))
    return out.reshape(3, 3)


def _tilt_deg(quat):
    return float(np.degrees(np.arccos(np.clip(_rotation(quat)[2, 2], -1.0, 1.0))))


@pytest.mark.parametrize(
    ("flags", "planar"),
    (
        (_flags(), False),  # cup P&P does not set it
        (_flags("_BOX_PACK_TASK"), True),
    ),
)
def test_only_box_pack_uses_the_planar_hold(flags, planar):
    assert sar._ready_pin_is_planar(SimpleNamespace(**flags)) is planar


def test_cup_pick_and_place_keeps_the_historical_pin_byte_for_byte():
    rng = np.random.default_rng(7)
    qpos = rng.normal(size=30)
    qvel = rng.normal(size=28)
    state = SimpleNamespace(
        d=SimpleNamespace(qpos=qpos.copy(), qvel=qvel.copy()),
        CUPa=np.int32(9), cdof=np.int32(8),
        cup=np.array([0.01, -0.62, 0.034]),
        _READY_PRIMARY_QUAT=np.array([0.0, 1.0, 0.0, 0.0]),
        **_flags(),
    )
    expected_qpos, expected_qvel = qpos.copy(), qvel.copy()
    # The line settle_and_ready.py ran before the planar hold existed.
    expected_qpos[9:12] = state.cup
    expected_qpos[12:16] = state._READY_PRIMARY_QUAT
    expected_qvel[8:14] = 0.0

    sar._pin_ready_target(state)

    assert state.d.qpos.tobytes() == expected_qpos.tobytes()
    assert state.d.qvel.tobytes() == expected_qvel.tobytes()


@pytest.mark.parametrize("planar_flag", _TASK_FLAGS)
def test_the_planar_hold_pins_x_y_and_heading_and_leaves_the_rest_to_contact(planar_flag):
    stored = np.concatenate([[-0.005, -0.775, 0.034], _quat([0, 0, 1], 90.0)])
    current_quat = _mul(_quat([0, 0, 1], 30.0), _quat([1, 0, 0], 10.0))
    qpos = np.zeros(20)
    qpos[5:12] = np.concatenate([[-0.002, -0.771, 0.0337], current_quat])
    qvel = np.zeros(18)
    omega_world = np.array([0.3, -0.2, 1.5])
    qvel[4:10] = np.concatenate([[0.01, -0.02, -0.03], _rotation(current_quat).T @ omega_world])
    untouched_qpos, untouched_qvel = qpos.copy(), qvel.copy()
    state = SimpleNamespace(
        d=SimpleNamespace(qpos=qpos, qvel=qvel), CUPa=5, cdof=4,
        _TASK_OBJECT_TARGET="cup", _READY_OBJECT_QPOS={"cup": (5, 4, stored)},
        **_flags(planar_flag),
    )

    sar._pin_ready_target(state)

    held = qpos[8:12]
    assert np.allclose(qpos[5:7], stored[:2], atol=0.0)
    assert qpos[7] == untouched_qpos[7]  # height follows contact
    assert np.degrees(sar._twist_about_world_z(held)) == pytest.approx(90.0, abs=1e-9)
    assert _tilt_deg(held) == pytest.approx(10.0, abs=1e-9)  # roll/pitch follow contact
    assert np.allclose(qvel[4:6], 0.0) and qvel[6] == untouched_qvel[6]
    # Roll/pitch rates survive (turning with the object); only spin about
    # world z is removed. World z seen from the body is the same before and
    # after a turn about world z.
    world_z_in_body = _rotation(current_quat)[2, :]
    old_body = untouched_qvel[7:10]
    expected = old_body - (world_z_in_body @ old_body) * world_z_in_body
    assert np.allclose(qvel[7:10], expected, atol=1e-12)
    assert (_rotation(held) @ qvel[7:10])[2] == pytest.approx(0.0, abs=1e-12)
    others = np.ones(20, bool)
    others[5:12] = False
    assert np.array_equal(qpos[others], untouched_qpos[others])


def test_the_planar_hold_keeps_heading_for_a_part_lying_on_its_side():
    # A body-axis heading is undefined when that axis points up; the twist is not.
    stored = np.concatenate([[0.1, -0.5, 0.05], _quat([0, 0, 1], -20.0)])
    on_side = _mul(_quat([0, 0, 1], 40.0), _quat([0, 1, 0], 90.0))
    qpos = np.concatenate([[0.1, -0.5, 0.02], on_side])
    qvel = np.zeros(6)

    sar._hold_planar_pose(qpos, qvel, 0, 0, stored)

    assert np.degrees(sar._twist_about_world_z(qpos[3:7])) == pytest.approx(-20.0, abs=1e-9)
    assert _tilt_deg(qpos[3:7]) == pytest.approx(90.0, abs=1e-9)


_PART_XML = (
    '<mujoco><option gravity="0 0 0"/><worldbody><body name="part">'
    '<freejoint/><geom type="box" size="0.03 0.02 0.01"/></body></worldbody></mujoco>'
)


def _world_angular_velocity(model, data):
    mujoco.mj_forward(model, data)
    out = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, out, 0)
    return out[:3]


def test_the_planar_hold_removes_world_z_spin_as_mujoco_defines_it():
    # Measured with MuJoCo's own world-frame body velocity, so a wrong guess
    # about the free joint's angular-velocity frame fails here.
    model = mujoco.MjModel.from_xml_string(_PART_XML)
    data = mujoco.MjData(model)
    tilted = _mul(_quat([0, 0, 1], 25.0), _quat(np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0), 60.0))
    data.qpos[:] = np.concatenate([[0.0, 0.0, 0.2], tilted])
    data.qvel[3:6] = [0.2, 2.0, 2.0]
    before = _world_angular_velocity(model, data)
    assert abs(before[2]) > 0.5  # the premise: a real world-z spin to remove
    stored = np.concatenate([[0.0, 0.0, 0.2], _quat([0, 0, 1], -40.0)])

    sar._hold_planar_pose(data.qpos, data.qvel, 0, 0, stored)
    after = _world_angular_velocity(model, data)

    assert after[2] == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(after[:2]) == pytest.approx(np.linalg.norm(before[:2]), abs=1e-12)


def test_a_tilted_part_spinning_about_world_z_is_held_still():
    model = mujoco.MjModel.from_xml_string(_PART_XML)
    data = mujoco.MjData(model)
    tilted = _mul(_quat([0, 0, 1], 25.0), _quat(np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0), 60.0))
    data.qpos[:] = np.concatenate([[0.0, 0.0, 0.2], tilted])
    data.qvel[3:6] = _rotation(tilted).T @ np.array([0.0, 0.0, 3.0])
    stored = data.qpos.copy()

    for _ in range(50):
        sar._hold_planar_pose(data.qpos, data.qvel, 0, 0, stored)
        mujoco.mj_step(model, data)
        drift = sar._twist_about_world_z(data.qpos[3:7]) - sar._twist_about_world_z(stored[3:7])
        assert abs(drift) < 1e-9
    assert _tilt_deg(data.qpos[3:7]) == pytest.approx(60.0, abs=1e-6)


def _drop_state(planar_flag):
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        '<geom name="table_collision" type="plane" size="1 1 0.1"/>'
        '<body name="part_main" pos="0.1 -0.2 0.12" quat="0.9238795 0 0 0.3826834">'
        '<freejoint name="part_joint0"/><geom type="box" size="0.03 0.02 0.01"/></body>'
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "part_joint0")
    state = SimpleNamespace(
        m=model, d=data, CUPa=model.jnt_qposadr[joint], cdof=model.jnt_dofadr[joint],
        _TASK_OBJECT_TARGET="cup",
        _TASK_OBJECT_BINDINGS={
            "cup": {"joint_id": joint, "qpos_address": int(model.jnt_qposadr[joint]),
                    "dof_address": int(model.jnt_dofadr[joint])},
        },
        **_flags(planar_flag),
    )
    initialize_ready_object_registry(state)
    return state


def _ready_drive_then_restore(state, steps=400):
    for _ in range(steps):
        sar._pin_ready_target(state)
        mujoco.mj_step(state.m, state.d)
    if sar._ready_pin_is_planar(state):
        sar._keep_ready_objects_where_they_landed(state)
    _restore_ready_objects_stationary(state)


def test_a_dropped_target_under_the_planar_hold_starts_the_policy_resting_on_its_support():
    state = _drop_state("_BOX_PACK_TASK")
    spawn = state.d.qpos[0:7].copy()

    _ready_drive_then_restore(state)

    assert state.d.qpos[2] == pytest.approx(0.01, abs=5e-4)  # on the table, not at 0.12 m
    assert np.array_equal(state.d.qpos[0:2], spawn[:2])
    assert sar._twist_about_world_z(state.d.qpos[3:7]) == pytest.approx(
        sar._twist_about_world_z(spawn[3:7]), abs=1e-9
    )
    assert np.all(state.d.qvel[0:6] == 0.0)
    mujoco.mj_step(state.m, state.d)
    assert abs(float(state.d.qpos[2]) - 0.01) < 5e-4  # and it stays there


def test_the_historical_pin_still_restores_the_stored_pose():
    # Cup P&P / recorded: unchanged, including the stored height.
    state = _drop_state(None)
    stored = state.d.qpos[0:7].copy()

    _ready_drive_then_restore(state)

    assert np.array_equal(state.d.qpos[0:7], stored)


def _settle_and_ready_run():
    path = PACKAGE_ROOT / "rollout" / "stages" / "settle_and_ready.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")


def _calls(node, name):
    return [
        call for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == name
    ]


def test_the_ready_drive_holds_the_target_only_through_the_pin_helper():
    run = _settle_and_ready_run()
    pins = _calls(run, "_pin_ready_target")
    assert len(pins) == 1
    loop = next(
        node for node in ast.walk(run)
        if isinstance(node, ast.For) and any(call in pins for call in ast.walk(node))
    )
    steps = [
        call for call in ast.walk(loop)
        if isinstance(call, ast.Call) and ast.unparse(call.func) == "state.env.step"
    ]
    assert steps and pins[0].lineno < steps[0].lineno
    # No second, inline write of the target's pose anywhere in the stage.
    target_writes = [
        node for node in ast.walk(run)
        if isinstance(node, ast.Assign)
        and any("state.CUPa" in ast.unparse(target) for target in node.targets)
    ]
    assert target_writes == []


def test_planar_tasks_restore_the_landed_pose_and_nothing_else_does():
    run = _settle_and_ready_run()
    keeps = _calls(run, "_keep_ready_objects_where_they_landed")
    restores = _calls(run, "_restore_ready_objects_stationary")
    assert len(keeps) == 1 and len(restores) == 1
    guard = next(
        node for node in ast.walk(run)
        if isinstance(node, ast.If)
        and any(isinstance(stmt, ast.Expr) and stmt.value is keeps[0] for stmt in node.body)
    )
    assert ast.unparse(guard.test) == "_ready_pin_is_planar(state)"
    assert keeps[0].lineno < restores[0].lineno
