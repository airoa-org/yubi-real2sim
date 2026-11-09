# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""What the arms hold while task objects settle (rollout/state/settle.py).

FR3 drives both arms to a validated ready pose (settle_and_ready.run) and
holds the last command of that trajectory while the task objects settle, then
re-checks the ready pose. The FR3 path must stay byte-identical to the
historical hold and gate it replaced.
"""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from robosuite_env.robots.profiles import get_robot_profile
from robosuite_env.rollout.control.fr3_strategy import make_fr3_controller_strategy
from robosuite_env.rollout.state import settle

# Two arms of two joints with a finger between them, so the right arm's joints
# (qpos 0, 1) and the left arm's (qpos 3, 4) are not one contiguous block.
_ACTUATED = ("r_j1", "r_j2", "r_finger", "l_j1", "l_j2", "l_finger")
_ARM_QI = {"R": np.array([0, 1]), "L": np.array([3, 4])}
_FINGER_QI = {"R": 2, "L": 5}
_HOME = {"R": np.array([0.31, -0.22]), "L": np.array([-0.43, 0.27])}

# A recorded profile's quiet thresholds, with shorter windows.
_SETTLE = dict(
    dynamic_object_settle_max_sec=2.0,
    dynamic_object_settle_min_sec=0.2,
    dynamic_object_settle_stable_sec=0.2,
    dynamic_object_settle_linear_speed_mm_s=1.0,
    dynamic_object_settle_angular_speed_rad_s=0.03,
    dynamic_object_settle_linear_acceleration_m_s2=0.1,
    dynamic_object_settle_angular_acceleration_rad_s2=1.0,
    dynamic_object_settle_position_drift_mm=0.25,
    dynamic_object_settle_orientation_drift_deg=0.1,
)


def _arm_xml(side, y):
    return (
        f'<body name="{side}_link1" pos="0 {y} 0.5">'
        f'<joint name="{side}_j1" axis="0 0 1"/>'
        '<geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.02"/>'
        f'<body name="{side}_link2" pos="0.1 0 0">'
        f'<joint name="{side}_j2" axis="0 1 0"/>'
        '<geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.02"/>'
        f'<body name="{side}_hand" pos="0.1 0 0">'
        f'<joint name="{side}_finger" axis="0 1 0" range="0 0.7854"/>'
        '<geom type="box" size="0.01 0.01 0.01" mass="0.05"/>'
        "</body></body></body>"
    )


def _model():
    actuators = "".join(
        f'<position name="{joint}_servo" joint="{joint}" kp="{5 if "finger" in joint else 400}"/>'
        for joint in _ACTUATED
    )
    return mujoco.MjModel.from_xml_string(
        "<mujoco>"
        '<option timestep="0.002" integrator="implicitfast"/>'
        '<default><joint damping="1" armature="0.01"/>'
        '<geom contype="0" conaffinity="0" mass="0.2"/></default>'
        "<worldbody>"
        '<geom name="table_collision" type="plane" size="1 1 0.1" contype="1" conaffinity="1"/>'
        + _arm_xml("r", -0.5)
        + _arm_xml("l", 0.5)
        + '<body name="cup" pos="0 0 0.02"><freejoint name="cup_joint"/>'
        '<geom type="box" size="0.03 0.03 0.02" mass="0.1" contype="1" conaffinity="1"/>'
        "</body></worldbody>"
        f"<actuator>{actuators}</actuator>"
        "</mujoco>"
    )


class _FR3Robot:
    """One of FR3's two robots: one arm and its hand."""

    def __init__(self):
        self.parts = []

    def create_action_vector(self, parts):
        self.parts.append({k: np.array(v, dtype=np.float64) for k, v in parts.items()})
        return np.concatenate([parts["right"], parts["right_gripper"]])


class _Env:
    """Position servos on both arms, stepped at the 50 Hz control tick."""

    control_freq = 50
    action_dim = 6

    def __init__(self, model, data, robots):
        self.m, self.d, self.robots = model, data, robots
        self.actions = []

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.actions.append(action.copy())
        self.d.ctrl[[0, 1]] = action[[0, 1]]
        self.d.ctrl[2] = (action[2] + 1.0) * 0.5 * 0.7854
        self.d.ctrl[[3, 4]] = action[[3, 4]]
        self.d.ctrl[5] = (action[5] + 1.0) * 0.5 * 0.7854
        for _ in range(10):
            mujoco.mj_step(self.m, self.d)


def _state(embodiment, *, robots, ready_segments, ready_gate):
    model = _model()
    data = mujoco.MjData(model)
    for key in ("R", "L"):
        data.qpos[_ARM_QI[key]] = _HOME[key]
        data.ctrl[_ARM_QI[key]] = _HOME[key]
    mujoco.mj_forward(model, data)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cup_joint")
    state = SimpleNamespace(
        m=model,
        d=data,
        EMBODIMENT=embodiment,
        env=_Env(model, data, robots),
        A=SimpleNamespace(**_SETTLE),
        ARM={
            key: {"qi": _ARM_QI[key], "fj": [_FINGER_QI[key], _FINGER_QI[key]]}
            for key in ("R", "L")
        },
        CUPa=int(model.jnt_qposadr[joint]),
        cdof=int(model.jnt_dofadr[joint]),
        DYNAMIC_YUBI_HOOK=None,
        FR3_STRATEGY=make_fr3_controller_strategy("joint_ik"),
        _TASK_OBJECT_NAMES=("cup",),
        _TASK_OBJECT_BINDINGS={
            "cup": {
                "joint_id": joint,
                "qpos_address": int(model.jnt_qposadr[joint]),
                "dof_address": int(model.jnt_dofadr[joint]),
            }
        },
        _READY_JOINT_SEGMENTS=ready_segments,
        _joint_ik_ready_gate_report=ready_gate,
    )
    settle.initialize_ready_object_registry(state)
    settle.initialize_settle_contact_registry(state)
    return state


@pytest.fixture(autouse=True)
def _no_rule_grasp_welds(monkeypatch):
    # The fixture model has no rule-grasp welds to inspect; the pristine check
    # is not what these tests are about.
    monkeypatch.setattr(settle, "_assert_rule_grasp_pristine", lambda state, context: None)


def _pre_fix_hold_action(state, grip_hold):
    # The lines _settle_ready_objects ran before the hold became a helper.
    arm_hold={
        key:np.asarray(
            state._READY_JOINT_SEGMENTS[key].commands[-1],dtype=np.float64
        )
        for key in ("R","L")
    }
    return settle.dynamic_env_action(state,arm_hold,grip_hold)


def _pre_fix_ready_gate(state):
    # The expression _validate_settled_objects evaluated before.
    return (
        state._joint_ik_ready_gate_report(state)
        if state.FR3_STRATEGY.uses_joint_ik else None
    )


def _fr3_settle(robot_name):
    embodiment = get_robot_profile(robot_name).embodiment
    segments = {
        key: SimpleNamespace(
            commands=np.stack(
                [_HOME[key], _HOME[key] + [0.02, -0.01], _HOME[key] + [0.05, -0.03]]
            )
        )
        for key in ("R", "L")
    }
    calls = []

    def ready_gate(state):
        calls.append(state)
        return {
            "R": {"position_error_mm": 0.25, "orientation_error_deg": 0.5},
            "L": {"position_error_mm": 0.75, "orientation_error_deg": 1.25},
        }

    state = _state(
        embodiment,
        robots=[_FR3Robot(), _FR3Robot()],
        ready_segments=segments,
        ready_gate=ready_gate,
    )
    report = settle._settle_ready_objects(state)
    return state, report, calls


@pytest.mark.parametrize("robot_name", ("fr3", "fr3_duo"))
def test_fr3_object_settle_hold_and_gate_are_unchanged(monkeypatch, robot_name):
    now_state, now_report, now_calls = _fr3_settle(robot_name)
    monkeypatch.setattr(settle, "_object_settle_hold_action", _pre_fix_hold_action)
    monkeypatch.setattr(settle, "_settled_ready_gate", _pre_fix_ready_gate)
    ref_state, ref_report, ref_calls = _fr3_settle(robot_name)

    # Each FR3 holds the end of its own ready trajectory, exactly.
    for robot, key in zip(now_state.env.robots, ("R", "L")):
        [parts] = robot.parts
        expected = np.asarray(now_state._READY_JOINT_SEGMENTS[key].commands[-1], np.float64)
        assert parts["right"].tobytes() == expected.tobytes()
    assert [a.tobytes() for a in now_state.env.actions] == [
        a.tobytes() for a in ref_state.env.actions
    ]
    assert now_state.d.qpos.tobytes() == ref_state.d.qpos.tobytes()
    assert now_state.d.qvel.tobytes() == ref_state.d.qvel.tobytes()
    assert now_report == ref_report
    assert now_report["ready_gate"]["L"]["orientation_error_deg"] == 1.25
    assert len(now_calls) == len(ref_calls) == 1
