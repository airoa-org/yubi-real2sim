# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Run-summary fields that name the arm controller and report object lift.

``cup_lift_mm`` follows one fixed joint relative to its own pre-loop height, so
it names its joint and datum; box tasks report the lift of their own target.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robosuite_env.rendering.wrist_render_profiles import wrist_image_intrinsics
from robosuite_env.robots.fr3.runtime_ik import (
    controller_path_summary,
    controller_summary_name,
)
from robosuite_env.rollout.stages import finalize_outputs as fo


REPO_ROOT = Path(__file__).resolve().parents[3]


# ------------------------------------------------------------ arm controller


def _args(**overrides):
    values = {
        "physics": "dynamic",
        "state_log_replay": None,
        "ik_replay": False,
        "fr3_controller": "joint_ik",
        "fr3_nullspace": "home",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("robot", ["fr3", "fr3_duo"])
@pytest.mark.parametrize("physics", ["dynamic", "kinematic"])
@pytest.mark.parametrize("nullspace", ["home", "off"])
def test_fr3_rigs_keep_the_fr3_controller_name(robot, physics, nullspace):
    state = SimpleNamespace(
        ROBOT=robot,
        EMBODIMENT="fr3",
        A=_args(physics=physics, fr3_nullspace=nullspace),
    )

    fields = fo.arm_controller_fields(state)

    assert fields["arm_controller"] == controller_summary_name(physics, "joint_ik", nullspace)
    assert fields["arm_controller_detail"]["path"] == controller_path_summary(physics, "joint_ik")
    assert fields["arm_controller_detail"]["rig"] == robot


@pytest.mark.parametrize("robot", ["fr3", "fr3_duo"])
def test_state_log_replay_is_named_as_a_replay_on_every_rig(robot):
    state = SimpleNamespace(
        ROBOT=robot,
        EMBODIMENT="fr3",
        A=_args(state_log_replay="/runs/source/logs/sim_log.jsonl"),
    )

    fields = fo.arm_controller_fields(state)

    assert fields["arm_controller"] == "generation_state_log_qpos_replay"


def test_an_unknown_embodiment_is_refused():
    state = SimpleNamespace(ROBOT="ur5", EMBODIMENT="ur5", A=_args())

    with pytest.raises(ValueError, match="ur5"):
        fo.arm_controller_fields(state)


# ------------------------------------------------------------ run() wiring


def _run_summary_dict():
    tree = ast.parse(Path(fo.__file__).read_text(encoding="utf-8"))
    run = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    for node in ast.walk(run):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "summ" for t in node.targets)
            and isinstance(node.value, ast.Dict)
        ):
            return node.value
    raise AssertionError("run() builds no summ dict literal")


def test_run_summary_takes_controller_and_lift_fields_from_the_helpers():
    summary = _run_summary_dict()
    keys = {
        key.value: value
        for key, value in zip(summary.keys, summary.values)
        if isinstance(key, ast.Constant)
    }
    splats = [ast.unparse(value) for key, value in zip(summary.keys, summary.values) if key is None]

    assert "arm_controller_fields(state)" in splats
    assert "legacy_target_lift_fields(state, cup_pos)" in splats
    assert "task_lift_fields(state._RECS, _success_summary.get('success_criterion'))" in splats
    assert "task_binding_fields(state)" in splats
    # No second, hand-written copy of any helper-owned key.
    assert not set(keys) & {
        "arm_controller",
        "arm_controller_detail",
        "cup_lift_mm",
        "cup_displacement_mm",
        "cup_lift_mm_definition",
        "task_lift_mm",
        "task_lift",
        "task_target",
        "task_receptacle",
    }
    assert "state.EMBODIMENT == 'fr3'" in ast.unparse(keys["fr3_controller"])


def test_run_summary_records_the_rendered_policy_wrist_intrinsics():
    summary = _run_summary_dict()
    keys = {
        key.value: ast.unparse(value)
        for key, value in zip(summary.keys, summary.values)
        if isinstance(key, ast.Constant)
    }
    assert keys["policy_wrist_intrinsics"] == "policy_wrist_intrinsics(state)"

    profile = "franka-pinhole-wrist-rgb"
    state = SimpleNamespace(
        POLICY_SOURCE=True,
        A=SimpleNamespace(policy_image_source="sim", policy_yubi_render_profile=profile),
    )
    record = fo.policy_wrist_intrinsics(state)
    assert record == {
        side: wrist_image_intrinsics(profile, side).as_dict() for side in ("left", "right")
    }
    assert json.loads(json.dumps(record)) == record
    # Recorded wrist video is not this renderer's image, and replay has no policy.
    state.A.policy_image_source = "lerobot"
    assert fo.policy_wrist_intrinsics(state) is None
    state.A.policy_image_source = "sim"
    state.POLICY_SOURCE = False
    assert fo.policy_wrist_intrinsics(state) is None


def test_run_summary_records_the_wrist_profile_of_a_run_without_a_policy():
    # Display wrist insets render with the profile, so null would leave rerender guessing.
    summary = _run_summary_dict()
    keys = {
        key.value: ast.unparse(value)
        for key, value in zip(summary.keys, summary.values)
        if isinstance(key, ast.Constant)
    }
    assert keys["policy_yubi_render_profile"] == "state.A.policy_yubi_render_profile"
    assert keys["policy_wrist_backend"] == "state.A.policy_wrist_backend"
    # The replay of this run selects the same KB4 near cull from it (rollout.replay.recorded_wrist).
    assert keys["gs_kb4_near_cull"] == "state.A.gs_kb4_near_cull"
    assert "kb4_near_meters()" in keys["gs_kb4_near_m"]


@pytest.mark.parametrize(
    ("binding", "expected"),
    (
        ({"target": "cup", "receptacle": "plate"}, {"task_target": "cup", "task_receptacle": "plate"}),
        ({"target": "pen", "receptacle": "box"}, {"task_target": "pen", "task_receptacle": "box"}),
    ),
)
def test_summary_serializes_the_resolved_task_binding(binding, expected):
    assert fo.task_binding_fields(SimpleNamespace(_TASK_BINDING=binding)) == expected


@pytest.mark.parametrize("binding", (None, {}, {"target": "pen"}, {"target": "", "receptacle": "box"}))
def test_summary_refuses_an_invalid_task_binding(binding):
    with pytest.raises(RuntimeError, match="task binding"):
        fo.task_binding_fields(SimpleNamespace(_TASK_BINDING=binding))


# ------------------------------------------------------------ legacy lift


def test_legacy_cup_lift_keeps_its_value_and_names_its_joint_and_datum():
    # A recorded run whose plain cup launched from inside the table.
    state = SimpleNamespace(
        cup=np.array([0.05, 0.10, 0.07669]),
        CUPa=21,
        _TASK_OBJECT_BINDINGS={
            "support_plate": {"joint_name": "support_plate_joint0", "qpos_address": 28},
            "cup": {"joint_name": "cube_joint0", "qpos_address": 21},
        },
        m=None,
    )
    cup_pos = np.array(
        [[0.05, 0.10, 0.13348], [0.05, 0.10, 0.10686], [0.05, 0.11, 0.10885]]
    )

    fields = fo.legacy_target_lift_fields(state, cup_pos)

    assert fields["cup_lift_mm"] == round((0.13348 - 0.07669) * 1000.0, 3)
    assert fields["cup_displacement_mm"] == round(
        float(np.linalg.norm(cup_pos - state.cup[None, :], axis=1).max() * 1000.0), 3
    )
    definition = fields["cup_lift_mm_definition"]
    assert definition["joint"] == "cube_joint0"
    assert definition["table_relative"] is False
    assert definition["follows_task_stage"] is False
    assert definition["replacement"] == ["task_lift_mm", "task_lift"]
    for consumer in definition["kept_for"]:
        assert (REPO_ROOT / consumer).is_file(), consumer
        assert "cup_lift_mm" in (REPO_ROOT / consumer).read_text(encoding="utf-8")


def test_legacy_cup_lift_names_the_joint_from_the_compiled_model():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
<mujoco>
  <worldbody>
    <body name="container"><freejoint name="fixture_box_joint0"/><geom size="0.1"/></body>
    <body name="tape"><freejoint name="tape_joint0"/><geom size="0.02"/></body>
  </worldbody>
</mujoco>
"""
    )
    tape = model.joint("tape_joint0").id
    state = SimpleNamespace(
        cup=np.zeros(3),
        CUPa=int(model.jnt_qposadr[tape]),
        _TASK_OBJECT_BINDINGS=None,
        m=model,
    )

    fields = fo.legacy_target_lift_fields(state, np.zeros((0, 3)))

    assert fields["cup_lift_mm"] is None
    assert fields["cup_displacement_mm"] is None
    assert fields["cup_lift_mm_definition"]["joint"] == "tape_joint0"


# ------------------------------------------------------------ box lift


def test_box_pack_reports_the_prompted_object_above_the_table():
    heights = [(0.5, 0.0), (0.4, 0.0), (25.0, -0.1), (80.0, 0.2), (12.0, 0.1)]
    records = [
        {
            "frame": frame,
            "success": {
                "task": "object_in_box",
                "metrics": {
                    "object": "tape",
                    "object_height_mm": obj,
                    "box_height_mm": box,
                    "table_top_z_mm": 34.0,
                },
            },
        }
        for frame, (obj, box) in enumerate(heights)
    ]

    fields = fo.task_lift_fields(records, {"name": "object_in_box"})

    block = fields["task_lift"]
    assert fields["task_lift_mm"] == 80.0
    assert set(block["objects"]) == {"tape"}
    assert block["objects"]["tape"]["max_frame"] == 3
    assert block["box"]["min_mm"] == -0.1
    assert block["source"] == "success.metrics.object_height_mm"


@pytest.mark.parametrize("records", [
    [{"frame": 0, "success": {"task": "cup_on_plate", "metrics": {"support_gap_mm": 14.2}}}],
    [{"frame": 0, "cup_pos": [0.0, 0.0, 0.1]}],
])
def test_tasks_without_table_heights_report_no_task_lift(records):
    fields = fo.task_lift_fields(records, None)

    assert fields["task_lift_mm"] is None
    assert fields["task_lift"]["available"] is False
