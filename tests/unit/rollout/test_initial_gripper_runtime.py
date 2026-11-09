# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Runtime wiring for the schema-v2 declared initial gripper angle.

The rollout stages construct MuJoCo state on import, so these contracts are
checked statically, the same way the other structured-runtime contracts are.
The numeric behaviour of the declared angle itself is covered by
``tests/unit/robots/test_fr3_kinematics.py``.
"""

from tests._helpers.worker_ast import *  # noqa: F403

SETTLE_PATH = PACKAGE_ROOT / "rollout" / "stages" / "settle_and_ready.py"
SETTLE_SOURCE = SETTLE_PATH.read_text(encoding="utf-8")
SETTLE_TREE = ast.parse(SETTLE_SOURCE, filename=str(SETTLE_PATH))
SETTLE_STATE_PATH = PACKAGE_ROOT / "rollout" / "state" / "settle.py"
SETTLE_STATE_SOURCE = SETTLE_STATE_PATH.read_text(encoding="utf-8")

def test_session_declares_the_initial_gripper_fields() -> None:
    assert {"_INITIAL_GRIP_Q", "_READY_GRIP_CMD"} <= _session_field_names()

def test_build_model_resolves_the_declared_gripper_through_one_helper() -> None:
    # The MimicGen conflict check lives in that helper, so the stage must not
    # reach into ``pose.grip_by_arm`` directly and bypass it.
    assert _call_lines(BUILD_MODEL_TREE, "resolve_initial_grip_targets")
    assert not {
        ast.unparse(node.value) for node in ast.walk(BUILD_MODEL_TREE)
        if isinstance(node, ast.Attribute) and node.attr == "grip_by_arm"
    }

def test_the_ready_drive_grip_command_has_exactly_one_source() -> None:
    # The historical fully-open command survives as the default of
    # ``_READY_GRIP_CMD``; every other use must read that one variable, so a
    # schema-1 profile keeps a byte-identical ready drive and a schema-2
    # profile changes it in exactly one place.
    assert SETTLE_SOURCE.count('{"R":0.7854,"L":0.7854}') == 1
    assert SETTLE_SOURCE.count("_READY_GRIP_CMD") >= 3
    assert "set_targets(dict(_READY_GRIP_CMD)" in SETTLE_SOURCE
    assert "dict(state._READY_GRIP_CMD)," in SETTLE_SOURCE

def test_the_declared_gripper_is_forced_after_the_ready_drive() -> None:
    assert (
        '_force_finger_state(state,state._INITIAL_GRIP_Q["L"],'
        'state._INITIAL_GRIP_Q["R"])'
    ) in SETTLE_SOURCE

def test_mimicgen_and_grasp_test_keep_priority_over_the_declared_gripper() -> None:
    grasp_test = SETTLE_SOURCE.index("if state.A.grasp_test:\n        _force_finger_state")
    policy_init = SETTLE_SOURCE.index("elif state.POLICY_SOURCE and state.POLICY_INIT_GRIP is not None:")
    declared = SETTLE_SOURCE.index("elif state._INITIAL_GRIP_Q is not None:")
    fallback = SETTLE_SOURCE.index("        _reset_yubi_from_current(state)")

    assert grasp_test < policy_init < declared < fallback

def test_the_settled_init_state_contract_still_binds_the_pose_file_hash() -> None:
    # Switching a profile to a different registered pose changes this hash, so
    # a cached deterministic init state built from the previous pose cannot be
    # reused silently.
    assert "initial_arm_joints_sha256" in SETTLE_STATE_SOURCE
    assert "_INITIAL_ARM_JOINT_POSE.source_sha256" in SETTLE_STATE_SOURCE

def test_finalize_records_the_applied_initial_gripper() -> None:
    assert "initial_gripper_raw_rad" in STAGE_FINALIZE_SOURCE
    assert _undeclared_session_attribute_writes((STAGE_FINALIZE_TREE,)) == set()
