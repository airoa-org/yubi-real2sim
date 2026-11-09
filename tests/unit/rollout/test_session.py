# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Tests for the RolloutSession state contract."""

from tests._helpers.worker_ast import *  # noqa: F403

def test_session_fields_keep_runtime_probe_defaults() -> None:
    fields = _session_field_names()
    assert {
        "_FR3_DYNAMIC_CONTROL_TICKS_PER_ACTION",
        "_FR3_DYNAMIC_ENV_STEP_COUNT",
        "_cid",
        "_POLICY_WRIST_EPHEMERAL",
    } <= fields

def test_session_fields_cover_stage_attribute_writes() -> None:
    stage_trees = tuple(
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in (
            PACKAGE_ROOT / "rollout" / "stages" / "run_frames.py",
            PACKAGE_ROOT / "rollout" / "stages" / "finalize_outputs.py",
        )
    )
    assert _undeclared_session_attribute_writes(stage_trees) == set()

def test_session_field_checker_reports_undeclared_state_write() -> None:
    synthetic_tree = ast.parse(
        """
def run(state):
    state.njmup = 5
"""
    )
    assert _undeclared_session_attribute_writes((synthetic_tree,)) == {"njmup"}
