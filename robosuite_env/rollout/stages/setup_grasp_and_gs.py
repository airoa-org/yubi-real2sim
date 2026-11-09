# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Grasp/contact registries and GS client setup stage."""

from robosuite_env.rollout.build.collision_gate import configure_arm_desk_collision_gate
from robosuite_env.rollout.build.geoms import initialize_geometry_sets
from robosuite_env.rollout.control.rule_grasp import _new_rule_grasp_state
from robosuite_env.rollout.render.gs_client import setup_gs_client
from robosuite_env.rollout.state.dumps import dump_state_once as _dump_state_once
from robosuite_env.rollout.state.settle import (
    initialize_ready_object_registry,
    initialize_settle_contact_registry,
)

def run(state):
    """Set up grasp/contact instrumentation and the 3DGS background client."""
    setup_gs_client(state)

    initialize_geometry_sets(state)

    state.RULE_GRASP=_new_rule_grasp_state()
    state._dump_state_once = _dump_state_once

    # A single-stage rollout for LeRobot task_index 1..5 cannot start from the
    # calibrated frame-zero placement, so an optional chained-run stage
    # boundary is applied before the ready-object registry anchors the initial
    # condition. Every downstream settle and load-bearing gate then runs
    # unchanged on the new placement.
    initialize_ready_object_registry(state)
    configure_arm_desk_collision_gate(state)
    initialize_settle_contact_registry(state)


__all__ = ["run"]
