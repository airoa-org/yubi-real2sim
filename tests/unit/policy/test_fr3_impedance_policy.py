# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The deployed mapper consumes quaternion xyz as raw velocity coefficients."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("scipy")

from robosuite_env.rollout.observe.policy_client import (
    _build_policy_target_trajectory, _prepare_policy_action_chunk,
)


def state(controller):
    return SimpleNamespace(EMBODIMENT="fr3", A=SimpleNamespace(
        fr3_controller=controller, policy_action_clamp=False,
        policy_pose_repr="delta", policy_delta_anchor="observation"))


def test_twist_noop_accepts_zero_quaternion_and_needs_no_pose_anchor():
    actions = np.zeros((16, 16))
    controller = state("cartesian_impedance")
    chunk, telemetry = _prepare_policy_action_chunk(controller, actions)
    np.testing.assert_array_equal(chunk, actions)
    assert not telemetry["enabled"]
    assert _build_policy_target_trajectory(controller, chunk, {}) is None
    # A zero quaternion still cannot represent an IK pose delta.
    with pytest.raises(ValueError, match="quaternion"):
        _prepare_policy_action_chunk(state("joint_ik"), actions)


def test_twist_mapper_does_not_normalize_or_flip_quaternion_coefficients():
    actions = np.zeros((2, 16))
    actions[:, 3:7] = [.1, -.2, .3, -2.]
    actions[:, 10:14] = [-.4, .2, .3, 0.]
    chunk, _ = _prepare_policy_action_chunk(state("cartesian_impedance"), actions)
    np.testing.assert_array_equal(chunk, actions)
    actions[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _prepare_policy_action_chunk(state("cartesian_impedance"), actions)
