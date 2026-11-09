# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""FR3 ``--fr3-controller`` values shared by the config schema and the rollout worker."""

FR3_CONTROLLERS = ("joint_ik", "cartesian_impedance")
FR3_CMD_TWIST_CONTROLLERS = frozenset({"cartesian_impedance"})
FR3_IMPEDANCE_HOOK_CONTROLLERS = frozenset({"cartesian_impedance"})  # torques from CartesianImpedanceHook


def fr3_uses_cmd_twist_torque(controller: str) -> bool:
    """Whether the policy sends cmd_twist actions that run at a 1 ms physics step."""
    return controller in FR3_CMD_TWIST_CONTROLLERS


def fr3_uses_impedance_hook(controller: str) -> bool:
    """Whether CartesianImpedanceHook computes the arm torques and reads ``fr3_impedance``."""
    return controller in FR3_IMPEDANCE_HOOK_CONTROLLERS


def validate_fr3_controller(controller: str) -> None:
    """Reject at startup a controller that this simulator cannot drive."""
    if controller not in FR3_CONTROLLERS:
        raise ValueError(f"FR3 controller must be one of {FR3_CONTROLLERS}, got {controller!r}")
    if fr3_uses_cmd_twist_torque(controller) and not fr3_uses_impedance_hook(controller):
        raise ValueError(f"FR3 controller {controller!r} sends cmd_twist actions but has no torque hook")
