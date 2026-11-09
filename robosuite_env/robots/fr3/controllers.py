# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""FR3 controller sets for the worker, kept in `real2sim` because its wheel does not ship `robosuite_env`."""

from real2sim.fr3_controllers import (
    FR3_CMD_TWIST_CONTROLLERS,
    FR3_CONTROLLERS,
    FR3_IMPEDANCE_HOOK_CONTROLLERS,
    fr3_uses_cmd_twist_torque,
    fr3_uses_impedance_hook,
    validate_fr3_controller,
)

__all__ = [
    "FR3_CMD_TWIST_CONTROLLERS",
    "FR3_CONTROLLERS",
    "FR3_IMPEDANCE_HOOK_CONTROLLERS",
    "fr3_uses_cmd_twist_torque",
    "fr3_uses_impedance_hook",
    "validate_fr3_controller",
]
