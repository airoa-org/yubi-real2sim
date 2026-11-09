# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""FR3 joint-IK strategy used by the rollout worker."""

from dataclasses import dataclass

from robosuite_env.robots.fr3.controllers import FR3_CONTROLLERS


@dataclass(frozen=True)
class FR3ControllerStrategy:
    """Controller-dependent decisions for one rollout run."""

    # The validated --fr3-controller value.
    controller: str
    # Absolute whole-pose nullspace joint IK path: joint-space commands,
    # jerk-limited trajectories, desk-collision gate, ready-gate reports,
    # per-frame joint telemetry, and initial joint command seeding.
    uses_joint_ik: bool
    supports_joint_ready: bool = True


def make_fr3_controller_strategy(controller: str) -> FR3ControllerStrategy:
    """Build the strategy for one ``--fr3-controller`` value."""
    name = str(controller)
    if name not in FR3_CONTROLLERS:
        raise ValueError(f"FR3 controller must be one of {FR3_CONTROLLERS}, got {name!r}")
    return FR3ControllerStrategy(
        controller=name,
        uses_joint_ik=name == "joint_ik",
    )
