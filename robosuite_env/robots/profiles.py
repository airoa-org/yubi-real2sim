#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Robot-specific naming and construction details for the shared YUBI policy path.

The policy and task layers are hand-centric.  This module is the deliberately
small embodiment boundary for the two-FR3 rigs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

from robosuite_env.robots.fr3.controllers import FR3_CONTROLLERS
from robosuite_env.policy.policy_io import FR3_EEF_TO_HAND_ROOT_M


@dataclass(frozen=True)
class ArmProfile:
    """Resolved MuJoCo names for one logical policy hand."""

    joint_names: Tuple[str, ...]
    eef_body: str
    policy_body: str
    policy_offset_m: Tuple[float, float, float]
    base_body: str
    gripper_prefix: str
    body_prefixes: Tuple[str, ...]

    @property
    def finger_joint_names(self) -> Tuple[str, str]:
        return (
            f"{self.gripper_prefix}_finger_joint_left",
            f"{self.gripper_prefix}_finger_joint_right",
        )

    @property
    def finger_actuator_names(self) -> Tuple[str, str]:
        return (
            f"{self.gripper_prefix}_finger_l",
            f"{self.gripper_prefix}_finger_r",
        )


@dataclass(frozen=True)
class BaseSlot:
    """One physically mounted robot root this rig calibrates.

    Slots are not arms. The FR3 rigs stand two robots on the table and have
    two. The name is the key a calibration file uses, and the index says which
    robosuite robot it places.
    """

    name: str
    robot_index: int


@dataclass(frozen=True)
class RobotProfile:
    """Construction and logical-name contract for one supported embodiment."""

    name: str
    #: Mechanism family this rig is built from. Two rigs can be separate
    #: embodiments -- their own calibration, home pose, scene and desk -- while
    #: driving the same arms through the same controller: the FR3 Duo is two
    #: FR3s on a shared tilted mount. Runtime code that guards *mechanism*
    #: behaviour (joint_ik, the FR3 controller, initial arm joints) tests this;
    #: only code that means one specific rig tests ``name``.
    embodiment: str
    robosuite_robots: Tuple[str, ...]
    gripper_types: object
    arms: Dict[str, ArmProfile]
    #: Mounted roots, in the order a calibration file is expected to name them.
    base_slots: Tuple[BaseSlot, ...]

    def controller_action_dim_for(self, controller: str | None = None) -> int:
        """Return the action width for the selected robosuite controller.

        Whole-pose joint_ik consumes seven joint targets plus one gripper
        command per arm (16 total).
        """
        if controller is not None and controller not in FR3_CONTROLLERS:
            raise ValueError(
                f"{self.name} controller must be one of {FR3_CONTROLLERS}, got {controller!r}"
            )
        return 16

    @property
    def wrist_prefixes(self) -> Dict[str, str]:
        return {side: arm.gripper_prefix for side, arm in self.arms.items()}

    def runtime_gripper_types(self, *, calibrated_yubi: bool = False):
        """Return robosuite's embodiment-specific calibrated YUBI argument."""
        if not calibrated_yubi:
            return self.gripper_types
        return ("YubiGripper", "YubiGripperLeft")

    def load_controller_config(self):
        """Load the repo-local robosuite controller for this embodiment."""
        if self.embodiment == "fr3":
            from .fr3 import model

            return model.load_fr3_controller_config()
        raise AssertionError(f"unhandled robot embodiment {self.embodiment!r}")

    def load_nullspace_home_qpos(self):
        """Load the posture the whole-pose IK null-space is drawn toward.

        The FR3 arms run through a damped least-squares solver; each rig
        supplies the posture it rests in.
        """
        if self.embodiment == "fr3":
            from .fr3 import model

            return model.FR3_HOME_QPOS
        raise AssertionError(f"unhandled robot embodiment {self.embodiment!r}")


SUPPORTED_ROBOT_NAMES: Tuple[str, ...] = ("fr3", "fr3_duo")


def normalize_robot_name(value: str) -> str:
    """Return the canonical lower-case CLI/config robot name."""
    name = str(value).strip().lower()
    if name not in SUPPORTED_ROBOT_NAMES:
        expected = " or ".join(repr(known) for known in SUPPORTED_ROBOT_NAMES)
        raise ValueError(f"unsupported robot {value!r}; expected {expected}")
    return name


def robot_embodiment(value: str) -> str:
    """Return the mechanism family behind a canonical robot name.

    Call sites that hold only a recorded name -- an asset bundle request, a
    replayed ``--robot`` argument -- still need to ask "is this an FR3?"
    without constructing a profile. Answering from the registry keeps the
    embodiment of a rig in one place.
    """

    return get_robot_profile(normalize_robot_name(value)).embodiment


def _fr3_profile() -> RobotProfile:
    return RobotProfile(
        name="fr3",
        embodiment="fr3",
        robosuite_robots=("FR3", "FR3"),
        gripper_types=("YubiGripper", "YubiGripperLeft"),
        arms={
            "R": ArmProfile(
                joint_names=tuple(f"robot0_joint{i}" for i in range(1, 8)),
                eef_body="gripper0_right_eef",
                policy_body="gripper0_right_eef",
                policy_offset_m=FR3_EEF_TO_HAND_ROOT_M,
                base_body="robot0_link0",
                gripper_prefix="gripper0_right",
                body_prefixes=("robot0_", "gripper0_right_"),
            ),
            "L": ArmProfile(
                joint_names=tuple(f"robot1_joint{i}" for i in range(1, 8)),
                eef_body="gripper1_right_eef",
                policy_body="gripper1_right_eef",
                policy_offset_m=FR3_EEF_TO_HAND_ROOT_M,
                base_body="robot1_link0",
                gripper_prefix="gripper1_right",
                body_prefixes=("robot1_", "gripper1_right_"),
            ),
        },
        # Two pedestals, one robosuite robot each.
        base_slots=(BaseSlot("R", 0), BaseSlot("L", 1)),
    )


def _fr3_duo_profile() -> RobotProfile:
    """Two FR3s on the shared tilted Duo mount.

    Mechanically this is the FR3 profile: same robosuite robots, same YUBI
    hands, same joint and body names. What makes it a separate rig is
    everything the profile does not own -- its base calibration carries a full
    rpy rather than a yaw, and it brings its own registered scene, desk and
    home pose. Deriving it keeps that difference from being restated here.
    """

    return replace(_fr3_profile(), name="fr3_duo")


_ROBOT_PROFILE_BUILDERS = {
    "fr3": _fr3_profile,
    "fr3_duo": _fr3_duo_profile,
}


def get_robot_profile(value: str) -> RobotProfile:
    name = normalize_robot_name(value)
    return _ROBOT_PROFILE_BUILDERS[name]()
