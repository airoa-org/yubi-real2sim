# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Rig identity and mechanism family are two different questions.

FR3 and the FR3 Duo are two rigs built from the same
mechanism: the Duo is a pair of FR3 arms on a shared tilted mount, with its own
base calibration, home pose, registered scene and desk. Keeping them apart lets
each carry its own calibration without a scene-name guess, but a runtime guard
that reads ``robot == "fr3"`` is usually asking about the mechanism -- and
would silently stop applying to the Duo.
"""

import ast
import unittest

import numpy as np
from pathlib import Path

from robosuite_env.contracts.assets import runtime_bundle_ids
from robosuite_env.robots.profiles import (
    SUPPORTED_ROBOT_NAMES,
    get_robot_profile,
    robot_embodiment,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


class EmbodimentRegistryTest(unittest.TestCase):
    def test_two_rigs_one_mechanism(self):
        self.assertEqual(SUPPORTED_ROBOT_NAMES, ("fr3", "fr3_duo"))
        self.assertEqual(robot_embodiment("fr3"), "fr3")
        self.assertEqual(robot_embodiment("fr3_duo"), "fr3")

    def test_duo_is_derived_from_fr3_rather_than_restated(self):
        """Only the name may differ; a copy would drift on the next FR3 edit."""
        fr3 = get_robot_profile("fr3")
        duo = get_robot_profile("fr3_duo")

        self.assertNotEqual(duo.name, fr3.name)
        self.assertEqual(duo.embodiment, fr3.embodiment)
        self.assertEqual(duo.robosuite_robots, fr3.robosuite_robots)
        self.assertEqual(duo.gripper_types, fr3.gripper_types)
        self.assertEqual(duo.arms, fr3.arms)

    def test_both_fr3_rigs_request_the_tracked_arm_bundle(self):
        """Both FR3 rigs ship the same tracked arm meshes."""
        for name in ("fr3", "fr3_duo"):
            self.assertIn("fr3", runtime_bundle_ids(name, "cup_plate"))


class NullspaceHomePostureTest(unittest.TestCase):
    """Every rig has to answer what posture its IK rests toward.

    The whole-pose solver is damped least squares over the arms the profile
    names, so it serves any rig; only the null-space target differs.
    """

    def test_each_rig_supplies_a_seven_joint_posture(self):
        for name in SUPPORTED_ROBOT_NAMES:
            home = get_robot_profile(name).load_nullspace_home_qpos()
            if isinstance(home, dict):
                self.assertEqual(set(home), {"R", "L"})
                for side, values in home.items():
                    self.assertEqual(len(values), 7, f"{name}/{side}")
            else:
                self.assertEqual(len(home), 7, name)

    def test_both_fr3_rigs_rest_toward_the_same_posture(self):
        self.assertTrue(
            np.allclose(
                get_robot_profile("fr3").load_nullspace_home_qpos(),
                get_robot_profile("fr3_duo").load_nullspace_home_qpos(),
            )
        )


class MechanismGuardsTest(unittest.TestCase):
    """Keep rig identity out of the guards that mean "is this an FR3?".

    ``state.ROBOT`` holds more than one value. A guard spelled
    ``state.ROBOT == "fr3"`` reads as if nothing changed while quietly turning
    itself off for the Duo -- taking the FR3 controller, the joint_ik strategy,
    the initial arm joints and most of the run summary with it. There is no
    error to notice: the run completes and reports less.
    """

    def test_no_runtime_guard_compares_the_rig_name_to_fr3(self):
        offenders = []
        for path in sorted((REPO_ROOT / "robosuite_env").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Compare):
                    continue
                left = node.left
                if not (
                    isinstance(left, ast.Attribute)
                    and left.attr == "ROBOT"
                    and isinstance(left.value, ast.Name)
                    and left.value.id == "state"
                ):
                    continue
                for op, comparator in zip(node.ops, node.comparators):
                    if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(
                        comparator, ast.Constant
                    ):
                        offenders.append(
                            f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
                        )
        self.assertEqual(
            offenders,
            [],
            "compare state.EMBODIMENT for mechanism behaviour; state.ROBOT is "
            "the rig identity and excludes fr3_duo from every FR3 guard",
        )


if __name__ == "__main__":
    unittest.main()
