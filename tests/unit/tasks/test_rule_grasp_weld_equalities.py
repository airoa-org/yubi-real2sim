# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The inactive weld pair every manipulation task declares up front."""

import unittest
import xml.etree.ElementTree as ET

from robosuite_env.tasks.equalities import (
    RULE_GRASP_WELD_NAMES,
    add_rule_grasp_weld_equalities,
)


def _task_xml():
    return ET.fromstring(
        """
        <mujoco>
          <worldbody>
            <body name="table"/>
            <body name="cube_main"/>
            <body name="plate_main"/>
          </worldbody>
        </mujoco>
        """
    )


class RuleGraspWeldEqualityTest(unittest.TestCase):
    def test_declares_one_inactive_weld_per_hand(self):
        root = _task_xml()
        welds = add_rule_grasp_weld_equalities(
            root, placeholder_bodies=("cube_main", "plate_main")
        )
        self.assertEqual(len(welds), 2)
        declared = {weld.get("name"): weld for weld in welds}
        self.assertEqual(set(declared), set(RULE_GRASP_WELD_NAMES.values()))
        for weld in welds:
            self.assertEqual(weld.get("active"), "false")
            self.assertEqual(weld.get("body1"), "cube_main")
            self.assertEqual(weld.get("body2"), "plate_main")

    def test_the_caller_names_the_pair_so_the_xml_is_stable(self):
        """An unrelated arena body must not move the declared pair.

        The pair is inert, but it is hashed into model_physics_sha256, so
        letting document order pick it made the fingerprint depend on edits
        that have nothing to do with grasping.
        """

        root = _task_xml()
        worldbody = root.find("worldbody")
        worldbody.insert(0, ET.Element("body", name="new_arena_fixture"))
        (weld, _) = add_rule_grasp_weld_equalities(
            root, placeholder_bodies=("cube_main", "plate_main")
        )
        self.assertEqual(weld.get("body1"), "cube_main")

    def test_rejects_a_body_the_task_does_not_declare(self):
        with self.assertRaisesRegex(ValueError, "not in the task XML"):
            add_rule_grasp_weld_equalities(
                _task_xml(), placeholder_bodies=("cube_main", "absent")
            )

    def test_rejects_a_body_welded_to_itself(self):
        with self.assertRaisesRegex(ValueError, "two distinct body names"):
            add_rule_grasp_weld_equalities(
                _task_xml(), placeholder_bodies=("cube_main", "cube_main")
            )

    def test_rejects_a_name_that_is_already_taken(self):
        root = _task_xml()
        equality = ET.SubElement(root, "equality")
        ET.SubElement(
            equality, "weld", name=RULE_GRASP_WELD_NAMES["R"], body1="a", body2="b"
        )
        with self.assertRaisesRegex(ValueError, "already exists"):
            add_rule_grasp_weld_equalities(
                root, placeholder_bodies=("cube_main", "plate_main")
            )


if __name__ == "__main__":
    unittest.main()
