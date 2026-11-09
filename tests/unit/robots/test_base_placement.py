# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""What a base calibration has to prove before it is allowed to place a rig.

A calibration that merely parses still puts a robot somewhere wrong, and the
run finishes and reports a score. Every check here converts one of those quiet
outcomes into a refusal at load time.
"""

import json
import tempfile
import unittest
from pathlib import Path


from robosuite_env.paths import OBJECT_CALIBRATION_ROOT
from robosuite_env.robots.base_placement import (
    BasePlacementError,
    load_robot_base_placement,
)
from robosuite_env.robots.profiles import get_robot_profile


FLAT = OBJECT_CALIBRATION_ROOT / "fr3_base_from_blend.json"
DUO = OBJECT_CALIBRATION_ROOT / "fr3_base_duo.json"


def _write(tmp: Path, document) -> Path:
    path = tmp / "calibration.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class CheckedInCalibrationsTest(unittest.TestCase):
    def test_each_file_declares_the_rig_it_belongs_to(self):
        for path, robot in ((FLAT, "fr3"), (DUO, "fr3_duo")):
            placement = load_robot_base_placement(path, robot=robot)
            self.assertEqual(placement.robot, robot)
            self.assertEqual(
                set(placement.poses),
                {slot.name for slot in get_robot_profile(robot).base_slots},
            )

    def test_a_rig_cannot_borrow_another_rigs_calibration(self):
        """The Duo mount tilts; the flat pedestals do not.

        Loading one for the other is the mistake that leaves both arms upright
        on a mount that is not there -- or tilted on a table that is flat.
        """
        with self.assertRaises(BasePlacementError):
            load_robot_base_placement(DUO, robot="fr3")
        with self.assertRaises(BasePlacementError):
            load_robot_base_placement(FLAT, robot="fr3_duo")


class SchemaTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    _LEFT = {"pos": [-0.3, -0.3, 0.0], "yaw_deg": 45.0}

    def _base(self, **overrides):
        document = {
            "schema_version": 2,
            "frame": "table",
            "robot": "fr3",
            "base": {
                "R": {"pos": [0.3, -0.3, 0.0], "yaw_deg": 135.0},
                "L": {"pos": [-0.3, -0.3, 0.0], "yaw_deg": 45.0},
            },
        }
        document.update(overrides)
        return document

    def test_version_one_files_are_refused_rather_than_guessed_at(self):
        with self.assertRaises(BasePlacementError):
            load_robot_base_placement(_write(self.tmp, self._base(schema_version=1)))

    def test_a_pose_in_another_frame_is_refused(self):
        with self.assertRaises(BasePlacementError):
            load_robot_base_placement(_write(self.tmp, self._base(frame="world")))

    def test_the_slot_names_must_match_the_rig_exactly(self):
        """A missing or extra slot is a half-applied layout, not a default."""
        for base in ({"R": {"pos": [0, 0, 0], "yaw_deg": 0.0}},
                     {"R": {"pos": [0, 0, 0], "yaw_deg": 0.0},
                      "L": {"pos": [0, 0, 0], "yaw_deg": 0.0},
                      "spare": {"pos": [0, 0, 0], "yaw_deg": 0.0}},
                     {}):
            with self.assertRaises(BasePlacementError):
                load_robot_base_placement(_write(self.tmp, self._base(base=base)))

    def test_exactly_one_orientation_spelling_is_required(self):
        for entry in ({"pos": [0, 0, 0]},
                      {"pos": [0, 0, 0], "yaw_deg": 0.0, "rpy_deg": [0, 0, 0]}):
            with self.assertRaises(BasePlacementError):
                load_robot_base_placement(
                    _write(self.tmp, self._base(base={"R": entry, "L": self._LEFT}))
                )

    def test_malformed_numbers_are_refused(self):
        for entry in ({"pos": [0, 0], "yaw_deg": 0.0},
                      {"pos": [0, 0, 0], "rpy_deg": [0, 0]},
                      {"pos": [0, 0, float("nan")], "yaw_deg": 0.0}):
            with self.assertRaises(BasePlacementError):
                load_robot_base_placement(
                    _write(self.tmp, self._base(base={"R": entry, "L": self._LEFT}))
                )

    def test_an_unknown_rig_is_refused(self):
        with self.assertRaises(BasePlacementError):
            load_robot_base_placement(_write(self.tmp, self._base(robot="panda")))

    def test_the_file_digest_is_recorded(self):
        placement = load_robot_base_placement(_write(self.tmp, self._base()))
        self.assertEqual(len(placement.source_sha256), 64)


if __name__ == "__main__":
    unittest.main()
