# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""FR3 base-pose orientation: URDF rpy in, robosuite euler out.

The two-pedestal layouts only ever set a yaw, and ``Rz(y)`` is the same matrix
whichever order the three rotations compose in.  The Duo mount is the first
pose with a non-zero roll and pitch, so it is the first one that can tell the
two conventions apart -- these tests pin the distinction.
"""

import unittest

import numpy as np

from robosuite_env.paths import OBJECT_CALIBRATION_ROOT
from robosuite_env.robots.base_placement import (
    load_robot_base_placement,
    resolve_robot_base_placement,
)
from robosuite_env.tasks.scene import (
    place_robots_at_calibrated_bases,
    place_robots_at_stock_offsets,
    robosuite_base_euler,
)


DUO_BASE_JSON = OBJECT_CALIBRATION_ROOT / "fr3_base_duo.json"

#: The mount's two base orientations, in the URDF order the file quotes them.
#: ``DuoCalibrationFileTest`` pins these against the checked-in calibration, so
#: the conversion tests below cannot drift into testing an invented pose.
DUO_RPY_DEG = {
    "R": (51.1651, -10.0268, -63.4721),
    "L": (-51.1651, -10.0268, -116.5279),
}


def _rx(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _ry(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _urdf(rpy):
    """URDF/blend convention: ``Rz(y) @ Ry(p) @ Rx(r)``."""
    r, p, y = rpy
    return _rz(y) @ _ry(p) @ _rx(r)


def _robosuite(euler):
    """What robosuite's ``euler2mat`` builds: ``Rx(r) @ Ry(p) @ Rz(y)``."""
    a, b, c = euler
    return _rx(a) @ _ry(b) @ _rz(c)


class RobosuiteEulerTest(unittest.TestCase):
    def test_yaw_only_is_a_fixed_point(self):
        """Yaw-only mounts, as in the flat two-pedestal scenes, are unaffected."""
        for yaw_deg in (0.0, 45.0, 135.0, -63.4721, 180.0):
            rpy = [0.0, 0.0, np.radians(yaw_deg)]
            np.testing.assert_allclose(robosuite_base_euler(rpy), rpy, atol=1e-12)

    def test_reproduces_the_urdf_rotation(self):
        """The returned triple must rebuild the same matrix under robosuite."""
        for rpy_deg in (
            DUO_RPY_DEG["R"],
            DUO_RPY_DEG["L"],
            (30.0, 20.0, 10.0),
            (-5.0, 85.0, 200.0),
        ):
            rpy = np.radians(rpy_deg)
            np.testing.assert_allclose(
                _robosuite(robosuite_base_euler(rpy)), _urdf(rpy), atol=1e-12
            )

    def test_passing_urdf_angles_through_would_mis_orient_the_mount(self):
        """The naive path is measurably wrong for a mount with roll and pitch."""
        rpy = np.radians(DUO_RPY_DEG["R"])
        naive = _robosuite(rpy)
        correct = _urdf(rpy)
        # The mount's own z axis lands over 40 degrees away.
        angle = np.degrees(
            np.arccos(np.clip(naive[:, 2] @ correct[:, 2], -1.0, 1.0))
        )
        self.assertGreater(angle, 40.0)


class DuoCalibrationFileTest(unittest.TestCase):
    """Bind the conversion tests above to the mount they claim to describe."""

    def test_the_checked_in_calibration_holds_those_poses(self):
        loaded = load_robot_base_placement(DUO_BASE_JSON, robot="fr3_duo")
        for side, rpy_deg in DUO_RPY_DEG.items():
            np.testing.assert_allclose(
                np.degrees(loaded.pose(side).rpy_rad), rpy_deg, atol=1e-9
            )

    def test_the_arms_are_mirrored_about_the_mount_axis(self):
        """Left yaw is -116.53, not the right arm's -63.47 negated.

        Negating the yaw is the plausible-looking mistake: it keeps the roll
        mirror-symmetric and still compiles, so only the file itself settles it.
        """
        right_yaw, left_yaw = DUO_RPY_DEG["R"][2], DUO_RPY_DEG["L"][2]
        self.assertAlmostEqual(right_yaw + left_yaw, -180.0, places=6)
        self.assertNotAlmostEqual(left_yaw, -right_yaw, places=3)


class _StubModel:
    """Record what ``place_task_robots`` asks robosuite to do.

    The real ``ManipulatorModel`` needs a compiled MJCF, which would drag the
    FR3 mesh bundle into a test whose whole subject
    is arithmetic on a base pose.
    """

    base_xpos_offset = {"table": staticmethod(lambda length: (-0.5 - length / 2.0, 0.0, 0.0))}

    def __init__(self, class_name="FR3"):
        self.__class__.__name__ = class_name
        self.base_xpos = None
        self.base_ori = None

    def set_base_xpos(self, value):
        self.base_xpos = np.asarray(value, dtype=float)

    def set_base_ori(self, value):
        self.base_ori = np.asarray(value, dtype=float)


class _StubRobot:
    def __init__(self, class_name="FR3"):
        self.robot_model = _StubModel(class_name)


TABLE_TOP = np.array([0.0, 0.0, 0.8])
TABLE_LENGTH = 0.8


class PlaceTaskRobotsGoldenTest(unittest.TestCase):
    """Pin the world poses ``place_task_robots`` produces.

    The tests above stop at the rpy conversion; these pin what is put on the
    table, so a refactor that moves an arm by a millimetre has to say so out
    loud.
    """

    def test_flat_two_pedestal_world_poses(self):
        robots = [_StubRobot(), _StubRobot()]
        place_robots_at_calibrated_bases(
            robots,
            table_top=TABLE_TOP,
            placement=resolve_robot_base_placement("fr3"),
        )
        np.testing.assert_allclose(
            robots[0].robot_model.base_xpos,
            [0.340958774, -0.269104302, 0.801631439],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            robots[0].robot_model.base_ori, [0.0, 0.0, 2.356194508], atol=1e-9
        )
        np.testing.assert_allclose(
            robots[1].robot_model.base_xpos,
            [-0.359041214, -0.269104302, 0.801631439],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            robots[1].robot_model.base_ori, [0.0, 0.0, 0.785398146], atol=1e-9
        )

    def test_duo_mount_world_poses(self):
        """The only layout whose roll and pitch are non-zero.

        The Duo rig resolves its own file from the registry.
        """
        robots = [_StubRobot(), _StubRobot()]
        place_robots_at_calibrated_bases(
            robots,
            table_top=TABLE_TOP,
            placement=resolve_robot_base_placement("fr3_duo"),
        )
        np.testing.assert_allclose(
            robots[0].robot_model.base_xpos, [-0.05018, -0.0369, 0.988585], atol=1e-9
        )
        np.testing.assert_allclose(
            robots[0].robot_model.base_ori,
            [0.384997202, -0.841597860, -0.849833232],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            robots[1].robot_model.base_xpos, [0.05018, -0.0369, 0.988585], atol=1e-9
        )
        np.testing.assert_allclose(
            robots[1].robot_model.base_ori,
            [0.384997202, 0.841597860, -2.291759422],
            atol=1e-9,
        )

    def test_nominal_placement_discards_the_table_top(self):
        """The stock path is not the calibrated path in different clothes.

        It reads only ``table_length``, never sets an orientation, and spreads
        robots on a fixed 0.5 m pitch. A caller that lands here by accident
        gets a robot 0.8 m below the table it is meant to work on, with no
        exception raised.
        """
        robots = [_StubRobot(), _StubRobot()]
        place_robots_at_stock_offsets(robots, table_length=TABLE_LENGTH)
        np.testing.assert_allclose(robots[0].robot_model.base_xpos, [-0.9, -0.25, 0.0])
        np.testing.assert_allclose(robots[1].robot_model.base_xpos, [-0.9, 0.25, 0.0])
        self.assertIsNone(robots[0].robot_model.base_ori)
        self.assertIsNone(robots[1].robot_model.base_ori)

    def test_a_rig_cannot_be_placed_with_the_wrong_number_of_robots(self):
        with self.assertRaises(ValueError):
            place_robots_at_calibrated_bases(
                [_StubRobot()],
                table_top=TABLE_TOP,
                placement=resolve_robot_base_placement("fr3"),
            )


if __name__ == "__main__":
    unittest.main()
