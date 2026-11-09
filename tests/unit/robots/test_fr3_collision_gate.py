# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import pytest

try:
    import mujoco
    from robosuite_env.robots.fr3.collision_gate import ArmDeskCollisionGate
except ImportError:  # pragma: no cover - heavy dependency absent from CI
    mujoco = None
    ArmDeskCollisionGate = None


_MODEL_XML = """
<mujoco model="arm_desk_gate_test">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <default>
    <joint limited="true" range="-0.5 0.5"/>
    <geom contype="1" conaffinity="1"/>
  </default>
  <worldbody>
    <geom name="gw_desk_coll_0" type="box" pos="0 0 0"
          size="0.5 0.5 0.05"/>
    <body name="robot0_link1" pos="0 0 0.25">
      <joint name="joint1" type="slide" axis="0 0 1"/>
      <geom name="robot0_link1_collision" type="sphere" size="0.05"/>
      <body name="robot0_link2">
        <inertial pos="0 0 0" mass="0.01"
                  diaginertia="0.00001 0.00001 0.00001"/>
        <joint name="joint2" type="hinge" axis="1 0 0"/>
        <body name="robot0_link3">
          <inertial pos="0 0 0" mass="0.01"
                    diaginertia="0.00001 0.00001 0.00001"/>
          <joint name="joint3" type="hinge" axis="0 1 0"/>
          <body name="robot0_link4">
            <inertial pos="0 0 0" mass="0.01"
                      diaginertia="0.00001 0.00001 0.00001"/>
            <joint name="joint4" type="hinge" axis="0 0 1"/>
            <body name="robot0_link5">
              <inertial pos="0 0 0" mass="0.01"
                        diaginertia="0.00001 0.00001 0.00001"/>
              <joint name="joint5" type="hinge" axis="1 0 0"/>
              <body name="robot0_link6">
                <inertial pos="0 0 0" mass="0.01"
                          diaginertia="0.00001 0.00001 0.00001"/>
                <joint name="joint6" type="hinge" axis="0 1 0"/>
                <body name="robot0_link7">
                  <inertial pos="0 0 0" mass="0.01"
                            diaginertia="0.00001 0.00001 0.00001"/>
                  <joint name="joint7" type="hinge" axis="0 0 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="cup" pos="0 0 0.25">
      <freejoint/>
      <geom name="cup_collision" type="sphere" size="0.06"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.mark.usefixtures("mujoco")
class ArmDeskCollisionGateTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(_MODEL_XML)
        self.arm_qpos = np.asarray(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        f"joint{index}",
                    )
                ]
                for index in range(1, 8)
            ],
            dtype=np.int64,
        )
        self.arm_geom = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "robot0_link1_collision",
        )
        self.desk_geom = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "gw_desk_coll_0",
        )
        self.gate = ArmDeskCollisionGate(
            self.model,
            arm_qpos_indices={"R": self.arm_qpos},
            arm_geom_ids={"R": [self.arm_geom]},
            desk_geom_ids=[self.desk_geom],
            clearance_m=0.01,
        )

    def test_signed_distance_ignores_overlapping_cup(self):
        checks = self.gate.check_path(
            "R",
            np.asarray(
                [
                    [0.0, 0, 0, 0, 0, 0, 0],
                    [-0.145, 0, 0, 0, 0, 0, 0],
                    [-0.25, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.float64,
            ),
        )
        safe, inside_clearance, colliding = checks

        self.assertEqual(len(checks), 3)
        self.assertEqual([check.safe for check in checks], [True, False, False])
        self.assertTrue(safe.safe)
        self.assertAlmostEqual(safe.minimum_distance_m, 0.15, places=9)
        self.assertEqual(safe.arm_geom_name, "robot0_link1_collision")
        self.assertEqual(safe.desk_geom_name, "gw_desk_coll_0")
        self.assertEqual(safe.evaluated_pairs, 1)
        self.assertFalse(inside_clearance.safe)
        self.assertGreater(inside_clearance.minimum_distance_m, 0.0)
        self.assertLess(inside_clearance.minimum_distance_m, 0.01)
        self.assertFalse(colliding.safe)
        self.assertLess(colliding.minimum_distance_m, 0.0)

    def test_filter_selects_largest_safe_prefix(self):
        requested = np.asarray([-0.25, 0, 0, 0, 0, 0, 0])
        result = self.gate.filter_target(
            "R",
            requested,
            np.zeros(7),
            path_samples=16,
            bisection_iterations=12,
            retreat_rad=np.radians(0.25),
        )

        self.assertTrue(result.accepted)
        self.assertTrue(result.clipped)
        self.assertEqual(result.reason, "arm_desk_collision_clipped")
        self.assertGreater(result.selected_fraction, 0.54)
        self.assertLess(result.selected_fraction, 0.6)
        self.assertFalse(result.requested_check.safe)
        self.assertTrue(result.selected_check.safe)
        self.assertIsNotNone(result.blocked_check)
        self.assertIsNotNone(result.blocked_fraction)
        self.assertGreaterEqual(
            result.selected_check.minimum_distance_m,
            0.01 - 1e-9,
        )
        np.testing.assert_allclose(
            result.selected_target,
            result.selected_fraction * requested,
            atol=1e-12,
        )

    def test_safe_target_is_accepted_without_clipping(self):
        requested = np.asarray([-0.05, 0, 0, 0, 0, 0, 0])
        result = self.gate.filter_target(
            "R",
            requested,
            np.zeros(7),
        )

        self.assertTrue(result.accepted)
        self.assertFalse(result.clipped)
        self.assertIsNone(result.reason)
        self.assertEqual(result.selected_fraction, 1.0)
        self.assertIsNone(result.blocked_fraction)
        self.assertIsNone(result.blocked_check)
        self.assertGreaterEqual(result.collision_checks, 2)
        np.testing.assert_array_equal(result.selected_target, requested)

    def test_safe_endpoint_beyond_desk_does_not_skip_path_collision(self):
        requested = np.asarray([-0.5, 0, 0, 0, 0, 0, 0])
        result = self.gate.filter_target(
            "R",
            requested,
            np.zeros(7),
            path_samples=32,
            bisection_iterations=12,
            retreat_rad=np.radians(0.25),
        )

        self.assertTrue(result.requested_check.safe)
        self.assertTrue(result.accepted)
        self.assertTrue(result.clipped)
        self.assertEqual(result.reason, "arm_desk_collision_clipped")
        self.assertGreater(result.selected_fraction, 0.25)
        self.assertLess(result.selected_fraction, 0.3)
        self.assertTrue(result.selected_check.safe)
        self.assertLess(result.selected_fraction, 1.0)

    def test_no_safe_progress_holds_reference(self):
        reference = np.asarray([-0.14, 0, 0, 0, 0, 0, 0])
        requested = np.asarray([-0.2, 0, 0, 0, 0, 0, 0])
        result = self.gate.filter_target(
            "R",
            requested,
            reference,
            path_samples=8,
            bisection_iterations=8,
        )

        self.assertFalse(result.accepted)
        self.assertFalse(result.clipped)
        self.assertEqual(result.reason, "arm_desk_collision_hold")
        self.assertEqual(result.selected_fraction, 0.0)
        np.testing.assert_array_equal(result.selected_target, reference)

    def test_scratch_checks_do_not_modify_live_data(self):
        live = mujoco.MjData(self.model)
        live.qpos[:] = self.model.qpos0
        live.qpos[self.arm_qpos] = np.linspace(0.01, 0.07, 7)
        live.qvel[:] = np.linspace(-0.2, 0.2, self.model.nv)
        live.qacc_warmstart[:] = np.linspace(0.3, 0.7, self.model.nv)
        mujoco.mj_forward(self.model, live)
        before = {
            "qpos": live.qpos.copy(),
            "qvel": live.qvel.copy(),
            "qacc_warmstart": live.qacc_warmstart.copy(),
            "xpos": live.xpos.copy(),
        }

        self.gate.filter_target(
            "R",
            np.asarray([-0.25, 0, 0, 0, 0, 0, 0]),
            np.zeros(7),
            base_qpos=live.qpos,
        )

        for field, expected in before.items():
            np.testing.assert_array_equal(getattr(live, field), expected)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown side"):
            self.gate.check("L", np.zeros(7))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.gate.check("R", np.full(7, np.nan))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.gate.filter_target(
                "R",
                np.zeros(7),
                np.zeros(7),
                path_samples=0,
            )


if __name__ == "__main__":
    unittest.main()
