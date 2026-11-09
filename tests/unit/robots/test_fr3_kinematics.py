# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path

import numpy as np
import pytest

try:
    import mujoco
    from robosuite_env.robots.fr3.absolute_ik import AbsoluteFR3IK
    from robosuite_env.robots.fr3.runtime_ik import solve_and_pin_nullspace_ik
except ImportError:  # pragma: no cover - heavy dependency absent from CI
    mujoco = None
    AbsoluteFR3IK = None
    solve_and_pin_nullspace_ik = None


_SEVEN_DOF_ARM_XML = """
<mujoco model="seven_dof_test_arm">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <default>
    <joint type="hinge" limited="true" range="-2.7 2.7"
           damping="0.1" armature="0.01"/>
    <geom type="capsule" size="0.015" density="500"/>
  </default>
  <worldbody>
    <body name="link1">
      <joint name="joint1" axis="0 0 1"/>
      <geom fromto="0 0 0 0 0 0.18"/>
      <body name="link2" pos="0 0 0.18">
        <joint name="joint2" axis="0 1 0"/>
        <geom fromto="0 0 0 0.18 0 0"/>
        <body name="link3" pos="0.18 0 0">
          <joint name="joint3" axis="1 0 0"/>
          <geom fromto="0 0 0 0.16 0 0"/>
          <body name="link4" pos="0.16 0 0">
            <joint name="joint4" axis="0 1 0"/>
            <geom fromto="0 0 0 0.14 0 0"/>
            <body name="link5" pos="0.14 0 0">
              <joint name="joint5" axis="1 0 0"/>
              <geom fromto="0 0 0 0.12 0 0"/>
              <body name="link6" pos="0.12 0 0">
                <joint name="joint6" axis="0 1 0"/>
                <geom fromto="0 0 0 0.10 0 0"/>
                <body name="link7" pos="0.10 0 0">
                  <joint name="joint7" axis="1 0 0"/>
                  <geom fromto="0 0 0 0.08 0 0"/>
                  <body name="eef" pos="0.08 0 0">
                    <geom type="sphere" size="0.02"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


# joint1 and joint2 are coincident and coaxial, so only their sum affects the
# TCP.  This makes the redundant solution family exact and gives the
# null-space posture test a deterministic, non-flaky oracle.
_EXACT_REDUNDANCY_XML = """
<mujoco model="exact_redundancy_test_arm">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <default>
    <joint type="hinge" limited="true" range="-2.7 2.7"/>
    <geom type="capsule" size="0.015" density="500"/>
  </default>
  <worldbody>
    <body name="link1">
      <joint name="joint1" axis="0 0 1"/>
      <joint name="joint2" axis="0 0 1"/>
      <geom fromto="0 0 0 0 0 0.18"/>
      <body name="link3" pos="0 0 0.18">
        <joint name="joint3" axis="0 1 0"/>
        <geom fromto="0 0 0 0.18 0 0"/>
        <body name="link4" pos="0.18 0 0">
          <joint name="joint4" axis="1 0 0"/>
          <geom fromto="0 0 0 0.16 0 0"/>
          <body name="link5" pos="0.16 0 0">
            <joint name="joint5" axis="0 1 0"/>
            <geom fromto="0 0 0 0.14 0 0"/>
            <body name="link6" pos="0.14 0 0">
              <joint name="joint6" axis="1 0 0"/>
              <geom fromto="0 0 0 0.12 0 0"/>
              <body name="link7" pos="0.12 0 0">
                <joint name="joint7" axis="0 1 0"/>
                <geom fromto="0 0 0 0.10 0 0"/>
                <body name="eef" pos="0.10 0 0">
                  <geom type="sphere" size="0.02"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _quat_matrix(quaternion_wxyz):
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(
        matrix, np.ascontiguousarray(quaternion_wxyz, dtype=np.float64)
    )
    return matrix.reshape(3, 3)


@pytest.mark.usefixtures("mujoco")
@unittest.skipIf(mujoco is None, "MuJoCo is not installed")
class AbsoluteFR3IKTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(_SEVEN_DOF_ARM_XML)
        self.qpos_indices = np.asarray(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}"
                    )
                ]
                for i in range(1, 8)
            ],
            dtype=int,
        )
        self.dof_indices = np.asarray(
            [
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}"
                    )
                ]
                for i in range(1, 8)
            ],
            dtype=int,
        )
        self.joint_ranges = np.asarray(
            [
                self.model.jnt_range[
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}"
                    )
                ]
                for i in range(1, 8)
            ],
            dtype=np.float64,
        )
        self.eef_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "eef"
        )
        self.eef_to_tcp = np.asarray([0.025, -0.012, 0.04])
        self.arm_specs = {
            "R": {
                "qpos_indices": self.qpos_indices,
                "dof_indices": self.dof_indices,
                "joint_ranges": self.joint_ranges,
                "eef_body_id": self.eef_body_id,
            }
        }

    def _tcp_fk(self, q):
        data = mujoco.MjData(self.model)
        data.qpos[self.qpos_indices] = q
        mujoco.mj_forward(self.model, data)
        eef_quat = data.xquat[self.eef_body_id].copy()
        tcp_pos = (
            data.xpos[self.eef_body_id].copy()
            + _quat_matrix(eef_quat) @ self.eef_to_tcp
        )
        return tcp_pos, eef_quat

    def test_reachable_fk_roundtrip_and_quaternion_sign(self):
        target_q = np.asarray([0.3, -0.55, 0.4, -0.7, 0.35, 0.6, -0.25])
        seed_q = target_q + np.asarray(
            [-0.12, 0.14, -0.08, 0.1, -0.11, 0.09, 0.13]
        )
        target_pos, target_quat = self._tcp_fk(target_q)
        solver = AbsoluteFR3IK(
            self.model,
            self.arm_specs,
            self.eef_to_tcp,
            iterations=80,
            damping=1e-5,
        )

        positive = solver.solve("R", target_pos, target_quat, seed_q)
        negative = solver.solve("R", target_pos, -target_quat, seed_q)

        self.assertTrue(positive.converged, positive)
        self.assertTrue(negative.converged, negative)
        self.assertLess(positive.pos_error_mm, 0.1)
        self.assertLess(positive.ori_error_deg, 0.1)
        self.assertLess(negative.pos_error_mm, 0.1)
        self.assertLess(negative.ori_error_deg, 0.1)
        np.testing.assert_allclose(positive.q, negative.q, atol=1e-8)
        self.assertIsNone(positive.fallback_reason)
        self.assertGreaterEqual(positive.min_limit_margin_rad, 0.0)
        self.assertTrue(np.isfinite(positive.condition_number))

    def test_continuity_seed_is_used_and_hard_clipped_to_joint_limits(self):
        target_q = np.asarray([0.2, -0.3, 0.25, -0.5, 0.2, 0.4, -0.1])
        target_pos, target_quat = self._tcp_fk(target_q)
        continuity = target_q.copy()
        continuity[0] = 99.0
        solver = AbsoluteFR3IK(
            self.model,
            self.arm_specs,
            self.eef_to_tcp,
            iterations=1,
            max_step_deg=1.0,
        )

        result = solver.solve(
            "R",
            target_pos,
            target_quat,
            seed_q=np.full(7, -99.0),
            continuity_q=continuity,
        )

        self.assertTrue(np.all(result.q >= self.joint_ranges[:, 0]))
        self.assertTrue(np.all(result.q <= self.joint_ranges[:, 1]))
        self.assertGreaterEqual(result.min_limit_margin_rad, 0.0)
        self.assertTrue(np.all(np.isfinite(result.q)))

    def test_unreachable_target_is_finite_and_nonconverged(self):
        solver = AbsoluteFR3IK(
            self.model,
            self.arm_specs,
            self.eef_to_tcp,
            iterations=12,
        )

        result = solver.solve(
            "R",
            [10.0, -8.0, 12.0],
            [1.0, 0.0, 0.0, 0.0],
            np.zeros(7),
        )

        self.assertFalse(result.converged)
        self.assertIn(result.fallback_reason, {"max_iterations", "joint_limits"})
        self.assertTrue(np.all(np.isfinite(result.q)))
        self.assertTrue(np.isfinite(result.pos_error_mm))
        self.assertTrue(np.isfinite(result.ori_error_deg))
        self.assertGreater(result.pos_error_mm, 1000.0)
        self.assertTrue(np.all(result.q >= self.joint_ranges[:, 0]))
        self.assertTrue(np.all(result.q <= self.joint_ranges[:, 1]))

    def test_solver_does_not_modify_other_mjdata_for_same_model(self):
        live_data = mujoco.MjData(self.model)
        live_data.qpos[:] = np.linspace(-0.4, 0.4, self.model.nq)
        live_data.qvel[:] = np.linspace(0.7, -0.7, self.model.nv)
        live_data.qacc_warmstart[:] = np.linspace(1.0, 2.0, self.model.nv)
        live_data.ctrl[:] = np.linspace(-0.5, 0.5, self.model.nu)
        mujoco.mj_forward(self.model, live_data)
        snapshot = {
            "qpos": live_data.qpos.copy(),
            "qvel": live_data.qvel.copy(),
            "qacc_warmstart": live_data.qacc_warmstart.copy(),
            "ctrl": live_data.ctrl.copy(),
            "time": float(live_data.time),
            "xpos": live_data.xpos.copy(),
        }
        target_q = np.asarray([0.4, -0.6, 0.3, -0.8, 0.2, 0.5, -0.2])
        target_pos, target_quat = self._tcp_fk(target_q)
        solver = AbsoluteFR3IK(
            self.model, self.arm_specs, self.eef_to_tcp, iterations=60
        )

        solver.solve("R", target_pos, target_quat, np.zeros(7))

        for name in ("qpos", "qvel", "qacc_warmstart", "ctrl", "xpos"):
            np.testing.assert_array_equal(getattr(live_data, name), snapshot[name])
        self.assertEqual(live_data.time, snapshot["time"])

    def test_nonfinite_and_zero_quaternion_inputs_are_rejected(self):
        solver = AbsoluteFR3IK(
            self.model, self.arm_specs, self.eef_to_tcp
        )

        with self.assertRaisesRegex(ValueError, "finite"):
            solver.solve(
                "R", [np.nan, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], np.zeros(7)
            )
        with self.assertRaisesRegex(ValueError, "non-zero norm"):
            solver.solve("R", [0.0, 0.0, 0.0], np.zeros(4), np.zeros(7))
        with self.assertRaisesRegex(ValueError, "finite"):
            solver.solve(
                "R",
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                np.full(7, np.inf),
            )

    def test_runtime_adapter_pins_the_selected_solution(self):
        target_q = np.asarray([0.3, -0.55, 0.4, -0.7, 0.35, 0.6, -0.25])
        target_pos, target_quat = self._tcp_fk(target_q)
        live_data = mujoco.MjData(self.model)
        live_data.qpos[self.qpos_indices] = target_q + np.asarray(
            [-0.12, 0.14, -0.08, 0.1, -0.11, 0.09, 0.13]
        )
        mujoco.mj_forward(self.model, live_data)
        solver = AbsoluteFR3IK(
            self.model,
            self.arm_specs,
            self.eef_to_tcp,
            iterations=80,
            nullspace_gain=0.2,
            home_q=np.zeros(7),
        )

        def error_fn(_side, position, quaternion):
            eef_quaternion = live_data.xquat[self.eef_body_id].copy()
            actual_position = (
                live_data.xpos[self.eef_body_id].copy()
                + _quat_matrix(eef_quaternion) @ self.eef_to_tcp
            )
            position_error = float(np.linalg.norm(position - actual_position))
            orientation_error = 2.0 * np.arccos(
                np.clip(abs(np.dot(quaternion, eef_quaternion)), 0.0, 1.0)
            )
            return position_error, float(orientation_error)

        q, error, converged = solve_and_pin_nullspace_ik(
            solver,
            self.model,
            live_data,
            "R",
            {"qi": self.qpos_indices},
            target_pos,
            target_quat,
            error_fn,
        )

        self.assertTrue(converged)
        np.testing.assert_array_equal(live_data.qpos[self.qpos_indices], q)
        self.assertLess(error[0], 5e-4)
        self.assertLess(error[1], np.radians(0.5))

    def test_nullspace_selects_home_when_seed_already_satisfies_tcp(self):
        model = mujoco.MjModel.from_xml_string(_EXACT_REDUNDANCY_XML)
        joint_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index}"
                )
                for index in range(1, 8)
            ],
            dtype=int,
        )
        qpos_indices = model.jnt_qposadr[joint_ids].astype(int)
        dof_indices = model.jnt_dofadr[joint_ids].astype(int)
        joint_ranges = model.jnt_range[joint_ids].copy()
        eef_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "eef"
        )
        arm_specs = {
            "R": {
                "qpos_indices": qpos_indices,
                "dof_indices": dof_indices,
                "joint_ranges": joint_ranges,
                "eef_body_id": eef_body_id,
            }
        }

        seed = np.asarray([0.8, -0.8, 0.35, -0.55, 0.45, 0.6, -0.4])
        home = np.asarray([0.0, 0.0, 0.35, -0.55, 0.45, 0.6, -0.4])
        fk_data = mujoco.MjData(model)
        fk_data.qpos[qpos_indices] = seed
        mujoco.mj_forward(model, fk_data)
        target_position = fk_data.xpos[eef_body_id].copy()
        target_quaternion = fk_data.xquat[eef_body_id].copy()
        solver = AbsoluteFR3IK(
            model,
            arm_specs,
            np.zeros(3),
            iterations=80,
            damping=1e-5,
            nullspace_gain=0.25,
            home_q=home,
        )

        result = solver.solve(
            "R",
            target_position,
            target_quaternion,
            seed_q=np.full(7, -1.5),
            continuity_q=seed,
        )

        self.assertTrue(result.converged, result)
        self.assertGreater(result.iterations, 0)
        self.assertLess(result.pos_error_mm, 1e-6)
        self.assertLess(result.ori_error_deg, 1e-6)
        self.assertLess(
            result.home_distance_rad,
            1e-3 * float(np.linalg.norm(seed - home)),
        )
        np.testing.assert_allclose(result.q[2:], home[2:], atol=1e-10)

    def test_structural_projector_preserves_primary_task(self):
        solver = AbsoluteFR3IK(
            self.model,
            self.arm_specs,
            self.eef_to_tcp,
        )
        spec = solver.arm_specs["R"]
        q = np.asarray([0.3, -0.55, 0.4, -0.7, 0.35, 0.6, -0.25])
        solver._set_arm(spec, q)
        jacobian = solver._arm_jacobian(spec)

        _, nullspace = solver._inverse_and_structural_nullspace(jacobian)

        np.testing.assert_allclose(nullspace, nullspace.T, atol=1e-12)
        np.testing.assert_allclose(
            nullspace @ nullspace, nullspace, atol=1e-12
        )
        np.testing.assert_allclose(
            jacobian @ nullspace, np.zeros((6, 7)), atol=1e-12
        )

    def test_positive_nullspace_gain_requires_valid_home(self):
        with self.assertRaisesRegex(ValueError, "home_q is required"):
            AbsoluteFR3IK(
                self.model,
                self.arm_specs,
                self.eef_to_tcp,
                nullspace_gain=0.1,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            AbsoluteFR3IK(
                self.model,
                self.arm_specs,
                self.eef_to_tcp,
                nullspace_gain=0.1,
                home_q=np.full(7, np.nan),
            )
        with self.assertRaisesRegex(ValueError, "outside joint limits"):
            AbsoluteFR3IK(
                self.model,
                self.arm_specs,
                self.eef_to_tcp,
                nullspace_gain=0.1,
                home_q=np.full(7, 99.0),
            )

if __name__ == "__main__":
    unittest.main()


import json
import unittest

import pytest

try:
    from robosuite_env.robots.fr3 import model as fr3_robot
    from robosuite_env.robots.fr3.runtime_ik import (
        controller_path_summary,
        controller_summary_name,
    )
except ImportError:  # pragma: no cover - heavy dependencies absent from CI
    fr3_robot = None
    controller_path_summary = None
    controller_summary_name = None


@pytest.mark.usefixtures("mujoco", "robosuite")
class Fr3ControllerConfigTest(unittest.TestCase):
    def test_canonical_home_pose_is_stable_and_inside_fr3_limits(self):
        expected = np.asarray(
            [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]
        )
        np.testing.assert_array_equal(fr3_robot.FR3_HOME_QPOS, expected)
        self.assertFalse(fr3_robot.FR3_HOME_QPOS.flags.writeable)

    def test_joint_ik_controller_accepts_absolute_fr3_joint_targets(self):
        path = Path(fr3_robot.FR3_CONTROLLER_CONFIGS["joint_ik"])
        config = json.loads(path.read_text(encoding="utf-8"))
        arm = config["body_parts"]["arms"]["right"]

        self.assertEqual(arm["type"], "JOINT_POSITION")
        self.assertEqual(arm["input_type"], "absolute")
        self.assertEqual(len(arm["input_min"]), 7)
        self.assertEqual(len(arm["input_max"]), 7)
        self.assertEqual(arm["kp"], 900)
        loaded = fr3_robot.load_fr3_controller_config("joint_ik")
        loaded_arm = loaded["body_parts"]["right"]
        self.assertEqual(loaded_arm["type"], "JOINT_POSITION")
        self.assertEqual(loaded_arm["input_type"], "absolute")
        self.assertEqual(loaded_arm["kp"], 900)
        self.assertEqual(loaded_arm["damping_ratio"], 1)
        self.assertNotIn("kd", loaded_arm)

        overridden = fr3_robot.load_fr3_controller_config(
            "joint_ik",
            joint_kp=400,
            joint_kd=80,
        )
        overridden_arm = overridden["body_parts"]["right"]
        self.assertEqual(overridden_arm["kp"], 400.0)
        self.assertEqual(overridden_arm["kd"], 80.0)
        self.assertEqual(overridden_arm["damping_ratio"], 1)

    def test_unknown_controller_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown FR3 controller mode"):
            fr3_robot.load_fr3_controller_config("unknown")

    def test_controller_summaries_match_all_fr3_routes(self):
        expected = {
            ("kinematic", "joint_ik"): (
                "fr3_mujoco_home_nullspace_ik",
                "mujoco_home_nullspace_ik_to_direct_qpos_pin",
            ),
            ("dynamic", "joint_ik"): (
                "fr3_absolute_home_nullspace_joint_ik",
                "absolute_whole_pose_ik_to_stateful_jerk_limited_joint_position_pd",
            ),
        }

        for selection, summaries in expected.items():
            with self.subTest(physics=selection[0], controller=selection[1]):
                self.assertEqual(controller_summary_name(*selection), summaries[0])
                self.assertEqual(controller_path_summary(*selection), summaries[1])

    def test_nullspace_off_changes_only_joint_ik_names(self):
        from robosuite_env.robots.fr3.runtime_ik import effective_nullspace_gain

        self.assertEqual(
            controller_summary_name("dynamic", "joint_ik", "off"),
            "fr3_absolute_joint_ik",
        )
        self.assertEqual(
            controller_summary_name("kinematic", "joint_ik", "off"),
            "fr3_mujoco_ik",
        )
        self.assertEqual(effective_nullspace_gain("home", 0.2), 0.2)
        self.assertEqual(effective_nullspace_gain("off", 0.2), 0.0)
        with self.assertRaises(ValueError):
            effective_nullspace_gain("maybe", 0.2)
        with self.assertRaises(ValueError):
            controller_summary_name("dynamic", "joint_ik", "maybe")
        with self.assertRaises(ValueError):
            controller_summary_name("dynamic", "osc")


if __name__ == "__main__":
    unittest.main()


import unittest


from robosuite_env.robots.fr3.initial_pose import (
    GRIPPER_JOINT_ORDER,
    GRIPPER_MAX_RAD,
    JOINT_ORDER,
    joint_ready_gate_report,
    parse_initial_arm_joint_pose_payload,
    resolve_initial_grip_targets,
    validate_joint_ranges,
)


def _payload():
    return {
        "schema_version": 1,
        "units": "rad",
        "joint_order": list(JOINT_ORDER),
        "left_arm": [1.0, 0.2, -0.1, -2.4, 1.3, 3.9, 0.4],
        "right_arm": [-1.0, 0.3, 0.4, -2.5, -1.0, 3.7, 1.1],
    }


def _v2_payload(**gripper):
    payload = _payload()
    payload["schema_version"] = 2
    payload["gripper"] = {
        "coordinate": "raw",
        "joint_order": list(GRIPPER_JOINT_ORDER),
        "left": 0.7,
        "right": 0.7,
        **gripper,
    }
    return payload


class InitialArmJointPoseTest(unittest.TestCase):
    def test_human_left_right_maps_to_sim_L_R_without_wrapping_j6(self):
        pose = parse_initial_arm_joint_pose_payload(
            _payload(), source_path="test-data/pose.json", source_sha256="abc"
        )
        np.testing.assert_allclose(pose.q_by_arm["L"], _payload()["left_arm"])
        np.testing.assert_allclose(pose.q_by_arm["R"], _payload()["right_arm"])
        self.assertEqual(pose.q_by_arm["L"][5], 3.9)
        self.assertEqual(pose.q_by_arm["R"][5], 3.7)

    def test_schema_is_strict(self):
        for key in ("left_arm", "right_arm", "joint_order", "units"):
            payload = _payload()
            payload.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse_initial_arm_joint_pose_payload(
                    payload, source_path="pose.json", source_sha256="abc"
                )
        payload = _payload()
        payload["extra"] = 1
        with self.assertRaises(ValueError):
            parse_initial_arm_joint_pose_payload(
                payload, source_path="pose.json", source_sha256="abc"
            )

    def test_rejects_nonfinite_and_boolean_values(self):
        for bad in (float("nan"), True):
            payload = _payload()
            payload["left_arm"][2] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_initial_arm_joint_pose_payload(
                    payload, source_path="pose.json", source_sha256="abc"
                )

    def test_joint_range_validation(self):
        pose = parse_initial_arm_joint_pose_payload(
            _payload(), source_path="pose.json", source_sha256="abc"
        )
        ranges = {
            "R": np.tile([-5.0, 5.0], (7, 1)),
            "L": np.tile([-5.0, 5.0], (7, 1)),
        }
        validate_joint_ranges(pose.q_by_arm, ranges)
        ranges["L"][5] = [0.4, 3.8]
        with self.assertRaisesRegex(ValueError, "L.*j6"):
            validate_joint_ranges(pose.q_by_arm, ranges)

    def test_ready_gate_reports_each_joint_and_failure(self):
        pose = parse_initial_arm_joint_pose_payload(
            _payload(), source_path="pose.json", source_sha256="abc"
        )
        actual = {key: value.copy() for key, value in pose.q_by_arm.items()}
        actual["R"][3] += np.radians(0.2)
        passed = joint_ready_gate_report(
            actual, pose.q_by_arm, tolerance_deg=0.5
        )
        self.assertTrue(passed["passed"])
        self.assertAlmostEqual(passed["arms"]["R"]["max_error_deg"], 0.2)
        failed = joint_ready_gate_report(
            actual, pose.q_by_arm, tolerance_deg=0.1
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["arms"]["R"]["max_error_joint"], 4)


class InitialArmJointPoseSchemaV2Test(unittest.TestCase):
    """schema_version 2 adds one optional gripper block and nothing else."""

    def _parse(self, payload):
        return parse_initial_arm_joint_pose_payload(
            payload, source_path="pose.json", source_sha256="abc"
        )

    def test_schema_1_stays_valid_and_declares_no_gripper(self):
        pose = self._parse(_payload())

        self.assertEqual(pose.schema_version, 1)
        self.assertIsNone(pose.grip_by_arm)
        self.assertIsNone(resolve_initial_grip_targets(pose))

    def test_schema_1_still_rejects_the_gripper_block(self):
        payload = _v2_payload()
        payload["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            self._parse(payload)

    def test_schema_2_without_a_gripper_block_declares_no_gripper(self):
        payload = _payload()
        payload["schema_version"] = 2
        pose = self._parse(payload)

        self.assertEqual(pose.schema_version, 2)
        self.assertIsNone(pose.grip_by_arm)

    def test_schema_2_maps_the_gripper_block_to_sim_L_R(self):
        pose = self._parse(_v2_payload(left=0.7, right=0.25))

        self.assertEqual(pose.grip_by_arm, {"L": 0.7, "R": 0.25})
        self.assertEqual(
            resolve_initial_grip_targets(pose), {"R": 0.25, "L": 0.7}
        )

    def test_a_mimicgen_replay_refuses_a_second_initial_gripper(self):
        pose = self._parse(_v2_payload())
        with self.assertRaisesRegex(ValueError, "MimicGen"):
            resolve_initial_grip_targets(
                pose, mimicgen_init_episode="episode_000000"
            )
        # A schema-1 file declares none, so MimicGen keeps working.
        self.assertIsNone(
            resolve_initial_grip_targets(
                self._parse(_payload()), mimicgen_init_episode="episode_000000"
            )
        )

    def test_only_the_raw_motor_coordinate_is_accepted(self):
        with self.assertRaisesRegex(ValueError, "gripper.coordinate"):
            self._parse(_v2_payload(coordinate="calibrated"))

    def test_gripper_joint_order_is_pinned_to_the_real_motor_names(self):
        self.assertEqual(
            GRIPPER_JOINT_ORDER,
            ("hand_left_motor_joint", "hand_right_motor_joint"),
        )
        with self.assertRaisesRegex(ValueError, "gripper.joint_order"):
            self._parse(
                _v2_payload(joint_order=list(reversed(GRIPPER_JOINT_ORDER)))
            )

    def test_gripper_angles_outside_the_runtime_clamp_are_rejected(self):
        for bad in (-0.01, GRIPPER_MAX_RAD + 1e-6, float("nan"), True, "0.7"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self._parse(_v2_payload(left=bad))

    def test_gripper_block_keys_are_strict(self):
        payload = _v2_payload()
        payload["gripper"]["extra"] = 1
        with self.assertRaisesRegex(ValueError, "gripper unknown keys"):
            self._parse(payload)
        payload = _v2_payload()
        del payload["gripper"]["right"]
        with self.assertRaisesRegex(ValueError, "gripper missing keys"):
            self._parse(payload)

    def test_unsupported_schema_versions_are_rejected(self):
        for version in (0, 3, "2", None, True, 2.0):
            payload = _payload()
            payload["schema_version"] = version
            with self.subTest(version=version), self.assertRaisesRegex(
                ValueError, "schema_version must be one of"
            ):
                self._parse(payload)


if __name__ == "__main__":
    unittest.main()


import unittest

from robosuite_env.policy.policy_io import FR3_EEF_TO_HAND_ROOT_M
from robosuite_env.robots.profiles import get_robot_profile, normalize_robot_name


class RobotProfilesTest(unittest.TestCase):
    def test_fr3_preserves_two_independent_arm_mapping(self):
        profile = get_robot_profile("FR3")

        self.assertEqual(profile.name, "fr3")
        self.assertEqual(profile.robosuite_robots, ("FR3", "FR3"))
        self.assertEqual(
            profile.gripper_types,
            ("YubiGripper", "YubiGripperLeft"),
        )
        self.assertEqual(profile.arms["R"].joint_names[0], "robot0_joint1")
        self.assertEqual(profile.arms["L"].joint_names[0], "robot1_joint1")
        self.assertEqual(profile.arms["R"].policy_offset_m, FR3_EEF_TO_HAND_ROOT_M)
        self.assertEqual(profile.arms["L"].policy_offset_m, FR3_EEF_TO_HAND_ROOT_M)
        self.assertEqual(profile.wrist_prefixes["R"], "gripper0_right")
        self.assertEqual(profile.wrist_prefixes["L"], "gripper1_right")
        self.assertEqual(profile.controller_action_dim_for("joint_ik"), 16)
        with self.assertRaises(ValueError):
            profile.controller_action_dim_for("osc")
        self.assertEqual(
            profile.runtime_gripper_types(calibrated_yubi=True),
            ("YubiGripper", "YubiGripperLeft"),
        )

    def test_robot_name_normalization_is_cli_case_insensitive(self):
        # Every supported embodiment normalizes from any CLI spelling, and
        # nothing else is accepted.
        self.assertEqual(normalize_robot_name(" FR3 "), "fr3")
        self.assertEqual(normalize_robot_name("fr3"), "fr3")
        self.assertEqual(normalize_robot_name(" FR3_DUO "), "fr3_duo")
        for unknown in ("panda", "ur5", "fr3x", ""):
            with self.assertRaisesRegex(ValueError, "unsupported robot"):
                normalize_robot_name(unknown)


if __name__ == "__main__":
    unittest.main()
