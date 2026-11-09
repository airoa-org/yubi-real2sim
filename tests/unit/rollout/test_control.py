# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np
import pytest

from robosuite_env.rollout.control.joint_trajectory import (
    MINIMUM_JERK_MAX_ACCELERATION_COEFFICIENT,
    MINIMUM_JERK_MAX_VELOCITY_COEFFICIENT,
    RUCKIG_VERSION,
    StatefulJerkLimitedTrajectory,
    bounded_minimum_jerk_segment,
    minimum_jerk_blend,
)

_RUCKIG_AVAILABLE = importlib.util.find_spec("ruckig") is not None


class JointTrajectoryTest(unittest.TestCase):
    def test_minimum_jerk_endpoints_and_monotonicity(self):
        values = minimum_jerk_blend(np.linspace(0.0, 1.0, 101))

        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 1.0)
        self.assertTrue(np.all(np.diff(values) >= 0.0))

    def test_segment_has_requested_number_of_commands_and_exact_endpoint(self):
        segment = bounded_minimum_jerk_segment(
            np.zeros(3),
            np.array([0.01, -0.02, 0.03]),
            steps=5,
            duration_s=0.1,
            max_velocity_rad_s=2.0,
            max_acceleration_rad_s2=100.0,
        )

        self.assertEqual(segment.commands.shape, (5, 3))
        np.testing.assert_allclose(segment.commands[-1], segment.executed_goal)
        np.testing.assert_allclose(segment.executed_goal, segment.requested_goal)

    def test_common_scale_respects_continuous_velocity_and_acceleration(self):
        max_velocity = 0.6
        max_acceleration = 8.0
        segment = bounded_minimum_jerk_segment(
            np.zeros(2),
            np.array([0.4, -0.2]),
            steps=5,
            duration_s=0.1,
            max_velocity_rad_s=max_velocity,
            max_acceleration_rad_s2=max_acceleration,
        )

        self.assertLess(segment.scale, 1.0)
        self.assertLessEqual(segment.max_velocity_rad_s, max_velocity + 1e-12)
        self.assertLessEqual(
            segment.max_acceleration_rad_s2, max_acceleration + 1e-12
        )
        ratios = segment.executed_goal / segment.requested_goal
        np.testing.assert_allclose(ratios, np.full(2, segment.scale))

    def test_reported_extrema_match_quintic_coefficients(self):
        delta = 0.02
        duration = 0.1
        segment = bounded_minimum_jerk_segment(
            [0.0],
            [delta],
            steps=5,
            duration_s=duration,
            max_velocity_rad_s=100.0,
            max_acceleration_rad_s2=1000.0,
        )

        self.assertAlmostEqual(
            segment.max_velocity_rad_s,
            MINIMUM_JERK_MAX_VELOCITY_COEFFICIENT * delta / duration,
        )
        self.assertAlmostEqual(
            segment.max_acceleration_rad_s2,
            MINIMUM_JERK_MAX_ACCELERATION_COEFFICIENT * delta / duration**2,
        )

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "steps"):
            bounded_minimum_jerk_segment(
                [0.0],
                [1.0],
                steps=0,
                duration_s=0.1,
                max_velocity_rad_s=1.0,
                max_acceleration_rad_s2=1.0,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            bounded_minimum_jerk_segment(
                [0.0],
                [np.nan],
                steps=1,
                duration_s=0.1,
                max_velocity_rad_s=1.0,
                max_acceleration_rad_s2=1.0,
            )


@pytest.mark.usefixtures("ruckig")
@unittest.skipUnless(_RUCKIG_AVAILABLE, f"ruckig=={RUCKIG_VERSION} is not installed")
class StatefulJointTrajectoryTest(unittest.TestCase):
    """StatefulJerkLimitedTrajectory wraps ruckig, which the CI lockfile omits."""

    def test_stateful_trajectory_carries_derivatives_across_retarget(self):
        trajectory = StatefulJerkLimitedTrajectory(
            np.zeros(3),
            delta_time_s=0.02,
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=4.0,
            max_jerk_rad_s3=20.0,
            position_lower=np.full(3, -2.0),
            position_upper=np.full(3, 2.0),
        )
        self.assertEqual(
            trajectory._ruckig_module.__version__,
            RUCKIG_VERSION,
        )
        self.assertTrue(trajectory.set_target([0.8, -0.4, 0.2]).accepted)
        samples = [trajectory.step() for _ in range(20)]
        before = (
            trajectory.position.copy(),
            trajectory.velocity.copy(),
            trajectory.acceleration.copy(),
        )
        update = trajectory.set_target([-0.5, 0.3, -0.1])
        self.assertTrue(update.accepted)
        np.testing.assert_array_equal(trajectory.position, before[0])
        np.testing.assert_array_equal(trajectory.velocity, before[1])
        np.testing.assert_array_equal(trajectory.acceleration, before[2])
        samples.extend(trajectory.step() for _ in range(300))

        for sample in samples:
            self.assertLessEqual(
                float(np.max(np.abs(sample.velocity))),
                1.0 + 1e-9,
            )
            self.assertLessEqual(
                float(np.max(np.abs(sample.acceleration))),
                4.0 + 1e-8,
            )
            self.assertLessEqual(
                float(np.max(np.abs(sample.jerk))),
                20.0 + 1e-7,
            )
            self.assertTrue(np.all(sample.position >= -2.0 - 1e-9))
            self.assertTrue(np.all(sample.position <= 2.0 + 1e-9))
        np.testing.assert_allclose(
            trajectory.position,
            [-0.5, 0.3, -0.1],
            atol=1e-10,
        )
        np.testing.assert_allclose(trajectory.velocity, 0.0, atol=1e-10)
        np.testing.assert_allclose(
            trajectory.acceleration, 0.0, atol=1e-10
        )

    def test_stateful_trajectory_rejects_unsafe_target_and_keeps_plan(self):
        trajectory = StatefulJerkLimitedTrajectory(
            [0.0],
            delta_time_s=0.02,
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=2.0,
            max_jerk_rad_s3=10.0,
            position_lower=[-1.0],
            position_upper=[1.0],
        )
        self.assertTrue(trajectory.set_target([0.8]).accepted)
        for _ in range(10):
            trajectory.step()
        active_before = trajectory.active_target.copy()
        update = trajectory.set_target([1.1])

        self.assertFalse(update.accepted)
        self.assertEqual(update.reason, "target_outside_hard_joint_limits")
        np.testing.assert_array_equal(
            trajectory.active_target, active_before
        )
        self.assertEqual(trajectory.target_rejection_count, 1)
        sample = trajectory.step()
        np.testing.assert_array_equal(sample.active_target, active_before)
        self.assertLessEqual(sample.predicted_position_max[0], 1.0 + 1e-9)

    def test_collision_preview_rejection_keeps_validated_remainder(self):
        def make_trajectory():
            return StatefulJerkLimitedTrajectory(
                [0.0],
                delta_time_s=0.02,
                max_velocity_rad_s=0.7,
                max_acceleration_rad_s2=12.0,
                max_jerk_rad_s3=120.0,
                position_lower=[-1.0],
                position_upper=[1.0],
            )

        rejected = make_trajectory()
        control = make_trajectory()
        full_preview = rejected.preview_target(
            [0.8],
            sample_period_s=0.02,
            horizon_s=None,
        )
        self.assertTrue(full_preview.update.accepted)
        for trajectory in (rejected, control):
            self.assertTrue(trajectory.set_target([0.8]).accepted)
            for _ in range(10):
                trajectory.step()

        validated_remainder = full_preview.positions[10:]
        unsafe_replan = rejected.preview_target(
            [-0.8],
            sample_period_s=0.02,
            horizon_s=None,
        )
        self.assertTrue(unsafe_replan.update.accepted)
        self.assertGreater(len(validated_remainder), 0)
        self.assertTrue(np.all(validated_remainder[:, 0] >= 0.0))
        self.assertTrue(np.any(unsafe_replan.positions[:, 0] < 0.0))
        before = {
            name: getattr(rejected, name).copy()
            for name in ("position", "velocity", "acceleration", "active_target")
        }

        update = rejected.reject_target(
            [-0.8],
            reason="trajectory_preview_collision_hold",
        )

        self.assertFalse(update.accepted)
        self.assertEqual(
            update.reason,
            "trajectory_preview_collision_hold",
        )
        np.testing.assert_array_equal(update.requested_target, [-0.8])
        np.testing.assert_array_equal(update.active_target, before["active_target"])
        for name, expected in before.items():
            np.testing.assert_array_equal(getattr(rejected, name), expected)
        self.assertEqual(rejected.target_rejection_count, 1)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            rejected.reject_target([0.0], reason="")
        for expected_position in validated_remainder:
            rejected_sample = rejected.step()
            control_sample = control.step()
            np.testing.assert_allclose(
                rejected_sample.position,
                expected_position,
                atol=1e-15,
                rtol=0.0,
            )
            for field in (
                "position",
                "velocity",
                "acceleration",
                "jerk",
                "active_target",
                "predicted_position_min",
                "predicted_position_max",
            ):
                np.testing.assert_array_equal(
                    getattr(rejected_sample, field),
                    getattr(control_sample, field),
                )
            self.assertFalse(rejected_sample.new_calculation)
            self.assertFalse(control_sample.new_calculation)
            self.assertGreaterEqual(float(rejected_sample.position[0]), 0.0)
        self.assertTrue(rejected_sample.finished)
        np.testing.assert_allclose(rejected.position, [0.8], atol=1e-12)

    def test_target_preview_matches_online_steps_without_mutating_state(self):
        trajectory = StatefulJerkLimitedTrajectory(
            [0.0, 0.0],
            delta_time_s=0.02,
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=2.0,
            max_jerk_rad_s3=10.0,
            position_lower=[-1.0, -1.0],
            position_upper=[1.0, 1.0],
        )
        before = {
            name: getattr(trajectory, name).copy()
            for name in ("position", "velocity", "acceleration", "active_target")
        }

        preview = trajectory.preview_target(
            [0.8, -0.4],
            sample_period_s=0.02,
            horizon_s=0.1,
        )

        self.assertTrue(preview.update.accepted)
        np.testing.assert_allclose(
            preview.sample_times_s,
            [0.02, 0.04, 0.06, 0.08, 0.1],
            atol=1e-15,
        )
        self.assertEqual(preview.positions.shape, (5, 2))
        for name, expected in before.items():
            np.testing.assert_array_equal(getattr(trajectory, name), expected)

        self.assertTrue(trajectory.set_target([0.8, -0.4]).accepted)
        for expected_position in preview.positions:
            sample = trajectory.step()
            np.testing.assert_allclose(
                sample.position,
                expected_position,
                atol=1e-15,
                rtol=0.0,
            )

        with self.assertRaisesRegex(ValueError, "positive"):
            trajectory.preview_target(
                [0.0, 0.0],
                sample_period_s=0.0,
            )

    def test_full_preview_covers_nonzero_velocity_brake_past_one_tick(self):
        trajectory = StatefulJerkLimitedTrajectory(
            [0.0],
            delta_time_s=0.02,
            max_velocity_rad_s=0.7,
            max_acceleration_rad_s2=12.0,
            max_jerk_rad_s3=120.0,
            position_lower=[-1.0],
            position_upper=[1.0],
        )
        self.assertTrue(trajectory.set_target([0.8]).accepted)
        for _ in range(5):
            trajectory.step()
        self.assertGreater(abs(float(trajectory.velocity[0])), 0.0)
        brake_target = trajectory.position.copy()
        before = {
            name: getattr(trajectory, name).copy()
            for name in ("position", "velocity", "acceleration", "active_target")
        }

        preview = trajectory.preview_target(
            brake_target,
            sample_period_s=0.02,
            horizon_s=None,
        )

        self.assertTrue(preview.update.accepted)
        self.assertGreater(preview.update.trajectory_duration_s, 0.1)
        self.assertGreater(len(preview.positions), 5)
        self.assertGreater(
            float(np.max(preview.positions[:, 0])),
            float(brake_target[0]) + 1e-12,
        )
        self.assertTrue(np.any(np.diff(preview.positions[:, 0]) < -1e-12))
        np.testing.assert_allclose(
            preview.positions[-1],
            brake_target,
            atol=1e-12,
        )
        self.assertAlmostEqual(
            float(preview.sample_times_s[-1]),
            float(preview.update.trajectory_duration_s),
            places=12,
        )
        for name, expected in before.items():
            np.testing.assert_array_equal(getattr(trajectory, name), expected)

        self.assertTrue(trajectory.set_target(brake_target).accepted)
        for expected_position in preview.positions:
            sample = trajectory.step()
            np.testing.assert_allclose(
                sample.position,
                expected_position,
                atol=1e-15,
                rtol=0.0,
            )
        self.assertTrue(sample.finished)
        np.testing.assert_allclose(sample.velocity, [0.0], atol=1e-12)
        np.testing.assert_allclose(sample.acceleration, [0.0], atol=1e-12)

    def test_stateful_trajectory_rejects_in_range_target_with_unsafe_extrema(
        self,
    ):
        trajectory = StatefulJerkLimitedTrajectory(
            [-0.4715034642592388, 0.7556868105081043],
            velocity=[0.7620345158512221, 0.9627147159561722],
            acceleration=[1.8608198588066918, -1.3469106047593815],
            delta_time_s=0.02,
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=2.0,
            max_jerk_rad_s3=10.0,
            position_lower=[-1.0, -1.0],
            position_upper=[1.0, 1.0],
        )
        active_before = trajectory.active_target.copy()
        update = trajectory.set_target(
            [-0.42430926042530626, 0.9937910252724493]
        )

        self.assertFalse(update.accepted)
        self.assertEqual(
            update.reason,
            "predicted_position_extrema_outside_hard_limits",
        )
        self.assertTrue(np.all(update.requested_target <= 1.0))
        self.assertGreater(update.predicted_position_max[1], 1.0)
        np.testing.assert_array_equal(
            trajectory.active_target, active_before
        )
        samples = [trajectory.step() for _ in range(300)]
        self.assertTrue(
            all(np.all(sample.position <= 1.0 + 1e-9) for sample in samples)
        )
        self.assertTrue(
            all(np.all(sample.position >= -1.0 - 1e-9) for sample in samples)
        )
        np.testing.assert_allclose(
            trajectory.position,
            active_before,
            atol=1e-10,
        )

    def test_stateful_trajectory_rejects_unsafe_initial_braking_state(self):
        cases = (
            ([0.9], [1.0], [0.0]),
            (
                [0.6031661142487459],
                [0.6838266831208479],
                [1.913365620785981],
            ),
        )
        for position, velocity, acceleration in cases:
            with self.subTest(
                position=position,
                velocity=velocity,
                acceleration=acceleration,
            ):
                with self.assertRaisesRegex(
                    ValueError, "no joint-limit-safe braking trajectory"
                ):
                    StatefulJerkLimitedTrajectory(
                        position,
                        velocity=velocity,
                        acceleration=acceleration,
                        delta_time_s=0.02,
                        max_velocity_rad_s=1.0,
                        max_acceleration_rad_s2=2.0,
                        max_jerk_rad_s3=10.0,
                        position_lower=[-1.0],
                        position_upper=[1.0],
                    )

    def test_stateful_extrema_include_interior_velocity_root(self):
        trajectory = StatefulJerkLimitedTrajectory(
            [0.6031661142487459],
            velocity=[0.6838266831208479],
            acceleration=[1.913365620785981],
            delta_time_s=0.02,
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=2.0,
            max_jerk_rad_s3=10.0,
            position_lower=[-10.0],
            position_upper=[10.0],
        )

        self.assertAlmostEqual(
            trajectory.last_target_update.predicted_position_max[0],
            1.0285786725945836,
            places=12,
        )

    def test_zero_duration_brake_profile_does_not_add_zero_position(self):
        trajectory = StatefulJerkLimitedTrajectory(
            [-1.0],
            delta_time_s=0.02,
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=2.0,
            max_jerk_rad_s3=10.0,
            position_lower=[-2.0],
            position_upper=[-0.1],
        )
        update = trajectory.set_target([-0.5])

        self.assertTrue(update.accepted)
        self.assertLess(update.predicted_position_max[0], 0.0)
        for _ in range(300):
            trajectory.step()
        np.testing.assert_allclose(
            trajectory.position,
            [-0.5],
            atol=1e-10,
        )

    def test_nonzero_brake_profile_uses_continuous_real_position(self):
        trajectory = StatefulJerkLimitedTrajectory(
            [0.12021345201537781],
            velocity=[0.9901930104706482],
            acceleration=[1.7957746997510613],
            delta_time_s=0.02,
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=2.0,
            max_jerk_rad_s3=10.0,
            position_lower=[0.1],
            position_upper=[1.0],
        )

        self.assertGreaterEqual(
            trajectory.last_target_update.predicted_position_min[0],
            0.1,
        )
        self.assertAlmostEqual(
            trajectory.last_target_update.predicted_position_max[0],
            0.7605929199208658,
            places=12,
        )

    def test_stateful_trajectory_is_deterministic_under_rapid_retargeting(self):
        def run():
            trajectory = StatefulJerkLimitedTrajectory(
                np.zeros(2),
                delta_time_s=0.02,
                max_velocity_rad_s=1.0,
                max_acceleration_rad_s2=5.0,
                max_jerk_rad_s3=30.0,
                position_lower=[-2.0, -2.0],
                position_upper=[2.0, 2.0],
            )
            rows = []
            targets = ([0.7, -0.4], [-0.6, 0.5], [0.2, -0.1])
            for index in range(90):
                if index % 7 == 0:
                    update = trajectory.set_target(
                        targets[(index // 7) % len(targets)]
                    )
                    self.assertTrue(update.accepted)
                sample = trajectory.step()
                rows.append(
                    np.r_[
                        sample.position,
                        sample.velocity,
                        sample.acceleration,
                        sample.jerk,
                    ]
                )
            return np.asarray(rows)

        np.testing.assert_array_equal(run(), run())

    def test_stateful_trajectory_rejects_invalid_configuration(self):
        common = dict(
            position=[0.0, 0.0],
            delta_time_s=0.02,
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=2.0,
            max_jerk_rad_s3=10.0,
            position_lower=[-1.0, -1.0],
            position_upper=[1.0, 1.0],
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            StatefulJerkLimitedTrajectory(
                **{**common, "position_upper": [1.0]}
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            StatefulJerkLimitedTrajectory(
                **{**common, "max_jerk_rad_s3": 0.0}
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            StatefulJerkLimitedTrajectory(
                **{**common, "position": [0.0, np.nan]}
            )
        for field in (
            "max_velocity_rad_s",
            "max_acceleration_rad_s2",
            "max_jerk_rad_s3",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ValueError, "finite positive"
                ):
                    StatefulJerkLimitedTrajectory(
                        **{**common, field: np.inf}
                    )
        with self.assertRaisesRegex(ValueError, "smaller"):
            StatefulJerkLimitedTrajectory(
                **{**common, "position_lower": [1.0, -1.0]}
            )


if __name__ == "__main__":
    unittest.main()


import unittest

from robosuite_env.rollout.control.grasp_intent import (
    classify_policy_gripper_intent,
    evaluate_contact_angle_grasp,
    latch_policy_gripper_intents,
)


class LatchHeadroomGuardTest(unittest.TestCase):
    """A grip with the jaws at their stop is one the policy could never undo."""

    def test_latch_at_the_stop_reports_no_reachable_release(self):
        from robosuite_env.rollout.control.grasp_intent import (
            YUBI_JAW_MAX_RAD,
            evaluate_contact_angle_grasp,
        )

        contact = YUBI_JAW_MAX_RAD - 0.001
        predicate = evaluate_contact_angle_grasp(
            contact - 0.01,
            {"left": contact, "right": contact},
            {"left": contact, "right": contact},
            0.001,
            0.05,
            YUBI_JAW_MAX_RAD,
        )

        self.assertIs(predicate["closing_below_contact"], True)
        self.assertIs(predicate["release_reachable"], False)

    def test_a_contact_angle_past_the_stop_is_also_unreachable(self):
        from robosuite_env.rollout.control.grasp_intent import (
            YUBI_JAW_MAX_RAD,
            evaluate_contact_angle_grasp,
        )

        # A latch can land slightly beyond the stop, here 0.001 rad, so no
        # command can ever exceed the contact angle.
        contact = YUBI_JAW_MAX_RAD + 0.001
        predicate = evaluate_contact_angle_grasp(
            0.75,
            {"left": contact, "right": contact},
            {"left": contact, "right": contact},
            0.001,
            0.05,
            YUBI_JAW_MAX_RAD,
        )

        self.assertIs(predicate["release_reachable"], False)

    def test_a_mid_range_grip_keeps_its_headroom(self):
        from robosuite_env.rollout.control.grasp_intent import (
            YUBI_JAW_MAX_RAD,
            evaluate_contact_angle_grasp,
        )

        predicate = evaluate_contact_angle_grasp(
            0.40, {"left": 0.42}, {"left": 0.42}, 0.001, 0.05, YUBI_JAW_MAX_RAD
        )

        self.assertIs(predicate["closing_below_contact"], True)
        self.assertIs(predicate["release_reachable"], True)
        self.assertEqual(predicate["release_thresholds_rad"], {"left": 0.47})

    def test_without_a_cap_every_grip_looks_reachable(self):
        from robosuite_env.rollout.control.grasp_intent import (
            evaluate_contact_angle_grasp,
        )

        contact = 0.7844
        predicate = evaluate_contact_angle_grasp(
            0.75, {"left": contact}, {"left": contact}, 0.001, 0.05
        )

        self.assertIs(predicate["release_reachable"], True)


class ReleaseThresholdCapTest(unittest.TestCase):
    """A full-open command always lets go, whatever the latch angle was."""

    def test_threshold_is_capped_at_the_jaw_maximum(self):
        from robosuite_env.rollout.control.grasp_intent import (
            YUBI_JAW_MAX_RAD,
            evaluate_contact_angle_grasp,
        )

        # Latched near the limit: contact + release_epsilon lands beyond any
        # angle the jaws can be commanded to.
        contact = YUBI_JAW_MAX_RAD - 0.01
        predicate = evaluate_contact_angle_grasp(
            YUBI_JAW_MAX_RAD,
            {"left": contact, "right": contact},
            {"left": contact, "right": contact},
            0.001,
            0.05,
            YUBI_JAW_MAX_RAD,
        )

        self.assertIs(predicate["opened_above_contact"], True)
        self.assertEqual(
            predicate["release_thresholds_rad"],
            {"left": YUBI_JAW_MAX_RAD, "right": YUBI_JAW_MAX_RAD},
        )

    def test_without_the_cap_that_same_grasp_can_never_release(self):
        from robosuite_env.rollout.control.grasp_intent import (
            YUBI_JAW_MAX_RAD,
            evaluate_contact_angle_grasp,
        )

        contact = YUBI_JAW_MAX_RAD - 0.01
        predicate = evaluate_contact_angle_grasp(
            YUBI_JAW_MAX_RAD,
            {"left": contact, "right": contact},
            {"left": contact, "right": contact},
            0.001,
            0.05,
        )

        self.assertIs(predicate["opened_above_contact"], False)

    def test_the_cap_leaves_a_mid_range_grasp_alone(self):
        from robosuite_env.rollout.control.grasp_intent import (
            YUBI_JAW_MAX_RAD,
            evaluate_contact_angle_grasp,
        )

        held = evaluate_contact_angle_grasp(
            0.44, {"left": 0.40}, {"left": 0.40}, 0.001, 0.05, YUBI_JAW_MAX_RAD
        )
        released = evaluate_contact_angle_grasp(
            0.46, {"left": 0.40}, {"left": 0.40}, 0.001, 0.05, YUBI_JAW_MAX_RAD
        )

        self.assertEqual(held["release_thresholds_rad"], {"left": 0.45})
        self.assertIs(held["opened_above_contact"], False)
        self.assertIs(released["opened_above_contact"], True)


class RuleGraspFingerPinTest(unittest.TestCase):
    """The jaws stop where the object latched, and stay there."""

    class _Data:
        def __init__(self, qpos):
            import numpy as np

            self.qpos = np.asarray(qpos, dtype=float)
            self.qvel = np.zeros_like(self.qpos)
            self.qacc = np.zeros_like(self.qpos)
            self.qacc_warmstart = np.zeros_like(self.qpos)
            # The ratchet limits how fast it may pull a jaw back, so it reads
            # the clock. One env.step of simulated time has passed since the
            # last call, which at 1.5 rad/s is 0.03 rad of allowance -- more
            # than any of these fakes asks the clamp to travel.
            self.time = 0.02

    class _State:
        def __init__(self, data, adrs, pinned, dof_adrs=None):
            self.d = data
            # Deliberately different from the qpos addresses: the model keeps
            # separate ``fj`` and ``fd`` arrays and they only coincide by
            # accident of compilation order, so a fake that reuses one for both
            # hides an indexing bug rather than catching it.
            self.ARM = {"R": {"fj": adrs, "fd": dof_adrs or [a - 1 for a in adrs]}}
            self.m = SimpleNamespace(opt=SimpleNamespace(cone=0))
            self.RULE_GRASP = {
                "pinned_finger_q": pinned,
                "ratchet_finger_q": None if pinned is None else list(pinned),
                # One env.step behind the data clock, so the rate-limited clamp
                # has a real allowance. Zero elapsed now means zero movement.
                "ratchet_last_sec": 0.0,
            }

    def test_finger_qpos_reads_each_joint_not_their_mean(self):
        from robosuite_env.rollout.control.rule_grasp import _finger_qpos

        data = self._Data([0.0, 0.40, 0.0, 0.50])
        state = self._State(data, [1, 3], None)

        self.assertEqual(_finger_qpos(state, "R"), [0.40, 0.50])

    def test_a_jaw_that_tried_to_open_is_pushed_back_and_stopped(self):
        """The ratchet's whole job: a latched grasp does not let go by itself.

        Back toward the floor, not onto it. A jaw levered wide open arrives a
        long way out, and closing that gap in one assignment drives the pad
        through whatever opened it; the allowance is 1.5 rad/s, so one 20 ms
        update buys 0.03 rad and the rest comes on later calls.
        """
        from robosuite_env.rollout.control.rule_grasp import _ratchet_grasped_fingers

        data = self._Data([0.0, 0.62, 0.0, 0.71])
        # qpos addresses 1 and 3; the fake's dof addresses are 0 and 2. The
        # motion that has to stop lives at the dof addresses.
        data.qvel[0] = 3.0
        data.qacc[2] = -9.0
        state = self._State(data, [1, 3], [0.40, 0.50])

        _ratchet_grasped_fingers(state, "R")

        self.assertAlmostEqual(data.qpos[1], 0.62 - 0.03)
        self.assertAlmostEqual(data.qpos[3], 0.71 - 0.03)
        self.assertEqual(state.RULE_GRASP["ratchet_finger_q"], [0.40, 0.50],
                         "the floor is kept while the jaw travels back to it")
        self.assertEqual(data.qvel[0], 0.0)
        self.assertEqual(data.qacc[2], 0.0)

    def test_a_jaw_is_not_moved_at_all_when_no_time_has_passed(self):
        """No allowance without elapsed time.

        Zero elapsed time gives zero allowance, not an unlimited one: the next
        physics step starts from the clamped configuration, so anything the
        clamp moved would be pushed into the object.
        """
        from robosuite_env.rollout.control.rule_grasp import _ratchet_grasped_fingers

        data = self._Data([0.0, 0.62, 0.0, 0.71])
        state = self._State(data, [1, 3], [0.40, 0.50])
        state.RULE_GRASP["ratchet_last_sec"] = data.time

        _ratchet_grasped_fingers(state, "R")

        self.assertEqual(list(data.qpos), [0.0, 0.62, 0.0, 0.71])

    def test_an_overshot_limit_is_not_recorded_as_the_floor(self):
        """The solver can push a jaw a hair past its stop on a hard contact.

        Recorded as the floor, that would have the ratchet driving the jaw back
        through its own physical limit for the rest of the grasp.
        """
        from robosuite_env.rollout.control.rule_grasp import _ratchet_grasped_fingers

        data = self._Data([0.0, -0.0249, 0.0, 0.50])
        state = self._State(data, [1, 3], [0.40, 0.50])

        _ratchet_grasped_fingers(state, "R")

        self.assertEqual(state.RULE_GRASP["ratchet_finger_q"][0], 0.0)

    def test_a_jaw_that_closed_further_keeps_the_ground_it_gained(self):
        """Smaller is more closed, and closing further is allowed.

        Jaws held at the latch angle would make the gripper channel of the
        observation a constant for the length of a grasp, and a pen picked up
        crooked could never be squeezed straight.
        """
        from robosuite_env.rollout.control.rule_grasp import _ratchet_grasped_fingers

        data = self._Data([0.0, 0.31, 0.0, 0.44])
        state = self._State(data, [1, 3], [0.40, 0.50])

        _ratchet_grasped_fingers(state, "R")

        self.assertEqual(list(data.qpos), [0.0, 0.31, 0.0, 0.44])
        self.assertEqual(state.RULE_GRASP["ratchet_finger_q"], [0.31, 0.44])

    def test_the_ratchet_does_not_alias_the_latch_angles(self):
        """The latch angles are telemetry and must survive the jaws closing."""
        from robosuite_env.rollout.control.rule_grasp import _ratchet_grasped_fingers

        data = self._Data([0.0, 0.31, 0.0, 0.44])
        state = self._State(data, [1, 3], [0.40, 0.50])

        _ratchet_grasped_fingers(state, "R")

        self.assertEqual(state.RULE_GRASP["pinned_finger_q"], [0.40, 0.50])

    def test_velocity_is_zeroed_at_the_dof_address_not_the_qpos_one(self):
        """``fj`` indexes qpos and ``fd`` indexes qvel; they are not the same.

        They can coincide on one build, so using one for both passes
        there and silently writes into an unrelated joint on any model where
        they do not.
        """
        from robosuite_env.rollout.control.rule_grasp import _ratchet_grasped_fingers

        data = self._Data([0.0, 0.62, 0.0, 0.71])
        data.qvel[:] = 5.0
        data.qacc[:] = 5.0
        data.qacc_warmstart[:] = 5.0
        state = self._State(data, [1, 3], [0.40, 0.50], dof_adrs=[0, 2])

        _ratchet_grasped_fingers(state, "R")

        self.assertEqual(list(data.qvel), [0.0, 5.0, 0.0, 5.0])
        self.assertEqual(list(data.qacc), [0.0, 5.0, 0.0, 5.0])
        self.assertEqual(list(data.qacc_warmstart), [0.0, 5.0, 0.0, 5.0])

    def test_a_rebound_inside_the_deadband_is_left_alone(self):
        """A pad settling on an object oscillates; chasing every excursion
        teleports the jaw back into the object and drives penetration."""
        from robosuite_env.rollout.control.rule_grasp import (
            RULE_GRASP_RATCHET_DEADBAND_RAD,
            _ratchet_grasped_fingers,
        )

        wobble = 0.5 * RULE_GRASP_RATCHET_DEADBAND_RAD
        data = self._Data([0.0, 0.40 + wobble, 0.0, 0.50 - wobble])
        state = self._State(data, [1, 3], [0.40, 0.50])

        _ratchet_grasped_fingers(state, "R")

        self.assertEqual(list(data.qpos), [0.0, 0.40 + wobble, 0.0, 0.50 - wobble])
        self.assertEqual(state.RULE_GRASP["ratchet_finger_q"], [0.40, 0.50])

    def test_the_ratchet_is_a_no_op_before_anything_latched(self):
        from robosuite_env.rollout.control.rule_grasp import _ratchet_grasped_fingers

        data = self._Data([0.0, 0.62, 0.0, 0.71])
        state = self._State(data, [1, 3], None)

        _ratchet_grasped_fingers(state, "R")

        self.assertEqual(list(data.qpos), [0.0, 0.62, 0.0, 0.71])

    def test_the_weld_leaves_the_held_object_free_to_turn(self):
        """torquescale 0. At 1 the object freezes at its first-contact pose."""
        from robosuite_env.rollout.control.rule_grasp import (
            RULE_GRASP_WELD_TORQUESCALE,
        )

        self.assertEqual(RULE_GRASP_WELD_TORQUESCALE, 0.0)

    def test_the_closing_phase_carries_the_object_weight(self):
        """Between latch and weld only friction holds it, and that is not enough.

        The closing phase exists to give contact time to settle an object that
        was caught badly -- which is exactly long enough for a lightly held pen
        to fall out of the hand. Cancelling the weight leaves the rest to the
        solver: it can still roll and be turned square by the pads, it just
        does not sink while that happens.
        """
        import numpy as np
        from types import SimpleNamespace
        from robosuite_env.rollout.control.rule_grasp import (
            _carry_weight_while_closing,
            _drop_carried_weight,
        )

        state = SimpleNamespace(
            m=SimpleNamespace(
                body_subtreemass=np.array([0.0, 0.25]),
                opt=SimpleNamespace(gravity=np.array([0.0, 0.0, -9.81])),
            ),
            d=SimpleNamespace(xfrc_applied=np.zeros((2, 6))),
            RULE_GRASP={},
        )

        _carry_weight_while_closing(state, 1)
        np.testing.assert_allclose(
            state.d.xfrc_applied[1], [0.0, 0.0, 0.25 * 9.81, 0.0, 0.0, 0.0]
        )
        self.assertEqual(state.RULE_GRASP["carried_body_id"], 1,
                         "a carried body has to be findable again")

        # Named by nobody: the drop takes what is on record, so a caller that
        # has already reset the grasp fields still gives the weight back.
        _drop_carried_weight(state)
        np.testing.assert_array_equal(state.d.xfrc_applied[1], np.zeros(6))
        self.assertIsNone(state.RULE_GRASP["carried_body_id"])

    def test_a_tilted_or_absent_gravity_is_followed_not_assumed(self):
        """9.81 down is a property of this model, not of the code."""
        import numpy as np
        from types import SimpleNamespace
        from robosuite_env.rollout.control.rule_grasp import (
            _carry_weight_while_closing,
        )

        state = SimpleNamespace(
            m=SimpleNamespace(
                body_subtreemass=np.array([0.0, 2.0]),
                opt=SimpleNamespace(gravity=np.array([1.0, 0.0, 0.0])),
            ),
            d=SimpleNamespace(xfrc_applied=np.zeros((2, 6))),
            RULE_GRASP={},
        )

        _carry_weight_while_closing(state, 1)
        np.testing.assert_allclose(state.d.xfrc_applied[1][:3], [-2.0, 0.0, 0.0])

    def test_the_stall_window_is_shorter_than_the_time_cap(self):
        """Otherwise the cap fires first and the weld never waits for a stall.

        Both are simulated seconds rather than call counts, because the update
        runs once per ``mj_step`` on the kinematic path and once per
        ``env.step`` -- ten MuJoCo steps -- on a dynamic one. A count
        would have meant two different durations depending on the rig.
        """
        from robosuite_env.rollout.control.rule_grasp import (
            RULE_GRASP_CLOSING_MAX_SEC,
            RULE_GRASP_STALL_RATE_RAD_PER_SEC,
            RULE_GRASP_STALL_SEC,
        )

        self.assertLess(RULE_GRASP_STALL_SEC, RULE_GRASP_CLOSING_MAX_SEC)
        self.assertGreater(RULE_GRASP_STALL_RATE_RAD_PER_SEC, 0.0)


class RuleGraspClosingPhaseTest(unittest.TestCase):
    """What may and may not happen between the latch and the weld.

    The closing phase is the window where the object is held by contact alone.
    Everything here is about not welding in it when the hand is not actually
    gripping: a weld fired while the pads are on one face would hold an
    object the hand is only touching.
    """

    def _state(self, *, time=0.0, jaws=(0.30, 0.30)):
        import numpy as np
        from types import SimpleNamespace

        qpos = np.array([0.0, jaws[0], 0.0, jaws[1]], dtype=float)
        data = SimpleNamespace(
            time=float(time),
            qpos=qpos,
            qvel=np.zeros(4),
            qacc=np.zeros(4),
            qacc_warmstart=np.zeros(4),
            xfrc_applied=np.zeros((2, 6)),
            # A welded grasp re-checks its equality every call, so the fake
            # needs one that is active.
            eq_active=np.ones(16, dtype=np.uint8),
            xpos=np.zeros((8, 3)),
            xquat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (8, 1)),
        )
        state = SimpleNamespace(
            d=data,
            m=SimpleNamespace(
                body_subtreemass=np.array([0.0, 0.25]),
                opt=SimpleNamespace(gravity=np.array([0.0, 0.0, -9.81])),
            ),
            # eef, xpos and xquat because a welded grasp now reads the
            # object's pose in the hand to decide whether it is still turning.
            ARM={"R": {"fj": [1, 3], "fd": [1, 3], "eef": 4}},
            RULE_GRASP={
                "active": True,
                "hand": "R",
                "weld_eq_id": None,
                "object_body_id": 1,
                "ratchet_finger_q": [jaws[0], jaws[1]],
                "closing_started_sec": 0.0,
                "closing_last_sec": 0.0,
                "ratchet_last_sec": None,
                "closing_stalled_sec": 0.0,
                "closing_lost_contact_sec": 0.0,
                "closing_last_rad": sum(jaws),
                "events": [],
            },
        )
        return state

    @staticmethod
    def _contacts(opposed, body_id=1):
        return {"opposed": opposed, "object_body_id": body_id,
                "both": True, "normal_dot": -1.0 if opposed else 1.0}

    def _no_weld_allowed(self):
        from robosuite_env.rollout.control import rule_grasp

        def _boom(*args, **kwargs):
            raise AssertionError("welded while the pads were not opposed")

        return mock.patch.object(
            rule_grasp, "_activate_rule_grasp_weld", _boom)

    def test_a_stalled_jaw_does_not_weld_once_opposition_is_lost(self):
        """The abandon and weld windows must use the same boundary.

        If abandoning needed the contact to have been gone for *longer* than
        the window while welding needed the jaws to have been stopped for *at
        least* it, a grasp that had already stalled, then slipped off the
        object, would spend a tick where it was too early to abandon and late
        enough to weld -- and weld onto an object it was no longer holding.
        """
        from robosuite_env.rollout.control.rule_grasp import (
            RULE_GRASP_STALL_SEC,
            _apply_rule_grasp_hold,
        )

        state = self._state(time=0.02)
        state.RULE_GRASP["closing_stalled_sec"] = RULE_GRASP_STALL_SEC

        with self._no_weld_allowed():
            _apply_rule_grasp_hold(state, 0.0, 5, 0, self._contacts(False))

        self.assertIsNone(state.RULE_GRASP["weld_eq_id"])
        self.assertTrue(state.RULE_GRASP["active"])

    def test_the_time_cap_does_not_weld_onto_an_unopposed_object_either(self):
        """The other way in. The cap is a backstop, not an override."""
        from robosuite_env.rollout.control.rule_grasp import (
            RULE_GRASP_CLOSING_MAX_SEC,
            _apply_rule_grasp_hold,
        )

        state = self._state(time=RULE_GRASP_CLOSING_MAX_SEC + 1.0)
        state.RULE_GRASP["closing_last_sec"] = RULE_GRASP_CLOSING_MAX_SEC + 0.98

        with self._no_weld_allowed():
            _apply_rule_grasp_hold(state, 0.0, 5, 0, self._contacts(False))

        self.assertIsNone(state.RULE_GRASP["weld_eq_id"])

    def test_losing_the_pads_restarts_both_clocks_together(self):
        """Freezing one clock and resetting the other would build a trap.

        If the stall clock survived a dropped sample while the opposed clock
        restarted, after one flicker the stall clock could mature first, so
        the very next opposed sample would meet the weld condition with almost
        no opposed time behind it -- and be abandoned for it. A grip that had
        just steadied itself would be thrown away and never given the chance
        to earn a fresh window.

        Kept in step, "the jaws stopped" can only ever mean "the jaws stopped
        while actually holding it".
        """
        from robosuite_env.rollout.control.rule_grasp import _apply_rule_grasp_hold

        state = self._state(time=0.02)
        state.RULE_GRASP["closing_stalled_sec"] = 0.03
        state.RULE_GRASP["closing_opposed_sec"] = 0.03

        with self._no_weld_allowed():
            _apply_rule_grasp_hold(state, 0.0, 5, 0, self._contacts(False))

        self.assertEqual(state.RULE_GRASP["closing_stalled_sec"], 0.0)
        self.assertEqual(state.RULE_GRASP["closing_opposed_sec"], 0.0)

    def test_one_dropped_sample_does_not_throw_the_grasp_away(self):
        """One dropped sample amid steady contact, at a 20 ms dynamic cadence."""
        from robosuite_env.rollout.control import rule_grasp
        from robosuite_env.rollout.control.rule_grasp import _apply_rule_grasp_hold

        state = self._state(time=0.0)
        state.RULE_GRASP["closing_last_sec"] = 0.0
        welds = []
        now = 0.0

        def _weld(*args, **kwargs):
            welds.append(round(now, 3))
            return 9

        with mock.patch.object(rule_grasp, "_activate_rule_grasp_weld", _weld):
            # Two opposed samples, one dropout, then a settled grip.
            for opposed in (True, True, False, True, True, True, True):
                now += 0.02
                state.d.time = now
                _apply_rule_grasp_hold(state, 0.0, 5, 0, self._contacts(opposed))

        self.assertTrue(state.RULE_GRASP["active"],
                        "the grasp was abandoned after a single dropped sample")
        self.assertTrue(welds, "and it never got another window to weld in")

    def test_nothing_is_held_up_while_the_pads_are_off_the_object(self):
        """The weight carry is for a grasp in progress, not for a dropped object."""
        import numpy as np
        from robosuite_env.rollout.control.rule_grasp import _apply_rule_grasp_hold

        from robosuite_env.rollout.control.rule_grasp import (
            _carry_weight_while_closing,
        )

        state = self._state(time=0.02)
        _carry_weight_while_closing(state, 1)
        self.assertGreater(state.d.xfrc_applied[1, 2], 0.0)

        with self._no_weld_allowed():
            _apply_rule_grasp_hold(state, 0.0, 5, 0, self._contacts(False))

        np.testing.assert_array_equal(state.d.xfrc_applied[1], np.zeros(6))

    def test_a_proper_pinch_still_welds_when_the_jaws_stop(self):
        """The refusals above must not have cost the thing the phase is for."""
        from robosuite_env.rollout.control import rule_grasp
        from robosuite_env.rollout.control.rule_grasp import (
            RULE_GRASP_STALL_SEC,
            _apply_rule_grasp_hold,
        )

        state = self._state(time=RULE_GRASP_STALL_SEC + 0.02)
        state.RULE_GRASP["closing_stalled_sec"] = RULE_GRASP_STALL_SEC
        state.RULE_GRASP["closing_last_sec"] = RULE_GRASP_STALL_SEC
        # Held throughout, which is what earns the weld.
        state.RULE_GRASP["closing_opposed_sec"] = RULE_GRASP_STALL_SEC

        with mock.patch.object(
            rule_grasp, "_activate_rule_grasp_weld", lambda *a, **k: 7
        ):
            _apply_rule_grasp_hold(state, 0.0, 5, 0, self._contacts(True))

        self.assertEqual(state.RULE_GRASP["weld_eq_id"], 7)
        events = [e["event"] for e in state.RULE_GRASP["events"]]
        self.assertIn("weld", events)

    def test_a_flickering_grip_never_welds_however_long_it_flickers(self):
        """40 ms off, 20 ms on, three times -- at a 20 ms dynamic cadence.

        The counters are consecutive, not cumulative: cumulative counters would
        reach their window without the pads ever holding the object for one,
        and the third isolated opposed sample would weld with the reason
        "jaws_stopped_closing", outside the guard on the one-second cap.

        Nothing here is ever lost for long enough to abandon either: the loss
        counter is reset by each opposed sample, so this case sits between the
        two guards rather than being one either of them catches.
        """
        from robosuite_env.rollout.control.rule_grasp import _apply_rule_grasp_hold

        state = self._state(time=0.0)
        state.RULE_GRASP["closing_last_sec"] = 0.0

        now = 0.0
        with self._no_weld_allowed():
            for _ in range(3):
                for _ in range(2):  # 40 ms with the pads off the object
                    now += 0.02
                    state.d.time = now
                    _apply_rule_grasp_hold(state, 0.0, 1, 0, self._contacts(False))
                    if not state.RULE_GRASP["active"]:
                        break
                if not state.RULE_GRASP["active"]:
                    break
                now += 0.02  # one 20 ms sample with them back on it
                state.d.time = now
                _apply_rule_grasp_hold(state, 0.0, 1, 0, self._contacts(True))

        self.assertIsNone(state.RULE_GRASP["weld_eq_id"])

    def test_an_abandoned_grasp_is_not_taken_again_in_the_same_breath(self):
        """Clearing the fields drops the caller into the loop that latches.

        The contacts that were just refused are still there, so without a
        refusal on record the same grasp is taken again in the same update --
        an abandon and a latch, over and over, forever.
        """
        from robosuite_env.rollout.control.rule_grasp import (
            _abandon_rule_grasp,
            _relatch_refused,
        )

        state = self._state()
        contacts = self._contacts(True)

        _abandon_rule_grasp(state, "R", contacts)

        self.assertFalse(state.RULE_GRASP["active"])
        self.assertEqual(state.RULE_GRASP["refused_body_id"]["R"], 1)
        self.assertTrue(_relatch_refused(state, "R", contacts),
                        "the same grip on the same object is still refused")

        # Off it, or holding it some other way, and the refusal lifts.
        self.assertFalse(_relatch_refused(state, "R", self._contacts(False)))
        self.assertIsNone(state.RULE_GRASP["refused_body_id"]["R"])

    def test_one_hands_refusal_is_not_cleared_by_the_other_hand(self):
        """rule_grasp_hand is "auto", so the loop visits both hands.

        With one shared field, the idle hand's empty contact list read as "not
        opposed" and cleared the refusal the other hand had just recorded -- so
        the abandoned grasp was taken again on the very next update. The
        same-call loop, moved one tick later and no less endless.
        """
        from robosuite_env.rollout.control.rule_grasp import (
            _abandon_rule_grasp,
            _relatch_refused,
        )

        state = self._state()
        contacts = self._contacts(True)
        _abandon_rule_grasp(state, "R", contacts)

        # The idle left hand, touching nothing at all.
        nothing = {"opposed": False, "object_body_id": None, "both": False}
        self.assertFalse(_relatch_refused(state, "L", nothing))

        self.assertTrue(_relatch_refused(state, "R", contacts),
                        "the left hand cleared the right hand's refusal")

    def test_the_deadline_finalises_the_grasp_instead_of_dropping_it(self):
        """A latch is a decision; nothing after it should quietly undo that.

        At the deadline the grasp is finalised rather than abandoned: full
        weld, jaws held, released only when the policy opens. Abandoning there
        would drop grasps that latched but had not yet passed the seating test.
        """
        from robosuite_env.rollout.control import rule_grasp
        from robosuite_env.rollout.control.rule_grasp import (
            RULE_GRASP_CLOSING_MAX_SEC,
            _apply_rule_grasp_hold,
        )

        for label, contacts in (
            ("lost for good", {"opposed": False, "object_body_id": 1,
                               "both": True}),
            ("held but never settling", {"opposed": True, "object_body_id": 1,
                                         "both": True}),
        ):
            state = self._welded()
            state.RULE_GRASP["object_body_id"] = 1
            engaged = []
            with mock.patch.object(
                rule_grasp, "_tighten_rule_grasp_weld",
                lambda st, hand: (engaged.append(round(float(st.d.time), 2)),
                                  st.RULE_GRASP.__setitem__("rotation_pinned",
                                                            True)),
            ):
                now = 0.0
                # Creep the jaws so the stall test never fires on its own.
                while now < RULE_GRASP_CLOSING_MAX_SEC + 0.1:
                    now += 0.02
                    state.d.time = now
                    state.d.qpos[1] = 0.30 - now * 0.05
                    state.d.qpos[3] = 0.30 - now * 0.05
                    _apply_rule_grasp_hold(state, 0.0, 1, 0, dict(contacts))

            self.assertTrue(engaged, f"the grasp was dropped: {label}")
            self.assertTrue(state.RULE_GRASP["active"], label)
            self.assertIsNotNone(state.RULE_GRASP["weld_eq_id"], label)
            reasons = [e.get("reason") for e in state.RULE_GRASP["events"]]
            self.assertNotIn("never_held_the_object_while_seating", reasons)

    def test_abandoning_takes_the_weld_down_with_it(self):
        """The weld goes on at the latch, so an abandon has one to undo.

        Clearing weld_eq_id without deactivating the equality leaves it holding
        the object while the state machine believes nothing is held, and the
        next latch dies on "rule-grasp weld was already active".
        """
        from robosuite_env.rollout.control import rule_grasp
        from robosuite_env.rollout.control.rule_grasp import _abandon_rule_grasp

        state = self._welded()
        state.RULE_GRASP["object_body_id"] = 1
        taken_down = []
        with mock.patch.object(
            rule_grasp, "_deactivate_rule_grasp_weld",
            lambda st, hand: taken_down.append(st.RULE_GRASP["weld_eq_id"]),
        ):
            _abandon_rule_grasp(state, "R", self._contacts(True))

        self.assertEqual(taken_down, [3], "the equality was left active")
        self.assertIsNone(state.RULE_GRASP["weld_eq_id"])

    def test_a_hand_refuses_what_it_latched_not_what_it_is_touching(self):
        """The latched body rolled clear and another object is under the pads.

        Refusing that one would blame a body this hand never grasped -- and
        leave the one it did give up on free to be taken again.
        """
        from robosuite_env.rollout.control.rule_grasp import _abandon_rule_grasp

        state = self._state()
        state.RULE_GRASP["object_body_id"] = 1
        elsewhere = {"opposed": True, "object_body_id": 7, "both": True}

        _abandon_rule_grasp(state, "R", elsewhere)

        self.assertEqual(state.RULE_GRASP["refused_body_id"]["R"], 1)

    def _welded(self):
        """A latched, translation-welded grasp, with a hand and object to read."""
        state = self._state(time=0.0)
        state.RULE_GRASP["weld_eq_id"] = 3
        state.RULE_GRASP["object_body_id"] = 1
        return state

    def _close(self, state, apertures, start=0.0, contacts=None):
        """Drive the total jaw aperture, one reading per 20 ms update."""
        from robosuite_env.rollout.control.rule_grasp import _apply_rule_grasp_hold

        now = start
        for aperture in apertures:
            now += 0.02
            state.d.time = now
            half = aperture / 2.0
            state.d.qpos[1] = half
            state.d.qpos[3] = half
            cs = dict(self._contacts(True)) if contacts is None else dict(contacts)
            _apply_rule_grasp_hold(state, 0.0, 1, 0, cs)
        return now

    def test_the_grasp_finishes_when_the_jaws_stop_gaining_aperture(self):
        """Seating is over when the jaws can close no further under load.

        While a crooked object turns toward the orientation the pads can hold,
        the jaws gain angle. Once they cannot, that motion is finished, and
        rotation at constant aperture is symmetry or drift rather than more
        engagement. The jaws stop where they came to rest and the orientation
        is pinned at the same instant -- one transition, no threshold of its
        own beyond the stall test the closing phase already used.
        """
        from robosuite_env.rollout.control import rule_grasp

        state = self._welded()
        engaged = []
        with mock.patch.object(
            rule_grasp, "_tighten_rule_grasp_weld",
            lambda st, hand: (engaged.append(round(float(st.d.time), 3)),
                              st.RULE_GRASP.__setitem__("rotation_pinned", True)),
        ):
            # Still closing: 0.02 rad of aperture per update is 1 rad/s.
            now = self._close(state, [0.60, 0.58, 0.56, 0.54, 0.52])
            self.assertEqual(engaged, [], "finished while the jaws were still closing")
            # Stalled.
            self._close(state, [0.52, 0.52, 0.52, 0.52], start=now)

        self.assertTrue(engaged, "never finished though the jaws had stopped")
        self.assertEqual(state.RULE_GRASP["stopped_finger_q"], [0.26, 0.26])

    def test_the_jaws_are_then_held_where_they_stopped(self):
        """The jaw itself is held, not only the recorded floor.

        The floor is only what the clamp pulls back *to*; a jaw left free to
        close below it would drive the fingers into a cup whose origin is
        welded and cannot move away.
        """
        from robosuite_env.rollout.control.rule_grasp import _ratchet_grasped_fingers

        state = self._welded()
        state.RULE_GRASP["stopped_finger_q"] = [0.26, 0.26]
        state.d.qpos[1] = 0.10          # driven far past where it stopped
        state.d.qpos[3] = 0.10
        # dof addresses, which this fake keeps at [1, 3] alongside the qpos
        # ones -- they are separate arrays and only coincide by accident of
        # compilation order.
        state.d.qvel[1] = -5.0
        state.d.qacc[3] = -30.0

        _ratchet_grasped_fingers(state, "R")

        self.assertEqual(state.d.qpos[1], 0.26)
        self.assertEqual(state.d.qpos[3], 0.26)
        self.assertEqual(state.d.qvel[1], 0.0)
        self.assertEqual(state.d.qacc[3], 0.0)
        self.assertEqual(state.d.qacc_warmstart[1], 0.0)

    def test_losing_the_object_does_not_count_as_having_settled(self):
        """Lost contact is not a stall.

        Contacts thinning out, the pads coming off, the selector falling back
        to another object -- every one of those looks like "not moving any
        more". Tightening there welds the hand to something it is not holding,
        for the rest of the episode.
        """
        from robosuite_env.rollout.control import rule_grasp

        for label, contacts in (
            ("not opposed", {"opposed": False, "object_body_id": 1, "both": True}),
            ("no contact at all", {"opposed": False, "object_body_id": None,
                                   "both": False}),
            ("a different object", {"opposed": True, "object_body_id": 9,
                                    "both": True}),
        ):
            state = self._welded()
            pinned = []
            with mock.patch.object(
                rule_grasp, "_tighten_rule_grasp_weld",
                lambda st, hand: pinned.append(1),
            ):
                # Perfectly still for far longer than the dwell.
                self._close(state, [0.52] * 12, contacts=contacts)
            self.assertEqual(pinned, [], f"tightened while {label}")

    def test_a_grip_that_keeps_slipping_never_accumulates_a_dwell(self):
        """Consecutive, not cumulative: the wait is for a settled grip."""
        from robosuite_env.rollout.control import rule_grasp

        state = self._welded()
        pinned = []
        held = {"opposed": True, "object_body_id": 1, "both": True}
        lost = {"opposed": False, "object_body_id": 1, "both": True}
        with mock.patch.object(
            rule_grasp, "_tighten_rule_grasp_weld",
            lambda st, hand: pinned.append(1),
        ):
            now = 0.0
            for _ in range(6):
                now = self._close(state, [0.52, 0.52], start=now, contacts=held)
                now = self._close(state, [0.52], start=now, contacts=lost)

        self.assertEqual(pinned, [], "a dwell survived the grip being lost")

    def test_frozen_time_cannot_manufacture_a_dwell(self):
        """The replay paths call mj_forward only, so d.time never moves.

        A rate is undefined there. Counting calls instead would invent a
        cadence matching neither the 2 ms nor the 20 ms runtime one.
        """
        from robosuite_env.rollout.control import rule_grasp
        from robosuite_env.rollout.control.rule_grasp import _apply_rule_grasp_hold

        state = self._welded()
        pinned = []
        with mock.patch.object(
            rule_grasp, "_tighten_rule_grasp_weld",
            lambda st, hand: pinned.append(1),
        ):
            for _ in range(20):
                _apply_rule_grasp_hold(state, 0.0, 1, 0, dict(self._contacts(True)))

        self.assertEqual(pinned, [], "a frozen clock produced a dwell")

    def test_a_fresh_state_satisfies_its_own_pristine_check(self):
        """The constructor and the invariant have to agree by construction.

        Every field the clear and the check know must also be set by the
        constructor: a missing key reads as None, which is neither False nor
        0.0.
        """
        from robosuite_env.rollout.control.rule_grasp import (
            _cleared_rule_grasp_fields,
            _new_rule_grasp_state,
        )

        fresh = _new_rule_grasp_state()
        for key, value in _cleared_rule_grasp_fields().items():
            self.assertIn(key, fresh)
            self.assertEqual(fresh[key], value, key)

    def test_tightening_re_reads_the_orientation_but_not_the_translation(self):
        """The latch pose is the crooked one -- that is the point of phase one.

        Writing torquescale 1 against the orientation recorded at the latch
        would snap the object back to however it was first caught.
        """
        import numpy as np
        from types import SimpleNamespace
        from robosuite_env.rollout.control import rule_grasp
        from robosuite_env.rollout.control.rule_grasp import (
            _tighten_rule_grasp_weld,
        )

        eq_data = np.zeros((4, 11))
        eq_data[3, 3:6] = [9.0, 9.0, 9.0]          # the latch translation, kept
        eq_data[3, 6:10] = [0.0, 1.0, 0.0, 0.0]
        state = SimpleNamespace(
            m=SimpleNamespace(eq_data=eq_data),
            d=SimpleNamespace(),
            ARM={"R": {"eef": 5}},
            RULE_GRASP={"weld_eq_id": 3, "object_body_id": 7},
        )
        with mock.patch.object(
            rule_grasp, "_relpose",
            lambda d, a, b: (np.array([0.1, 0.2, 0.3]),
                             np.array([1.0, 0.0, 0.0, 0.0]))):
            _tighten_rule_grasp_weld(state, "R")

        np.testing.assert_allclose(
            eq_data[3, 3:6], [9.0, 9.0, 9.0],
            err_msg="the latch translation was overwritten; only the "
                    "orientation is being enabled here")
        np.testing.assert_allclose(eq_data[3, 6:10], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(eq_data[3, 10], 1.0)
        self.assertTrue(state.RULE_GRASP["rotation_pinned"])

    def test_the_ratchet_closes_a_forced_jaw_at_a_bounded_speed(self):
        """Ten substeps run between calls; the jaw comes back over several.

        Putting a jaw levered wide open straight back onto its floor in one
        assignment drives the pad through whatever pushed it open, which is a
        penetration source rather than a hold. The floor is not given up -- the
        jaw simply takes a few calls to reach it.
        """
        from robosuite_env.rollout.control.rule_grasp import (
            RULE_GRASP_RATCHET_MAX_RATE_RAD_PER_SEC,
            _ratchet_grasped_fingers,
        )

        state = self._state(time=0.02, jaws=(0.30, 0.30))
        state.RULE_GRASP["ratchet_finger_q"] = [0.10, 0.10]
        state.RULE_GRASP["ratchet_last_sec"] = 0.0  # one env.step ago

        _ratchet_grasped_fingers(state, "R")

        step = RULE_GRASP_RATCHET_MAX_RATE_RAD_PER_SEC * 0.02
        self.assertAlmostEqual(state.d.qpos[1], 0.30 - step)
        self.assertGreater(state.d.qpos[1], 0.10, "teleported onto the floor")
        self.assertEqual(state.RULE_GRASP["ratchet_finger_q"][0], 0.10,
                         "the floor was given up")

    def test_the_ratchet_never_overshoots_its_floor(self):
        """Bounded speed, not a fixed step: it stops at the floor."""
        from robosuite_env.rollout.control.rule_grasp import (
            _ratchet_grasped_fingers,
        )

        state = self._state(time=1.0, jaws=(0.3005, 0.3005))
        state.RULE_GRASP["ratchet_finger_q"] = [0.30, 0.30]
        state.RULE_GRASP["ratchet_last_sec"] = 0.0  # a whole second of allowance

        _ratchet_grasped_fingers(state, "R")

        self.assertAlmostEqual(state.d.qpos[1], 0.30)


class RuleGraspWeldTest(unittest.TestCase):
    """Rule grasp retargets inactive welds and never teleports the object."""

    _MODEL_XML = """
    <mujoco model="rule_grasp_weld_test">
      <option gravity="0 0 0"/>
      <worldbody>
        <body name="object" pos="1 2 3" quat="0.7071067811865476 0 0 0.7071067811865476">
          <freejoint name="object_free"/>
          <geom name="object_geom" type="sphere" size="0.01" mass="0.1"/>
        </body>
        <body name="right_eef" pos="1 3 3" quat="0 0 0 1">
          <freejoint name="right_eef_free"/>
          <geom name="right_eef_geom" type="sphere" size="0.01" mass="0.1"/>
          <body name="right_finger_left">
            <joint name="right_finger_left_joint" type="slide" axis="1 0 0"/>
            <geom type="sphere" size="0.005" mass="0.01"/>
          </body>
          <body name="right_finger_right">
            <joint name="right_finger_right_joint" type="slide" axis="1 0 0"/>
            <geom type="sphere" size="0.005" mass="0.01"/>
          </body>
        </body>
        <body name="left_eef" pos="-1 0 0">
          <freejoint name="left_eef_free"/>
          <geom name="left_eef_geom" type="sphere" size="0.01" mass="0.1"/>
        </body>
      </worldbody>
      <equality>
        <connect name="preexisting_equality" body1="object" body2="right_eef"
                 anchor="1 2 3" active="false"/>
      </equality>
    </mujoco>
    """

    @classmethod
    def _compiled_model(cls):
        from robosuite_env.tasks.equalities import add_rule_grasp_weld_equalities

        root = ET.fromstring(cls._MODEL_XML)
        equality = root.find("equality")
        before = len(equality) if equality is not None else 0
        welds = add_rule_grasp_weld_equalities(
            root, placeholder_bodies=("object", "right_eef")
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data, before, welds

    @classmethod
    def _state(cls, grasp_mode="rule"):
        from robosuite_env.rollout.control.rule_grasp import _new_rule_grasp_state

        model, data, _, _ = cls._compiled_model()
        finger_qpos = [
            int(model.jnt_qposadr[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name
            )])
            for name in ("right_finger_left_joint", "right_finger_right_joint")
        ]
        finger_dof = [
            int(model.jnt_dofadr[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name
            )])
            for name in ("right_finger_left_joint", "right_finger_right_joint")
        ]
        data.qpos[finger_qpos] = [0.50, 0.51]
        mujoco.mj_forward(model, data)
        state = SimpleNamespace(
            m=model,
            d=data,
            A=SimpleNamespace(
                grasp_mode=grasp_mode,
                rule_grasp_hand="R",
                rule_grasp_policy_eps_rad=0.001,
                rule_grasp_release_eps_rad=0.05,
                rule_grasp_effort_eps=0.001,
                state_dump_on_rule_grasp=False,
            ),
            ARM={
                "R": {
                    "eef": mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_BODY, "right_eef"
                    ),
                    "fj": finger_qpos,
                    "fd": finger_dof,
                }
            },
            RULE_GRASP=_new_rule_grasp_state(),
        )
        return state

    def test_declaration_preserves_existing_equality_and_adds_two_inactive_welds(self):
        from robosuite_env.tasks.equalities import RULE_GRASP_WELD_NAMES

        model, data, before, welds = self._compiled_model()

        self.assertEqual(before, 1)
        self.assertEqual(len(welds), 2)
        self.assertEqual(model.neq, before + 2)
        self.assertGreaterEqual(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, "preexisting_equality"
            ),
            0,
        )
        for name in RULE_GRASP_WELD_NAMES.values():
            eq_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, name
            )
            self.assertGreaterEqual(eq_id, 0)
            self.assertEqual(model.eq_type[eq_id], mujoco.mjtEq.mjEQ_WELD)
            self.assertEqual(model.eq_active0[eq_id], 0)
            self.assertEqual(data.eq_active[eq_id], 0)
            np.testing.assert_allclose(model.eq_solref[eq_id], [0.005, 1.0])
            np.testing.assert_allclose(
                model.eq_solimp[eq_id, :3], [0.99, 0.999, 0.001]
            )
            self.assertEqual(model.eq_data[eq_id, 10], 1.0)

    def test_relpose_returns_body2_in_body1_coordinates(self):
        from robosuite_env.rollout.control.rule_grasp import _relpose

        model, data, _, _ = self._compiled_model()
        object_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        right_eef = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_eef"
        )

        relative_position, relative_quaternion = _relpose(
            data, object_body, right_eef
        )

        np.testing.assert_allclose(relative_position, [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            relative_quaternion,
            [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
            atol=1e-12,
        )

    def test_latch_activates_weld_and_release_preserves_object_velocity(self):
        from robosuite_env.rollout.control.rule_grasp import (
            _relpose,
            RULE_GRASP_STALL_SEC,
            _rule_grasp_update,
        )

        state = self._state()
        object_body = mujoco.mj_name2id(
            state.m, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        contacts = {
            "left": 1,
            "right": 1,
            "both": True,
            "same_object": True,
            "inner_pad_only": True,
            # Opposed pads: the grip this test is about. Two pads resting on
            # one face report the same shape with ``opposed`` false, and the
            # test below covers that case.
            "normal_dot": -0.98,
            "opposed": True,
            "object_body_id": object_body,
            "object_body": "object",
            "contacts": [],
        }
        object_joint = mujoco.mj_name2id(
            state.m, mujoco.mjtObj.mjOBJ_JOINT, "object_free"
        )
        object_qpos = int(state.m.jnt_qposadr[object_joint])
        object_dof = int(state.m.jnt_dofadr[object_joint])
        object_qpos_before_latch = state.d.qpos[object_qpos : object_qpos + 7].copy()
        state.d.qvel[object_dof : object_dof + 6] = [
            0.01, -0.02, 0.03, -0.04, 0.05, -0.06
        ]
        object_qvel_before_latch = state.d.qvel[object_dof : object_dof + 6].copy()
        # Hand first: at torquescale 0 that is what holds the object's own
        # origin at a fixed point in the hand. The other order pins a point of
        # the object to the hand's origin, and the object orbits the hand as it
        # turns.
        expected_relpos, expected_relquat = _relpose(
            state.d, state.ARM["R"]["eef"], object_body
        )
        with mock.patch(
            "robosuite_env.rollout.control.rule_grasp._rule_grasp_contact_sides",
            return_value=contacts,
        ):
            latched = _rule_grasp_update(
                state,
                {"R": 0.40},
                grip_intents={
                    "R": {
                        "target_rad": 0.40,
                        "value": -0.10,
                        "direction": "closing",
                        "source": "test",
                    }
                },
            )

            # A latch pins the object at once: translation is welded at latch
            # so a tape roll cannot roll back out of the jaws. At torquescale 0 the
            # orientation is still the solver's, so the jaws can go on closing
            # and a crooked pen can still be turned square.
            self.assertIs(latched["active"], True)
            self.assertEqual(latched["hold_mode"], "weld_equality")
            self.assertIsNotNone(latched["weld_eq_id"])

            # Hold the jaws where they are and let the stall window run out.
            # d.time does not advance without mj_step, so move it by hand: the
            # window is a duration, not a number of calls.
            for _ in range(3):
                state.d.time += RULE_GRASP_STALL_SEC
                welded = _rule_grasp_update(
                    state,
                    {"R": 0.40},
                    grip_intents={
                        "R": {
                            "target_rad": 0.40,
                            "value": -0.10,
                            "direction": "closing",
                            "source": "test",
                        }
                    },
                )
            eq_id = state.RULE_GRASP["weld_eq_id"]
            self.assertEqual(welded["phase"], "welded")
            self.assertEqual(welded["hold_mode"], "weld_equality")
            self.assertIsNotNone(eq_id)
            self.assertEqual(state.d.eq_active[eq_id], 1)
            self.assertEqual(state.m.eq_obj1id[eq_id], state.ARM["R"]["eef"])
            self.assertEqual(state.m.eq_obj2id[eq_id], object_body)
            np.testing.assert_allclose(
                state.m.eq_data[eq_id, 3:6], expected_relpos, atol=1e-12
            )
            np.testing.assert_allclose(
                state.m.eq_data[eq_id, 6:10], expected_relquat, atol=1e-12
            )
            np.testing.assert_array_equal(
                state.d.qpos[object_qpos : object_qpos + 7],
                object_qpos_before_latch,
            )
            np.testing.assert_array_equal(
                state.d.qvel[object_dof : object_dof + 6],
                object_qvel_before_latch,
            )

            state.d.qvel[object_dof : object_dof + 6] = [
                0.1, -0.2, 0.3, -0.4, 0.5, -0.6
            ]
            velocity_before_release = state.d.qvel.copy()

            released = _rule_grasp_update(
                state,
                {"R": 0.70},
                grip_intents={
                    "R": {
                        "target_rad": 0.70,
                        "value": 0.20,
                        "direction": "opening",
                        "source": "test",
                    }
                },
            )

        self.assertIs(released["active"], False)
        self.assertIsNone(released["weld_eq_id"])
        self.assertEqual(state.d.eq_active[eq_id], 0)
        np.testing.assert_array_equal(state.d.qvel, velocity_before_release)

    def test_physics_mode_leaves_declared_welds_inactive_and_untouched(self):
        from robosuite_env.rollout.control.rule_grasp import _rule_grasp_update
        from robosuite_env.tasks.equalities import RULE_GRASP_WELD_NAMES

        state = self._state(grasp_mode="physics")
        object_ids_before = state.m.eq_obj1id.copy()
        hand_ids_before = state.m.eq_obj2id.copy()
        data_before = state.m.eq_data.copy()

        status = _rule_grasp_update(state, {"R": 0.40})

        self.assertEqual(status["mode"], "physics")
        self.assertIsNone(status["hold_mode"])
        for name in RULE_GRASP_WELD_NAMES.values():
            eq_id = mujoco.mj_name2id(
                state.m, mujoco.mjtObj.mjOBJ_EQUALITY, name
            )
            self.assertEqual(state.m.eq_active0[eq_id], 0)
            self.assertEqual(state.d.eq_active[eq_id], 0)
        np.testing.assert_array_equal(state.m.eq_obj1id, object_ids_before)
        np.testing.assert_array_equal(state.m.eq_obj2id, hand_ids_before)
        np.testing.assert_array_equal(state.m.eq_data, data_before)


class PolicyGripperIntentTest(unittest.TestCase):
    def test_direction_classification_table(self):
        cases = (
            (0.4, "closing", -0.1),
            (0.6, "opening", 0.1),
            (0.5005, "neutral", 0.0005),
            (0.499, "neutral", -0.001),
            (0.501, "neutral", 0.001),
        )
        for target, direction, value in cases:
            with self.subTest(target=target):
                intent = classify_policy_gripper_intent(target, 0.5, 0.001)
                self.assertEqual(intent["direction"], direction)
                self.assertEqual(intent["source"], "policy_target_error")
                self.assertAlmostEqual(intent["value"], value)

    def test_negative_epsilon_is_rejected(self):
        with self.assertRaisesRegex(ValueError, ">= 0"):
            classify_policy_gripper_intent(0.4, 0.5, -0.001)

    def test_non_finite_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            classify_policy_gripper_intent(float("nan"), 0.5, 0.001)

    def test_source_action_intent_is_latched(self):
        latched = latch_policy_gripper_intents(
            {"R": 0.4, "L": 0.6},
            {"R": 0.5, "L": 0.5},
            0.001,
        )

        self.assertEqual(latched["R"]["direction"], "closing")
        self.assertEqual(latched["L"]["direction"], "opening")
        self.assertIs(latched["R"]["latched"], True)
        self.assertEqual(latched["R"]["sample_phase"], "action_start")
        self.assertEqual(
            classify_policy_gripper_intent(0.4, 0.3, 0.001)["direction"],
            "opening",
        )
        self.assertEqual(latched["R"]["direction"], "closing")

    def test_contact_angle_grasp_requires_target_below_both_fingers(self):
        predicate = evaluate_contact_angle_grasp(
            0.49,
            {"left": 0.50, "right": 0.51},
            {"left": 0.50, "right": 0.51},
            0.001,
        )

        self.assertIs(predicate["closing_below_contact"], True)
        self.assertIs(predicate["opened_above_contact"], False)
        self.assertEqual(predicate["closing_sides"], {"left": True, "right": True})

        one_side_not_closing = evaluate_contact_angle_grasp(
            0.505,
            {"left": 0.50, "right": 0.51},
            {"left": 0.50, "right": 0.51},
            0.001,
        )
        self.assertIs(one_side_not_closing["closing_below_contact"], False)

    def test_contact_angle_grasp_clears_when_the_target_opens_either_side(self):
        # 0.505 sits above the left contact angle plus epsilon but below the
        # right one, so the left side alone ends the grasp.
        predicate = evaluate_contact_angle_grasp(
            0.505,
            {"left": 0.50, "right": 0.51},
            {"left": 0.502, "right": 0.50},
            0.001,
        )

        self.assertIs(predicate["opened_above_contact"], True)
        self.assertEqual(predicate["opened_sides"], {"left": True, "right": False})

    def test_contact_angle_grasp_ignores_measured_angles_when_deciding(self):
        """A held object stops yielding, so the fingers get pushed open by the
        contact they can no longer compress.  Releasing on that measurement
        would make latching the very thing that ends the grasp."""
        forced_open = evaluate_contact_angle_grasp(
            0.45,
            {"left": 0.50, "right": 0.50},
            {"left": 0.62, "right": 0.62},
            0.001,
            0.05,
        )

        self.assertIs(forced_open["opened_above_contact"], False)
        self.assertIs(forced_open["closing_below_contact"], True)
        self.assertEqual(
            forced_open["actual_angles_rad"], {"left": 0.62, "right": 0.62}
        )

    def test_contact_angle_grasp_uses_independent_release_margin(self):
        # Identical measured angles in both calls: only the policy target moves
        # across the left side's release threshold of 0.50 + 0.05.
        held = evaluate_contact_angle_grasp(
            0.549,
            {"left": 0.50, "right": 0.51},
            {"left": 0.52, "right": 0.52},
            0.001,
            0.05,
        )
        released = evaluate_contact_angle_grasp(
            0.551,
            {"left": 0.50, "right": 0.51},
            {"left": 0.52, "right": 0.52},
            0.001,
            0.05,
        )

        self.assertIs(held["opened_above_contact"], False)
        self.assertIs(released["opened_above_contact"], True)
        self.assertEqual(released["opened_sides"], {"left": True, "right": False})
        self.assertEqual(released["release_epsilon_rad"], 0.05)

    def test_contact_angle_grasp_threshold_boundaries_are_stable(self):
        predicate = evaluate_contact_angle_grasp(
            0.499,
            {"left": 0.50, "right": 0.50},
            {"left": 0.501, "right": 0.501},
            0.001,
        )

        self.assertIs(predicate["closing_below_contact"], False)
        self.assertIs(predicate["opened_above_contact"], False)

    def test_contact_angle_grasp_rejects_mismatched_finger_keys(self):
        with self.assertRaisesRegex(ValueError, "same non-empty keys"):
            evaluate_contact_angle_grasp(
                0.4,
                {"left": 0.5, "right": 0.5},
                {"left": 0.5},
                0.001,
            )


if __name__ == "__main__":
    unittest.main()


import unittest

from robosuite_env.rollout.timing import (
    constant_control_ticks_per_source_action,
    control_ticks_for_source_action,
    exact_state_replay_playback_clock,
    nearest_control_ticks_for_source_action,
    resolve_dynamic_timing,
)


class DynamicTimingTest(unittest.TestCase):
    def test_state_replay_playback_clock_does_not_quantize_30hz(self):
        clock = exact_state_replay_playback_clock(30.0)

        self.assertEqual(clock.frame_frequency_hz, 30.0)
        self.assertAlmostEqual(clock.frame_dt_s, 1.0 / 30.0)
        self.assertEqual(clock.output_fps, 30)

    def test_state_replay_playback_clock_rejects_lossy_rates(self):
        for invalid_rate in (False, 0.0, -1.0, float("nan"), 29.5):
            with self.subTest(invalid_rate=invalid_rate):
                with self.assertRaisesRegex(ValueError, "state replay frame rate"):
                    exact_state_replay_playback_clock(invalid_rate)

    def test_default_is_decoupled_from_policy_chunk_length(self):
        timing = resolve_dynamic_timing(16, 0, 4)

        self.assertEqual(timing.control_repeats, 1)
        self.assertEqual(timing.subframes, 1)
        self.assertEqual(timing.render_repeats, (0,))

    def test_action_rate_derives_control_steps_from_rate_ratio(self):
        for control_hz, expected_repeats in ((20.0, 2), (50.0, 5)):
            with self.subTest(control_hz=control_hz):
                timing = resolve_dynamic_timing(
                    16, 0, 1, action_hz=10.0, control_hz=control_hz
                )
                self.assertEqual(timing.control_repeats, expected_repeats)
                self.assertEqual(timing.subframes, 1)
                self.assertEqual(timing.render_repeats, (expected_repeats - 1,))

    def test_one_control_step_clips_subframes(self):
        timing = resolve_dynamic_timing(16, 1, 4)

        self.assertEqual(timing.control_repeats, 1)
        self.assertEqual(timing.subframes, 1)
        self.assertEqual(timing.render_repeats, (0,))

    def test_negative_control_repeats_are_rejected(self):
        with self.assertRaisesRegex(ValueError, ">= 0"):
            resolve_dynamic_timing(16, -1, 4)

    def test_non_integral_rate_ratio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            resolve_dynamic_timing(
                16, 0, 1, action_hz=12.0, control_hz=20.0
            )
    def test_fixed_rate_source_scheduler_is_exact_for_divisor_rates(self):
        self.assertEqual(
            [control_ticks_for_source_action(index, 10.0) for index in range(4)],
            [5, 5, 5, 5],
        )
        self.assertEqual(
            [control_ticks_for_source_action(index, 5.0) for index in range(3)],
            [10, 10, 10],
        )

    def test_fixed_rate_source_scheduler_preserves_15hz_average(self):
        ticks = [
            control_ticks_for_source_action(index, 15.0)
            for index in range(15)
        ]

        self.assertEqual(ticks, [3, 3, 4] * 5)
        self.assertEqual(sum(ticks), 50)

    def test_fr3_30hz_source_is_exactly_phase_scheduled_on_50hz_control(self):
        ticks = [
            nearest_control_ticks_for_source_action(
                index,
                30.0,
                control_frequency_hz=50.0,
            )
            for index in range(30)
        ]

        self.assertEqual(ticks, [2, 1, 2] * 10)
        self.assertEqual(sum(ticks), 50)
        self.assertAlmostEqual(sum(ticks) / 50.0, len(ticks) / 30.0)
        expected_cumulative_ticks = {
            1: 2,
            2: 3,
            3: 5,
            16: 27,
            32: 53,
            300: 500,
        }
        scheduled = ticks + [
            nearest_control_ticks_for_source_action(index, 30.0, control_frequency_hz=50.0)
            for index in range(30, 300)
        ]

        for action_count, expected_ticks in expected_cumulative_ticks.items():
            with self.subTest(action_count=action_count):
                cumulative_ticks = sum(scheduled[:action_count])
                phase_error_s = (
                    cumulative_ticks / 50.0 - action_count / 30.0
                )
                self.assertEqual(cumulative_ticks, expected_ticks)
                self.assertLess(abs(phase_error_s), 0.5 / 50.0)

    def test_nearest_schedule_uses_explicit_half_up_boundaries(self):
        ticks = [
            nearest_control_ticks_for_source_action(
                index,
                2.0,
                control_frequency_hz=3.0,
            )
            for index in range(4)
        ]

        self.assertEqual(ticks, [2, 1, 2, 1])

    def test_fixed_rate_source_scheduler_rejects_unrepresentable_rates(self):
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            control_ticks_for_source_action(0, 51.0)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            control_ticks_for_source_action(-1, 10.0)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            nearest_control_ticks_for_source_action(0, 51.0)

    def test_scalar_tick_count_is_only_reported_for_divisor_rates(self):
        self.assertEqual(constant_control_ticks_per_source_action(5.0), 10)
        self.assertEqual(constant_control_ticks_per_source_action(10.0), 5)
        self.assertIsNone(constant_control_ticks_per_source_action(15.0))


if __name__ == "__main__":
    unittest.main()
