# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import json
import unittest

import numpy as np

from robosuite_env.policy import policy_io as pio
from robosuite_env.policy.rel2abs import apply_delta_local


def same_rotation(q0, q1, atol=1e-9):
    q0 = pio.normalize_quat_wxyz(q0)
    q1 = pio.normalize_quat_wxyz(q1)
    return abs(float(np.dot(q0, q1))) >= 1.0 - atol


def _pose_xyzw(position, axis, degrees):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    half_angle = np.deg2rad(float(degrees)) / 2.0
    quat = np.concatenate([axis * np.sin(half_angle), [np.cos(half_angle)]])
    return np.concatenate([np.asarray(position, dtype=float), quat])


class PolicyIoTest(unittest.TestCase):
    def test_yubi_action_layout(self):
        self.assertEqual(pio.LEFT_ACTION_SLICE, slice(0, 7))
        self.assertEqual(pio.RIGHT_ACTION_SLICE, slice(7, 14))
        self.assertEqual(pio.GRIP_L_INDEX, 14)
        self.assertEqual(pio.GRIP_R_INDEX, 15)

    def test_action_chunk_validation(self):
        chunk = np.zeros((4, 16), dtype=float)
        self.assertIs(pio.validate_action_chunk(chunk), chunk)
        with self.assertRaises(ValueError):
            pio.validate_action_chunk(np.zeros(16))
        with self.assertRaises(ValueError):
            pio.validate_action_chunk(np.zeros((4, 15)))
        pio.validate_exec_horizon(chunk, 4)
        with self.assertRaises(ValueError):
            pio.validate_exec_horizon(chunk, 0)
        with self.assertRaises(ValueError):
            pio.validate_exec_horizon(chunk, 5)
        for invalid in (np.nan, np.inf, -np.inf):
            bad = chunk.copy()
            bad[0, 0] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "finite"
            ):
                pio.validate_action_chunk(bad)

    def test_unclamped_fr3_action_validation_preserves_valid_raw_deltas(self):
        chunk = np.zeros((2, 16), dtype=np.float32)
        chunk[:, 6] = 2.0
        chunk[:, 13] = 3.0
        chunk[:, :3] = [0.2, -0.1, 0.3]

        validated = pio.validate_unclamped_fr3_policy_action_chunk(chunk)

        self.assertIs(validated, chunk)
        np.testing.assert_array_equal(validated, chunk)

        chunk = np.zeros((1, 16), dtype=float)
        chunk[0, 6] = 1.0
        with self.assertRaisesRegex(ValueError, "quaternion must be non-zero"):
            pio.validate_unclamped_fr3_policy_action_chunk(chunk)

    def test_fr3_policy_action_clamp_matches_deploy_relative_pose_limits(self):
        chunk = np.zeros((2, 20), dtype=np.float32)
        chunk[:, 6] = 1.0
        chunk[:, 13] = 1.0
        chunk[:, 14:] = np.arange(12, dtype=np.float32).reshape(2, 6)
        chunk[0, :3] = [0.06, 0.08, 0.0]
        chunk[0, 7:10] = [-0.09, 0.0, 0.12]
        rotation_axis = np.array([1.0, -2.0, 2.0]) / 3.0
        raw_half_angle = np.deg2rad(40.0) / 2.0
        chunk[0, 3:7] = np.r_[
            rotation_axis * np.sin(raw_half_angle),
            np.cos(raw_half_angle),
        ]
        chunk[0, 10:14] = -chunk[0, 3:7]
        original = chunk.copy()

        clamped = pio.clamp_fr3_policy_action_chunk(chunk)

        np.testing.assert_array_equal(chunk, original)
        self.assertEqual(clamped.dtype, chunk.dtype)
        self.assertAlmostEqual(
            float(np.linalg.norm(clamped[0, :3])),
            pio.FR3_POLICY_MAX_REL_POSE_TRANSLATION_M,
            places=7,
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(clamped[0, 7:10])),
            pio.FR3_POLICY_MAX_REL_POSE_TRANSLATION_M,
            places=7,
        )
        np.testing.assert_allclose(clamped[0, :3] / clamped[0, 0], [1.0, 4.0 / 3.0, 0.0])
        np.testing.assert_allclose(clamped[0, 7:10] / clamped[0, 9], [-0.75, 0.0, 1.0])
        for quaternion in (clamped[0, 3:7], clamped[0, 10:14]):
            angle = 2.0 * np.arctan2(np.linalg.norm(quaternion[:3]), quaternion[3])
            self.assertAlmostEqual(
                float(angle),
                pio.FR3_POLICY_MAX_REL_POSE_ROTATION_RAD,
                places=7,
            )
            np.testing.assert_allclose(
                quaternion[:3] / np.linalg.norm(quaternion[:3]),
                rotation_axis,
                atol=1e-6,
            )
        np.testing.assert_array_equal(clamped[:, 14:], chunk[:, 14:])

    def test_fr3_policy_action_clamp_keeps_small_steps_and_sanitizes_zero_quat(self):
        chunk = np.zeros((1, pio.ACTION_DIM), dtype=np.float64)
        chunk[0, :7] = [0.01, -0.02, 0.03, 0.0, 0.0, 0.0, 1.0]
        chunk[0, 7:10] = [0.001, 0.002, -0.003]
        chunk[0, 14:] = [0.25, 0.75]

        clamped = pio.clamp_fr3_policy_action_chunk(chunk)

        np.testing.assert_array_equal(clamped[0, :7], chunk[0, :7])
        np.testing.assert_array_equal(clamped[0, 7:10], chunk[0, 7:10])
        np.testing.assert_array_equal(clamped[0, 10:14], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(clamped[0, 14:], chunk[0, 14:])

    def test_fr3_policy_action_clamp_rejects_invalid_input_or_limits(self):
        chunk = np.zeros((1, pio.ACTION_DIM), dtype=float)
        chunk[0, 6] = chunk[0, 13] = 1.0
        for kwargs in (
            {"max_translation_m": -0.01},
            {"max_translation_m": np.inf},
            {"max_rotation_rad": -0.01},
            {"max_rotation_rad": np.pi + 0.01},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                pio.clamp_fr3_policy_action_chunk(chunk, **kwargs)

        chunk[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "must be finite"):
            pio.clamp_fr3_policy_action_chunk(chunk)

    def test_fr3_policy_action_clamp_telemetry_reports_actual_interventions(self):
        chunk = np.zeros((3, pio.ACTION_DIM), dtype=np.float32)
        chunk[:, 6] = 1.0
        chunk[:, 13] = 1.0
        chunk[0, :3] = [0.06, 0.08, 0.0]
        chunk[0, 3:7] = [np.sin(0.2), 0.0, 0.0, np.cos(0.2)]
        chunk[1, 7:10] = [0.0, -0.12, 0.0]
        chunk[1, 10:14] = [0.0, np.sin(0.25), 0.0, np.cos(0.25)]
        chunk[2, 3:7] = 0.0
        original = chunk.copy()

        clamped, telemetry = pio.clamp_fr3_policy_action_chunk_with_telemetry(
            chunk
        )
        established = pio.clamp_fr3_policy_action_chunk(chunk)

        np.testing.assert_array_equal(chunk, original)
        np.testing.assert_array_equal(clamped, established)
        self.assertTrue(telemetry["enabled"])
        self.assertEqual(telemetry["measurement"], "applied")
        self.assertTrue(telemetry["intervention_occurred"])
        self.assertEqual(telemetry["rows_received"], 3)
        self.assertEqual(telemetry["rows_evaluated"], 3)
        self.assertEqual(telemetry["rows_clipped"], 2)
        self.assertEqual(telemetry["arm_rows_evaluated"], 6)
        self.assertEqual(telemetry["arm_rows_clipped"], 2)
        self.assertEqual(telemetry["translation_arm_rows_clipped"], 2)
        self.assertEqual(telemetry["rotation_arm_rows_clipped"], 2)
        self.assertEqual(telemetry["zero_quaternion_arm_rows_sanitized"], 1)

        left = telemetry["arms"]["left"]
        self.assertEqual(left["rows_clipped"], 1)
        self.assertEqual(left["translation_rows_clipped"], 1)
        self.assertEqual(left["rotation_rows_clipped"], 1)
        self.assertEqual(left["zero_quaternion_rows_sanitized"], 1)
        self.assertAlmostEqual(left["max_translation_input_norm_m"], 0.1, places=7)
        self.assertAlmostEqual(
            left["max_translation_output_norm_m"],
            pio.FR3_POLICY_MAX_REL_POSE_TRANSLATION_M,
            places=7,
        )
        self.assertAlmostEqual(left["max_translation_clipped_amount_m"], 0.05, places=7)
        self.assertAlmostEqual(left["max_rotation_input_rad"], 0.4, places=6)
        self.assertAlmostEqual(
            left["max_rotation_output_rad"],
            pio.FR3_POLICY_MAX_REL_POSE_ROTATION_RAD,
            places=6,
        )
        self.assertAlmostEqual(
            left["max_rotation_clipped_amount_rad"],
            0.4 - pio.FR3_POLICY_MAX_REL_POSE_ROTATION_RAD,
            places=6,
        )

        right = telemetry["arms"]["right"]
        self.assertEqual(right["rows_clipped"], 1)
        self.assertEqual(right["translation_rows_clipped"], 1)
        self.assertEqual(right["rotation_rows_clipped"], 1)
        self.assertAlmostEqual(right["max_translation_input_norm_m"], 0.12, places=7)
        self.assertAlmostEqual(right["max_translation_clipped_amount_m"], 0.07, places=7)
        self.assertAlmostEqual(right["max_rotation_input_rad"], 0.5, places=6)
        json.dumps(telemetry, allow_nan=False)

    def test_fr3_policy_action_clamp_telemetry_aggregates_and_marks_disabled(self):
        clipped_chunk = np.zeros((2, pio.ACTION_DIM), dtype=np.float64)
        clipped_chunk[:, 6] = clipped_chunk[:, 13] = 1.0
        clipped_chunk[0, 0] = 0.10
        _, clipped = pio.clamp_fr3_policy_action_chunk_with_telemetry(
            clipped_chunk
        )

        safe_chunk = np.zeros((1, pio.ACTION_DIM), dtype=np.float64)
        safe_chunk[:, 6] = safe_chunk[:, 13] = 1.0
        safe_chunk[0, 7] = 0.01
        _, safe = pio.clamp_fr3_policy_action_chunk_with_telemetry(safe_chunk)
        summary = pio.aggregate_fr3_policy_action_clamp_telemetry(
            [clipped, safe], enabled=True
        )

        self.assertTrue(summary["enabled"])
        self.assertTrue(summary["intervention_occurred"])
        self.assertEqual(summary["chunks_received"], 2)
        self.assertEqual(summary["chunks_evaluated"], 2)
        self.assertEqual(summary["chunks_with_clipping"], 1)
        self.assertEqual(summary["rows_received"], 3)
        self.assertEqual(summary["rows_clipped"], 1)
        self.assertEqual(summary["arms"]["left"]["translation_rows_clipped"], 1)
        self.assertEqual(summary["arms"]["right"]["translation_rows_clipped"], 0)
        self.assertAlmostEqual(
            summary["arms"]["left"]["max_translation_input_norm_m"], 0.10
        )

        disabled = pio.fr3_policy_action_clamp_not_applied_telemetry(safe_chunk)
        disabled_summary = pio.aggregate_fr3_policy_action_clamp_telemetry(
            [disabled], enabled=False
        )
        self.assertFalse(disabled_summary["enabled"])
        self.assertEqual(disabled_summary["measurement"], "not_applied")
        self.assertFalse(disabled_summary["intervention_occurred"])
        self.assertEqual(disabled_summary["chunks_received"], 1)
        self.assertEqual(disabled_summary["chunks_evaluated"], 0)
        self.assertEqual(disabled_summary["rows_received"], 1)
        self.assertEqual(disabled_summary["rows_evaluated"], 0)
        self.assertEqual(disabled_summary["rows_clipped"], 0)
        self.assertIsNone(
            disabled_summary["arms"]["left"]["max_translation_input_norm_m"]
        )
        json.dumps(disabled_summary, allow_nan=False)

        with self.assertRaisesRegex(ValueError, "enabled state disagrees"):
            pio.aggregate_fr3_policy_action_clamp_telemetry(
                [disabled], enabled=True
            )

    def test_pose_xyzw_compose_and_inverse_with_ninety_degree_rotation(self):
        root_half = np.sqrt(0.5)
        first = np.array([1.0, 2.0, 3.0, 0.0, 0.0, root_half, root_half])
        second = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        composed = pio.compose_pose_xyzw(first, second)
        identity = pio.compose_pose_xyzw(pio.invert_pose_xyzw(first), first)

        np.testing.assert_allclose(composed[:3], [1.0, 3.0, 3.0], atol=1e-12)
        self.assertTrue(same_rotation(composed[3:7], first[3:7]))
        np.testing.assert_allclose(identity, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], atol=1e-12)

    def test_left_reference_rebase_random_se3_preserves_complete_segment(self):
        rng = np.random.default_rng(20260715)

        def random_pose(position_scale=0.3):
            quat = rng.normal(size=4)
            quat /= np.linalg.norm(quat)
            return np.concatenate([rng.normal(scale=position_scale, size=3), quat])

        for _ in range(25):
            source_h = random_pose()
            destination_h = random_pose()
            actions = np.zeros((4, 18), dtype=np.float64)
            for row in actions:
                row[:7] = random_pose(position_scale=0.02)
                row[7:14] = random_pose(position_scale=0.02)
                row[3:7] *= rng.uniform(0.25, 2.0)
                row[10:14] *= rng.uniform(0.25, 2.0)
            actions[:, 14:] = rng.normal(size=(4, 4))

            rebased = pio.rebase_yubi_action_segment_left_reference(
                actions, source_h, destination_h
            )
            source_end = pio.propagate_yubi_inter_hand_state(source_h, actions)
            destination_end = pio.propagate_yubi_inter_hand_state(destination_h, rebased)

            np.testing.assert_allclose(destination_end[:3], source_end[:3], atol=2e-11)
            self.assertTrue(same_rotation(destination_end[3:7], source_end[3:7], atol=2e-11))
            np.testing.assert_array_equal(rebased[0, :7], actions[0, :7])
            np.testing.assert_array_equal(rebased[0, 14:], actions[0, 14:])
            np.testing.assert_array_equal(rebased[1:], actions[1:])

    def test_left_reference_identity_rebase_is_bit_exact(self):
        inter_hand = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
        actions = np.zeros((3, 17), dtype=np.float32)
        actions[:, 6] = 1.0
        actions[:, 13] = 1.0
        actions[:, 14:] = np.arange(9, dtype=np.float32).reshape(3, 3)

        rebased = pio.rebase_yubi_action_segment_left_reference(
            actions, inter_hand, inter_hand.copy()
        )

        self.assertEqual(rebased.dtype, actions.dtype)
        np.testing.assert_array_equal(rebased, actions)

    def test_pose_left_multiply_affine_matches_se3_without_normalizing_operand(self):
        root_half = np.sqrt(0.5)
        correction = np.array([0.3, -0.2, 0.1, 0.0, 0.0, root_half, root_half])
        latent_pose = np.array([0.4, 0.5, -0.6, 0.2, -0.4, 0.7, 1.3])

        matrix, bias = pio.pose_left_multiply_affine_xyzw(correction)
        affine_result = matrix @ latent_pose + bias
        manual_position = (
            correction[:3]
            + pio.quat_wxyz_to_matrix(pio.xyzw_to_wxyz(correction[3:7])) @ latent_pose[:3]
        )
        manual_quat = pio.wxyz_to_xyzw(
            pio.quat_mul_wxyz(
                pio.xyzw_to_wxyz(correction[3:7]),
                pio.xyzw_to_wxyz(latent_pose[3:7]),
            )
        )

        np.testing.assert_allclose(affine_result[:3], manual_position, atol=1e-12)
        np.testing.assert_allclose(affine_result[3:7], manual_quat, atol=1e-12)
        self.assertNotAlmostEqual(float(np.linalg.norm(affine_result[3:7])), 1.0, places=3)

    def test_left_reference_helpers_validate_pose_units_and_action_nonzero_quaternions(self):
        identity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        with self.assertRaisesRegex(ValueError, r"shape \(7,\)"):
            pio.compose_pose_xyzw(np.zeros(6), identity)
        nonfinite = identity.copy()
        nonfinite[0] = np.nan
        with self.assertRaisesRegex(ValueError, "must be finite"):
            pio.invert_pose_xyzw(nonfinite)
        nonunit = identity.copy()
        nonunit[6] = 2.0
        with self.assertRaisesRegex(ValueError, "unit length"):
            pio.left_reference_action_correction(nonunit, identity)

        actions = np.zeros((1, 16), dtype=float)
        actions[0, 6] = 1.0
        actions[0, 13] = 1.0
        scaled_actions = actions.copy()
        scaled_actions[0, 6] = 2.0
        scaled_actions[0, 13] = 0.5
        np.testing.assert_allclose(
            pio.propagate_yubi_inter_hand_state(identity, scaled_actions),
            pio.propagate_yubi_inter_hand_state(identity, actions),
            atol=1e-12,
        )
        bad_actions = actions.copy()
        bad_actions[0, 10:14] = 0.0
        with self.assertRaisesRegex(ValueError, "right quaternion.*non-zero"):
            pio.propagate_yubi_inter_hand_state(identity, bad_actions)
        with self.assertRaisesRegex(ValueError, "at least one action"):
            pio.rebase_yubi_action_segment_left_reference(
                actions[:0], identity, identity
            )

    def test_rf_static_prefix_source_respects_observation_and_live_anchor_modes(self):
        active_h = _pose_xyzw([0.3, -0.2, 0.1], [0.2, 0.8, -0.3], 37.0)
        target_h = _pose_xyzw([-0.1, 0.4, 0.2], [-0.7, 0.1, 0.6], 29.0)
        chunk = np.zeros((2, 16), dtype=float)
        chunk[0, pio.LEFT_ACTION_SLICE] = _pose_xyzw(
            [0.02, -0.01, 0.03], [1.0, 0.2, -0.1], 11.0
        )
        chunk[0, pio.RIGHT_ACTION_SLICE] = _pose_xyzw(
            [-0.03, 0.04, 0.01], [-0.3, 0.9, 0.2], 17.0
        )
        chunk[1, pio.LEFT_ACTION_SLICE] = _pose_xyzw(
            [0.01, 0.02, -0.02], [0.2, -0.5, 1.0], 7.0
        )
        chunk[1, pio.RIGHT_ACTION_SLICE] = _pose_xyzw(
            [0.04, -0.02, 0.03], [0.8, 0.1, -0.4], 13.0
        )

        planned, planned_mode = pio.rf_conditioning_source_inter_hand(
            active_h, chunk, 1, target_h, policy_delta_anchor="observation"
        )
        live, live_mode = pio.rf_conditioning_source_inter_hand(
            active_h, chunk, 1, target_h, policy_delta_anchor="live"
        )

        np.testing.assert_allclose(
            planned,
            pio.propagate_yubi_inter_hand_state(active_h, chunk[:1]),
            atol=1e-12,
        )
        np.testing.assert_allclose(live, target_h, atol=1e-15)
        self.assertEqual(planned_mode, "active_observation_planned_boundary")
        self.assertEqual(live_mode, "target_observation_live")
        with self.assertRaisesRegex(ValueError, "policy_delta_anchor"):
            pio.rf_conditioning_source_inter_hand(
                active_h, chunk, 1, target_h, policy_delta_anchor="invalid"
            )

    def test_rf_overlap_tail_and_execution_start(self):
        chunk = np.arange(32 * 16, dtype=np.float32).reshape(32, 16)

        for exec_steps, overlap, chunk_steps in (
            (16, 1, 32),
            (16, 4, 32),
            (16, 8, 32),
            (8, 8, 16),
        ):
            with self.subTest(exec_steps=exec_steps, overlap=overlap):
                execution_start = pio.validate_rf_overlap(
                    exec_steps,
                    overlap,
                    chunk_steps,
                )
                np.testing.assert_array_equal(
                    pio.rf_overlap_conditioning_tail(
                        chunk[:chunk_steps],
                        execution_start=0,
                        exec_steps=exec_steps,
                        overlap_steps=overlap,
                    ),
                    chunk[exec_steps - overlap : exec_steps],
                )
                np.testing.assert_array_equal(
                    pio.rf_overlap_conditioning_tail(
                        chunk[:chunk_steps],
                        execution_start=execution_start,
                        exec_steps=exec_steps,
                        overlap_steps=overlap,
                    ),
                    chunk[exec_steps : exec_steps + overlap],
                )
                self.assertEqual(
                    list(range(execution_start, execution_start + exec_steps)),
                    list(range(overlap, overlap + exec_steps)),
                )

    def test_rf_overlap_validation(self):
        self.assertEqual(pio.validate_rf_overlap(16, 0, 16), 0)
        self.assertEqual(pio.validate_rf_overlap(16, 16, 32), 16)
        with self.assertRaises(ValueError):
            pio.validate_rf_overlap(16, -1, 32)
        with self.assertRaises(ValueError):
            pio.validate_rf_overlap(16, 17, 33)
        with self.assertRaises(ValueError):
            pio.validate_rf_overlap(16, 8, 23)
        with self.assertRaises(ValueError):
            pio.rf_overlap_conditioning_tail(
                np.zeros((20, 16)), execution_start=8, exec_steps=16, overlap_steps=8
            )

    def test_rf_overlap_scheduler_infers_midpoint_and_activates_at_boundary(self):
        cases = (
            (16, 8, 40, [8, 24], [16, 32]),
            (16, 1, 33, [15, 31], [16, 32]),
            (16, 4, 33, [12, 28], [16, 32]),
            (8, 8, 17, [0, 8, 16], [8, 16]),
        )
        for exec_steps, overlap, frame_count, expected_inference, expected_activation in cases:
            with self.subTest(exec_steps=exec_steps, overlap=overlap):
                block_step = 0
                inference_frames = []
                activation_frames = []
                for frame in range(frame_count):
                    infer_pending, activate_pending = pio.rf_overlap_scheduler_flags(
                        block_step,
                        exec_steps=exec_steps,
                        overlap_steps=overlap,
                    )
                    if activate_pending:
                        activation_frames.append(frame)
                        block_step = 0
                        infer_pending, _ = pio.rf_overlap_scheduler_flags(
                            block_step,
                            exec_steps=exec_steps,
                            overlap_steps=overlap,
                        )
                    if infer_pending:
                        inference_frames.append(frame)
                    block_step += 1

                self.assertEqual(inference_frames, expected_inference)
                self.assertEqual(activation_frames, expected_activation)

        for block_step in (0, 8, 16, 32):
            self.assertEqual(
                pio.rf_overlap_scheduler_flags(
                    block_step, exec_steps=16, overlap_steps=0
                ),
                (False, False),
            )
        with self.assertRaises(ValueError):
            pio.rf_overlap_scheduler_flags(-1, exec_steps=16, overlap_steps=8)

    def test_add_yubi_action_conditioning(self):
        prefix = np.arange(8 * 16, dtype=np.float64).reshape(8, 16)
        obs = {"prompt": "pick"}

        conditioned = pio.add_yubi_action_conditioning(obs, prefix)

        self.assertIsNot(conditioned, obs)
        self.assertNotIn("action.joint_states", obs)
        np.testing.assert_array_equal(
            conditioned["observation.pose.left_hand_root.relative"], prefix[:, :7]
        )
        np.testing.assert_array_equal(
            conditioned["observation.pose.right_hand_root.relative"], prefix[:, 7:14]
        )
        np.testing.assert_array_equal(conditioned["action.joint_states"], prefix[:, 14:16])
        self.assertEqual(conditioned["action.joint_states"].dtype, np.float32)

    def test_canonical_to_franka_constant(self):
        expected_R = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        )
        np.testing.assert_allclose(pio.R_CANON_TO_FRANKA, expected_R, atol=1e-10)
        self.assertTrue(same_rotation(pio.Q_CANON_TO_FRANKA_WXYZ, np.array([0.5, 0.5, 0.5, 0.5])))

    def test_fr3_hand_root_offsets_match_real_urdf(self):
        np.testing.assert_allclose(
            pio.FR3_EEF_TO_HAND_ROOT_M,
            [0.0, 0.0015, -0.0524],
            atol=0.0,
        )
        np.testing.assert_allclose(
            pio.FR3_GRIPPER_ROOT_TO_HAND_ROOT_M,
            [0.0, 0.0015, 0.0510],
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(pio.FR3_GRIPPER_ROOT_TO_HAND_ROOT_M)
            - np.asarray(pio.FR3_EEF_TO_HAND_ROOT_M),
            [0.0, 0.0, 0.1034],
            atol=1e-12,
        )

    def test_relative_hand_state_matches_dataset_formula(self):
        q_cf = pio.Q_CANON_TO_FRANKA_WXYZ
        q_fc = pio.quat_conj_wxyz(q_cf)
        q_l_canon = pio.normalize_quat_wxyz(np.array([0.92387953, 0.0, 0.0, 0.38268343]))
        q_r_canon = pio.normalize_quat_wxyz(np.array([0.8660254, 0.0, 0.5, 0.0]))
        q_l_grip = pio.normalize_quat_wxyz(pio.quat_mul_wxyz(q_l_canon, q_cf))
        q_r_grip = pio.normalize_quat_wxyz(pio.quat_mul_wxyz(q_r_canon, q_cf))
        p_l = np.array([0.1, -0.2, 0.3])
        p_r = np.array([0.4, 0.0, 0.5])

        got = pio.relative_hand_state(p_l, q_l_grip, p_r, q_r_grip, q_franka_to_canon_wxyz=q_fc)
        expected_dp = pio.quat_wxyz_to_matrix(q_l_canon).T @ (p_r - p_l)
        expected_q = pio.normalize_quat_wxyz(pio.quat_mul_wxyz(pio.quat_conj_wxyz(q_l_canon), q_r_canon))
        if expected_q[0] < 0.0:
            expected_q = -expected_q
        expected = np.concatenate([expected_dp, pio.wxyz_to_xyzw(expected_q)]).astype(np.float32)

        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_apply_policy_delta_matches_rel2abs_in_canonical_frame(self):
        q_cf = pio.Q_CANON_TO_FRANKA_WXYZ
        q_fc = pio.quat_conj_wxyz(q_cf)
        p0 = np.array([0.25, 0.05, 0.12])
        q0_canon = pio.normalize_quat_wxyz(np.array([0.96592583, 0.25881905, 0.0, 0.0]))
        q0_grip = pio.normalize_quat_wxyz(pio.quat_mul_wxyz(q0_canon, q_cf))
        rel = np.array([0.01, -0.02, 0.03, 0.0, 0.0, 0.08715574, 0.9961947])

        got_p, got_q_grip = pio.apply_policy_delta(
            p0,
            q0_grip,
            rel,
            q_franka_to_canon_wxyz=q_fc,
            q_canon_to_franka_wxyz=q_cf,
        )
        exp_p, exp_q_xyzw = apply_delta_local(p0, pio.wxyz_to_xyzw(q0_canon), rel)
        exp_q_grip = pio.normalize_quat_wxyz(pio.quat_mul_wxyz(pio.xyzw_to_wxyz(exp_q_xyzw), q_cf))

        np.testing.assert_allclose(got_p, exp_p, atol=1e-9)
        self.assertTrue(same_rotation(got_q_grip, exp_q_grip))

        p0 = np.array([0.1, 0.2, 0.3])
        q0_grip = pio.normalize_quat_wxyz(np.array([0.5, 0.5, 0.5, 0.5]))
        rel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        offset = np.array([0.01, 0.02, -0.03])

        p1, q1 = pio.apply_policy_delta(p0, q0_grip, rel, world_offset=offset)

        np.testing.assert_allclose(p1, p0 + offset, atol=1e-12)
        self.assertTrue(same_rotation(q1, q0_grip))

    def test_integrate_policy_delta_trajectory_carries_unachieved_residual(self):
        q_cf = pio.Q_CANON_TO_FRANKA_WXYZ
        anchor = np.array([0.1, 0.2, 0.3])
        actions = np.array(
            [
                [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [-0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        positions, quaternions = pio.integrate_policy_delta_trajectory(anchor, q_cf, actions)

        np.testing.assert_allclose(
            positions,
            anchor + np.array([[0.01, 0.0, 0.0], [0.03, 0.0, 0.0], [0.025, 0.0, 0.0]]),
            atol=1e-12,
        )
        for quaternion in quaternions:
            self.assertTrue(same_rotation(quaternion, q_cf))

        q_cf = pio.Q_CANON_TO_FRANKA_WXYZ
        rotating_actions = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
                [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        positions, _ = pio.integrate_policy_delta_trajectory(
            np.zeros(3),
            q_cf,
            rotating_actions,
        )

        np.testing.assert_allclose(positions[0], np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(positions[1], [0.0, 0.01, 0.0], atol=1e-12)

        overlap_actions = np.zeros((9, 7), dtype=float)
        overlap_actions[:, 6] = 1.0
        overlap_actions[:8, 0] = 0.01
        overlap_actions[8, 0] = 0.02

        positions, _ = pio.integrate_policy_delta_trajectory(
            np.array([1.0, 0.0, 0.0]),
            pio.Q_CANON_TO_FRANKA_WXYZ,
            overlap_actions,
        )

        np.testing.assert_allclose(positions[7], [1.08, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(positions[8], [1.10, 0.0, 0.0], atol=1e-12)

        with self.assertRaisesRegex(ValueError, "horizon, 7"):
            pio.integrate_policy_delta_trajectory(
                np.zeros(3), pio.Q_CANON_TO_FRANKA_WXYZ, np.zeros((2, 6))
            )
        with self.assertRaisesRegex(ValueError, "must be finite"):
            pio.integrate_policy_delta_trajectory(
                np.zeros(3),
                pio.Q_CANON_TO_FRANKA_WXYZ,
                np.array([[np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            )

    def test_anchored_policy_trajectory_composes_every_row_from_one_anchor(self):
        half = np.sqrt(0.5)
        anchor_position = np.array([1.0, 2.0, 3.0])
        anchor_quaternion = np.array([half, 0.0, 0.0, half])
        waypoints = np.array(
            [
                [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.02, 0.0, 0.0, 0.0, 0.0, half, half],
            ]
        )
        identity_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

        positions, quaternions = pio.compose_policy_anchored_trajectory(
            anchor_position,
            anchor_quaternion,
            waypoints,
            q_franka_to_canon_wxyz=identity_wxyz,
            q_canon_to_franka_wxyz=identity_wxyz,
        )

        # Both x offsets are rotated by the unchanged 90-degree observation
        # anchor. Row 1 is 2 cm from that anchor, not 1 + 2 = 3 cm from row 0.
        np.testing.assert_allclose(
            positions,
            [[1.0, 2.01, 3.0], [1.0, 2.02, 3.0]],
            atol=1e-12,
        )
        self.assertTrue(same_rotation(quaternions[0], anchor_quaternion))
        expected_second = pio.quat_mul_wxyz(
            anchor_quaternion, pio.xyzw_to_wxyz(waypoints[1, 3:7])
        )
        self.assertTrue(same_rotation(quaternions[1], expected_second))

        integrated_positions, _ = pio.integrate_policy_delta_trajectory(
            anchor_position,
            anchor_quaternion,
            waypoints,
            q_franka_to_canon_wxyz=identity_wxyz,
            q_canon_to_franka_wxyz=identity_wxyz,
        )
        self.assertFalse(np.allclose(positions[1], integrated_positions[1]))

        identity_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        valid = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "horizon, 7"):
            pio.compose_policy_anchored_trajectory(
                np.zeros(3), identity_wxyz, np.zeros((2, 6))
            )
        with self.assertRaisesRegex(ValueError, "waypoints must be finite"):
            invalid = valid.copy()
            invalid[0, 0] = np.nan
            pio.compose_policy_anchored_trajectory(
                np.zeros(3), identity_wxyz, invalid
            )
        with self.assertRaisesRegex(ValueError, "waypoint quaternion must be non-zero"):
            pio.compose_policy_anchored_trajectory(
                np.zeros(3), identity_wxyz, np.zeros((1, 7))
            )
        with self.assertRaisesRegex(ValueError, "anchor quaternion must be non-zero"):
            pio.compose_policy_anchored_trajectory(
                np.zeros(3), np.zeros(4), valid
            )

    def test_policy_pose_execution_contract_rejects_delta_only_features_for_anchored(self):
        self.assertEqual(
            pio.validate_policy_pose_execution_contract(
                "delta",
                robot="other_robot",
                policy_delta_anchor="live",
                rf_overlap=8,
                action_clamp=True,
            ),
            "delta",
        )
        self.assertEqual(
            pio.validate_policy_pose_execution_contract(
                "anchored",
                robot="fr3",
                policy_delta_anchor="observation",
                rf_overlap=0,
                action_clamp=False,
            ),
            "anchored",
        )
        invalid_contracts = (
            ({"robot": "other_robot", "policy_delta_anchor": "observation", "rf_overlap": 0,
              "action_clamp": False}, "only for FR3"),
            ({"robot": "fr3", "policy_delta_anchor": "live", "rf_overlap": 0,
              "action_clamp": False}, "fixed inference-time anchor"),
            ({"robot": "fr3", "policy_delta_anchor": "observation", "rf_overlap": 1,
              "action_clamp": False}, "rf_overlap=0"),
            ({"robot": "fr3", "policy_delta_anchor": "observation", "rf_overlap": 0,
              "action_clamp": True}, "no-policy-action-clamp"),
        )
        for kwargs, message in invalid_contracts:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    pio.validate_policy_pose_execution_contract("anchored", **kwargs)
        with self.assertRaisesRegex(ValueError, "representation"):
            pio.validate_policy_pose_execution_contract(
                "unknown",
                robot="fr3",
                policy_delta_anchor="observation",
                rf_overlap=0,
                action_clamp=False,
            )

    def test_resize_policy_image_pad_uses_injected_pad_fn(self):
        image = np.ones((4, 6, 3), dtype=np.uint8)

        def fake_pad(arr, height, width):
            self.assertEqual(arr.shape, image.shape)
            self.assertEqual(height, 224)
            self.assertEqual(width, 224)
            return np.zeros((height, width, 3), dtype=np.uint8)

        out = pio.resize_policy_image(image, "pad", pad_fn=fake_pad)
        self.assertEqual(out.shape, (224, 224, 3))
        self.assertEqual(out.dtype, np.uint8)
        with self.assertRaises(ValueError):
            pio.resize_policy_image(image, "bad-mode", pad_fn=fake_pad)

    def test_openpi_letterbox_content_geometry_matches_deployed_wrist_raster(self):
        self.assertEqual(
            pio.openpi_letterbox_content_xywh((539, 719), (224, 224)),
            (0, 28, 224, 167),
        )
        self.assertEqual(
            pio.openpi_letterbox_content_xywh((719, 539), (224, 224)),
            (28, 0, 167, 224),
        )
        self.assertEqual(
            pio.openpi_letterbox_content_xywh((480, 640), (224, 224)),
            (0, 28, 224, 168),
        )

    def test_policy_wrist_display_unpads_by_metadata_and_preserves_orientation(self):
        # Deliberately use non-black padding and black valid camera pixels.  A
        # pixel-scanning implementation would choose the wrong crop.
        policy_image = np.full((224, 224, 3), [253, 0, 251], dtype=np.uint8)
        content = policy_image[28:195]
        content[:84, :112] = [255, 0, 0]
        content[:84, 112:] = [0, 255, 0]
        content[84:, :112] = [0, 0, 255]
        content[84:, 112:] = [255, 255, 0]
        content[:12, 48:176] = 0
        original = policy_image.copy()

        display, transform = pio.prepare_policy_wrist_display_image(
            policy_image,
            resize_mode="pad",
            source_hw=(539, 719),
            output_hw=(135, 180),
        )

        self.assertEqual(display.shape, (135, 180, 3))
        self.assertEqual(transform.source_hw, (539, 719))
        self.assertEqual(transform.policy_hw, (224, 224))
        self.assertEqual(transform.policy_content_xywh, (0, 28, 224, 167))
        self.assertEqual(transform.output_content_xywh, (0, 0, 180, 135))
        self.assertEqual(transform.algorithm, "openpi_letterbox_unpad_aspect_fit_v1")
        np.testing.assert_array_equal(policy_image, original)
        np.testing.assert_array_equal(display[40, 30], [255, 0, 0])
        np.testing.assert_array_equal(display[40, 150], [0, 255, 0])
        np.testing.assert_array_equal(display[105, 30], [0, 0, 255])
        np.testing.assert_array_equal(display[105, 150], [255, 255, 0])
        self.assertFalse(np.any(np.all(display == [253, 0, 251], axis=2)))

    def test_policy_wrist_display_restores_stretched_source_aspect(self):
        policy_image = np.zeros((12, 12, 3), dtype=np.uint8)
        policy_image[:, :6] = [11, 22, 33]
        policy_image[:, 6:] = [101, 151, 201]

        display, transform = pio.prepare_policy_wrist_display_image(
            policy_image,
            resize_mode="stretch",
            source_hw=(3, 4),
            output_hw=(100, 200),
        )

        self.assertEqual(transform.policy_content_xywh, (0, 0, 12, 12))
        self.assertEqual(transform.output_content_xywh, (33, 0, 133, 100))
        np.testing.assert_array_equal(display[:, :33], 0)
        np.testing.assert_array_equal(display[:, 166:], 0)
        np.testing.assert_array_equal(display[50, 50], [11, 22, 33])
        np.testing.assert_array_equal(display[50, 150], [101, 151, 201])

    def test_policy_wrist_display_rejects_ambiguous_or_invalid_geometry(self):
        valid = np.zeros((224, 224, 3), dtype=np.uint8)
        invalid_calls = (
            ({"image": valid.astype(np.float32), "resize_mode": "pad",
              "source_hw": (539, 719)}, "uint8"),
            ({"image": valid[:, :, 0], "resize_mode": "pad",
              "source_hw": (539, 719)}, "shape"),
            ({"image": valid, "resize_mode": "pad", "source_hw": (0, 719)},
             "positive"),
            ({"image": valid, "resize_mode": "pad", "source_hw": (539.0, 719)},
             "integers"),
            ({"image": valid, "resize_mode": "pad", "source_hw": (539, 719),
              "output_hw": (135,)}, "pair"),
            ({"image": valid, "resize_mode": "guess", "source_hw": (539, 719)},
             "unknown"),
        )
        for kwargs, message in invalid_calls:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                pio.prepare_policy_wrist_display_image(**kwargs)

        with self.assertRaisesRegex(ValueError, "empty"):
            pio.openpi_letterbox_content_xywh((1, 10**9), (224, 224))

    def test_adjust_image_contrast_around_mid_gray(self):
        image = np.array([[[0, 127, 255]]], dtype=np.uint8)

        adjusted = pio.adjust_image_contrast(image, 0.7)

        np.testing.assert_array_equal(adjusted, [[[38, 127, 217]]])
        np.testing.assert_array_equal(pio.adjust_image_contrast(image, 1.0), image)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            pio.adjust_image_contrast(image, -0.1)

    def test_sim_policy_wrist_contrast_precedes_black_padding(self):
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        seen = []

        def fake_pad(arr, height, width):
            seen.append(arr.copy())
            out = np.zeros((height, width, 3), dtype=np.uint8)
            out[28:196] = 91
            return out

        out = pio.preprocess_sim_policy_wrist_image(
            image,
            contrast_factor=0.7,
            resize_mode="pad",
            pad_fn=fake_pad,
        )

        np.testing.assert_array_equal(seen[0], np.full_like(image, 38))
        np.testing.assert_array_equal(out[:28], 0)
        np.testing.assert_array_equal(out[196:], 0)
        np.testing.assert_array_equal(out[28:196], 91)


if __name__ == "__main__":
    unittest.main()
import unittest


from robosuite_env.policy.rf_action_inpainting import get_conditioned_sample_actions


class RFActionProjectionTest(unittest.TestCase):
    def test_pytorch_policy_is_rejected_explicitly(self):
        class FakePytorchPolicy:
            _is_pytorch_model = True

        with self.assertRaisesRegex(NotImplementedError, "JAX Pi0/Pi0.5"):
            get_conditioned_sample_actions(FakePytorchPolicy())


if __name__ == "__main__":
    unittest.main()
