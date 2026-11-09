# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pytest

from robosuite_env.policy import policy_batch_infer as pbi
from robosuite_env.policy import policy_io as pio


@pytest.fixture()
def batch_server(msgpack):
    from robosuite_env.policy import policy_batch_server

    globals()["pbs"] = policy_batch_server
    return policy_batch_server


class FakeBatchPolicy:
    def __init__(self):
        self.calls = []

    def infer_many(self, obs_list):
        self.calls.append([obs["env_id"] for obs in obs_list])
        out = []
        for obs in obs_list:
            chunk = np.full((4, 16), float(obs["env_id"]), dtype=np.float32)
            out.append({"actions": chunk, "policy_timing": {"fake": True}})
        return out


class FakeNoiseBatchPolicy:
    def __init__(self):
        self.calls = []

    def infer_many(self, obs_list, *, noise=None):
        assert noise is not None
        arr = np.asarray(noise, dtype=np.float32)
        self.calls.append(
            {
                "env_ids": [obs["env_id"] for obs in obs_list],
                "noise": arr.copy(),
            }
        )
        return [{"actions": arr[index].copy()} for index in range(len(obs_list))]


class FakeSequentialPolicy:
    def infer(self, obs):
        return {"actions": np.asarray(obs["actions"])}


def _pose_xyzw(position, axis, degrees):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    half_angle = np.deg2rad(float(degrees)) / 2.0
    quat = np.concatenate([axis * np.sin(half_angle), [np.cos(half_angle)]])
    return np.concatenate([np.asarray(position, dtype=np.float64), quat])


def _lightweight_data_factory(asset_id):
    return types.SimpleNamespace(
        assets=types.SimpleNamespace(asset_id=asset_id),
        repo_id="dataset/fallback",
        resolve_norm_stats_asset_id=lambda: asset_id,
    )



class PolicyBatchInferTest(unittest.TestCase):
    def test_config_norm_stats_resolution_ignores_decoys(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint"
            asset_id = "dataset/specific_policy"
            expected = checkpoint / "assets" / asset_id / "norm_stats.json"
            expected.parent.mkdir(parents=True)
            expected.write_text('{"norm_stats": "expected"}\n')
            decoy = checkpoint / "assets" / "other" / "norm_stats.json"
            decoy.parent.mkdir(parents=True)
            decoy.write_text('{"norm_stats": "decoy"}\n')
            data_factory = _lightweight_data_factory(asset_id)
            train_config = types.SimpleNamespace(
                name="specific_policy",
                data=data_factory,
                assets_dirs=Path(tmp) / "config-assets",
                model=object(),
            )

            resolved, resolved_asset_id = pbi.resolve_config_norm_stats_path(
                train_config,
                checkpoint,
            )

        self.assertEqual(resolved, expected)
        self.assertEqual(resolved_asset_id, asset_id)
        self.assertNotEqual(resolved, decoy)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint"
            decoy = checkpoint / "assets" / "other" / "norm_stats.json"
            decoy.parent.mkdir(parents=True)
            decoy.write_text('{"norm_stats": "decoy"}\n')
            data_factory = _lightweight_data_factory("dataset/missing_policy")
            train_config = types.SimpleNamespace(
                name="missing_policy",
                data=data_factory,
                assets_dirs=Path(tmp) / "config-assets",
                model=object(),
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "dataset/missing_policy/norm_stats.json",
            ):
                pbi.resolve_config_norm_stats_path(train_config, checkpoint)

    def test_custom_factory_uses_the_asset_id_returned_by_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint"
            exact_asset_id = "dataset/from_create"
            expected = checkpoint / "assets" / exact_asset_id / "norm_stats.json"
            expected.parent.mkdir(parents=True)
            expected.write_text('{"norm_stats": "expected"}\n')
            create = mock.Mock(
                return_value=types.SimpleNamespace(asset_id=exact_asset_id)
            )
            data_factory = types.SimpleNamespace(
                assets=types.SimpleNamespace(asset_id="dataset/wrong_static_field"),
                repo_id="dataset/wrong_repo",
                create=create,
            )
            model = object()
            assets_dirs = Path(tmp) / "config-assets"
            train_config = types.SimpleNamespace(
                name="custom_policy",
                data=data_factory,
                assets_dirs=assets_dirs,
                model=model,
            )

            resolved, resolved_asset_id = pbi.resolve_config_norm_stats_path(
                train_config,
                checkpoint,
            )

        self.assertEqual(resolved, expected)
        self.assertEqual(resolved_asset_id, exact_asset_id)
        create.assert_called_once_with(assets_dirs, model)

    def test_policy_loader_delegates_exact_config_asset_to_openpi(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint"
            asset_id = "dataset/specific_policy"
            norm_stats = checkpoint / "assets" / asset_id / "norm_stats.json"
            norm_stats.parent.mkdir(parents=True)
            # The loader resolves the state quaternion hemisphere from these
            # stats, so they must be a real 9D YUBI state (qw straddles zero).
            state_q01 = [-0.5] * 9 + [0.0] * 23
            state_q99 = [0.5] * 9 + [0.0] * 23
            norm_stats.write_bytes(
                json.dumps(
                    {"norm_stats": {"state": {"q01": state_q01, "q99": state_q99}}}
                ).encode()
                + b"\n"
            )
            expected_sha256 = hashlib.sha256(norm_stats.read_bytes()).hexdigest()

            data_factory = _lightweight_data_factory(asset_id)
            config = types.SimpleNamespace(
                name="specific_policy",
                data=data_factory,
                assets_dirs=Path(tmp) / "config-assets",
                model=types.SimpleNamespace(action_horizon=32),
            )
            loaded_policy = types.SimpleNamespace()
            create_trained_policy = mock.Mock(return_value=loaded_policy)

            policy_config_module = types.ModuleType("openpi.policies.policy_config")
            policy_config_module.create_trained_policy = create_trained_policy
            policies_module = types.ModuleType("openpi.policies")
            policies_module.policy_config = policy_config_module
            openpi_module = types.ModuleType("openpi")
            openpi_module.policies = policies_module
            deploy_config_module = types.ModuleType("robosuite_env.policy.yubi_openpi.config_yubi")
            deploy_config_module.get_config = mock.Mock(return_value=config)

            with (
                mock.patch.dict(
                    "sys.modules",
                    {
                        "openpi": openpi_module,
                        "openpi.policies": policies_module,
                        "openpi.policies.policy_config": policy_config_module,
                        "robosuite_env.policy.yubi_openpi.config_yubi": deploy_config_module,
                    },
                ),
                mock.patch.object(
                    pbi.os,
                    "walk",
                    side_effect=AssertionError("norm stats must not be searched"),
                ),
                # The locked-OpenPI guard has its own tests
                # (test_locked_imports.py); the fake OpenPI here is not an
                # installed distribution, so record the call instead.
                mock.patch(
                    "real2sim.locked_imports.require_locked_imports"
                ) as require_locked,
            ):
                result = pbi.load_openpi_policy(
                    checkpoint,
                    config="specific_policy",
                    config_src="deploy",
                    action_horizon=32,
                    instruction="test instruction",
                    repo_root=Path(tmp),
                )

        self.assertIs(result, loaded_policy)
        require_locked.assert_called_once()
        create_trained_policy.assert_called_once_with(
            config,
            checkpoint.resolve(),
            default_prompt="test instruction",
        )
        metadata = pbi.policy_load_metadata(result)
        self.assertEqual(metadata["norm_stats_asset_id"], asset_id)
        self.assertEqual(metadata["norm_stats_path"], str(norm_stats))
        self.assertEqual(metadata["norm_stats_sha256"], expected_sha256)
        self.assertEqual(metadata["policy_state_quat_hemisphere"], "positive")
        resolution = metadata["policy_state_quat_hemisphere_resolution"]
        self.assertEqual(resolution["classification"], "mixed")
        self.assertEqual(resolution["norm_stats_sha256"], expected_sha256)

    def test_policy_sample_num_steps_uses_configured_value_and_default(self):
        configured = types.SimpleNamespace(_sample_kwargs={"num_steps": np.int64(12)})
        defaulted = types.SimpleNamespace(_sample_kwargs={})

        self.assertEqual(pbi.policy_sample_num_steps(configured), 12)
        self.assertEqual(pbi.policy_sample_num_steps(defaulted), 10)
        for invalid in (0, True, 1.5):
            with self.subTest(invalid=invalid):
                policy = types.SimpleNamespace(_sample_kwargs={"num_steps": invalid})
                with self.assertRaisesRegex(ValueError, "num_steps"):
                    pbi.policy_sample_num_steps(policy)

    def test_stack_and_unstack_pytrees(self):
        items = [
            {"image": {"left": np.zeros((2, 3), dtype=np.uint8)}, "state": np.array([1.0, 2.0])},
            {"image": {"left": np.ones((2, 3), dtype=np.uint8)}, "state": np.array([3.0, 4.0])},
        ]

        stacked = pbi.stack_pytrees(items)
        self.assertEqual(stacked["image"]["left"].shape, (2, 2, 3))
        self.assertEqual(stacked["state"].shape, (2, 2))

        unstacked = pbi.unstack_pytree(stacked, 2)
        np.testing.assert_array_equal(unstacked[0]["image"]["left"], items[0]["image"]["left"])
        np.testing.assert_array_equal(unstacked[1]["state"], items[1]["state"])

    def test_prepare_action_conditioning_supports_mixed_batch(self):
        prefix = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)

        actions, mask = pbi.prepare_action_conditioning(
            [prefix, None], action_horizon=5, action_dim=4
        )

        self.assertEqual(actions.shape, (2, 5, 4))
        self.assertEqual(mask.shape, (2, 5, 1))
        np.testing.assert_array_equal(actions[0, :2], prefix)
        np.testing.assert_array_equal(actions[0, 2:], 0.0)
        np.testing.assert_array_equal(actions[1], 0.0)
        np.testing.assert_array_equal(mask[0, :, 0], [True, True, False, False, False])
        self.assertFalse(mask[1].any())

        actions, mask = pbi.prepare_action_conditioning(
            [None, None], action_horizon=5, action_dim=4
        )
        self.assertIsNone(actions)
        self.assertIsNone(mask)

        with self.assertRaises(ValueError):
            pbi.prepare_action_conditioning(
                [np.zeros((2, 3))], action_horizon=5, action_dim=4
            )
        with self.assertRaises(ValueError):
            pbi.prepare_action_conditioning(
                [np.zeros((6, 4))], action_horizon=5, action_dim=4
            )

    def test_conditioning_does_not_silently_use_custom_infer_many(self):
        prefix = np.zeros((8, 16), dtype=np.float32)
        obs = pio.add_yubi_action_conditioning({"prompt": "pick", "env_id": 7}, prefix)
        policy = FakeBatchPolicy()

        with self.assertRaisesRegex(TypeError, "requires OpenPI Policy internals"):
            pbi.infer_many(policy, [obs])
        self.assertEqual(policy.calls, [])

        for key in (
            "observation.pose.left_hand_root.relative",
            "observation.pose.right_hand_root.relative",
            "action.joint_states",
        ):
            with self.subTest(key=key):
                policy = FakeBatchPolicy()
                obs = {"env_id": 7, key: np.zeros((8, 2), dtype=np.float32)}

                with self.assertRaisesRegex(TypeError, "requires OpenPI Policy internals"):
                    pbi.infer_many(policy, [obs])
                self.assertEqual(policy.calls, [])

    def test_unconditioned_request_still_uses_custom_infer_many(self):
        policy = FakeBatchPolicy()

        result = pbi.infer_many(policy, [{"env_id": 7}])

        self.assertEqual(policy.calls, [[7]])
        np.testing.assert_array_equal(
            result[0]["actions"], np.full((4, 16), 7.0, dtype=np.float32)
        )

    def test_custom_infer_many_broadcasts_fixed_noise(self):
        policy = FakeNoiseBatchPolicy()
        noise = np.arange(4 * 16, dtype=np.float32).reshape(4, 16)

        results = pbi.infer_many(
            policy,
            [{"env_id": 1}, {"env_id": 2}],
            noise=noise,
        )

        self.assertEqual(policy.calls[0]["noise"].shape, (2, 4, 16))
        np.testing.assert_array_equal(results[0]["actions"], noise)
        np.testing.assert_array_equal(results[1]["actions"], noise)

    def test_fixed_noise_is_not_silently_ignored_by_sequential_policy(self):
        with self.assertRaisesRegex(TypeError, "fixed RF noise requires"):
            pbi.infer_many(
                FakeSequentialPolicy(),
                [{"actions": np.zeros((4, 16), dtype=np.float32)}],
                noise=np.zeros((1, 4, 16), dtype=np.float32),
            )


# --------------------------------------------------------------------------
# Fail-closed validation helpers.
# --------------------------------------------------------------------------


class PolicyBatchInferValidationTest(unittest.TestCase):
    def test_fixed_noise_leading_dim_must_match_the_batch(self):
        noise = np.zeros((3, 4, 16), dtype=np.float32)

        self.assertIsNone(pbi._prepare_noise(None, 2))
        self.assertEqual(pbi._prepare_noise(noise, 3).shape, (3, 4, 16))
        broadcast = pbi._prepare_noise(np.zeros((4, 16), dtype=np.float32), 2)
        self.assertEqual(broadcast.shape, (2, 4, 16))
        with self.assertRaisesRegex(ValueError, "does not match batch size 2"):
            pbi._prepare_noise(noise, 2)

    def test_stack_and_unstack_reject_mismatched_pytrees(self):
        with self.assertRaisesRegex(ValueError, "cannot stack an empty sequence"):
            pbi.stack_pytrees([])
        with self.assertRaisesRegex(ValueError, "different dict keys"):
            pbi.stack_pytrees([{"a": 1}, {"b": 1}])
        with self.assertRaisesRegex(ValueError, "different tuple arity"):
            pbi.stack_pytrees([(1, 2), (1,)])
        with self.assertRaisesRegex(ValueError, "different list arity"):
            pbi.stack_pytrees([[1, 2], [1]])
        with self.assertRaisesRegex(ValueError, "batched leaf has leading dim 3"):
            pbi.unstack_pytree(np.zeros((3, 2)), 2)

    def test_prepare_action_conditioning_rejects_bad_prefix_geometry(self):
        with self.assertRaisesRegex(
            ValueError, r"normalized action prefix must have shape \[steps, 4\]"
        ):
            pbi.prepare_action_conditioning(
                [np.zeros((2, 3), dtype=np.float32)], action_horizon=5, action_dim=4
            )
        with self.assertRaisesRegex(
            ValueError, r"action prefix length must be in \[1, 2\]"
        ):
            pbi.prepare_action_conditioning(
                [np.zeros((3, 4), dtype=np.float32)], action_horizon=2, action_dim=4
            )


if __name__ == "__main__":
    unittest.main()
