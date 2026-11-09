# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

import numpy as np
import pytest

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


class FakeSlotSensitivePolicy:
    """Make fixed batch width and stable client slots observable in tests."""

    def __init__(self):
        self.calls = []

    def infer_many(self, obs_list):
        self.calls.append(
            [(obs["client"], obs["request"]) for obs in obs_list]
        )
        return [
            {
                "actions": np.full(
                    (2, 3),
                    obs["client"] * 1000 + obs["request"] * 10 + slot,
                    dtype=np.float32,
                )
            }
            for slot, obs in enumerate(obs_list)
        ]


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


class PolicyBatchInfraTest(unittest.TestCase):
    @pytest.mark.usefixtures("batch_server")
    def test_server_main_resolves_repository_root_after_package_move(self):
        fake_server = mock.Mock()
        fake_server.server_address = ("127.0.0.1", 8765)
        with (
            mock.patch.object(
                pbs.pbi,
                "load_openpi_policy",
                return_value=object(),
            ) as load_policy,
            mock.patch.object(
                pbs.pbi,
                "policy_sample_num_steps",
                return_value=10,
            ),
            mock.patch.object(
                pbs.pbi,
                "policy_load_metadata",
                return_value={
                    "norm_stats_asset_id": "dataset/specific_policy",
                    "norm_stats_path": (
                        "/tmp/checkpoint/assets/dataset/specific_policy/"
                        "norm_stats.json"
                    ),
                    "norm_stats_sha256": "abc123",
                },
            ),
            mock.patch.object(
                pbs,
                "BatchPolicyTCPServer",
                return_value=fake_server,
            ) as server_factory,
        ):
            result = pbs.main(
                [
                    "--policy",
                    "/tmp/checkpoint",
                    "--config",
                    "pi05_yubi_pickplace",
                    "--batch-size",
                    "1",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(load_policy.call_args.args, ("/tmp/checkpoint",))
        self.assertEqual(load_policy.call_args.kwargs["config"], "pi05_yubi_pickplace")
        self.assertEqual(load_policy.call_args.kwargs["config_src"], "deploy")
        self.assertEqual(load_policy.call_args.kwargs["action_horizon"], 0)
        self.assertEqual(
            load_policy.call_args.kwargs["repo_root"],
            Path(__file__).resolve().parents[3],
        )
        server_metadata = server_factory.call_args.kwargs["metadata"]
        self.assertEqual(
            server_metadata["norm_stats_asset_id"],
            "dataset/specific_policy",
        )
        self.assertEqual(server_metadata["norm_stats_sha256"], "abc123")
        fake_server.serve_forever.assert_called_once_with()

    @pytest.mark.usefixtures("batch_server")
    def test_server_requires_a_local_checkpoint(self):
        parser = pbs.build_arg_parser()
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--batch-size", "1"])
            # The portable WebSocket policy source was removed; the option
            # must not be accepted silently.
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    ["--portable-policy-server", "127.0.0.1:8000", "--batch-size", "1"]
                )

        with self.assertRaises(SystemExit) as raised:
            pbs.main(["--policy", "/tmp/checkpoint", "--batch-size", "1"])
        self.assertEqual(raised.exception.code, 2)

    @pytest.mark.usefixtures("batch_server")
    def test_rf_conditioning_prefix_survives_batch_transport(self):
        prefix = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
        obs = pio.add_yubi_action_conditioning({"prompt": "pick"}, prefix)

        received = pbs.unpack_message(pbs.pack_message({"obs": obs}))["obs"]

        np.testing.assert_array_equal(
            received["observation.pose.left_hand_root.relative"], prefix[:, :7]
        )
        np.testing.assert_array_equal(
            received["observation.pose.right_hand_root.relative"], prefix[:, 7:14]
        )
        np.testing.assert_array_equal(received["action.joint_states"], prefix[:, 14:16])
        self.assertEqual(received["action.joint_states"].dtype, np.float32)






    @pytest.mark.usefixtures("batch_server")
    def test_coordinator_batches_requests_and_preserves_mapping(self):
        policy = FakeBatchPolicy()
        coord = pbs.BatchCoordinator(policy, batch_size=2, max_wait_sec=None)
        results = {}
        errors = []

        def submit(env_id):
            try:
                results[env_id] = coord.submit(f"env-{env_id}", env_id, {"env_id": env_id})
            except Exception as exc:  # pragma: no cover - surfaced by assertion below.
                errors.append(exc)

        threads = [threading.Thread(target=submit, args=(i,)) for i in (1, 2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)
        coord.close()

        self.assertFalse(errors)
        self.assertEqual(sorted(results), [1, 2])
        self.assertEqual(policy.calls, [[1, 2]])
        np.testing.assert_array_equal(results[1]["actions"], np.full((4, 16), 1.0, dtype=np.float32))
        np.testing.assert_array_equal(results[2]["actions"], np.full((4, 16), 2.0, dtype=np.float32))
        self.assertEqual(results[1]["batch_policy"]["batch_size"], 2)
        self.assertEqual(results[2]["batch_policy"]["client_ids"], ["env-1", "env-2"])

        def run_ordered(entries, client_order):
            ordered_policy = FakeBatchPolicy()
            ordered_coord = pbs.BatchCoordinator(
                ordered_policy,
                batch_size=2,
                max_wait_sec=None,
                client_order=client_order,
            )
            ordered_results = {}
            ordered_errors = []

            def submit_ordered(client_id, request_id, env_id):
                try:
                    ordered_results[client_id] = ordered_coord.submit(
                        client_id,
                        request_id,
                        {"env_id": env_id},
                    )
                except Exception as exc:  # pragma: no cover - assertion below.
                    ordered_errors.append(exc)

            ordered_threads = [
                threading.Thread(target=submit_ordered, args=entry)
                for entry in entries
            ]
            for thread in ordered_threads:
                thread.start()
            for thread in ordered_threads:
                thread.join(timeout=5.0)
            ordered_coord.close()
            self.assertFalse(ordered_errors)
            return ordered_policy, ordered_results

        ordered_policy, ordered_results = run_ordered(
            [("env-2", 2, 2), ("env-1", 1, 1)],
            ["env-1", "env-2"],
        )
        self.assertEqual(ordered_policy.calls, [[1, 2]])
        for result in ordered_results.values():
            self.assertEqual(
                result["batch_policy"]["client_ids"],
                ["env-1", "env-2"],
            )

        ordered_policy, _ = run_ordered(
            [
                ("scene-b_run_01", 1, 2),
                ("scene-a_run_01", 1, 1),
            ],
            ["scene-a", "scene-b"],
        )
        self.assertEqual(ordered_policy.calls, [[1, 2]])

    @pytest.mark.usefixtures("batch_server")
    def test_retired_tcp_client_is_padded_while_survivors_stay_bit_identical(self):
        client_ids = [f"env-{index}" for index in range(5)]

        def run_wave(*, stop_client=None, stop_after=None, second_wave=False):
            policy = FakeSlotSensitivePolicy()
            server = pbs.BatchPolicyTCPServer(
                policy,
                host="127.0.0.1",
                port=0,
                batch_size=5,
                max_wait_sec=None,
                client_order=[*client_ids, *(f"next-{i}" for i in range(5))],
            )
            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.start()
            host, port = server.server_address
            clients = {
                client_id: pbs.BatchPolicyClient(
                    f"{host}:{port}", client_id=client_id, timeout_sec=3.0
                )
                for client_id in client_ids
            }
            actions = {client_id: [] for client_id in client_ids}
            errors = []

            def infer(client_id, request):
                try:
                    result = clients[client_id].infer(
                        {"client": int(client_id.rsplit("-", 1)[1]), "request": request}
                    )
                    actions[client_id].append(result["actions"].copy())
                    self.assertEqual(result["batch_policy"]["batch_size"], 5)
                except Exception as exc:  # pragma: no cover - asserted below.
                    errors.append(exc)

            try:
                active = list(client_ids)
                for request in range(1, 5):
                    threads = [
                        threading.Thread(target=infer, args=(client_id, request))
                        for client_id in active
                    ]
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join(timeout=3.0)
                    self.assertTrue(all(not thread.is_alive() for thread in threads))
                    if stop_after == request:
                        clients[stop_client].close()
                        active.remove(stop_client)
                self.assertFalse(errors)
                for client in clients.values():
                    client.close()

                if second_wave:
                    next_clients = {
                        f"next-{index}": pbs.BatchPolicyClient(
                            f"{host}:{port}",
                            client_id=f"next-{index}",
                            timeout_sec=3.0,
                        )
                        for index in range(5)
                    }
                    next_results = {}

                    def infer_next(index):
                        next_results[index] = next_clients[f"next-{index}"].infer(
                            {"client": index + 5, "request": 1}
                        )

                    next_threads = [
                        threading.Thread(target=infer_next, args=(index,))
                        for index in range(5)
                    ]
                    for thread in next_threads:
                        thread.start()
                    for thread in next_threads:
                        thread.join(timeout=3.0)
                    self.assertEqual(sorted(next_results), list(range(5)))
                    for client in next_clients.values():
                        client.close()
            finally:
                for client in clients.values():
                    client.close()
                server.shutdown()
                server_thread.join(timeout=3.0)
            return policy, actions

        baseline_policy, baseline = run_wave()
        retired_policy, retired = run_wave(
            stop_client="env-2", stop_after=2, second_wave=True
        )

        self.assertEqual(len(baseline_policy.calls), 4)
        self.assertEqual(len(retired_policy.calls), 5)
        self.assertTrue(all(len(call) == 5 for call in retired_policy.calls))
        self.assertEqual(len(retired["env-2"]), 2)
        for client_id in ("env-0", "env-1", "env-3", "env-4"):
            for actual, expected in zip(retired[client_id], baseline[client_id]):
                np.testing.assert_array_equal(actual, expected)

    @pytest.mark.usefixtures("batch_server")
    def test_crashed_tcp_client_fails_the_strict_wave_without_waiting(self):
        client_ids = [f"env-{index}" for index in range(5)]
        server = pbs.BatchPolicyTCPServer(
            FakeSlotSensitivePolicy(),
            host="127.0.0.1",
            port=0,
            batch_size=5,
            max_wait_sec=None,
            client_order=client_ids,
        )
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()
        host, port = server.server_address
        clients = {
            client_id: pbs.BatchPolicyClient(
                f"{host}:{port}", client_id=client_id, timeout_sec=3.0
            )
            for client_id in client_ids
        }
        crashed = clients["env-2"]
        crashed._sock.shutdown(_socket.SHUT_RDWR)
        crashed._sock.close()
        crashed._closed = True
        deadline = time.monotonic() + 2.0
        while server.coordinator._fatal_error is None and time.monotonic() < deadline:
            time.sleep(0.005)

        errors = {}

        def infer(client_id):
            try:
                clients[client_id].infer({"client": 1, "request": 1})
            except Exception as exc:  # pragma: no branch - required outcome.
                errors[client_id] = str(exc)

        started = time.monotonic()
        threads = [
            threading.Thread(target=infer, args=(client_id,))
            for client_id in client_ids
            if client_id != "env-2"
        ]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2.0)
            elapsed = time.monotonic() - started
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertLess(elapsed, 2.0)
            self.assertEqual(set(errors), set(client_ids) - {"env-2"})
            for message in errors.values():
                self.assertIn("env-2", message)
                self.assertIn("disconnected without a clean close", message)
        finally:
            for client in clients.values():
                client.close()
            server.shutdown()
            server_thread.join(timeout=3.0)

    @pytest.mark.usefixtures("batch_server")
    def test_failed_registration_ack_fails_the_strict_wave(self):
        client_ids = [f"env-{index}" for index in range(5)]
        server = pbs.BatchPolicyTCPServer(
            FakeSlotSensitivePolicy(),
            host="127.0.0.1",
            port=0,
            batch_size=5,
            max_wait_sec=None,
            client_order=client_ids,
        )
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()
        host, port = server.server_address
        clients = {
            client_id: pbs.BatchPolicyClient(
                f"{host}:{port}", client_id=client_id, timeout_sec=3.0
            )
            for client_id in client_ids[:4]
        }
        raw_client = _socket.create_connection((host, port), timeout=3.0)
        raw_client.settimeout(3.0)
        self.assertEqual(pbs.recv_packet(raw_client)["type"], "hello")
        real_send_packet = pbs.send_packet

        def fail_registered(sock, payload):
            if payload.get("type") == "registered":
                raise OSError("registration acknowledgment failed")
            return real_send_packet(sock, payload)

        try:
            with mock.patch.object(pbs, "send_packet", side_effect=fail_registered):
                real_send_packet(
                    raw_client,
                    {"type": "register", "client_id": client_ids[-1]},
                )
                response = pbs.recv_packet(raw_client)
                self.assertEqual(response["type"], "error")

            errors = {}

            def infer(client_id):
                try:
                    clients[client_id].infer({"client": 1, "request": 1})
                except Exception as exc:  # pragma: no branch - required outcome.
                    errors[client_id] = str(exc)

            threads = [
                threading.Thread(target=infer, args=(client_id,))
                for client_id in client_ids[:4]
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2.0)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(set(errors), set(client_ids[:4]))
            for message in errors.values():
                self.assertIn(client_ids[-1], message)
                self.assertIn("registration acknowledgment failed", message)
        finally:
            raw_client.close()
            for client in clients.values():
                client.close()
            server.shutdown()
            server_thread.join(timeout=3.0)

    @pytest.mark.usefixtures("batch_server")
    def test_keyed_noise_bank_persists_and_reuses_exact_float32_noise(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = pbs.KeyedNoiseBank(
                seed=17,
                action_horizon=4,
                action_dim=16,
                directory=temp_dir,
            )
            noise_a, meta_a = first.get("env-1", 3)
            noise_b, meta_b = first.get("env-2", 3)

            second = pbs.KeyedNoiseBank(
                seed=17,
                action_horizon=4,
                action_dim=16,
                directory=temp_dir,
            )
            noise_a_reused, meta_a_reused = second.get("env-1", 3)

            self.assertEqual(noise_a.dtype, np.float32)
            self.assertEqual(noise_a.shape, (4, 16))
            np.testing.assert_array_equal(noise_a_reused, noise_a)
            self.assertFalse(np.array_equal(noise_a, noise_b))
            self.assertFalse(meta_a["reused"])
            self.assertFalse(meta_b["reused"])
            self.assertTrue(meta_a_reused["reused"])
            self.assertEqual(meta_a_reused["sha256"], meta_a["sha256"])
            self.assertEqual(meta_a["shape"], [4, 16])
            self.assertEqual(meta_a["dtype"], np.dtype(np.float32).str)
            self.assertEqual(len(list(Path(temp_dir).glob("*.npy"))), 2)

    @pytest.mark.usefixtures("batch_server")
    def test_policy_action_shape_supports_model_and_config_attributes(self):
        direct = types.SimpleNamespace(
            _model=types.SimpleNamespace(action_horizon=32, action_dim=16)
        )
        configured = types.SimpleNamespace(
            _model=types.SimpleNamespace(
                config=types.SimpleNamespace(action_horizon=24, action_dim=12)
            )
        )

        self.assertEqual(pbs.policy_action_shape(direct), (32, 16))
        self.assertEqual(pbs.policy_action_shape(configured), (24, 12))

        horizon_fallback = types.SimpleNamespace(
            _model=types.SimpleNamespace(action_dim=16)
        )
        missing_dim = types.SimpleNamespace(_model=types.SimpleNamespace())

        self.assertEqual(
            pbs.policy_action_shape(horizon_fallback, fallback_action_horizon=32),
            (32, 16),
        )
        with self.assertRaisesRegex(ValueError, "positive policy action shape"):
            pbs.policy_action_shape(missing_dim, fallback_action_horizon=32)

    @pytest.mark.usefixtures("batch_server")
    def test_coordinator_records_keyed_noise_and_action_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "inference.jsonl"
            bank = pbs.KeyedNoiseBank(
                seed=23,
                action_horizon=4,
                action_dim=16,
                directory=Path(temp_dir) / "noise",
            )
            policy = FakeNoiseBatchPolicy()
            coord = pbs.BatchCoordinator(
                policy,
                batch_size=2,
                max_wait_sec=None,
                client_order=["env-1", "env-2"],
                noise_bank=bank,
                inference_manifest=manifest,
            )
            results = {}
            errors = []

            def submit(env_id):
                try:
                    results[env_id] = coord.submit(
                        f"env-{env_id}", 1, {"env_id": env_id}
                    )
                except Exception as exc:  # pragma: no cover - surfaced below.
                    errors.append(exc)

            threads = [threading.Thread(target=submit, args=(i,)) for i in (2, 1)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5.0)
            coord.close()

            self.assertFalse(errors)
            self.assertEqual(policy.calls[0]["env_ids"], [1, 2])
            for index, env_id in enumerate((1, 2)):
                batch_meta = results[env_id]["batch_policy"]
                self.assertEqual(batch_meta["noise"]["client_id"], f"env-{env_id}")
                self.assertEqual(batch_meta["noise"]["sequence"], 1)
                self.assertEqual(
                    batch_meta["action_sha256"],
                    pbs.array_sha256(policy.calls[0]["noise"][index]),
                )
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual([row["client_id"] for row in rows], ["env-1", "env-2"])
            self.assertEqual(
                [row["action_sha256"] for row in rows],
                [results[i]["batch_policy"]["action_sha256"] for i in (1, 2)],
            )


if __name__ == "__main__":
    unittest.main()

import importlib

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PICKPLACE_ASSET_ID = (
    "yubi/fix_actions_tf/frame_souce_table/"
    "socks_cup_penholder_box_merged"
)


def _load_deploy_config_module():
    # The production import: the deploy config is part of the installed package.
    return importlib.import_module("robosuite_env.policy.yubi_openpi.config_yubi")


@pytest.mark.parametrize(
    ("config_name", "asset_id", "action_dim"),
    (
        ("pi05_yubi_pickplace", PICKPLACE_ASSET_ID, None),
        ("pi05_cup", "pi05_cup", 32),
        ("pi05_yubi_pickplace_typo", None, None),
    ),
)
def test_deploy_config_pins_checkpoint_norm_stats_asset(
    openpi,
    config_name: str,
    asset_id: str | None,
    action_dim: int | None,
) -> None:
    config_yubi = _load_deploy_config_module()

    if asset_id is None:
        with pytest.raises(ValueError, match="Unknown YUBI config"):
            config_yubi.get_config(config_name)
        return

    config = config_yubi.get_config(config_name)
    assert config.name == config_name
    assert config.model.action_horizon == 32
    if action_dim is not None:
        assert config.model.action_dim == action_dim
    assert config.data.repo_id == asset_id
    assert config.data.assets.asset_id == asset_id


# --------------------------------------------------------------------------
# Wire transport, client protocol, and coordinator lifecycle.
# --------------------------------------------------------------------------

import socket as _socket
import time


def _detached_client(pbs_module):
    """Build a client without connecting so the wire protocol can be driven."""

    client = pbs_module.BatchPolicyClient.__new__(pbs_module.BatchPolicyClient)
    client.client_id = "env-1"
    client.timeout_sec = None
    client._seq = 0
    client._lock = threading.Lock()
    client._sock = object()
    return client


class PolicyBatchTransportTest(unittest.TestCase):
    @pytest.mark.usefixtures("batch_server")
    def test_parse_address_accepts_scheme_prefixes_and_bare_hosts(self):
        self.assertEqual(pbs.parse_address("10.0.0.4:9100"), ("10.0.0.4", 9100))
        self.assertEqual(pbs.parse_address("tcp://10.0.0.4:9100"), ("10.0.0.4", 9100))
        self.assertEqual(pbs.parse_address("policy://host:1"), ("host", 1))
        self.assertEqual(pbs.parse_address("host"), ("host", 8765))
        self.assertEqual(pbs.parse_address("host", 9000), ("host", 9000))
        self.assertEqual(pbs.parse_address(":9100"), ("127.0.0.1", 9100))
        self.assertEqual(pbs.parse_address("  host:2  "), ("host", 2))

    @pytest.mark.usefixtures("batch_server")
    def test_msgpack_transport_preserves_nested_dtypes_and_scalars(self):
        message = {
            "type": "infer",
            "client_id": "env-1",
            "sequence": 3,
            "obs": {
                "image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
                "state": np.linspace(0.0, 1.0, 5, dtype=np.float32),
                "frame": 8,
                "replan": True,
                "scale": np.float32(1.5),
            },
        }

        restored = pbs.unpack_message(pbs.pack_message(message))

        obs = restored["obs"]
        self.assertEqual(obs["image"].dtype, np.uint8)
        self.assertEqual(obs["image"].shape, (2, 2, 3))
        np.testing.assert_array_equal(obs["image"], message["obs"]["image"])
        self.assertEqual(obs["state"].dtype, np.float32)
        np.testing.assert_allclose(obs["state"], message["obs"]["state"])
        self.assertEqual(obs["frame"], 8)
        self.assertIs(obs["replan"], True)
        self.assertEqual(np.dtype(obs["scale"].dtype), np.float32)

    @pytest.mark.usefixtures("batch_server")
    def test_msgpack_transport_rejects_dtypes_it_cannot_reconstruct(self):
        for value in (
            np.zeros(2, dtype=np.complex64),
            np.array([("a", 1)], dtype=[("name", "U1"), ("value", "i4")]),
            np.array([object()], dtype=object),
        ):
            with self.assertRaisesRegex(
                ValueError, "unsupported dtype for msgpack numpy transport"
            ):
                pbs.pack_message({"obs": value})

    @pytest.mark.usefixtures("batch_server")
    def test_packet_framing_round_trips_and_fails_closed(self):
        left, right = _socket.socketpair()
        try:
            pbs.send_packet(left, {"type": "hello", "metadata": {"ah": 32}})
            self.assertEqual(
                pbs.recv_packet(right), {"type": "hello", "metadata": {"ah": 32}}
            )
            left.sendall(pbs._HEADER.pack(0))
            with self.assertRaisesRegex(ValueError, "invalid packet size 0"):
                pbs.recv_packet(right)
            left.close()
            with self.assertRaisesRegex(EOFError, "socket closed while reading packet"):
                pbs.recv_packet(right)
        finally:
            left.close()
            right.close()

    @pytest.mark.usefixtures("batch_server")
    def test_client_infer_sends_the_request_contract_and_attaches_batch_metadata(self):
        client = _detached_client(pbs)
        sent = []
        response = {
            "type": "infer_result",
            "sequence": 1,
            "result": {"actions": np.zeros((4, 16), dtype=np.float32)},
            "batch": {"batch_index": 7, "client_ids": ["env-1"]},
        }
        with mock.patch.object(pbs, "send_packet", lambda sock, msg: sent.append(msg)), \
                mock.patch.object(pbs, "recv_packet", lambda sock: response):
            result = client.infer({"env_id": 1})

        self.assertEqual(
            sent,
            [{"type": "infer", "client_id": "env-1", "sequence": 1, "obs": {"env_id": 1}}],
        )
        self.assertEqual(result["batch_policy"], response["batch"])

    @pytest.mark.usefixtures("batch_server")
    def test_client_infer_rejects_server_errors_and_stale_sequences(self):
        client = _detached_client(pbs)
        with mock.patch.object(pbs, "send_packet", lambda sock, msg: None), \
                mock.patch.object(
                    pbs, "recv_packet", lambda sock: {"type": "error", "message": "boom"}
                ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                client.infer({})

        with mock.patch.object(pbs, "send_packet", lambda sock, msg: None), \
                mock.patch.object(pbs, "recv_packet", lambda sock: {"type": "pong"}):
            with self.assertRaisesRegex(
                RuntimeError, "unexpected policy batch server response"
            ):
                client.infer({})

        stale = {"type": "infer_result", "sequence": 1, "result": {}, "batch": {}}
        with mock.patch.object(pbs, "send_packet", lambda sock, msg: None), \
                mock.patch.object(pbs, "recv_packet", lambda sock: stale):
            with self.assertRaisesRegex(
                RuntimeError, "policy batch server sequence mismatch"
            ):
                client.infer({})

    @pytest.mark.usefixtures("batch_server")
    def test_client_close_waits_for_ack_and_never_raises(self):
        client = _detached_client(pbs)
        client._sock = mock.Mock()
        sent = []
        received = []
        with mock.patch.object(pbs, "send_packet", lambda sock, msg: sent.append(msg)), \
                mock.patch.object(
                    pbs,
                    "recv_packet",
                    lambda sock: received.append(sock)
                    or {"type": "closed", "client_id": "env-1"},
                ):
            client.close()
        self.assertEqual(sent, [{"type": "close", "client_id": "env-1"}])
        self.assertEqual(received, [client._sock])

        broken = _detached_client(pbs)
        broken._sock = mock.Mock()
        broken._sock.close.side_effect = OSError("already gone")
        with mock.patch.object(
            pbs, "send_packet", mock.Mock(side_effect=OSError("closed"))
        ):
            broken.close()

    @pytest.mark.usefixtures("batch_server")
    def test_array_sha256_separates_dtype_and_shape(self):
        flat = np.zeros(6, dtype=np.float32)
        wide = np.zeros((2, 3), dtype=np.float32)
        tall = np.zeros((3, 2), dtype=np.float32)
        doubles = np.zeros((2, 3), dtype=np.float64)
        digests = {
            pbs.array_sha256(flat),
            pbs.array_sha256(wide),
            pbs.array_sha256(tall),
            pbs.array_sha256(doubles),
        }
        self.assertEqual(len(digests), 4)
        self.assertEqual(pbs.array_sha256(wide), pbs.array_sha256(wide.copy()))

    @pytest.mark.usefixtures("batch_server")
    def test_runtime_determinism_metadata_records_a_failed_jax_probe(self):
        with mock.patch.dict(sys.modules, {"jax": None}):
            metadata = pbs.runtime_determinism_metadata()

        self.assertEqual(metadata["numpy_version"], np.__version__)
        self.assertRegex(
            metadata["jax_probe_error"], r"^(ModuleNotFoundError|ImportError): "
        )
        self.assertNotIn("jax_version", metadata)
        self.assertNotIn("jax_probe_skipped", metadata)
        for key in (
            "python_hash_seed",
            "xla_flags",
            "cublas_workspace_config",
            "cuda_visible_devices",
            "omp_num_threads",
            "mkl_num_threads",
            "openblas_num_threads",
        ):
            self.assertIn(key, metadata)

    @pytest.mark.usefixtures("batch_server")
    def test_policy_action_shape_requires_a_positive_dimension(self):
        model = types.SimpleNamespace(action_horizon=32, action_dim=None, config=None)
        with self.assertRaisesRegex(ValueError, "positive policy action shape"):
            pbs.policy_action_shape(types.SimpleNamespace(_model=model))

    @pytest.mark.usefixtures("batch_server")
    def test_keyed_noise_bank_rejects_a_nonpositive_action_shape(self):
        with self.assertRaisesRegex(ValueError, "positive action shape required"):
            pbs.KeyedNoiseBank(seed=1, action_horizon=0, action_dim=16)
        with self.assertRaisesRegex(ValueError, "positive action shape required"):
            pbs.KeyedNoiseBank(seed=1, action_horizon=32, action_dim=0)

    @pytest.mark.usefixtures("batch_server")
    def test_keyed_noise_bank_without_a_directory_stays_in_memory(self):
        bank = pbs.KeyedNoiseBank(seed=7, action_horizon=4, action_dim=16)

        first, first_meta = bank.get("env-1", 0)
        second, second_meta = bank.get("env-1", 0)

        np.testing.assert_array_equal(first, second)
        self.assertIsNone(first_meta["path"])
        self.assertFalse(first_meta["reused"])
        self.assertFalse(second_meta["reused"])
        self.assertEqual(first_meta["shape"], [4, 16])
        self.assertEqual(first_meta["mode"], "keyed_noise_bank")

    @pytest.mark.usefixtures("batch_server")
    def test_keyed_noise_bank_rejects_a_corrupt_on_disk_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = pbs.KeyedNoiseBank(
                seed=7, action_horizon=4, action_dim=16, directory=tmp
            )
            bank.get("env-1", 0)
            path = next(Path(tmp).glob("seq_000000_*.npy"))
            np.save(path, np.zeros((2, 2), dtype=np.float64), allow_pickle=False)

            with self.assertRaisesRegex(ValueError, "invalid noise bank entry"):
                bank.get("env-1", 0)

    @pytest.mark.usefixtures("batch_server")
    def test_coordinator_rejects_a_nonpositive_batch_size(self):
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            pbs.BatchCoordinator(FakeBatchPolicy(), batch_size=0, max_wait_sec=None)

    @pytest.mark.usefixtures("batch_server")
    def test_coordinator_rejects_submissions_after_close(self):
        coord = pbs.BatchCoordinator(FakeBatchPolicy(), batch_size=1, max_wait_sec=None)
        coord.close()

        with self.assertRaisesRegex(RuntimeError, "policy batch coordinator is closed"):
            coord.submit("env-1", 0, {"env_id": 1})

    @pytest.mark.usefixtures("batch_server")
    def test_coordinator_fails_pending_requests_when_it_is_closed(self):
        coord = pbs.BatchCoordinator(FakeBatchPolicy(), batch_size=2, max_wait_sec=None)
        errors = []

        def worker():
            try:
                coord.submit("env-1", 0, {"env_id": 1})
            except RuntimeError as exc:
                errors.append(str(exc))

        thread = threading.Thread(target=worker)
        thread.start()
        for _ in range(200):
            with coord._cv:
                if coord._pending:
                    break
            time.sleep(0.005)
        coord.close()
        thread.join(timeout=5.0)

        self.assertEqual(errors, ["policy batch coordinator closed"])

    @pytest.mark.usefixtures("batch_server")
    def test_coordinator_partial_batch_is_released_after_max_wait(self):
        policy = FakeBatchPolicy()
        coord = pbs.BatchCoordinator(policy, batch_size=4, max_wait_sec=0.05)
        try:
            result = coord.submit("env-1", 0, {"env_id": 1})
        finally:
            coord.close()

        self.assertEqual(policy.calls, [[1]])
        self.assertEqual(result["batch_policy"]["batch_size"], 1)
        self.assertEqual(result["batch_policy"]["requested_batch_size"], 4)

    @pytest.mark.usefixtures("batch_server")
    def test_coordinator_reports_a_policy_result_count_mismatch(self):
        class ShortPolicy:
            def infer_many(self, obs_list):
                return []

        coord = pbs.BatchCoordinator(ShortPolicy(), batch_size=1, max_wait_sec=None)
        try:
            with self.assertRaisesRegex(RuntimeError, "policy returned 0 results"):
                coord.submit("env-1", 0, {"env_id": 1})
        finally:
            coord.close()


if __name__ == "__main__":
    unittest.main()
