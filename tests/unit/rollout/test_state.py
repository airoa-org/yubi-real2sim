# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import copy
import math
import unittest

import numpy as np
import pytest

from robosuite_env.rollout.state.render_state import (
    RENDER_STATE_CONTRACT_SCHEMA,
    RENDER_STATE_SCHEMA,
    build_render_state_contract,
    build_render_state_summary_contract,
    canonical_inventory_sha256,
    encode_render_state_row,
    validate_render_state_contract,
    validate_render_state_row,
    validate_render_state_rows,
)


class RenderStateContractTest(unittest.TestCase):
    def setUp(self):
        self.inventory = [
            {
                "name": "plate",
                "role": "distractor",
                "joint_name": "plate_joint",
                "qpos_address": 14,
                "dof_address": 12,
            },
            {
                "name": "cup",
                "role": "target",
                "joint_name": "cup_joint",
                "qpos_address": 7,
                "dof_address": 6,
            },
        ]
        self.contract = build_render_state_contract(
            model_signature="model-signature-a",
            nq=21,
            target="cup",
            inventory=self.inventory,
        )
        self.qpos = [float(index) / 10.0 for index in range(21)]
        self.qpos[10:14] = [1.0, 0.0, 0.0, 0.0]
        half = math.sqrt(0.5)
        self.qpos[17:21] = [half, 0.0, half, 0.0]
        self.row = {
            "frame": 0,
            "arms": {"R": {"q": [0.0] * 7}},
            **encode_render_state_row(self.qpos, self.contract),
        }

    def test_contract_and_inventory_sha_are_canonical_and_order_independent(self):
        reversed_contract = build_render_state_contract(
            model_signature="model-signature-a",
            nq=21,
            target="cup",
            inventory=list(reversed(self.inventory)),
        )
        self.assertEqual(self.contract, reversed_contract)
        self.assertEqual(
            [entry["name"] for entry in self.contract["inventory"]],
            ["cup", "plate"],
        )
        self.assertEqual(
            canonical_inventory_sha256(self.inventory),
            self.contract["inventory_sha256"],
        )
        self.assertEqual(self.contract["schema"], RENDER_STATE_CONTRACT_SCHEMA)
        self.assertEqual(self.contract["state_schema"], RENDER_STATE_SCHEMA)

    def test_contract_rejects_invalid_inventory_and_target(self):
        for field in ("qpos_address", "dof_address"):
            inventory = copy.deepcopy(self.inventory)
            inventory[0][field] = True
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "bool"):
                build_render_state_contract(
                    model_signature="sig", nq=21, target="cup", inventory=inventory
                )

        inventory = copy.deepcopy(self.inventory)
        inventory[0]["unexpected"] = 1
        with self.assertRaisesRegex(ValueError, "extra"):
            build_render_state_contract(
                model_signature="sig", nq=21, target="cup", inventory=inventory
            )

        inventory = copy.deepcopy(self.inventory)
        inventory[0]["qpos_address"] = 10
        with self.assertRaisesRegex(ValueError, "overlapping"):
            build_render_state_contract(
                model_signature="sig", nq=21, target="cup", inventory=inventory
            )

        with self.assertRaisesRegex(ValueError, "absent"):
            build_render_state_contract(
                model_signature="sig", nq=21, target="bottle", inventory=self.inventory
            )
        with self.assertRaisesRegex(ValueError, "role='target'"):
            build_render_state_contract(
                model_signature="sig", nq=21, target="plate", inventory=self.inventory
            )

    def test_contract_rejects_schema_signature_inventory_and_noncanonical_order(self):
        cases = []
        wrong_schema = copy.deepcopy(self.contract)
        wrong_schema["schema"] = "old-schema"
        cases.append((wrong_schema, "schema"))
        wrong_row_schema = copy.deepcopy(self.contract)
        wrong_row_schema["state_schema"] = "old-state-schema"
        cases.append((wrong_row_schema, "row schema"))
        wrong_sha = copy.deepcopy(self.contract)
        wrong_sha["inventory_sha256"] = "0" * 64
        cases.append((wrong_sha, "SHA-256"))
        wrong_order = copy.deepcopy(self.contract)
        wrong_order["inventory"].reverse()
        cases.append((wrong_order, "not canonical"))
        for payload, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                validate_render_state_contract(payload)

    def test_encode_preserves_full_precision_and_all_named_slices(self):
        precise = np.asarray(self.qpos, dtype=np.float64)
        precise[0] = np.nextafter(1.0, 2.0)
        payload = encode_render_state_row(precise, self.contract)
        self.assertEqual(payload["render_state"]["qpos"][0], precise[0])
        self.assertEqual(
            payload["task_objects"]["cup"]["qpos"],
            payload["render_state"]["qpos"][7:14],
        )
        self.assertEqual(
            payload["task_objects"]["plate"]["qpos"],
            payload["render_state"]["qpos"][14:21],
        )
        self.assertEqual(sorted(payload["task_objects"]), ["cup", "plate"])

    def test_row_rejects_bool_nonfinite_and_wrong_shapes(self):
        cases = []
        boolean = copy.deepcopy(self.row)
        boolean["render_state"]["qpos"][0] = True
        cases.append((boolean, "bool"))
        nonfinite = copy.deepcopy(self.row)
        nonfinite["render_state"]["qpos"][0] = np.inf
        cases.append((nonfinite, "finite"))
        wrong_full_shape = copy.deepcopy(self.row)
        wrong_full_shape["render_state"]["qpos"].pop()
        cases.append((wrong_full_shape, "shape"))
        wrong_object_shape = copy.deepcopy(self.row)
        wrong_object_shape["task_objects"]["cup"]["qpos"].pop()
        cases.append((wrong_object_shape, "shape"))
        bool_nq = copy.deepcopy(self.row)
        bool_nq["render_state"]["nq"] = True
        cases.append((bool_nq, "nq mismatch"))
        for payload, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                validate_render_state_row(payload, self.contract)

    def test_row_rejects_schema_signature_inventory_and_missing_extra_objects(self):
        cases = []
        wrong_schema = copy.deepcopy(self.row)
        wrong_schema["render_state"]["schema"] = "legacy"
        cases.append((wrong_schema, "schema"))
        wrong_signature = copy.deepcopy(self.row)
        wrong_signature["render_state"]["model_signature"] = "other"
        cases.append((wrong_signature, "signature"))
        wrong_inventory = copy.deepcopy(self.row)
        wrong_inventory["render_state"]["inventory_sha256"] = "f" * 64
        cases.append((wrong_inventory, "inventory SHA-256"))
        missing = copy.deepcopy(self.row)
        del missing["task_objects"]["plate"]
        cases.append((missing, "missing=\\['plate'\\]"))
        extra = copy.deepcopy(self.row)
        extra["task_objects"]["bowl"] = {"qpos": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}
        cases.append((extra, "extra=\\['bowl'\\]"))
        extra_state_key = copy.deepcopy(self.row)
        extra_state_key["task_objects"]["cup"]["pos"] = [0.0] * 3
        cases.append((extra_state_key, "extra"))
        for payload, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                validate_render_state_row(payload, self.contract)

    def test_row_rejects_zero_nonunit_and_slice_mismatched_quaternions(self):
        zero = copy.deepcopy(self.row)
        zero["render_state"]["qpos"][10:14] = [0.0] * 4
        zero["task_objects"]["cup"]["qpos"][3:7] = [0.0] * 4
        with self.assertRaisesRegex(ValueError, "zero quaternion"):
            validate_render_state_row(zero, self.contract)

        nonunit = copy.deepcopy(self.row)
        nonunit["render_state"]["qpos"][10:14] = [2.0, 0.0, 0.0, 0.0]
        nonunit["task_objects"]["cup"]["qpos"][3:7] = [2.0, 0.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "non-unit quaternion"):
            validate_render_state_row(nonunit, self.contract)

        mismatch = copy.deepcopy(self.row)
        mismatch["task_objects"]["cup"]["qpos"][0] = np.nextafter(
            mismatch["task_objects"]["cup"]["qpos"][0], math.inf
        )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_render_state_row(mismatch, self.contract)

    def test_summary_validates_every_row_count_completion_and_expected_contract(self):
        row_1 = copy.deepcopy(self.row)
        row_1["frame"] = 1
        summary = build_render_state_summary_contract(
            self.contract, rows_written=2, complete=True
        )
        report = validate_render_state_rows(
            [self.row, row_1], summary, expected_contract=self.contract
        )
        self.assertEqual(report["rows_written"], 2)
        self.assertEqual(report["task_object_names"], ["cup", "plate"])

        bad_second = copy.deepcopy(row_1)
        bad_second["task_objects"]["plate"]["qpos"][0] += 1e-12
        with self.assertRaisesRegex(ValueError, "row 1.*exactly match"):
            validate_render_state_rows([self.row, bad_second], summary)

        wrong_count = copy.deepcopy(summary)
        wrong_count["rows_written"] = 1
        with self.assertRaisesRegex(ValueError, "rows_written mismatch"):
            validate_render_state_rows([self.row, row_1], wrong_count)

        incomplete = copy.deepcopy(summary)
        incomplete["complete"] = False
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_render_state_rows([self.row, row_1], incomplete)

        bool_count = copy.deepcopy(summary)
        bool_count["rows_written"] = True
        with self.assertRaisesRegex(ValueError, "bool"):
            validate_render_state_rows([self.row, row_1], bool_count)

        other_contract = build_render_state_contract(
            model_signature="model-signature-b",
            nq=21,
            target="cup",
            inventory=self.inventory,
        )
        with self.assertRaisesRegex(ValueError, "expected contract"):
            validate_render_state_rows(
                [self.row, row_1], summary, expected_contract=other_contract
            )

    def test_missing_contract_is_allowed_only_for_explicit_optional_legacy_rows(self):
        self.assertIsNone(validate_render_state_contract(None, required=False))
        self.assertIsNone(
            validate_render_state_row({"frame": 0}, None, required=False)
        )
        self.assertIsNone(
            validate_render_state_rows(
                [{"frame": 0}], None, required=False
            )
        )
        with self.assertRaisesRegex(ValueError, "required"):
            validate_render_state_contract(None)
        with self.assertRaisesRegex(ValueError, "required"):
            validate_render_state_rows([], None)
        with self.assertRaisesRegex(ValueError, "without a contract"):
            validate_render_state_row(self.row, None, required=False)
        with self.assertRaisesRegex(ValueError, "without a summary contract"):
            validate_render_state_rows([self.row], None, required=False)


if __name__ == "__main__":
    unittest.main()


import unittest


from robosuite_env.rollout.state.mujoco import (
    INTEGRATION_STATE_KEY,
    capture_integration_state,
    forward_preserving_warmstart,
    restore_integration_or_legacy_state,
)

try:
    import mujoco
except ImportError:  # pragma: no cover - exercised only in environments without MuJoCo
    mujoco = None


_CONTACT_MODEL = """
<mujoco>
  <option timestep="0.002" solver="PGS" iterations="1" tolerance="0"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 0.045">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"
            friction="1 0.1 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.mark.usefixtures("mujoco")
@unittest.skipIf(mujoco is None, "MuJoCo is not installed")
class MujocoIntegrationStateTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(_CONTACT_MODEL)

    def _settled_data(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        for _ in range(20):
            mujoco.mj_step(self.model, data)
        return data

    def _integration_values(self, data):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        values = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, data, values, spec)
        return values

    def test_complete_integration_state_roundtrip_and_next_step_are_exact(self):
        source = self._settled_data()
        block = capture_integration_state(mujoco, self.model, source)
        payload = {INTEGRATION_STATE_KEY: block}
        snapshot = self._integration_values(source)

        restored = mujoco.MjData(self.model)
        restored.qacc_warmstart[:] = 123.0
        meta = restore_integration_or_legacy_state(
            mujoco, self.model, restored, payload
        )

        np.testing.assert_array_equal(self._integration_values(restored), snapshot)
        self.assertEqual(meta["source"], "mjSTATE_INTEGRATION")
        self.assertEqual(block["spec"], int(mujoco.mjtState.mjSTATE_INTEGRATION))
        self.assertEqual(block["size"], len(block["values"]))

        mujoco.mj_step(self.model, source)
        expected_next = self._integration_values(source)
        forward_preserving_warmstart(
            mujoco, self.model, restored, meta["warmstart"]
        )
        mujoco.mj_step(self.model, restored)
        np.testing.assert_array_equal(
            self._integration_values(restored), expected_next
        )

    def test_legacy_snapshot_without_warmstart_is_canonicalized(self):
        data = self._settled_data()
        data.qacc_warmstart[:] = np.arange(self.model.nv) + 1.0
        data.qfrc_applied[:] = 7.0
        data.xfrc_applied[:] = 9.0

        meta = restore_integration_or_legacy_state(
            mujoco, self.model, data, {}
        )

        self.assertEqual(meta["source"], "legacy-zero-warmstart")
        np.testing.assert_array_equal(data.qacc_warmstart, np.zeros(self.model.nv))
        np.testing.assert_array_equal(data.qfrc_applied, np.zeros(self.model.nv))
        np.testing.assert_array_equal(
            data.xfrc_applied, np.zeros((self.model.nbody, 6))
        )

    def test_legacy_snapshot_can_restore_explicit_warmstart(self):
        data = self._settled_data()
        warmstart = np.linspace(-1.0, 1.0, self.model.nv)

        meta = restore_integration_or_legacy_state(
            mujoco,
            self.model,
            data,
            {"qacc_warmstart": warmstart.tolist()},
        )

        self.assertEqual(meta["source"], "legacy-saved-warmstart")
        np.testing.assert_array_equal(data.qacc_warmstart, warmstart)

    def test_rejects_integration_state_metadata_mismatch(self):
        data = self._settled_data()
        block = capture_integration_state(mujoco, self.model, data)
        block["size"] += 1

        with self.assertRaisesRegex(ValueError, "size mismatch"):
            restore_integration_or_legacy_state(
                mujoco, self.model, data, {INTEGRATION_STATE_KEY: block}
            )


if __name__ == "__main__":
    unittest.main()


import json
import os
import stat
import tempfile
import threading
import unittest
from unittest import mock

from robosuite_env.rollout.state.cache import (
    SETTLED_INIT_STATE_OBJECTS,
    SETTLED_INIT_STATE_SCHEMA,
    atomic_write_json,
    validate_saved_settled_init_payload,
)


def _contract(objects=SETTLED_INIT_STATE_OBJECTS):
    return {
        "schema": SETTLED_INIT_STATE_SCHEMA,
        "expected_objects": list(objects),
        "model_physics_sha256": "model-sha",
        "stable_samples_required": 10,
        "thresholds": {
            "linear_speed_m_s": 0.001,
            "angular_speed_rad_s": 0.02,
        },
    }


def _payload(objects=SETTLED_INIT_STATE_OBJECTS):
    names = tuple(objects)
    object_states = {
        name: {"qpos": [0.0] * 7, "qvel": [0.0] * 6, "qacc": [0.0] * 6}
        for name in names
    }
    return {
        "version": 3,
        "object_settle_contract": _contract(names),
        "objects": object_states,
        "object_settle": {
            "source": "physics_settle",
            "validated": True,
            "stable_samples": 10,
            "thresholds": _contract(names)["thresholds"],
            "objects": object_states,
            "support": {
                name: {"supported": True, "contacts": [{}]}
                for name in names
            },
            "rule_grasp_unchanged": True,
        },
        "rule_grasp": {
            "active": False,
            "hand": None,
            "hold_mode": "weld_equality",
            "weld_eq_id": None,
            "events": [],
        },
    }


class InitStateCacheTest(unittest.TestCase):
    def test_atomic_write_publishes_complete_private_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.json")
            payload = {"value": list(range(100))}
            atomic_write_json(path, payload)

            with open(path, encoding="utf-8") as stream:
                self.assertEqual(json.load(stream), payload)
            self.assertEqual(
                stat.S_IMODE(os.stat(path).st_mode),
                0o600,
            )
            self.assertEqual(os.listdir(directory), ["state.json"])

    def test_failed_write_keeps_previous_complete_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.json")
            atomic_write_json(path, {"generation": "old"})
            with mock.patch(
                "robosuite_env.rollout.state.cache.json.dump",
                side_effect=RuntimeError("injected"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    atomic_write_json(path, {"generation": "new"})

            with open(path, encoding="utf-8") as stream:
                self.assertEqual(json.load(stream), {"generation": "old"})
            self.assertEqual(os.listdir(directory), ["state.json"])

    def test_concurrent_publish_never_leaves_mixed_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.json")
            payloads = [
                {"writer": writer, "values": [writer] * 500}
                for writer in range(8)
            ]
            threads = [
                threading.Thread(target=atomic_write_json, args=(path, payload))
                for payload in payloads
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            with open(path, encoding="utf-8") as stream:
                result = json.load(stream)
            self.assertIn(result, payloads)
            self.assertEqual(os.listdir(directory), ["state.json"])

    def test_settled_payload_contract_accepts_complete_state(self):
        validate_saved_settled_init_payload(_payload(), _contract())
        names = ("bowl", "cup", "fork", "plate", "spoon")
        validate_saved_settled_init_payload(
            _payload(names),
            _contract(names),
        )

    def test_settled_payload_contract_rejects_invalid_expected_inventory(self):
        for names in ((), ("cup", "cup"), ("",)):
            with self.subTest(names=names):
                with self.assertRaises(ValueError):
                    validate_saved_settled_init_payload(
                        _payload(("cup",)),
                        _contract(names),
                    )

    def test_settled_payload_contract_rejects_each_provenance_gap(self):
        mutations = (
            lambda value: value.update(version=2),
            lambda value: value.pop("object_settle_contract"),
            lambda value: value["object_settle"].update(source="init_state"),
            lambda value: value["object_settle"].update(stable_samples=9),
            lambda value: value["object_settle"]["support"]["plate"].update(
                supported=False
            ),
            lambda value: value["object_settle"]["objects"]["plate"][
                "qvel"
            ].__setitem__(0, 1.0),
            lambda value: value["rule_grasp"].update(active=True),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                value = json.loads(json.dumps(_payload()))
                mutate(value)
                with self.assertRaises(ValueError):
                    validate_saved_settled_init_payload(value, _contract())


if __name__ == "__main__":
    unittest.main()


import unittest


from robosuite_env.rollout.state.object_settle import (
    ConsecutiveQuietWindow,
    FreeJointMotion,
    QuietThresholds,
    resolve_transitive_support,
)


class ObjectSettleTest(unittest.TestCase):
    def setUp(self):
        self.thresholds = QuietThresholds(
            linear_speed_m_s=0.001,
            angular_speed_rad_s=0.01,
            linear_acceleration_m_s2=0.1,
            angular_acceleration_rad_s2=0.1,
        )
        self.quiet = FreeJointMotion(0.0005, 0.005, 0.05, 0.05)

    def test_motion_uses_linear_and_angular_norms(self):
        motion = FreeJointMotion.from_vectors(
            [3.0, 4.0, 0.0, 0.0, 0.0, 12.0],
            [0.0, 0.0, 5.0, 8.0, 15.0, 0.0],
        )
        self.assertEqual(motion.linear_speed_m_s, 5.0)
        self.assertEqual(motion.angular_speed_rad_s, 12.0)
        self.assertEqual(motion.linear_acceleration_m_s2, 5.0)
        self.assertEqual(motion.angular_acceleration_rad_s2, 17.0)

    def test_motion_rejects_wrong_shapes_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            FreeJointMotion.from_vectors(np.zeros(5), np.zeros(6))
        bad = np.zeros(6)
        bad[2] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            FreeJointMotion.from_vectors(np.zeros(6), bad)

    def test_all_four_thresholds_are_required(self):
        self.assertTrue(self.thresholds.accepts(self.quiet))
        for motion in (
            FreeJointMotion(0.002, 0.005, 0.05, 0.05),
            FreeJointMotion(0.0005, 0.02, 0.05, 0.05),
            FreeJointMotion(0.0005, 0.005, 0.2, 0.05),
            FreeJointMotion(0.0005, 0.005, 0.05, 0.2),
        ):
            self.assertFalse(self.thresholds.accepts(motion))

    def test_acceleration_can_be_telemetry_only(self):
        contact_chatter = FreeJointMotion(
            linear_speed_m_s=0.0005,
            angular_speed_rad_s=0.005,
            linear_acceleration_m_s2=10.0,
            angular_acceleration_rad_s2=100.0,
        )
        self.assertFalse(self.thresholds.accepts(contact_chatter))
        self.assertTrue(
            self.thresholds.accepts(
                contact_chatter,
                include_acceleration=False,
            )
        )
        moving = FreeJointMotion(0.002, 0.005, 0.0, 0.0)
        self.assertFalse(
            self.thresholds.accepts(
                moving,
                include_acceleration=False,
            )
        )

    def test_quiet_window_requires_all_objects_and_consecutive_samples(self):
        gate = ConsecutiveQuietWindow(self.thresholds, required_samples=3)
        quiet_objects = {"cup": self.quiet, "plate": self.quiet}
        self.assertFalse(gate.observe(quiet_objects))
        self.assertFalse(gate.observe(quiet_objects))
        moving_plate = {
            **quiet_objects,
            "plate": FreeJointMotion(0.01, 0.0, 0.0, 0.0),
        }
        self.assertFalse(gate.observe(moving_plate))
        self.assertEqual(gate.consecutive_samples, 0)
        self.assertFalse(gate.observe(quiet_objects))
        self.assertFalse(gate.observe(quiet_objects))
        self.assertTrue(gate.observe(quiet_objects))
        gate.reset()
        self.assertEqual(gate.consecutive_samples, 0)
        self.assertFalse(gate.observe(quiet_objects, extra_quiet=False))
        self.assertEqual(gate.consecutive_samples, 0)

    def test_quiet_window_can_ignore_instantaneous_contact_acceleration(self):
        gate = ConsecutiveQuietWindow(
            self.thresholds,
            required_samples=2,
            include_acceleration=False,
        )
        chatter = {
            "cup": FreeJointMotion(0.0005, 0.005, 10.0, 100.0),
            "plate": FreeJointMotion(0.0001, 0.001, 20.0, 200.0),
        }
        self.assertFalse(gate.observe(chatter))
        self.assertTrue(gate.observe(chatter))
        chatter["cup"] = FreeJointMotion(0.0005, 0.02, 0.0, 0.0)
        self.assertFalse(gate.observe(chatter))
        self.assertEqual(gate.consecutive_samples, 0)

    def test_invalid_gate_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            QuietThresholds(-1.0, 0.0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            ConsecutiveQuietWindow(self.thresholds, required_samples=0)
        gate = ConsecutiveQuietWindow(self.thresholds, required_samples=1)
        with self.assertRaisesRegex(ValueError, "at least one"):
            gate.observe({})

    def test_transitive_support_resolves_stacks_and_floating_components(self):
        result = resolve_transitive_support(
            ("base", "middle", "top", "floating_a", "floating_b"),
            {"base"},
            {
                "base": {"middle"},
                "middle": {"base", "top"},
                "top": {"middle"},
                "floating_a": {"floating_b"},
                "floating_b": {"floating_a"},
            },
        )

        self.assertEqual(
            result,
            {
                "base": {
                    "supported": True,
                    "support_path": ("base", "static_desk"),
                },
                "middle": {
                    "supported": True,
                    "support_path": ("middle", "base", "static_desk"),
                },
                "top": {
                    "supported": True,
                    "support_path": (
                        "top",
                        "middle",
                        "base",
                        "static_desk",
                    ),
                },
                "floating_a": {"supported": False, "support_path": ()},
                "floating_b": {"supported": False, "support_path": ()},
            },
        )

    def test_transitive_support_uses_object_order_for_equal_paths(self):
        result = resolve_transitive_support(
            ("root_b", "bridge", "root_a", "top"),
            {"root_a", "root_b"},
            {
                "root_b": {"bridge"},
                "bridge": {"root_a", "root_b", "top"},
                "root_a": {"bridge"},
                "top": {"bridge"},
            },
        )

        self.assertEqual(
            result["bridge"]["support_path"],
            ("bridge", "root_b", "static_desk"),
        )
        self.assertEqual(
            result["top"]["support_path"],
            ("top", "bridge", "root_b", "static_desk"),
        )

    def test_transitive_support_rejects_invalid_graphs(self):
        valid_adjacency = {"a": {"b"}, "b": {"a"}}
        invalid_cases = (
            (
                (),
                (),
                {},
                "at least one",
            ),
            (
                ("a", "a"),
                (),
                {"a": ()},
                "duplicate",
            ),
            (
                ("a", "b"),
                {"unknown"},
                valid_adjacency,
                "unknown object",
            ),
            (
                ("a", "b"),
                (),
                {"a": {"a"}, "b": ()},
                "self edge",
            ),
            (
                ("a", "b"),
                (),
                {"a": {"unknown"}, "b": ()},
                "unknown object",
            ),
            (
                ("a", "b"),
                (),
                {"a": {"b"}, "b": ()},
                "symmetric",
            ),
            (
                ("a", "b"),
                (),
                {"a": ()},
                "missing object",
            ),
            (
                ("a", "b"),
                (),
                {"a": (), 1: ()},
                "keys must be non-empty strings",
            ),
        )
        for object_names, directly_supported, adjacency, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    resolve_transitive_support(
                        object_names,
                        directly_supported,
                        adjacency,
                    )


if __name__ == "__main__":
    unittest.main()
