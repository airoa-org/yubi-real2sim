# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pytest

try:
    import mujoco
    from robosuite_env.robots.yubi.calibration import (
        GripperCalibrationError,
        apply_left_gripper_calibration,
        apply_right_gripper_calibration,
        load_gripper_calibration,
    )
except ImportError:  # pragma: no cover - heavy dependency absent from CI
    mujoco = None
    GripperCalibrationError = None
    apply_left_gripper_calibration = None
    apply_right_gripper_calibration = None
    load_gripper_calibration = None


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = REPO_ROOT / "robosuite_env" / "assets"
CALIBRATION_PATH = ASSET_DIR / "gripper_calib_rad.json"
LEFT_XML = ASSET_DIR / "yubi_left.xml"
EXPECTED_SHA256 = (
    "12f726e13e21f0c79c4dd05bfb5b4fea65ebc7783189e3bc44bc652d0db18d2f"
)


def _joint_ids(model):
    names = ("finger_joint_left", "finger_joint_right")
    return [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in names
    ]


@pytest.mark.usefixtures("mujoco")
class GripperCalibrationTest(unittest.TestCase):
    def test_checked_in_calibration_loads_with_exact_provenance(self):
        calibration = load_gripper_calibration(CALIBRATION_PATH)

        self.assertEqual(calibration.sha256, EXPECTED_SHA256)
        self.assertAlmostEqual(calibration.left_offset_rad, -0.10213766992092133)
        self.assertAlmostEqual(calibration.right_offset_rad, -0.10213766992092133)
        self.assertEqual(
            calibration.reference_joints,
            ("finger_joint_left", "finger_joint_right"),
        )
        self.assertEqual(
            Path(calibration.reference_xml).name,
            "yubi_left.xml",
        )

    def test_loader_rejects_inconsistent_alias_and_wrong_reference_asset(self):
        payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        cases = (
            ("alias", {"left": 0.0}, "same calibration"),
            ("right_alias", {"right": 0.0}, "same calibration"),
            ("missing_right", {"right_offset_rad": None}, "finite number"),
            (
                "reference",
                {"reference_xml": "other/yubi_right.xml"},
                "does not match",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label, update, message in cases:
                with self.subTest(label=label):
                    candidate = root / f"{label}.json"
                    candidate.write_text(
                        json.dumps(payload | update),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(GripperCalibrationError, message):
                        load_gripper_calibration(candidate)

    def test_body_frame_offset_matches_raw_plus_calibration_without_state_changes(self):
        calibration = load_gripper_calibration(CALIBRATION_PATH)
        reference_model = mujoco.MjModel.from_xml_path(str(LEFT_XML))
        calibrated_model = mujoco.MjModel.from_xml_path(str(LEFT_XML))
        reference_data = mujoco.MjData(reference_model)
        calibrated_data = mujoco.MjData(calibrated_model)
        joint_ids = _joint_ids(calibrated_model)
        body_quat_before = calibrated_model.body_quat.copy()
        qpos0_before = calibrated_model.qpos0.copy()
        ranges_before = calibrated_model.jnt_range.copy()
        ctrlrange_before = calibrated_model.actuator_ctrlrange.copy()

        report = apply_left_gripper_calibration(
            calibrated_model,
            compiled_joint_ids=joint_ids,
            compiled_joint_names=["finger_joint_left", "finger_joint_right"],
            calibration=calibration,
            runtime_reference_xml=LEFT_XML,
        )

        self.assertTrue(report["enabled"])
        self.assertEqual(report["sha256"], EXPECTED_SHA256)
        np.testing.assert_array_equal(calibrated_model.qpos0, qpos0_before)
        np.testing.assert_array_equal(calibrated_model.jnt_range, ranges_before)
        np.testing.assert_array_equal(
            calibrated_model.actuator_ctrlrange, ctrlrange_before
        )
        changed_body_ids = {
            int(calibrated_model.jnt_bodyid[joint_id]) for joint_id in joint_ids
        }
        for body_id in range(calibrated_model.nbody):
            if body_id not in changed_body_ids:
                np.testing.assert_array_equal(
                    calibrated_model.body_quat[body_id],
                    body_quat_before[body_id],
                )

        finger_geom_ids = [
            geom_id
            for geom_id in range(reference_model.ngeom)
            if (
                mujoco.mj_id2name(
                    reference_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )
                or ""
            ).startswith("finger_")
        ]
        for raw_angle in (0.0, 0.3, 0.7854):
            with self.subTest(raw_angle=raw_angle):
                reference_data.qpos[:] = raw_angle + calibration.left_offset_rad
                calibrated_data.qpos[:] = raw_angle
                mujoco.mj_forward(reference_model, reference_data)
                mujoco.mj_forward(calibrated_model, calibrated_data)
                np.testing.assert_allclose(
                    calibrated_data.geom_xpos[finger_geom_ids],
                    reference_data.geom_xpos[finger_geom_ids],
                    rtol=0.0,
                    atol=1e-15,
                )
                np.testing.assert_allclose(
                    calibrated_data.geom_xmat[finger_geom_ids],
                    reference_data.geom_xmat[finger_geom_ids],
                    rtol=0.0,
                    atol=1e-15,
                )
                np.testing.assert_array_equal(
                    calibrated_data.qpos,
                    np.full(2, raw_angle),
                )

        right_xml = ASSET_DIR / "yubi_right.xml"
        right_reference = mujoco.MjModel.from_xml_path(str(right_xml))
        right_calibrated = mujoco.MjModel.from_xml_path(str(right_xml))
        report = apply_right_gripper_calibration(
            right_calibrated,
            compiled_joint_ids=_joint_ids(right_calibrated),
            compiled_joint_names=["finger_joint_left", "finger_joint_right"],
            calibration=calibration,
            runtime_reference_xml=right_xml,
        )
        self.assertEqual(report["logical_hand"], "R")
        self.assertEqual(report["offset_rad"], calibration.right_offset_rad)
        right_reference_data = mujoco.MjData(right_reference)
        right_calibrated_data = mujoco.MjData(right_calibrated)
        right_finger_geoms = [
            geom_id
            for geom_id in range(right_reference.ngeom)
            if (
                mujoco.mj_id2name(
                    right_reference, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )
                or ""
            ).startswith("finger_")
        ]
        right_reference_data.qpos[:] = calibration.right_offset_rad
        mujoco.mj_forward(right_reference, right_reference_data)
        mujoco.mj_forward(right_calibrated, right_calibrated_data)
        np.testing.assert_allclose(
            right_calibrated_data.geom_xpos[right_finger_geoms],
            right_reference_data.geom_xpos[right_finger_geoms],
            rtol=0.0,
            atol=1e-15,
        )

    def test_second_application_is_rejected_instead_of_doubling_offset(self):
        calibration = load_gripper_calibration(CALIBRATION_PATH)
        model = mujoco.MjModel.from_xml_path(str(LEFT_XML))
        joint_ids = _joint_ids(model)
        arguments = {
            "compiled_joint_ids": joint_ids,
            "compiled_joint_names": ["finger_joint_left", "finger_joint_right"],
            "calibration": calibration,
            "runtime_reference_xml": LEFT_XML,
        }
        apply_left_gripper_calibration(model, **arguments)

        with self.assertRaisesRegex(GripperCalibrationError, "already be applied"):
            apply_left_gripper_calibration(model, **arguments)


if __name__ == "__main__":
    unittest.main()


import unittest


from robosuite_env.robots.yubi.control.robosuite_hook import DynamicYubiQfrcHook


class _Config:
    def resolve_torque_limit(self, opening):
        return 0.8 if opening else 0.6


class _Hand:
    def __init__(self, side, events, torque):
        self.side = side
        self.events = events
        self.torque = torque
        self.cfg = _Config()
        self.telemetry = {}
        self.targets = []

    def set_action(self, action, t):
        self.targets.append((action, t))

    def compute_torque(self, t, q, qd):
        self.events.append(("compute", self.side, t, q, qd))
        self.telemetry = {
            "q_user_goal": q + 0.1,
            "q_ref": q + 0.1,
            "e": 0.1,
            "tau_raw": self.torque * 2,
            "current_equiv_A": self.torque / 2,
            "saturated": True,
            "sat_ratio": 2.0,
        }
        return self.torque


class _Controller:
    def __init__(self, events):
        self.right = _Hand("right", events, 0.4)
        self.left = _Hand("left", events, -0.2)


class _Data:
    def __init__(self):
        self.time = 6.0
        self.qpos = np.array([0.2, 0.4, 0.5, 0.7])
        self.qvel = np.array([1.0, 3.0, -2.0, 2.0, 0.0])
        self.qfrc_applied = np.full(5, 9.0)


class DynamicYubiQfrcHookTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.data = _Data()
        self.controller = _Controller(self.events)
        self.arms = {
            "R": {"fj": [0, 1], "fd": [0, 1]},
            "L": {"fj": [2, 3], "fd": [2, 3]},
        }

        def original(action, policy_step=False):
            self.events.append(("original", policy_step, tuple(action)))

        self.hook = DynamicYubiQfrcHook(
            original, self.data, self.arms, self.controller, expected_dt=0.002
        )

    def test_original_runs_first_and_finger_forces_are_overwritten(self):
        self.hook.set_targets({"R": 0.7, "L": 0.5})
        self.hook([1.0, 2.0], policy_step=True)

        self.assertEqual(self.events[0][0], "original")
        self.assertEqual(self.controller.right.targets, [(0.7, 6.0)])
        self.assertEqual(self.controller.left.targets, [(0.5, 6.0)])
        np.testing.assert_allclose(self.data.qfrc_applied, [0.2, 0.2, -0.1, -0.1, 9.0])
        self.assertEqual(self.hook.tick_count, 1)
        self.assertEqual(self.hook.target_update_count, 1)

        samples = self.hook.drain_samples()
        self.assertEqual(len(samples), 1)
        self.assertAlmostEqual(samples[0]["sides"]["R"]["q"], 0.3)
        self.assertAlmostEqual(samples[0]["sides"]["R"]["qd"], 2.0)
        self.assertAlmostEqual(samples[0]["sides"]["R"]["jaw_tau"], 0.2)
        self.assertEqual(self.hook.drain_samples(), [])

    def test_one_compute_per_call_and_cadence_is_checked(self):
        self.hook.set_targets({"R": 0.7, "L": 0.5})
        self.hook([0.0])
        self.data.time += 0.002
        self.hook([0.0])
        self.data.time += 0.004
        self.hook([0.0])

        self.assertEqual(self.hook.tick_count, 3)
        self.assertEqual(len([event for event in self.events if event[0] == "compute"]), 6)
        self.assertEqual(self.hook.cadence_violation_count, 1)

        self.hook.reset_timing(clear_samples=True, reset_counts=True)
        self.assertEqual(self.hook.tick_count, 0)
        self.assertEqual(self.hook.target_update_count, 0)
        self.assertEqual(self.hook.cadence_violation_count, 0)
        self.assertEqual(self.hook.drain_samples(), [])


if __name__ == "__main__":
    unittest.main()


from pathlib import Path

import pytest

try:
    import mujoco
except ImportError:  # pragma: no cover - heavy dependency absent from CI
    mujoco = None


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = REPO_ROOT / "robosuite_env/assets"
RIGHT_XML = ASSET_DIR / "yubi_right.xml"
LEFT_XML = ASSET_DIR / "yubi_left.xml"
MANIFEST_PATH = ASSET_DIR / "fr3_yubi_right_registered/manifest.json"


@pytest.mark.usefixtures("mujoco")
def test_registered_mount_compiles_bilaterally_and_matches_blender_bounds():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_camera_positions = {
        RIGHT_XML: [0.0, 0.0408, -0.0554],
        LEFT_XML: [0.0, 0.04125, -0.05542],
    }
    expected_body_positions = {
        "right_gripper": [0.0, 0.0, 0.0],
        "hand_root": [0.0, 0.0015, 0.051],
        "yubi_geom": manifest["chain"][
            "colored_yubi_model_origin_in_physical_root_m"
        ],
        "finger_left": [0.015, 0.001, 0.0345],
        "finger_right": [-0.015, 0.001, 0.0345],
    }
    for xml_path, expected_camera_position in expected_camera_positions.items():
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        for body_name, expected_position in expected_body_positions.items():
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            np.testing.assert_allclose(
                model.body_pos[body_id],
                expected_position,
                rtol=0.0,
                atol=1e-6 if body_name == "yubi_geom" else 1e-12,
            )
        hand_root_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "hand_root"
        )
        np.testing.assert_allclose(
            model.body_quat[hand_root_id],
            [0.5, 0.5, 0.5, 0.5],
            rtol=0.0,
            atol=0.0,
        )
        for camera_id in range(model.ncam):
            np.testing.assert_allclose(
                model.cam_pos[camera_id],
                expected_camera_position,
                rtol=0.0,
                atol=1e-12,
            )

        for stem in ("yubi_mount_flange", "yubi_mount_attachment"):
            geom_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"{stem}_visual"
            )
            mesh_id = int(model.geom_dataid[geom_id])
            vertex_start = int(model.mesh_vertadr[mesh_id])
            vertex_count = int(model.mesh_vertnum[mesh_id])
            vertices = np.asarray(
                model.mesh_vert[vertex_start : vertex_start + vertex_count], dtype=float
            )
            rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
            world = vertices @ rotation.T + data.geom_xpos[geom_id]
            key = stem.replace("yubi_mount", "right_yubi")
            reference = manifest["visual_meshes"][key]
            np.testing.assert_allclose(
                world.min(axis=0), reference["bounds_min_m"], rtol=0.0, atol=3e-9
            )
            np.testing.assert_allclose(
                world.max(axis=0), reference["bounds_max_m"], rtol=0.0, atol=3e-9
            )

        mount_collision_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            for geom_id in range(model.ngeom)
        ]
        mount_collision_names = [
            name
            for name in mount_collision_names
            if name.startswith("yubi_mount_") and "_collision_" in name
        ]
        assert len(mount_collision_names) == 32
