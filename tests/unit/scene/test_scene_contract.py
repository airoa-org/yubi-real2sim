# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from xml.etree import ElementTree as ET

import numpy as np
import pytest

import robosuite_env.contracts.scene as scene_contract_module
from robosuite_env.contracts.scene import (
    TABLE_Z_METHOD,
    TABLE_Z_MIN_VERTICES,
    TABLE_Z_TOLERANCE_M,
    SceneContractError,
    build_scene_contract,
    compute_tree_sha256,
    measure_gs_table_plane_z,
    ply_vertex_count,
    scene_tag,
    validate_scene_contract,
)

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", ("room1", "room2"))
def test_plain_scene_names_survive_tag_normalization(name: str) -> None:
    assert scene_tag(name) == name


def _write_plane_ply(
    path: Path,
    w2gs: np.ndarray,
    *,
    table_z_world: float,
    count: int = 4000,
    text: bool = False,
) -> None:
    """Write a binary 3DGS-style PLY whose dominant surface is a table plane.

    Points are laid out deterministically in MuJoCo world (a dense grid on
    the table top at ``table_z_world`` plus sparse clutter above it) and then
    mapped into GS space through ``w2gs`` so that inv(w2gs) recovers them.
    """

    from plyfile import PlyData, PlyElement

    side = int(math.ceil(math.sqrt(count)))
    grid = np.linspace(-0.3, 0.3, side)
    gx, gy = np.meshgrid(grid, grid, indexing="ij")
    table = np.stack(
        [gx.ravel(), gy.ravel() - 0.10, np.full(gx.size, table_z_world)],
        axis=1,
    )[:count]
    clutter_count = max(count // 10, 1)
    t = np.linspace(0.0, 1.0, clutter_count)
    clutter = np.stack(
        [
            0.2 * np.sin(7.0 * t),
            -0.10 + 0.2 * np.cos(5.0 * t),
            table_z_world + 0.05 + 0.4 * t,
        ],
        axis=1,
    )
    world = np.concatenate([table, clutter], axis=0)
    homogeneous = np.concatenate(
        [world, np.ones((world.shape[0], 1))],
        axis=1,
    )
    gs = homogeneous @ np.asarray(w2gs, dtype=np.float64).T
    gs = gs[:, :3] / gs[:, 3:4]
    vertices = np.empty(
        gs.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("f_dc_0", "f4")],
    )
    vertices["x"] = gs[:, 0]
    vertices["y"] = gs[:, 1]
    vertices["z"] = gs[:, 2]
    vertices["f_dc_0"] = 0.5
    PlyData([PlyElement.describe(vertices, "vertex")], text=text).write(
        os.fspath(path)
    )


def _scaled_identity(scale: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] *= scale
    return matrix


class _SceneFixtureTest(unittest.TestCase):
    """Temporary schema-v2 scene with a header-only fixture PLY."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.scene = self.root / "scene"
        self.scene.mkdir()

        self.ply = self.scene / "point_cloud.ply"
        self.ply.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
        self.w2gs = self.scene / "gs_w2gs.npy"
        np.save(self.w2gs, np.eye(4, dtype=np.float64))

        self.collision = self.scene / "table_collision"
        self.collision.mkdir()
        for index in range(2):
            (self.collision / f"table_{index:03d}.obj").write_text(
                f"v {index} 0 0\nv {index} 1 0\nv {index} 0 1\nf 1 2 3\n",
                encoding="utf-8",
            )
        collision_files = sorted(self.collision.glob("*.obj"))
        self.spec = {
            "schema_version": 2,
            "id": "franka-white-pinhole-a1b2c3",
            "name": "room2",
            "ply_path": "point_cloud.ply",
            "ply_sha256": _sha256(self.ply),
            "w2gs": "gs_w2gs.npy",
            "w2gs_sha256": _sha256(self.w2gs),
            # The header-only fixture PLY is below TABLE_Z_MIN_VERTICES, so
            # the declared value is accepted without measurement.
            "calibration_table_z_m": 0.0,
            "calibration_table_z_method": TABLE_Z_METHOD,
            "table": {
                "appearance": "embedded_in_3dgs",
                "collision": {
                    "directory": "table_collision",
                    "glob": "*.obj",
                    "count": 2,
                    "tree_sha256": compute_tree_sha256(
                        self.collision,
                        collision_files,
                    ),
                    "frame": "mujoco_world",
                },
            },
        }
        self.manifest = self.scene / "background.json"
        self._write_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_manifest(self) -> None:
        self.manifest.write_text(
            json.dumps(self.spec, indent=2) + "\n",
            encoding="utf-8",
        )


class SceneContractTest(_SceneFixtureTest):
    def test_plain_scene_names_need_no_underscore(self):
        for name in ("room1", "room2"):
            with self.subTest(name=name):
                self.spec["name"] = name
                self._write_manifest()
                self.assertEqual(validate_scene_contract(self.scene)["name"], name)

    def test_valid_v2_resolves_one_bound_scene(self):
        report = validate_scene_contract(self.scene)

        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["id"], self.spec["id"])
        self.assertEqual(report["name"], self.spec["name"])
        self.assertEqual(report["manifest_path"], self.manifest.resolve())
        self.assertEqual(report["background_dir"], self.scene.resolve())
        self.assertEqual(report["ply_path"], self.ply.resolve())
        self.assertEqual(report["w2gs_path"], self.w2gs.resolve())
        np.testing.assert_allclose(report["w2gs_matrix"], np.eye(4))
        self.assertEqual(report["collision_directory"], self.collision.resolve())
        self.assertEqual(
            report["collision_files"],
            tuple(path.resolve() for path in sorted(self.collision.glob("*.obj"))),
        )
        self.assertRegex(report["contract_digest"], r"^[0-9a-f]{64}$")

        from_file = validate_scene_contract(self.manifest)
        self.assertEqual(from_file["contract_digest"], report["contract_digest"])

        with self.assertRaisesRegex(SceneContractError, "does not match"):
            validate_scene_contract(
                self.scene,
                preverified_ply_sha256="0" * 64,
            )
        for digest in ("f" * 63, "F" * 64, "not-a-digest"):
            with self.subTest(digest=digest):
                with self.assertRaisesRegex(
                    SceneContractError,
                    "preverified_ply_sha256 must be a lowercase SHA-256 digest",
                ):
                    validate_scene_contract(
                        self.scene,
                        preverified_ply_sha256=digest,
                    )

    def test_matching_preverified_ply_hash_avoids_reading_ply_again(self):
        original_sha256 = scene_contract_module._sha256

        def reject_ply_rehash(path: Path) -> str:
            if path.resolve() == self.ply.resolve():
                raise AssertionError("PLY content was hashed twice")
            return original_sha256(path)

        with mock.patch.object(
            scene_contract_module,
            "_sha256",
            side_effect=reject_ply_rehash,
        ):
            report = validate_scene_contract(
                self.scene,
                preverified_ply_sha256=self.spec["ply_sha256"],
            )

        self.assertEqual(report["ply_sha256"], self.spec["ply_sha256"])

    def test_v2_accepts_valid_inline_uniform_scale_transform(self):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] *= 2.5
        matrix[:3, 3] = [1.0, -2.0, 3.0]
        self.spec.pop("w2gs")
        self.spec.pop("w2gs_sha256")
        self.spec["mat_w2gs"] = matrix.tolist()
        self._write_manifest()

        report = validate_scene_contract(self.scene)

        self.assertIsNone(report["w2gs_path"])
        np.testing.assert_allclose(report["w2gs_matrix"], matrix)

    def test_v2_accepts_float32_registration_roundoff(self):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.asarray(
            [
                [1.76153882, -2.98584331, -0.24853053],
                [-0.14695420, 0.20185396, -3.46665607],
                [2.99256355, 1.76749618, -0.02394055],
            ]
        )
        self.spec.pop("w2gs")
        self.spec.pop("w2gs_sha256")
        self.spec["mat_w2gs"] = matrix.tolist()
        self._write_manifest()

        report = validate_scene_contract(self.scene)

        np.testing.assert_allclose(report["w2gs_matrix"], matrix)

    def test_legacy_v1_is_readable_only_when_not_strict(self):
        self.spec = {
            "schema_version": 1,
            "name": "legacy_scene",
            "ply_path": "point_cloud.ply",
            "mat_w2gs": np.eye(4).tolist(),
        }
        self._write_manifest()

        with self.assertRaisesRegex(SceneContractError, "official scene requires"):
            validate_scene_contract(self.scene)

        report = validate_scene_contract(self.scene, strict=False)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["collision_files"], ())
        self.assertIsNone(report["collision_directory"])

        self.spec = {
            "schema_version": 1,
            "name": "legacy_scene",
            "ply_path": "point_cloud.ply",
            "w2gs": "gs_w2gs.npy",
            "w2gs_sha256": _sha256(self.w2gs),
        }
        self._write_manifest()

        report = validate_scene_contract(self.scene, strict=False)
        self.assertEqual(report["w2gs_path"], self.w2gs.resolve())

        self.spec["w2gs_sha256"] = "0" * 64
        self._write_manifest()
        with self.assertRaisesRegex(SceneContractError, "does not match"):
            validate_scene_contract(self.scene, strict=False)

    def test_does_not_discover_nested_background_recursively(self):
        with self.assertRaisesRegex(SceneContractError, "background.json not found"):
            validate_scene_contract(self.root)

        unrelated_json = self.root / "scene.json"
        unrelated_json.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(SceneContractError, "exact background.json"):
            validate_scene_contract(unrelated_json)

    def test_relative_paths_are_manifest_relative_and_env_paths_expand(self):
        external = self.root / "external_collision"
        self.collision.rename(external)
        files = sorted(external.glob("*.obj"))
        collision_spec = self.spec["table"]["collision"]
        collision_spec["directory"] = "${SCENE_CONTRACT_TEST_COLLISION}"
        collision_spec["tree_sha256"] = compute_tree_sha256(external, files)
        self._write_manifest()

        with mock.patch.dict(
            os.environ,
            {"SCENE_CONTRACT_TEST_COLLISION": str(external)},
        ):
            report = validate_scene_contract(self.scene)

        self.assertEqual(report["ply_path"], self.ply.resolve())
        self.assertEqual(report["collision_directory"], external.resolve())

    def test_v2_rejects_external_or_symlinked_scene_payloads(self):
        external = self.root / "external.ply"
        external.write_bytes(self.ply.read_bytes())
        cases = {
            "parent traversal": "../external.ply",
            "absolute path": str(external),
        }
        symlink = self.scene / "linked.ply"
        symlink.symlink_to(external)
        cases["escaping symlink"] = symlink.name

        for label, value in cases.items():
            with self.subTest(label=label):
                self.spec["ply_path"] = value
                self.spec["ply_sha256"] = _sha256(external)
                self._write_manifest()
                with self.assertRaisesRegex(SceneContractError, "inside the scene"):
                    validate_scene_contract(self.scene)

        self.spec["ply_path"] = self.ply.name
        self.spec["ply_sha256"] = _sha256(self.ply)
        external_w2gs = self.root / "external.npy"
        np.save(external_w2gs, np.eye(4, dtype=np.float64))
        self.spec["w2gs"] = str(external_w2gs)
        self.spec["w2gs_sha256"] = _sha256(external_w2gs)
        self._write_manifest()
        with self.assertRaisesRegex(SceneContractError, "inside the scene"):
            validate_scene_contract(self.scene)

    def test_rejects_missing_or_incorrect_content_hashes(self):
        cases = (
            ("ply_sha256", None, "lowercase SHA-256"),
            ("ply_sha256", "0" * 64, "does not match"),
            ("w2gs_sha256", None, "lowercase SHA-256"),
            ("w2gs_sha256", "f" * 64, "does not match"),
        )
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                original = self.spec.get(field)
                if value is None:
                    self.spec.pop(field, None)
                else:
                    self.spec[field] = value
                self._write_manifest()
                with self.assertRaisesRegex(SceneContractError, message):
                    validate_scene_contract(self.scene)
                self.spec[field] = original

    def test_rejects_invalid_world_to_gs_matrices(self):
        self.spec.pop("w2gs")
        self.spec.pop("w2gs_sha256")
        matrices = {
            "wrong shape": [[1.0, 0.0], [0.0, 1.0]],
            "boolean": [
                [True, False, False, False],
                [False, True, False, False],
                [False, False, True, False],
                [False, False, False, True],
            ],
            "not finite": [
                [float("nan"), 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "bad bottom row": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            "reflection": [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "non-uniform scale": [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
        for label, matrix in matrices.items():
            with self.subTest(label=label):
                self.spec["mat_w2gs"] = matrix
                self._write_manifest()
                with self.assertRaises(SceneContractError):
                    validate_scene_contract(self.scene)

    def test_rejects_both_or_neither_transform_source(self):
        self.spec["mat_w2gs"] = np.eye(4).tolist()
        self._write_manifest()
        with self.assertRaisesRegex(SceneContractError, "exactly one"):
            validate_scene_contract(self.scene)

        self.spec.pop("mat_w2gs")
        self.spec.pop("w2gs")
        self.spec.pop("w2gs_sha256")
        self._write_manifest()
        with self.assertRaisesRegex(SceneContractError, "exactly one"):
            validate_scene_contract(self.scene)

    def test_v2_rejects_legacy_transform_and_point_cloud_overrides(self):
        cases = {
            "ply": "different.ply",
            "gs_w2gs": "different.npy",
            "alignment": {"rot": "x90"},
            "gs_rot": "x90",
            "gs_pivot": [0.0, 0.0, 0.8],
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                self.spec[field] = value
                self._write_manifest()
                with self.assertRaisesRegex(
                    SceneContractError,
                    "aliases outside the canonical contract",
                ):
                    validate_scene_contract(self.scene)
                self.spec.pop(field)

    def test_rejects_any_table_visual_field(self):
        self.spec["table"]["visual"] = "table.glb"
        self._write_manifest()

        with self.assertRaisesRegex(SceneContractError, "permits only"):
            validate_scene_contract(self.scene)
        self.spec["table"].pop("visual")
        self.spec["table"]["appearance"] = "mujoco_visual"
        self._write_manifest()

        with self.assertRaisesRegex(SceneContractError, "embedded_in_3dgs"):
            validate_scene_contract(self.scene)

    def test_rejects_missing_empty_or_wrong_collision_selection(self):
        collision_spec = self.spec["table"]["collision"]
        cases = (
            ("missing directory", "directory", "missing", "does not name a directory"),
            ("empty glob", "glob", "*.stl", "matched no files"),
            ("zero count", "count", 0, "positive integer"),
            ("wrong count", "count", 3, "count mismatch"),
            ("wrong tree hash", "tree_sha256", "0" * 64, "does not match"),
            ("wrong frame", "frame", "gs", "mujoco_world"),
            ("escaping glob", "glob", "../*.obj", "stay inside"),
        )
        for label, field, value, message in cases:
            with self.subTest(label=label):
                original = collision_spec[field]
                collision_spec[field] = value
                self._write_manifest()
                with self.assertRaisesRegex(SceneContractError, message):
                    validate_scene_contract(self.scene)
                collision_spec[field] = original

    def test_contract_digest_binds_manifest_and_selected_content(self):
        first = validate_scene_contract(self.scene)["contract_digest"]
        self.spec["id"] = "franka-white-pinhole-different"
        self._write_manifest()
        second = validate_scene_contract(self.scene)["contract_digest"]

        self.assertNotEqual(first, second)

    def test_build_scene_contract_upgrades_legacy_without_writing(self):
        legacy = {
            "name": "room2",
            "ply_path": "point_cloud.ply",
            "w2gs": "gs_w2gs.npy",
            "notes": "preserved",
        }
        self.manifest.write_text(json.dumps(legacy), encoding="utf-8")

        payload = build_scene_contract(
            self.scene,
            table_collision_directory=self.collision,
        )

        self.assertEqual(json.loads(self.manifest.read_text()), legacy)
        self.assertEqual(payload["schema_version"], 2)
        self.assertRegex(
            payload["id"],
            r"^room2-[0-9a-f]{12}$",
        )
        self.assertEqual(payload["notes"], "preserved")
        self.assertEqual(payload["table"]["appearance"], "embedded_in_3dgs")
        self.assertEqual(payload["table"]["collision"]["count"], 2)

        self.manifest.write_text(json.dumps(payload), encoding="utf-8")
        report = validate_scene_contract(self.scene)
        self.assertEqual(report["collision_files"], tuple(sorted(self.collision.glob("*.obj"))))

    def test_build_scene_contract_bakes_legacy_calibration_offset(self):
        raw = np.asarray(
            [
                [0.0, -2.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 2.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        np.save(self.w2gs, raw)
        self.manifest.write_text(
            json.dumps(
                {
                    "name": "legacy_offset_scene",
                    "ply_path": "point_cloud.ply",
                    "w2gs": "gs_w2gs.npy",
                }
            ),
            encoding="utf-8",
        )
        offset = np.asarray([0.25, -0.5, 0.8], dtype=np.float64)

        payload = build_scene_contract(
            self.scene,
            table_collision_directory=self.collision,
            calibration_world_offset=offset,
        )

        translation = np.eye(4, dtype=np.float64)
        translation[:3, 3] = offset
        expected = raw @ translation
        self.assertNotIn("w2gs", payload)
        self.assertNotIn("w2gs_sha256", payload)
        np.testing.assert_allclose(payload["mat_w2gs"], expected)

        self.manifest.write_text(json.dumps(payload), encoding="utf-8")
        report = validate_scene_contract(self.scene)
        np.testing.assert_allclose(report["w2gs_matrix"], expected)

    def test_upgrade_preserves_external_collision_environment_reference(self):
        external = self.root / "external" / "coacd"
        external.parent.mkdir()
        self.collision.rename(external)
        self.manifest.write_text(
            json.dumps(
                {
                    "name": "portable_external_scene",
                    "ply_path": "point_cloud.ply",
                    "w2gs": "gs_w2gs.npy",
                }
            ),
            encoding="utf-8",
        )

        with mock.patch.dict(
            os.environ,
            {"SCENE_EXTERNAL_COLLISION": str(external.parent)},
        ):
            payload = build_scene_contract(
                self.scene,
                table_collision_directory=(
                    "${SCENE_EXTERNAL_COLLISION}/coacd"
                ),
            )

        self.assertEqual(
            payload["table"]["collision"]["directory"],
            "${SCENE_EXTERNAL_COLLISION}/coacd",
        )

    def test_generated_scene_id_binds_transform_and_collision_content(self):
        legacy = {
            "name": "same_scene_name",
            "ply_path": "point_cloud.ply",
            "mat_w2gs": np.eye(4).tolist(),
        }
        self.manifest.write_text(json.dumps(legacy), encoding="utf-8")
        first = build_scene_contract(
            self.scene,
            table_collision_directory=self.collision,
        )["id"]
        second = build_scene_contract(
            self.scene,
            table_collision_directory=self.collision,
            calibration_world_offset=(0.0, 0.0, 0.8),
        )["id"]

        self.assertNotEqual(first, second)

    def test_build_scene_contract_rejects_already_v2_scene(self):
        with self.assertRaisesRegex(SceneContractError, "already uses schema v2"):
            build_scene_contract(
                self.scene,
                table_collision_directory=self.collision,
            )

    def test_build_scene_contract_rejects_uncomposed_runtime_alignment_fields(self):
        legacy = {
            "name": "legacy_scene",
            "ply_path": "point_cloud.ply",
            "w2gs": "gs_w2gs.npy",
        }
        cases = {
            "alignment": {"rot": "x90"},
            "transform": {"matrix": np.eye(4).tolist()},
            "w2gs_matrix": np.eye(4).tolist(),
            "world_to_gs_matrix": np.eye(4).tolist(),
            "rot_R": np.eye(3).tolist(),
            "rotation_matrix": np.eye(3).tolist(),
            "rot": "x90",
            "gs_rot": "x90",
            "rotation": "x90",
            "pivot": [0.0, 0.0, 0.8],
            "rot_piv": [0.0, 0.0, 0.8],
            "rotation_pivot": [0.0, 0.0, 0.8],
            "gs_pivot": [0.0, 0.0, 0.8],
            "pivz": 0.8,
            "gs_pivz": 0.8,
            "pivot_z": 0.8,
            "rot_pivz": 0.8,
            "use_legacy_alignment": True,
            "legacy_alignment": True,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                self.manifest.write_text(
                    json.dumps({**legacy, field: value}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    SceneContractError,
                    "cannot be safely migrated",
                ):
                    build_scene_contract(
                        self.scene,
                        table_collision_directory=self.collision,
                    )

    def test_upgrade_write_validates_payload_before_replacing_manifest(self):
        legacy = {
            "name": "legacy_scene",
            "ply_path": "point_cloud.ply",
            "w2gs": "gs_w2gs.npy",
        }
        original_manifest = json.dumps(legacy, indent=2) + "\n"
        self.manifest.write_text(original_manifest, encoding="utf-8")
        real_sha256 = scene_contract_module._sha256
        changed_ply = False

        def hash_then_change_ply(path):
            nonlocal changed_ply
            digest = real_sha256(path)
            if Path(path).resolve() == self.ply.resolve() and not changed_ply:
                changed_ply = True
                self.ply.write_bytes(
                    b"ply\nformat ascii 1.0\ncomment changed\nend_header\n"
                )
            return digest

        with mock.patch.object(
            scene_contract_module,
            "_sha256",
            side_effect=hash_then_change_ply,
        ):
            with self.assertRaisesRegex(SceneContractError, "ply_sha256 does not match"):
                scene_contract_module._main(
                    [
                        "upgrade",
                        str(self.scene),
                        "--table-collision-dir",
                        str(self.collision),
                        "--write",
                    ]
                )

        self.assertTrue(changed_ply)
        self.assertEqual(
            self.manifest.read_text(encoding="utf-8"),
            original_manifest,
        )

    def test_build_scene_contract_rejects_invalid_calibration_offset(self):
        legacy = {
            "name": "legacy_scene",
            "ply_path": "point_cloud.ply",
            "mat_w2gs": np.eye(4).tolist(),
        }
        self.manifest.write_text(json.dumps(legacy), encoding="utf-8")
        for value in ((0.0, 1.0), (0.0, float("nan"), 1.0)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    SceneContractError,
                    "three finite numbers",
                ):
                    build_scene_contract(
                        self.scene,
                        table_collision_directory=self.collision,
                        calibration_world_offset=value,
                    )


class SceneTableHeightTest(_SceneFixtureTest):
    """calibration_table_z_m: the GS table plane must sit at MuJoCo table_z=0."""

    def _use_inline_matrix(self, matrix: np.ndarray) -> None:
        self.spec.pop("w2gs", None)
        self.spec.pop("w2gs_sha256", None)
        self.spec["mat_w2gs"] = matrix.tolist()

    def _write_scene_ply(self, w2gs: np.ndarray, **kwargs) -> None:
        _write_plane_ply(self.ply, w2gs, **kwargs)
        self.spec["ply_sha256"] = _sha256(self.ply)

    def _source_and_variant(
        self, *, source_table_z: float, variant_table_z: float
    ) -> tuple[Path, dict]:
        """A clean source scene and a variant whose own cloud moved its table."""

        scaled = _scaled_identity(3.5)
        self._write_scene_ply(scaled, table_z_world=source_table_z)
        self._use_inline_matrix(scaled)
        self._write_manifest()
        variant = self.root / "variant"
        variant.mkdir(exist_ok=True)
        _write_plane_ply(
            variant / "point_cloud.ply", scaled, table_z_world=variant_table_z
        )
        spec = json.loads(json.dumps(self.spec))
        spec["id"] = "variant-a1b2c3"
        spec["name"] = "variant"
        spec["ply_sha256"] = _sha256(variant / "point_cloud.ply")
        spec["table"]["collision"]["directory"] = str(self.collision)
        return variant, spec

    def _write_variant(self, variant: Path, spec: dict) -> None:
        (variant / "background.json").write_text(json.dumps(spec), encoding="utf-8")

    def test_variant_is_calibrated_by_its_source_not_its_own_cloud(self):
        # A displaced tabletop can lose the probe's modal bin to the floor, so
        # the variant's own cloud "measures" -0.8 m while its transform is the
        # validated source's, byte for byte.
        variant, spec = self._source_and_variant(
            source_table_z=0.0, variant_table_z=-0.8
        )
        validate_scene_contract(self.scene)
        self._write_variant(variant, spec)
        with self.assertRaisesRegex(SceneContractError, "disagrees"):
            validate_scene_contract(variant)

        spec["calibration_source"] = {
            "manifest": str(self.manifest),
            "ply_sha256": self.spec["ply_sha256"],
        }
        self._write_variant(variant, spec)
        report = validate_scene_contract(variant)
        self.assertAlmostEqual(report["gs_table_plane_z_m"], 0.0, delta=2e-3)
        self.assertEqual(report["calibration_source_manifest"], self.manifest.resolve())
        self.assertIsNone(validate_scene_contract(self.scene)["calibration_source_manifest"])

    def test_calibration_source_still_rejects_another_table_z_convention(self):
        # The source is what gets measured, so a source registered for the
        # +0.8 m convention is still caught through every variant of it.
        variant, spec = self._source_and_variant(
            source_table_z=0.8, variant_table_z=0.0
        )
        spec["calibration_source"] = {
            "manifest": str(self.manifest),
            "ply_sha256": self.spec["ply_sha256"],
        }
        self._write_variant(variant, spec)
        with self.assertRaisesRegex(
            SceneContractError,
            r"calibration_table_z_m=\+0\.0000 m disagrees .* \+0\.80\d\d m",
        ):
            validate_scene_contract(variant)

    def test_calibration_source_must_pin_and_share_the_source(self):
        variant, base_spec = self._source_and_variant(
            source_table_z=0.0, variant_table_z=-0.8
        )
        pinned = {
            "manifest": str(self.manifest),
            "ply_sha256": self.spec["ply_sha256"],
        }

        def attempt(mutate, pattern):
            spec = json.loads(json.dumps(base_spec))
            spec["calibration_source"] = dict(pinned)
            mutate(spec)
            self._write_variant(variant, spec)
            with self.assertRaisesRegex(SceneContractError, pattern):
                validate_scene_contract(variant)

        with self.subTest("another transform"):
            attempt(
                lambda spec: spec.update(mat_w2gs=_scaled_identity(3.6).tolist()),
                "transform differs from its calibration_source",
            )
        with self.subTest("wrong pin"):
            attempt(
                lambda spec: spec["calibration_source"].update(ply_sha256="0" * 64),
                "does not match the ply_sha256 declared",
            )
        with self.subTest("another declared table height"):
            attempt(
                lambda spec: spec.update(calibration_table_z_m=0.01),
                "differs from its calibration_source",
            )
        with self.subTest("unknown field"):
            attempt(
                lambda spec: spec["calibration_source"].update(scene="clean"),
                "permits only manifest and ply_sha256",
            )
        with self.subTest("chained source"):
            chained = dict(self.spec, calibration_source=dict(pinned))
            self.manifest.write_text(json.dumps(chained), encoding="utf-8")
            attempt(lambda spec: None, "is itself a variant")
            self._write_manifest()
        with self.subTest("source cloud replaced after pinning"):
            _write_plane_ply(self.ply, _scaled_identity(3.5), table_z_world=0.02)
            attempt(lambda spec: None, "scene field ply_sha256 does not match")

    def test_calibration_source_must_have_a_measured_table_plane(self):
        # A fixture-sized source is accepted on its own without a measurement.
        # It must not lend that acceptance to a variant whose cloud puts the
        # table somewhere else entirely.
        scaled = _scaled_identity(3.5)
        self._use_inline_matrix(scaled)
        self._write_manifest()
        self.assertIsNone(validate_scene_contract(self.scene)["gs_table_plane_z_m"])

        variant = self.root / "variant"
        variant.mkdir()
        _write_plane_ply(variant / "point_cloud.ply", scaled, table_z_world=-0.8)
        spec = json.loads(json.dumps(self.spec))
        spec["id"] = "variant-a1b2c3"
        spec["name"] = "variant"
        spec["ply_sha256"] = _sha256(variant / "point_cloud.ply")
        spec["table"]["collision"]["directory"] = str(self.collision)
        spec["calibration_source"] = {
            "manifest": str(self.manifest),
            "ply_sha256": self.spec["ply_sha256"],
        }
        self._write_variant(variant, spec)
        with self.assertRaisesRegex(SceneContractError, "too few vertices"):
            validate_scene_contract(variant)

    def test_calibration_source_is_validated_as_a_whole_scene(self):
        # The source's own bindings count too: a source whose collision tree
        # no longer matches its digest cannot vouch for a variant.
        variant, spec = self._source_and_variant(
            source_table_z=0.0, variant_table_z=-0.8
        )
        self.spec["table"]["collision"]["tree_sha256"] = "0" * 64
        self._write_manifest()
        spec["calibration_source"] = {
            "manifest": str(self.manifest),
            "ply_sha256": self.spec["ply_sha256"],
        }
        self._write_variant(variant, spec)
        with self.assertRaisesRegex(SceneContractError, "tree_sha256 does not match"):
            validate_scene_contract(variant)

    def test_rejects_missing_or_non_finite_calibration_table_z(self):
        cases = {
            "missing": None,
            "string": "0.0",
            "boolean": False,
            "nan": float("nan"),
            "infinite": float("inf"),
        }
        for label, value in cases.items():
            with self.subTest(label=label):
                if value is None:
                    self.spec.pop("calibration_table_z_m", None)
                else:
                    self.spec["calibration_table_z_m"] = value
                self.manifest.write_text(
                    json.dumps(self.spec, allow_nan=True),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    SceneContractError,
                    "must declare calibration_table_z_m as a finite",
                ):
                    validate_scene_contract(self.scene)
        self.spec["calibration_table_z_m"] = 0.0

    def test_rejects_unknown_measurement_method(self):
        self.spec["calibration_table_z_method"] = "eyeballed"
        self._write_manifest()

        with self.assertRaisesRegex(SceneContractError, "calibration_table_z_method"):
            validate_scene_contract(self.scene)

        # The method is optional and defaults to the only supported method.
        self.spec.pop("calibration_table_z_method")
        self._write_manifest()
        self.assertEqual(validate_scene_contract(self.scene)["calibration_table_z_m"], 0.0)

    def test_declared_value_must_match_runtime_table_z_convention(self):
        for value in (TABLE_Z_TOLERANCE_M, -TABLE_Z_TOLERANCE_M):
            with self.subTest(value=value):
                self.spec["calibration_table_z_m"] = value
                self._write_manifest()
                self.assertEqual(
                    validate_scene_contract(self.scene)["calibration_table_z_m"],
                    value,
                )
        for value in (TABLE_Z_TOLERANCE_M + 1e-3, 0.8, -0.787):
            with self.subTest(value=value):
                self.spec["calibration_table_z_m"] = value
                self._write_manifest()
                with self.assertRaisesRegex(SceneContractError, "table_z=0 convention"):
                    validate_scene_contract(self.scene)

    def test_measure_recovers_plane_through_scaled_and_rotated_w2gs(self):
        scaled = _scaled_identity(3.5)
        self._write_scene_ply(scaled, table_z_world=0.8)
        self.assertGreaterEqual(ply_vertex_count(self.ply), TABLE_Z_MIN_VERTICES)

        self.assertAlmostEqual(
            measure_gs_table_plane_z(self.ply, scaled),
            0.8,
            delta=2e-3,
        )

        angle = math.radians(37.0)
        rotated = np.eye(4, dtype=np.float64)
        rotated[:3, :3] = 3.5 * np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [0.0, 0.0, -1.0],
                [math.sin(angle), math.cos(angle), 0.0],
            ]
        )
        rotated[:3, 3] = [0.9, 3.7, 0.2]
        self._write_scene_ply(rotated, table_z_world=-0.013)
        self.assertAlmostEqual(
            measure_gs_table_plane_z(self.ply, rotated),
            -0.013,
            delta=2e-3,
        )
        # The measurement is a pure function of its inputs.
        self.assertEqual(
            measure_gs_table_plane_z(self.ply, rotated),
            measure_gs_table_plane_z(self.ply, rotated),
        )

        scaled = _scaled_identity(2.0)
        self._write_scene_ply(scaled, table_z_world=0.01, text=True)

        self.assertAlmostEqual(
            measure_gs_table_plane_z(self.ply, scaled),
            0.01,
            delta=2e-3,
        )

    def test_measure_rejects_point_clouds_without_a_table_probe(self):
        scaled = _scaled_identity(3.5)
        self._write_scene_ply(scaled, table_z_world=0.0)
        far_away = np.eye(4, dtype=np.float64)
        far_away[:3, 3] = [50.0, 50.0, 0.0]

        with self.assertRaisesRegex(SceneContractError, "table probe region"):
            measure_gs_table_plane_z(self.ply, far_away)

        self._write_scene_ply(scaled, table_z_world=5.0)
        with self.assertRaisesRegex(SceneContractError, r"within \[-1.5, 1.5\]"):
            measure_gs_table_plane_z(self.ply, scaled)

    def test_validate_rejects_w2gs_calibrated_for_another_table_z(self):
        # A scene registered with the table top at +0.8 m (the legacy GS
        # calibration convention) while the contract claims the z=0 runtime
        # convention: the bytes all verify, only the geometry is wrong.
        scaled = _scaled_identity(3.5)
        self._write_scene_ply(scaled, table_z_world=0.8)
        self._use_inline_matrix(scaled)
        self.spec["calibration_table_z_m"] = 0.0
        self._write_manifest()

        with self.assertRaisesRegex(
            SceneContractError,
            r"calibration_table_z_m=\+0\.0000 m disagrees .* \+0\.80\d\d m",
        ):
            validate_scene_contract(self.scene)

    def test_upgrade_refuses_unbaked_scene_and_records_baked_measurement(self):
        scaled = _scaled_identity(3.5)
        self._write_scene_ply(scaled, table_z_world=0.8)
        np.save(self.w2gs, scaled)
        self.manifest.write_text(
            json.dumps(
                {
                    "name": "legacy_scene",
                    "ply_path": "point_cloud.ply",
                    "w2gs": "gs_w2gs.npy",
                }
            ),
            encoding="utf-8",
        )

        # Without the offset the measured table top is +0.8 m: the upgrade
        # must not produce a contract that only documents the bad convention.
        with self.assertRaisesRegex(SceneContractError, "table_z=0 convention"):
            build_scene_contract(
                self.scene,
                table_collision_directory=self.collision,
            )

        payload = build_scene_contract(
            self.scene,
            table_collision_directory=self.collision,
            calibration_world_offset=(0.0, 0.0, 0.8),
        )

        self.assertEqual(payload["calibration_table_z_method"], TABLE_Z_METHOD)
        self.assertAlmostEqual(payload["calibration_table_z_m"], 0.0, delta=2e-3)
        self.assertIsInstance(payload["calibration_table_z_m"], float)

        self.manifest.write_text(json.dumps(payload), encoding="utf-8")
        report = validate_scene_contract(self.scene)
        self.assertEqual(report["calibration_table_z_m"], payload["calibration_table_z_m"])
        self.assertAlmostEqual(report["gs_table_plane_z_m"], 0.0, delta=2e-3)

    def test_upgrade_cli_writes_measured_table_z(self):
        scaled = _scaled_identity(3.5)
        self._write_scene_ply(scaled, table_z_world=0.787)
        self.manifest.write_text(
            json.dumps(
                {
                    "name": "legacy_scene",
                    "ply_path": "point_cloud.ply",
                    "mat_w2gs": scaled.tolist(),
                }
            ),
            encoding="utf-8",
        )

        self.assertEqual(
            scene_contract_module._main(
                [
                    "upgrade",
                    str(self.scene),
                    "--table-collision-dir",
                    str(self.collision),
                    "--calibration-world-offset",
                    "0",
                    "0",
                    "0.8",
                    "--write",
                ]
            ),
            0,
        )

        written = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(written["calibration_table_z_method"], TABLE_Z_METHOD)
        self.assertAlmostEqual(written["calibration_table_z_m"], -0.013, delta=2e-3)

    def test_measurement_tolerance_boundary(self):
        scaled = _scaled_identity(3.5)
        self._write_scene_ply(scaled, table_z_world=-0.03)
        self._use_inline_matrix(scaled)
        measured = measure_gs_table_plane_z(self.ply, scaled)
        self.assertAlmostEqual(measured, -0.03, delta=1e-3)

        self.spec["calibration_table_z_m"] = round(measured + TABLE_Z_TOLERANCE_M - 1e-3, 6)
        self._write_manifest()
        report = validate_scene_contract(self.scene)
        self.assertEqual(report["gs_table_plane_z_m"], measured)

        self.spec["calibration_table_z_m"] = round(measured + TABLE_Z_TOLERANCE_M + 1e-3, 6)
        self._write_manifest()
        with self.assertRaisesRegex(SceneContractError, "disagrees"):
            validate_scene_contract(self.scene)

    def test_tiny_ply_is_accepted_unmeasured(self):
        scaled = _scaled_identity(3.5)
        # 40 table points + 4 clutter points: below the fixture threshold.
        self._write_scene_ply(scaled, table_z_world=0.8, count=40)
        self.assertLess(ply_vertex_count(self.ply), TABLE_Z_MIN_VERTICES)
        self._use_inline_matrix(scaled)
        self.spec["calibration_table_z_m"] = 0.0
        self._write_manifest()

        report = validate_scene_contract(self.scene)
        self.assertEqual(report["calibration_table_z_m"], 0.0)
        self.assertIsNone(report["gs_table_plane_z_m"])

        self.manifest.write_text(
            json.dumps(
                {
                    "name": "tiny_scene",
                    "ply_path": "point_cloud.ply",
                    "mat_w2gs": scaled.tolist(),
                }
            ),
            encoding="utf-8",
        )
        payload = build_scene_contract(
            self.scene,
            table_collision_directory=self.collision,
        )
        self.assertEqual(payload["calibration_table_z_m"], 0.0)
        self.assertEqual(payload["calibration_table_z_method"], TABLE_Z_METHOD)

    def test_ply_vertex_count_reads_only_the_header(self):
        self.assertEqual(ply_vertex_count(self.ply), 0)
        self._write_scene_ply(_scaled_identity(1.0), table_z_world=0.0, count=123)
        self.assertEqual(ply_vertex_count(self.ply), 123 + 12)

        self.ply.write_bytes(b"ply\r\nformat binary_little_endian 1.0\r\nelement vertex 000042\r\nproperty float x\r\nend_header\r\n")
        self.assertEqual(ply_vertex_count(self.ply), 42)

        for label, payload in {
            "no magic": b"not a ply\n",
            "no end_header": b"ply\nformat ascii 1.0\nelement vertex 3\n",
            "bad count": b"ply\nelement vertex many\nend_header\n",
        }.items():
            with self.subTest(label=label):
                self.ply.write_bytes(payload)
                with self.assertRaises(SceneContractError):
                    ply_vertex_count(self.ply)

    def test_v2_rejects_non_ply_point_cloud(self):
        self.ply.write_bytes(b"not a point cloud")
        self.spec["ply_sha256"] = _sha256(self.ply)
        self._write_manifest()

        with self.assertRaisesRegex(SceneContractError, "not a PLY file"):
            validate_scene_contract(self.scene)


if __name__ == "__main__":
    unittest.main()


def _remove_table_arena_fallback_geometries():
    from robosuite_env.tasks.envs.cup_plate_registered import (
        remove_table_arena_fallback_geometries,
    )

    return remove_table_arena_fallback_geometries


def test_table_arena_fallback_geometry_is_removed_not_only_hidden(robosuite):
    root = ET.fromstring(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" contype="1" />
            <geom name="wall_front_visual" contype="0" />
            <body name="table">
              <geom name="table_collision" contype="1" />
              <geom name="table_visual" contype="0" />
              <geom name="table_leg1_visual" contype="0" />
            </body>
            <body name="task">
              <geom name="container_wall_xneg" contype="1" />
              <geom name="robot_visual" contype="0" />
              <geom name="gw_desk_coll_0" contype="1" />
            </body>
          </worldbody>
        </mujoco>
        """
    )

    removed = _remove_table_arena_fallback_geometries()(root)
    remaining = {geom.get("name") for geom in root.iter("geom")}

    assert removed == {
        "floor",
        "wall_front_visual",
        "table_collision",
        "table_visual",
        "table_leg1_visual",
    }
    assert remaining == {
        "container_wall_xneg",
        "robot_visual",
        "gw_desk_coll_0",
    }


def test_missing_table_arena_physics_fails_closed(robosuite):
    root = ET.fromstring("<mujoco><worldbody /></mujoco>")
    with pytest.raises(ValueError, match="ambiguous table physics"):
        _remove_table_arena_fallback_geometries()(root)
