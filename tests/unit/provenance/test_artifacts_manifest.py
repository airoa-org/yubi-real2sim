# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]

from real2sim import artifacts as real2sim_artifacts
from real2sim.artifacts import ManifestError, load_manifest, verify_artifact
from real2sim import launcher


@pytest.fixture(autouse=True)
def _isolate_cpu_fixtures_from_ambient_cuda(monkeypatch):
    """Do not let a scheduler allocation alter CPU fixture semantics."""

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)




def test_single_manifest_verifies_each_input_fixture_role():
    manifest = load_manifest(REPO_ROOT / "artifacts" / "manifest.yaml")
    selected = (
        "tiny-checkpoint-v1",
        "tiny-scene-v2",
        "tiny-registered-desk-v1",
        "tiny-policy-episode-v1",
        "tiny-repository-assets-v1",
    )
    records = [
        verify_artifact(manifest, artifact_id, repo_root=REPO_ROOT)
        for artifact_id in selected
    ]
    assert [record.artifact_id for record in records] == list(selected)
    assert all(len(record.sha256) == 64 and record.size > 0 for record in records)


def test_pending_production_input_is_fail_closed():
    manifest = load_manifest(REPO_ROOT / "artifacts" / "manifest.yaml")
    manifest["artifacts"]["public-scene-flat-reading-room-v2"]["status"] = "pending"
    with pytest.raises(ManifestError, match="pending"):
        verify_artifact(
            manifest,
            "public-scene-flat-reading-room-v2",
            repo_root=REPO_ROOT,
            environment={"REAL2SIM_INPUTS_ROOT": "/tmp/not-used"},
        )


def test_every_production_artifact_is_public():
    manifest = load_manifest(REPO_ROOT / "artifacts" / "manifest.yaml")
    # Distributed production inputs are the ones a clean checkout downloads.
    production = {
        artifact_id: entry
        for artifact_id, entry in manifest["artifacts"].items()
        if entry["access"] != "fixture"
    }
    assert production
    assert {entry["access"] for entry in production.values()} == {"public"}
    assert all("request" not in entry for entry in production.values())
    # The public SuperSplat rooms carry their creators' CC BY 4.0 licence and,
    # for AIRoA's background.json and table_collision/ in each archive,
    # Apache-2.0 (THIRD_PARTY_NOTICES C.4); their provenance and attribution
    # are checked in test_public_scenes.py.
    public_scenes = {artifact_id for artifact_id in production if artifact_id.startswith("public-scene-")}
    assert public_scenes == {
        f"public-scene-flat-{room}-v2" for room in ("reading-room", "a-room")
    }
    assert {production[artifact_id]["license"] for artifact_id in public_scenes} == {"CC-BY-4.0 AND Apache-2.0"}
    # The policy checkpoints that the release download catalog publishes are
    # Gemma Model Derivatives (fine-tuned from openpi pi0.5, built on
    # PaliGemma), so they carry the Gemma Terms of Use.
    gemma = {
        artifact_id
        for artifact_id, entry in production.items()
        if entry["license"] == "LicenseRef-Gemma-Terms-of-Use"
    }
    assert gemma == {"pi05-cup"}
    assert all("Gemma Model Derivative" in production[artifact_id]["description"] for artifact_id in gemma)
    catalog = json.loads((REPO_ROOT / "artifacts" / "release_downloads.json").read_text())
    published_checkpoints = {
        Path(pin["destination"]).name
        for pin in catalog["archives"].values()
        if pin["destination"].startswith("{inputs_root}/checkpoints/")
    }
    assert {
        Path(production[artifact_id]["local_path"]).name for artifact_id in gemma
    } == published_checkpoints
    archive = {
        artifact_id: entry
        for artifact_id, entry in production.items()
        if artifact_id not in public_scenes and artifact_id not in gemma
    }
    # The FR3 collision meshes and the cup/plate meshes are under the Apache
    # License 2.0 (THIRD_PARTY_NOTICES C.2, C.3).
    assert set(archive) == {
        "fr3-collision-franka-description-2fd6aaa8",
        "public-cup-plate-visual-v1",
        "public-cup-plate-collision-v1",
    }
    assert {entry["license"] for entry in archive.values()} == {"Apache-2.0"}

    # Every production input is published; no locally generated training-run
    # checkpoint is listed.
    assert not any(entry["access"] == "generated" for entry in manifest["artifacts"].values())
    assert not any(artifact_id.startswith("cf80-") for artifact_id in manifest["artifacts"])

    assert "restricted" not in real2sim_artifacts.ALLOWED_ACCESS
    assert "request" not in real2sim_artifacts.ALLOWED_SCHEMES
    assert "request" not in real2sim_artifacts.OPTIONAL_ENTRY_FIELDS


def test_production_inputs_resolve_under_inputs_root():
    # Every production input resolves under the single REAL2SIM_INPUTS_ROOT
    # through the {inputs_root} template field; the per-input variables
    # (REAL2SIM_CHECKPOINT, REAL2SIM_PINHOLE_WHITE_*) are not in the
    # manifest allowlist.
    manifest = load_manifest(REPO_ROOT / "artifacts" / "manifest.yaml")
    assert manifest["environment_allowlist"] == (
        "REAL2SIM_INPUTS_ROOT",
        "REAL2SIM_POLICY_EPISODE",
    )
    assert not real2sim_artifacts._LEGACY_INPUT_ENV_NAMES.intersection(
        manifest["environment_allowlist"]
    )
    # The published scenes are the two public SuperSplat rooms registered to
    # the FR3 Flat rig.
    # A fixture scene is a test double and a generated scene is built
    # locally; neither is downloaded by a clean checkout, so neither belongs
    # in this set.
    published_scene_ids = {
        artifact_id
        for artifact_id, entry in manifest["artifacts"].items()
        if entry["access"] not in {"fixture", "generated"}
        and "/scenes/" in entry["local_path"]
    }
    assert published_scene_ids == {
        "public-scene-flat-reading-room-v2",
        "public-scene-flat-a-room-v2",
    }
    assert "fr3-pinhole-white-scene-release" not in manifest["artifacts"]
    assert "fr3-pinhole-white-scene-v2-migrated" not in manifest["artifacts"]
    expected_layout = {
        "pi05-cup": (
            "{inputs_root}/checkpoints/pi05_cup"
        ),
        "public-scene-flat-a-room-v2": "{inputs_root}/scenes/flat_a_room",
    }
    for artifact_id, local_path in expected_layout.items():
        assert manifest["artifacts"][artifact_id]["local_path"] == local_path
    collision = manifest["artifacts"]["public-cup-plate-collision-v1"]
    assert collision["kind"] == "collection"
    assert collision["local_path"] == (
        "{repo_root}/robosuite_env/assets/trellis_cup_plate/coll"
    )
    assert len(collision["files"]) == 58
    assert (collision["size"], collision["sha256"]) == (
        692167,
        "7980bacec4f9e10eadb840a2a86c52513babd3da39e349be6f1e81f63ca98118",
    )
    for entry in manifest["artifacts"].values():
        assert "${" not in entry["local_path"], entry["id"]




def test_legacy_input_variables_are_rejected_everywhere(tmp_path: Path):
    # A stale profile or manifest that still names one of the retired
    # per-input variables must fail with the migration hint rather than
    # silently resolving through the process environment.
    source = yaml.safe_load(
        (REPO_ROOT / "config/release/fr3/flat/reading_room_cup_cup1trim.yaml").read_text(
            encoding="utf-8"
        )
    )
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    environment = {"REAL2SIM_INPUTS_ROOT": str(tmp_path / "inputs")}

    stale_reference = dict(source)
    stale_reference["policy"] = dict(source["policy"], checkpoint="${REAL2SIM_CHECKPOINT}")
    path = config_dir / "stale-reference.yaml"
    path.write_text(yaml.safe_dump(stale_reference, sort_keys=False), encoding="utf-8")
    with mock.patch.dict(os.environ, environment, clear=False):
        with pytest.raises(launcher.ConfigError, match="REAL2SIM_CHECKPOINT is retired"):
            launcher.load_rollout_config(path, repo_root=tmp_path)

    stale_allowlist = yaml.safe_load(yaml.safe_dump(source))
    stale_allowlist["reproducibility"]["environment"]["allowlist"].append(
        "REAL2SIM_PINHOLE_WHITE_SCENE"
    )
    path = config_dir / "stale-allowlist.yaml"
    path.write_text(yaml.safe_dump(stale_allowlist, sort_keys=False), encoding="utf-8")
    with mock.patch.dict(os.environ, environment, clear=False):
        with pytest.raises(
            launcher.ConfigError,
            match="allowlist names retired variables: REAL2SIM_PINHOLE_WHITE_SCENE is retired",
        ):
            launcher.load_rollout_config(path, repo_root=tmp_path)

    entry = {"id": "stale", "local_path": "${REAL2SIM_PINHOLE_WHITE_DESK_BUNDLE}"}
    with pytest.raises(ManifestError, match="REAL2SIM_PINHOLE_WHITE_DESK_BUNDLE is retired"):
        real2sim_artifacts.resolve_local_path(
            entry,
            repo_root=REPO_ROOT,
            environment_allowlist=("REAL2SIM_PINHOLE_WHITE_DESK_BUNDLE",),
            environment={"REAL2SIM_PINHOLE_WHITE_DESK_BUNDLE": "/old"},
        )

    # The process environment may still carry the retired names: they are
    # ignored (not allowlisted), never forwarded to the runner.
    with mock.patch.dict(
        os.environ, {**environment, "REAL2SIM_CHECKPOINT": "/old/checkpoint"}, clear=False
    ):
        plan = launcher.load_rollout_config(
            REPO_ROOT / "config/release/fr3/flat/reading_room_cup_cup1trim.yaml",
            timestamp="20260822_000000",
        )
        assert "REAL2SIM_CHECKPOINT" not in launcher.build_runner_environment(plan)
        assert plan["resolved"]["policy"]["checkpoint"] == str(
            tmp_path / "inputs" / "checkpoints" / "pi05_cup"
        )


def test_artifact_byte_change_is_detected(tmp_path):
    manifest = load_manifest(REPO_ROOT / "artifacts" / "manifest.yaml")
    copied = tmp_path / "checkpoint"
    copied.mkdir()
    (copied / "README.txt").write_bytes(
        (REPO_ROOT / "tests" / "fixtures" / "artifacts" / "checkpoint" / "README.txt").read_bytes()
        + b"tampered\n"
    )
    manifest["artifacts"]["tiny-checkpoint-v1"]["local_path"] = str(copied)
    with pytest.raises(ManifestError, match="mismatch"):
        verify_artifact(
            manifest,
            "tiny-checkpoint-v1",
            repo_root=REPO_ROOT,
            environment={},
        )


def test_artifact_verification_reuses_digest_without_reading_content(
    monkeypatch, tmp_path
):
    manifest = load_manifest(REPO_ROOT / "artifacts" / "manifest.yaml")
    copied = tmp_path / "checkpoint"
    copied.mkdir()
    source = (
        REPO_ROOT
        / "tests"
        / "fixtures"
        / "artifacts"
        / "checkpoint"
        / "README.txt"
    )
    (copied / "README.txt").write_bytes(source.read_bytes())
    manifest["artifacts"]["tiny-checkpoint-v1"]["local_path"] = str(copied)
    verification = real2sim_artifacts.verify_artifacts_with_snapshot(
        manifest,
        ["tiny-checkpoint-v1"],
        repo_root=REPO_ROOT,
        environment={},
    )

    def unexpected_hash(_path):
        raise AssertionError("artifact content must not be hashed twice")

    monkeypatch.setattr(real2sim_artifacts, "sha256_file", unexpected_hash)
    records = real2sim_artifacts.reuse_artifact_verification(
        verification,
        manifest,
        ["tiny-checkpoint-v1"],
        repo_root=REPO_ROOT,
        environment={},
    )

    assert records == list(verification.records)


def test_artifact_verification_rejects_post_doctor_file_change(tmp_path):
    manifest = load_manifest(REPO_ROOT / "artifacts" / "manifest.yaml")
    copied = tmp_path / "checkpoint"
    copied.mkdir()
    source = (
        REPO_ROOT
        / "tests"
        / "fixtures"
        / "artifacts"
        / "checkpoint"
        / "README.txt"
    )
    artifact_file = copied / "README.txt"
    artifact_file.write_bytes(source.read_bytes())
    manifest["artifacts"]["tiny-checkpoint-v1"]["local_path"] = str(copied)
    verification = real2sim_artifacts.verify_artifacts_with_snapshot(
        manifest,
        ["tiny-checkpoint-v1"],
        repo_root=REPO_ROOT,
        environment={},
    )

    artifact_file.write_bytes(artifact_file.read_bytes() + b"tampered\n")

    with pytest.raises(ManifestError, match="changed after doctor verification"):
        real2sim_artifacts.reuse_artifact_verification(
            verification,
            manifest,
            ["tiny-checkpoint-v1"],
            repo_root=REPO_ROOT,
            environment={},
        )


def test_artifact_verification_rejects_pending_manifest_entry(tmp_path):
    manifest = load_manifest(REPO_ROOT / "artifacts" / "manifest.yaml")
    copied = tmp_path / "checkpoint"
    copied.mkdir()
    source = (
        REPO_ROOT
        / "tests"
        / "fixtures"
        / "artifacts"
        / "checkpoint"
        / "README.txt"
    )
    (copied / "README.txt").write_bytes(source.read_bytes())
    entry = manifest["artifacts"]["tiny-checkpoint-v1"]
    entry["local_path"] = str(copied)
    verification = real2sim_artifacts.verify_artifacts_with_snapshot(
        manifest,
        ["tiny-checkpoint-v1"],
        repo_root=REPO_ROOT,
        environment={},
    )

    entry["status"] = "pending"
    with pytest.raises(ManifestError, match="pending and cannot be used"):
        real2sim_artifacts.reuse_artifact_verification(
            verification,
            manifest,
            ["tiny-checkpoint-v1"],
            repo_root=REPO_ROOT,
            environment={},
        )




def test_collection_manifest_rejects_parent_path_before_install(tmp_path):
    source = yaml.safe_load(
        (REPO_ROOT / "artifacts" / "manifest.yaml").read_text(encoding="utf-8")
    )
    collection = next(
        item
        for item in source["artifacts"]
        if item["id"] == "public-cup-plate-visual-v1"
    )
    collection["files"] = ["../outside-repository"]
    manifest_path = tmp_path / "unsafe-manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(source, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(ManifestError, match="canonical POSIX relative path"):
        load_manifest(manifest_path)


# ---------------------------------------------------------------------------
# REAL2SIM_INPUTS_ROOT / {inputs_root}: the single machine-local path root.
# ---------------------------------------------------------------------------


def test_manifest_entry_under_inputs_root_verifies_against_installed_bytes(tmp_path: Path):
    # The tiny fixture checkpoint copied under a temporary inputs root must
    # verify through an {inputs_root} local_path exactly like a production entry.
    import shutil

    manifest = load_manifest(REPO_ROOT / "artifacts" / "manifest.yaml")
    fixture = manifest["artifacts"]["tiny-checkpoint-v1"]
    root = tmp_path / "inputs"
    shutil.copytree(
        REPO_ROOT / "tests" / "fixtures" / "artifacts" / "checkpoint",
        root / "checkpoints" / "tiny",
    )
    manifest["environment_allowlist"] = tuple(manifest["environment_allowlist"]) + (
        "REAL2SIM_INPUTS_ROOT",
    )
    manifest["artifacts"]["tiny-under-root"] = {
        **fixture,
        "id": "tiny-under-root",
        "local_path": "{inputs_root}/checkpoints/tiny",
    }
    record = verify_artifact(
        manifest,
        "tiny-under-root",
        repo_root=REPO_ROOT,
        environment={"REAL2SIM_INPUTS_ROOT": str(root)},
    )
    assert record.path == str(root / "checkpoints" / "tiny")
    assert record.sha256 == fixture["sha256"]
    with pytest.raises(ManifestError, match="REAL2SIM_INPUTS_ROOT is unset"):
        verify_artifact(manifest, "tiny-under-root", repo_root=REPO_ROOT, environment={})


def test_external_runtime_assets_are_owned_by_the_install_manifest():
    manifest = yaml.safe_load(
        (REPO_ROOT / "artifacts" / "manifest.yaml").read_text(encoding="utf-8")
    )
    entries = {entry["id"]: entry for entry in manifest["artifacts"]}
    licensed_files = entries["public-cup-plate-visual-v1"]["files"]
    for relative_path in (
        "robosuite_env/assets/trellis_cup_plate/cup.png",
        "robosuite_env/assets/trellis_cup_plate/cup_vis.obj",
        "robosuite_env/assets/trellis_cup_plate/plate.png",
        "robosuite_env/assets/trellis_cup_plate/plate_vis.obj",
    ):
        assert relative_path in licensed_files

    collision_entry = entries["public-cup-plate-collision-v1"]
    assert collision_entry["kind"] == "collection"
    assert collision_entry["local_path"].endswith(
        "robosuite_env/assets/trellis_cup_plate/coll"
    )
    assert collision_entry["files"] == [
        f"cup_{index:03d}.obj" for index in range(38)
    ] + [
        f"plate_{index:03d}.obj" for index in range(20)
    ]

    public_files = entries["fr3-collision-franka-description-2fd6aaa8"]["files"]
    for relative_path in ("fr3_collision/link0.stl", "fr3_collision/link7.stl"):
        assert f"robosuite_env/assets/{relative_path}" in public_files

    for entry in entries.values():
        assert {
            "name",
            "version",
            "uri",
            "size",
            "sha256",
            "license",
            "access",
            "created_at",
            "generator_commit",
        } <= set(entry)


def test_an_undeclared_artifact_is_refused():
    manifest = load_manifest(REPO_ROOT / "artifacts" / "manifest.yaml")
    with pytest.raises(real2sim_artifacts.ManifestError, match="not declared in the manifest: unknown-artifact"):
        real2sim_artifacts.verify_artifact(manifest, "unknown-artifact", repo_root=REPO_ROOT)
