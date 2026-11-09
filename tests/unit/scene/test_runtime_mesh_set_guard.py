# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""A stale or partial copy of gitignored object meshes must fail the contract.

The other checks the asset contract applies read tracked text (roles, counts,
file names, pinned runtime geometry, MJCF sites). Without the pinned mesh set,
a stale mesh copy under the current box MJCF would load silently, with a
rocking base and an ~8.5 mm origin offset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import pytest

from real2sim.artifacts_binding import _validate_schema_v2_contracts
from real2sim.config._types import ConfigError
from robosuite_env.contracts.assets import (
    AssetContractError,
    RUNTIME_MESH_SET_SECTION,
    runtime_bundle_ids,
    runtime_mesh_set_sha256,
    validate_repository_assets,
)
from tests._helpers.asset_trees import (
    REPOSITORY_ASSET_ROOT,
    mesh_files,
    repin,
    requires_repository_meshes,
    stage_bundle_tree,
)


def _repository_manifest() -> dict:
    return json.loads(
        (REPOSITORY_ASSET_ROOT / "manifest.json").read_text(encoding="utf-8")
    )


def _pinned_bundles(manifest: dict) -> set[str]:
    return {
        bundle_id
        for bundle_id, definition in manifest["bundles"].items()
        if "derived_from" not in definition and "runtime_geometry" in definition
    }


def test_a_stale_office_mesh_fails_the_asset_contract(tmp_path: Path):
    root = stage_bundle_tree(tmp_path / "assets", ("box",))
    pin = repin(root, "box")
    report = validate_repository_assets(root, bundle_ids=["box"])
    assert report["bundles"]["box"][RUNTIME_MESH_SET_SECTION] == pin

    hull = root / "trellis_office" / "coll" / "box_007.obj"
    current = hull.read_bytes()
    hull.write_bytes(current + b"# hull from the previous build\n")
    with pytest.raises(
        AssetContractError, match=r"'box' runtime mesh set .* is not the pinned one"
    ):
        validate_repository_assets(root, bundle_ids=["box"])

    hull.write_bytes(current)
    visual = root / "trellis_office" / "box_vis.obj"
    visual.write_bytes(b"v 0 0 0.3059\n")
    with pytest.raises(AssetContractError, match="is not the pinned one"):
        validate_repository_assets(root, bundle_ids=["box"])


def test_the_repository_pins_refuse_any_other_mesh_copy(tmp_path: Path):
    """No fixture re-pin: the committed manifest itself rejects a foreign copy."""

    root = stage_bundle_tree(tmp_path / "assets", ("box", "pen"))
    for bundle_id in ("box", "pen"):
        with pytest.raises(
            AssetContractError, match=rf"'{bundle_id}' runtime mesh set"
        ):
            validate_repository_assets(root, bundle_ids=[bundle_id])


def test_a_bundle_with_runtime_geometry_must_pin_its_mesh_set(tmp_path: Path):
    root = stage_bundle_tree(tmp_path / "assets", ("box",))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    del manifest[RUNTIME_MESH_SET_SECTION]["box"]
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AssetContractError, match="pins no mesh set for 'box'"):
        validate_repository_assets(root, bundle_ids=["box"])


def test_derived_bundles_are_checked_against_their_parent_pin(tmp_path: Path):
    root = stage_bundle_tree(tmp_path / "assets", ("cup",))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["bundles"]["fixture_variant"] = {"derived_from": "cup", "mjcf": "cup.xml"}
    (root / "manifest.json").write_text(json.dumps(manifest))
    parent_pin = repin(root, "cup")
    report = validate_repository_assets(root, bundle_ids=["fixture_variant"])
    assert report["bundles"]["fixture_variant"][RUNTIME_MESH_SET_SECTION] == parent_pin

    (root / "trellis_cup_plate" / "coll" / "cup_000.obj").write_bytes(b"v 9 9 9\n")
    with pytest.raises(
        AssetContractError, match=r"'fixture_variant' runtime mesh set .*\['cup'\]"
    ):
        validate_repository_assets(root, bundle_ids=["fixture_variant"])


@pytest.mark.parametrize(
    ("section", "message"),
    (
        ([], "must be an object"),
        ({"no_such_bundle": "0" * 64}, "pins unknown bundle 'no_such_bundle'"),
        ({"fixture_variant": "0" * 64}, "pins derived bundle 'fixture_variant'"),
        ({"box": "not-a-digest"}, "invalid digest for 'box'"),
    ),
)
def test_the_pin_section_itself_is_validated(tmp_path: Path, section, message):
    root = stage_bundle_tree(tmp_path / "assets", ("box",))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["bundles"]["fixture_variant"] = {"derived_from": "cup", "mjcf": "cup.xml"}
    manifest[RUNTIME_MESH_SET_SECTION] = section
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AssetContractError, match=message):
        validate_repository_assets(root, bundle_ids=["box"])


def test_the_per_build_object_loader_refuses_a_stale_copy(tmp_path: Path, monkeypatch):
    from robosuite_env.tasks import objects

    root = stage_bundle_tree(tmp_path / "assets", ("tape",))
    monkeypatch.setattr(objects, "ASSETS_ROOT", root)

    with pytest.raises(AssetContractError, match="'tape' runtime mesh set"):
        objects._validate_object_bundle("tape")


def test_check_runtime_asset_contract_refuses_a_stale_copy(tmp_path: Path):
    bundle_ids = runtime_bundle_ids("fr3_duo", "box_pack")
    stage_bundle_tree(tmp_path / "robosuite_env" / "assets", bundle_ids)
    plan = {
        "repo_root": tmp_path,
        "resolved": {
            "robot": "fr3_duo",
            "task": {"objects": "box_pack"},
            "scenes": [],
        },
    }

    with pytest.raises(
        ConfigError,
        match=r"repository asset contract failed: .*'box' runtime mesh set",
    ):
        _validate_schema_v2_contracts(plan)


def test_the_documented_coreutils_recipe_reproduces_a_pin(tmp_path: Path):
    root = stage_bundle_tree(tmp_path / "assets", ("pen",))
    files = mesh_files(root, "pen")
    completed = subprocess.run(
        ["bash", "-c", 'sha256sum "$@" | LC_ALL=C sort -k2 | sha256sum', "_", *files],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.split()[0] == runtime_mesh_set_sha256(
        {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in files
        }
    )


def test_every_runtime_geometry_bundle_is_pinned():
    manifest = _repository_manifest()
    assert set(manifest[RUNTIME_MESH_SET_SECTION]) == _pinned_bundles(manifest)


@pytest.mark.parametrize(
    "bundle_id",
    [
        pytest.param(bundle_id, id=bundle_id, marks=requires_repository_meshes(bundle_id))
        for bundle_id in sorted(_repository_manifest()[RUNTIME_MESH_SET_SECTION])
    ],
)
def test_repository_pins_name_the_meshes_the_geometry_was_measured_from(bundle_id):
    """Each pin is the set whose collision hulls reproduce the pinned extents.

    One case per pinned mesh set, so an installed set is checked even when
    another set is not installed.
    """

    manifest = _repository_manifest()
    pin = manifest[RUNTIME_MESH_SET_SECTION][bundle_id]
    definition = manifest["bundles"][bundle_id]
    xml = ET.parse(REPOSITORY_ASSET_ROOT / definition["mjcf"]).getroot()
    vertices = []
    for mesh in xml.findall("./asset/mesh"):
        if "/coll/" not in mesh.get("file"):
            continue
        scale = [float(item) for item in (mesh.get("scale") or "1 1 1").split()]
        for line in (REPOSITORY_ASSET_ROOT / mesh.get("file")).read_text(
            encoding="utf-8"
        ).splitlines():
            if line.startswith("v "):
                xyz = [float(item) for item in line.split()[1:4]]
                vertices.append([a * s for a, s in zip(xyz, scale)])
    geometry = definition["runtime_geometry"]["source_collision_geometry"]
    for axis in range(3):
        assert abs(min(v[axis] for v in vertices) - geometry["bounds_min_m"][axis]) < 1e-9
        assert abs(max(v[axis] for v in vertices) - geometry["bounds_max_m"][axis]) < 1e-9

    report = validate_repository_assets(REPOSITORY_ASSET_ROOT, bundle_ids=[bundle_id])
    assert report["bundles"][bundle_id][RUNTIME_MESH_SET_SECTION] == pin
