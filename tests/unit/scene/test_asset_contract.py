# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import pytest

from robosuite_env.contracts.assets import (
    AssetContractError,
    OBJECT_RUNTIME_GEOMETRY,
    runtime_bundle_ids,
    runtime_geometry_for_object,
    validate_repository_assets,
)


REPOSITORY_ASSET_ROOT = Path(__file__).resolve().parents[3] / "robosuite_env" / "assets"
ASSET_ROOT = REPOSITORY_ASSET_ROOT


def test_runtime_bundle_selection_is_centralized():
    assert runtime_bundle_ids("fr3", "cup_plate") == (
        "yubi_right",
        "yubi_left",
        "fr3",
        "cup",
        "plate",
    )
    assert runtime_bundle_ids("fr3", " BOX_PACK ") == runtime_bundle_ids(
        "fr3", "box_pack"
    )
    assert runtime_bundle_ids("FR3", "cup_plate") == runtime_bundle_ids(
        "fr3", "cup_plate"
    )
    with pytest.raises(AssetContractError, match="unsupported simulation robot"):
        runtime_bundle_ids("miru", "cup_plate")


@pytest.mark.parametrize(
    ("robot", "rig"),
    [
        ("fr3_duo", ("yubi_right", "yubi_left", "fr3")),
        ("fr3", ("yubi_right", "yubi_left", "fr3")),
    ],
)
def test_each_task_bundle_set_is_exactly_the_objects_its_env_builds(robot, rig):
    """A run's asset contract covers what its env compiles and nothing else.

    The names are the ones each task's ``configure_*_task`` passes to
    ``OBJECT_ASSET_REGISTRY``; cup_plate and a positions JSON build the base
    ``CupPlateLiftGW`` pair.
    """

    from robosuite_env.tasks.box_pack import BOX_PACK_OBJECT_NAMES

    built = {
        "cup_plate": ("cup", "plate"),
        "robosuite_env/obj_calib/scene_positions_duo.json": ("cup", "plate"),
        "box_pack": BOX_PACK_OBJECT_NAMES,
    }
    for objects, names in built.items():
        assert runtime_bundle_ids(robot, objects) == (*rig, *names), objects


def test_runtime_geometry_api_is_immutable_for_canonical_objects():
    assert set(OBJECT_RUNTIME_GEOMETRY) == {
        "cup",
        "plate",
        "box",
        "tape",
        "pen",
        "lan",
    }
    cup = runtime_geometry_for_object("cup")
    plate = runtime_geometry_for_object("plate")
    assert runtime_geometry_for_object(" CUP ") is cup
    assert runtime_geometry_for_object("plate") is plate
    assert cup["source_collision_geometry"] == {
        "bounds_min_m": (-0.0365698154612, -0.0366110705429, -0.0003977204099),
        "bounds_max_m": (0.0365501498264, 0.0365593006575, 0.0846023112278),
        "z_min_m": -0.0003977204099,
        "z_max_m": 0.0846023112278,
        "xy_radius_m": 0.0368424624415595,
    }
    assert plate["source_collision_geometry"]["z_min_m"] == -0.0006518344298
    assert plate["source_collision_geometry"]["z_max_m"] == 0.01964160204
    assert (
        plate["source_collision_geometry"]["xy_radius_m"]
        == 0.0902683802941864
    )
    assert cup["semantic_sites_m"]["top_site"] == (0.0, 0.0, 0.08274)
    assert plate["semantic_sites_m"]["horizontal_radius_site"] == (
        0.089371,
        0.0,
        0.0,
    )
    with pytest.raises(TypeError):
        cup["source_collision_geometry"]["z_max_m"] = 1.0
    with pytest.raises(AssetContractError, match="no runtime geometry"):
        runtime_geometry_for_object("unknown")

    # MuJoCo stores mesh vertices as float32.  Its compiled observation must
    # remain a close audit value, never the canonical success/placement input.
    for geometry in (cup, plate):
        source = geometry["source_collision_geometry"]
        compiled = geometry["compiled_collision_observed"]
        for key in ("z_min_m", "z_max_m", "xy_radius_m"):
            assert abs(source[key] - compiled[key]) <= 5e-9
        for key in ("bounds_min_m", "bounds_max_m"):
            assert max(abs(a - b) for a, b in zip(source[key], compiled[key])) <= 5e-9


def _bundle(*, visual_prefix: str = "visual/", collision_prefix: str = "collision/"):
    return {
        "mjcf": "object.xml",
        "required_roles": ["visual", "collision"],
        "roles": {
            "visual": {
                "expected_count": 1,
                "allowed_prefixes": [visual_prefix],
            },
            "collision": {
                "expected_count": 1,
                "allowed_prefixes": [collision_prefix],
            },
        },
    }


def _write_manifest(
    root: Path,
    *,
    bundle: dict | None = None,
    forbidden_hashes: list[str] | None = None,
) -> Path:
    manifest = {
        "schema_version": 1,
        "bundles": {"object": bundle or _bundle()},
        "forbidden": {
            "path_globs": ["**/cuph_*.obj"],
            "candidate_sizes": [
                len(b"old forbidden collision mesh")
                if forbidden_hashes
                else 999999
            ],
            "sha256": forbidden_hashes or ["0" * 64],
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_valid_assets(root: Path) -> None:
    (root / "visual").mkdir(parents=True)
    (root / "collision").mkdir()
    (root / "visual" / "object.obj").write_bytes(b"visual mesh")
    (root / "collision" / "object.obj").write_bytes(b"collision mesh")
    (root / "texture.png").write_bytes(b"texture")
    (root / "object.xml").write_text(
        """<mujoco model="object">
  <asset>
    <texture name="texture" file="texture.png"/>
    <mesh name="object_visual" file="visual/object.obj"/>
    <mesh name="object_collision" file="collision/object.obj"/>
  </asset>
  <worldbody><body>
    <geom name="visual" mesh="object_visual" contype="0" conaffinity="0"/>
    <geom name="collision" mesh="object_collision"/>
  </body></worldbody>
</mujoco>
""",
        encoding="utf-8",
    )


def test_valid_bundle_classifies_roles_and_returns_content_digest(tmp_path: Path):
    root = tmp_path / "assets"
    _write_valid_assets(root)
    _write_manifest(root)

    first = validate_repository_assets(root, bundle_ids=["object"])
    assert first["bundles"]["object"]["roles"] == {
        "collision": {
            "count": 1,
            "files": ["collision/object.obj"],
        },
        "visual": {
            "count": 1,
            "files": ["visual/object.obj"],
        },
    }
    assert len(first["contract_digest"]) == 64
    assert {item["path"] for item in first["selected_files"]} == {
        "collision/object.obj",
        "object.xml",
        "texture.png",
        "visual/object.obj",
    }

    (root / "visual" / "object.obj").write_bytes(b"updated visual mesh")
    second = validate_repository_assets(root, bundle_ids=["object"])
    assert second["contract_digest"] != first["contract_digest"]


def test_derived_bundle_reuses_parent_geometry_contract(tmp_path: Path):
    root = tmp_path / "assets"
    _write_valid_assets(root)
    variant_xml = root / "variant.xml"
    variant_xml.write_text(
        (root / "object.xml").read_text(encoding="utf-8").replace(
            'model="object"', 'model="variant"'
        ),
        encoding="utf-8",
    )
    manifest_path = _write_manifest(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bundles"]["variant"] = {
        "derived_from": "object",
        "mjcf": "variant.xml",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_repository_assets(root, bundle_ids=["variant"])
    assert report["bundles"]["variant"]["mjcf"] == "variant.xml"
    assert report["bundles"]["variant"]["roles"] == {
        "collision": {"count": 1, "files": ["collision/object.obj"]},
        "visual": {"count": 1, "files": ["visual/object.obj"]},
    }

    manifest["bundles"]["variant"]["roles"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AssetContractError, match="may define only"):
        validate_repository_assets(root, bundle_ids=["variant"])


def test_missing_or_out_of_root_mjcf_file_fails_closed(tmp_path: Path):
    root = tmp_path / "assets"
    _write_valid_assets(root)
    _write_manifest(root)
    (root / "collision" / "object.obj").unlink()
    with pytest.raises(AssetContractError, match="is missing"):
        validate_repository_assets(root)

    outside = tmp_path / "outside.obj"
    outside.write_bytes(b"outside")
    xml = (root / "object.xml").read_text(encoding="utf-8")
    (root / "object.xml").write_text(
        xml.replace("collision/object.obj", "../outside.obj"),
        encoding="utf-8",
    )
    with pytest.raises(AssetContractError, match="escapes the asset root"):
        validate_repository_assets(root)


def test_wrong_role_directory_is_rejected(tmp_path: Path):
    root = tmp_path / "assets"
    _write_valid_assets(root)
    _write_manifest(root, bundle=_bundle(visual_prefix="approved_visual/"))
    with pytest.raises(AssetContractError, match="outside its allowed directories"):
        validate_repository_assets(root)


def test_one_mesh_cannot_be_visual_and_collision(tmp_path: Path):
    root = tmp_path / "assets"
    _write_valid_assets(root)
    _write_manifest(root)
    xml_path = root / "object.xml"
    xml_path.write_text(
        xml_path.read_text(encoding="utf-8").replace(
            '<geom name="collision" mesh="object_collision"/>',
            '<geom name="collision" mesh="object_collision"/>'
            '<geom name="wrong" mesh="object_visual"/>',
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssetContractError, match="both visual and collision"):
        validate_repository_assets(root)


def test_mjcf_include_is_rejected_instead_of_bypassing_role_checks(tmp_path: Path):
    root = tmp_path / "assets"
    _write_valid_assets(root)
    _write_manifest(root)
    (root / "extra.xml").write_text("<mujoco/>", encoding="utf-8")
    xml_path = root / "object.xml"
    xml_path.write_text(
        xml_path.read_text(encoding="utf-8").replace(
            '<asset>',
            '<include file="extra.xml"/><asset>',
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssetContractError, match="uses <include>"):
        validate_repository_assets(root)


@pytest.mark.parametrize("attribute", ["assetdir", "strippath"])
def test_compiler_path_rewrites_cannot_bypass_manifest_roles(
    tmp_path: Path,
    attribute: str,
):
    root = tmp_path / "assets"
    _write_valid_assets(root)
    _write_manifest(root)
    xml_path = root / "object.xml"
    xml_path.write_text(
        xml_path.read_text(encoding="utf-8").replace(
            '<mujoco model="object">',
            f'<mujoco model="object"><compiler {attribute}="true"/>',
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssetContractError, match="unsupported path resolution"):
        validate_repository_assets(root)


def test_duplicate_mesh_file_reference_is_rejected(tmp_path: Path):
    root = tmp_path / "assets"
    _write_valid_assets(root)
    bundle = _bundle()
    bundle["roles"]["visual"]["expected_count"] = 2
    _write_manifest(root, bundle=bundle)
    xml_path = root / "object.xml"
    xml_path.write_text(
        xml_path.read_text(encoding="utf-8")
        .replace(
            '<mesh name="object_visual" file="visual/object.obj"/>',
            '<mesh name="object_visual" file="visual/object.obj"/>'
            '<mesh name="object_visual_copy" file="visual/object.obj"/>',
        )
        .replace(
            '<geom name="visual" mesh="object_visual" contype="0" conaffinity="0"/>',
            '<geom name="visual" mesh="object_visual" contype="0" conaffinity="0"/>'
            '<geom name="visual_copy" mesh="object_visual_copy" '
            'contype="0" conaffinity="0"/>',
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssetContractError, match="same mesh file more than once"):
        validate_repository_assets(root)


def test_forbidden_path_is_rejected_even_when_unreferenced(tmp_path: Path):
    root = tmp_path / "assets"
    _write_valid_assets(root)
    _write_manifest(root)
    archive = root / "archive"
    archive.mkdir()
    forbidden = archive / "cuph_999.obj"
    forbidden.write_bytes(b"unused forbidden file")
    try:
        with pytest.raises(AssetContractError, match="forbidden asset path"):
            validate_repository_assets(root)
    finally:
        # The forbidden filename must not survive even in pytest's retained
        # temporary directories.
        forbidden.unlink(missing_ok=True)


def test_forbidden_content_is_rejected_after_rename(tmp_path: Path):
    root = tmp_path / "assets"
    _write_valid_assets(root)
    forbidden_content = b"old forbidden collision mesh"
    _write_manifest(
        root,
        forbidden_hashes=[hashlib.sha256(forbidden_content).hexdigest()],
    )
    (root / "collision" / "renamed_safe_looking.obj").write_bytes(forbidden_content)
    with pytest.raises(AssetContractError, match="renaming is not allowed"):
        validate_repository_assets(root)


def test_repository_manifest_pins_current_cup_to_38_coacd_parts(tmp_path: Path):
    source_manifest = json.loads(
        (REPOSITORY_ASSET_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    forbidden_hashes = source_manifest["forbidden"]["sha256"]
    candidate_sizes = source_manifest["forbidden"]["candidate_sizes"]
    assert forbidden_hashes
    assert len(set(forbidden_hashes)) == len(forbidden_hashes)
    assert candidate_sizes
    assert len(set(candidate_sizes)) == len(candidate_sizes)
    assert source_manifest["bundles"]["cup"]["roles"]["collision"] == {
        "expected_count": 38,
        "allowed_prefixes": ["trellis_cup_plate/coll/"],
        "expected_files": [
            f"trellis_cup_plate/coll/cup_{index:03d}.obj" for index in range(38)
        ],
    }
    assert source_manifest["bundles"]["cup"]["roles"]["visual"][
        "expected_files"
    ] == ["trellis_cup_plate/cup_vis.obj"]
    assert source_manifest["bundles"]["plate"]["roles"]["visual"][
        "expected_files"
    ] == ["trellis_cup_plate/plate_vis.obj"]
    assert source_manifest["bundles"]["plate"]["roles"]["collision"][
        "expected_files"
    ] == [
        f"trellis_cup_plate/coll/plate_{index:03d}.obj" for index in range(20)
    ]
    assert "cuph_" not in (REPOSITORY_ASSET_ROOT / "cup.xml").read_text(
        encoding="utf-8"
    )

    root = tmp_path / "assets"
    collision_root = root / "trellis_cup_plate" / "coll"
    collision_root.mkdir(parents=True)
    shutil.copyfile(REPOSITORY_ASSET_ROOT / "cup.xml", root / "cup.xml")
    for index in range(38):
        filename = f"cup_{index:03d}.obj"
        (collision_root / filename).write_text(
            f"v {index} 0 0\n",
            encoding="utf-8",
        )
    (root / "trellis_cup_plate" / "cup_vis.obj").write_bytes(b"visual")
    (root / "trellis_cup_plate" / "cup.png").write_bytes(b"texture")
    source_manifest["bundles"] = {"cup": source_manifest["bundles"]["cup"]}
    # These placeholder meshes are not the repository's pinned cup meshes, so
    # the fixture pins its own mesh set; the refusal of an unpinned copy is
    # covered by tests/unit/scene/test_runtime_mesh_set_guard.py.
    from robosuite_env.contracts.assets import (
        RUNTIME_MESH_SET_SECTION,
        runtime_mesh_set_sha256,
    )

    source_manifest[RUNTIME_MESH_SET_SECTION] = {
        "cup": runtime_mesh_set_sha256(
            {
                path.relative_to(root).as_posix(): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in (
                    root / "trellis_cup_plate" / "cup_vis.obj",
                    *sorted(collision_root.glob("cup_*.obj")),
                )
            }
        )
    }
    (root / "manifest.json").write_text(
        json.dumps(source_manifest),
        encoding="utf-8",
    )

    report = validate_repository_assets(root)
    collision = report["bundles"]["cup"]["roles"]["collision"]
    assert collision["count"] == 38
    assert collision["files"] == [
        f"trellis_cup_plate/coll/cup_{index:03d}.obj" for index in range(38)
    ]
    geometry = report["bundles"]["cup"]["runtime_geometry"]
    assert geometry["source_collision_geometry"] == {
        "bounds_min_m": [-0.0365698154612, -0.0366110705429, -0.0003977204099],
        "bounds_max_m": [0.0365501498264, 0.0365593006575, 0.0846023112278],
        "z_min_m": -0.0003977204099,
        "z_max_m": 0.0846023112278,
        "xy_radius_m": 0.0368424624415595,
    }

    shutil.copyfile(collision_root / "cup_037.obj", collision_root / "cup_999.obj")
    cup_xml = root / "cup.xml"
    cup_xml.write_text(
        cup_xml.read_text(encoding="utf-8").replace("cup_037.obj", "cup_999.obj"),
        encoding="utf-8",
    )
    with pytest.raises(AssetContractError, match="differs from expected_files"):
        validate_repository_assets(root)


def test_runtime_xml_assets_are_checkout_relative():
    for xml_name in (
        "cup.xml",
        "plate.xml",
        "fr3_robot.xml",
    ):
        xml_path = ASSET_ROOT / xml_name
        root = ET.parse(xml_path).getroot()
        file_attributes = [
            element.attrib["file"]
            for element in root.findall(".//*[@file]")
        ]

        assert file_attributes, f"{xml_name} has no file-backed assets"
        for raw_path in file_attributes:
            asset_path = Path(raw_path)
            assert not asset_path.is_absolute(), (
                f"{xml_name} must not depend on another user's checkout: {raw_path}"
            )
            assert ".." not in asset_path.parts, (
                f"{xml_name} asset must stay below the checkout asset root: {raw_path}"
            )


def test_object_registry_exposes_canonical_classes_and_xml_bindings():
    from robosuite_env.tasks import objects

    expected = {
        "cup": objects.CupObject, "plate": objects.PlateObject,
        "box": objects.BoxObject, "tape": objects.TapeObject,
        "pen": objects.PenObject, "lan": objects.LanCableObject,
    }
    assert objects.OBJECT_ASSET_REGISTRY == expected
    assert objects._VALIDATED_OBJECT_BUNDLES == {
        f"{name}.xml": name for name in expected
    }
