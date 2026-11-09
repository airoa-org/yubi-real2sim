# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Stage repository asset bundles with placeholder meshes for contract tests.

Also provides a marker that skips a test when the installed (gitignored)
files its MJCF references are absent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable
import xml.etree.ElementTree as ET

import pytest

from robosuite_env.contracts.assets import (
    RUNTIME_MESH_SET_SECTION,
    runtime_mesh_set_sha256,
)
from tests._helpers.repo import REPO_ROOT

REPOSITORY_ASSET_ROOT = REPO_ROOT / "robosuite_env" / "assets"


def stage_bundle_tree(root: Path, bundle_ids: Iterable[str]) -> Path:
    """Copy the real manifest and each bundle's MJCF; write placeholder payloads.

    Every tracked text input -- manifest, runtime geometry, MJCF sites, file
    lists -- is the repository's own, so only the gitignored payload bytes
    differ. Texture payload hashes are repinned in the fixture manifest; mesh
    and parent pins remain unchanged so their validation is still exercised.
    """

    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (REPOSITORY_ASSET_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    shutil.copyfile(REPOSITORY_ASSET_ROOT / "manifest.json", root / "manifest.json")
    for bundle_id in bundle_ids:
        definition = manifest["bundles"][bundle_id]
        pinned = set(definition.get("expected_sha256", {}))
        xml_name = definition["mjcf"]
        shutil.copyfile(REPOSITORY_ASSET_ROOT / xml_name, root / xml_name)
        for element in ET.parse(root / xml_name).getroot().iter():
            raw = element.get("file")
            if raw is None:
                continue
            target = root / raw
            target.parent.mkdir(parents=True, exist_ok=True)
            if raw in pinned:
                target.write_bytes(f"fixture texture {raw}".encode())
                definition["expected_sha256"][raw] = hashlib.sha256(target.read_bytes()).hexdigest()
            elif not target.exists():
                target.write_text(f"v 0 0 0\n# placeholder {raw}\n", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root


def mesh_files(root: Path, bundle_id: str) -> list[str]:
    """Return the asset-root-relative mesh files a bundle's MJCF selects."""

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    definition = manifest["bundles"][bundle_id]
    xml = ET.parse(root / definition["mjcf"]).getroot()
    return sorted(mesh.get("file") for mesh in xml.findall("./asset/mesh"))


def requires_repository_meshes(*bundle_ids: str) -> pytest.MarkDecorator:
    """Skip unless every listed bundle's file-backed MJCF assets are installed.

    Only absence skips: an installed file that is stale or wrong still runs the
    test, which then fails. The reason names each bundle that lacks a file,
    with the first file it lacks.
    """

    manifest = json.loads((REPOSITORY_ASSET_ROOT / "manifest.json").read_text())
    missing = {}
    for bundle_id in bundle_ids:
        xml_path = REPOSITORY_ASSET_ROOT / manifest["bundles"][bundle_id]["mjcf"]
        xml = ET.parse(xml_path).getroot()
        compiler = xml.find("./compiler")
        files = []
        for element in xml.iter():
            raw = element.get("file")
            if raw is None:
                continue
            directory = ""
            if compiler is not None and element.tag in {"mesh", "texture"}:
                directory = compiler.get(element.tag + "dir", "")
            files.append((xml_path.parent / directory / raw).relative_to(REPOSITORY_ASSET_ROOT))
        absent = [
            relative
            for relative in sorted(set(files))
            if not (REPOSITORY_ASSET_ROOT / relative).is_file()
        ]
        if absent:
            missing[bundle_id] = absent[0]
    return pytest.mark.skipif(
        bool(missing),
        reason="gitignored bundle files are not installed in this checkout: "
        + ", ".join(f"{bundle_id!r} (no {path})" for bundle_id, path in missing.items()),
    )


def mesh_set_digest(root: Path, bundle_id: str) -> str:
    return runtime_mesh_set_sha256(
        {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in mesh_files(root, bundle_id)
        }
    )


def repin(root: Path, bundle_id: str) -> str:
    """Pin the staged tree's current meshes as the bundle's mesh set."""

    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    digest = mesh_set_digest(root, bundle_id)
    manifest[RUNTIME_MESH_SET_SECTION][bundle_id] = digest
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return digest
