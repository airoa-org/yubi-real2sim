# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Missing payloads skip mesh compilation; existing bad payloads still run."""

import xml.etree.ElementTree as ET

import pytest

from tests._helpers import asset_trees


@pytest.mark.parametrize("bundle", ["cup", "plate", "box", "tape", "pen", "lan"])
def test_marker_detects_every_referenced_file(tmp_path, monkeypatch, bundle):
    root = asset_trees.stage_bundle_tree(tmp_path, [bundle])
    monkeypatch.setattr(asset_trees, "REPOSITORY_ASSET_ROOT", root)
    assert asset_trees.requires_repository_meshes(bundle).args == (False,)
    xml = ET.parse(root / f"{bundle}.xml").getroot()
    files = {element.get("file") for element in xml.iter() if element.get("file")}
    assert any(path.endswith(".png") for path in files)
    for relative in sorted(files):
        path = root / relative
        contents = path.read_bytes()
        path.unlink()
        marker = asset_trees.requires_repository_meshes(bundle)
        assert marker.args == (True,), relative
        assert bundle in marker.kwargs["reason"]
        assert relative in marker.kwargs["reason"]
        path.write_bytes(contents)
    # The fixture bytes are deliberately not valid meshes or textures: their
    # presence must enable the test so its contract/compilation check can fail.
    assert asset_trees.requires_repository_meshes(bundle).args == (False,)


def test_marker_checks_only_requested_bundles(tmp_path, monkeypatch):
    root = asset_trees.stage_bundle_tree(tmp_path, ["cup", "plate"])
    monkeypatch.setattr(asset_trees, "REPOSITORY_ASSET_ROOT", root)
    (root / "trellis_cup_plate/cup.png").unlink()
    assert asset_trees.requires_repository_meshes("plate").args == (False,)
    assert asset_trees.requires_repository_meshes("cup", "plate").args == (True,)
