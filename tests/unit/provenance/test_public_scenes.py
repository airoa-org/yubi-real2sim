# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The public scenes are pinned folders; their registration is only in the scene directory.

A public scene directory is one Hugging Face folder. Its ``background.json``
(registration, table-plane height, scene id) is not copied into the
repository: the repository keeps the attribution in
``assets/public_scenes/README.md`` and the integrity pins in
``artifacts/manifest.yaml`` and ``artifacts/release_downloads.json``.

The tests without the ``real_inputs_root`` fixture check the pins and the
README attribution in a clean checkout. ``test_installed_scene_matches_its_pin``
is an ``assets`` test: it checks an installed scene directory under
``$REAL2SIM_INPUTS_ROOT/scenes/`` and is skipped where none is installed.
Each scene directory's ``ATTRIBUTION.txt`` exists only there (the repository
pins it through the tree sha256 but keeps no copy), so its text is checked
there; CI checks only that a missing statement fails with a named assertion.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from real2sim.artifacts import sha256_tree
from robosuite_env.contracts.scene import scene_tag, validate_scene_contract


REPO_ROOT = Path(__file__).resolve().parents[3]
PUBLIC = REPO_ROOT / "assets" / "public_scenes"
CC_BY = "https://creativecommons.org/licenses/by/4.0/"
APACHE = "https://www.apache.org/licenses/LICENSE-2.0"
# ATTRIBUTION.txt ends its CC BY notice with this sentence; the licence of AIRoA's additions follows it.
CC_BY_BOUNDARY = "not licensed by CC BY 4.0."
ROOMS = {
    "reading_room": {
        "title": "Reading Room (No Lidar, Only Photos)",
        "creator": "Ethan",
        "url": "https://superspl.at/scene/9d370db9",
    },
    "a_room": {
        "title": "A Room",
        "creator": "parsa mahdavi",
        "url": "https://superspl.at/scene/3e754ea2",
    },
}
RIGS = {"flat": "franka_flat"}
SCENES = sorted(f"{rig}_{room}" for rig in RIGS for room in ROOMS)


def _artifact_id(scene: str) -> str:
    return "public-scene-" + scene.replace("_", "-") + "-v2"


def _room(scene: str) -> str:
    return scene.split("_", 1)[1]


def _manifest_entries() -> dict[str, dict]:
    manifest = yaml.safe_load((REPO_ROOT / "artifacts" / "manifest.yaml").read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in manifest["artifacts"]}


def _download_pins() -> dict[str, dict]:
    catalog = json.loads((REPO_ROOT / "artifacts" / "release_downloads.json").read_text(encoding="utf-8"))
    return catalog["archives"]


def _check_attribution(attribution: str, room: dict[str, str]) -> None:
    """Assert what a scene directory's ATTRIBUTION.txt states; each failure names the missing text."""
    for required in (f'"{room["title"]}"', room["creator"], room["url"], CC_BY, "Changes made:",
                     "obtained: 2026-11-04 via the scene page's Download button (PLY)", CC_BY_BOUNDARY):
        assert required in attribution, f"ATTRIBUTION.txt does not contain {required!r}"
    # After the CC BY notice: the licence of what this project adds, and the exclusion of
    # third-party works in the capture with a contact for removal requests.
    added = attribution.split(CC_BY_BOUNDARY, 1)[1]
    for required in ("background.json", "table_collision/", "Apache License, Version 2.0", APACHE,
                     "LICENSE-APACHE-2.0", "Third-party works", "ext-oss@airoa.org"):
        assert required in added, f"ATTRIBUTION.txt does not contain {required!r} after {CC_BY_BOUNDARY!r}"


def test_the_repository_keeps_no_copy_of_a_scene_directory():
    tracked = subprocess.run(
        ["git", "ls-files", "--", "assets/public_scenes"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        pytest.skip("not a Git checkout")
    assert tracked.stdout.split() == ["assets/public_scenes/README.md"]


@pytest.mark.parametrize("scene", SCENES)
def test_manifest_and_download_pin_the_same_tree(scene):
    entry = _manifest_entries()[_artifact_id(scene)]
    pin = _download_pins()[scene]
    assert entry["kind"] == "tree"
    assert entry["local_path"] == pin["destination"] == "{inputs_root}/scenes/" + scene
    assert pin["tree"] == {"size": entry["size"], "sha256": entry["sha256"]}
    # The room is CC BY 4.0; AIRoA's background.json and table_collision/ are Apache-2.0.
    assert (entry["license"], entry["access"], entry["status"]) == (
        "CC-BY-4.0 AND Apache-2.0",
        "public",
        "ready",
    )
    room = ROOMS[_room(scene)]
    assert f'"{room["title"]}" by {room["creator"]}' in entry["name"]
    assert entry["description"].startswith(room["url"] + " ")


def test_attribution_check_names_a_missing_cc_by_boundary():
    room = ROOMS["a_room"]
    # Every statement before the boundary is present, so the failure is the boundary itself.
    text = (f'"{room["title"]}" by {room["creator"]}, {room["url"]}, {CC_BY}. Changes made: '
            "obtained: 2026-11-04 via the scene page's Download button (PLY)")
    with pytest.raises(AssertionError, match=re.escape(repr(CC_BY_BOUNDARY))):
        _check_attribution(text, room)


def test_readme_credits_both_rooms():
    text = (PUBLIC / "README.md").read_text(encoding="utf-8")
    assert CC_BY in text
    for room in ROOMS.values():
        for required in (f'"{room["title"]}"', room["creator"], room["url"]):
            assert required in text, required
    for scene in SCENES:
        assert f"`{scene}`" in text and f"`{_artifact_id(scene)}`" in text


@pytest.mark.parametrize("scene", SCENES)
def test_installed_scene_matches_its_pin(scene, real_inputs_root):
    directory = real_inputs_root / "scenes" / scene
    if not (directory / "background.json").is_file():
        pytest.skip(f"public scene {scene} is not installed")
    entry = _manifest_entries()[_artifact_id(scene)]
    digest = sha256_tree(directory)
    assert (digest.size, digest.sha256) == (entry["size"], entry["sha256"])

    report = validate_scene_contract(directory, strict=True)
    rig = scene.split("_", 1)[0]
    assert report["name"] == RIGS[rig] and scene_tag(report["name"]) == RIGS[rig]
    assert report["id"].startswith(report["name"] + "-")
    spec = json.loads((directory / "background.json").read_text(encoding="utf-8"))
    assert spec["license"] == "CC-BY-4.0"
    room = ROOMS[_room(scene)]
    assert room["url"] in spec["source_scene"]
    _check_attribution((directory / "ATTRIBUTION.txt").read_text(encoding="utf-8"), room)
    licence = (directory / "LICENSE-APACHE-2.0").read_text(encoding="utf-8")
    assert "Apache License" in licence and "Version 2.0, January 2004" in licence
