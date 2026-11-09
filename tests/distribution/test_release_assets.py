# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Profile selection, pinned downloads and safe installation of release inputs."""

import json
from pathlib import Path
import shutil
import subprocess
import tomllib
import xml.etree.ElementTree as ET

import pytest
import yaml

from real2sim import release_assets

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def installer():
    return release_assets


def test_installer_is_the_declared_console_script():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entry = pyproject["project"]["scripts"]["real2sim-download-release"]
    assert entry == "real2sim.release_assets:cli"
    module_name, function = entry.split(":")
    assert release_assets.__name__ == module_name
    assert callable(getattr(release_assets, function))
    assert not (ROOT / "scripts" / "download_release_assets.py").exists()


def test_default_repo_root_prefers_the_editable_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = Path(release_assets.__file__).resolve().parents[2]
    if not (expected / release_assets.RELEASE_CATALOG).is_file():
        pytest.skip("real2sim is not installed in editable mode from a checkout")
    assert release_assets.default_repo_root() == expected


def test_default_repo_root_falls_back_to_the_working_directory(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    (checkout / "artifacts").mkdir(parents=True)
    (checkout / release_assets.RELEASE_CATALOG).write_text("{}")
    (checkout / "config").mkdir()
    monkeypatch.setattr(release_assets, "__file__", str(tmp_path / "site/real2sim/release_assets.py"))
    monkeypatch.chdir(checkout / "config")
    assert release_assets.default_repo_root() == checkout.resolve()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(release_assets.ReleaseAssetError, match="pass --repo-root"):
        release_assets.default_repo_root()


def test_all_profiles_select_their_declared_scene_and_checkpoint(installer):
    catalog = json.loads((ROOT / "artifacts/release_downloads.json").read_text())
    manifest = yaml.safe_load((ROOT / "artifacts/manifest.yaml").read_text())
    by_id = {a["id"]: a for a in manifest["artifacts"]}
    profiles = sorted((ROOT / "config/release").glob("*/*/*.yaml"))
    assert len(profiles) == 2
    for profile in profiles:
        data = yaml.safe_load(profile.read_text())
        selected = installer.select_archives([profile], ROOT, catalog)
        scene = data["scenes"][0]
        scene_name = Path(scene["path"]).name
        checkpoint = Path(data["policy"]["checkpoint"]).name
        assert scene_name in selected
        assert checkpoint in selected
        assert "fr3_collision" in selected
        for name, role in [(checkpoint, "checkpoint"), (scene_name, "scene." + scene["tag"])]:
            artifact = by_id[data["reproducibility"]["artifacts"][role]]
            pin = catalog["archives"][name]
            assert pin["destination"] == artifact["local_path"]
            assert pin["tree"] == {"size": artifact["size"], "sha256": artifact["sha256"]}
    selected = installer.select_archives(profiles, ROOT, catalog)
    assert len(selected) == len(set(selected)) == 5
    # The catalog pins only what a release profile installs.
    assert set(selected) == set(catalog["archives"])
    for profile in profiles:
        rig, room = profile.parent.name, profile.stem.split("_room_")[0] + "_room"
        scenes = [name for name in installer.select_archives([profile], ROOT, catalog) if name.endswith("_room")]
        assert scenes == [f"{rig}_{room}"]


def test_catalog_bundle_mapping_covers_every_untracked_mjcf_payload():
    catalog = json.loads((ROOT / "artifacts/release_downloads.json").read_text())
    asset_root = ROOT / "robosuite_env/assets"
    manifest = json.loads((asset_root / "manifest.json").read_text())
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "--", "robosuite_env/assets"], cwd=ROOT, text=True,
    ).splitlines())
    for bundle, archives in catalog["runtime_bundles"].items():
        xml_path = asset_root / manifest["bundles"][bundle]["mjcf"]
        for element in ET.parse(xml_path).getroot().iter():
            relative = element.get("file")
            if relative is None:
                continue
            path = (xml_path.parent / relative).relative_to(ROOT).as_posix()
            if path in tracked:
                continue
            assert any(
                ("{repo_root}/" + path).startswith(
                    catalog["archives"][archive]["destination"] + "/"
                )
                for archive in archives
            ), (bundle, relative)


@pytest.mark.parametrize("robot", ["fr3", "fr3_duo"])
@pytest.mark.parametrize("objects", ["cup_plate", "box_pack", " BOX_PACK "])
def test_installer_selects_the_runtime_bundles(robot, objects, tmp_path):
    from real2sim.artifacts_binding import _schema_v2_asset_bundle_ids

    catalog = json.loads((ROOT / "artifacts/release_downloads.json").read_text())
    data = yaml.safe_load((ROOT / "config/release/fr3/flat/reading_room_cup_cup1trim.yaml").read_text())
    data.update(robot=robot, task={"objects": objects})
    # Give every runtime bundle a distinct synthetic archive so omitted or
    # extra bundles remain observable even when release bundles share a tree.
    bundles = _schema_v2_asset_bundle_ids({"resolved": data})
    catalog["runtime_bundles"] = {bundle: ["payload_" + bundle] for bundle in bundles}
    for bundle in bundles:
        catalog["archives"]["payload_" + bundle] = {"tree": {"size": 1}}
    profile = tmp_path / "config/release" / robot / "flat/fixture.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text(yaml.safe_dump(data))
    selected = release_assets.select_archives([profile], tmp_path, catalog)
    assert set(selected) == {
        "flat_reading_room", "pi05_cup", *("payload_" + bundle for bundle in bundles),
    }


def test_omitted_robot_matches_schema_default_and_installer_cli(tmp_path, monkeypatch, capsys):
    from real2sim.artifacts_binding import _schema_v2_asset_bundle_ids
    from real2sim.config.schema import load_rollout_config

    source = ROOT / "config/release/fr3/flat/reading_room_cup_cup1trim.yaml"
    data = yaml.safe_load(source.read_text())
    del data["robot"]
    profile = tmp_path / "config/release/fr3/flat/no_robot.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text(yaml.safe_dump(data))
    shutil.copyfile(ROOT / "config/support-matrix.yaml", tmp_path / "config/support-matrix.yaml")
    catalog = json.loads((ROOT / "artifacts/release_downloads.json").read_text())
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/release_downloads.json").write_text(json.dumps(catalog))
    inputs = tmp_path / "inputs"
    monkeypatch.setenv("REAL2SIM_INPUTS_ROOT", str(inputs))

    # Resolve the real profile through the runtime schema, not a copied
    # default, then compare the installer with that runtime's bundle set.
    plan = load_rollout_config(profile, repo_root=tmp_path, timestamp="20260822_000000")
    assert plan["resolved"]["robot"] == "fr3"
    expected = {"flat_reading_room", "pi05_cup"}
    for bundle in _schema_v2_asset_bundle_ids(plan):
        expected.update(catalog["runtime_bundles"][bundle])
    selected = release_assets.select_archives([profile], tmp_path, catalog)
    assert set(selected) == expected
    assert selected == release_assets.select_archives([source], ROOT, catalog)

    assert release_assets.cli([
        "--repo-root", str(tmp_path), "--profile", str(profile),
        "--inputs-root", str(inputs), "--dry-run",
    ]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert {
        line.strip().split(":", 1)[0]
        for line in output.out.splitlines() if " bytes -> " in line
    } == expected
    assert not inputs.exists()
    assert not (tmp_path / ".env").exists()


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("failure", ["unpublished_objects", "missing_archive", "missing_mapping"])
def test_unavailable_bundles_fail_before_downloads(tmp_path, monkeypatch, capsys, dry_run, failure):
    catalog = json.loads((ROOT / "artifacts/release_downloads.json").read_text())
    data = yaml.safe_load((ROOT / "config/release/fr3/flat/reading_room_cup_cup1trim.yaml").read_text())
    if failure == "unpublished_objects":
        data["task"]["objects"] = "box_pack"
        expected = "no pinned downloads for runtime bundles: box, tape, pen, lan"
    elif failure == "missing_archive":
        del catalog["archives"]["trellis_cup_plate"]
        expected = "invalid pinned downloads for runtime bundle 'cup'"
    else:
        del catalog["runtime_bundles"]
        expected = "no pinned downloads for runtime bundles: yubi_right, yubi_left, fr3, cup, plate"
    profile = tmp_path / "config/release/fr3/flat/fixture.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text(yaml.safe_dump(data))
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/release_downloads.json").write_text(json.dumps(catalog))
    monkeypatch.setattr(release_assets, "_hub_snapshot", lambda **kw: pytest.fail("download attempted"))
    root = tmp_path / "inputs"
    args = ["--repo-root", str(tmp_path), "--profile", str(profile), "--inputs-root", str(root)]
    assert release_assets.cli(args + (["--dry-run"] if dry_run else [])) == 1
    assert expected in capsys.readouterr().err
    assert not root.exists()
    assert not (tmp_path / ".env").exists()


def test_a_profile_cannot_select_another_rigs_scene(installer, tmp_path):
    catalog = json.loads((ROOT / "artifacts/release_downloads.json").read_text())
    source = ROOT / "config/release/fr3/flat/reading_room_cup_cup1trim.yaml"
    data = yaml.safe_load(source.read_text())
    # A scene pinned for another rig is still refused; the release catalog
    # pins only Flat scenes, so pin one here.
    catalog["archives"]["duo_reading_room"] = {
        "destination": "{inputs_root}/scenes/duo_reading_room",
        "tree": {"size": 1, "sha256": "0" * 64},
    }
    data["scenes"][0]["path"] = "{inputs_root}/scenes/duo_reading_room"
    profile = tmp_path / "config/release/fr3/flat/wrong_rig_scene.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text(yaml.safe_dump(data))
    with pytest.raises(installer.ReleaseAssetError, match="no pinned flat scene download"):
        installer.select_archives([profile], tmp_path, catalog)
    data["scenes"][0]["path"] = "{inputs_root}/scenes/flat_reading_room"
    profile.write_text(yaml.safe_dump(data))
    assert "flat_reading_room" in installer.select_archives([profile], tmp_path, catalog)


def test_dry_run_dispatch_does_not_create_input_root(installer, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("REAL2SIM_INPUTS_ROOT", raising=False)
    root = tmp_path / "inputs"
    assert installer.cli([
        "--profile", "config/release/fr3/flat/reading_room_cup_cup1trim.yaml",
        "--inputs-root", str(root), "--env-file", str(tmp_path / ".env"), "--dry-run",
    ]) == 0
    assert not root.exists()
    assert "pi05_cup" in capsys.readouterr().out


REVISION = "1" * 40
CHECKPOINTS = "{inputs_root}/checkpoints/"


def make_pin(installer, tmp_path):
    """One checkpoint folder and its catalog pin."""
    source = tmp_path / "fixture"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"complete checkpoint\n")
    digest = installer.sha256_tree(source)
    return source, {"destination": CHECKPOINTS + "fixture",
                    "tree": {"size": digest.size, "sha256": digest.sha256}}


def repos(installer, revision=REVISION):
    return {"policies": installer.HubRepo("policies", "owner/name", "model", revision,
                                          ((CHECKPOINTS, ""),))}


def serve(installer, monkeypatch, source):
    """Serve ``source`` as the repository folder that every download requests."""
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        (folder,) = kwargs["allow_patterns"]
        shutil.copytree(source, Path(kwargs["local_dir"]) / folder.removesuffix("/*"),
                        dirs_exist_ok=True)
        return kwargs["local_dir"]

    monkeypatch.setattr(installer, "_hub_snapshot", snapshot)
    return calls


def test_install_verifies_bytes_reuses_and_refuses_different_data(installer, tmp_path, monkeypatch):
    source, pin = make_pin(installer, tmp_path)
    calls = serve(installer, monkeypatch, source)
    target = tmp_path / "installed/fixture"
    kwargs = dict(timeout=1, repos=repos(installer))
    installer.install("fixture", pin, target, tmp_path / "cache", **kwargs)
    assert (target / "weights.bin").read_bytes() == b"complete checkpoint\n"
    assert len(calls) == 1
    monkeypatch.setattr(installer, "_hub_snapshot", lambda **kw: pytest.fail("redownload"))
    installer.install("fixture", pin, target, tmp_path / "cache", **kwargs)
    serve(installer, monkeypatch, source)
    (target / "weights.bin").write_bytes(b"local changes")
    with pytest.raises(installer.ReleaseAssetError, match="not byte-identical"):
        installer.install("fixture", pin, target, tmp_path / "cache", **kwargs)
    assert (target / "weights.bin").read_bytes() == b"local changes"


def test_partial_install_can_be_completed_without_overwrite(installer, tmp_path, monkeypatch):
    source, pin = make_pin(installer, tmp_path)
    serve(installer, monkeypatch, source)
    target = tmp_path / "installed/fixture"
    target.mkdir(parents=True)
    installer.install("fixture", pin, target, tmp_path / "cache",
                      timeout=1, repos=repos(installer))
    assert installer.matches_tree(target, pin["tree"])


def test_unpinned_repository_is_never_requested(installer, tmp_path, monkeypatch):
    _, pin = make_pin(installer, tmp_path)
    monkeypatch.setattr(installer, "_hub_snapshot", lambda **kw: pytest.fail("requested"))
    with pytest.raises(installer.ReleaseAssetError, match="has no pinned revision"):
        installer.install("fixture", pin, tmp_path / "installed/fixture", tmp_path / "cache",
                          timeout=1, repos=repos(installer, "UNSET_BEFORE_PUBLICATION"))


def test_bad_digest_never_installs(installer, tmp_path, monkeypatch):
    source, pin = make_pin(installer, tmp_path)
    serve(installer, monkeypatch, source)
    pin["tree"]["sha256"] = "0" * 64
    target = tmp_path / "installed/fixture"
    with pytest.raises(installer.ReleaseAssetError, match="do not match the pinned tree"):
        installer.install("fixture", pin, target, tmp_path / "cache",
                          timeout=1, repos=repos(installer))
    assert not target.exists()


def test_the_installer_has_no_offline_archive_option(installer, tmp_path, capsys):
    # The release inputs are published only as folders on Hugging Face; no tar
    # archive exists for an offline install to read.
    with pytest.raises(SystemExit) as raised:
        installer.main(["--profile", "config/release/fr3/flat/a_room_cup_cup1trim.yaml",
                        "--archive-dir", str(tmp_path), "--dry-run"])
    assert raised.value.code == 2
    assert "unrecognized arguments: --archive-dir" in capsys.readouterr().err


def test_symlinked_parent_is_refused(installer, tmp_path):
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(installer.ReleaseAssetError, match="symlinks"):
        installer.real_path(tmp_path / "link/scene")


def test_inputs_root_comes_from_the_environment_without_the_option(installer, tmp_path, monkeypatch, capsys):
    root = tmp_path / "environment-inputs"
    monkeypatch.setenv("REAL2SIM_INPUTS_ROOT", str(root))
    profile = sorted((ROOT / "config/release").glob("*/*/*.yaml"))[0]
    assert installer.main([
        "--profile", str(profile), "--env-file", str(tmp_path / ".env"), "--dry-run",
    ]) == 0
    assert f"-> {root}/checkpoints/" in capsys.readouterr().out
    assert not root.exists()
