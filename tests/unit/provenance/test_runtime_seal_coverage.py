# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The runtime seal covers the repository data a moved-on rerender reads.

A moved-on rerender of an official run is accepted only when the checkout
still holds every file the launcher sealed. These tests seal a copy of this
checkout, edit the copy the way a later commit would, and ask the shared rule
(``robosuite_env.contracts.provenance``) whether the copy may stand in for the
run's code.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from real2sim import runtime_manifest
from robosuite_env.contracts import provenance as provenance_contract
from tests._helpers.repo import REPO_ROOT
from tests._helpers.sealed_checkout import (
    copy_covered_files,
    gaussian_wrapping,
    private_copy,
    renderer_code,
    replace as _replace,
    run_manifest,
    seal as _seal,
)


#: A generation commit the checkout has since moved past.
_GENERATION_COMMIT = "b" * 40
_RENDERER_COMMIT = "c" * 40

#: Repository data the replay reads from the checkout: the box MJCF, a gripper
#: mesh the cup profiles load, a robot control config, an object layout and
#: an initial pose.
DATA_FILES = (
    "robosuite_env/assets/box.xml",
    "robosuite_env/assets/yubi_right/visual_mj/dyn_l_finger_r__geometry0.obj",
    "robosuite_env/robots/yubi/control/configs/spec_nominal.yaml",
    "robosuite_env/obj_calib/layout_box_pack_duo_manual.json",
    "assets/initial_poses/fr3/flat/grip07.json",
)


@pytest.fixture(scope="module")
def pristine(tmp_path_factory) -> Path:
    """One copy of every file the seal covers, made once per module."""

    return copy_covered_files(tmp_path_factory.mktemp("pristine") / "checkout")


@pytest.fixture
def checkout(pristine: Path, tmp_path: Path) -> Path:
    return private_copy(pristine, tmp_path / "checkout")


def _check(checkout: Path, run_root: Path, *, commit: str = _RENDERER_COMMIT):
    return provenance_contract.official_generation_check(
        run_root,
        run_manifest({"commit": _GENERATION_COMMIT, "dirty": False, "path": str(checkout)}),
        renderer_code(checkout, commit=commit, dirty=commit != _GENERATION_COMMIT),
        checkout_root=checkout,
    )


def _matches(checkout: Path, run_root: Path) -> bool:
    return _check(checkout, run_root) is not None


def test_an_unchanged_checkout_matches_its_seal(checkout, tmp_path):
    _seal(checkout, tmp_path / "run")

    record = _check(checkout, tmp_path / "run")
    assert record is not None
    assert record["rule"] == provenance_contract.RULE_SEALED_SOURCES
    assert set(DATA_FILES) <= set(record["files"])


def test_the_seal_hashes_the_replay_data_files(checkout, tmp_path):
    _seal(checkout, tmp_path / "run")
    payload = json.loads(
        (tmp_path / "run" / provenance_contract.RUNTIME_SOURCE_MANIFEST_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    sealed = {Path(record["path"]).relative_to(checkout).as_posix() for record in payload["files"]}

    assert set(DATA_FILES) <= sealed


@pytest.mark.parametrize("relative", DATA_FILES)
def test_an_edited_data_file_refuses_a_moved_on_rerender(checkout, tmp_path, relative):
    _seal(checkout, tmp_path / "run")
    path = checkout / relative
    _replace(path, path.read_bytes() + b"\n")

    assert not _matches(checkout, tmp_path / "run")


def test_a_data_file_added_under_a_new_suffix_refuses_a_moved_on_rerender(checkout, tmp_path):
    _seal(checkout, tmp_path / "run")
    (checkout / "robosuite_env/assets/box_slab.stl").write_bytes(b"solid slab\n")

    assert not _matches(checkout, tmp_path / "run")


def test_bytecode_caches_in_a_data_tree_are_not_sealed(checkout, tmp_path):
    cache = checkout / "robosuite_env/robots/__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "stale.cpython-311.pyc").write_bytes(b"old")
    _seal(checkout, tmp_path / "run")
    (cache / "new.cpython-311.pyc").write_bytes(b"new")

    assert _matches(checkout, tmp_path / "run")


def _without_box_slab(text: str) -> str:
    """box.xml as it was before the box_023 base-slab collision geom returned."""

    lines = text.splitlines(keepends=True)
    kept = [line for line in lines if "p_coll_23" not in line and '"p_c23"' not in line]
    assert len(lines) - len(kept) == 2
    return "".join(kept)


def test_a_run_sealed_before_the_box_slab_returned_refuses_a_rerender_after_it(
    checkout, tmp_path
):
    box = checkout / "robosuite_env/assets/box.xml"
    current = box.read_bytes()
    _replace(box, _without_box_slab(current.decode("utf-8")).encode("utf-8"))
    _seal(checkout, tmp_path / "run")
    _replace(box, current)

    assert not _matches(checkout, tmp_path / "run")


def _source_only_selection(checkout: Path) -> dict:
    """The selection a launcher wrote before the data trees joined the seal."""

    selection = runtime_manifest.default_selection(checkout)
    selection["trees"] = [
        {"path": "robosuite_env/contracts", "suffixes": [".py"]},
        {"path": "robosuite_env/controllers", "suffixes": [".json"]},
        {"path": "robosuite_env/policy", "suffixes": [".py"]},
        {"path": "robosuite_env/rendering", "suffixes": [".py"]},
        {"path": "robosuite_env/rerender", "suffixes": [".py"]},
        {"path": "robosuite_env/robots", "suffixes": [".py"]},
        {"path": "robosuite_env/rollout", "suffixes": [".py"]},
        {"path": "robosuite_env/tasks", "suffixes": [".py"]},
        {"path": "scene", "suffixes": [".py"]},
    ]
    return selection


def test_a_seal_without_the_data_trees_allows_only_the_run_commit(
    checkout, tmp_path, monkeypatch
):
    run_root = tmp_path / "run"
    _seal(checkout, run_root, _source_only_selection(checkout))
    # The copy is not a Git checkout; let the run's commit track every file in
    # it, so the commit rule has no ignored sealed file left to compare.
    tracked = frozenset(
        path.relative_to(checkout).as_posix() for path in checkout.rglob("*") if path.is_file()
    )
    monkeypatch.setattr(
        provenance_contract, "_paths_in_commit", lambda _root, _commit: tracked
    )

    # Every file it sealed still matches, but it never hashed the assets.
    assert not _matches(checkout, run_root)
    # The run's own clean commit still rerenders it.
    record = _check(checkout, run_root, commit=_GENERATION_COMMIT)
    assert record is not None
    assert record["rule"] == provenance_contract.RULE_GENERATION_COMMIT


@pytest.mark.parametrize(
    ("coverage", "covered"),
    [
        ({"files": [], "trees": [{"path": "a", "suffixes": [".py"]}]}, True),
        ({"files": [], "trees": [{"path": "b", "suffixes": [".json"]}]}, True),
        ({"files": [], "trees": [{"path": "a", "suffixes": [".json"]}]}, False),
        ({"files": [], "trees": [{"path": "b", "suffixes": ["*"]}]}, True),
        ({"files": [], "trees": [{"path": "a", "suffixes": ["*"]}]}, False),
        ({"files": [], "trees": [{"path": "c", "suffixes": [".py"]}]}, False),
        ({"files": ["x.py"], "trees": []}, True),
        ({"files": ["y.py"], "trees": []}, False),
    ],
)
def test_selection_coverage(coverage, covered):
    selection = {
        "files": ["x.py"],
        "trees": [
            {"path": "a", "suffixes": [".py"]},
            {"path": "b", "suffixes": ["*"]},
        ],
    }

    assert runtime_manifest.selection_covers(selection, coverage) is covered


@pytest.mark.parametrize("suffixes", [["*", ".py"], ["**"], ["py"]])
def test_a_tree_selects_every_file_only_as_the_sole_suffix(tmp_path, suffixes):
    (tmp_path / "tree").mkdir()
    selection = {
        "repo_root": str(tmp_path.resolve()),
        "files": [],
        "external_files": [],
        "trees": [{"path": "tree", "suffixes": suffixes}],
    }

    with pytest.raises(runtime_manifest.RuntimeManifestError, match="suffixes"):
        runtime_manifest.capture_selection(selection)


# -- data-tree membership in a Git checkout ---------------------------------

_DATA_TREES = [
    {"path": "assets", "suffixes": [runtime_manifest.ALL_FILES]},
    {"path": "robosuite_env/assets", "suffixes": [runtime_manifest.ALL_FILES]},
]
#: A collision mesh the artifacts manifest declares as installed in the
#: checkout; the repository ignores it because an archive installs it.
_DECLARED_MESH = "robosuite_env/assets/fr3_collision/link0.stl"


def _git(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _git_checkout(root: Path) -> Path:
    """A committed checkout whose data trees also hold ignored local content."""

    _write(
        root / ".gitignore",
        b"assets/runtime/\n"
        b"robosuite_env/assets/fr3_collision\n"
        b"robosuite_env/assets/gs_backgrounds/**/point_cloud.ply\n",
    )
    shutil.copyfile(REPO_ROOT / "artifacts/manifest.yaml", _write(root / "artifacts/manifest.yaml", b""))
    _write(root / "assets/initpose/fr3/flat/pose.json", b"{}\n")
    _write(root / "robosuite_env/assets/box.xml", b"<mujoco/>\n")
    _write(root / "robosuite_env/assets/gs_backgrounds/scene/manifest.json", b"{}\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fixture")
    _write(root / _DECLARED_MESH, b"solid link0\n")
    # Ignored local content: run outputs, a scene point cloud, a nested
    # repository and a link out of the checkout.
    _write(root / "assets/runtime/run_01/frames.bin", b"\0" * 4096)
    _write(root / "assets/runtime/clone/.git/config", b"[core]\n")
    (root / "assets/runtime/linked").symlink_to("/")
    _write(root / "robosuite_env/assets/gs_backgrounds/scene/point_cloud.ply", b"ply\n")
    return root


def _data_selection(checkout: Path) -> dict:
    return {
        "repo_root": str(checkout.resolve()),
        "files": [],
        "external_files": [],
        "trees": _DATA_TREES,
    }


def _sealed_relative(capture: dict, checkout: Path) -> set[str]:
    return {
        Path(record["path"]).relative_to(checkout.resolve()).as_posix()
        for record in capture["files"]
    }


def test_a_data_tree_seals_tracked_and_declared_archive_files_only(tmp_path):
    checkout = _git_checkout(tmp_path / "checkout")

    capture = runtime_manifest.capture_selection(_data_selection(checkout))

    assert _sealed_relative(capture, checkout) == {
        "assets/initpose/fr3/flat/pose.json",
        "robosuite_env/assets/box.xml",
        "robosuite_env/assets/gs_backgrounds/scene/manifest.json",
        _DECLARED_MESH,
    }
    runtime_manifest.verify_capture(capture)


def test_an_untracked_file_git_does_not_ignore_is_sealed(tmp_path):
    checkout = _git_checkout(tmp_path / "checkout")
    _write(checkout / "robosuite_env/assets/new_mesh.obj", b"v 0 0 0\n")

    capture = runtime_manifest.capture_selection(_data_selection(checkout))

    assert "robosuite_env/assets/new_mesh.obj" in _sealed_relative(capture, checkout)


def test_ignored_local_content_does_not_refuse_a_moved_on_check(tmp_path):
    checkout = _git_checkout(tmp_path / "checkout")
    capture = runtime_manifest.capture_selection(_data_selection(checkout))
    _write(checkout / "assets/runtime/run_02/frames.bin", b"\1" * 4096)

    assert runtime_manifest.tree_members(checkout, _DATA_TREES) == sorted(
        _sealed_relative(capture, checkout)
    )


def test_a_tracked_symlink_in_a_data_tree_fails_with_what_to_do(tmp_path):
    checkout = _git_checkout(tmp_path / "checkout")
    (checkout / "robosuite_env/assets/linked.xml").symlink_to("box.xml")
    _git(checkout, "add", "robosuite_env/assets/linked.xml")

    with pytest.raises(runtime_manifest.RuntimeManifestError, match="Replace it with"):
        runtime_manifest.capture_selection(_data_selection(checkout))


def test_a_linked_archive_install_directory_fails_with_what_to_do(tmp_path):
    checkout = _git_checkout(tmp_path / "checkout")
    installed = checkout / "robosuite_env/assets/fr3_collision"
    shutil.move(installed, tmp_path / "shared_fr3_collision")
    installed.symlink_to(tmp_path / "shared_fr3_collision")

    with pytest.raises(
        runtime_manifest.RuntimeManifestError, match="real2sim-download-release"
    ):
        runtime_manifest.capture_selection(_data_selection(checkout))


def _commit_rule_check(checkout: Path, run_root: Path):
    head = _git(checkout, "rev-parse", "HEAD")
    manifest_code = run_manifest(
        {"commit": head, "dirty": False, "path": str(checkout.resolve())}
    )
    code = {
        "commit": head,
        "dirty": False,
        "gaussian_wrapping": gaussian_wrapping(checkout.resolve()),
    }
    record = provenance_contract.official_generation_check(
        run_root, manifest_code, code, checkout_root=checkout
    )
    return record, manifest_code, code


def test_the_run_commit_rule_compares_the_sealed_files_git_ignores(tmp_path):
    checkout = _git_checkout(tmp_path / "checkout")
    _seal(checkout, tmp_path / "run", _data_selection(checkout))

    record, manifest_code, code = _commit_rule_check(checkout, tmp_path / "run")

    assert record is not None
    assert record["rule"] == provenance_contract.RULE_GENERATION_COMMIT
    assert set(record["files"]) == {_DECLARED_MESH}
    assert (
        provenance_contract.official_generation_check_errors(
            tmp_path / "run", manifest_code, code, {**record, "unchanged_at_write": True}
        )
        == []
    )


def test_an_edited_ignored_mesh_at_the_run_commit_is_refused(tmp_path):
    checkout = _git_checkout(tmp_path / "checkout")
    _seal(checkout, tmp_path / "run", _data_selection(checkout))
    mesh = checkout / _DECLARED_MESH
    _replace(mesh, mesh.read_bytes() + b"endsolid\n")
    assert _git(checkout, "status", "--porcelain") == ""

    record, _manifest_code, _code = _commit_rule_check(checkout, tmp_path / "run")

    assert record is None


def test_completion_refuses_a_commit_rule_record_that_skipped_an_ignored_file(tmp_path):
    checkout = _git_checkout(tmp_path / "checkout")
    _seal(checkout, tmp_path / "run", _data_selection(checkout))
    record, manifest_code, code = _commit_rule_check(checkout, tmp_path / "run")
    assert record is not None

    errors = provenance_contract.official_generation_check_errors(
        tmp_path / "run",
        manifest_code,
        code,
        {**record, "files": {}, "unchanged_at_write": True},
    )

    assert any("outside the run's commit" in error for error in errors), errors


# -- the GaussianWrapping checkout that renders the 3DGS background ----------


def _gaussian_wrapping_checkout(checkout: Path) -> tuple[Path, str, str]:
    """A GaussianWrapping checkout at the gitlink with two commits; at the first."""

    gw = checkout / "third_party/GaussianWrapping"
    _write(gw / "gaussian_wrapping/gaussian_renderer/ours.py", b"debug = False\n")
    _git(gw, "init", "-q")
    _git(gw, "add", "-A")
    _git(gw, "commit", "-q", "-m", "launch")
    launch = _git(gw, "rev-parse", "HEAD")
    _write(gw / "gaussian_wrapping/gaussian_renderer/ours.py", b"debug = True\n")
    _git(gw, "commit", "-q", "-am", "bump")
    bumped = _git(gw, "rev-parse", "HEAD")
    _git(gw, "checkout", "-q", launch)
    return gw, launch, bumped


def _gw_check(checkout: Path, run_root: Path, launch: str, *, commit: str):
    checkout = checkout.resolve()
    manifest = run_manifest(
        {"commit": _GENERATION_COMMIT, "dirty": False, "path": str(checkout)},
        commit=launch,
    )
    code = renderer_code(checkout, commit=commit, dirty=commit != _GENERATION_COMMIT)
    code["gaussian_wrapping"] = provenance_contract.gaussian_wrapping_identity(checkout)
    record = provenance_contract.official_generation_check(
        run_root, manifest, code, checkout_root=checkout
    )
    return record, manifest, code


@pytest.mark.parametrize("commit", [_RENDERER_COMMIT, _GENERATION_COMMIT])
def test_the_run_gaussian_wrapping_commit_is_accepted(checkout, tmp_path, commit):
    _gw, launch, _bumped = _gaussian_wrapping_checkout(checkout)
    run_root = tmp_path / "run"
    if commit == _RENDERER_COMMIT:
        _seal(checkout, run_root)
    else:
        run_root.mkdir()

    record, _manifest, _code = _gw_check(checkout, run_root, launch, commit=commit)

    assert record is not None


@pytest.mark.parametrize("commit", [_RENDERER_COMMIT, _GENERATION_COMMIT])
def test_a_gaussian_wrapping_gitlink_bump_is_refused(checkout, tmp_path, commit):
    gw, launch, bumped = _gaussian_wrapping_checkout(checkout)
    run_root = tmp_path / "run"
    if commit == _RENDERER_COMMIT:
        _seal(checkout, run_root)
    else:
        run_root.mkdir()
    # A later commit bumps the gitlink and the checkout follows it.
    _git(gw, "checkout", "-q", bumped)

    record, manifest, code = _gw_check(checkout, run_root, launch, commit=commit)

    assert record is None
    accepted, _manifest, _code = _gw_check(checkout, run_root, bumped, commit=commit)
    assert accepted is not None, "only the GaussianWrapping commit differs"
    errors = provenance_contract.official_generation_check_errors(
        run_root, manifest, code, {**accepted, "unchanged_at_write": True}
    )
    assert any("GaussianWrapping checkout is at" in error for error in errors), errors


@pytest.mark.parametrize("commit", [_RENDERER_COMMIT, _GENERATION_COMMIT])
def test_a_locally_edited_gaussian_wrapping_file_is_refused(checkout, tmp_path, commit):
    gw, launch, _bumped = _gaussian_wrapping_checkout(checkout)
    run_root = tmp_path / "run"
    if commit == _RENDERER_COMMIT:
        _seal(checkout, run_root)
    else:
        run_root.mkdir()
    _write(gw / "gaussian_wrapping/gaussian_renderer/ours.py", b"debug = True\n")

    record, _manifest, code = _gw_check(checkout, run_root, launch, commit=commit)

    assert code["gaussian_wrapping"]["commit"] == launch
    assert code["gaussian_wrapping"]["dirty"] is True
    assert record is None
