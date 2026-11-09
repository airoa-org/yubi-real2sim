# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Exercise source bootstrapping with real local repositories, without downloads."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(path, *args):
    return subprocess.check_output(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "-c", "filter.lfs.process=", "-c", "filter.lfs.clean=cat",
         "-c", "filter.lfs.smudge=cat", "-c", "filter.lfs.required=false",
         "-c", "protocol.file.allow=always", "-C", str(path), *args],
        text=True, stderr=subprocess.PIPE,
    ).strip()


def _repository(path):
    path.mkdir()
    _git(path, "init", "-q")
    (path / "README").write_text("fixture\n")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "fixture")
    return path


def test_bootstrap_initializes_gaussian_wrapping_without_its_nested_submodules(tmp_path):
    # Nothing builds or imports GaussianWrapping's nested submodules
    # (THIRD_PARTY_NOTICES B.1), so bootstrap does not fetch them, and
    # production provenance records them without requiring them.
    from real2sim import provenance

    nested = _repository(tmp_path / "nested")
    sources = {}
    for name in ("GaussianWrapping", "franka_description"):
        path = _repository(tmp_path / name)
        if name == "GaussianWrapping":
            _git(path, "submodule", "add", str(nested), "submodules/nvdiffrast")
            _git(path, "commit", "-qam", "nested dependency")
        if name == "franka_description":
            (path / ".gitattributes").write_text("*.media filter=lfs\n")
            (path / "example.media").write_text("source-only LFS fixture\n")
            _git(path, "add", ".")
            _git(path, "commit", "-qm", "LFS fixture")
        sources[name] = path
    template = _repository(tmp_path / "template")
    shutil.copytree(REPO_ROOT / "scripts", template / "scripts")
    shutil.copytree(REPO_ROOT / "src", template / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (template / "config").mkdir()
    shutil.copy(REPO_ROOT / "config/support-matrix.yaml", template / "config")
    shutil.copy(REPO_ROOT / ".python-version", template)
    _git(template, "submodule", "add", str(sources["GaussianWrapping"]), "third_party/GaussianWrapping")
    _git(template, "add", ".")
    _git(template, "commit", "-qm", "bootstrap fixture")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", str(template), str(clone))
    original_head = _git(clone, "rev-parse", "HEAD")
    assert not (clone / "third_party/GaussianWrapping/.git").exists()
    env = {
        **os.environ, "GIT_ALLOW_PROTOCOL": "file", "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "filter.lfs.process",
        "GIT_CONFIG_VALUE_0": "missing-git-lfs-for-bootstrap-test",
    }
    for name, prefix in (
        ("franka_description", "FRANKA_DESCRIPTION"),
    ):
        env[f"{prefix}_URL"] = str(sources[name])
        env[f"{prefix}_REF"] = _git(sources[name], "rev-parse", "HEAD")
    for _ in range(2):
        result = subprocess.run(
            [str(clone / "scripts/bootstrap.sh"), "--skip-python"],
            cwd=clone, env=env, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert _git(clone, "rev-parse", "HEAD") == original_head
        for name in sources:
            path = clone / "third_party" / name
            assert _git(path, "rev-parse", "HEAD") == _git(sources[name], "rev-parse", "HEAD")
        gaussian_wrapping = clone / "third_party/GaussianWrapping"
        assert subprocess.check_output(
            ["git", "-C", str(gaussian_wrapping), "submodule", "status", "--recursive"],
            text=True,
        ).startswith("-")
        assert not any((gaussian_wrapping / "submodules/nvdiffrast").iterdir())
    assert _git(clone, "status", "--porcelain", "--untracked-files=no") == ""
    nested_commit = _git(nested, "rev-parse", "HEAD")
    for root, path in (
        (clone, "third_party/GaussianWrapping/submodules/nvdiffrast"),
        (gaussian_wrapping, "submodules/nvdiffrast"),
    ):
        records = provenance.collect_git_submodules(root, production=True)
        assert {"path": path, "commit": nested_commit, "status": "missing"} in records
    # OpenPI and robosuite run from their locked, installed distributions; no
    # checkout exists for an import to pick up instead.
    assert not (clone / "third_party/openpi").exists()
    assert not (clone / "third_party/robosuite").exists()


def test_bootstrap_rejects_a_directory_that_only_belongs_to_the_parent_repo(configured_sources):
    root, env, _ = configured_sources
    # A plain directory where bootstrap expects its own clone resolves to the
    # parent repository; bootstrap must refuse it rather than reuse that HEAD.
    (root / "third_party/franka_description").mkdir(parents=True)
    original_head = _git(root, "rev-parse", "HEAD")
    result = _bootstrap(root, env)
    assert result.returncode == 1
    assert "exists but is not a Git clone" in result.stderr
    assert _git(root, "rev-parse", "HEAD") == original_head


@pytest.fixture
def configured_sources(tmp_path):
    root = _repository(tmp_path / "root")
    shutil.copytree(REPO_ROOT / "scripts", root / "scripts")
    shutil.copytree(REPO_ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "config").mkdir()
    shutil.copy(REPO_ROOT / "config/support-matrix.yaml", root / "config")
    shutil.copy(REPO_ROOT / ".python-version", root)
    env = {**os.environ, "GIT_ALLOW_PROTOCOL": "file"}
    sources = {}
    for name, prefix in (
        ("franka_description", "FRANKA_DESCRIPTION"),
    ):
        sources[name] = _repository(tmp_path / name)
        env[f"{prefix}_URL"] = str(sources[name])
        env[f"{prefix}_REF"] = _git(sources[name], "rev-parse", "HEAD")
    sources["GaussianWrapping"] = _repository(tmp_path / "GaussianWrapping")
    (sources["GaussianWrapping"] / "README").write_text("pinned\n")
    _git(sources["GaussianWrapping"], "commit", "-qam", "pinned")
    _git(root, "submodule", "add", str(sources["GaussianWrapping"]), "third_party/GaussianWrapping")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "bootstrap fixture")
    _git(root, "submodule", "deinit", "-q", "--all")
    shutil.rmtree(root / ".git/modules")
    return root, env, sources


def _bootstrap(root, env, *args):
    return subprocess.run(
        [str(root / "scripts/bootstrap.sh"), "--skip-python", *args],
        cwd=root, env=env, capture_output=True, text=True,
    )


@pytest.mark.parametrize("update", [False, True])
def test_bootstrap_fetches_selected_source_only_with_update(configured_sources, tmp_path, update):
    root, env, sources = configured_sources
    result = _bootstrap(root, env)
    assert result.returncode == 0, result.stdout + result.stderr
    checkout = root / "third_party/franka_description"
    original_head = _git(checkout, "rev-parse", "HEAD")
    selected = tmp_path / "selected"
    _git(tmp_path, "clone", str(sources["franka_description"]), str(selected))
    (selected / "new-source-only").write_text("selected source\n")
    _git(selected, "add", ".")
    _git(selected, "commit", "-qm", "selected source pin")
    selected_ref = _git(selected, "rev-parse", "HEAD")
    env.update(FRANKA_DESCRIPTION_URL=str(selected), FRANKA_DESCRIPTION_REF=selected_ref)
    (checkout / "local-untracked").write_text("keep me\n")

    result = _bootstrap(root, env, *(["--update"] if update else []))

    assert (result.returncode == 0) == update, result.stdout + result.stderr
    assert _git(checkout, "rev-parse", "HEAD") == (selected_ref if update else original_head)
    assert _git(checkout, "remote", "get-url", "origin") == str(sources["franka_description"])
    assert (checkout / "local-untracked").read_text() == "keep me\n"
    assert subprocess.run(
        ["git", "-C", str(checkout), "symbolic-ref", "-q", "HEAD"],
        capture_output=True,
    ).returncode != 0


@pytest.mark.parametrize("staged", [False, True])
def test_bootstrap_rejects_dirty_checkout_before_fetch(configured_sources, tmp_path, staged):
    root, env, _ = configured_sources
    result = _bootstrap(root, env)
    assert result.returncode == 0, result.stdout + result.stderr
    checkout = root / "third_party/franka_description"
    original_head = _git(checkout, "rev-parse", "HEAD")
    (checkout / "README").write_text("local changes\n")
    if staged:
        _git(checkout, "add", "README")
    original_status = _git(checkout, "status", "--porcelain")
    env["FRANKA_DESCRIPTION_URL"] = str(tmp_path / "must-not-fetch")

    result = _bootstrap(root, env, "--update")

    assert result.returncode == 1
    assert "franka_description has tracked local changes" in result.stderr
    assert "must-not-fetch" not in result.stderr
    assert _git(checkout, "rev-parse", "HEAD") == original_head
    assert _git(checkout, "status", "--porcelain") == original_status
    assert (checkout / "README").read_text() == "local changes\n"


def test_bootstrap_checks_out_the_gitlink_and_repoints_an_older_clone(configured_sources, tmp_path):
    root, env, sources = configured_sources
    checkout = root / "third_party/GaussianWrapping"
    pin = _git(root, "rev-parse", "HEAD:third_party/GaussianWrapping")
    older = tmp_path / "older-source"
    _git(tmp_path, "clone", "-q", str(sources["GaussianWrapping"]), str(older))
    _git(older, "reset", "-q", "--hard", f"{pin}~1")
    _git(tmp_path, "clone", "-q", str(older), str(checkout))

    result = _bootstrap(root, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(checkout, "rev-parse", "HEAD") == pin
    assert _git(checkout, "remote", "get-url", "origin") == str(sources["GaussianWrapping"])
    assert _git(root, "submodule", "status", "--", "third_party/GaussianWrapping").startswith(pin)


def test_bootstrap_keeps_a_dirty_gaussian_wrapping_checkout(configured_sources):
    root, env, _ = configured_sources
    assert _bootstrap(root, env).returncode == 0
    checkout = root / "third_party/GaussianWrapping"
    (checkout / "README").write_text("local changes\n")

    result = _bootstrap(root, env)

    assert result.returncode == 1
    assert "GaussianWrapping has tracked local changes" in result.stderr
    assert (checkout / "README").read_text() == "local changes\n"


def test_bootstrap_rejects_the_removed_gaussian_wrapping_overrides(configured_sources):
    root, env, sources = configured_sources
    env["GAUSSIAN_WRAPPING_REF"] = _git(sources["GaussianWrapping"], "rev-parse", "HEAD")

    result = _bootstrap(root, env)

    assert result.returncode == 1
    assert "GAUSSIAN_WRAPPING_URL/REF are not read" in result.stderr
    assert not (root / "third_party/openpi").exists()
    assert not (root / "third_party/GaussianWrapping/.git").exists()
