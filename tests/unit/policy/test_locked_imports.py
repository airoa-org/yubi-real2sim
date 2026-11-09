# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Git dependencies must import from the installed distribution uv.lock pins.

Each case builds a real ``site`` directory with real ``.dist-info`` metadata and
puts it on ``sys.path``; nothing in ``importlib`` is mocked. The policy loader
and the launcher preflight are driven through their production entry points
with a shadowing checkout, and must refuse it before the deploy config or the
model loader is reached.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tomllib
import types
from unittest import mock

import pytest

from real2sim import locked_imports
from real2sim.locked_imports import (
    CHEAP,
    CHECKS,
    FULL,
    OPENPI_PACKAGES,
    ROBOSUITE_PACKAGES,
    LockedImportError,
    installed_records,
    locked_git_source,
    require_locked_imports,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK = REPO_ROOT / "uv.lock"


def _direct_url(distribution: str, **override) -> dict:
    locked = locked_git_source(LOCK, distribution)
    payload = {
        "url": locked.url,
        "vcs_info": {
            "vcs": "git",
            "commit_id": locked.commit,
            "requested_revision": locked.commit,
        },
    }
    if locked.subdirectory is not None:
        payload["subdirectory"] = locked.subdirectory
    payload.update(override)
    return payload


def _record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


#: Files every test package installs besides ``__init__.py``: a subpackage, so
#: edits and symlinks below the top level are covered.
PACKAGE_FILES = {"models/__init__.py": b"", "models/model.py": b"VALUE = 1\n"}


def _install(site: Path, module: str, distribution: str, direct_url: dict | None) -> None:
    package = site / module
    package.mkdir(parents=True)
    rows = []
    for path, data in {"__init__.py": b"", **PACKAGE_FILES}.items():
        target = package / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        rows.append(f"{module}/{path},{_record_hash(data)},{len(data)}")
    info = site / f"{distribution.replace('-', '_')}-0.1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 0.1.0\n",
        encoding="utf-8",
    )
    (info / "RECORD").write_text(
        "\n".join([*rows, f"{info.name}/METADATA,,", f"{info.name}/RECORD,,"]) + "\n",
        encoding="utf-8",
    )
    if direct_url is not None:
        (info / "direct_url.json").write_text(json.dumps(direct_url), encoding="utf-8")


def _shadow(root: Path, module: str) -> Path:
    package = root / module
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    return root


def _link_out(site: Path, module: str, source_root: Path, linked: str) -> Path:
    """Replace an installed package directory or file with a symlink.

    The symlink points into a source checkout at ``source_root``; the
    distribution's RECORD and direct_url.json stay as installed. Returns the
    checkout file the module now resolves to.
    """

    source = _shadow(source_root, module) / module
    installed = site / module
    if linked == "package-directory":
        shutil.rmtree(installed)
        installed.symlink_to(source, target_is_directory=True)
    else:
        (installed / "__init__.py").unlink()
        (installed / "__init__.py").symlink_to(source / "__init__.py")
    importlib.invalidate_caches()
    return (source / "__init__.py").resolve()


#: Where a symlinked installed path leads. A per-file symlink out of the
#: install directory is also what uv's ``symlink`` link mode leaves.
LINK_TARGETS = {
    # A source checkout beside the repository, outside it.
    "outside-repository": lambda env, tmp_path: tmp_path / "openpi-checkout" / "src",
    # The former third_party layout inside the repository.
    "inside-repository": lambda env, tmp_path: env.checkout / "third_party/openpi/src",
    # Another directory of the same site-packages that RECORD does not list.
    "unrecorded-install-path": lambda env, tmp_path: env.site / "unlocked-openpi",
}


#: Where the locked environment lives relative to the checked repository.
#: ``scripts/bootstrap.sh`` and CI's locked-runtime job install it at
#: ``<repo>/.venv``, so its site-packages is inside the checkout the guard is
#: given; a site directory outside the checkout is the other supported layout.
SITE_LAYOUTS = {
    "repository-venv": lambda tmp_path, checkout: (
        checkout / ".venv" / "lib" / "python3.11" / "site-packages"
    ),
    "outside-site": lambda tmp_path, checkout: tmp_path / "site",
}


@pytest.fixture(params=sorted(SITE_LAYOUTS))
def site_layout(request):
    return request.param


#: The cheap checks (the module file is listed in RECORD, direct_url.json names
#: the locked source) catch every case these tests run in both modes.
@pytest.fixture(params=[CHEAP, FULL])
def check_mode(request):
    return request.param


def _site_dir(layout: str, tmp_path: Path, checkout: Path) -> Path:
    site = SITE_LAYOUTS[layout](tmp_path, checkout)
    site.mkdir(parents=True)
    return site


@pytest.fixture
def isolated_openpi(monkeypatch, tmp_path, site_layout):
    """A clean import state with a locked-looking OpenPI installed in ``site``."""

    for name in list(sys.modules):
        if name.split(".")[0] in {"openpi", "openpi_client"} or name == "robosuite_env.policy.yubi_openpi.config_yubi":
            monkeypatch.delitem(sys.modules, name)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "uv.lock").write_bytes(LOCK.read_bytes())
    site = _site_dir(site_layout, tmp_path, checkout)
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    yield types.SimpleNamespace(site=site, checkout=checkout)
    importlib.invalidate_caches()


def _install_locked(site: Path) -> None:
    _install(site, "openpi", "openpi", _direct_url("openpi"))
    _install(site, "openpi_client", "openpi-client", _direct_url("openpi-client"))


def test_lock_and_project_source_name_the_same_openpi_commit():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    source = project["tool"]["uv"]["sources"]["openpi"]
    openpi = locked_git_source(LOCK, "openpi")
    client = locked_git_source(LOCK, "openpi-client")
    assert openpi.commit == client.commit == source["rev"]
    assert openpi.url == client.url == source["git"].removesuffix(".git")
    assert (openpi.subdirectory, client.subdirectory) == (None, "packages/openpi-client")


def test_installed_locked_distribution_passes(isolated_openpi, check_mode):
    _install_locked(isolated_openpi.site)

    verified = require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=check_mode)

    assert [record["distribution"] for record in verified] == ["openpi", "openpi-client"]
    for record, module in zip(verified, ("openpi", "openpi_client")):
        assert record["origin"] == str((isolated_openpi.site / module / "__init__.py").resolve())
        assert record["direct_url"]["vcs_info"]["commit_id"] == locked_git_source(
            LOCK, record["distribution"]
        ).commit


def test_checkout_inside_the_repository_is_refused(isolated_openpi, monkeypatch, check_mode):
    _install_locked(isolated_openpi.site)
    # The layout ensure_openpi_import_paths used to put first on sys.path.
    monkeypatch.syspath_prepend(
        str(_shadow(isolated_openpi.checkout / "third_party/openpi/src", "openpi"))
    )
    importlib.invalidate_caches()

    with pytest.raises(LockedImportError, match="inside the repository checkout"):
        require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=check_mode)


def test_unrecorded_module_beside_the_installed_distribution_is_refused(isolated_openpi, check_mode):
    # A stale module left in the same site-packages is not the locked install.
    _install_locked(isolated_openpi.site)
    record = isolated_openpi.site / "openpi-0.1.0.dist-info" / "RECORD"
    record.write_text(
        "".join(
            line
            for line in record.read_text(encoding="utf-8").splitlines(keepends=True)
            if not line.startswith("openpi/__init__.py,")
        ),
        encoding="utf-8",
    )

    with pytest.raises(LockedImportError, match="did not install"):
        require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=check_mode)


def test_record_entry_reaching_into_the_checkout_is_refused(isolated_openpi, monkeypatch, check_mode):
    # RECORD may list files outside the install directory (scripts, data); a
    # source checkout named that way is still not an installed module.
    _install_locked(isolated_openpi.site)
    source = _shadow(isolated_openpi.checkout / "third_party/openpi/src", "openpi")
    shadow = source / "openpi" / "__init__.py"
    record = isolated_openpi.site / "openpi-0.1.0.dist-info" / "RECORD"
    with record.open("a", encoding="utf-8") as stream:
        stream.write(f"{os.path.relpath(shadow, isolated_openpi.site)},,\n")
    monkeypatch.syspath_prepend(str(source))
    importlib.invalidate_caches()

    with pytest.raises(LockedImportError, match="inside the repository checkout"):
        require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=check_mode)


def test_editable_install_of_a_repository_checkout_is_refused(isolated_openpi, monkeypatch, check_mode):
    # What `pip install -e third_party/openpi` leaves: metadata in site-packages
    # and a .pth path entry that resolves the module to the checkout.
    source = _shadow(isolated_openpi.checkout / "third_party/openpi/src", "openpi")
    info = isolated_openpi.site / "openpi-0.1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: openpi\nVersion: 0.1.0\n", encoding="utf-8"
    )
    pth = "__editable__.openpi-0.1.0.pth"
    (isolated_openpi.site / pth).write_text(f"{source}\n", encoding="utf-8")
    (info / "RECORD").write_text(
        f"{pth},,\n{info.name}/METADATA,,\n{info.name}/RECORD,,\n", encoding="utf-8"
    )
    (info / "direct_url.json").write_text(
        json.dumps({"url": source.parent.as_uri(), "dir_info": {"editable": True}}),
        encoding="utf-8",
    )
    _install(
        isolated_openpi.site, "openpi_client", "openpi-client", _direct_url("openpi-client")
    )
    monkeypatch.syspath_prepend(str(source))
    importlib.invalidate_caches()

    with pytest.raises(LockedImportError, match="inside the repository checkout"):
        require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=check_mode)


@pytest.mark.parametrize("target", sorted(LINK_TARGETS))
@pytest.mark.parametrize("linked", ["package-directory", "module-file"])
def test_installed_path_symlinked_out_of_place_is_refused(
    isolated_openpi, tmp_path, linked, target, check_mode
):
    # The same install passes until one installed path becomes a symlink.
    _install_locked(isolated_openpi.site)
    require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=check_mode)
    origin = _link_out(
        isolated_openpi.site, "openpi", LINK_TARGETS[target](isolated_openpi, tmp_path), linked
    )

    # Whether the link target is in the checkout picks the message, not the outcome.
    if origin.is_relative_to(isolated_openpi.checkout.resolve()):
        message = "inside the repository checkout"
    else:
        message = "did not install; another sys.path entry shadows it, or a symlink"
    with pytest.raises(LockedImportError, match=message) as refused:
        require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=check_mode)
    assert str(origin) in str(refused.value)


def test_installed_file_edited_in_a_subpackage_is_refused(isolated_openpi):
    _install_locked(isolated_openpi.site)
    require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=FULL)
    edited = isolated_openpi.site / "openpi" / "models" / "model.py"
    edited.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(
        LockedImportError, match="differs from the file distribution 'openpi' installed"
    ) as refused:
        require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=FULL)
    assert str(edited) in str(refused.value)


@pytest.mark.parametrize("target", sorted(LINK_TARGETS))
@pytest.mark.parametrize("linked", ["subdirectory", "subpackage-file"])
def test_installed_path_below_the_top_level_symlinked_out_of_place_is_refused(
    isolated_openpi, tmp_path, linked, target
):
    # The link target holds identical bytes, so only the symlink is wrong.
    _install_locked(isolated_openpi.site)
    require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=FULL)
    installed = isolated_openpi.site / "openpi" / "models"
    copy = LINK_TARGETS[target](isolated_openpi, tmp_path) / "openpi" / "models"
    shutil.copytree(installed, copy)
    if linked == "subdirectory":
        shutil.rmtree(installed)
        installed.symlink_to(copy, target_is_directory=True)
    else:
        (installed / "model.py").unlink()
        (installed / "model.py").symlink_to(copy / "model.py")
    importlib.invalidate_caches()

    with pytest.raises(LockedImportError, match="is a symlink or lies below one"):
        require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=FULL)


@pytest.mark.parametrize("added", ["module", "symlink"])
def test_unlisted_module_or_symlink_in_the_installed_package_is_refused(
    isolated_openpi, tmp_path, added
):
    _install_locked(isolated_openpi.site)
    require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=FULL)
    path = isolated_openpi.site / "openpi" / "models" / "extra.py"
    if added == "module":
        path.write_text("VALUE = 3\n", encoding="utf-8")
        message = "is importable from the installed 'openpi' package"
    else:
        checkout = _shadow(tmp_path / "openpi-checkout", "openpi") / "openpi"
        path.symlink_to(checkout / "__init__.py")
        message = "is a symlink inside the installed 'openpi' package"

    with pytest.raises(LockedImportError, match=message):
        require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=FULL)


def test_shadowing_path_outside_the_repository_is_refused(isolated_openpi, monkeypatch, tmp_path, check_mode):
    _install_locked(isolated_openpi.site)
    monkeypatch.syspath_prepend(str(_shadow(tmp_path / "elsewhere", "openpi_client")))
    importlib.invalidate_caches()

    with pytest.raises(LockedImportError, match="did not install; another sys.path entry"):
        require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=check_mode)


def test_already_imported_shadow_is_refused(isolated_openpi, monkeypatch, tmp_path, check_mode):
    _install_locked(isolated_openpi.site)
    shadow = _shadow(tmp_path / "elsewhere", "openpi") / "openpi" / "__init__.py"
    monkeypatch.setitem(
        sys.modules, "openpi", types.SimpleNamespace(__file__=str(shadow))
    )

    with pytest.raises(LockedImportError, match="did not install"):
        require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=check_mode)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"vcs_info": {"vcs": "git", "commit_id": "0" * 40}}, "commit=" + "0" * 40),
        ({"url": "https://example.invalid/fork/openpi"}, "url=https://example.invalid/fork/openpi"),
        ({"vcs_info": None, "dir_info": {"editable": True}}, "commit=None"),
    ],
    ids=["wrong-commit", "wrong-url", "editable-checkout"],
)
def test_distribution_from_another_source_is_refused(isolated_openpi, override, message, check_mode):
    _install(isolated_openpi.site, "openpi", "openpi", _direct_url("openpi", **override))
    _install(
        isolated_openpi.site, "openpi_client", "openpi-client", _direct_url("openpi-client")
    )

    with pytest.raises(LockedImportError, match=message):
        require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=check_mode)


def test_distribution_without_direct_url_is_refused(isolated_openpi, check_mode):
    _install(isolated_openpi.site, "openpi", "openpi", None)
    _install(
        isolated_openpi.site, "openpi_client", "openpi-client", _direct_url("openpi-client")
    )

    with pytest.raises(LockedImportError, match="has no direct_url.json"):
        require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=check_mode)


def test_lock_without_full_commit_is_refused(isolated_openpi, check_mode):
    _install_locked(isolated_openpi.site)
    lock = isolated_openpi.checkout / "uv.lock"
    text = lock.read_text(encoding="utf-8")
    commit = locked_git_source(LOCK, "openpi").commit
    lock.write_text(
        text.replace(f'openpi?rev={commit}#{commit}"', 'openpi?rev=main#main"'),
        encoding="utf-8",
    )

    with pytest.raises(LockedImportError, match="without a full commit"):
        require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=check_mode)


def test_installed_records_describe_metadata_without_validating(isolated_openpi):
    _install(
        isolated_openpi.site,
        "openpi",
        "openpi",
        _direct_url("openpi", vcs_info={"vcs": "git", "commit_id": "1" * 40}),
    )
    _install(isolated_openpi.site, "openpi_client", "openpi-client", None)

    records = installed_records(OPENPI_PACKAGES)

    assert records[0]["direct_url"]["vcs_info"]["commit_id"] == "1" * 40
    assert records[1] == {
        "module": "openpi_client",
        "distribution": "openpi-client",
        "version": "0.1.0",
        "direct_url": None,
    }


def _checkpoint(root: Path) -> Path:
    stats = root / "checkpoint" / "assets" / "dataset" / "norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text("{}", encoding="utf-8")
    return root / "checkpoint"


def _refuse_before_config(isolated_openpi, monkeypatch, call, *, replace=None):
    """Run ``call`` with a repository checkout of OpenPI first on sys.path.

    ``replace``, when given, swaps in a checkout another way instead and
    returns the refusal message it expects.
    """

    from robosuite_env.policy import policy_batch_infer as pbi

    _install_locked(isolated_openpi.site)
    if replace is None:
        monkeypatch.syspath_prepend(
            str(_shadow(isolated_openpi.checkout / "third_party/openpi/src", "openpi"))
        )
        message = "inside the repository checkout"
    else:
        message = replace()
    importlib.invalidate_caches()
    deploy = types.ModuleType("robosuite_env.policy.yubi_openpi.config_yubi")
    deploy.get_config = mock.Mock(side_effect=AssertionError("deploy config reached"))
    monkeypatch.setitem(sys.modules, "robosuite_env.policy.yubi_openpi.config_yubi", deploy)
    guard = mock.Mock(wraps=locked_imports.require_locked_imports)
    monkeypatch.setattr(locked_imports, "require_locked_imports", guard)
    load = mock.Mock(side_effect=AssertionError("policy loaded"))
    with pytest.raises(LockedImportError, match=message):
        call(pbi, load)
    # The refusal came from the shared guard, for this checkout, before the
    # deploy config or any model loader ran.
    guard.assert_called_once_with(
        isolated_openpi.checkout.resolve(), OPENPI_PACKAGES, mode=CHEAP
    )
    deploy.get_config.assert_not_called()
    load.assert_not_called()
    assert "openpi" not in sys.modules


def test_policy_loader_refuses_a_shadowing_checkout_before_loading(
    isolated_openpi, monkeypatch, tmp_path
):
    checkpoint = _checkpoint(tmp_path)

    def call(pbi, load):
        policies = types.ModuleType("openpi.policies")
        policies.policy_config = types.SimpleNamespace(create_trained_policy=load)
        with mock.patch.dict(sys.modules, {"openpi.policies": policies}):
            pbi.load_openpi_policy(
                checkpoint,
                config="pi05_yubi_pickplace",
                repo_root=isolated_openpi.checkout,
            )

    _refuse_before_config(isolated_openpi, monkeypatch, call)


def test_policy_loader_refuses_an_installed_package_linked_out_before_loading(
    isolated_openpi, monkeypatch, tmp_path
):
    checkpoint = _checkpoint(tmp_path)

    def replace():
        _link_out(
            isolated_openpi.site,
            "openpi",
            LINK_TARGETS["outside-repository"](isolated_openpi, tmp_path),
            "package-directory",
        )
        return "did not install; another sys.path entry shadows it, or a symlink"

    def call(pbi, load):
        policies = types.ModuleType("openpi.policies")
        policies.policy_config = types.SimpleNamespace(create_trained_policy=load)
        with mock.patch.dict(sys.modules, {"openpi.policies": policies}):
            pbi.load_openpi_policy(
                checkpoint,
                config="pi05_yubi_pickplace",
                repo_root=isolated_openpi.checkout,
            )

    _refuse_before_config(isolated_openpi, monkeypatch, call, replace=replace)


def test_launcher_preflight_refuses_a_shadowing_checkout(
    isolated_openpi, monkeypatch, tmp_path
):
    from real2sim import preflight

    checkpoint = _checkpoint(tmp_path)
    plan = {
        "repo_root": str(isolated_openpi.checkout),
        "resolved": {
            "policy": {
                "checkpoint": str(checkpoint),
                "config_name": "pi05_yubi_pickplace",
                "config_source": "deploy",
                "action_horizon": 32,
            }
        },
    }

    def call(_pbi, _load):
        preflight._openpi_norm_stats_for_plan(plan)

    _refuse_before_config(isolated_openpi, monkeypatch, call)


@pytest.mark.parametrize("mode", ["--dry-run", "--check-runtime"])
def test_cli_reports_a_shadowing_openpi_as_a_configuration_error(
    isolated_openpi, monkeypatch, tmp_path, capsys, mode
):
    """real2sim-run exits 2 with the guard's message, not a traceback."""

    from real2sim import cli

    profile = REPO_ROOT / "config/release/fr3/flat/reading_room_cup_cup1trim.yaml"
    inputs = tmp_path / "inputs"
    # The profile's checkpoint, with norm stats, so auto resolution loads OpenPI.
    stats = inputs / "checkpoints/pi05_cup/assets/dataset/norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text("{}", encoding="utf-8")
    _install_locked(isolated_openpi.site)
    monkeypatch.syspath_prepend(str(_shadow(tmp_path / "openpi-checkout" / "src", "openpi")))
    importlib.invalidate_caches()
    monkeypatch.setenv("REAL2SIM_INPUTS_ROOT", str(inputs))
    env_file = tmp_path / "empty.env"
    env_file.write_text("", encoding="utf-8")

    assert cli.main(["--config", str(profile), "--env-file", str(env_file), mode]) == 2

    err = capsys.readouterr().err
    assert "locked OpenPI check failed" in err
    assert "another sys.path entry shadows it" in err
    assert "Traceback" not in err
    assert "openpi" not in sys.modules


def test_production_provenance_requires_the_locked_openpi(isolated_openpi, monkeypatch):
    from real2sim.provenance import (
        LOCKED_RUN_IMPORTS,
        ProvenanceError,
        collect_locked_imports,
    )

    # Run manifests record robosuite too; this test drives the OpenPI pair.
    assert LOCKED_RUN_IMPORTS == OPENPI_PACKAGES + ROBOSUITE_PACKAGES
    _install_locked(isolated_openpi.site)
    records = collect_locked_imports(
        isolated_openpi.checkout, production=True, packages=OPENPI_PACKAGES
    )
    assert [record["verified"] for record in records] == [True, True]

    monkeypatch.syspath_prepend(
        str(_shadow(isolated_openpi.checkout / "third_party/openpi/src", "openpi"))
    )
    importlib.invalidate_caches()
    with pytest.raises(ProvenanceError, match="inside the repository checkout"):
        collect_locked_imports(
            isolated_openpi.checkout, production=True, packages=OPENPI_PACKAGES
        )
    fixture = collect_locked_imports(
        isolated_openpi.checkout, production=False, packages=OPENPI_PACKAGES
    )
    assert [record["verified"] for record in fixture] == [False, False]


def test_policy_loader_records_the_openpi_it_verified(isolated_openpi, monkeypatch, tmp_path):
    """The process that loads the model publishes what it imported.

    A batch server forwards this metadata in its hello packet and ready file,
    and an in-process policy reports it to the worker.
    """

    import hashlib

    from robosuite_env.policy import policy_batch_infer as pbi

    _install_locked(isolated_openpi.site)
    importlib.invalidate_caches()
    checkpoint = _checkpoint(tmp_path)
    stats = checkpoint / "assets" / "dataset" / "norm_stats.json"
    digest = hashlib.sha256(stats.read_bytes()).hexdigest()
    deploy = types.ModuleType("robosuite_env.policy.yubi_openpi.config_yubi")
    deploy.get_config = lambda name: types.SimpleNamespace(
        name=name,
        model=types.SimpleNamespace(action_horizon=32),
        data=types.SimpleNamespace(resolve_norm_stats_asset_id=lambda: "dataset"),
    )
    monkeypatch.setitem(sys.modules, "robosuite_env.policy.yubi_openpi.config_yubi", deploy)
    monkeypatch.setattr(
        pbi.pio,
        "policy_state_quat_hemisphere_from_norm_stats",
        lambda path: {
            "hemisphere": "positive",
            "classification": "fixture",
            "qw_q01": 0.5,
            "qw_q99": 0.9,
            "norm_stats_sha256": digest,
        },
    )
    guard = mock.Mock(wraps=locked_imports.require_locked_imports)
    monkeypatch.setattr(locked_imports, "require_locked_imports", guard)
    policies = types.ModuleType("openpi.policies")
    policies.policy_config = types.SimpleNamespace(
        create_trained_policy=lambda *a, **k: types.SimpleNamespace()
    )
    with mock.patch.dict(sys.modules, {"openpi.policies": policies}):
        policy = pbi.load_openpi_policy(
            checkpoint, config="pi05_yubi_pickplace", repo_root=isolated_openpi.checkout
        )

    guard.assert_called_once_with(
        isolated_openpi.checkout.resolve(), OPENPI_PACKAGES, mode=CHEAP
    )
    recorded = pbi.policy_load_metadata(policy)["locked_imports"]
    assert [(r["module"], r["distribution"]) for r in recorded] == [
        ("openpi", "openpi"),
        ("openpi_client", "openpi-client"),
    ]
    assert recorded[0]["origin"].startswith(str(isolated_openpi.site))
    assert recorded[0]["direct_url"]["vcs_info"]["commit_id"] == _direct_url("openpi")["vcs_info"]["commit_id"]


def test_worker_records_the_openpi_it_verified(isolated_openpi, monkeypatch):
    """The rollout worker keeps the guard's records for sim_summary.json."""

    import inspect

    from robosuite_env.rollout.stages import init_policy_render

    _install_locked(isolated_openpi.site)
    importlib.invalidate_caches()
    earlier = {"module": "already-recorded"}
    state = types.SimpleNamespace(REPO=isolated_openpi.checkout, LOCKED_IMPORTS=[earlier])
    records = init_policy_render.record_locked_openpi(state)
    assert [r["module"] for r in records] == ["openpi", "openpi_client"]
    assert state.LOCKED_IMPORTS == [earlier, *records]

    # The same worker refuses, and records nothing, once a checkout shadows it.
    monkeypatch.syspath_prepend(
        str(_shadow(isolated_openpi.checkout / "third_party/openpi/src", "openpi"))
    )
    importlib.invalidate_caches()
    for name in [n for n in sys.modules if n.split(".")[0] == "openpi"]:
        monkeypatch.delitem(sys.modules, name)
    fresh = types.SimpleNamespace(REPO=isolated_openpi.checkout)
    with pytest.raises(LockedImportError, match="inside the repository checkout"):
        init_policy_render.record_locked_openpi(fresh)
    assert not hasattr(fresh, "LOCKED_IMPORTS")

    # The policy stage records before it contacts a batch server or loads.
    source = inspect.getsource(init_policy_render.run)
    assert source.index("record_locked_openpi(state)") < source.index("BatchPolicyClient(")
    assert source.index("record_locked_openpi(state)") < source.index("load_openpi_policy(")
    summary = inspect.getsource(
        importlib.import_module("robosuite_env.rollout.stages.finalize_outputs")
    )
    assert 'summ["locked_imports"]=getattr(state,"LOCKED_IMPORTS",None)' in summary
@pytest.fixture
def isolated_robosuite(monkeypatch, tmp_path, site_layout):
    """Hide an already imported robosuite and install a locked-looking one."""

    # Only the top-level entry: the guard reads it, and leaving the submodules
    # keeps robosuite's registries intact for the rest of the session.
    monkeypatch.delitem(sys.modules, "robosuite", raising=False)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "uv.lock").write_bytes(LOCK.read_bytes())
    site = _site_dir(site_layout, tmp_path, checkout)
    _install(site, "robosuite", "robosuite", _direct_url("robosuite"))
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    yield types.SimpleNamespace(site=site, checkout=checkout)
    importlib.invalidate_caches()


def test_installed_locked_robosuite_passes(isolated_robosuite, check_mode):
    [record] = require_locked_imports(isolated_robosuite.checkout, ROBOSUITE_PACKAGES, mode=check_mode)

    assert record["origin"] == str(
        (isolated_robosuite.site / "robosuite" / "__init__.py").resolve()
    )
    assert record["direct_url"]["vcs_info"]["commit_id"] == locked_git_source(
        LOCK, "robosuite"
    ).commit


def test_robosuite_checkout_in_the_repository_is_refused(isolated_robosuite, monkeypatch, check_mode):
    # The former third_party/robosuite submodule layout.
    monkeypatch.syspath_prepend(
        str(_shadow(isolated_robosuite.checkout / "third_party/robosuite", "robosuite"))
    )
    importlib.invalidate_caches()

    with pytest.raises(LockedImportError, match="inside the repository checkout"):
        require_locked_imports(isolated_robosuite.checkout, ROBOSUITE_PACKAGES, mode=check_mode)


def test_robosuite_from_another_commit_is_refused(isolated_robosuite, tmp_path, monkeypatch, check_mode):
    other = tmp_path / "other-site"
    _install(
        other,
        "robosuite",
        "robosuite",
        _direct_url("robosuite", vcs_info={"vcs": "git", "commit_id": "2" * 40}),
    )
    monkeypatch.syspath_prepend(str(other))
    importlib.invalidate_caches()

    with pytest.raises(LockedImportError, match="commit=" + "2" * 40):
        require_locked_imports(isolated_robosuite.checkout, ROBOSUITE_PACKAGES, mode=check_mode)


def test_launcher_preflight_refuses_an_unlocked_robosuite(isolated_robosuite, monkeypatch):
    from real2sim import preflight
    from real2sim.config._types import ConfigError

    plan = {"repo_root": str(isolated_robosuite.checkout)}
    # The locked install passes, including at <repo>/.venv (repository-venv).
    preflight.validate_locked_imports(plan)

    monkeypatch.syspath_prepend(
        str(_shadow(isolated_robosuite.checkout / "third_party/robosuite", "robosuite"))
    )
    importlib.invalidate_caches()
    with pytest.raises(ConfigError, match="locked import check failed.*inside the repository"):
        preflight.validate_locked_imports(plan)


@pytest.mark.parametrize("linked", ["package-directory", "module-file"])
def test_launcher_preflight_refuses_robosuite_linked_out_of_its_install(
    isolated_robosuite, tmp_path, linked
):
    from real2sim import preflight
    from real2sim.config._types import ConfigError

    plan = {"repo_root": str(isolated_robosuite.checkout)}
    preflight.validate_locked_imports(plan)

    # A robosuite source checkout beside the repository, outside it.
    origin = _link_out(
        isolated_robosuite.site, "robosuite", tmp_path / "robosuite-checkout", linked
    )
    with pytest.raises(
        ConfigError, match="locked import check failed.*did not install"
    ) as refused:
        preflight.validate_locked_imports(plan)
    assert str(origin) in str(refused.value)


def test_cli_check_runtime_reports_an_unlocked_robosuite_as_a_configuration_error(
    isolated_robosuite, monkeypatch, tmp_path, capsys
):
    """real2sim-run --check-runtime exits 2 with the guard's message, not a traceback.

    The production doctor and the host path checks run before the robosuite
    check and need staged inputs this host does not have, so they are stubbed.
    """

    from real2sim import cli, preflight

    monkeypatch.setattr(preflight, "validate_production_preflight", lambda *a, **k: None)
    monkeypatch.setattr(preflight, "validate_runtime_paths", lambda plan: {})
    monkeypatch.syspath_prepend(str(_shadow(tmp_path / "robosuite-checkout", "robosuite")))
    importlib.invalidate_caches()
    env_file = tmp_path / "empty.env"
    env_file.write_text("", encoding="utf-8")
    profile = REPO_ROOT / "config/release/fixture_v2.yaml"

    assert cli.main(["--config", str(profile), "--env-file", str(env_file), "--check-runtime"]) == 2

    err = capsys.readouterr().err
    assert "locked import check failed" in err
    assert "another sys.path entry shadows it" in err
    assert "Traceback" not in err


def test_robosuite_file_edited_below_the_top_level_is_refused(isolated_robosuite):
    from real2sim import preflight
    from real2sim.config._types import ConfigError

    plan = {"repo_root": str(isolated_robosuite.checkout)}
    preflight.validate_locked_imports(plan)
    (isolated_robosuite.site / "robosuite" / "models" / "model.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )

    with pytest.raises(ConfigError, match="locked import check failed.*differs from the file"):
        preflight.validate_locked_imports(plan)


def test_robosuite_subdirectory_symlinked_to_a_checkout_is_refused(isolated_robosuite, tmp_path):
    from real2sim import preflight
    from real2sim.config._types import ConfigError

    plan = {"repo_root": str(isolated_robosuite.checkout)}
    preflight.validate_locked_imports(plan)
    installed = isolated_robosuite.site / "robosuite" / "models"
    copy = tmp_path / "robosuite-checkout" / "robosuite" / "models"
    shutil.copytree(installed, copy)
    shutil.rmtree(installed)
    installed.symlink_to(copy, target_is_directory=True)
    importlib.invalidate_caches()

    with pytest.raises(ConfigError, match="locked import check failed.*symlink or lies below one"):
        preflight.validate_locked_imports(plan)


def test_launch_preflight_runs_the_robosuite_check_before_the_ruckig_check(monkeypatch):
    from real2sim import preflight

    calls = []
    monkeypatch.setattr(preflight, "validate_production_preflight", lambda *a, **k: None)
    monkeypatch.setattr(preflight, "validate_runtime_paths", lambda plan: calls.append("paths"))
    monkeypatch.setattr(
        preflight,
        "validate_locked_imports",
        mock.Mock(side_effect=preflight.ConfigError("refused")),
    )
    monkeypatch.setattr(
        preflight, "validate_joint_trajectory_runtime", lambda plan: calls.append("ruckig")
    )
    with pytest.raises(preflight.ConfigError, match="refused"):
        preflight.validate_launch_preflight({"repo_root": str(REPO_ROOT)})
    assert calls == ["paths"]
    preflight.validate_locked_imports.assert_called_once()


def test_worker_refuses_an_unlocked_robosuite_before_building_the_model(monkeypatch):
    pytest.importorskip("mujoco")
    pytest.importorskip("cv2")
    pytest.importorskip("robosuite")
    # The worker entry point prepends its own directories; keep them local.
    monkeypatch.setattr(sys, "path", list(sys.path))
    from robosuite_env.rollout import worker

    built = []
    monkeypatch.setattr(worker, "RolloutSession", lambda: types.SimpleNamespace())
    monkeypatch.setattr(worker, "validate_args", lambda session: None)
    monkeypatch.setattr(worker, "build_model_env", lambda session: built.append(session))
    guard = mock.Mock(side_effect=LockedImportError("refused robosuite"))
    monkeypatch.setattr(worker, "require_locked_imports", guard)

    with pytest.raises(LockedImportError, match="refused robosuite"):
        worker.main()
    guard.assert_called_once_with(worker.REPO, ROBOSUITE_PACKAGES, mode=CHEAP)
    assert built == []


def test_production_callers_accept_the_locked_install(isolated_openpi, monkeypatch, tmp_path):
    """The launcher preflight and the policy loader pass the locked install.

    With the ``repository-venv`` layout this is the environment bootstrap
    creates: OpenPI imports from ``<repo>/.venv/.../site-packages``, inside the
    checkout both callers hand to the guard.
    """

    import hashlib

    from real2sim import preflight
    from real2sim.provenance import collect_locked_imports
    from robosuite_env.policy import policy_batch_infer as pbi

    _install_locked(isolated_openpi.site)
    importlib.invalidate_caches()
    checkpoint = _checkpoint(tmp_path)
    stats = (checkpoint / "assets" / "dataset" / "norm_stats.json").resolve()
    deploy = types.ModuleType("robosuite_env.policy.yubi_openpi.config_yubi")
    deploy.get_config = mock.Mock(
        side_effect=lambda name: types.SimpleNamespace(
            name=name,
            model=types.SimpleNamespace(action_horizon=32),
            data=types.SimpleNamespace(resolve_norm_stats_asset_id=lambda: "dataset"),
        )
    )
    monkeypatch.setitem(sys.modules, "robosuite_env.policy.yubi_openpi.config_yubi", deploy)
    guard = mock.Mock(wraps=locked_imports.require_locked_imports)
    monkeypatch.setattr(locked_imports, "require_locked_imports", guard)
    plan = {
        "repo_root": str(isolated_openpi.checkout),
        "resolved": {
            "policy": {
                "checkpoint": str(checkpoint),
                "config_name": "pi05_yubi_pickplace",
                "config_source": "deploy",
                "action_horizon": 32,
            }
        },
    }

    assert preflight._openpi_norm_stats_for_plan(plan) == (stats, "dataset")

    monkeypatch.setattr(
        pbi.pio,
        "policy_state_quat_hemisphere_from_norm_stats",
        lambda path: {
            "hemisphere": "positive",
            "classification": "fixture",
            "qw_q01": 0.5,
            "qw_q99": 0.9,
            "norm_stats_sha256": hashlib.sha256(stats.read_bytes()).hexdigest(),
        },
    )
    load = mock.Mock(return_value=types.SimpleNamespace())
    policies = types.ModuleType("openpi.policies")
    policies.policy_config = types.SimpleNamespace(create_trained_policy=load)
    with mock.patch.dict(sys.modules, {"openpi.policies": policies}):
        pbi.load_openpi_policy(
            checkpoint, config="pi05_yubi_pickplace", repo_root=isolated_openpi.checkout
        )

    load.assert_called_once()
    assert guard.call_args_list == [
        mock.call(isolated_openpi.checkout.resolve(), OPENPI_PACKAGES, mode=CHEAP)
    ] * 2
    records = collect_locked_imports(
        isolated_openpi.checkout, production=True, packages=OPENPI_PACKAGES
    )
    assert [record["verified"] for record in records] == [True, True]
    site = isolated_openpi.site.resolve()
    assert all(Path(record["origin"]).is_relative_to(site) for record in records)


def test_the_installed_environment_passes_the_guard():
    """The real environment running these tests passes, wherever it is installed.

    CI's locked-runtime job installs it at ``<repo>/.venv``, the layout
    bootstrap creates, so there the verified files are inside ``REPO_ROOT``.
    """

    packages = OPENPI_PACKAGES + ROBOSUITE_PACKAGES
    for package in packages:
        try:
            importlib.metadata.distribution(package.distribution)
        except importlib.metadata.PackageNotFoundError:
            pytest.skip(f"{package.distribution} is not installed in this environment")

    records = require_locked_imports(REPO_ROOT, packages, mode=FULL)
    assert require_locked_imports(REPO_ROOT, packages, mode=CHEAP)

    assert [record["distribution"] for record in records] == [
        package.distribution for package in packages
    ]
    if Path(sys.prefix).resolve().is_relative_to(REPO_ROOT):
        assert all(Path(record["origin"]).is_relative_to(REPO_ROOT) for record in records)


# Which mode each caller runs. The full check reads every installed byte, so a
# launch runs it once (preflight, recorded by the manifest) and each rollout
# worker and policy loader runs only the cheap checks.


def test_check_mode_has_no_default_and_takes_one_of_two_values(isolated_openpi):
    _install_locked(isolated_openpi.site)

    with pytest.raises(TypeError, match="mode"):
        require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES)
    with pytest.raises(ValueError, match="mode must be one of"):
        require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode="both")


def test_cheap_mode_reads_no_installed_file_and_says_so(isolated_openpi):
    """Cheap mode verifies the import origin and direct_url.json only.

    An installed file edited below the top level passes it, and its records
    name exactly the checks that ran; the full mode refuses the same install.
    """

    _install_locked(isolated_openpi.site)
    (isolated_openpi.site / "openpi" / "models" / "model.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    hashed = mock.Mock(side_effect=AssertionError("cheap mode hashed a file"))
    with mock.patch.object(locked_imports, "_file_hash", hashed):
        records = require_locked_imports(
            isolated_openpi.checkout, OPENPI_PACKAGES, mode=CHEAP
        )

    assert [(r["check"], r["checks"]) for r in records] == [(CHEAP, list(CHECKS[CHEAP]))] * 2
    hashed.assert_not_called()
    with pytest.raises(LockedImportError, match="differs from the file distribution"):
        require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=FULL)


def _refuse_full_check(monkeypatch):
    """Make any full-mode check in this process fail the test."""

    def full(*args, **kwargs):
        raise AssertionError("a rollout worker or policy loader ran the full check")

    monkeypatch.setattr(locked_imports, "_verify_installed_files", full)


def test_worker_and_policy_loader_run_only_the_cheap_check(isolated_openpi, monkeypatch):
    from robosuite_env.policy import policy_batch_infer as pbi
    from robosuite_env.rollout.stages import init_policy_render

    _install_locked(isolated_openpi.site)
    importlib.invalidate_caches()
    _refuse_full_check(monkeypatch)

    state = types.SimpleNamespace(REPO=isolated_openpi.checkout)
    records = init_policy_render.record_locked_openpi(state)
    assert [(r["check"], r["checked_in"]) for r in records] == [
        (CHEAP, "rollout worker")
    ] * 2
    # The policy loader (in-process or batch server) and the launcher's
    # norm-stats resolution share this entry point.
    assert [r["check"] for r in pbi.ensure_locked_openpi(isolated_openpi.checkout)] == [
        CHEAP
    ] * 2


def _production_plan(checkout: Path) -> dict:
    return {
        "repo_root": str(checkout),
        "resolved": {"reproducibility": {"production": True}},
    }


def test_launch_preflight_runs_the_full_check_for_production(isolated_openpi, monkeypatch):
    from real2sim import preflight, provenance
    from real2sim.config._types import ConfigError

    monkeypatch.setattr(provenance, "LOCKED_RUN_IMPORTS", OPENPI_PACKAGES)
    _install_locked(isolated_openpi.site)
    importlib.invalidate_caches()
    guard = mock.Mock(wraps=locked_imports.require_locked_imports)
    monkeypatch.setattr(locked_imports, "require_locked_imports", guard)

    records = preflight.validate_locked_imports(_production_plan(isolated_openpi.checkout))

    guard.assert_called_once_with(str(isolated_openpi.checkout), OPENPI_PACKAGES, mode=FULL)
    assert [(r["check"], r["checks"]) for r in records] == [(FULL, list(CHECKS[FULL]))] * 2
    # A fixture launch checks only robosuite, which every rollout builds with,
    # and its manifest records the packages unverified.
    fixture_guard = mock.Mock(return_value=[])
    monkeypatch.setattr(locked_imports, "require_locked_imports", fixture_guard)
    for fixture in ({"resolved": {"reproducibility": {"production": False}}}, {"resolved": {}}):
        assert preflight.validate_locked_imports(fixture) == []
    assert fixture_guard.call_args_list == [
        mock.call(preflight.REPO_ROOT, ROBOSUITE_PACKAGES, mode=FULL)
    ] * 2
    assert guard.call_count == 1
    monkeypatch.setattr(locked_imports, "require_locked_imports", guard)

    edited = isolated_openpi.site / "openpi" / "models" / "model.py"
    edited.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="locked import check failed: .*differs from"):
        preflight.validate_locked_imports(_production_plan(isolated_openpi.checkout))


def test_launch_preflight_reports_the_full_check_records(monkeypatch):
    from real2sim import preflight

    records = [{"module": "openpi", "check": FULL}]
    monkeypatch.setattr(preflight, "validate_production_preflight", lambda *a, **k: None)
    monkeypatch.setattr(preflight, "validate_runtime_paths", lambda plan: {})
    monkeypatch.setattr(preflight, "validate_locked_imports", lambda plan: records)
    monkeypatch.setattr(preflight, "validate_joint_trajectory_runtime", lambda plan: None)

    assert preflight.validate_launch_preflight({})["_locked_imports"] is records


def test_manifest_records_the_launch_full_check_without_hashing_again(
    isolated_openpi, monkeypatch
):
    from real2sim import preflight, provenance
    from real2sim.provenance import ProvenanceError, collect_locked_imports

    monkeypatch.setattr(provenance, "LOCKED_RUN_IMPORTS", OPENPI_PACKAGES)
    _install_locked(isolated_openpi.site)
    importlib.invalidate_caches()
    launch = preflight.validate_locked_imports(_production_plan(isolated_openpi.checkout))
    hashed = mock.Mock(side_effect=AssertionError("the manifest hashed the files again"))
    monkeypatch.setattr(locked_imports, "_file_hash", hashed)

    records = collect_locked_imports(
        isolated_openpi.checkout,
        production=True,
        packages=OPENPI_PACKAGES,
        launch_records=launch,
    )

    hashed.assert_not_called()
    assert [
        (r["verified"], r["check"], r["checks"], r["checked_in"]) for r in records
    ] == [(True, FULL, list(CHECKS[FULL]), "launcher preflight")] * 2
    assert [r["origin"] for r in records] == [r["origin"] for r in launch]
    # Only full-check records for exactly the recorded packages are reused.
    cheap = require_locked_imports(isolated_openpi.checkout, OPENPI_PACKAGES, mode=CHEAP)
    with pytest.raises(ProvenanceError, match="did not come from the full check"):
        collect_locked_imports(
            isolated_openpi.checkout, production=True, packages=OPENPI_PACKAGES,
            launch_records=cheap,
        )
    with pytest.raises(ProvenanceError, match="cover"):
        collect_locked_imports(
            isolated_openpi.checkout, production=True, packages=OPENPI_PACKAGES,
            launch_records=launch[:1],
        )


def test_manifest_without_launch_records_runs_the_full_check(isolated_openpi):
    from real2sim.provenance import ProvenanceError, collect_locked_imports

    _install_locked(isolated_openpi.site)
    importlib.invalidate_caches()
    records = collect_locked_imports(
        isolated_openpi.checkout, production=True, packages=OPENPI_PACKAGES
    )
    assert [(r["check"], r["checked_in"]) for r in records] == [(FULL, "run manifest")] * 2

    (isolated_openpi.site / "openpi" / "models" / "model.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    with pytest.raises(ProvenanceError, match="differs from the file distribution"):
        collect_locked_imports(
            isolated_openpi.checkout, production=True, packages=OPENPI_PACKAGES
        )


def test_prepare_run_manifest_records_the_preflight_full_check(monkeypatch, tmp_path):
    """The launcher hands the preflight's records to the manifest builder."""

    from real2sim import launch_provenance
    from real2sim.config._types import ConfigError
    from real2sim.provenance import ProvenanceError

    records = [{"module": "openpi", "check": FULL}]
    seen = {}

    def build(**kwargs):
        seen.update(kwargs)
        raise ProvenanceError("stop after the call")

    monkeypatch.setattr(
        launch_provenance, "verify_plan_artifacts",
        lambda *a, **k: ({"path": "m.yaml", "sha256": "0" * 64}, []),
    )
    monkeypatch.setattr(launch_provenance, "build_runner_environment", lambda *a, **k: {})
    monkeypatch.setattr(launch_provenance, "build_run_manifest", build)
    plan = {
        "repo_root": str(tmp_path),
        "config_repo_path": "config.yaml",
        "config_sha256": "0" * 64,
        "timestamp": "20261102T000000Z",
        "resolved": {
            "reproducibility": {
                "production": True,
                "lockfiles": [],
                "source_dependencies": [],
                "accelerator_profile": {"kind": "cpu"},
            },
            "runtime": {"gaussian_wrapping_repo": None},
        },
    }

    with pytest.raises(ConfigError, match="stop after the call"):
        launch_provenance.prepare_run_manifest(plan, {"_locked_imports": records})

    assert seen["launch_locked_imports"] is records


def test_worker_records_the_cheap_robosuite_check_before_building_the_model(monkeypatch):
    pytest.importorskip("mujoco")
    pytest.importorskip("cv2")
    pytest.importorskip("robosuite")
    monkeypatch.setattr(sys, "path", list(sys.path))
    from robosuite_env.rollout import worker

    session = types.SimpleNamespace()
    record = {"module": "robosuite", "check": CHEAP, "checks": list(CHECKS[CHEAP])}
    monkeypatch.setattr(worker, "RolloutSession", lambda: session)
    monkeypatch.setattr(worker, "validate_args", lambda session: None)

    def check(root, packages, *, mode):
        assert mode == CHEAP, "the rollout worker ran the full check"
        return [dict(record)]

    guard = mock.Mock(side_effect=check)
    monkeypatch.setattr(worker, "require_locked_imports", guard)

    class Built(Exception):
        pass

    def build(session):
        raise Built

    monkeypatch.setattr(worker, "build_model_env", build)
    with pytest.raises(Built):
        worker.main()
    guard.assert_called_once_with(worker.REPO, ROBOSUITE_PACKAGES, mode=CHEAP)
    assert session.LOCKED_IMPORTS == [{**record, "checked_in": "rollout worker"}]
