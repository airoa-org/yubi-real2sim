# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Refuse to import a Git dependency from anything but the build uv.lock pins.

``uv.lock`` records each Git source (OpenPI, its client, robosuite) at a full commit,
and the locked environment installs it as a distribution. A checkout on
``sys.path`` (one under ``third_party/``, a ``PYTHONPATH`` entry or the working
directory) can still win the import and silently replace the locked code.
``require_locked_imports`` is the one check the launcher and the rollout
processes share. Each caller names one of two modes, and each record it
returns states the mode (``check``) and the checks that ran (``checks``).

``CHEAP`` accepts a module only from a file the distribution's ``RECORD``
lists, found at that same path in the distribution's install directory (the
one holding its ``.dist-info``) after every symlink is resolved, and only if
the distribution's ``direct_url.json`` names the Git URL, subdirectory and
commit ``uv.lock`` locks. It reads two metadata files and resolves one path
per package. It refuses a source tree or checkout that shadows the install,
an editable or hand-installed package, an unlisted top-level module, a
top-level package directory or module file replaced by a symlink, and an
environment that was not re-synced after ``uv.lock`` changed. Every rollout
worker and policy loader runs this mode before it imports the package.

``FULL`` runs the cheap checks and also reads every file ``RECORD`` lists for
the package: each must be at its installed path with no symlink on the way
and must still match its recorded hash, and the package directory may hold no
symlink and no importable file ``RECORD`` does not list (``__pycache__`` is
skipped). That adds an installed file edited in place, and a file or
directory below the top level replaced by a symlink to a checkout inside or
outside the repository. It reads every installed byte (robosuite installs
its model assets, 943 files and 547 MB at the locked commit), so it runs once
per launch, in the launcher process: ``real2sim.preflight
.validate_locked_imports`` runs it (for robosuite on every launch, and also
for OpenPI on a production launch), and the run manifest records that same
result (``check: "full"``, ``checked_in: "launcher preflight"``) instead of
hashing again. The rollout worker records its own robosuite and OpenPI
checks with ``check: "cheap"`` and ``checked_in: "rollout worker"`` in
``sim_summary.json``.

The rule is the same wherever the environment lives, including
``<repo>/.venv``, which bootstrap creates inside the repository checkout. Copy
and hardlink installs keep every file inside the install directory (CI's
locked runtime sets ``UV_LINK_MODE=hardlink``); uv's ``symlink`` link mode
points each file into the uv cache instead, so both modes refuse it.
"""

from __future__ import annotations

import base64
import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import sys
import tomllib
from typing import Any, Sequence
from urllib.parse import parse_qs, urlsplit


@dataclass(frozen=True)
class LockedPackage:
    """A top-level import name and the locked distribution that provides it."""

    module: str
    distribution: str


#: The OpenPI packages the policy imports.
OPENPI_PACKAGES: tuple[LockedPackage, ...] = (
    LockedPackage("openpi", "openpi"),
    LockedPackage("openpi_client", "openpi-client"),
)

#: The simulator framework every rollout imports.
ROBOSUITE_PACKAGES: tuple[LockedPackage, ...] = (
    LockedPackage("robosuite", "robosuite"),
)

#: ``require_locked_imports`` mode: the module's file is one ``RECORD`` lists
#: at its installed path, and ``direct_url.json`` names the locked source.
CHEAP = "cheap"
#: ``require_locked_imports`` mode: the cheap checks, plus every installed file
#: of the package against its ``RECORD`` hash, with no symlink and no unlisted
#: importable file in the package directory.
FULL = "full"
#: The checks each mode runs, as each verified record's ``checks`` lists them.
CHECKS: dict[str, tuple[str, ...]] = {
    CHEAP: ("module_file_listed_in_record", "direct_url_names_locked_source"),
    FULL: (
        "module_file_listed_in_record",
        "installed_files_match_record_hashes",
        "no_symlink_or_unlisted_importable_file",
        "direct_url_names_locked_source",
    ),
}

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class LockedImportError(RuntimeError):
    """An import does not resolve to the locked, installed distribution."""


@dataclass(frozen=True)
class LockedGitSource:
    distribution: str
    url: str
    subdirectory: str | None
    commit: str


def _normalized_url(url: str) -> str:
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    return f"{parts.scheme}://{parts.netloc}{path}"


def locked_git_source(lock_path: str | Path, distribution: str) -> LockedGitSource:
    """Read one Git package source, at its full commit, from ``uv.lock``."""

    path = Path(lock_path)
    try:
        lock = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise LockedImportError(f"cannot read the package lock {path}: {exc}") from exc
    entries = [
        package
        for package in lock.get("package", ())
        if package.get("name") == distribution
    ]
    if len(entries) != 1:
        raise LockedImportError(
            f"{path} must lock exactly one {distribution!r} package, found {len(entries)}"
        )
    source = entries[0].get("source") or {}
    git = source.get("git")
    if not isinstance(git, str):
        raise LockedImportError(f"{path} does not lock {distribution!r} to a Git source")
    parts = urlsplit(git)
    commit = parts.fragment
    if _COMMIT_RE.fullmatch(commit) is None:
        raise LockedImportError(
            f"{path} locks {distribution!r} without a full commit: {git}"
        )
    subdirectory = parse_qs(parts.query).get("subdirectory", [None])[0]
    return LockedGitSource(
        distribution=distribution,
        url=_normalized_url(git),
        subdirectory=subdirectory,
        commit=commit,
    )


def _module_origin(module_name: str) -> Path:
    """Where ``import module_name`` resolves now, without importing it."""

    module = sys.modules.get(module_name)
    if module is not None:
        origin = getattr(module, "__file__", None)
        if not origin:
            raise LockedImportError(
                f"{module_name!r} is already imported without a source file"
            )
        return Path(origin).resolve()
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError) as exc:
        raise LockedImportError(f"cannot resolve {module_name!r}: {exc}") from exc
    if spec is None:
        raise LockedImportError(
            f"{module_name!r} is not installed; install the locked environment "
            "with `uv sync --frozen`"
        )
    if spec.origin is None or not spec.has_location:
        raise LockedImportError(
            f"{module_name!r} resolves to a namespace package "
            f"{list(spec.submodule_search_locations or ())}, not an installed "
            "distribution"
        )
    return Path(spec.origin).resolve()


def _installed_at(
    origin: Path, install_dir: Path, distribution: importlib.metadata.Distribution
) -> bool:
    """Whether ``distribution`` installed ``origin`` at that path.

    ``origin`` and ``install_dir`` are both fully resolved. ``RECORD`` paths
    are compared as the installer wrote them, relative to the install
    directory, without following any symlink. A symlink that redirects an
    installed package directory or file, to a checkout inside or outside the
    repository or to any other path, changes the resolved origin, so the
    origin no longer matches the path the installer recorded.
    """

    if not origin.is_relative_to(install_dir):
        return False
    relative = PurePosixPath(origin.relative_to(install_dir).as_posix())
    return any(
        PurePosixPath(posixpath.normpath(file.as_posix())) == relative
        for file in distribution.files or ()
    )


#: Files Python can import from a package directory.
_IMPORTABLE_SUFFIXES = (".py", ".pyc", ".so", ".pyd")


def _record_rows(
    distribution: importlib.metadata.Distribution, name: str
) -> dict[str, str]:
    """``RECORD`` as ``{normalized path: hash}``, paths as the installer wrote them."""

    text = distribution.read_text("RECORD")
    if not text:
        raise LockedImportError(f"distribution {name!r} has no RECORD")
    return {
        posixpath.normpath(row[0]): (row[1] if len(row) > 1 else "")
        for row in csv.reader(text.splitlines())
        if row
    }


def _file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")


def _verify_installed_files(
    module_name: str,
    name: str,
    origin: Path,
    install_dir: Path,
    distribution: importlib.metadata.Distribution,
) -> None:
    """Every file ``name`` installed for ``module_name`` is in place and unchanged.

    ``origin`` is the module's file, already known to be one ``RECORD`` lists.
    For a package this covers every ``RECORD`` entry below its directory
    except ``__pycache__``; for a single-file module, that file. Each must be
    a regular file at its installed path with no symlink on the way (a
    resolved path that differs from the installed one means a symlinked file
    or directory) and must match its ``RECORD`` hash. The package directory is
    then walked without following links: a symlink anywhere in it, or an
    importable file ``RECORD`` does not list, is refused.
    """

    relative = PurePosixPath(origin.relative_to(install_dir).as_posix())
    package = relative.parent if relative.name == "__init__.py" else None
    rows = _record_rows(distribution, name)
    if package is None:
        files = {path: rows[path] for path in rows if path == relative.as_posix()}
    else:
        prefix = package.as_posix() + "/"
        files = {
            path: digest
            for path, digest in rows.items()
            if path.startswith(prefix) and "__pycache__" not in PurePosixPath(path).parts
        }
    for path, recorded in sorted(files.items()):
        installed = install_dir / path
        if installed.resolve() != installed:
            raise LockedImportError(
                f"{installed} is a symlink or lies below one; distribution {name!r} "
                "installed a regular file there"
            )
        if not installed.is_file():
            raise LockedImportError(
                f"{installed}, which distribution {name!r} installed, is missing"
            )
        algorithm, _, expected = recorded.partition("=")
        if algorithm not in hashlib.algorithms_guaranteed or not expected:
            raise LockedImportError(
                f"distribution {name!r} RECORD lists no usable hash for {path}"
            )
        if _file_hash(installed, algorithm) != expected:
            raise LockedImportError(
                f"{installed} differs from the file distribution {name!r} installed "
                f"({algorithm} mismatch); reinstall the locked environment with "
                "`uv sync --frozen`"
            )
    if package is None:
        return
    for directory, dirnames, filenames in os.walk(install_dir / package):
        here = Path(directory)
        for child in (*dirnames, *filenames):
            if (here / child).is_symlink():
                raise LockedImportError(
                    f"{here / child} is a symlink inside the installed "
                    f"{module_name!r} package"
                )
        dirnames[:] = [child for child in dirnames if child != "__pycache__"]
        for filename in filenames:
            path = (here / filename).relative_to(install_dir).as_posix()
            if filename.endswith(_IMPORTABLE_SUFFIXES) and path not in files:
                raise LockedImportError(
                    f"{here / filename} is importable from the installed "
                    f"{module_name!r} package, but distribution {name!r} did not "
                    "install it"
                )


def _direct_url(distribution: importlib.metadata.Distribution, name: str) -> dict[str, Any]:
    text = distribution.read_text("direct_url.json")
    if not text:
        raise LockedImportError(
            f"distribution {name!r} has no direct_url.json; it was not installed "
            "from the locked Git source"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LockedImportError(
            f"distribution {name!r} has an unreadable direct_url.json: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise LockedImportError(f"distribution {name!r} direct_url.json is not an object")
    return payload


def installed_records(packages: Sequence[LockedPackage]) -> list[dict[str, Any]]:
    """Describe the installed distributions without validating them."""

    records = []
    for package in packages:
        try:
            distribution = importlib.metadata.distribution(package.distribution)
        except importlib.metadata.PackageNotFoundError:
            records.append(
                {"module": package.module, "distribution": package.distribution,
                 "version": None, "direct_url": None}
            )
            continue
        try:
            direct_url: dict[str, Any] | None = _direct_url(
                distribution, package.distribution
            )
        except LockedImportError:
            direct_url = None
        records.append(
            {
                "module": package.module,
                "distribution": package.distribution,
                "version": distribution.version,
                "direct_url": direct_url,
            }
        )
    return records


def require_locked_imports(
    repo_root: str | Path,
    packages: Sequence[LockedPackage],
    *,
    mode: str,
    lock_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Fail closed unless each package imports from its locked distribution.

    ``mode`` is ``CHEAP`` or ``FULL``; it has no default, so each caller
    states which checks it runs. For each package both modes check, in order,
    that the module's file, with every symlink resolved, is a file the
    distribution's ``RECORD`` lists at that same path in its install directory
    (so no other ``sys.path`` entry, source or editable checkout, or symlink
    to one replaces it; a module inside ``repo_root``, such as the
    ``<repo>/.venv`` install bootstrap creates, gets the same check and a
    message naming the checkout). ``FULL`` then checks that every file
    ``RECORD`` lists for the package is at its installed path, not reached
    through a symlink, and matches its recorded hash, with no symlink or
    unlisted importable file in the package directory. Both modes then check
    that the distribution's ``direct_url.json`` names the Git URL,
    subdirectory and commit ``uv.lock`` locks. It returns the verified
    records, each with ``check`` (the mode) and ``checks`` (``CHECKS[mode]``).
    """

    if mode not in CHECKS:
        raise ValueError(
            f"locked import check mode must be one of {sorted(CHECKS)}, got {mode!r}"
        )
    root = Path(repo_root).resolve()
    lock = Path(lock_path) if lock_path is not None else root / "uv.lock"
    verified = []
    for package in packages:
        module_name, name = package.module, package.distribution
        locked = locked_git_source(lock, name)
        origin = _module_origin(module_name)
        try:
            distribution: importlib.metadata.Distribution | None = (
                importlib.metadata.distribution(name)
            )
        except importlib.metadata.PackageNotFoundError:
            distribution = None
        install_dir: Path | None = None
        installed = False
        if distribution is not None:
            install_dir = Path(str(distribution.locate_file(""))).resolve()
            installed = _installed_at(origin, install_dir, distribution)
        if origin.is_relative_to(root) and not installed:
            raise LockedImportError(
                f"{module_name!r} resolves inside the repository checkout ({origin}) "
                f"to a file distribution {name!r} did not install; run the installed "
                f"{name!r} distribution locked at {locked.commit}"
            )
        if distribution is None:
            raise LockedImportError(
                f"{module_name!r} resolves to {origin}, but distribution {name!r} "
                "is not installed"
            )
        if not installed:
            raise LockedImportError(
                f"{module_name!r} resolves to {origin}, which distribution {name!r} "
                "did not install; another sys.path entry shadows it, or a symlink "
                f"replaces a file it installed in {install_dir}"
            )
        if mode == FULL:
            _verify_installed_files(module_name, name, origin, install_dir, distribution)
        direct_url = _direct_url(distribution, name)
        vcs = direct_url.get("vcs_info") or {}
        observed = (
            _normalized_url(str(direct_url.get("url", ""))),
            direct_url.get("subdirectory"),
            vcs.get("commit_id"),
        )
        expected = (locked.url, locked.subdirectory, locked.commit)
        if vcs.get("vcs") != "git" or observed != expected:
            raise LockedImportError(
                f"distribution {name!r} was installed from "
                f"url={observed[0]} subdirectory={observed[1]} commit={observed[2]}; "
                f"{lock} locks url={expected[0]} subdirectory={expected[1]} "
                f"commit={expected[2]}"
            )
        verified.append(
            {
                "module": module_name,
                "distribution": name,
                "version": distribution.version,
                "origin": str(origin),
                "direct_url": direct_url,
                "check": mode,
                "checks": list(CHECKS[mode]),
            }
        )
    return verified
