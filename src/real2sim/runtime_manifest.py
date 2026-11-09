# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Seal and verify the source/config generation used by rollout workers.

The batch runner starts a fresh simulator process for every rollout wave.  A
developer edit while a long batch is running would otherwise let later waves
import a different checkout.  This module records an explicit runtime
dependency set -- the sources a worker imports and the repository data files
it and a post-hoc replay read -- and verifies both its exact membership and
every file's stable SHA-256/state before another wave is accepted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = 1
MANIFEST_FILE = "runtime_source_manifest.json"

#: A tree spec suffix that selects the data files of a tree: every file Git
#: lists there (tracked, or untracked and not ignored) and every archive file
#: the repository declares as installed there (:data:`ARCHIVE_DECLARATIONS`),
#: except inside ``__pycache__`` directories. Other ignored content -- local
#: run outputs, scene point clouds, nested ``.git`` directories -- is not part
#: of a generation. Outside a Git checkout every file under the tree is.
ALL_FILES = "*"

#: The repository files that declare where an archive installs files into the
#: checkout: the artifacts manifest (``{repo_root}`` collections and trees) and
#: the release download catalog (``{repo_root}`` archive destinations).
ARTIFACT_MANIFEST = "artifacts/manifest.yaml"
RELEASE_DOWNLOAD_CATALOG = "artifacts/release_downloads.json"
_REPO_ROOT_FIELD = "{repo_root}"

# Deliberately exclude tests, training utilities, and unrelated release tools.
# These are the executable/import roots used by run_policy_batch.sh, its policy
# proxy, rollout workers, and rollout finalizers, and every repository data
# directory a worker or a post-hoc replay reads: the MuJoCo models, meshes and
# textures, object calibration and layouts, controller and robot configs,
# initial poses and task placements, and the grid video templates. A data tree
# seals its data files (see ALL_FILES) under any suffix, so a file added under
# a new suffix is a change too.
DEFAULT_TREE_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("assets", (ALL_FILES,)),
    ("config/render_templates", (ALL_FILES,)),
    ("robosuite_env/assets", (ALL_FILES,)),
    ("robosuite_env/contracts", (".py",)),
    ("robosuite_env/controllers", (ALL_FILES,)),
    ("robosuite_env/obj_calib", (ALL_FILES,)),
    ("robosuite_env/policy", (".py",)),
    ("robosuite_env/rendering", (".py",)),
    ("robosuite_env/rerender", (".py",)),
    ("robosuite_env/robots", (ALL_FILES,)),
    ("robosuite_env/rollout", (".py",)),
    ("robosuite_env/tasks", (".py",)),
    ("scene", (".py",)),
)
DEFAULT_FILES: tuple[str, ...] = (
    # Read by a post-hoc rerender: the profile support matrix, and the lock the
    # locked-import guard compares the installed OpenPI and robosuite against.
    "config/support-matrix.yaml",
    "robosuite_env/__init__.py",
    # Core renderer source: the GS server argv and /health identity check used
    # by the rollout worker's fallback launcher and the post-hoc renderer.
    "robosuite_env/gs_launch.py",
    "robosuite_env/paths.py",
    "robosuite_env/sim.py",
    "robosuite_env/tools/__init__.py",
    "robosuite_env/tools/make_rollout_grid_video.py",
    "robosuite_env/tools/plot_repeat_background_summary.py",
    "robosuite_env/tools/plot_sim_trajectories.py",
    "robosuite_env/tools/rerender_third_person.py",
    "robosuite_env/tools/summarize_policy_rollouts.py",
    "robosuite_env/tools/validate_rollout_video.py",
    "scripts/run_policy_batch.sh",
    "src/real2sim/__init__.py",
    "src/real2sim/_bootstrap.py",
    "src/real2sim/artifacts.py",
    "src/real2sim/artifacts_binding.py",
    "src/real2sim/cli.py",
    "src/real2sim/config/__init__.py",
    "src/real2sim/config/_types.py",
    "src/real2sim/config/controllers.py",
    "src/real2sim/config/initial_arm_joints.py",
    "src/real2sim/config/keys.py",
    "src/real2sim/config/parameters.py",
    "src/real2sim/config/paths.py",
    "src/real2sim/config/reproducibility.py",
    "src/real2sim/config/schema.py",
    "src/real2sim/config/shards.py",
    "src/real2sim/config/task_objects.py",
    "src/real2sim/env_file.py",
    "src/real2sim/fr3_controllers.py",
    "src/real2sim/gpu_binding.py",
    "src/real2sim/start_state_guard.py",
    "src/real2sim/launch_provenance.py",
    "src/real2sim/launcher.py",
    "src/real2sim/locked_imports.py",
    "src/real2sim/preflight.py",
    "src/real2sim/provenance.py",
    "src/real2sim/reporting/__init__.py",
    "src/real2sim/reporting/completion.py",
    "src/real2sim/reporting/dry_run.py",
    "src/real2sim/reporting/posthoc_provenance.py",
    "src/real2sim/reporting/summary.py",
    "src/real2sim/runner/__init__.py",
    "src/real2sim/runner/argv.py",
    "src/real2sim/runner/environment.py",
    "src/real2sim/runner/execute.py",
    "src/real2sim/runtime_manifest.py",
    "src/real2sim/support_matrix.py",
    "src/real2sim/wrist_profiles.py",
    "uv.lock",
)
_STATE_KEYS = ("device", "inode", "mode", "size", "mtime_ns", "ctime_ns")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class RuntimeManifestError(RuntimeError):
    """A runtime generation could not be sealed or no longer matches."""


class RuntimeSourceDriftError(RuntimeManifestError):
    """The sealed files are intact but the working tree has moved.

    This is the one failure a running batch is allowed to survive: the
    checkout changed while the batch was running, and the rollouts themselves
    are unaffected. Every other failure this module
    raises means the manifest is missing, unreadable, malformed or
    self-inconsistent, and those must still stop the run -- a broken manifest
    is not evidence that anything is fine. The two are separated by type here
    and by exit code in ``main`` so the shell runner can tell them apart
    instead of treating any non-zero exit as drift.
    """


@contextmanager
def _drift_context(*, enabled: bool) -> Iterator[None]:
    """Read the working tree, optionally calling any on-disk failure drift.

    ``enabled`` is the caller's statement that a record already exists for what
    is being read, which is what makes a missing or replaced path an edit
    rather than a broken configuration.
    """

    try:
        yield
    except RuntimeSourceDriftError:
        raise
    except RuntimeManifestError as exc:
        if not enabled:
            raise
        raise RuntimeSourceDriftError(str(exc)) from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _file_state(observed: os.stat_result) -> dict[str, int]:
    return {
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "mode": int(observed.st_mode),
        "size": int(observed.st_size),
        "mtime_ns": int(observed.st_mtime_ns),
        "ctime_ns": int(observed.st_ctime_ns),
    }


def _stable_file_snapshot(
    path: Path,
    *,
    retain_content: bool,
) -> tuple[dict[str, Any], bytes | None]:
    """Read one non-symlink file and reject concurrent replacement.

    Every failure here is raised as a plain manifest error. Whether it means
    the tree moved is not a property of the file -- the same read serves
    sealing, where an unreadable path is a configuration error, and verifying,
    where it is an ordinary edit. Only the caller knows which, and
    ``_drift_context`` is where verification says so.
    """

    try:
        before_path = path.lstat()
    except OSError as exc:
        raise RuntimeManifestError(f"critical runtime file is unreadable: {path}: {exc}") from exc
    if stat.S_ISLNK(before_path.st_mode):
        raise RuntimeManifestError(f"critical runtime file must not be a symlink: {path}")
    if not stat.S_ISREG(before_path.st_mode):
        raise RuntimeManifestError(f"critical runtime path is not a regular file: {path}")

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeManifestError(f"critical runtime file could not be opened: {path}: {exc}") from exc
    digest = hashlib.sha256()
    content = bytearray() if retain_content else None
    try:
        # Report read failures as manifest errors (exit 2), like every other
        # failure here, so the shell can classify them by exit code.
        try:
            opened_before = os.fstat(descriptor)
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                if content is not None:
                    content.extend(chunk)
            opened_after = os.fstat(descriptor)
        except OSError as exc:
            raise RuntimeManifestError(
                f"critical runtime file could not be read: {path}: {exc}"
            ) from exc
        close_failure = None
    finally:
        # A close that raises must not replace whatever the read handler above
        # already raised, and must not be swallowed either: a read that
        # succeeded and a close that failed is not a successful read.
        try:
            os.close(descriptor)
        except OSError as exc:
            close_failure = exc
    if close_failure is not None:
        raise RuntimeManifestError(
            f"critical runtime file could not be closed: {path}: {close_failure}"
        ) from close_failure
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise RuntimeManifestError(
            f"critical runtime file disappeared while hashing: {path}: {exc}"
        ) from exc

    states = tuple(
        _file_state(value)
        for value in (before_path, opened_before, opened_after, after_path)
    )
    if any(value != states[0] for value in states[1:]):
        raise RuntimeManifestError(f"critical runtime file changed while hashing: {path}")
    return (
        {
            "path": str(path),
            "sha256": digest.hexdigest(),
            "state": states[0],
        },
        bytes(content) if content is not None else None,
    )


def _stable_file_record(path: Path) -> dict[str, Any]:
    """Hash one non-symlink regular file and reject concurrent replacement."""

    record, _content = _stable_file_snapshot(path, retain_content=False)
    return record


def _canonical_regular_file(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = Path(os.path.abspath(candidate))
    # Do not call resolve(): retaining the lexical path lets lstat reject a
    # symlink instead of silently sealing only its current target.
    try:
        observed = candidate.lstat()
    except OSError as exc:
        raise RuntimeManifestError(f"critical runtime file is missing: {candidate}: {exc}") from exc
    if stat.S_ISLNK(observed.st_mode):
        raise RuntimeManifestError(f"critical runtime file must not be a symlink: {candidate}")
    if not stat.S_ISREG(observed.st_mode):
        raise RuntimeManifestError(f"critical runtime path is not a regular file: {candidate}")
    return candidate


def tree_selects(path: Path, tree_root: Path, suffixes: Iterable[str]) -> bool:
    """Whether a tree spec with ``suffixes`` rooted at ``tree_root`` selects ``path``.

    Sealing and the moved-on rerender check both use this, so they agree on
    which files a tree holds.
    """

    suffixes = tuple(suffixes)
    if ALL_FILES in suffixes:
        return "__pycache__" not in path.relative_to(tree_root).parts
    return path.suffix in suffixes


def _git_listed_files(root: Path) -> frozenset[str] | None:
    """Files Git lists in the checkout at ``root``: tracked, or untracked and not ignored.

    ``None`` when ``root`` is not the top level of a Git checkout.
    """

    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if not top or Path(top).resolve() != root.resolve():
        return None
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeManifestError(
            f"could not list the files Git tracks in {root}: {exc}"
        ) from exc
    return frozenset(os.fsdecode(item) for item in listed.split(b"\0") if item)


def declared_archive_paths(root: Path) -> tuple[frozenset[str], frozenset[str]]:
    """The archive files and directories the repository declares as installed in it.

    Returns ``(files, directories)`` relative to ``root``: the members of each
    ``{repo_root}`` collection and the ``{repo_root}`` file or tree of every
    other entry in :data:`ARTIFACT_MANIFEST`, and each ``{repo_root}``
    destination in :data:`RELEASE_DOWNLOAD_CATALOG`. These are the gitignored
    payloads the replay loads (meshes, textures, collision hulls), so a data
    tree seals them although Git does not list them.
    """

    files: set[str] = set()
    directories: set[str] = set()

    def relative(template: Any, *, label: str) -> str | None:
        if not isinstance(template, str) or not template.startswith(_REPO_ROOT_FIELD):
            return None
        value = template[len(_REPO_ROOT_FIELD):].strip("/")
        if value and (Path(value).is_absolute() or ".." in Path(value).parts):
            raise RuntimeManifestError(f"{label} is not inside the checkout: {template!r}")
        return value

    manifest_path = root / ARTIFACT_MANIFEST
    if manifest_path.is_file():
        from real2sim.artifacts import ManifestError, load_manifest

        try:
            entries = load_manifest(manifest_path)["artifacts"].values()
        except ManifestError as exc:
            raise RuntimeManifestError(
                f"cannot read the archive declarations in {manifest_path}: {exc}"
            ) from exc
        for entry in entries:
            base = relative(entry.get("local_path"), label=f"{ARTIFACT_MANIFEST} {entry['id']}")
            if base is None:
                continue
            if entry.get("kind") == "collection":
                for member in entry.get("files") or ():
                    files.add(Path(base, member).as_posix())
            elif entry.get("kind") == "file":
                files.add(base)
            elif base:
                directories.add(base)
    catalog_path = root / RELEASE_DOWNLOAD_CATALOG
    if catalog_path.is_file():
        try:
            archives = json.loads(catalog_path.read_text(encoding="utf-8"))["archives"]
            pins = list(archives.items())
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
            raise RuntimeManifestError(
                f"cannot read the archive declarations in {catalog_path}: {exc}"
            ) from exc
        for name, pin in pins:
            base = relative(
                pin.get("destination") if isinstance(pin, dict) else None,
                label=f"{RELEASE_DOWNLOAD_CATALOG} {name}",
            )
            if base:
                directories.add(base)
    return frozenset(files), frozenset(directories)


def _symlink_error(path: Path, *, how: str) -> RuntimeManifestError:
    return RuntimeManifestError(
        f"critical runtime source tree must not contain symlinks: {path} "
        f"(a symlink {how}). The seal "
        "hashes regular files only and does not follow links. Replace it with "
        "the file or directory it points to (install archives with "
        "real2sim-download-release), "
        "or remove it, or, if it is local content the runtime does not read "
        "and Git does not track, ignore it in .gitignore, before launching"
    )


def _first_symlink(root: Path, relative: str) -> Path | None:
    """The first path from ``root`` down to ``root / relative`` that is a symlink."""

    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            return current
    return None


def _under(relative: str, prefix: str) -> bool:
    return prefix == "" or relative == prefix or relative.startswith(prefix + "/")


def tree_members(root: Path, trees: Iterable[Mapping[str, Any]]) -> list[str]:
    """The files the tree specs ``trees`` select in the checkout at ``root``.

    Sealing and the moved-on rerender check both use this, so they agree on
    which files a tree holds. A suffix tree selects every file with one of
    its suffixes. A data tree (:data:`ALL_FILES`) selects the files Git lists
    there plus the archive files the repository declares as installed there
    (:func:`declared_archive_paths`); in a directory that is not a Git
    checkout it selects every file except inside ``.git``. A symlink among
    the selected files, or on the way to a declared archive file, fails with
    what to do about it.
    """

    root = Path(root)
    trees = list(trees)
    listing: frozenset[str] | None = None
    declared: tuple[frozenset[str], frozenset[str]] | None = None
    if any(ALL_FILES in tree["suffixes"] for tree in trees):
        listing = _git_listed_files(root)
        if listing is not None:
            declared = declared_archive_paths(root)
    selected: set[str] = set()
    for tree in trees:
        tree_path = tree["path"]
        suffixes = tuple(tree["suffixes"])
        tree_root = root / tree_path
        if ALL_FILES not in suffixes or listing is None:
            for candidate in _walk_tree(tree_root, skip_git=ALL_FILES in suffixes):
                if candidate.is_symlink():
                    raise _symlink_error(candidate, how="in a sealed tree")
                if tree_selects(candidate, tree_root, suffixes) and candidate.is_file():
                    selected.add(candidate.relative_to(root).as_posix())
            continue
        assert declared is not None
        declared_files, declared_directories = declared
        candidates = {item for item in listing if _under(item, tree_path)}
        candidates |= {item for item in declared_files if _under(item, tree_path)}
        for directory in sorted(declared_directories):
            if not _under(directory, tree_path):
                continue
            # A declared archive directory that is a link, or sits below one,
            # would seal whatever the link points to today.
            linked = _first_symlink(root, directory)
            if linked is not None:
                raise _symlink_error(linked, how="on the way to an installed archive")
            if not (root / directory).is_dir():
                continue
            for candidate in _walk_tree(root / directory, skip_git=True):
                if candidate.is_symlink():
                    raise _symlink_error(candidate, how="in an installed archive")
                if candidate.is_file():
                    candidates.add(candidate.relative_to(root).as_posix())
        for relative in sorted(candidates):
            if relative.endswith("/"):
                raise RuntimeManifestError(
                    "critical runtime source tree holds a nested Git checkout "
                    f"that is not ignored: {root / relative}. Ignore it in "
                    ".gitignore or move it out of the tree before launching"
                )
            candidate = root / relative
            linked = _first_symlink(root, relative)
            if linked is not None:
                raise _symlink_error(
                    linked, how="that Git lists or the repository declares, or above one"
                )
            if not candidate.exists():
                # A tracked file deleted in the working tree is not read.
                continue
            if tree_selects(candidate, tree_root, suffixes) and candidate.is_file():
                selected.add(relative)
    return sorted(selected)


def required_coverage() -> dict[str, Any]:
    """The files and trees this checkout's runtime reads, as a seal names them."""

    return {
        "files": sorted(set(DEFAULT_FILES)),
        "trees": [
            {"path": path, "suffixes": list(suffixes)}
            for path, suffixes in DEFAULT_TREE_SPECS
        ],
    }


def selection_covers(selection: Mapping[str, Any], coverage: Mapping[str, Any]) -> bool:
    """Whether a sealed ``selection`` hashes everything ``coverage`` names.

    Every covered file must be a sealed file, and every covered tree must be a
    sealed tree at the same path that selects at least the same files. A seal
    written before a tree or file joined the coverage does not cover it.
    """

    try:
        sealed_files = set(selection["files"])
        sealed_trees = {
            tree["path"]: set(tree["suffixes"]) for tree in selection["trees"]
        }
        if not set(coverage["files"]) <= sealed_files:
            return False
        for tree in coverage["trees"]:
            sealed = sealed_trees.get(tree["path"])
            if sealed is None:
                return False
            if ALL_FILES not in sealed and (
                ALL_FILES in tree["suffixes"] or not set(tree["suffixes"]) <= sealed
            ):
                return False
    except (KeyError, TypeError):
        return False
    return True


def default_selection(
    repo_root: str | Path,
    *,
    extra_files: Iterable[str | Path] = (),
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    files = list(DEFAULT_FILES)
    external_files: list[str] = []
    for raw_path in extra_files:
        path = _canonical_regular_file(raw_path)
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            external_files.append(str(path))
        else:
            files.append(relative)
    return {
        "repo_root": str(root),
        "files": sorted(set(files)),
        "external_files": sorted(set(external_files)),
        "trees": [
            {"path": path, "suffixes": list(suffixes)}
            for path, suffixes in DEFAULT_TREE_SPECS
        ],
    }


def _validate_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeManifestError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise RuntimeManifestError(f"{label} is not a canonical relative path: {value!r}")
    return value


def _require_canonical_repo_root(raw_root: str) -> None:
    """Check the recorded root against the filesystem as it is right now."""

    root = Path(raw_root)
    try:
        resolved = root.resolve()
    except OSError as exc:
        raise RuntimeManifestError(
            f"runtime manifest repo_root is unreadable: {raw_root}: {exc}"
        ) from exc
    if resolved != root:
        raise RuntimeManifestError(
            f"runtime manifest repo_root is not canonical: {raw_root}"
        )


def _validated_selection(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "repo_root",
        "files",
        "external_files",
        "trees",
    }:
        raise RuntimeManifestError("runtime manifest selection has an invalid structure")
    raw_root = value["repo_root"]
    if not isinstance(raw_root, str) or not Path(raw_root).is_absolute():
        raise RuntimeManifestError("runtime manifest repo_root must be absolute")
    root = Path(raw_root)
    if str(root) != raw_root:
        raise RuntimeManifestError("runtime manifest repo_root is not canonical")
    # root.resolve() reads the filesystem, so a worktree moved or swapped for a
    # symlink after sealing would fail here -- outside the drift context, and
    # therefore as corruption. The lexical part of the check stays; the part
    # that depends on what is on disk now happens during discovery, where
    # verification can call it an edit and sealing still fails closed.

    raw_files = value["files"]
    if not isinstance(raw_files, list):
        raise RuntimeManifestError("runtime manifest files must be a list")
    files = [_validate_relative_path(item, label="selection.files item") for item in raw_files]
    if files != sorted(set(files)):
        raise RuntimeManifestError("runtime manifest files must be sorted and unique")

    raw_external = value["external_files"]
    if not isinstance(raw_external, list):
        raise RuntimeManifestError("runtime manifest external_files must be a list")
    external_files: list[str] = []
    for item in raw_external:
        if not isinstance(item, str) or not Path(item).is_absolute():
            raise RuntimeManifestError("selection.external_files items must be absolute paths")
        path = Path(item)
        if str(path) != item:
            raise RuntimeManifestError(
                f"selection.external_files item is not canonical: {item!r}"
            )
        external_files.append(item)
    if external_files != sorted(set(external_files)):
        raise RuntimeManifestError("runtime manifest external_files must be sorted and unique")

    raw_trees = value["trees"]
    if not isinstance(raw_trees, list):
        raise RuntimeManifestError("runtime manifest trees must be a list")
    trees: list[dict[str, Any]] = []
    for index, raw_tree in enumerate(raw_trees):
        if not isinstance(raw_tree, dict) or set(raw_tree) != {"path", "suffixes"}:
            raise RuntimeManifestError(f"selection.trees[{index}] has an invalid structure")
        tree_path = _validate_relative_path(
            raw_tree["path"], label=f"selection.trees[{index}].path"
        )
        suffixes = raw_tree["suffixes"]
        if (
            not isinstance(suffixes, list)
            or not suffixes
            or not (
                suffixes == [ALL_FILES]
                or all(
                    isinstance(suffix, str)
                    and suffix.startswith(".")
                    and "/" not in suffix
                    for suffix in suffixes
                )
            )
            or suffixes != sorted(set(suffixes))
        ):
            raise RuntimeManifestError(
                f"selection.trees[{index}].suffixes must be sorted unique suffixes "
                f"or [{ALL_FILES!r}]"
            )
        trees.append({"path": tree_path, "suffixes": suffixes})
    if [tree["path"] for tree in trees] != sorted(
        {tree["path"] for tree in trees}
    ):
        raise RuntimeManifestError("runtime manifest trees must be path-sorted and unique")
    return {
        "repo_root": str(root),
        "files": files,
        "external_files": external_files,
        "trees": trees,
    }


def _walk_tree(tree_root: Path, *, skip_git: bool = False) -> Iterator[Path]:
    """Yield every entry under ``tree_root``, refusing to skip what it cannot read.

    Path.rglob swallows PermissionError and simply omits the subtree. Sealing
    would then record a generation with a directory missing from it and call
    that a complete capture, and a later verification would not notice files
    added beneath it. A directory this module cannot list is a failure, not an
    empty directory. ``skip_git`` leaves out ``.git`` entries and what is
    below them.
    """

    stack = [tree_root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda path: str(path))
        except OSError as exc:
            raise RuntimeManifestError(
                f"critical runtime source tree could not be listed: {current}: {exc}"
            ) from exc
        for entry in entries:
            if skip_git and entry.name == ".git":
                continue
            yield entry
            if entry.is_dir() and not entry.is_symlink():
                stack.append(entry)


def _discover_selected_paths(selection: Mapping[str, Any]) -> list[Path]:
    _require_canonical_repo_root(selection["repo_root"])
    validated = _validated_selection(dict(selection))
    root = Path(validated["repo_root"])
    selected: set[Path] = set()
    for relative in validated["files"]:
        selected.add(_canonical_regular_file(root / relative))
    for absolute in validated["external_files"]:
        selected.add(_canonical_regular_file(absolute))
    for tree in validated["trees"]:
        tree_root = root / tree["path"]
        if not tree_root.is_dir() or tree_root.is_symlink():
            raise RuntimeManifestError(
                f"critical runtime source tree is missing or not a real directory: {tree_root}"
            )
    for relative in tree_members(root, validated["trees"]):
        selected.add(_canonical_regular_file(root / relative))
    return sorted(selected, key=lambda path: str(path))


def capture_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Capture a stable runtime generation without writing it."""

    validated = _validated_selection(dict(selection))
    before = _discover_selected_paths(validated)
    records = [_stable_file_record(path) for path in before]
    after = _discover_selected_paths(validated)
    if before != after:
        raise RuntimeManifestError("critical runtime file set changed while hashing")
    return {
        "schema_version": SCHEMA_VERSION,
        "selection": validated,
        "files": records,
    }


def _validate_record(record: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "state"}:
        raise RuntimeManifestError(f"runtime manifest files[{index}] has an invalid structure")
    path = record["path"]
    digest = record["sha256"]
    state = record["state"]
    if not isinstance(path, str) or not Path(path).is_absolute() or str(Path(path)) != path:
        raise RuntimeManifestError(f"runtime manifest files[{index}].path is not canonical")
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise RuntimeManifestError(f"runtime manifest files[{index}].sha256 is invalid")
    if (
        not isinstance(state, dict)
        or set(state) != set(_STATE_KEYS)
        or not all(isinstance(state[key], int) for key in _STATE_KEYS)
    ):
        raise RuntimeManifestError(f"runtime manifest files[{index}].state is invalid")
    return record


def verify_capture(
    capture: Mapping[str, Any],
    *,
    tree_changes_are_drift: bool = False,
) -> dict[str, Any]:
    """Verify a captured generation, including the selection's exact set.

    ``tree_changes_are_drift`` belongs to the caller, because only the caller
    knows which operation this is. Checking an existing manifest against the
    tree is a verification: the record proves each path was a readable regular
    file when it was sealed, so a path now missing, replaced or unreadable is
    an ordinary edit and a running batch survives it. The same function also
    checks the baseline while sealing, and there the identical failure is a
    configuration error that must stop the launch -- nothing has been sealed
    yet, so there is no record to argue the path was ever fine. Defaulting to
    False keeps sealing fail-closed: a caller must ask for drift.
    """

    if not isinstance(capture, Mapping) or set(capture) != {
        "schema_version",
        "selection",
        "files",
    }:
        raise RuntimeManifestError("runtime generation capture has an invalid structure")
    if capture["schema_version"] != SCHEMA_VERSION:
        raise RuntimeManifestError(
            f"unsupported runtime generation schema: {capture['schema_version']!r}"
        )
    selection = _validated_selection(capture["selection"])
    raw_records = capture["files"]
    if not isinstance(raw_records, list) or not raw_records:
        raise RuntimeManifestError("runtime generation capture has no files")
    records = [
        _validate_record(record, index=index)
        for index, record in enumerate(raw_records)
    ]
    recorded_paths = [Path(record["path"]) for record in records]
    if recorded_paths != sorted(set(recorded_paths), key=lambda path: str(path)):
        raise RuntimeManifestError("runtime generation file records must be path-sorted and unique")
    # Nothing in this module decides whether a failure means the tree moved --
    # including the three comparisons below. A capture that changed between
    # being taken and being sealed is a launch that cannot be trusted. Only
    # _drift_context converts, and only when the caller enabled it.
    #
    # Everything above reads the manifest; everything below reads the working
    # tree. Past this line the manifest has already proved well-formed, and it
    # recorded every selected path as a readable regular file at seal time. So
    # a path that is now missing, a symlink, a directory, unreadable or
    # unopenable is not a broken record -- it is the tree having moved (an
    # ordinary `git checkout` or `rm`), which is drift.
    with _drift_context(enabled=tree_changes_are_drift):
        discovered_paths = _discover_selected_paths(selection)
        if discovered_paths != recorded_paths:
            missing = sorted(
                str(path) for path in set(recorded_paths) - set(discovered_paths)
            )
            added = sorted(
                str(path) for path in set(discovered_paths) - set(recorded_paths)
            )
            raise RuntimeManifestError(
                "critical runtime file set changed"
                f"; missing={missing or 'none'}; added={added or 'none'}"
            )
        for record, path in zip(records, discovered_paths, strict=True):
            current = _stable_file_record(path)
            if current["state"] != record["state"]:
                raise RuntimeManifestError(
                    f"critical runtime file state changed: {path}"
                )
            if current["sha256"] != record["sha256"]:
                raise RuntimeManifestError(
                    f"critical runtime file content changed: {path}"
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "file_count": len(records),
        "verified_at_utc": _utc_now(),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o640,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise RuntimeManifestError(f"could not atomically write runtime manifest {path}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return digest


def write_manifest(path: str | Path, capture: Mapping[str, Any]) -> str:
    """Verify a capture, add metadata, and atomically publish the manifest."""

    verification = verify_capture(capture)
    payload = dict(capture)
    payload["created_at_utc"] = verification["verified_at_utc"]
    return _atomic_write_json(Path(path), payload)


def create_manifest(
    *,
    repo_root: str | Path,
    output_path: str | Path,
    extra_files: Iterable[str | Path] = (),
) -> tuple[dict[str, Any], str]:
    selection = default_selection(repo_root, extra_files=extra_files)
    capture = capture_selection(selection)
    digest = write_manifest(output_path, capture)
    return capture, digest


def verify_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    manifest_path = Path(path)
    if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise RuntimeManifestError("expected runtime manifest SHA-256 is invalid")
    manifest_record, manifest_bytes = _stable_file_snapshot(
        manifest_path,
        retain_content=True,
    )
    assert manifest_bytes is not None
    observed_digest = manifest_record["sha256"]
    if observed_digest != expected_sha256:
        raise RuntimeManifestError(
            "runtime manifest content changed"
            f"; expected={expected_sha256}; observed={observed_digest}"
        )
    try:
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeManifestError(f"runtime manifest is invalid: {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "created_at_utc",
        "selection",
        "files",
    }:
        raise RuntimeManifestError("runtime manifest has an invalid top-level structure")
    created = payload.pop("created_at_utc")
    if not isinstance(created, str) or not created:
        raise RuntimeManifestError("runtime manifest created_at_utc is invalid")
    result = verify_capture(payload, tree_changes_are_drift=True)
    result.update(
        {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": observed_digest,
        }
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="atomically create a runtime manifest")
    create.add_argument("--repo-root", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--extra-file", action="append", default=[], type=Path)
    verify = subparsers.add_parser("verify", help="fail if the sealed generation changed")
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--expected-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "create":
            capture, digest = create_manifest(
                repo_root=args.repo_root,
                output_path=args.output,
                extra_files=args.extra_file,
            )
            result = {
                "file_count": len(capture["files"]),
                "manifest": str(args.output.resolve()),
                "manifest_sha256": digest,
                "status": "created",
            }
        else:
            result = verify_manifest(
                args.manifest,
                expected_sha256=args.expected_sha256,
            )
    except RuntimeSourceDriftError as exc:
        # Exit 3 is "the tree moved, the manifest is fine". The shell runner
        # survives this one and keeps the rollouts; it must still stop on the
        # exit 2 below, which covers a missing, unreadable, malformed or
        # self-inconsistent manifest. Collapsing both into one code is how a
        # corrupt manifest would come to read as a harmless edit.
        print(f"runtime source drift: {exc}", file=sys.stderr)
        return 3
    except RuntimeManifestError as exc:
        print(f"runtime manifest error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
