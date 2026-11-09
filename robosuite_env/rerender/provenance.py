# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Git/source identity capture and immutable file-record assertions."""


from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

# The canonical implementations live in robosuite_env.contracts.provenance.
# They are bound under the historical underscore spelling because that is the
# name robosuite_env.rerender re-exports (and that rerender_third_person,
# real2sim.launcher, and the rerender tests import); there is no second
# definition of either function here.
from robosuite_env.contracts.provenance import (
    file_state as _file_state,
    gaussian_wrapping_identity,
    git_tracked_diff_identity,
    sha256_file as _sha256_file,
)
from robosuite_env.rerender.common import (
    REPO_ROOT,
    RerenderError,
    _COMMIT_RE,
    _CORE_RENDERER_SOURCE_PATHS,
)


def _file_record(path: Path, *, include_state: bool) -> dict[str, Any]:
    """Hash a stable regular file and capture its canonical identity."""

    path = path.resolve()
    try:
        before = _file_state(path)
        digest = _sha256_file(path)
        after = _file_state(path)
    except OSError as exc:
        raise RerenderError(f"cannot fingerprint {path}: {exc}") from exc
    if before != after:
        raise RerenderError(f"file changed while it was fingerprinted: {path}")
    record: dict[str, Any] = {"path": str(path), "sha256": digest}
    if include_state:
        record["state"] = after
    return record


def _git_head_commit() -> str:
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RerenderError(f"cannot identify rerender source checkout: {exc}") from exc
    commit = commit_result.stdout.strip()
    if _COMMIT_RE.fullmatch(commit) is None:
        raise RerenderError(f"git returned an invalid commit id: {commit!r}")
    return commit


def _git_tracked_diff_identity(commit: str) -> dict[str, Any]:
    """Hash the exact tracked patch from ``commit`` to index/worktree."""

    try:
        return git_tracked_diff_identity(commit)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RerenderError(f"cannot fingerprint tracked source diff: {exc}") from exc


def _core_renderer_source_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative_path in _CORE_RENDERER_SOURCE_PATHS:
        path = REPO_ROOT / relative_path
        if not path.is_file():
            raise RerenderError(f"core renderer source does not exist: {path}")
        records[relative_path] = _file_record(path, include_state=True)
    return records


def _git_code_identity() -> dict[str, Any]:
    """Bind commit, dirty state, exact core bytes, the tracked patch, and GaussianWrapping."""

    commit = _git_head_commit()
    first_diff = _git_tracked_diff_identity(commit)
    source_files = _core_renderer_source_records()
    second_diff = _git_tracked_diff_identity(commit)
    if first_diff != second_diff or _git_head_commit() != commit:
        raise RerenderError("tracked renderer checkout changed while fingerprinted")
    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RerenderError(f"cannot identify rerender source checkout: {exc}") from exc
    return {
        "commit": commit,
        "dirty": bool(status_result.stdout),
        "core_source_files": source_files,
        "tracked_diff": second_diff,
        "gaussian_wrapping": gaussian_wrapping_identity(REPO_ROOT),
    }


def _warn_source_moved(message: str) -> None:
    """Report that a source moved, without refusing to draw the video.

    A source file may change while a run is in progress; the mismatch is
    reported on stderr and the render continues. Only a malformed identity or
    an unreadable file is fatal, because those mean the check could not be
    made at all.
    """

    print(f"WARNING: {message}; rendering anyway", file=sys.stderr)


def _assert_record_current(
    record: dict[str, Any], *, label: str, verify_hash: bool = False
) -> bool:
    """Recheck one source record; ``False`` (with a warning) when it moved."""

    path = Path(record["path"])
    try:
        if "state" in record and _file_state(path) != record["state"]:
            _warn_source_moved(f"{label} changed after source validation: {path}")
            return False
        if verify_hash and _sha256_file(path) != record["sha256"]:
            _warn_source_moved(f"{label} changed after source validation: {path}")
            return False
    except OSError as exc:
        raise RerenderError(f"cannot recheck {label} {path}: {exc}") from exc
    return True


def _assert_renderer_code_current(identity: dict[str, Any]) -> bool:
    """Recheck renderer code against the pre-GPU source snapshot.

    A malformed identity or an unreadable file fails. A change is reported
    and returned as ``False``: the video is still written, and the
    provenance records that its checkout changed while it rendered.
    """

    commit = identity.get("commit")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise RerenderError("renderer identity has no valid commit")
    unchanged = True
    if _git_head_commit() != commit:
        unchanged = False
        _warn_source_moved(
            "renderer checkout commit changed after source validation"
        )

    source_files = identity.get("core_source_files")
    if not isinstance(source_files, dict) or set(source_files) != set(
        _CORE_RENDERER_SOURCE_PATHS
    ):
        raise RerenderError("renderer identity has no exact core source file set")
    for relative_path in _CORE_RENDERER_SOURCE_PATHS:
        record = source_files[relative_path]
        expected_path = (REPO_ROOT / relative_path).resolve()
        if not isinstance(record, dict) or record.get("path") != str(expected_path):
            raise RerenderError(
                f"renderer source identity has an invalid path for {relative_path}"
            )
        unchanged = (
            _assert_record_current(
                record,
                label=f"renderer source {relative_path}",
                verify_hash=True,
            )
            and unchanged
        )

    gaussian_wrapping = identity.get("gaussian_wrapping")
    if not isinstance(gaussian_wrapping, dict):
        raise RerenderError("renderer identity has no GaussianWrapping record")
    current_gaussian_wrapping = gaussian_wrapping_identity(REPO_ROOT)
    if current_gaussian_wrapping != gaussian_wrapping:
        unchanged = False
        _warn_source_moved(
            "GaussianWrapping checkout changed after source validation: "
            f"expected {gaussian_wrapping!r}, observed {current_gaussian_wrapping!r}"
        )

    expected_diff = identity.get("tracked_diff")
    if not isinstance(expected_diff, dict):
        raise RerenderError("renderer identity has no tracked source diff")
    current_diff = _git_tracked_diff_identity(commit)
    if expected_diff != current_diff:
        unchanged = False
        _warn_source_moved(
            "tracked source diff changed after source validation: "
            f"expected {expected_diff!r}, observed {current_diff!r}"
        )
    return unchanged
