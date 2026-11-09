# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Copies of this checkout that a test may seal and then edit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from real2sim import runtime_manifest
from robosuite_env.contracts import provenance as provenance_contract
from tests._helpers.repo import REPO_ROOT


def covered_paths() -> list[str]:
    """Every file the runtime seal covers in this checkout, relative to it."""

    coverage = runtime_manifest.required_coverage()
    paths = set(coverage["files"])
    paths.update(runtime_manifest.tree_members(REPO_ROOT, coverage["trees"]))
    return sorted(paths)


def copy_covered_files(destination: Path) -> Path:
    """Copy every covered file into ``destination`` and return it."""

    for relative in covered_paths():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, target)
    return destination


def private_copy(pristine: Path, destination: Path) -> Path:
    """Hardlink ``pristine`` into ``destination``; edit it with :func:`replace`."""

    shutil.copytree(pristine, destination, copy_function=os.link)
    return destination


def replace(path: Path, data: bytes) -> None:
    """Rewrite ``path`` without touching the pristine copy it may be linked to."""

    path.unlink()
    path.write_bytes(data)


def seal(checkout: Path, run_root: Path, selection: dict | None = None) -> None:
    """Write the run seal the launcher captures for ``checkout``."""

    run_root.mkdir(parents=True, exist_ok=True)
    capture = runtime_manifest.capture_selection(
        selection or runtime_manifest.default_selection(checkout)
    )
    (run_root / provenance_contract.RUNTIME_SOURCE_MANIFEST_FILENAME).write_text(
        json.dumps(capture), encoding="utf-8"
    )


def renderer_code(checkout: Path, *, commit: str, dirty: bool = True) -> dict[str, Any]:
    """A renderer identity of ``checkout`` with its core files' records."""

    files = {}
    for relative in provenance_contract.CORE_RENDERER_SOURCE_PATHS:
        path = checkout / relative
        stat = path.stat()
        files[relative] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "state": {
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
                "mode": int(stat.st_mode),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "ctime_ns": int(stat.st_ctime_ns),
            },
        }
    return {
        "commit": commit,
        "dirty": dirty,
        "core_source_files": files,
        "tracked_diff": {
            "base_commit": commit,
            "sha256": hashlib.sha256(b"x" if dirty else b"").hexdigest(),
            "size_bytes": 1 if dirty else 0,
        },
        "gaussian_wrapping": gaussian_wrapping(checkout),
    }


#: The GaussianWrapping commit test runs record at launch and test renderers hold.
GAUSSIAN_WRAPPING_COMMIT = "a" * 40
GAUSSIAN_WRAPPING_GITLINK = "third_party/GaussianWrapping"


def gaussian_wrapping(checkout: Path, *, commit: str = GAUSSIAN_WRAPPING_COMMIT) -> dict[str, Any]:
    """A clean GaussianWrapping record of ``checkout`` at ``commit``."""

    return {"path": str(Path(checkout) / GAUSSIAN_WRAPPING_GITLINK), "commit": commit, "dirty": False}


def run_manifest(code: dict[str, Any], **launch: Any) -> dict[str, Any]:
    """A run manifest with ``code`` that recorded a clean GaussianWrapping at launch."""

    return {
        "code": code,
        "source_dependencies": [gaussian_wrapping(Path(code["path"]), **launch)],
    }
