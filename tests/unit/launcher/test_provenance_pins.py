# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The robosuite pin is declared once, in pyproject, and locked in uv.lock."""

from __future__ import annotations

import configparser
import re
import subprocess
import tomllib
from pathlib import Path

from real2sim.locked_imports import locked_git_source

REPO_ROOT = Path(__file__).resolve().parents[3]


def _project_rev() -> str:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    source = project["tool"]["uv"]["sources"]["robosuite"]
    assert source["git"] == "https://github.com/ARISE-Initiative/robosuite"
    return source["rev"]


def test_lock_pins_the_project_robosuite_revision():
    locked = locked_git_source(REPO_ROOT / "uv.lock", "robosuite")
    assert locked.url == "https://github.com/ARISE-Initiative/robosuite"
    assert locked.subdirectory is None
    assert locked.commit == _project_rev()


def test_no_robosuite_checkout_competes_with_the_lock():
    # A gitlink would be a second, independently movable declaration and a
    # checkout an import could pick up instead of the installed distribution.
    gitmodules = configparser.ConfigParser()
    gitmodules.read(REPO_ROOT / ".gitmodules")
    assert not [
        section for section in gitmodules.sections() if "robosuite" in section.lower()
    ]
    tracked = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--stage", "--", "third_party/robosuite"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert tracked == ""


def test_provenance_document_names_the_same_revision():
    provenance = (REPO_ROOT / "PROVENANCE.md").read_text(encoding="utf-8")
    revisions = set(re.findall(r"\b[0-9a-f]{40}\b", provenance))
    assert revisions == {_project_rev()}
