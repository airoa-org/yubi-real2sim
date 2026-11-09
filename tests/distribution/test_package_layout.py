# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Every first-party Python directory is a package that the wheel ships.

The project sets ``namespaces = false``, so a directory without ``__init__.py``
is silently left out of the wheel even though checkout imports still work.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import tomllib

from setuptools import find_packages


REPO_ROOT = Path(__file__).resolve().parents[2]


def _declared_packages() -> set[str]:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    find = config["tool"]["setuptools"]["packages"]["find"]
    assert find.get("namespaces") is False
    packages: set[str] = set()
    for where in find["where"]:
        packages.update(find_packages(where=str(REPO_ROOT / where), include=find["include"]))
    return packages


def _tracked_module_directories() -> set[str]:
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", "src/real2sim", "robosuite_env", "scene"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode().split("\0")
    directories = set()
    for name in tracked:
        if not name.endswith(".py"):
            continue
        parts = Path(name).parent.parts
        if parts[0] == "src":
            parts = parts[1:]
        directories.add(".".join(parts))
    return directories


def test_every_tracked_python_directory_is_a_shipped_package() -> None:
    missing = sorted(_tracked_module_directories() - _declared_packages())
    assert not missing, f"Python directories the wheel would omit: {missing}"


def test_installed_rerender_dependencies_are_shipped() -> None:
    packages = _declared_packages()
    assert {"robosuite_env.rerender", "robosuite_env.tools"} <= packages
