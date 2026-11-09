# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""First-party code imports first-party packages from the installed distribution.

``real2sim``, ``robosuite_env`` and ``scene`` are declared packages, installed
editable in a checkout, so no tracked code puts a directory on ``sys.path`` and
no profile, runner script or launcher default hands its children a
``PYTHONPATH`` (issue #57). The allowlist below names the one remaining path
edit; the scan must find exactly that, so a new edit fails and so does the
allowlisted one disappearing (then the entry has to go too).

Tests are scanned like any other code. Their scoped shadows go through
``monkeypatch.syspath_prepend`` or ``monkeypatch.setattr``, which pytest
undoes, and are not ``sys.path`` edits. ``third_party/`` is not first-party.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]

#: (path, qualified function, operation) -> count. Each entry says why.
ALLOWED_SYS_PATH_EDITS = Counter(
    {
        # GaussianWrapping is not installable, and its top-level ``scene``
        # package clashes with ours; see the comment at the inserts.
        ("scene/gs_background_server.py", "_make_gpu_renderer.initialize", "sys.path.insert"): 3,
    }
)
# Blender scripts run in Blender's own interpreter and may need a path adapter
# of their own. The repository tracks none outside third_party/, so there is no
# entry for one; add it here, with its reason, together with such a script.

_PATH_MUTATORS = frozenset({"insert", "append", "extend", "remove", "pop", "clear"})
# Code paths whose PYTHONPATH settings reach a launched process. Tests build
# child environments with deliberate shadows and are not checked for this.
_RUNTIME_ROOTS = ("src/", "robosuite_env/", "scene/", "scripts/")


def _tracked(*patterns: str) -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", *patterns],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode()
    return sorted(
        name for name in listed.split("\0") if name and not name.startswith("third_party/")
    )


class _ImportRootEdits(ast.NodeVisitor):
    """Collect sys.path edits, site.addsitedir calls and PYTHONPATH settings."""

    def __init__(self) -> None:
        self.scope: list[str] = []
        self.sys_names: set[str] = set()
        self.path_names: set[str] = set()
        self.site_names: set[str] = set()
        self.addsitedir_names: set[str] = set()
        self.path_edits: list[tuple[str, str]] = []
        self.pythonpath_settings: list[tuple[str, int]] = []

    # -- name tracking ---------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound = alias.asname or alias.name
            if alias.name == "sys":
                self.sys_names.add(bound)
            elif alias.name == "site":
                self.site_names.add(bound)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            bound = alias.asname or alias.name
            if node.module == "sys" and alias.name == "path":
                self.path_names.add(bound)
            elif node.module == "site" and alias.name == "addsitedir":
                self.addsitedir_names.add(bound)

    def _scoped(self, node: ast.AST) -> None:
        self.scope.append(node.name)  # type: ignore[attr-defined]
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = _scoped
    visit_AsyncFunctionDef = _scoped
    visit_ClassDef = _scoped

    # -- detection -------------------------------------------------------
    def _is_sys_path(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.path_names
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "path"
            and isinstance(node.value, ast.Name)
            and node.value.id in self.sys_names
        )

    def _record(self, operation: str) -> None:
        self.path_edits.append((".".join(self.scope) or "<module>", operation))

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _PATH_MUTATORS and self._is_sys_path(func.value):
            self._record(f"sys.path.{func.attr}")
        elif (
            isinstance(func, ast.Attribute)
            and func.attr == "addsitedir"
            and isinstance(func.value, ast.Name)
            and func.value.id in self.site_names
        ) or (isinstance(func, ast.Name) and func.id in self.addsitedir_names):
            self._record("site.addsitedir")
        elif (
            isinstance(func, ast.Name)
            and func.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in self.sys_names
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "path"
        ):
            self._record("setattr(sys, 'path')")
        elif (
            isinstance(func, ast.Attribute)
            and func.attr in {"putenv", "setdefault"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "PYTHONPATH"
        ):
            self.pythonpath_settings.append((func.attr, node.lineno))
        # dict(os.environ, PYTHONPATH=...), os.environ.update(PYTHONPATH=...)
        if any(keyword.arg == "PYTHONPATH" for keyword in node.keywords):
            self.pythonpath_settings.append(("PYTHONPATH=...", node.lineno))
        self.generic_visit(node)

    def _check_target(self, target: ast.AST) -> None:
        if self._is_sys_path(target):
            self._record("sys.path =")
        elif isinstance(target, ast.Subscript) and self._is_sys_path(target.value):
            self._record("sys.path[...] =")
        elif (
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "PYTHONPATH"
        ):
            self.pythonpath_settings.append(("[PYTHONPATH] =", target.lineno))

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_target(target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_target(node.target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_target(node.target)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key in node.keys:
            if isinstance(key, ast.Constant) and key.value == "PYTHONPATH":
                self.pythonpath_settings.append(("{PYTHONPATH: ...}", key.lineno))
        self.generic_visit(node)


def _scan(source: str, filename: str) -> _ImportRootEdits:
    visitor = _ImportRootEdits()
    visitor.visit(ast.parse(source, filename=filename))
    return visitor


def _python_sources() -> list[str]:
    return _tracked("*.py")


def test_the_scanner_sees_every_form_of_import_root_edit() -> None:
    source = """
import os
import site
import sys
import sys as other_sys
from site import addsitedir
from sys import path as search_path

sys.path.insert(0, "a")
other_sys.path.append("b")
search_path.extend(["c"])
sys.path[:0] = ["d"]
sys.path += ["e"]
sys.path = ["f"]
site.addsitedir("g")
addsitedir("h")
setattr(sys, "path", [])

def launch():
    environment = {"PYTHONPATH": "src"}
    environment["PYTHONPATH"] = "src"
    os.environ.setdefault("PYTHONPATH", "src")
    os.putenv("PYTHONPATH", "src")
    child = dict(os.environ, PYTHONPATH="src")
    os.environ.update(PYTHONPATH="src")

def harmless(monkeypatch):
    monkeypatch.syspath_prepend("x")
    monkeypatch.setattr(sys, "path", list(sys.path))
    print(sys.path[0], "sys.path.insert(0, 'string, not code')")
    os.environ.pop("PYTHONPATH", None)
"""
    found = _scan(source, "<fixture>")
    assert [operation for _, operation in found.path_edits] == [
        "sys.path.insert",
        "sys.path.append",
        "sys.path.extend",
        "sys.path[...] =",
        "sys.path =",
        "sys.path =",
        "site.addsitedir",
        "site.addsitedir",
        "setattr(sys, 'path')",
    ]
    assert {scope for scope, _ in found.path_edits} == {"<module>"}
    assert [kind for kind, _ in found.pythonpath_settings] == [
        "{PYTHONPATH: ...}",
        "[PYTHONPATH] =",
        "setdefault",
        "putenv",
        "PYTHONPATH=...",
        "PYTHONPATH=...",
    ]


def test_only_the_gaussian_wrapping_adapter_edits_sys_path() -> None:
    found: Counter[tuple[str, str, str]] = Counter()
    for relative in _python_sources():
        path = REPO_ROOT / relative
        for scope, operation in _scan(path.read_text(encoding="utf-8"), relative).path_edits:
            found[(relative, scope, operation)] += 1
    unexpected = found - ALLOWED_SYS_PATH_EDITS
    missing = ALLOWED_SYS_PATH_EDITS - found
    assert not unexpected, (
        "sys.path edits outside the allowlist; import the installed package instead "
        f"(or add an entry with its reason): {dict(unexpected)}"
    )
    assert not missing, (
        f"allowlisted sys.path edits are gone; remove their entries: {dict(missing)}"
    )


def test_runtime_code_sets_no_pythonpath() -> None:
    settings = []
    for relative in _python_sources():
        if not relative.startswith(_RUNTIME_ROOTS):
            continue
        path = REPO_ROOT / relative
        for kind, line in _scan(path.read_text(encoding="utf-8"), relative).pythonpath_settings:
            settings.append(f"{relative}:{line} {kind}")
    assert not settings, settings


_SHELL_PYTHONPATH = re.compile(r"(?<![\w$])(?:export\s+)?PYTHONPATH\s*\+?=")


def test_shell_scripts_set_no_pythonpath() -> None:
    scripts = _tracked("*.sh")
    assert "scripts/run_policy_batch.sh" in scripts
    settings = [
        f"{relative}:{number}: {line.strip()}"
        for relative in scripts
        for number, line in enumerate(
            (REPO_ROOT / relative).read_text(encoding="utf-8").splitlines(), start=1
        )
        if _SHELL_PYTHONPATH.search(line)
    ]
    assert not settings, settings


def test_profiles_pass_no_pythonpath_to_their_children() -> None:
    profiles = _tracked("config/*.yaml", "config/*.yml")
    assert "config/release/fr3/flat/a_room_cup_cup1trim.yaml" in profiles
    offending = [
        relative
        for relative in profiles
        if "PYTHONPATH" in (REPO_ROOT / relative).read_text(encoding="utf-8")
    ]
    assert not offending, offending


def test_documented_profiles_pass_no_pythonpath() -> None:
    offending = []
    for relative in _tracked("README.md", "docs/*.md"):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for block in re.findall(r"```ya?ml\n(.*?)```", text, flags=re.DOTALL):
            if "PYTHONPATH" in block:
                offending.append(relative)
    assert not offending, offending


def test_no_path_configuration_files_are_tracked() -> None:
    assert _tracked("*.pth") == []
