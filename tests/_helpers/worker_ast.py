# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Static and behavioral contracts for the structured rollout runtime.

These tests deliberately inspect the worker without importing it: importing the
production worker parses CLI arguments and constructs MuJoCo state.  AST checks
let the clean-install test suite enforce package ownership and startup ordering
without requiring GPUs, MuJoCo, Blender, or a policy checkpoint.
"""
from __future__ import annotations
import ast
from dataclasses import fields as dataclass_fields
import os  # noqa: F401 -- test modules star-import these names from here
from pathlib import Path
import subprocess  # noqa: F401
import sys  # noqa: F401
from types import SimpleNamespace  # noqa: F401
from robosuite_env.rollout.render.overview import resolve_overview_render_plan  # noqa: F401
from robosuite_env.rollout.session import RolloutSession
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / 'robosuite_env'
SIM_PATH = PACKAGE_ROOT / 'sim.py'
WORKER_PATH = PACKAGE_ROOT / 'rollout' / 'worker.py'
SESSION_PATH = PACKAGE_ROOT / 'rollout' / 'session.py'
PREFLIGHT_PATH = PACKAGE_ROOT / 'rollout' / 'preflight.py'
SETUP_ARGS_PATH = PACKAGE_ROOT / 'rollout' / 'setup' / 'args.py'
BUILD_MODEL_PATH = PACKAGE_ROOT / 'rollout' / 'build' / 'model.py'
CLI_PATH = PACKAGE_ROOT / 'rollout' / 'cli.py'
RUNNER_PATH = REPO_ROOT / 'scripts' / 'run_policy_batch.sh'
LAUNCHER_PATH = REPO_ROOT / 'src' / 'real2sim' / 'reporting' / 'completion.py'
SIM_SOURCE = SIM_PATH.read_text(encoding='utf-8')
WORKER_SOURCE = WORKER_PATH.read_text(encoding='utf-8')
SESSION_SOURCE = SESSION_PATH.read_text(encoding='utf-8')
PREFLIGHT_SOURCE = PREFLIGHT_PATH.read_text(encoding='utf-8')
SETUP_ARGS_SOURCE = SETUP_ARGS_PATH.read_text(encoding='utf-8')
BUILD_MODEL_SOURCE = BUILD_MODEL_PATH.read_text(encoding='utf-8')
CLI_SOURCE = CLI_PATH.read_text(encoding='utf-8')
RUNNER_SOURCE = RUNNER_PATH.read_text(encoding='utf-8')
LAUNCHER_SOURCE = LAUNCHER_PATH.read_text(encoding='utf-8')
SIM_TREE = ast.parse(SIM_SOURCE, filename=str(SIM_PATH))
WORKER_TREE = ast.parse(WORKER_SOURCE, filename=str(WORKER_PATH))
SESSION_TREE = ast.parse(SESSION_SOURCE, filename=str(SESSION_PATH))
STAGE_FINALIZE_PATH = PACKAGE_ROOT / 'rollout' / 'stages' / 'finalize_outputs.py'
STAGE_FINALIZE_SOURCE = STAGE_FINALIZE_PATH.read_text(encoding='utf-8')
STAGE_FINALIZE_TREE = ast.parse(STAGE_FINALIZE_SOURCE, filename=str(STAGE_FINALIZE_PATH))
PREFLIGHT_TREE = ast.parse(PREFLIGHT_SOURCE, filename=str(PREFLIGHT_PATH))
SETUP_ARGS_TREE = ast.parse(SETUP_ARGS_SOURCE, filename=str(SETUP_ARGS_PATH))
BUILD_MODEL_TREE = ast.parse(BUILD_MODEL_SOURCE, filename=str(BUILD_MODEL_PATH))
CLI_TREE = ast.parse(CLI_SOURCE, filename=str(CLI_PATH))
LAUNCHER_TREE = ast.parse(LAUNCHER_SOURCE, filename=str(LAUNCHER_PATH))
_WORKER_SOURCE_FILES = tuple(sorted((path for path in (PACKAGE_ROOT / 'rollout').rglob('*.py') if path.name != '__init__.py' and path != CLI_PATH)))
_WORKER_SOURCES = {path: path.read_text(encoding='utf-8') for path in _WORKER_SOURCE_FILES}
_WORKER_TREES = {path: ast.parse(source, filename=str(path)) for path, source in _WORKER_SOURCES.items()}
STRICT_RUNTIME_OPTIONS = ('--strict-assets', '--official-run-manifest', '--official-scene-tag', '--official-worker-contract', '--gs-calibration-world-offset', '--gs-server-token')
STRUCTURED_RUNTIME_DIRECTORIES = ('contracts', 'policy', 'rendering', 'robots', 'rollout', 'tasks')

def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update((alias.name for alias in node.names))
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules

def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None

def _call_lines(tree: ast.AST, name: str) -> list[int]:
    return sorted((node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) and _call_name(node) == name))

def _string_keyed_dict(node: ast.Dict) -> dict[str, ast.expr]:
    result: dict[str, ast.expr] = {}
    assert len(node.keys) == len(node.values)
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            result[key.value] = value
    return result

def _argument_calls(*trees: ast.AST) -> dict[str, list[ast.Call]]:
    result: dict[str, list[ast.Call]] = {}
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) != 'add_argument':
                continue
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    if argument.value.startswith('--'):
                        result.setdefault(argument.value, []).append(node)
    return result

def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((item.value for item in call.keywords if item.arg == name), None)

def _shell_function(source: str, name: str) -> str:
    start = source.index(f'{name}() {{')
    end = source.index('\n}\n', start) + len('\n}')
    return source[start:end]

def _sim_thread_caps_definitions() -> str:
    """The runner's shared simulation thread caps, for tests that run its functions."""
    start = RUNNER_SOURCE.index('SIM_THREAD_VARIABLES=(')
    end = RUNNER_SOURCE.index('\n)\n', start) + len('\n)')
    return '\n'.join((RUNNER_SOURCE[start:end], _shell_function(RUNNER_SOURCE, 'export_sim_thread_caps'), _shell_function(RUNNER_SOURCE, 'sim_helper_python')))
_SESSION_SCRATCH_GLOBALS = frozenset({'key', '_key', 'g', 'act', '_', 'k', 's', 'c', 'exc', 'fd', 'name', 'binding', 'ph', 'p', 'q', 'h', 'u', 'x', 'f', 'line', 'row'})

def _session_field_names() -> set[str]:
    return {item.name for item in dataclass_fields(RolloutSession)}

def _undeclared_session_attribute_writes(trees: tuple[ast.AST, ...]) -> set[str]:
    stored = {node.attr for tree in trees for node in ast.walk(tree) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and (node.value.id == 'state') and isinstance(node.ctx, ast.Store)}
    return stored - _SESSION_SCRATCH_GLOBALS - _session_field_names()

__all__ = tuple(name for name in globals() if not name.startswith("__"))
