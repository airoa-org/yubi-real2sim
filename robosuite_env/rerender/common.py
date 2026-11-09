# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Shared constants, dataclasses, and small helpers for post-hoc rerendering.

Split out of ``robosuite_env/tools/rerender_third_person.py``; every name is
re-exported unchanged from :mod:`robosuite_env.rerender`.
"""


from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from real2sim.config.task_objects import validate_task_objects
from robosuite_env.gs_launch import GS_SERVER_SCRIPT


REPO_ROOT = Path(__file__).resolve().parents[2]


GRID_TOOL = REPO_ROOT / "robosuite_env" / "tools" / "make_rollout_grid_video.py"
#: Module the post-hoc renderer runs (``python -m``) to build scene grids.
GRID_TOOL_MODULE = "robosuite_env.tools.make_rollout_grid_video"


# Single-sourced from robosuite_env.gs_launch, which also owns the argv that
# starts this executable, so the path cannot drift between the two launchers.
GS_SERVER = GS_SERVER_SCRIPT


GRID_TEMPLATES = {
    10: REPO_ROOT
    / "config"
    / "render_templates"
    / "rollout_grid_clear_headers_4x3.json",
    30: REPO_ROOT
    / "config"
    / "render_templates"
    / "rollout_grid_clear_headers_6x6.json",
    100: REPO_ROOT
    / "config"
    / "render_templates"
    / "rollout_grid_clear_headers_10x12.json",
}


# Canonical definitions live in :mod:`robosuite_env.contracts.provenance`;
# the underscore names below are compatibility aliases for this package's
# historical re-exports.
from robosuite_env.contracts.provenance import (  # noqa: E402
    BATCH_POLICY_LINEAGE_FILENAMES,
    CORE_RENDERER_SOURCE_PATHS,
)


_BATCH_POLICY_LINEAGE_FILENAMES = BATCH_POLICY_LINEAGE_FILENAMES


_CORE_RENDERER_SOURCE_PATHS = CORE_RENDERER_SOURCE_PATHS


class RerenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceRun:
    run_dir: Path
    environment: dict[str, str]
    argv: list[str]
    summary: dict[str, Any]
    state_rows: list[dict[str, Any]]
    expected_frames: int
    official: bool
    official_authority: dict[str, Any] | None
    source_records: dict[str, Any]
    renderer_code: dict[str, Any]


@dataclass(frozen=True)
class GpuBinding:
    """One allocation-local CUDA slot bound to its host EGL device."""

    allocation_slot: int
    cuda_token: str
    egl_physical_index: int


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


_GS_OFF_VALUES = {"", "none", "off", "false", "0", "no"}


def _resolved_recorded_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RerenderError(f"{label} is missing")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    try:
        return candidate.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise RerenderError(f"{label} is invalid: {value!r}") from exc


def _canonical_video_path(run_dir: Path) -> Path:
    """Return the replay video path rooted in the selected live run tree."""

    return (run_dir / f"{run_dir.parent.name}_{run_dir.name}.mp4").resolve()


def _last_option_value(argv: list[str], option: str) -> str | None:
    """Return the final CLI value for ``--option value`` or ``--option=value``."""

    value: str | None = None
    index = 0
    equals_prefix = f"{option}="
    while index < len(argv):
        token = argv[index]
        if token == option:
            if index + 1 >= len(argv):
                raise RerenderError(f"{option} is missing its value")
            value = argv[index + 1]
            index += 2
            continue
        if token.startswith(equals_prefix):
            value = token[len(equals_prefix) :]
            if not value:
                raise RerenderError(f"{option} is missing its value")
        index += 1
    return value


def _rollout_expected_frames(
    argv: list[str],
    summary: Any,
    configured_frames: int,
    *,
    label: str,
) -> int:
    """Return the exact frame count this one rollout was allowed to write.

    Every rollout writes ``--frames`` frames. Validate any recorded task input.
    """

    objects = _last_option_value(argv, "--objects")
    if objects is not None:
        try:
            validate_task_objects(objects)
        except ValueError as exc:
            raise RerenderError(f"{label}: {exc}") from exc
    return configured_frames


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RerenderError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RerenderError(f"{label} must contain a JSON object: {path}")
    return value
