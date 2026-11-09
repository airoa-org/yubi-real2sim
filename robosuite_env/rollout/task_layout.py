# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Provenance of the hand-authored layout a container or box rollout used.

The environment records the file it validated and the digest of those exact
bytes (``env.task_layout_record``, see ``tasks/layout_selection.py``). The run
summary names that record only after re-resolving the selection the way every
other consumer does -- the worker flag, then the harness-only override, then
the default -- and checking both agree. A record the
selection no longer names, or a file that changed after it was loaded, is
refused: its digest would describe a scene this rollout never had.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from robosuite_env.contracts.provenance import sha256_file
from robosuite_env.tasks.layout_selection import (
    BOX_PACK_TASK,
    LAYOUT_PARAMETERS,
    layout_sweep_offset_mm,
    select_task_layout,
)


def loaded_task_layout(state: Any) -> dict[str, Any] | None:
    """Return the layout provenance for this rollout, or ``None`` for other tasks."""

    task = str(getattr(state.A, "objects", "") or "")
    if task not in LAYOUT_PARAMETERS:
        return None
    env = getattr(state, "env", None)
    loaded = getattr(env, "task_layout_record", None)
    if not isinstance(loaded, Mapping) or not {"path", "sha256", "scene"} <= set(loaded):
        raise RuntimeError(
            f"the {task} environment recorded no layout it loaded; refusing to "
            "name one in this rollout's provenance"
        )
    parameter = LAYOUT_PARAMETERS[task]
    configured = str(getattr(state.A, parameter, "") or "").strip()
    selection = select_task_layout(task, configured, robot=getattr(state, "ROBOT", None))
    path = os.path.realpath(selection.path)
    if path != str(loaded["path"]):
        raise RuntimeError(
            f"the {task} layout selection names {path}, but the environment "
            f"loaded {loaded['path']}; refusing to record either as this "
            "rollout's provenance"
        )
    digest = sha256_file(Path(path))
    if digest != str(loaded["sha256"]):
        raise RuntimeError(
            f"the {task} layout {path} changed after the environment loaded it "
            f"(sha256 {str(loaded['sha256'])[:12]} at load, {digest[:12]} now); "
            "refusing to record its digest as this rollout's provenance"
        )
    record = {
        "task": task,
        "parameter": parameter,
        "configured": configured or None,
        "selected_by": selection.selected_by,
        "path": path,
        "sha256": digest,
        "scene": str(loaded["scene"]),
        "running_scene": getattr(env, "gw_scene_tag", None),
    }
    layout_index = getattr(state.A, "box_pack_layout_index", None)
    if task != BOX_PACK_TASK or layout_index is None:
        return record
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("layout root is not an object")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        reason = f"layout metadata could not be read: {type(exc).__name__}: {exc}"
        offsets = {
            "pen_offset_mm": (None, reason),
            "box_offset_mm": (None, reason),
        }
    else:
        offsets = {
            key: layout_sweep_offset_mm(payload, key)
            for key in ("pen_offset_mm", "box_offset_mm")
        }
    record.update(
        {
            "index": int(layout_index),
            "pen_offset_mm": offsets["pen_offset_mm"][0],
            "box_offset_mm": offsets["box_offset_mm"][0],
            "pen_offset_mm_reason": offsets["pen_offset_mm"][1],
            "box_offset_mm_reason": offsets["box_offset_mm"][1],
        }
    )
    return record


__all__ = ["loaded_task_layout"]
