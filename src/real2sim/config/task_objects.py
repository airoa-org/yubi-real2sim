# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Shared validation for named task presets and cup/plate position files."""

from pathlib import Path

SUPPORTED_TASKS = ("cup_plate", "box_pack")


def validate_task_objects(value: str, *, require_file: bool = False) -> None:
    """Reject unknown inputs; JSON content checks remain in preflight."""
    if value in SUPPORTED_TASKS:
        return
    if value.endswith(".json") and (not require_file or Path(value).is_file()):
        return
    choices = ", ".join(SUPPORTED_TASKS)
    raise ValueError(f"unknown task.objects {value!r}; expected {choices}, or a positions JSON")
