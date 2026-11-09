# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Redirect an initial-pose path from the previous layout.

Initial poses moved from ``config/*_initial_arm_joints.json`` to
``assets/initial_poses/<robot>/<scene>/<name>.json``.
``assets/initial_poses/legacy_paths.json`` maps every old repository path to
its new one; a profile or a worker command line that still spells an old path
is pointed at the moved file, which holds the same joint values.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..paths import REPOSITORY_ROOT

LEGACY_PATHS_FILE = REPOSITORY_ROOT / "assets" / "initial_poses" / "legacy_paths.json"


def redirect_legacy_pose_path(path: str | None) -> str | None:
    """Return the moved file for an old pose path, else ``path`` unchanged."""

    if not path or Path(path).exists():
        return path
    mapping = json.loads(LEGACY_PATHS_FILE.read_text(encoding="utf-8"))["paths"]
    text = str(path).replace("\\", "/")
    for old, new in mapping.items():
        if text == old or text.endswith("/" + old):
            return str(REPOSITORY_ROOT / new)
    return path
