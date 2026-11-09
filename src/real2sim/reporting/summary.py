# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Summary subprocess orchestration."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from real2sim._bootstrap import REPO_ROOT
from real2sim.config.paths import SUMMARY_TOOL
from real2sim.reporting.completion import rollout_completion_errors

def _summary_paths(run_root: Path) -> dict[str, Path]:
    prefix = "rollout_success_summary"
    return {
        "json": run_root / f"{prefix}.json",
        "csv": run_root / f"{prefix}.csv",
        "png": run_root / f"{prefix}.png",
    }

def summarize_run(plan: Mapping[str, Any]) -> tuple[int, dict[str, Any] | None, dict[str, Path]]:
    run_root = Path(plan["run_root"])
    paths = _summary_paths(run_root)
    python_path = str(plan["resolved"]["policy"]["python"])
    completed = subprocess.run(
        [
            python_path,
            str(SUMMARY_TOOL),
            str(run_root),
            "--out-dir",
            str(run_root),
            "--prefix",
            "rollout_success_summary",
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    if completed.returncode != 0 or not paths["json"].is_file():
        return completed.returncode or 1, None, paths
    try:
        summary = json.loads(paths["json"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 1, None, paths
    completion_errors = rollout_completion_errors(plan)
    if completion_errors:
        print("rollout completion validation failed:", file=sys.stderr)
        for error in completion_errors:
            print(f"  - {error}", file=sys.stderr)
        return 1, summary, paths
    return 0, summary, paths
