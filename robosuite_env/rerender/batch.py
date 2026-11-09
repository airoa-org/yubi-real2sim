# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Batch-level finalization: wrist insets, outputs.csv, and scene grids."""


from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from robosuite_env.rollout.replay_policy_insets import (
    ReplayPolicyInsetError,
    load_replay_policy_insets,
)

from robosuite_env.rerender.common import (
    GRID_TEMPLATES,
    GRID_TOOL_MODULE,
    REPO_ROOT,
    RerenderError,
    _canonical_video_path,
    _last_option_value,
)
from robosuite_env.rerender.replay_argv import parse_run_command
from robosuite_env.rerender.sources import _load_state_rows


def resolve_wrist_insets(
    run_dir: Path,
    requested: str,
    *,
    generation_summary: dict[str, Any] | None = None,
    state_rows: list[dict[str, Any]] | None = None,
) -> tuple[str, Path | None, int, str | None]:
    """Validate the requested saved/off wrist mode before MuJoCo or EGL."""

    if requested == "off":
        return "off", None, 0, None
    if requested != "saved":
        raise RerenderError(f"unknown wrist inset mode: {requested}")

    # With RF overlap, a newly captured display observation can precede the
    # active chunk's obs_log recorded in the same state row. Until that log
    # contract carries the exact displayed pair, fail closed. Bind both the
    # generation summary and recorded command so a missing/defaulted summary
    # field cannot silently select the wrong schedule.
    try:
        _, recorded_argv = parse_run_command(run_dir)
        command_value = _last_option_value(recorded_argv, "--rf-overlap")
        alias_value = _last_option_value(recorded_argv, "--action-overlap")
        if command_value is not None and alias_value is not None:
            if int(command_value) != int(alias_value):
                raise ValueError(
                    "--rf-overlap and --action-overlap disagree"
                )
        recorded_overlap = int(
            command_value if command_value is not None else alias_value or 0
        )
    except (OSError, ValueError, TypeError, IndexError, RerenderError) as exc:
        raise RerenderError(
            f"{run_dir}: cannot verify recorded rf_overlap: {exc}"
        ) from exc

    generation_summary_path = run_dir / "sim_summary.json"
    if generation_summary is not None or generation_summary_path.is_file():
        try:
            summary = (
                generation_summary
                if generation_summary is not None
                else json.loads(generation_summary_path.read_text(encoding="utf-8"))
            )
            if "rf_overlap" not in summary:
                raise ValueError("generation summary has no rf_overlap field")
            raw_overlap = summary["rf_overlap"]
            if isinstance(raw_overlap, bool):
                raise ValueError("generation summary rf_overlap must be an integer")
            overlap = int(raw_overlap)
            if overlap != recorded_overlap:
                raise ValueError(
                    f"generation summary rf_overlap {overlap} differs from "
                    f"recorded command {recorded_overlap}"
                )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            overlap_error = f"cannot verify generation rf_overlap: {exc}"
        else:
            overlap_error = (
                "saved inset replay is not exact when generation rf_overlap "
                f"is {overlap}"
                if overlap != 0
                else None
            )
        if overlap_error is not None:
            raise RerenderError(f"{run_dir}: {overlap_error}")
    else:
        overlap_error = (
            "saved inset replay is not exact when generation rf_overlap "
            f"is {recorded_overlap}"
            if recorded_overlap != 0
            else None
        )
        if overlap_error is not None:
            raise RerenderError(f"{run_dir}: {overlap_error}")

    sim_log = run_dir / "logs" / "sim_log.jsonl"
    policy_obs_dir = run_dir / "logs" / "policy_obs"
    try:
        schedule = load_replay_policy_insets(
            state_rows if state_rows is not None else _load_state_rows(sim_log),
            policy_obs_dir,
        )
    except (RerenderError, ReplayPolicyInsetError) as exc:
        raise RerenderError(
            f"{run_dir}: saved wrist inset validation failed: {exc}"
        ) from exc
    return "saved", policy_obs_dir.resolve(), schedule.unique_pairs, None


def update_outputs_csv_after_rerender(
    run_root: Path,
    runs: list[Path],
) -> bool:
    """Point selected batch rows at their validated post-hoc artifacts.

    ``run_policy_batch.sh`` writes ``outputs.csv`` before a separately invoked
    post-hoc pass.  Without this update the row keeps pointing at the live
    renderer's legacy ``run_XX/video_validation.json`` path even though the
    validated derivative lives under ``run_XX/rerender``.  Preserve every
    unrelated row and column, and replace the CSV atomically only after all
    selected validation files exist.
    """

    outputs_path = run_root.resolve() / "outputs.csv"
    if not outputs_path.is_file():
        return False

    selected: dict[tuple[str, str], Path] = {}
    for raw_run_dir in runs:
        run_dir = raw_run_dir.resolve()
        try:
            relative = run_dir.relative_to(run_root.resolve())
        except ValueError as exc:
            raise RerenderError(
                f"selected run is outside the batch root: {run_dir}"
            ) from exc
        if len(relative.parts) != 2:
            raise RerenderError(
                f"selected run must be <scene>/run_NN below the batch root: "
                f"{run_dir}"
            )
        key = (relative.parts[0], relative.parts[1])
        if key in selected:
            raise RerenderError(f"duplicate selected rollout: {key!r}")
        validation_path = run_dir / "rerender" / "video_validation.json"
        if not validation_path.is_file() or validation_path.stat().st_size <= 0:
            raise RerenderError(
                f"validated rerender report is missing or empty: {validation_path}"
            )
        selected[key] = validation_path.resolve()

    try:
        with outputs_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            required = {"scene", "run", "video", "video_validation"}
            if fieldnames is None or not required.issubset(fieldnames):
                raise RerenderError(
                    f"{outputs_path} is missing required columns: "
                    f"{sorted(required)}"
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        if isinstance(exc, RerenderError):
            raise
        raise RerenderError(f"cannot read {outputs_path}: {exc}") from exc

    matched: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("scene") or ""), str(row.get("run") or ""))
        validation_path = selected.get(key)
        if validation_path is None:
            continue
        if key in matched:
            raise RerenderError(
                f"{outputs_path} contains duplicate selected rollout {key!r}"
            )
        run_dir = run_root.resolve() / key[0] / key[1]
        row["video"] = str(_canonical_video_path(run_dir).resolve())
        row["video_validation"] = str(validation_path)
        matched.add(key)

    missing = sorted(set(selected) - matched)
    if missing:
        raise RerenderError(
            f"{outputs_path} has no row for selected rollout(s): {missing!r}"
        )

    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=outputs_path.parent,
            prefix=f".{outputs_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_name = stream.name
            writer = csv.DictWriter(
                stream,
                fieldnames=fieldnames,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, outputs_path)
    except (OSError, csv.Error) as exc:
        if temp_name is not None:
            try:
                Path(temp_name).unlink()
            except OSError:
                pass
        raise RerenderError(f"cannot update {outputs_path}: {exc}") from exc
    return True


def _recorded_scene_dir(run_dir: Path, argv: list[str]) -> Path:
    # The worker's argparse keeps the last value and accepts --opt=value, so
    # the GS server must be started for that same scene.
    raw_path = _last_option_value(argv, "--gs-background")
    if not raw_path:
        raise RerenderError(
            f"{run_dir}: recorded command has no --gs-background value"
        )
    scene_dir = Path(raw_path)
    if not scene_dir.is_absolute():
        scene_dir = REPO_ROOT / scene_dir
    return scene_dir.resolve()


def build_scene_grids(
    run_root: Path,
    runs: list[Path],
    expected_frames_by_scene: dict[str, int],
    grid_template: Path | None = None,
    ragged_scenes: frozenset[str] | set[str] | tuple[str, ...] = (),
) -> list[str]:
    """Build each supported per-scene comparison grid.

    The checked-in templates intentionally support 10-, 30-, and 100-rollout
    comparisons. The exact selected run names are forwarded to the grid tool,
    so a 10-run subset cannot accidentally include stale videos from other
    runs below the same batch root. Other cardinalities remain valid rerender
    results; they just do not have a canonical grid layout. An explicit
    ``grid_template`` replaces the count-based mapping for every scene and
    must exist.
    """

    if grid_template is not None:
        grid_template = Path(grid_template).resolve()
        if not grid_template.is_file():
            raise RerenderError(
                f"--grid-template file does not exist: {grid_template}"
            )

    runs_by_scene: dict[str, list[Path]] = {}
    for run_dir in runs:
        runs_by_scene.setdefault(run_dir.parent.name, []).append(run_dir)

    failures: list[str] = []
    multiple_scenes = len(runs_by_scene) > 1
    for scene_tag, scene_runs in runs_by_scene.items():
        def run_sort_key(path: Path) -> tuple[int, str]:
            match = re.fullmatch(r"run_(\d+)", path.name)
            return (
                int(match.group(1)) if match is not None else sys.maxsize,
                path.name,
            )

        scene_runs = sorted(scene_runs, key=run_sort_key)
        repeat_count = len(scene_runs)
        if scene_tag in ragged_scenes:
            print(
                f"[rerender] grid skipped for {scene_tag}: "
                "rollouts have per-rollout lengths",
                flush=True,
            )
            continue
        expected_frames = expected_frames_by_scene.get(scene_tag)
        if expected_frames is None or expected_frames <= 0:
            failures.append(f"{scene_tag}: no valid configured frame count")
            continue
        template = (
            grid_template
            if grid_template is not None
            else GRID_TEMPLATES.get(repeat_count)
        )
        if template is None:
            print(
                f"[rerender] grid skipped for {scene_tag}: no checked-in "
                f"template for {repeat_count} runs",
                flush=True,
            )
            continue

        # Match scripts/run_policy_batch.sh: a single scene publishes at the
        # batch root; multi-scene batches keep each grid under its scene tag.
        grid_root = run_root / scene_tag if multiple_scenes else run_root
        grid_root.mkdir(parents=True, exist_ok=True)
        output = grid_root / f"rollout_grid_{repeat_count}.mp4"
        stdout_path = grid_root / f"rollout_grid_{repeat_count}_stdout.json"
        command = [
            sys.executable,
            "-m",
            GRID_TOOL_MODULE,
            str(run_root),
            "--output",
            str(output),
            "--title",
            scene_tag,
            "--scene",
            scene_tag,
            "--runs",
            ",".join(run_dir.name for run_dir in scene_runs),
            "--template",
            str(template),
            "--expected-frames",
            str(expected_frames),
        ]
        print(
            f"[rerender] building {scene_tag} grid from {repeat_count} runs: "
            f"{output}",
            flush=True,
        )
        with stdout_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode != 0:
            failures.append(
                f"{scene_tag}: grid build exited {completed.returncode}; "
                f"see {stdout_path}"
            )
    return failures
