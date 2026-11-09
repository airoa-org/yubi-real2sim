# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Source run discovery, validation, and lineage capture before any GPU."""


from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from real2sim.runner.argv import worker_command_head_length
from robosuite_env.contracts.provenance import (
    official_generation_check,
)
from robosuite_env.contracts.worker import (
    WorkerContractError,
    validate_worker_contract,
)
from robosuite_env.rollout.state.render_state import (
    validate_render_state_rows,
)

from robosuite_env.rerender.cadence import (
    _fr3_policy_action_playback_contract,
    _frame_hz_playback_contract,
)
from robosuite_env.rerender.common import (
    REPO_ROOT,
    RerenderError,
    SourceRun,
    _BATCH_POLICY_LINEAGE_FILENAMES,
    _COMMIT_RE,
    _last_option_value,
    _load_json_object,
    _resolved_recorded_path,
    _rollout_expected_frames,
)
from robosuite_env.rerender.provenance import (
    _assert_record_current,
    _file_record,
    _file_state,
    _git_code_identity,
)
from robosuite_env.rerender.registration import _capture_summary_gs_registration
from robosuite_env.rerender.replay_argv import (
    _with_moved_condition_assets,
    _without_option,
    normalize_recorded_replay_options,
    parse_run_command,
)


def _load_state_rows_with_record(
    sim_log: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse and hash a state log in one streaming pass."""

    sim_log = sim_log.resolve()
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    try:
        before = _file_state(sim_log)
        with sim_log.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                value = json.loads(raw_line)
                if not isinstance(value, dict):
                    raise RerenderError(
                        f"{sim_log}:{line_number}: state row is not a JSON object"
                    )
                rows.append(value)
        after = _file_state(sim_log)
    except RerenderError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RerenderError(f"cannot read state log {sim_log}: {exc}") from exc
    if before != after:
        raise RerenderError(f"state log changed while it was read: {sim_log}")
    if not rows:
        raise RerenderError(f"state log is empty: {sim_log}")
    return rows, {
        "path": str(sim_log),
        "sha256": digest.hexdigest(),
        "state": after,
    }


def _load_state_rows(sim_log: Path) -> list[dict[str, Any]]:
    rows, _ = _load_state_rows_with_record(sim_log)
    return rows


def _load_generation_summary(run_dir: Path) -> dict:
    path = run_dir / "sim_summary.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RerenderError(f"cannot read generation summary {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RerenderError(f"generation summary is not a JSON object: {path}")
    return value


def _capture_batch_policy_lineage(run_dir: Path) -> dict[str, Any]:
    """Snapshot the exact optional policy lineage files at the batch root."""

    batch_root = run_dir.resolve().parents[1]
    files: dict[str, dict[str, Any]] = {}
    for filename in _BATCH_POLICY_LINEAGE_FILENAMES:
        candidate = batch_root / filename
        if candidate.is_symlink():
            raise RerenderError(f"policy lineage file must not be a symlink: {candidate}")
        if not candidate.exists():
            continue
        if not candidate.is_file():
            raise RerenderError(f"policy lineage path is not a file: {candidate}")
        files[filename] = _file_record(candidate, include_state=True)
    return {
        "batch_root": str(batch_root),
        "optional_filenames": list(_BATCH_POLICY_LINEAGE_FILENAMES),
        "files": files,
    }


def _assert_batch_policy_lineage_current(lineage: dict[str, Any]) -> None:
    """Verify both lineage-file presence and bytes remain unchanged."""

    raw_root = lineage.get("batch_root")
    if not isinstance(raw_root, str):
        raise RerenderError("batch policy lineage has no root")
    batch_root = Path(raw_root)
    expected_names = lineage.get("optional_filenames")
    if expected_names != list(_BATCH_POLICY_LINEAGE_FILENAMES):
        raise RerenderError("batch policy lineage has an invalid candidate file set")
    records = lineage.get("files")
    if not isinstance(records, dict):
        raise RerenderError("batch policy lineage has no file records")

    observed_names: set[str] = set()
    for filename in _BATCH_POLICY_LINEAGE_FILENAMES:
        candidate = batch_root / filename
        if candidate.is_symlink():
            raise RerenderError(f"policy lineage file became a symlink: {candidate}")
        if candidate.exists():
            if not candidate.is_file():
                raise RerenderError(f"policy lineage path is not a file: {candidate}")
            observed_names.add(filename)
    if observed_names != set(records):
        raise RerenderError(
            "batch policy lineage presence changed after source validation: "
            f"expected {sorted(records)}, observed {sorted(observed_names)}"
        )
    for filename, record in records.items():
        expected_path = (batch_root / filename).resolve()
        if not isinstance(record, dict) or record.get("path") != str(expected_path):
            raise RerenderError(f"policy lineage identity is invalid for {filename}")
        _assert_record_current(
            record,
            label=f"policy lineage {filename}",
            verify_hash=True,
        )


def _option_present(argv: list[str], option: str) -> bool:
    prefix = f"{option}="
    return any(token == option or token.startswith(prefix) for token in argv)


def _validate_official_source(
    run_dir: Path,
    argv: list[str],
    summary: dict[str, Any],
    renderer_code: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind an official derivative to its exact launcher authority."""

    label = f"{run_dir}: official source"
    if summary.get("execution_class") != "official":
        raise RerenderError(f"{label} has no execution_class='official'")
    if "--strict-assets" not in argv:
        raise RerenderError(f"{label} command has no --strict-assets")

    expected_manifest = (run_dir.parents[1] / "run_manifest.json").resolve()
    expected_worker = (run_dir / "official_worker_contract.json").resolve()
    expected_scene_tag = run_dir.parent.name
    recorded_manifest = _resolved_recorded_path(
        _last_option_value(argv, "--official-run-manifest"),
        label="recorded --official-run-manifest",
    )
    recorded_worker = _resolved_recorded_path(
        _last_option_value(argv, "--official-worker-contract"),
        label="recorded --official-worker-contract",
    )
    recorded_scene_tag = _last_option_value(argv, "--official-scene-tag")
    summary_manifest = _resolved_recorded_path(
        summary.get("official_run_manifest"),
        label="summary official_run_manifest",
    )
    summary_worker = _resolved_recorded_path(
        summary.get("official_worker_contract"),
        label="summary official_worker_contract",
    )
    if recorded_manifest != expected_manifest or summary_manifest != expected_manifest:
        raise RerenderError(f"{label} does not bind the exact run_manifest.json")
    if recorded_worker != expected_worker or summary_worker != expected_worker:
        raise RerenderError(f"{label} does not bind the exact worker contract")
    if recorded_scene_tag != expected_scene_tag:
        raise RerenderError(
            f"{label} scene tag {recorded_scene_tag!r} != {expected_scene_tag!r}"
        )

    manifest_record = _file_record(expected_manifest, include_state=False)
    worker_record = _file_record(expected_worker, include_state=False)
    manifest = _load_json_object(expected_manifest, label="official run manifest")
    worker = _load_json_object(expected_worker, label="official worker contract")
    contract_argv = worker.get("argv")
    if not isinstance(contract_argv, list) or any(
        not isinstance(item, str) for item in contract_argv
    ):
        raise RerenderError(f"{label} worker contract argv is invalid")
    head_length = worker_command_head_length(argv)
    if head_length is None:
        raise RerenderError(f"{label} RUN_COMMAND has an unexpected command head")
    recorded_worker_argv, removed = _without_option(
        argv[head_length:], "--official-worker-contract"
    )
    if removed != 1 or recorded_worker_argv != contract_argv:
        raise RerenderError(f"{label} RUN_COMMAND differs from its worker contract")
    try:
        validate_worker_contract(expected_worker, expected_manifest, contract_argv)
    except (WorkerContractError, OSError, ValueError, TypeError) as exc:
        raise RerenderError(f"{label} worker authority is invalid: {exc}") from exc

    code = manifest.get("code")
    manifest_commit = code.get("commit") if isinstance(code, dict) else None
    manifest_dirty = code.get("dirty") if isinstance(code, dict) else None
    if (
        not isinstance(manifest_commit, str)
        or _COMMIT_RE.fullmatch(manifest_commit) is None
        or manifest_dirty is not False
    ):
        raise RerenderError(f"{label} manifest has no clean 40-hex code commit")
    # The record of this check goes into the rerender provenance, and
    # completion reporting verifies that record against the run's seal rather
    # than the checkout it finds then, so both give this answer.
    generation_check = official_generation_check(
        expected_manifest.parent, manifest, renderer_code, checkout_root=REPO_ROOT
    )
    if generation_check is None:
        raise RerenderError(
            f"{label} requires clean checkout {manifest_commit}, or runtime "
            "sources that all match its runtime source seal; "
            f"current checkout is {renderer_code}"
        )

    try:
        scenes = manifest["preflight"]["contracts"]["scenes"]
        scene_authority = scenes[expected_scene_tag]
    except (KeyError, TypeError) as exc:
        raise RerenderError(
            f"{label} manifest has no scene authority for {expected_scene_tag}"
        ) from exc
    if not isinstance(scene_authority, dict):
        raise RerenderError(f"{label} scene authority is not an object")
    return {
        "run_manifest": manifest,
        "run_manifest_path": expected_manifest,
        "worker_contract_path": expected_worker,
        "scene": scene_authority,
        "code_commit": manifest_commit,
    }, {
        "run_manifest": manifest_record,
        "worker_contract": worker_record,
        "code_commit": manifest_commit,
        "official_generation_check": generation_check,
    }


def prepare_source_run(
    run_dir: Path, *, renderer_code: dict[str, Any] | None = None
) -> SourceRun:
    """Validate one complete generation run before any GPU process starts."""

    run_dir = run_dir.resolve()
    command_path = run_dir / "RUN_COMMAND.md"
    summary_path = run_dir / "sim_summary.json"
    state_path = run_dir / "logs" / "sim_log.jsonl"
    command_record = _file_record(command_path, include_state=True)
    environment, argv = parse_run_command(run_dir)
    _assert_record_current(command_record, label="run command")

    raw_frames = _last_option_value(argv, "--frames")
    try:
        configured_frames = int(raw_frames) if raw_frames is not None else 0
    except ValueError as exc:
        raise RerenderError(f"{run_dir}: final --frames is not an integer") from exc
    if configured_frames <= 0:
        raise RerenderError(f"{run_dir}: final --frames must be a positive integer")

    summary_record = _file_record(summary_path, include_state=True)
    summary = _load_generation_summary(run_dir)
    _assert_record_current(summary_record, label="generation summary")
    # Reject unreproducible visuals, and a pre-move condition path whose moved
    # file is not the bytes the run sealed, before a GPU/server starts, using
    # the same guards as argv rewriting. Keep the original argv for source
    # provenance.
    normalize_recorded_replay_options(argv, summary)
    _with_moved_condition_assets(argv, run_dir)
    expected_frames = _rollout_expected_frames(
        argv, summary, configured_frames, label=str(run_dir)
    )
    summary_frames = summary.get("frames")
    if isinstance(summary_frames, bool) or summary_frames != expected_frames:
        detail = (
            f"expected final --frames={configured_frames}"
            if expected_frames == configured_frames
            else (
                f"expected the {expected_frames} frames this chained rollout "
                f"stopped at, within its --frames={configured_frames} budget"
            )
        )
        raise RerenderError(
            f"{run_dir}: generation summary frames={summary_frames!r}, {detail}"
        )
    if summary.get("wall_stopped_early") is not False:
        raise RerenderError(
            f"{run_dir}: generation summary wall_stopped_early="
            f"{summary.get('wall_stopped_early')!r}, expected false"
        )

    gs_registration = None
    if _option_present(argv, "--gs-background") or _option_present(argv, "--gs"):
        gs_registration = _capture_summary_gs_registration(
            argv,
            summary,
            label=f"{run_dir}: generation summary",
        )

    cadence_contract = _fr3_policy_action_playback_contract(argv, summary)
    if cadence_contract is None:
        # Refuse, before any GPU work, a --frame-hz source whose recorded
        # presentation clock its replay could not reproduce. Validation
        # rebuilds the same contract from this source to check the output.
        _frame_hz_playback_contract(argv, summary)

    rows, state_record = _load_state_rows_with_record(state_path)
    if len(rows) != expected_frames:
        raise RerenderError(
            f"{run_dir}: state log has {len(rows)} rows, "
            f"expected {expected_frames}"
        )
    for index, row in enumerate(rows):
        frame = row.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame != index:
            raise RerenderError(
                f"{run_dir}: state log row {index + 1} has frame={frame!r}, "
                f"expected {index}"
            )

    try:
        render_state_validation = validate_render_state_rows(
            rows,
            summary.get("render_state_log"),
            required=False,
        )
    except ValueError as exc:
        raise RerenderError(
            f"{run_dir}: render-state source contract is invalid: {exc}"
        ) from exc

    policy_lineage = _capture_batch_policy_lineage(run_dir)
    code_identity = dict(renderer_code or _git_code_identity())
    execution_class = summary.get("execution_class")
    if not isinstance(execution_class, str) or execution_class not in {
        "diagnostic",
        "official",
    }:
        raise RerenderError(
            f"{run_dir}: generation summary execution_class must be "
            f"'diagnostic' or 'official', got {execution_class!r}"
        )
    official = execution_class == "official"
    # The batch runner records --official-scene-tag for diagnostic workers too;
    # it identifies a scene but grants no official authority by itself. Strict
    # mode, manifest/worker paths, and summary authority must still agree with
    # execution_class so a partially corrupted official run cannot be treated
    # as diagnostic.
    strict_authority_signals = [
        label
        for label, present in (
            (
                "summary official_run_manifest",
                summary.get("official_run_manifest") not in (None, ""),
            ),
            (
                "summary official_worker_contract",
                summary.get("official_worker_contract") not in (None, ""),
            ),
            ("--strict-assets", _option_present(argv, "--strict-assets")),
            (
                "--official-run-manifest",
                _option_present(argv, "--official-run-manifest"),
            ),
            (
                "--official-worker-contract",
                _option_present(argv, "--official-worker-contract"),
            ),
        )
        if present
    ]
    if not official and strict_authority_signals:
        raise RerenderError(
            f"{run_dir}: diagnostic source has inconsistent official authority: "
            + ", ".join(strict_authority_signals)
        )
    authority: dict[str, Any] | None = None
    official_records: dict[str, Any] = {}
    if official:
        authority, official_records = _validate_official_source(
            run_dir, argv, summary, code_identity
        )

    source_records: dict[str, Any] = {
        "generation_summary": summary_record,
        "state_log": {**state_record, "rows": len(rows)},
        "run_command": command_record,
        "batch_policy_lineage": policy_lineage,
        "expected_frames": expected_frames,
        **official_records,
    }
    if cadence_contract is not None:
        source_records["policy_action_playback_contract"] = cadence_contract
    if render_state_validation is not None:
        source_records["render_state_log_validation"] = render_state_validation
    if gs_registration is not None:
        source_records["gs_registration"] = gs_registration
    return SourceRun(
        run_dir=run_dir,
        environment=environment,
        argv=argv,
        summary=summary,
        state_rows=rows,
        expected_frames=expected_frames,
        official=official,
        official_authority=authority,
        source_records=source_records,
        renderer_code=code_identity,
    )


def discover_runs(run_root: Path) -> list[Path]:
    runs = sorted(
        candidate.parent
        for candidate in run_root.glob("*/run_*/RUN_COMMAND.md")
    )
    if not runs:
        raise RerenderError(f"no */run_*/RUN_COMMAND.md under {run_root}")
    return runs


def _select_run_paths(arguments: argparse.Namespace) -> tuple[list[Path], bool]:
    run_root = arguments.run_root.resolve()
    all_runs = discover_runs(run_root)
    runs = all_runs
    partial_selection = False
    if arguments.runs:
        raw_names = str(arguments.runs).split(",")
        wanted_names = [name.strip() for name in raw_names]
        if not wanted_names or any(not name for name in wanted_names):
            raise RerenderError(
                "--runs must contain non-empty comma-separated run names"
            )
        invalid = [
            name for name in wanted_names if re.fullmatch(r"run_\d+", name) is None
        ]
        if invalid:
            raise RerenderError(
                "--runs entries must match run_<integer>: " + ", ".join(invalid)
            )
        duplicates = sorted(
            {name for name in wanted_names if wanted_names.count(name) > 1}
        )
        if duplicates:
            raise RerenderError(
                "--runs contains duplicate entries: " + ", ".join(duplicates)
            )
        wanted = set(wanted_names)
        runs = [run for run in all_runs if run.name in wanted]
        missing = wanted - {run.name for run in runs}
        if missing:
            raise RerenderError(f"unknown --runs entries: {sorted(missing)}")
        partial_selection = len(runs) < len(all_runs)
    return runs, partial_selection


def prepare_selected_sources(
    arguments: argparse.Namespace,
) -> tuple[list[SourceRun], bool]:
    """Validate every selected source before acquiring or touching a GPU."""

    runs, partial_selection = _select_run_paths(arguments)
    renderer_code = _git_code_identity()
    sources = [
        prepare_source_run(run_dir, renderer_code=renderer_code)
        for run_dir in runs
    ]
    return sources, partial_selection
