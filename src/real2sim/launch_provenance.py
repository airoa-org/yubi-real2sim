# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Launch manifest persistence and runtime source sealing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from real2sim._bootstrap import REPO_ROOT
from real2sim.artifacts import ArtifactVerification
from real2sim.config._types import ConfigError
from real2sim.config.paths import (
    CONFIG_SNAPSHOT_FILE,
    RESOLVED_CONFIG_FILE,
    RUN_MANIFEST_FILE,
)
from real2sim.preflight import validate_launch_preflight
from real2sim.provenance import (
    ProvenanceError,
    build_run_manifest,
    write_run_manifest,
)
from real2sim.runner.argv import _official_static_sim_argv
from real2sim.runner.environment import (
    build_all_shard_environments,
    build_runner_environment,
)
from real2sim.artifacts_binding import verify_plan_artifacts
from real2sim.runtime_manifest import (
    MANIFEST_FILE as RUNTIME_SOURCE_MANIFEST_FILE,
    RuntimeManifestError,
    capture_selection as capture_runtime_selection,
    default_selection as runtime_source_selection,
    verify_capture as verify_runtime_capture,
    write_manifest as write_runtime_manifest,
)

RUNTIME_CHAIN_FILES_ENV = "REAL2SIM_RUNTIME_CHAIN_FILES"

def prepare_run_manifest(
    plan: Mapping[str, Any],
    preflight_report: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Collect complete schema-v2 provenance before any output is created."""

    reproducibility = plan["resolved"].get("reproducibility")
    if reproducibility is None:
        return None
    if reproducibility["production"] and preflight_report is None:
        preflight_report = validate_launch_preflight(plan)
    contract_report = (
        preflight_report.get("contracts")
        if preflight_report is not None
        else None
    )
    artifact_verification = (
        preflight_report.get("_artifact_verification")
        if preflight_report is not None
        else None
    )
    if artifact_verification is not None and not isinstance(
        artifact_verification, ArtifactVerification
    ):
        raise ConfigError("preflight artifact verification has an invalid type")
    manifest, records = verify_plan_artifacts(
        plan,
        contract_report,
        artifact_verification=artifact_verification,
    )
    assert manifest is not None
    gpu_bindings = (
        preflight_report.get("gpu_bindings")
        if preflight_report is not None
        else None
    )
    environment = build_runner_environment(plan, gpu_bindings=gpu_bindings)
    resolved_bytes = json.dumps(
        plan["resolved"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    config_record = {
        "path": plan["config_repo_path"],
        "sha256": plan["config_sha256"],
        "resolved_sha256": hashlib.sha256(resolved_bytes).hexdigest(),
        "timestamp": plan["timestamp"],
    }
    try:
        run_manifest = build_run_manifest(
            repo_root=Path(plan["repo_root"]),
            config=config_record,
            environment=environment,
            artifact_manifest_path=str(manifest["path"]),
            artifact_manifest_sha256=manifest["sha256"],
            artifacts=records,
            lockfiles=reproducibility["lockfiles"],
            source_dependencies=reproducibility["source_dependencies"],
            accelerator_profile=reproducibility["accelerator_profile"],
            production=reproducibility["production"],
            gaussian_wrapping_repo=plan["resolved"]["runtime"]["gaussian_wrapping_repo"],
            launch_locked_imports=(
                preflight_report.get("_locked_imports")
                if preflight_report is not None
                else None
            ),
        )
    except ProvenanceError as exc:
        raise ConfigError(f"run provenance validation failed: {exc}") from exc
    if preflight_report is None:
        run_manifest["preflight"] = None
    else:
        serializable_preflight = dict(preflight_report)
        serializable_preflight.pop("_artifact_verification", None)
        # The top-level locked_imports field records these.
        serializable_preflight.pop("_locked_imports", None)
        run_manifest["preflight"] = serializable_preflight
    shards = plan["resolved"]["runner"].get("shards")
    if isinstance(shards, list) and shards:
        shard_environments, aggregation_environment = (
            build_all_shard_environments(plan)
        )
        shard_records = []
        static_argv_by_shard = {}
        for shard in shards:
            name = str(shard["name"])
            shard_environment = shard_environments[name]
            static_argv = _official_static_sim_argv(shard_environment)
            static_argv_by_shard[name] = static_argv
            shard_records.append(
                {
                    "name": name,
                    "repeat_start": int(shard["repeat_start"]),
                    "repeat_stop": int(shard["repeat_stop"]),
                    "environment": shard_environment,
                    "static_argv": static_argv,
                }
            )
        run_manifest["shard_execution"] = {
            "schema_version": 1,
            "shards": shard_records,
            "aggregation_environment": aggregation_environment,
        }
        run_manifest["launcher_context"] = {
            "shards_override": plan.get("shards_override"),
            "environment": dict(plan.get("shard_resolution_environment", {})),
        }
        run_manifest["worker_contract"] = {
            "schema_version": 2,
            "static_argv_by_shard": static_argv_by_shard,
        }
    else:
        run_manifest["worker_contract"] = {
            "schema_version": 1,
            "static_argv": _official_static_sim_argv(environment),
        }
    return run_manifest

def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)

def persist_config_provenance(
    plan: Mapping[str, Any],
    run_manifest: Mapping[str, Any] | None = None,
) -> None:
    run_root = Path(plan["run_root"])
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / CONFIG_SNAPSHOT_FILE).write_bytes(plan["raw_bytes"])
    _write_json(run_root / RESOLVED_CONFIG_FILE, plan["resolved"])
    if run_manifest is not None:
        write_run_manifest(run_root / RUN_MANIFEST_FILE, run_manifest)

def _runtime_chain_files(
    environment: Mapping[str, str] | None = None,
) -> list[Path]:
    source = os.environ if environment is None else environment
    raw_value = source.get(RUNTIME_CHAIN_FILES_ENV, "")
    return [
        Path(value).expanduser()
        for value in raw_value.split(os.pathsep)
        if value
    ]

def _existing_runtime_input(
    value: Any,
    *,
    repo_root: Path,
) -> Path | None:
    if not isinstance(value, str) or not value or value.startswith("--"):
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        return candidate if candidate.is_file() else None
    except OSError:
        return None

def runtime_config_input_files(plan: Mapping[str, Any]) -> list[Path]:
    """Return config/control files that a fresh worker may reopen each wave."""

    resolved = plan["resolved"]
    simulation = resolved["simulation"]
    repo_root = Path(plan["repo_root"])
    run_root = Path(plan["run_root"])
    candidates: list[Any] = [
        plan["config_path"],
        run_root / CONFIG_SNAPSHOT_FILE,
        run_root / RESOLVED_CONFIG_FILE,
        resolved["task"]["objects"],
        simulation["gripper_calibration"],
        resolved["render"]["grid_template"],
        *_runtime_chain_files(),
    ]
    # A per-run box layout list is selected by the batch runner; every
    # listed layout is a runtime input.
    box_layouts = (simulation.get("parameters") or {}).get("box_pack_layout_json")
    if isinstance(box_layouts, list):
        candidates.extend(box_layouts)
    # Tasks whose object poses come from a default layout file rather than
    # from argv: a fresh worker reopens it every wave, so it has to be sealed
    # like any other runtime input.
    default_layouts = {
        "box_pack": "layout_box_pack_duo_manual.json",
    }
    default_layout = default_layouts.get(resolved["task"]["objects"])
    if default_layout:
        candidates.append(
            repo_root / "robosuite_env" / "obj_calib" / default_layout
        )
    official_manifest = run_root / RUN_MANIFEST_FILE
    if official_manifest.is_file():
        candidates.append(official_manifest)
    # Schema-v1 keeps file-valued selectors in the exact argv.  Selecting only
    # values that are existing regular files avoids treating numeric/string
    # option values as paths while still binding joints, desk, and calibration
    # files used by each new simulator process.
    candidates.extend(simulation["cli_args"])

    selected: set[Path] = set()
    for value in candidates:
        if isinstance(value, Path):
            candidate = value
            if not candidate.is_absolute():
                candidate = repo_root / candidate
            if not candidate.is_file():
                continue
        else:
            candidate = _existing_runtime_input(value, repo_root=repo_root)
            if candidate is None:
                continue
        selected.add(Path(os.path.abspath(candidate)))
    return sorted(selected, key=lambda path: str(path))

def capture_launch_runtime_generation(
    config_path: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Capture source/config before parsing or runtime preflight begins."""

    extra_files: list[str | Path] = [config_path]
    extra_files.extend(_runtime_chain_files(environment))
    selection = runtime_source_selection(repo_root, extra_files=extra_files)
    return capture_runtime_selection(selection)

def seal_runtime_source_manifest(
    plan: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> tuple[Path, str]:
    """Atomically seal one unchanged generation after config snapshots exist."""

    verify_runtime_capture(baseline)
    config_path = Path(plan["config_path"])
    try:
        current_config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeManifestError(
            f"rollout config is unreadable while sealing runtime generation: {exc}"
        ) from exc
    if current_config_digest != plan["config_sha256"]:
        raise RuntimeManifestError("rollout config changed after it was parsed")

    selection = runtime_source_selection(
        plan["repo_root"],
        extra_files=runtime_config_input_files(plan),
    )
    capture = capture_runtime_selection(selection)
    manifest_path = Path(plan["run_root"]) / RUNTIME_SOURCE_MANIFEST_FILE
    digest = write_runtime_manifest(manifest_path, capture)
    return manifest_path.resolve(), digest
