# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Bind one official simulator process to the batch runner's exact argv."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping, Sequence
from unittest.mock import patch

from robosuite_env.contracts.provenance import sha256_file
from real2sim.gpu_binding import (
    CUDA_VISIBLE_DEVICES_ENV,
    GPU_BINDINGS_ENV,
    GpuBindingError,
    validate_gpu_contract,
)


class WorkerContractError(RuntimeError):
    """Raised when an official worker was not launched as predeclared."""


_RETIRED_STATIC_OPTIONS = ("--fr3-contact-control", "--ik-url")


def _without_retired_shard_environments(shard_execution: dict) -> dict:
    """Drop retired launcher fields from each recorded shard environment.

    Manifests written before a field was retired still carry it; the rebuilt
    environments do not. Everything else must still match exactly.
    """

    from real2sim.runner.environment import without_retired_environment

    normalized = dict(shard_execution)
    shards = normalized.get("shards")
    if isinstance(shards, list):
        normalized["shards"] = [
            {**shard, "environment": without_retired_environment(shard["environment"])}
            if isinstance(shard, dict) and isinstance(shard.get("environment"), dict)
            else shard
            for shard in shards
        ]
    aggregation = normalized.get("aggregation_environment")
    if isinstance(aggregation, dict):
        normalized["aggregation_environment"] = without_retired_environment(aggregation)
    return normalized


# The canonical single-source implementation; the private alias keeps this
# module's call sites (and the tests that patch them) unchanged.
_sha256 = sha256_file


def _paths(
    contract_path: str | os.PathLike[str],
    run_manifest_path: str | os.PathLike[str],
) -> tuple[Path, Path]:
    contract = Path(contract_path).expanduser().resolve()
    manifest = Path(run_manifest_path).expanduser().resolve()
    if manifest.name != "run_manifest.json" or not manifest.is_file():
        raise WorkerContractError(f"official run manifest is missing: {manifest}")
    run_root = manifest.parent
    try:
        relative = contract.relative_to(run_root)
    except ValueError as exc:
        raise WorkerContractError(
            f"worker contract must be inside official run root {run_root}: {contract}"
        ) from exc
    if not relative.parts or relative == Path("run_manifest.json"):
        raise WorkerContractError("worker contract path is invalid")
    return contract, manifest


def _load_authority(
    manifest: Path,
    contract: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Load and verify the launcher files that authorize worker arguments."""

    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        config = payload["config"]
        environment = payload["resolved_environment"]
        worker = payload["worker_contract"]
        scenes = payload["preflight"]["contracts"]["scenes"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise WorkerContractError(
            f"official run manifest lacks worker authority: {manifest}: {exc}"
        ) from exc
    if (
        not isinstance(config, dict)
        or not isinstance(environment, dict)
        or not isinstance(worker, dict)
        or not isinstance(scenes, dict)
        or worker.get("schema_version") not in {1, 2}
    ):
        raise WorkerContractError("official run manifest worker authority is invalid")
    if worker["schema_version"] == 1:
        static_vectors = [worker.get("static_argv")]
    else:
        raw_vectors = worker.get("static_argv_by_shard")
        static_vectors = list(raw_vectors.values()) if isinstance(raw_vectors, dict) else []
    if not static_vectors or any(
        not isinstance(vector, list)
        or any(not isinstance(item, str) for item in vector)
        for vector in static_vectors
    ):
        raise WorkerContractError("official run manifest worker authority is invalid")
    for option in _RETIRED_STATIC_OPTIONS:
        for static_argv in static_vectors:
            if option in static_argv or any(
                item.startswith(f"{option}=") for item in static_argv
            ):
                raise WorkerContractError(
                    f"official simulator arguments contain retired option {option}"
                )

    repo_root = Path(__file__).resolve().parents[2]
    raw_config_path = config.get("path")
    if not isinstance(raw_config_path, str):
        raise WorkerContractError("official rollout config path is invalid")
    relative_config_path = Path(raw_config_path)
    if (
        relative_config_path.is_absolute()
        or not relative_config_path.parts
        or relative_config_path.parts[0] != "config"
        or relative_config_path.suffix not in {".yaml", ".yml"}
    ):
        raise WorkerContractError("official rollout config must be under config/")
    config_path = (repo_root / relative_config_path).resolve()
    try:
        config_path.relative_to(repo_root)
    except ValueError as exc:
        raise WorkerContractError("official rollout config escapes the repository") from exc
    if not config_path.is_file() or _sha256(config_path) != config.get("sha256"):
        raise WorkerContractError("official rollout config changed after preflight")
    timestamp = config.get("timestamp")
    if not isinstance(timestamp, str) or re.fullmatch(
        r"[0-9]{8}_[0-9]{6}", timestamp
    ) is None:
        raise WorkerContractError("official rollout timestamp is invalid")
    resolved_path = manifest.parent / "resolved_rollout_config.json"
    try:
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkerContractError(
            f"official resolved rollout config is unreadable: {resolved_path}"
        ) from exc
    if not isinstance(resolved, dict) or resolved.get("schema_version") != 2:
        raise WorkerContractError("official resolved rollout config must use schema v2")
    resolved_digest = hashlib.sha256(
        json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if resolved_digest != config.get("resolved_sha256"):
        raise WorkerContractError("official resolved rollout config changed after preflight")
    output = resolved.get("output")
    if (
        not isinstance(output, dict)
        or Path(output.get("run_root", "")).resolve() != manifest.parent
    ):
        raise WorkerContractError("official resolved rollout config uses another run root")

    # Re-resolve the checked-in YAML with the launcher's captured environment.
    # This prevents a hand-edited manifest and resolved JSON from authorizing a
    # different controller, OBS pipeline, or simulator option.
    reproducibility = resolved.get("reproducibility")
    source_environment = (
        reproducibility.get("environment")
        if isinstance(reproducibility, dict)
        else None
    )
    if not isinstance(source_environment, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in source_environment.items()
    ):
        raise WorkerContractError("official captured environment is invalid")
    try:
        from real2sim.launcher import (
            _official_static_sim_argv,
            build_all_shard_environments,
            build_runner_environment,
            load_rollout_config,
        )
        from real2sim.runner.environment import without_retired_environment

        launcher_context = payload.get("launcher_context")
        if launcher_context is None:
            launcher_context = {"shards_override": None, "environment": {}}
        if not isinstance(launcher_context, dict):
            raise WorkerContractError("official launcher context is invalid")
        context_environment = launcher_context.get("environment", {})
        shards_override = launcher_context.get("shards_override")
        if not isinstance(context_environment, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in context_environment.items()
        ):
            raise WorkerContractError("official launcher context environment is invalid")
        rebuild_environment = dict(source_environment)
        rebuild_environment.update(context_environment)
        with patch.dict(os.environ, rebuild_environment, clear=True):
            rebuilt_plan = load_rollout_config(
                config_path,
                repo_root=repo_root,
                timestamp=timestamp,
                shards_override=shards_override,
            )
        if rebuilt_plan["resolved"] != resolved:
            raise WorkerContractError(
                "official resolved rollout config differs from its checked-in YAML"
            )
        expected_environment = build_runner_environment(rebuilt_plan)
        raw_gpu_bindings = environment.get(GPU_BINDINGS_ENV)
        if raw_gpu_bindings:
            if not isinstance(raw_gpu_bindings, str):
                raise WorkerContractError(
                    "official launcher GPU binding contract is invalid"
                )
            parsed_gpu_bindings = json.loads(raw_gpu_bindings)
            if not isinstance(parsed_gpu_bindings, dict):
                raise WorkerContractError(
                    "official launcher GPU binding contract is invalid"
                )
            validated_gpu_bindings = validate_gpu_contract(
                parsed_gpu_bindings, expected_environment
            )
            expected_environment = build_runner_environment(
                rebuilt_plan, gpu_bindings=validated_gpu_bindings
            )
        if worker["schema_version"] == 1:
            expected_static_argv = _official_static_sim_argv(expected_environment)
            if without_retired_environment(environment) != expected_environment:
                raise WorkerContractError(
                    "official launcher environment differs from the checked-in rollout config"
                )
            if worker["static_argv"] != expected_static_argv:
                raise WorkerContractError(
                    "official simulator arguments differ from the checked-in rollout config"
                )
        else:
            shard_execution = payload.get("shard_execution")
            shards = rebuilt_plan["resolved"]["runner"].get("shards")
            if (
                not isinstance(shard_execution, dict)
                or shard_execution.get("schema_version") != 1
                or not isinstance(shards, list)
                or not shards
            ):
                raise WorkerContractError("official shard execution authority is invalid")
            generated_environments, aggregation_environment = (
                build_all_shard_environments(rebuilt_plan)
            )
            expected_records = []
            expected_static_by_shard = {}
            for shard in shards:
                name = str(shard["name"])
                shard_environment = generated_environments[name]
                shard_static = _official_static_sim_argv(shard_environment)
                expected_static_by_shard[name] = shard_static
                expected_records.append(
                    {
                        "name": name,
                        "repeat_start": int(shard["repeat_start"]),
                        "repeat_stop": int(shard["repeat_stop"]),
                        "environment": shard_environment,
                        "static_argv": shard_static,
                    }
                )
            expected_shard_execution = {
                "schema_version": 1,
                "shards": expected_records,
                "aggregation_environment": aggregation_environment,
            }
            if without_retired_environment(environment) != expected_environment:
                raise WorkerContractError(
                    "official launcher environment differs from the checked-in rollout config"
                )
            if _without_retired_shard_environments(shard_execution) != expected_shard_execution or worker != {
                "schema_version": 2,
                "static_argv_by_shard": expected_static_by_shard,
            }:
                raise WorkerContractError(
                    "official shard execution differs from the checked-in rollout config"
                )

            selected_name: str | None = None
            run_match = re.fullmatch(r"run_([0-9]+)", contract.parent.name)
            if run_match is not None:
                repeat = int(run_match.group(1))
                for shard in shards:
                    if int(shard["repeat_start"]) <= repeat <= int(shard["repeat_stop"]):
                        selected_name = str(shard["name"])
                        break
            elif (
                contract.parent.name == "deterministic_init_state_prep"
                and contract.parent.parent.parent.name == ".shards"
            ):
                selected_name = contract.parent.parent.name
            elif (
                contract.name == "official_worker_contract.json"
                and re.fullmatch(r"layout_[0-9]{2,}", contract.parent.name)
                is not None
                and contract.parent.parent.name == "deterministic_init_state_prep"
                and contract.parent.parent.parent.parent.name == ".shards"
            ):
                # .shards/<name>/deterministic_init_state_prep/layout_NN/
                # is the only extra nesting list-mode preparation introduces.
                selected_name = contract.parent.parent.parent.name
            if selected_name not in generated_environments:
                raise WorkerContractError(
                    "official worker path does not select one configured shard"
                )
            payload = dict(payload)
            payload["resolved_environment"] = generated_environments[selected_name]
            payload["worker_contract"] = {
                "schema_version": 1,
                "static_argv": expected_static_by_shard[selected_name],
            }
            payload["_selected_shard"] = selected_name
    except WorkerContractError:
        raise
    except Exception as exc:
        raise WorkerContractError(
            f"official rollout authority could not be reconstructed: {exc}"
        ) from exc
    resolved_scenes = resolved.get("scenes")
    if not isinstance(resolved_scenes, list) or not resolved_scenes:
        raise WorkerContractError("official resolved config has no scenes")
    expected_scene_tags: list[str] = []
    for item in resolved_scenes:
        if not isinstance(item, dict) or not isinstance(item.get("tag"), str):
            raise WorkerContractError("official resolved scene list is invalid")
        tag = item["tag"]
        expected_scene_tags.append(tag)
        report = scenes.get(tag)
        if not isinstance(report, dict) or not isinstance(
            report.get("manifest_path"), str
        ):
            raise WorkerContractError(f"official scene authority is invalid: {tag}")
        configured_path = Path(str(item.get("path", ""))).resolve()
        expected_manifest = (
            configured_path
            if configured_path.name == "background.json"
            else configured_path / "background.json"
        )
        if Path(report["manifest_path"]).resolve() != expected_manifest:
            raise WorkerContractError(
                f"official scene authority differs from the rollout config: {tag}"
            )
    if len(expected_scene_tags) != len(set(expected_scene_tags)) or set(
        expected_scene_tags
    ) != set(scenes):
        raise WorkerContractError(
            "official scene authority differs from the resolved scene set"
        )
    return payload, resolved


def _pairs(arguments: list[str], options: Sequence[str]) -> dict[str, str]:
    expected_length = len(options) * 2
    if len(arguments) != expected_length:
        raise WorkerContractError("official worker has an unexpected dynamic argv shape")
    result: dict[str, str] = {}
    for index, option in enumerate(options):
        position = index * 2
        if arguments[position] != option:
            raise WorkerContractError(
                f"official worker expected {option}, got {arguments[position]!r}"
            )
        result[option] = arguments[position + 1]
    return result


def _inside(path: str, root: Path, field: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise WorkerContractError(f"{field} must stay inside {root}: {candidate}") from exc
    return candidate


def _validate_state_path(value: str, environment: dict[str, object]) -> Path:
    configured = str(environment.get("INIT_STATE_JSON", ""))
    candidate = Path(value).expanduser().resolve()
    if configured:
        if candidate != Path(configured).expanduser().resolve():
            raise WorkerContractError("worker init state differs from launcher config")
        return candidate
    state_root = str(environment.get("INIT_STATE_ROOT", "")) or str(
        Path(str(environment["EXPORTS_ROOT"])) / "deterministic-init-state"
    )
    candidate = _inside(value, Path(state_root), "automatic init state")
    # The launcher names the cache after the rig it belongs to, because a
    # cached state is bound to one compiled model. Pinning the pattern to the
    # rig this run declares keeps a Duo state from being read back as a Flat
    # one.
    robot_tag = str(environment.get("ROBOT", "")).strip().lower()
    if not robot_tag:
        raise WorkerContractError(
            "automatic init state requires the launcher to declare ROBOT"
        )
    if re.fullmatch(
        rf"{re.escape(robot_tag)}_yubi_policy_deterministic_init_state_[a-z0-9_]+\.json",
        candidate.name,
    ) is None:
        raise WorkerContractError(
            f"automatic init state filename is invalid for {robot_tag}: "
            f"{candidate.name}"
        )
    return candidate


def _validated_render_gpu_bindings(
    authority: dict[str, object],
    environment: dict[str, object],
    recorded_gpu_bindings: Mapping[str, object] | None,
) -> tuple[dict[str, object] | None, bool]:
    """Return canonical lock authority and whether it came from runtime."""

    raw_gpu_bindings: object = environment.get(GPU_BINDINGS_ENV)
    runtime_gpu_bindings = False
    if raw_gpu_bindings is not None and raw_gpu_bindings != "":
        if recorded_gpu_bindings is not None:
            raise WorkerContractError(
                "worker GPU binding cannot replace launcher GPU authority"
            )
    elif recorded_gpu_bindings is not None:
        raw_gpu_bindings = recorded_gpu_bindings
        runtime_gpu_bindings = True
    else:
        raw_gpu_bindings = os.environ.get(GPU_BINDINGS_ENV)
        runtime_gpu_bindings = bool(raw_gpu_bindings)
    if raw_gpu_bindings is None or raw_gpu_bindings == "":
        return None, False
    try:
        if isinstance(raw_gpu_bindings, str):
            parsed_gpu_bindings = json.loads(raw_gpu_bindings)
        elif isinstance(raw_gpu_bindings, Mapping):
            parsed_gpu_bindings = dict(raw_gpu_bindings)
        else:
            raise TypeError("GPU binding contract must contain a JSON object")
        if not isinstance(parsed_gpu_bindings, dict):
            raise TypeError("GPU binding contract must contain a JSON object")
        validation_environment = environment
        if runtime_gpu_bindings and authority.get("_selected_shard") is not None:
            validation_environment = dict(environment)
            visible = parsed_gpu_bindings.get("cuda_visible_devices")
            if isinstance(visible, str):
                validation_environment[CUDA_VISIBLE_DEVICES_ENV] = visible
        validated_gpu_bindings = validate_gpu_contract(
            parsed_gpu_bindings,
            validation_environment,
        )
    except (json.JSONDecodeError, GpuBindingError, TypeError) as exc:
        raise WorkerContractError(
            f"official GPU binding contract is invalid: {exc}"
        ) from exc
    return validated_gpu_bindings, runtime_gpu_bindings


def _validate_dynamic_argv(
    contract: Path,
    authority: dict[str, object],
    resolved: dict[str, object],
    argv: Sequence[str],
    *,
    recorded_gpu_bindings: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    worker = authority["worker_contract"]
    assert isinstance(worker, dict)
    static = worker["static_argv"]
    assert isinstance(static, list)
    arguments = list(argv)
    if arguments[: len(static)] != static:
        raise WorkerContractError(
            "simulator static arguments differ from the resolved rollout config"
        )
    suffix = arguments[len(static) :]
    environment = authority["resolved_environment"]
    assert isinstance(environment, dict)
    raw_scenes = authority["preflight"]["contracts"]["scenes"]
    assert isinstance(raw_scenes, dict)
    run_root = (
        contract.parents[2]
        if contract.parent.name.startswith("run_")
        else manifest_root(authority)
    )
    runtime_gpu_bindings: dict[str, object] | None = None

    def consume_box_layout(
        values: list[str], *, repeat: int | None = None
    ) -> list[str]:
        simulation = resolved.get("simulation")
        parameters = (
            simulation.get("parameters") if isinstance(simulation, dict) else None
        )
        layouts = (
            parameters.get("box_pack_layout_json")
            if isinstance(parameters, dict)
            else None
        )
        if not isinstance(layouts, list):
            return values
        if len(values) < 4 or values[0] != "--box-pack-layout-json" or values[2] != "--box-pack-layout-index":
            raise WorkerContractError(
                "official box-pack layout-list worker lacks its selected layout and index"
            )
        try:
            index = int(values[3])
        except ValueError as exc:
            raise WorkerContractError("box-pack layout index is invalid") from exc
        expected_index = index if repeat is None else (repeat - 1) % len(layouts)
        if index != expected_index or not 0 <= index < len(layouts):
            raise WorkerContractError("box-pack layout index differs from run_NN")
        if os.path.realpath(values[1]) != os.path.realpath(str(layouts[index])):
            raise WorkerContractError(
                "box-pack selected layout differs from the resolved layout list"
            )
        return values[4:]

    is_layout_prep = contract.parent.parent.name == "deterministic_init_state_prep"
    if contract.parent.name == "deterministic_init_state_prep" or is_layout_prep:
        if contract.name != "official_worker_contract.json":
            raise WorkerContractError(
                "official init-state worker contract path is invalid"
            )
        selected_shard = authority.get("_selected_shard")
        expected_prep_directory = manifest_root(authority)
        if isinstance(selected_shard, str):
            expected_prep_directory = (
                expected_prep_directory / ".shards" / selected_shard
            )
        expected_prep_directory /= "deterministic_init_state_prep"
        if is_layout_prep:
            if re.fullmatch(r"layout_[0-9]{2,}", contract.parent.name) is None:
                raise WorkerContractError(
                    "official layout init-state worker contract path is invalid"
                )
            expected_prep_directory /= contract.parent.name
        if contract.parent != expected_prep_directory:
            raise WorkerContractError(
                "official init-state worker contract path is invalid"
            )
        if str(environment.get("DETERMINISTIC_SIM", "")) != "1":
            raise WorkerContractError(
                "official init-state worker requires deterministic simulation"
            )
        if str(environment.get("INIT_STATE_JSON", "")):
            raise WorkerContractError(
                "official init-state worker cannot replace a configured state"
            )
        original_suffix = suffix
        suffix = consume_box_layout(suffix)
        if is_layout_prep and (
            len(original_suffix) < 4
            or contract.parent.name != f"layout_{int(original_suffix[3]):02d}"
        ):
            raise WorkerContractError(
                "official layout init-state worker directory differs from its index"
            )
        options = (
            "--gs-background",
            "--official-scene-tag",
            "--gs-port",
            "--gs-server-token",
        )
        if len(suffix) < 8:
            raise WorkerContractError("official init-state worker argv is incomplete")
        values = _pairs(suffix[:8], options)
        remainder = suffix[8:]
        # The runner appends an explicit --no-render after the scene block
        # only when rendering is enabled; for newer batches the launcher-owned
        # static prefix already carries exactly one --no-render and the prep
        # command must not repeat it. Either placement disables rendering.
        if remainder[:1] == ["--no-render"]:
            remainder = remainder[1:]
        elif "--no-render" not in static:
            raise WorkerContractError("official init-state worker must use --no-render")
        tail_options = (
            "--init-state-in",
            "--init-state-out",
            "--out",
            "--logdir",
            "--summary-path",
        )
        if "--init-state-only" not in remainder:
            raise WorkerContractError("official init-state worker lacks --init-state-only")
        marker = remainder.index("--init-state-only")
        if marker != 4:
            raise WorkerContractError("official init-state worker argv order is invalid")
        tail = _pairs(remainder[:4] + remainder[marker + 1 :], tail_options)
        values.update(tail)
        if tail["--init-state-in"] != "":
            raise WorkerContractError("official init-state creation must clear its input")
        _validate_state_path(tail["--init-state-out"], environment)
        resolved_scenes = resolved.get("scenes")
        if not isinstance(resolved_scenes, list) or not resolved_scenes:
            raise WorkerContractError("official resolved config has no scenes")
        first_scene = resolved_scenes[0]
        first_tag = first_scene.get("tag") if isinstance(first_scene, dict) else None
        if values["--official-scene-tag"] != first_tag:
            raise WorkerContractError("init-state worker must use the first configured scene")
        expected_directory = contract.parent
        expected_paths = {
            "--out": expected_directory / "init_state_only.mp4",
            "--logdir": expected_directory / "logs",
            "--summary-path": expected_directory / "sim_summary.json",
        }
    else:
        run_match = re.fullmatch(r"run_([0-9]+)", contract.parent.name)
        if contract.name != "official_worker_contract.json" or run_match is None:
            raise WorkerContractError("official rollout worker contract path is invalid")
        repeat = int(run_match.group(1))
        resolved_runner = resolved.get("runner")
        repeats = (
            resolved_runner.get("repeats")
            if isinstance(resolved_runner, dict)
            else None
        )
        if (
            isinstance(repeats, bool)
            or not isinstance(repeats, int)
            or not 1 <= repeat <= repeats
            or contract.parent.name != f"run_{repeat:02d}"
        ):
            raise WorkerContractError(
                "official rollout worker repeat is outside the configured range"
            )
        has_init_state = suffix[:1] == ["--init-state-in"]
        if has_init_state:
            if len(suffix) < 2:
                raise WorkerContractError("--init-state-in requires a value")
            _validate_state_path(suffix[1], environment)
            suffix = suffix[2:]
        if str(environment.get("DETERMINISTIC_SIM", "")) == "1" and not has_init_state:
            raise WorkerContractError(
                "official deterministic rollout requires --init-state-in"
            )
        suffix = consume_box_layout(suffix, repeat=repeat)
        options = (
            "--policy-batch-client-id",
            "--gs-background",
            "--official-scene-tag",
            "--gs-port",
            "--gs-server-token",
            "--render-lock",
            "--out",
            "--logdir",
            "--summary-path",
        )
        expected_size = len(options) * 2
        if len(suffix) != expected_size:
            raise WorkerContractError("official rollout worker argv shape is invalid")
        values = _pairs(suffix[: len(options) * 2], options)
        tag = contract.parent.parent.name
        run = contract.parent.name
        if values["--policy-batch-client-id"] != f"{tag}_{run}":
            raise WorkerContractError("policy batch client id differs from worker path")
        if values["--official-scene-tag"] != tag:
            raise WorkerContractError("official scene tag differs from worker path")
        expected_directory = contract.parent
        expected_paths = {
            "--out": expected_directory / f"{tag}_{run}.mp4",
            "--logdir": expected_directory / "logs",
            "--summary-path": expected_directory / "sim_summary.json",
        }

        render_lock = Path(values["--render-lock"]).expanduser().resolve()
        render_root = Path(str(environment["RENDER_LOCK_ROOT"])).expanduser().resolve()
        lock_match = re.fullmatch(r"mujoco_render_faces_gpu_([0-9]+)\.lock", render_lock.name)
        validated_gpu_bindings, from_runtime = _validated_render_gpu_bindings(
            authority,
            environment,
            recorded_gpu_bindings,
        )
        if validated_gpu_bindings is not None:
            allowed_gpus = [
                str(entry["egl_physical_index"])
                for entry in validated_gpu_bindings["roles"]["sim"]
            ]
            if from_runtime:
                runtime_gpu_bindings = validated_gpu_bindings
        else:
            # Compatibility for CPU fixtures and pre-binding diagnostic manifests.
            allowed_gpus = str(environment["SIM_CUDA_DEVICES"]).split(",")
        if (
            lock_match is None
            or lock_match.group(1) not in allowed_gpus
            or render_lock != render_root / render_lock.name
        ):
            raise WorkerContractError("worker render lock GPU is not configured")

    tag = values["--official-scene-tag"]
    scene = raw_scenes.get(tag)
    if not isinstance(scene, dict):
        raise WorkerContractError(f"worker selects unknown official scene {tag!r}")
    expected_background = Path(str(scene["manifest_path"])).resolve().parent
    if Path(values["--gs-background"]).resolve() != expected_background:
        raise WorkerContractError("worker 3DGS background differs from scene contract")
    try:
        port = int(values["--gs-port"])
    except ValueError as exc:
        raise WorkerContractError("worker 3DGS port is invalid") from exc
    if not 1 <= port <= 65535:
        raise WorkerContractError("worker 3DGS port is outside 1..65535")
    if re.fullmatch(r"[0-9a-f]{64}", values["--gs-server-token"]) is None:
        raise WorkerContractError("worker 3DGS server token is invalid")
    for option, expected in expected_paths.items():
        if Path(values[option]).resolve() != expected.resolve():
            raise WorkerContractError(f"worker {option} differs from its output path")
    if run_root.resolve() != manifest_root(authority):
        raise WorkerContractError("worker contract belongs to another run root")
    return runtime_gpu_bindings


def manifest_root(authority: dict[str, object]) -> Path:
    environment = authority["resolved_environment"]
    assert isinstance(environment, dict)
    return Path(str(environment["RUN_ROOT"])).expanduser().resolve()


def write_worker_contract(
    contract_path: str | os.PathLike[str],
    run_manifest_path: str | os.PathLike[str],
    argv: Sequence[str],
) -> Path:
    """Atomically publish the exact arguments for one simulator process."""

    contract, manifest = _paths(contract_path, run_manifest_path)
    authority, resolved = _load_authority(manifest, contract)
    arguments = list(argv)
    if any(
        not isinstance(item, str)
        or "\0" in item
        or "\n" in item
        or "\r" in item
        for item in arguments
    ):
        raise WorkerContractError("worker argv must contain safe strings")
    runtime_gpu_bindings = _validate_dynamic_argv(
        contract,
        authority,
        resolved,
        arguments,
    )
    payload = {
        "schema_version": 2 if runtime_gpu_bindings is not None else 1,
        "run_manifest_sha256": _sha256(manifest),
        "argv": arguments,
    }
    if runtime_gpu_bindings is not None:
        payload["gpu_bindings"] = runtime_gpu_bindings
    contract.parent.mkdir(parents=True, exist_ok=True)
    temporary = contract.with_name(f".{contract.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, contract)
    return contract


def validate_worker_contract(
    contract_path: str | os.PathLike[str],
    run_manifest_path: str | os.PathLike[str],
    argv: Sequence[str],
) -> dict[str, object]:
    """Verify the current simulator argv against its launcher-owned record."""

    contract, manifest = _paths(contract_path, run_manifest_path)
    authority, resolved = _load_authority(manifest, contract)
    try:
        payload = json.loads(contract.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkerContractError(
            f"official worker contract is unreadable: {contract}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise WorkerContractError("official worker contract has an invalid structure")
    schema_version = payload.get("schema_version")
    expected_fields = {
        "schema_version",
        "run_manifest_sha256",
        "argv",
    }
    if schema_version == 2:
        expected_fields.add("gpu_bindings")
    if (
        isinstance(schema_version, bool)
        or schema_version not in {1, 2}
        or set(payload) != expected_fields
    ):
        raise WorkerContractError("official worker contract has an invalid structure")
    expected_manifest_hash = payload["run_manifest_sha256"]
    if expected_manifest_hash != _sha256(manifest):
        raise WorkerContractError("official run manifest changed after worker declaration")
    expected_argv = payload["argv"]
    if (
        not isinstance(expected_argv, list)
        or any(not isinstance(item, str) for item in expected_argv)
    ):
        raise WorkerContractError("official worker contract argv must be a string array")
    if list(argv) != expected_argv:
        raise WorkerContractError(
            "simulator arguments differ from the launcher-owned worker contract"
        )
    recorded_gpu_bindings = None
    if schema_version == 2:
        recorded_gpu_bindings = payload["gpu_bindings"]
        if not isinstance(recorded_gpu_bindings, Mapping):
            raise WorkerContractError("official worker GPU binding is invalid")
    _validate_dynamic_argv(
        contract,
        authority,
        resolved,
        expected_argv,
        recorded_gpu_bindings=recorded_gpu_bindings,
    )
    return payload


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    writer = subparsers.add_parser("write")
    writer.add_argument("--contract", required=True)
    writer.add_argument("--run-manifest", required=True)
    writer.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = list(args.argv)
    if arguments[:1] == ["--"]:
        arguments.pop(0)
    write_worker_contract(args.contract, args.run_manifest, arguments)
    return 0


__all__ = [
    "WorkerContractError",
    "manifest_root",
    "validate_worker_contract",
    "write_worker_contract",
]


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except WorkerContractError as exc:
        raise SystemExit(f"official worker contract error: {exc}") from exc
