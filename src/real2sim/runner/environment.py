# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Build the sealed environment passed to the batch runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from real2sim.config.keys import _ROBOT_CLI_NAMES
from real2sim.config.paths import RUN_MANIFEST_FILE
from real2sim.gpu_binding import (
    CUDA_VISIBLE_DEVICES_ENV,
    GPU_BINDINGS_ENV,
    contract_json,
    validate_gpu_contract,
)

RUNTIME_SOURCE_MANIFEST_ENV = "REAL2SIM_RUNTIME_SOURCE_MANIFEST"
RUNTIME_SOURCE_MANIFEST_SHA256_ENV = "REAL2SIM_RUNTIME_SOURCE_MANIFEST_SHA256"

# Set when the launcher could not seal this run and decided to start anyway.
# The absence of the two variables above cannot carry that decision: the runner
# treats a missing manifest as "derive one", which would seal the tree that had
# already moved and report it verified. This says the omission was deliberate.
RUNTIME_SOURCE_UNSEALED_ENV = "REAL2SIM_RUNTIME_SOURCE_UNSEALED"

# Launcher variables that earlier versions exported and this one neither sets
# nor reads. Run manifests that still record them are replayed and re-rendered
# with these treated as historical fields rather than drift. OpenPI itself is
# checked by real2sim.locked_imports in every process that imports it,
# whatever these fields say. Names match exactly: a name that differs in case
# or in any underscore is a different key and still drift.
RETIRED_RUNNER_ENVIRONMENT = frozenset({"OPENPI_SRC"})
_ENVIRONMENT_NAME = re.compile(r"[A-Z][A-Z0-9_]*")


def is_retired_runner_environment_key(key: object) -> bool:
    """Whether ``key`` is exactly a launcher variable an earlier version exported."""

    if not isinstance(key, str) or _ENVIRONMENT_NAME.fullmatch(key) is None:
        return False
    return key in RETIRED_RUNNER_ENVIRONMENT


def without_retired_environment(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recorded launcher environment without its retired fields."""

    return {
        key: value
        for key, value in environment.items()
        if not is_retired_runner_environment_key(key)
    }

def build_runner_environment(
    plan: Mapping[str, Any],
    *,
    gpu_bindings: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    resolved = plan["resolved"]
    runner = resolved["runner"]
    policy = resolved["policy"]
    task = resolved["task"]
    simulation = resolved["simulation"]
    render = resolved["render"]
    runtime = resolved["runtime"]
    environment = dict(resolved["reproducibility"]["environment"])
    environment.pop(GPU_BINDINGS_ENV, None)
    # Per-run layout selection is launcher-owned. An allowlisted or ambient
    # value must never turn a scalar profile into list mode.
    environment.pop("BOX_PACK_LAYOUT_JSON_LIST", None)
    repo_root = Path(plan["repo_root"])
    environment_bin = Path(policy["python"]).parent
    environment["PATH"] = os.pathsep.join(
        (str(environment_bin), environment.get("PATH", ""))
    ).rstrip(os.pathsep)

    def environment_path(name: str, default: Path) -> str:
        candidate = Path(environment.get(name) or default).expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        return str(Path(os.path.abspath(candidate)))

    values: dict[str, Any] = {
        "ROBOT": _ROBOT_CLI_NAMES[resolved["robot"]],
        "OPENPI_DATA_HOME": environment_path(
            "OPENPI_DATA_HOME", repo_root / "runtime-cache" / "openpi"
        ),
        "PY": policy["python"],
        "CKPT": policy["checkpoint"],
        "CONFIG": policy["config_name"],
        "CONFIG_SRC": policy["config_source"],
        "OBJECTS": task["objects"],
        "INSTRUCTION": task["instruction"],
        "POLICY_IMAGE_EPISODE": task["policy_image_episode"] or "",
        "GOAL_CONDITION": task["goal_condition"],
        "SCORE_CONDITION": task["score_condition"],
        "RUN_ROOT": str(plan["run_root"]),
        "REPEATS": runner["repeats"],
        "PARALLEL_REPEATS": runner["parallel_repeats"],
        "POLICY_BATCH_SIZE": runner["policy_batch_size"],
        "FRAMES": runner["frames"],
        "MAX_WALL_SEC": runner["max_wall_sec"],
        "ACTION_HORIZON": policy["action_horizon"],
        "EXEC_STEPS": policy["exec_steps"],
        "POLICY_NOISE_SEED": "" if policy["noise_seed"] is None else policy["noise_seed"],
        "POLICY_DETERMINISTIC_XLA": int(policy["deterministic_xla"]),
        "DETERMINISTIC_SIM": int(simulation["deterministic"]),
        "DETERMINISTIC_SEED": simulation["seed"],
        "INIT_STATE_JSON": simulation["init_state"] or "",
        "RF_OVERLAP": simulation["rf_overlap"],
        "RF_OVERLAP_SAMPLER": simulation["rf_overlap_sampler"],
        "RF_OVERLAP_PREV_WEIGHT": (
            ""
            if simulation["rf_overlap_prev_weight"] is None
            else simulation["rf_overlap_prev_weight"]
        ),
        "RF_OVERLAP_STEP_SHIFT": simulation["rf_overlap_step_shift"],
        "GRIP_TORQUE_MULT": simulation["grip_torque_mult"],
        "GRASP_MODE": simulation["grasp_mode"],
        "GRIPPER_CALIBRATION": simulation["gripper_calibration"] or "",
        "EXTRA_SIM_ARGS": "",
        "EXTRA_SIM_ARGS_JSON": json.dumps(
            simulation["cli_args"], ensure_ascii=False, separators=(",", ":")
        ),
        "WHITE_DESK_TEX": render["desk_texture"] or "",
        "VIEW_AZIMUTH": render["view_azimuth"],
        "VIEW_ELEVATION": render["view_elevation"],
        "VIEW_DISTANCE": render["view_distance"],
        "VIEW_LOOKAT": " ".join(str(item) for item in render["view_lookat"]),
        "RENDER_WRIST": int(render["wrist"]),
        "WRIST_GS": int(render["wrist_gs"]),
        "SHOW_ORIGIN": int(render["origin"]),
        "RUN_ANALYSIS": int(render["analysis"]),
        "CLEAN_VIDEO": int(render["clean_video"]),
        "POSTHOC_RENDER_REQUIRED": int(render["posthoc_required"]),
        "SERVER_HOST": runtime["server_host"],
        "SERVER_PORT": runtime["server_port"],
        "GS_PORT_BASE": runtime["gs_port_base"],
        "GS_START_TIMEOUT_SEC": runtime["gs_start_timeout_sec"],
        "GS_BG_COLOR": " ".join(
            repr(float(value)) for value in runtime["gs_background_color"]
        ),
        "SIM_NUM_THREADS": runtime["sim_num_threads"],
        "GAUSSIAN_WRAPPING_REPO": runtime["gaussian_wrapping_repo"],
        "GS_SCENE_DIRS": os.pathsep.join(scene["path"] for scene in resolved["scenes"]),
    }
    box_layouts = simulation["parameters"].get("box_pack_layout_json")
    values["BOX_PACK_LAYOUT_JSON_LIST"] = (
        json.dumps(box_layouts, ensure_ascii=False, separators=(",", ":"))
        if isinstance(box_layouts, list)
        else ""
    )
    gs_cuda_devices = runtime["gs_cuda_devices"] or ""
    # Pin every batch-runner input that otherwise has a shell default.
    # This makes the YAML-derived environment the complete execution
    # contract and lets the runner reject ambient overrides reliably.
    values.update(
        {
            "EXPORTS_ROOT": str(repo_root / "outputs"),
            "INIT_STATE_ROOT": "",
            "INIT_STATE_CACHE_SCHEMA": "initv5_named_task_object_settle",
            "MIMICGEN_INIT_DATASET_ROOT": "",
            "MIMICGEN_INIT_STATE_LOG_ROOT": "",
            "MIMICGEN_INIT_EPISODE_OFFSET": "0",
            "POLICY_NOISE_BANK_DIR": "",
            "POLICY_INFERENCE_MANIFEST": "",
            "GRID_TEMPLATE": render["grid_template"],
            "GS_CUDA_DEVICES": gs_cuda_devices,
            "GS_SERVERS_PER_GPU": str(runtime.get("gs_servers_per_gpu", 1)),
            "POLICY_CUDA_DEVICE": (
                runtime["policy_cuda_device"]
                or gs_cuda_devices.split(",", 1)[0]
            ),
            "SIM_CUDA_DEVICES": (
                runtime["sim_cuda_devices"] or gs_cuda_devices
            ),
            "RENDER_LOCK_ROOT": runtime["render_lock_root"]
            or f"/tmp/real2sim-render-locks-{os.getuid()}",
            "REAL2SIM_STRICT_LAUNCH": "1",
            "REAL2SIM_OFFICIAL_RUN_MANIFEST": str(
                Path(plan["run_root"]) / RUN_MANIFEST_FILE
            ),
        }
    )
    if runner["mode"] == "fast_parallel":
        batch_client_order = [
            f"{scene['tag']}_run_{repeat:02d}"
            for scene in resolved["scenes"]
            for repeat in range(1, runner["repeats"] + 1)
        ]
    else:
        batch_client_order = [scene["tag"] for scene in resolved["scenes"]]
    values["BATCH_CLIENT_ORDER"] = ",".join(batch_client_order)
    environment.update({key: str(value) for key, value in values.items()})
    if gpu_bindings is not None:
        environment[GPU_BINDINGS_ENV] = contract_json(
            validate_gpu_contract(gpu_bindings, environment)
        )
    return environment


def build_shard_environment(
    plan: Mapping[str, Any],
    shard: Mapping[str, Any],
    *,
    aggregate: bool = False,
) -> dict[str, str]:
    """Build one target-allocation environment from a resolved shard."""

    environment = build_runner_environment(plan)
    # The launcher's allocation boundary is not the target srun step's boundary.
    # Let Slurm install the target CUDA visibility and let the batch runner seal
    # its exact CUDA/EGL mapping after the step begins.
    environment.pop(CUDA_VISIBLE_DEVICES_ENV, None)
    environment.pop(GPU_BINDINGS_ENV, None)
    runner = plan["resolved"]["runner"]
    scenes = plan["resolved"]["scenes"]
    repeat_start = 1 if aggregate else int(shard["repeat_start"])
    repeat_stop = int(runner["repeats"]) if aggregate else int(shard["repeat_stop"])
    environment.update(
        {
            "REPEAT_START": str(repeat_start),
            "REPEAT_STOP": str(repeat_stop),
            "PARALLEL_REPEATS": str(shard["parallel_repeats"]),
            "POLICY_BATCH_SIZE": str(shard["policy_batch_size"]),
            "SERVER_PORT": str(shard["server_port"]),
            "GS_PORT_BASE": str(shard["gs_port_base"]),
            "GS_CUDA_DEVICES": str(shard["gs_cuda_devices"]),
            # servers_per_gpu is read from runtime, as in the unsharded
            # branch above, so this keeps what that branch set.
            "GS_SERVERS_PER_GPU": environment["GS_SERVERS_PER_GPU"],
            "POLICY_CUDA_DEVICE": str(shard["policy_cuda_device"]),
            "SIM_CUDA_DEVICES": str(shard["sim_cuda_devices"]),
            "REAL2SIM_SHARD_NAME": "aggregate" if aggregate else str(shard["name"]),
            "REAL2SIM_AGGREGATE_ONLY": "1" if aggregate else "0",
        }
    )
    if runner["mode"] == "fast_parallel":
        environment["BATCH_CLIENT_ORDER"] = ",".join(
            f"{scene['tag']}_run_{repeat:02d}"
            for scene in scenes
            for repeat in range(repeat_start, repeat_stop + 1)
        )
    else:
        environment["BATCH_CLIENT_ORDER"] = ",".join(
            scene["tag"] for scene in scenes
        )
    return environment


def build_all_shard_environments(
    plan: Mapping[str, Any],
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Return every generation environment and the one aggregation environment."""

    shards = plan["resolved"]["runner"].get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("plan has no resolved rollout shards")
    generation = {
        str(shard["name"]): build_shard_environment(plan, shard)
        for shard in shards
    }
    aggregate = build_shard_environment(plan, shards[0], aggregate=True)
    return generation, aggregate
