# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Stable JSON dry-run projection."""

from __future__ import annotations

from typing import Any, Mapping

from real2sim.runner.environment import build_runner_environment


def format_dry_run_table(plan: Mapping[str, Any]) -> str:
    """Render the operator-facing subset of a resolved plan as a table."""

    resolved = plan["resolved"]
    policy = resolved["policy"]
    runner = resolved["runner"]
    runtime = resolved["runtime"]
    controllers = resolved["controllers"]
    render = resolved["render"]
    environment = build_runner_environment(plan)
    rows: list[tuple[str, Any]] = [
        ("checkpoint", policy["checkpoint"]),
    ]
    rows.extend(
        (f"scene[{scene['tag']}]", scene["path"])
        for scene in resolved["scenes"]
    )
    rows.extend(
        (f"controller.{name}", controllers[name])
        for name in (
            "fr3_controller",
            "fr3_nullspace",
            "yubi_controller",
        )
    )
    rows.extend(
        [
            ("action_horizon", policy["action_horizon"]),
            ("exec_steps", policy["exec_steps"]),
            ("repeats", runner["repeats"]),
            ("policy_batch_size", runner["policy_batch_size"]),
            ("expected_rollouts", plan["expected_rollouts"]),
            ("server", f"{runtime['server_host']}:{runtime['server_port']}"),
            ("gs_port_base", runtime["gs_port_base"]),
            ("gs_cuda_devices", environment["GS_CUDA_DEVICES"]),
            ("policy_cuda_device", environment["POLICY_CUDA_DEVICE"]),
            ("sim_cuda_devices", environment["SIM_CUDA_DEVICES"]),
            ("render.requested_enabled", render["requested_enabled"]),
            ("render.enabled", render["enabled"]),
            ("render.third_person", render["third_person"]),
            ("render.posthoc_required", render["posthoc_required"]),
            ("run_root", plan["run_root"]),
        ]
    )
    label_width = max(len(label) for label, _ in rows)
    return "\n".join(
        f"{label:<{label_width}} | {value}" for label, value in rows
    ) + "\n"


def dry_run_view(plan: Mapping[str, Any]) -> dict[str, Any]:
    environment = build_runner_environment(plan)
    selected_keys = sorted(
        key
        for key in environment
        if key
        in {
            "ROBOT",
            "PY",
            "CKPT",
            "CONFIG",
            "CONFIG_SRC",
            "OBJECTS",
            "INSTRUCTION",
            "GOAL_CONDITION",
            "SCORE_CONDITION",
            "RUN_ROOT",
            "REPEATS",
            "PARALLEL_REPEATS",
            "POLICY_BATCH_SIZE",
            "FRAMES",
            "MAX_WALL_SEC",
            "ACTION_HORIZON",
            "EXEC_STEPS",
            "RF_OVERLAP",
            "RF_OVERLAP_SAMPLER",
            "RF_OVERLAP_PREV_WEIGHT",
            "RF_OVERLAP_STEP_SHIFT",
            "GRASP_MODE",
            "GRIPPER_CALIBRATION",
            "CLEAN_VIDEO",
            "POSTHOC_RENDER_REQUIRED",
            "GS_SCENE_DIRS",
            "BATCH_CLIENT_ORDER",
            "GS_CUDA_DEVICES",
            "POLICY_CUDA_DEVICE",
            "SIM_CUDA_DEVICES",
            "RENDER_LOCK_ROOT",
            "EXTRA_SIM_ARGS_JSON",
            "SERVER_HOST",
            "SERVER_PORT",
            "GS_PORT_BASE",
            "GS_START_TIMEOUT_SEC",
            "GS_BG_COLOR",
        }
    )
    view = {
        "config": str(plan["config_path"]),
        "config_sha256": plan["config_sha256"],
        "schema_version": plan["resolved"]["schema_version"],
        "reproducibility_mode": (
            plan["resolved"]["reproducibility"]["mode"]
            if plan["resolved"].get("reproducibility") is not None
            else "legacy-unverified"
        ),
        "runner": str(plan["runner_script"]),
        "run_root": str(plan["run_root"]),
        "expected_rollouts": plan["expected_rollouts"],
        "controllers": dict(plan["resolved"]["controllers"]),
        "render": {
            "requested_enabled": plan["resolved"]["render"][
                "requested_enabled"
            ],
            "enabled": plan["resolved"]["render"]["enabled"],
            "third_person": plan["resolved"]["render"]["third_person"],
            "forced_posthoc": plan["resolved"]["render"]["forced_posthoc"],
            "posthoc_required": plan["resolved"]["render"][
                "posthoc_required"
            ],
        },
        "environment": {key: environment[key] for key in selected_keys},
    }
    shards = plan["resolved"]["runner"].get("shards")
    if isinstance(shards, list):
        view["shards"] = [dict(shard) for shard in shards]
    return view
