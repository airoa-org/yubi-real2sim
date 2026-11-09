# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""YAML loading and resolved rollout plan construction."""

from __future__ import annotations

from datetime import datetime
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from real2sim._bootstrap import REPO_ROOT, yaml
from real2sim.config._types import (
    ConfigError,
    _boolean,
    _integer,
    _mapping,
    _number,
    _optional_integer,
    _optional_number,
    _optional_string,
    _string,
    _unknown_key_error,
)
from real2sim.config.controllers import (
    _resolve_controller_selectors,
    _validate_nullspace_selection,
)
from real2sim.config.keys import (
    _CONFIG_SOURCES,
    _REQUIRED_SECTION_KEYS,
    _RETIRED_SECTION_KEYS,
    _ROBOT_CLI_NAMES,
    _SCENE_KEYS,
    _SECTION_KEYS,
    _TOP_LEVEL_KEYS,
)
from real2sim.config.parameters import (
    _validate_v2_parameters,
    resolve_policy_state_quat_hemisphere_parameter,
)
from real2sim.config.paths import (
    _ENV_REFERENCE_RE,
    GRID_TEMPLATES,
    POLICY_EVAL_GRID_TEMPLATE,
    POSTHOC_RENDER_BATCH_THRESHOLD,
    RUNNERS,
    _render_template,
    _resolved_path,
)
from real2sim.config.reproducibility import (
    _validate_reproducibility,
    reject_secret_like_keys,
)
from real2sim.config.shards import resolve_shards
from real2sim.config.task_objects import validate_task_objects
from real2sim.fr3_controllers import (
    FR3_IMPEDANCE_HOOK_CONTROLLERS,
    fr3_uses_cmd_twist_torque,
    fr3_uses_impedance_hook,
    validate_fr3_controller,
)


class _RawConfig(dict[str, Any]):
    """Parsed config mapping carrying the bytes it was loaded from."""

    __slots__ = ("raw_bytes",)

    def __init__(self, values: Mapping[str, Any], raw_bytes: bytes) -> None:
        super().__init__(values)
        self.raw_bytes = raw_bytes


_SCHEMA_V2_REJECTED_SECTION_KEYS = {
    "runner": frozenset({"parallel_repeats"}),
    "simulation": frozenset({"extra_args", "grasp_mode"}),
    "render": frozenset({"desk_texture"}),
}


def _accepted_section_keys(name: str) -> frozenset[str]:
    return _SECTION_KEYS[name] - _SCHEMA_V2_REJECTED_SECTION_KEYS.get(name, frozenset())


def _runtime_port(
    value: Any,
    path: str,
    context: Mapping[str, Any],
) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return _integer(value, path, minimum=1, maximum=65535)
    if isinstance(value, str):
        reference = _ENV_REFERENCE_RE.fullmatch(value)
        if reference is None or reference.group("braced") is None:
            raise ConfigError(
                f"{path} must be an integer or a ${{VAR}} template",
                path=path,
                got=value,
            )
        rendered = _render_template(value, path, context)
        try:
            parsed = int(rendered)
        except ValueError as exc:
            raise ConfigError(
                f"{path} must render to an integer",
                path=path,
                got=rendered,
            ) from exc
        return _integer(parsed, path, minimum=1, maximum=65535)
    return _integer(value, path, minimum=1, maximum=65535)


def _section(
    config: Mapping[str, Any],
    name: str,
    *,
    required_keys: frozenset[str] | None = None,
) -> dict[str, Any]:
    value = _mapping(config.get(name), name)
    retired = _RETIRED_SECTION_KEYS.get(name, {})
    for key in sorted(set(value) & retired.keys()):
        raise ConfigError(retired[key])
    allowed_keys = _accepted_section_keys(name)
    unknown = sorted(set(value) - _SECTION_KEYS[name])
    missing = sorted((required_keys or _REQUIRED_SECTION_KEYS[name]) - set(value))
    if unknown:
        raise _unknown_key_error(
            f"{name} contains unknown keys: {', '.join(unknown)}",
            path=name,
            allowed=sorted(allowed_keys),
            got=unknown,
        )
    if missing:
        raise ConfigError(
            f"{name} is missing required keys: {', '.join(missing)}",
            path=name,
            allowed=sorted(allowed_keys),
            got=missing,
        )
    return value

def _resolve_plan(
    config: Mapping[str, Any],
    *,
    raw_bytes: bytes,
    path: str | Path,
    repo_root: Path = REPO_ROOT,
    timestamp: str | None = None,
    shards_override: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load, validate, resolve, and derive one rollout launch plan."""

    path = Path(path).expanduser().resolve()
    reject_secret_like_keys(config)
    schema_version = _integer(config.get("schema_version"), "schema_version", minimum=1)
    if schema_version != 2:
        raise ConfigError(
            "schema_version must be 2 (schema v1 was removed; see docs/configuration.md)",
            path="schema_version",
        )
    unknown = sorted(set(config) - _TOP_LEVEL_KEYS)
    optional_top_level = {"description", "robot"}
    missing = sorted(_TOP_LEVEL_KEYS - optional_top_level - set(config))
    if unknown:
        raise _unknown_key_error(
            f"config contains unknown keys: {', '.join(unknown)}",
            path="config",
            allowed=sorted(_TOP_LEVEL_KEYS),
            got=unknown,
        )
    if missing:
        raise ConfigError(
            f"config is missing required keys: {', '.join(missing)}",
            path="config",
            allowed=sorted(_TOP_LEVEL_KEYS),
            got=missing,
        )

    name = _string(config["name"], "name")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,63}", name) is None:
        raise ConfigError(
            "name must be 3-64 lowercase letters, digits, underscores, or hyphens",
            path="name",
        )
    description = _string(config.get("description", ""), "description", allow_empty=True)
    robot = _string(config.get("robot", "fr3"), "robot")
    if robot not in _ROBOT_CLI_NAMES:
        raise ConfigError(
            f"robot must be one of: {', '.join(sorted(_ROBOT_CLI_NAMES))}",
            path="robot",
            allowed=sorted(_ROBOT_CLI_NAMES),
            got=robot,
        )

    runner_required = _REQUIRED_SECTION_KEYS["runner"] - frozenset({"parallel_repeats"})
    runner = _section(config, "runner", required_keys=runner_required)
    if shards_override is not None:
        runner = dict(runner)
        runner["shards"] = [dict(item) for item in shards_override]
    policy = _section(config, "policy")
    task = _section(config, "task")
    try:
        validate_task_objects(_string(task["objects"], "task.objects"))
    except ValueError as exc:
        raise ConfigError(str(exc), path="task.objects") from exc
    simulation_required = (
        _REQUIRED_SECTION_KEYS["simulation"] - frozenset({"extra_args"}) | frozenset({"parameters"})
    )
    render_required = _REQUIRED_SECTION_KEYS["render"] - frozenset({"desk_texture"})
    runtime_required = _REQUIRED_SECTION_KEYS["runtime"]
    simulation = _section(config, "simulation", required_keys=simulation_required)
    render = _section(config, "render", required_keys=render_required)
    runtime = _section(config, "runtime", required_keys=runtime_required)
    output = _section(config, "output")
    if "parallel_repeats" in runner:
        raise ConfigError(
            "schema_version: 2 uses runner.policy_batch_size; "
            "parallel repeats per scene are derived",
            path="runner.parallel_repeats",
        )
    if "extra_args" in simulation:
        raise ConfigError(
            "schema_version: 2 forbids simulation.extra_args",
            path="simulation.extra_args",
        )
    if "grasp_mode" in simulation:
        raise ConfigError(
            "schema_version: 2 stores grasp_mode in simulation.parameters",
            path="simulation.grasp_mode",
        )
    if "desk_texture" in render:
        raise ConfigError(
            "schema_version: 2 forbids render.desk_texture; table appearance "
            "is embedded in the 3DGS scene",
            path="render.desk_texture",
        )

    effective_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    context = {
        "repo_root": str(repo_root.resolve()),
        "config_dir": str(path.parent),
        "timestamp": effective_timestamp,
        "robot": robot,
    }
    reproducibility = _validate_reproducibility(
        config["reproducibility"],
        raw_config=config,
        context=context,
        config_dir=path.parent,
    )
    context["_environment"] = reproducibility["environment"]
    context["inputs_root"] = reproducibility["inputs_root"]

    mode = _string(runner["mode"], "runner.mode")
    if mode not in RUNNERS:
        raise ConfigError(
            f"runner.mode must be one of: {', '.join(sorted(RUNNERS))}",
            path="runner.mode",
            allowed=sorted(RUNNERS),
            got=mode,
        )
    repeats = _integer(runner["repeats"], "runner.repeats", minimum=1)
    configured_policy_batch_size = _integer(
        runner["policy_batch_size"],
        "runner.policy_batch_size",
        minimum=1,
    )
    frames = _integer(runner["frames"], "runner.frames", minimum=1)
    max_wall_sec = _integer(runner["max_wall_sec"], "runner.max_wall_sec", minimum=1)

    raw_scenes = config["scenes"]
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ConfigError("scenes must be a non-empty YAML list", path="scenes")
    scenes: list[dict[str, str]] = []
    seen_tags: set[str] = set()
    seen_paths: set[str] = set()
    for index, raw_scene in enumerate(raw_scenes):
        scene = _mapping(raw_scene, f"scenes[{index}]")
        unknown_scene = sorted(set(scene) - _SCENE_KEYS)
        missing_scene = sorted(_SCENE_KEYS - set(scene))
        if unknown_scene or missing_scene:
            detail = []
            if unknown_scene:
                detail.append(f"unknown={unknown_scene}")
            if missing_scene:
                detail.append(f"missing={missing_scene}")
            raise ConfigError(
                f"invalid scenes[{index}] keys: {' '.join(detail)}",
                path=f"scenes[{index}]",
                allowed=sorted(_SCENE_KEYS),
                got=unknown_scene or missing_scene,
            )
        tag = _string(scene["tag"], f"scenes[{index}].tag")
        if re.fullmatch(r"[a-z0-9][a-z0-9_]{1,95}", tag) is None:
            raise ConfigError(f"scenes[{index}].tag must be a lowercase runner scene tag")
        scene_path = _resolved_path(
            scene["path"], f"scenes[{index}].path", context, path.parent
        )
        if os.pathsep in str(scene_path):
            raise ConfigError(
                f"scenes[{index}].path must not contain the runner delimiter {os.pathsep!r}"
            )
        if tag in seen_tags or str(scene_path) in seen_paths:
            raise ConfigError("scene tags and resolved paths must be unique")
        seen_tags.add(tag)
        seen_paths.add(str(scene_path))
        scenes.append({"tag": tag, "path": str(scene_path)})

    policy_batch_size = configured_policy_batch_size
    if policy_batch_size % len(scenes) != 0:
        raise ConfigError(
            "runner.policy_batch_size must be divisible by len(scenes) so "
            "each comparison scene has equal concurrency",
            path="runner.policy_batch_size",
        )
    parallel_repeats = policy_batch_size // len(scenes)
    if parallel_repeats > repeats or repeats % parallel_repeats != 0:
        raise ConfigError(
            "runner.repeats must be divisible by and >= "
            "runner.policy_batch_size / len(scenes)",
            path="runner.repeats",
        )
    if mode == "sequential_waves" and parallel_repeats != 1:
        raise ConfigError(
            "sequential_waves requires runner.policy_batch_size: len(scenes)",
            path="runner.policy_batch_size",
        )

    expected_rollouts = _integer(
        runner["expected_rollouts"], "runner.expected_rollouts", minimum=1
    )
    derived_rollouts = len(scenes) * repeats
    if expected_rollouts != derived_rollouts:
        raise ConfigError(
            "runner.expected_rollouts must equal len(scenes) * repeats "
            f"({len(scenes)} * {repeats} = {derived_rollouts})",
            path="runner.expected_rollouts",
        )

    action_horizon = _integer(
        policy["action_horizon"], "policy.action_horizon", minimum=1
    )
    exec_steps = _integer(policy["exec_steps"], "policy.exec_steps", minimum=1)
    rf_overlap = _integer(simulation["rf_overlap"], "simulation.rf_overlap", minimum=0)
    rf_sampler = _string(simulation["rf_overlap_sampler"], "simulation.rf_overlap_sampler")
    if rf_sampler != "single":
        raise ConfigError(
            "schema_version 2 requires simulation.rf_overlap_sampler: single so "
            "runner.policy_batch_size is the exact policy model batch size",
            path="simulation.rf_overlap_sampler",
        )
    rf_prev_weight = _optional_number(
        simulation["rf_overlap_prev_weight"],
        "simulation.rf_overlap_prev_weight",
        minimum=0.0,
    )
    if rf_prev_weight is not None and rf_prev_weight > 1.0:
        raise ConfigError(
            "simulation.rf_overlap_prev_weight must be <= 1",
            path="simulation.rf_overlap_prev_weight",
        )
    rf_step_shift = _integer(
        simulation["rf_overlap_step_shift"],
        "simulation.rf_overlap_step_shift",
        minimum=0,
    )
    if rf_overlap > exec_steps:
        raise ConfigError(
            "simulation.rf_overlap must not exceed policy.exec_steps",
            path="simulation.rf_overlap",
        )
    minimum_horizon = exec_steps + rf_overlap
    if action_horizon < minimum_horizon:
        raise ConfigError(
            f"policy.action_horizon must be >= {minimum_horizon} for the selected overlap mode",
            path="policy.action_horizon",
        )
    if rf_step_shift != 0:
        raise ConfigError(
            "single overlap sampling requires simulation.rf_overlap_step_shift: 0",
            path="simulation.rf_overlap_step_shift",
        )
    if rf_prev_weight is not None:
        raise ConfigError(
            "single overlap sampling requires simulation.rf_overlap_prev_weight: null",
            path="simulation.rf_overlap_prev_weight",
        )

    config_name = _string(policy["config_name"], "policy.config_name")
    config_source = _string(policy["config_source"], "policy.config_source")
    if config_source not in _CONFIG_SOURCES:
        raise ConfigError(
            "policy.config_source must be deploy or training",
            path="policy.config_source",
            allowed=sorted(_CONFIG_SOURCES),
            got=config_source,
        )
    python_path = _resolved_path(policy["python"], "policy.python", context, path.parent)
    checkpoint = _resolved_path(
        policy["checkpoint"], "policy.checkpoint", context, path.parent
    )
    noise_seed = _optional_integer(policy["noise_seed"], "policy.noise_seed", minimum=0)
    policy_image_episode_raw = _optional_string(
        task.get("policy_image_episode"), "task.policy_image_episode"
    )
    policy_image_episode = (
        str(
            _resolved_path(
                policy_image_episode_raw,
                "task.policy_image_episode",
                context,
                path.parent,
            )
        )
        if policy_image_episode_raw is not None
        else None
    )
    init_state_value = simulation["init_state"]
    if isinstance(init_state_value, Mapping):
        init_state_keys = set(init_state_value)
        expected_robot_keys = set(_ROBOT_CLI_NAMES)
        if init_state_keys != expected_robot_keys:
            raise ConfigError(
                "simulation.init_state robot mapping must contain exactly: "
                + ", ".join(sorted(expected_robot_keys))
            )
        init_state_value = init_state_value[robot]
    init_state_raw = _optional_string(
        init_state_value,
        f"simulation.init_state[{robot}]" if isinstance(simulation["init_state"], Mapping)
        else "simulation.init_state",
    )
    init_state = (
        str(_resolved_path(init_state_raw, "simulation.init_state", context, path.parent))
        if init_state_raw is not None
        else None
    )
    gripper_calibration_raw = _optional_string(
        simulation.get("gripper_calibration"), "simulation.gripper_calibration"
    )
    gripper_calibration = (
        str(
            _resolved_path(
                gripper_calibration_raw,
                "simulation.gripper_calibration",
                context,
                path.parent,
            )
        )
        if gripper_calibration_raw is not None
        else None
    )
    desk_texture = None
    gaussian_repo = _resolved_path(
        runtime["gaussian_wrapping_repo"],
        "runtime.gaussian_wrapping_repo",
        context,
        path.parent,
    )
    gs_cuda_devices_raw = _optional_string(
        runtime.get("gs_cuda_devices"), "runtime.gs_cuda_devices"
    )
    gs_cuda_devices = (
        _render_template(
            gs_cuda_devices_raw, "runtime.gs_cuda_devices", context
        )
        if gs_cuda_devices_raw is not None
        else None
    )
    if gs_cuda_devices is not None and re.fullmatch(
        r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", gs_cuda_devices
    ) is None:
        raise ConfigError(
            "runtime.gs_cuda_devices must be a comma-separated list of canonical non-negative GPU selectors",
            path="runtime.gs_cuda_devices",
        )
    if gs_cuda_devices is not None and len(set(gs_cuda_devices.split(","))) != len(
        gs_cuda_devices.split(",")
    ):
        raise ConfigError(
            "runtime.gs_cuda_devices must not contain duplicate GPU selectors",
            path="runtime.gs_cuda_devices",
        )
    gs_servers_per_gpu = _integer(
        runtime.get("gs_servers_per_gpu", 1),
        "runtime.gs_servers_per_gpu",
        minimum=1,
    )
    policy_cuda_device_raw = _optional_string(
        runtime.get("policy_cuda_device"), "runtime.policy_cuda_device"
    )
    policy_cuda_device = (
        _render_template(
            policy_cuda_device_raw, "runtime.policy_cuda_device", context
        )
        if policy_cuda_device_raw is not None
        else None
    )
    if policy_cuda_device is not None and re.fullmatch(
        r"0|[1-9][0-9]*", policy_cuda_device
    ) is None:
        raise ConfigError(
            "runtime.policy_cuda_device must be one canonical non-negative GPU selector",
            path="runtime.policy_cuda_device",
        )
    sim_cuda_devices_raw = _optional_string(
        runtime.get("sim_cuda_devices"), "runtime.sim_cuda_devices"
    )
    sim_cuda_devices = (
        _render_template(
            sim_cuda_devices_raw, "runtime.sim_cuda_devices", context
        )
        if sim_cuda_devices_raw is not None
        else None
    )
    if sim_cuda_devices is not None and re.fullmatch(
        r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", sim_cuda_devices
    ) is None:
        raise ConfigError(
            "runtime.sim_cuda_devices must be a comma-separated list of canonical non-negative GPU selectors",
            path="runtime.sim_cuda_devices",
        )
    if sim_cuda_devices is not None and len(set(sim_cuda_devices.split(","))) != len(
        sim_cuda_devices.split(",")
    ):
        raise ConfigError(
            "runtime.sim_cuda_devices must not contain duplicate GPU selectors",
            path="runtime.sim_cuda_devices",
        )
    render_lock_root_raw = _optional_string(
        runtime.get("render_lock_root"), "runtime.render_lock_root"
    )
    render_lock_root = (
        str(
            _resolved_path(
                render_lock_root_raw,
                "runtime.render_lock_root",
                context,
                path.parent,
            )
        )
        if render_lock_root_raw is not None
        else None
    )
    if reproducibility["production"]:
        missing_devices = [
            name
            for name, value in (
                ("gs_cuda_devices", gs_cuda_devices),
                ("sim_cuda_devices", sim_cuda_devices),
            )
            if value is None
        ]
        if missing_devices:
            raise ConfigError(
                "production schema v2 requires explicit runtime GPU assignments: "
                + ", ".join(missing_devices)
            )
    output_root = _resolved_path(output["root"], "output.root", context, path.parent)
    run_name = _render_template(output["run_name"], "output.run_name", context)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,127}", run_name) is None:
        raise ConfigError(
            "output.run_name must render to one safe path component",
            path="output.run_name",
        )
    run_root = output_root / run_name

    server_port = _runtime_port(
        runtime["server_port"], "runtime.server_port", context
    )
    gs_port_base = _runtime_port(
        runtime["gs_port_base"], "runtime.gs_port_base", context
    )
    resolved_shards = None
    if "shards" in runner:
        resolved_shards = resolve_shards(
            runner["shards"],
            repeats=repeats,
            scene_count=len(scenes),
            runner_mode=mode,
            default_policy_batch_size=configured_policy_batch_size,
            default_server_port=server_port,
            default_gs_port_base=gs_port_base,
            gs_servers_per_gpu=gs_servers_per_gpu,
        )

    view_lookat_raw = render["view_lookat"]
    if not isinstance(view_lookat_raw, list) or len(view_lookat_raw) != 3:
        raise ConfigError(
            "render.view_lookat must contain exactly three numbers",
            path="render.view_lookat",
        )
    view_lookat = [
        _number(item, f"render.view_lookat[{index}]")
        for index, item in enumerate(view_lookat_raw)
    ]
    render_wrist = _boolean(render["wrist"], "render.wrist")
    render_wrist_gs = _boolean(render["wrist_gs"], "render.wrist_gs")
    render_clean_video = _boolean(render["clean_video"], "render.clean_video")
    render_requested_enabled = _boolean(
        render.get("enabled", True), "render.enabled"
    )
    # Optional post-hoc third-person opt-out; absent means "render it".
    render_third_person = _boolean(
        render.get("third_person", True), "render.third_person"
    )
    grid_template = GRID_TEMPLATES.get(repeats, GRID_TEMPLATES[10])
    if render_wrist_gs and not render_wrist:
        raise ConfigError(
            "render.wrist_gs: true requires render.wrist: true",
            path="render.wrist_gs",
        )
    if render_clean_video and render_wrist:
        raise ConfigError(
            "render.clean_video: true requires render.wrist: false",
            path="render.clean_video",
        )
    typed_parameters, cli_args = _validate_v2_parameters(
        simulation["parameters"],
        context=context,
        config_dir=path.parent,
    )
    box_layouts = typed_parameters.get("box_pack_layout_json")
    if isinstance(box_layouts, list) and len(box_layouts) != repeats:
        raise ConfigError(
            "simulation.parameters.box_pack_layout_json list length must equal "
            f"runner.repeats ({repeats}); got {len(box_layouts)}",
            path="simulation.parameters.box_pack_layout_json",
            got=len(box_layouts),
        )
    if isinstance(box_layouts, list) and simulation.get("init_state") is not None:
        raise ConfigError(
            "simulation.parameters.box_pack_layout_json list requires "
            "simulation.init_state: null so each layout gets its own "
            "content-bound deterministic init state",
            path="simulation.init_state",
        )
    # ``auto`` needs the checkpoint's norm stats, and a batch-server worker
    # never loads the checkpoint, so the launcher resolves it here.
    cli_args, policy_state_quat_hemisphere = (
        resolve_policy_state_quat_hemisphere_parameter(
            typed_parameters,
            cli_args,
            checkpoint=checkpoint,
            config_name=config_name,
            config_source=config_source,
            action_horizon=action_horizon,
            repo_root=repo_root,
        )
    )
    # ``requested_enabled`` is final-video intent, not merely the YAML field.
    # An explicit ``--no-render`` selector is an explicit opt-out and
    # therefore disables both live and deferred rendering.
    render_requested_enabled = bool(
        render_requested_enabled and "--no-render" not in cli_args
    )
    render_explicitly_disabled = not render_requested_enabled
    render_batch_size = max(
        [int(shard["policy_batch_size"]) for shard in resolved_shards]
        if resolved_shards is not None
        else [policy_batch_size]
    )
    render_forced_posthoc = bool(
        not render_explicitly_disabled
        and render_batch_size >= POSTHOC_RENDER_BATCH_THRESHOLD
    )
    if render_forced_posthoc and repeats == 10:
        # 10-rollout (and larger) policy batches are finalized post-hoc by scripts/run_policy_batch.sh
        # with the policy-eval template (checkpoint/instruction info panel),
        # not the clear-headers template used for live rendering.
        grid_template = POLICY_EVAL_GRID_TEMPLATE
    render_enabled = bool(
        render_requested_enabled
        and "--no-render" not in cli_args
        and not render_forced_posthoc
    )
    if render_requested_enabled and repeats not in GRID_TEMPLATES:
        raise ConfigError(
            "video-producing schema-v2 runs require a checked-in per-scene "
            "grid template; "
            f"supported repeat counts are {', '.join(map(str, sorted(GRID_TEMPLATES)))}",
            path="runner.repeats",
        )
    grasp_mode = typed_parameters["grasp_mode"]
    controllers, cli_args = _resolve_controller_selectors(simulation, cli_args)
    _validate_nullspace_selection(controllers, typed_parameters)
    try:
        validate_fr3_controller(controllers["fr3_controller"])
    except ValueError as exc:
        raise ConfigError(str(exc), path="simulation.fr3_controller") from exc
    if fr3_uses_cmd_twist_torque(controllers["fr3_controller"]):
        if "mujoco_timestep" not in typed_parameters:
            typed_parameters["mujoco_timestep"] = .001
            cli_args.extend(["--mujoco-timestep", "0.001"])
        if typed_parameters["physics"] != "dynamic" or robot not in {"fr3", "fr3_duo"}:
            raise ConfigError("Cartesian impedance requires a dynamic FR3 rig")
        if typed_parameters["policy_action_clamp"]:
            raise ConfigError("Cartesian impedance cmd_twist requires policy_action_clamp: false")
    if "fr3_impedance" in typed_parameters and not fr3_uses_impedance_hook(controllers["fr3_controller"]):
        raise ConfigError(
            f"fr3_impedance requires fr3_controller: {' or '.join(sorted(FR3_IMPEDANCE_HOOK_CONTROLLERS))}"
        )

    # Keep the execution contract canonical even when the flag is supplied
    # more than once. This also avoids duplicate flags when the post-hoc
    # policy for 10-rollout (and larger) batches adds the same setting.
    cli_args = [argument for argument in cli_args if argument != "--no-render"]
    if not render_enabled:
        cli_args.append("--no-render")

    try:
        config_repo_path = path.relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ConfigError("rollout config must live inside this repository") from exc
    config_repo_parts = Path(config_repo_path).parts
    if (
        len(config_repo_parts) < 2
        or config_repo_parts[0] != "config"
        or Path(config_repo_path).suffix not in {".yaml", ".yml"}
    ):
        raise ConfigError(
            "rollout config must be a YAML file under config/",
            path="config_path",
        )

    resolved_runner = {
        "mode": mode,
        "repeats": repeats,
        "parallel_repeats": parallel_repeats,
        "policy_batch_size": policy_batch_size,
        "frames": frames,
        "max_wall_sec": max_wall_sec,
        "expected_rollouts": expected_rollouts,
    }
    if resolved_shards is not None:
        resolved_runner["shards"] = resolved_shards

    # What the renderer paints where no gaussian covers a pixel. Carving splats
    # out of a scene turns those holes into far field, so the colour is part of
    # what a rollout was scored under and belongs in the sealed config.
    raw_background_color = runtime.get("gs_background_color", [0.0, 0.0, 0.0])
    if not isinstance(raw_background_color, list) or len(raw_background_color) != 3:
        raise ConfigError(
            "runtime.gs_background_color must be [r, g, b]",
            path="runtime.gs_background_color",
        )
    gs_background_color = [
        _number(value, f"runtime.gs_background_color[{index}]", minimum=0.0)
        for index, value in enumerate(raw_background_color)
    ]
    if any(value > 1.0 for value in gs_background_color):
        raise ConfigError(
            "runtime.gs_background_color components must be within 0..1",
            path="runtime.gs_background_color",
        )

    resolved = {
        "schema_version": schema_version,
        "robot": robot,
        "name": name,
        "description": description,
        "runner": resolved_runner,
        "policy": {
            "python": str(python_path),
            "checkpoint": str(checkpoint),
            "config_name": config_name,
            "config_source": config_source,
            "action_horizon": action_horizon,
            "exec_steps": exec_steps,
            "noise_seed": noise_seed,
            "deterministic_xla": _boolean(
                policy["deterministic_xla"], "policy.deterministic_xla"
            ),
        },
        "task": {
            "objects": _string(task["objects"], "task.objects"),
            "instruction": _string(task["instruction"], "task.instruction"),
            "goal_condition": _string(task["goal_condition"], "task.goal_condition"),
            "score_condition": _string(task["score_condition"], "task.score_condition"),
            "policy_image_episode": policy_image_episode,
        },
        "scenes": scenes,
        "controllers": dict(controllers),
        "simulation": {
            "deterministic": _boolean(
                simulation["deterministic"], "simulation.deterministic"
            ),
            "seed": _integer(simulation["seed"], "simulation.seed", minimum=0),
            "init_state": init_state,
            "rf_overlap": rf_overlap,
            "rf_overlap_sampler": rf_sampler,
            "rf_overlap_prev_weight": rf_prev_weight,
            "rf_overlap_step_shift": rf_step_shift,
            "grip_torque_mult": _number(
                simulation["grip_torque_mult"],
                "simulation.grip_torque_mult",
                minimum=0.0,
            ),
            "grasp_mode": grasp_mode,
            **controllers,
            "gripper_calibration": gripper_calibration,
            "parameters": typed_parameters,
            # Resolution of parameters.policy_state_quat_hemisphere: the value,
            # and for ``auto`` the norm stats (path, sha256, qw q01/q99) it came from.
            "policy_state_quat_hemisphere": policy_state_quat_hemisphere,
            "cli_args": cli_args,
        },
        "render": {
            "desk_texture": str(desk_texture) if desk_texture is not None else None,
            "view_azimuth": _number(render["view_azimuth"], "render.view_azimuth"),
            "view_elevation": _number(render["view_elevation"], "render.view_elevation"),
            "view_distance": _number(
                render["view_distance"], "render.view_distance", minimum=0.01
            ),
            "view_lookat": view_lookat,
            "wrist": render_wrist,
            "wrist_gs": render_wrist_gs,
            "origin": _boolean(render["origin"], "render.origin"),
            "analysis": _boolean(render["analysis"], "render.analysis"),
            "clean_video": render_clean_video,
            "requested_enabled": render_requested_enabled,
            "enabled": render_enabled,
            "third_person": render_third_person,
            "forced_posthoc": render_forced_posthoc,
            # ``render.third_person: false`` opts out of the deferred
            # third-person pass even when a 10-rollout (or larger) batch forces post-hoc
            # rendering; the batch runner then finalizes without video.
            "posthoc_required": bool(
                render_forced_posthoc and render_third_person
            ),
            "grid_template": str(grid_template),
        },
        "runtime": {
            "server_host": _string(runtime["server_host"], "runtime.server_host"),
            "server_port": server_port,
            "gs_port_base": gs_port_base,
            "gs_cuda_devices": gs_cuda_devices,
            "gs_servers_per_gpu": gs_servers_per_gpu,
            "policy_cuda_device": policy_cuda_device,
            "sim_cuda_devices": sim_cuda_devices,
            "render_lock_root": render_lock_root,
            "gs_start_timeout_sec": _integer(
                runtime["gs_start_timeout_sec"],
                "runtime.gs_start_timeout_sec",
                minimum=1,
            ),
            "gs_background_color": gs_background_color,
            "sim_num_threads": _integer(
                runtime["sim_num_threads"], "runtime.sim_num_threads", minimum=1
            ),
            "gaussian_wrapping_repo": str(gaussian_repo),
        },
        "output": {"root": str(output_root), "run_name": run_name, "run_root": str(run_root)},
        "reproducibility": reproducibility,
        "provenance": {
            "config_path": config_repo_path,
            "config_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        },
    }
    plan = {
        "config_path": path,
        "config_repo_path": config_repo_path,
        "config_sha256": resolved["provenance"]["config_sha256"],
        "timestamp": effective_timestamp,
        "repo_root": repo_root.resolve(),
        "raw_bytes": raw_bytes,
        "resolved": resolved,
        "runner_script": RUNNERS[mode],
        "run_root": run_root,
        "expected_rollouts": expected_rollouts,
    }
    if shards_override is not None:
        plan["shards_override"] = [dict(item) for item in shards_override]
    raw_shards = runner.get("shards")
    if isinstance(raw_shards, list) and any(
        isinstance(item, Mapping) and item.get("slurm_job_id") is None
        for item in raw_shards
    ):
        # Worker-authority reconstruction occurs inside an srun step, where the
        # scheduler may replace SLURM_JOB_ID. Preserve the launcher-side fallback
        # that was used to resolve the checked-in config.
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id is not None:
            plan["shard_resolution_environment"] = {
                "SLURM_JOB_ID": slurm_job_id
            }
    return plan


def load_raw_config(config_path: str | Path) -> tuple[bytes, dict[str, Any]]:
    """Read and parse a YAML config without resolving its runtime plan."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}", path="config")
    raw_bytes = path.read_bytes()
    try:
        loaded = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}", path="config") from exc
    return raw_bytes, _RawConfig(_mapping(loaded, "config"), raw_bytes)


def resolve_plan(
    config: Mapping[str, Any],
    *,
    path: str | Path,
    repo_root: Path = REPO_ROOT,
    timestamp: str | None = None,
    raw_bytes: bytes | None = None,
    shards_override: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve a parsed config mapping into the stable launch plan."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}", path="config")
    if raw_bytes is None:
        raw_bytes = getattr(config, "raw_bytes", None)
    if not isinstance(raw_bytes, bytes):
        raise ConfigError(
            "resolve_plan requires the raw bytes returned by load_raw_config",
            path="config",
        )
    return _resolve_plan(
        config,
        raw_bytes=raw_bytes,
        path=config_path,
        repo_root=repo_root,
        timestamp=timestamp,
        shards_override=shards_override,
    )


def load_rollout_config(
    config_path: str | Path,
    *,
    repo_root: Path = REPO_ROOT,
    timestamp: str | None = None,
    shards_override: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load, validate, resolve, and derive one rollout launch plan."""

    raw_bytes, config = load_raw_config(config_path)
    return _resolve_plan(
        config,
        raw_bytes=raw_bytes,
        path=config_path,
        repo_root=repo_root,
        timestamp=timestamp,
        shards_override=shards_override,
    )
