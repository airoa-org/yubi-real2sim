# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Schema-v2 simulation parameter validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from real2sim.config._types import (
    ConfigError,
    _boolean,
    _choice,
    _integer,
    _mapping,
    _number,
    _number_list,
    _optional_string,
    _string,
    _rgb_color_transform,
    _unknown_key_error,
)
from real2sim.config.initial_arm_joints import (
    check_pose_signature,
    resolve_initial_arm_joint_pose,
)
from real2sim.config.keys import (
    _V2_OPTIONAL_PARAMETER_DEFAULTS,
    _V2_PARAMETER_KEYS,
    _V2_TASK_SCOPED_PARAMETER_KEYS,
)
from real2sim.config.paths import _resolved_path
from real2sim.locked_imports import LockedImportError
from real2sim.wrist_profiles import WRIST_BACKENDS, WRIST_RENDER_PROFILE_NAMES
from scene.gs_background_protocol import DEFAULT_KB4_NEAR_CULL

#: Values of ``simulation.parameters.policy_state_quat_hemisphere``. A resolved
#: value is one of the last two (``robosuite_env.policy.policy_io``).
POLICY_STATE_QUAT_HEMISPHERE_CHOICES = ("auto", "positive", "negative")
#: Worker option that carries a resolved hemisphere other than the default.
POLICY_STATE_QUAT_HEMISPHERE_FLAG = "--policy-state-quat-hemisphere"
_POLICY_STATE_QUAT_HEMISPHERE_PATH = "simulation.parameters.policy_state_quat_hemisphere"



#: Highest runnable task index per task family. Box packing: the six into-box
#: prompts (0-5); the registered out-of-box prompts need a different start
#: layout and are not launchable. Kept as literals so the launcher does not
#: import the simulator; a unit test pins the box value to the task module.
_TASK_INDEX_MAXIMA = {"box_pack_task_index": 5}

def resolve_policy_state_quat_hemisphere_parameter(
    typed_parameters: Mapping[str, Any],
    cli_args: Sequence[str],
    *,
    checkpoint: str | Path,
    config_name: str,
    config_source: str,
    action_horizon: int,
    repo_root: str | Path,
) -> tuple[list[str], dict[str, Any]]:
    """Resolve the state quaternion hemisphere and render its worker argument.

    ``auto`` reads the norm-stats asset selected by the OpenPI config (q99[qw]
    < 0 -> negative, q01[qw] > 0 -> positive, mixed -> positive) and fails
    closed when the selected stats are unreadable or do not confirm the qw
    index. The launcher resolves it because a batch-server worker never loads
    the checkpoint. The worker argument is rendered only for ``negative``:
    ``positive`` is the worker default, so every profile that resolves positive
    keeps its exact historical argument vector.
    """

    from robosuite_env.policy import policy_batch_infer as pbi
    from robosuite_env.policy import policy_io as pio

    configured = typed_parameters["policy_state_quat_hemisphere"]
    if configured not in POLICY_STATE_QUAT_HEMISPHERE_CHOICES:
        raise ConfigError(
            f"{_POLICY_STATE_QUAT_HEMISPHERE_PATH} must be one of "
            f"{', '.join(POLICY_STATE_QUAT_HEMISPHERE_CHOICES)}",
            path=_POLICY_STATE_QUAT_HEMISPHERE_PATH,
            allowed=list(POLICY_STATE_QUAT_HEMISPHERE_CHOICES),
            got=configured,
        )
    if any(
        argument == POLICY_STATE_QUAT_HEMISPHERE_FLAG
        or argument.startswith(POLICY_STATE_QUAT_HEMISPHERE_FLAG + "=")
        for argument in cli_args
    ):
        raise ConfigError(
            f"{POLICY_STATE_QUAT_HEMISPHERE_FLAG} is rendered by the launcher from "
            f"{_POLICY_STATE_QUAT_HEMISPHERE_PATH}",
            path=_POLICY_STATE_QUAT_HEMISPHERE_PATH,
        )
    if configured == "auto":
        try:
            _train_config, norm_stats_path, asset_id = pbi.resolve_openpi_norm_stats(
                checkpoint,
                config=config_name,
                config_src=config_source,
                action_horizon=action_horizon,
                repo_root=repo_root,
            )
            resolution = pio.policy_state_quat_hemisphere_from_norm_stats(
                norm_stats_path
            )
        except pio.PolicyStateQuatHemisphereAbsent as exc:
            # A checkpoint with no norm stats at all (a placeholder, or inputs
            # not staged on this host) stays unresolved and renders no flag.
            # validate_policy_state_quat_hemisphere refuses it at check-runtime
            # and launch unless the checkpoint artifact is a placeholder fixture.
            return list(cli_args), {
                "configured": configured,
                "source": "unresolved",
                "hemisphere": None,
                "reason": str(exc),
            }
        except pio.PolicyStateQuatHemisphereError as exc:
            raise ConfigError(
                f"{_POLICY_STATE_QUAT_HEMISPHERE_PATH}: auto cannot be resolved "
                f"from policy.checkpoint {checkpoint}: {exc}",
                path=_POLICY_STATE_QUAT_HEMISPHERE_PATH,
            ) from exc
        except LockedImportError as exc:
            raise ConfigError(
                f"{_POLICY_STATE_QUAT_HEMISPHERE_PATH}: auto cannot resolve the "
                f"OpenPI config {config_name!r}: locked OpenPI check failed: {exc}",
                path=_POLICY_STATE_QUAT_HEMISPHERE_PATH,
            ) from exc
        except (ImportError, ValueError) as exc:
            raise ConfigError(
                f"{_POLICY_STATE_QUAT_HEMISPHERE_PATH}: auto cannot resolve the "
                f"OpenPI config {config_name!r} ({config_source}): {exc}",
                path=_POLICY_STATE_QUAT_HEMISPHERE_PATH,
            ) from exc
        record = {
            "configured": configured,
            "source": "checkpoint_norm_stats",
            "norm_stats_asset_id": asset_id,
            **resolution,
        }
    else:
        record = {
            "configured": configured,
            "source": "profile",
            "hemisphere": configured,
        }
    rendered = list(cli_args)
    if record["hemisphere"] != "positive":
        rendered.extend((POLICY_STATE_QUAT_HEMISPHERE_FLAG, record["hemisphere"]))
    return rendered, record


def _validate_v2_parameters(
    value: Any,
    *,
    context: Mapping[str, Any],
    config_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Validate schema-v2 typed simulator parameters and render exact CLI args."""

    parameters = _mapping(value, "simulation.parameters")
    unknown = sorted(set(parameters) - _V2_PARAMETER_KEYS)
    missing = sorted(
        _V2_PARAMETER_KEYS
        - _V2_OPTIONAL_PARAMETER_DEFAULTS.keys()
        - set(parameters)
    )
    if unknown:
        raise _unknown_key_error(
            "simulation.parameters contains unknown keys: " + ", ".join(unknown),
            path="simulation.parameters",
            allowed=sorted(_V2_PARAMETER_KEYS),
            got=unknown,
        )
    if missing:
        raise ConfigError(
            "simulation.parameters is missing required keys: " + ", ".join(missing),
            path="simulation.parameters",
            allowed=sorted(_V2_PARAMETER_KEYS),
            got=missing,
        )

    resolved: dict[str, Any] = {
        "physics": _choice(
            parameters["physics"],
            "simulation.parameters.physics",
            ("kinematic", "dynamic"),
        ),
        "grasp_mode": _choice(
            parameters["grasp_mode"], "simulation.parameters.grasp_mode", ("physics", "rule")
        ),
        "policy_anchor": _choice(
            parameters["policy_anchor"], "simulation.parameters.policy_anchor", ("deploy", "ready")
        ),
        "policy_image_source": _choice(
            parameters["policy_image_source"],
            "simulation.parameters.policy_image_source",
            ("sim", "lerobot"),
        ),
        "policy_image_resize": _choice(
            parameters["policy_image_resize"],
            "simulation.parameters.policy_image_resize",
            ("stretch", "pad"),
        ),
        "policy_yubi_render_profile": _choice(
            parameters.get(
                "policy_yubi_render_profile",
                _V2_OPTIONAL_PARAMETER_DEFAULTS["policy_yubi_render_profile"],
            ),
            "simulation.parameters.policy_yubi_render_profile",
            WRIST_RENDER_PROFILE_NAMES,
        ),
        "policy_wrist_contrast": _number(
            parameters["policy_wrist_contrast"],
            "simulation.parameters.policy_wrist_contrast",
            minimum=0.0,
        ),
        "policy_wrist_backend": _choice(
            parameters.get(
                "policy_wrist_backend",
                _V2_OPTIONAL_PARAMETER_DEFAULTS["policy_wrist_backend"],
            ),
            "simulation.parameters.policy_wrist_backend",
            WRIST_BACKENDS,
        ),
        "policy_obs_inset": _boolean(
            parameters["policy_obs_inset"], "simulation.parameters.policy_obs_inset"
        ),
        "policy_delta_anchor": _choice(
            parameters["policy_delta_anchor"],
            "simulation.parameters.policy_delta_anchor",
            ("observation", "live"),
        ),
        # The configured value; ``auto`` is resolved against the checkpoint by
        # resolve_policy_state_quat_hemisphere_parameter at plan resolution.
        "policy_state_quat_hemisphere": _choice(
            parameters.get(
                "policy_state_quat_hemisphere",
                _V2_OPTIONAL_PARAMETER_DEFAULTS["policy_state_quat_hemisphere"],
            ),
            "simulation.parameters.policy_state_quat_hemisphere",
            POLICY_STATE_QUAT_HEMISPHERE_CHOICES,
        ),
        "policy_action_clamp": _boolean(
            parameters.get(
                "policy_action_clamp",
                _V2_OPTIONAL_PARAMETER_DEFAULTS["policy_action_clamp"],
            ),
            "simulation.parameters.policy_action_clamp",
        ),
        "policy_pose_source": _choice(
            parameters["policy_pose_source"],
            "simulation.parameters.policy_pose_source",
            ("sim", "lerobot"),
        ),
        "policy_image_index": _choice(
            parameters["policy_image_index"],
            "simulation.parameters.policy_image_index",
            ("infer", "frame"),
        ),
        "policy_image_loop": _boolean(
            parameters["policy_image_loop"], "simulation.parameters.policy_image_loop"
        ),
        "no_obs_gs": _boolean(parameters["no_obs_gs"], "simulation.parameters.no_obs_gs"),
        "obs_gs_only": _boolean(
            parameters.get(
                "obs_gs_only", _V2_OPTIONAL_PARAMETER_DEFAULTS["obs_gs_only"]
            ),
            "simulation.parameters.obs_gs_only",
        ),
        "desk_world_aligned": _boolean(
            parameters["desk_world_aligned"], "simulation.parameters.desk_world_aligned"
        ),
        "dynamic_control_hz": _integer(
            parameters["dynamic_control_hz"],
            "simulation.parameters.dynamic_control_hz",
            minimum=1,
        ),
        "frame_hz": _number(
            parameters["frame_hz"], "simulation.parameters.frame_hz", minimum=0.0
        ),
        "dynamic_ready_sec": _number(
            parameters["dynamic_ready_sec"],
            "simulation.parameters.dynamic_ready_sec",
            minimum=0.0,
        ),
        "joint_ik_max_velocity": _number(
            parameters["joint_ik_max_velocity"],
            "simulation.parameters.joint_ik_max_velocity",
            minimum=0.0,
        ),
        "joint_ik_max_acceleration": _number(
            parameters["joint_ik_max_acceleration"],
            "simulation.parameters.joint_ik_max_acceleration",
            minimum=0.0,
        ),
        "joint_ik_max_jerk": _number(
            parameters["joint_ik_max_jerk"],
            "simulation.parameters.joint_ik_max_jerk",
            minimum=0.0,
        ),
        "joint_ik_nullspace_gain": _number(
            parameters["joint_ik_nullspace_gain"],
            "simulation.parameters.joint_ik_nullspace_gain",
            minimum=0.0,
        ),
        "joint_ik_ready_position_tol_mm": _number(
            parameters["joint_ik_ready_position_tol_mm"],
            "simulation.parameters.joint_ik_ready_position_tol_mm",
            minimum=0.0,
        ),
        "joint_ik_ready_orientation_tol_deg": _number(
            parameters["joint_ik_ready_orientation_tol_deg"],
            "simulation.parameters.joint_ik_ready_orientation_tol_deg",
            minimum=0.0,
        ),
        "dynamic_subframes": _integer(
            parameters["dynamic_subframes"],
            "simulation.parameters.dynamic_subframes",
            minimum=1,
        ),
        "substeps": _integer(
            parameters["substeps"], "simulation.parameters.substeps", minimum=1
        ),
        "mujoco_integrator": _choice(
            parameters["mujoco_integrator"],
            "simulation.parameters.mujoco_integrator",
            ("keep", "euler", "rk4", "implicit", "implicitfast"),
        ),
        "torque_scale": _number(
            parameters["torque_scale"], "simulation.parameters.torque_scale", minimum=0.0
        ),
        "grip_friction": (
            "keep"
            if parameters["grip_friction"] == "keep"
            else _number_list(
                parameters["grip_friction"],
                "simulation.parameters.grip_friction",
                length=3,
                minimum=0.0,
            )
        ),
        "rubber_friction": (
            "keep"
            if parameters["rubber_friction"] == "keep"
            else _number_list(
                parameters["rubber_friction"],
                "simulation.parameters.rubber_friction",
                length=3,
                minimum=0.0,
            )
        ),
        "solref": _number_list(
            parameters["solref"], "simulation.parameters.solref", length=2
        ),
        "grip_priority": _integer(
            parameters["grip_priority"], "simulation.parameters.grip_priority", minimum=0
        ),
        "rubber_solref": _number_list(
            parameters["rubber_solref"], "simulation.parameters.rubber_solref", length=2
        ),
        "rubber_solimp": _number_list(
            parameters["rubber_solimp"], "simulation.parameters.rubber_solimp", length=3
        ),
        "rubber_priority": _integer(
            parameters["rubber_priority"], "simulation.parameters.rubber_priority", minimum=0
        ),
        "cup_mass": _number(
            parameters["cup_mass"], "simulation.parameters.cup_mass", minimum=0.0
        ),
        "rule_grasp_hand": _choice(
            parameters["rule_grasp_hand"],
            "simulation.parameters.rule_grasp_hand",
            ("auto", "R", "L"),
        ),
        "rule_grasp_effort_eps": _number(
            parameters["rule_grasp_effort_eps"],
            "simulation.parameters.rule_grasp_effort_eps",
            minimum=0.0,
        ),
        "rule_grasp_policy_eps_rad": _number(
            parameters["rule_grasp_policy_eps_rad"],
            "simulation.parameters.rule_grasp_policy_eps_rad",
            minimum=0.0,
        ),
        "rule_grasp_release_eps_rad": _number(
            parameters["rule_grasp_release_eps_rad"],
            "simulation.parameters.rule_grasp_release_eps_rad",
            minimum=0.0,
        ),
        "grasp_contact_tol_mm": _number(
            parameters["grasp_contact_tol_mm"],
            "simulation.parameters.grasp_contact_tol_mm",
            minimum=0.0,
        ),
        "debug_grasp_contact": _boolean(
            parameters["debug_grasp_contact"],
            "simulation.parameters.debug_grasp_contact",
        ),
        "gs_render_timeout_sec": _number(
            parameters["gs_render_timeout_sec"],
            "simulation.parameters.gs_render_timeout_sec",
            minimum=0.001,
        ),
        "gs_render_attempts": _integer(
            parameters["gs_render_attempts"],
            "simulation.parameters.gs_render_attempts",
            minimum=1,
        ),
        "gs_render_retry_delay_sec": _number(
            parameters["gs_render_retry_delay_sec"],
            "simulation.parameters.gs_render_retry_delay_sec",
            minimum=0.0,
        ),
        "gs_composite_mode": _choice(
            parameters.get(
                "gs_composite_mode",
                _V2_OPTIONAL_PARAMETER_DEFAULTS["gs_composite_mode"],
            ),
            "simulation.parameters.gs_composite_mode",
            ("depth", "foreground"),
        ),
        "gs_color_gain_rgb": _rgb_color_transform(
            parameters.get(
                "gs_color_gain_rgb",
                _V2_OPTIONAL_PARAMETER_DEFAULTS["gs_color_gain_rgb"],
            ),
            "simulation.parameters.gs_color_gain_rgb",
        ),
        "gs_color_gamma_rgb": _rgb_color_transform(
            parameters.get(
                "gs_color_gamma_rgb",
                _V2_OPTIONAL_PARAMETER_DEFAULTS["gs_color_gamma_rgb"],
            ),
            "simulation.parameters.gs_color_gamma_rgb",
        ),
        "gs_calibration_world_offset": _number_list(
            parameters["gs_calibration_world_offset"],
            "simulation.parameters.gs_calibration_world_offset",
            length=3,
        ),
        "success_mode": _choice(
            parameters["success_mode"], "simulation.parameters.success_mode", ("bbox",)
        ),
        "table_z": _number(parameters["table_z"], "simulation.parameters.table_z"),
        "world_offset": _number_list(
            parameters["world_offset"], "simulation.parameters.world_offset", length=3
        ),
        "ik_jump_reject": _choice(
            parameters["ik_jump_reject"],
            "simulation.parameters.ik_jump_reject",
            ("off", "hold", "retry-hold"),
        ),
        "ik_jump_retries": _integer(
            parameters["ik_jump_retries"],
            "simulation.parameters.ik_jump_retries",
            minimum=0,
        ),
    }
    if parameters.get("mujoco_timestep") is not None:
        value = _number(parameters["mujoco_timestep"], "simulation.parameters.mujoco_timestep", minimum=0.0)
        if value <= 0 or value > .002:
            raise ConfigError("mujoco_timestep must be positive and at most 0.002 seconds")
        resolved["mujoco_timestep"] = value
    if parameters.get("fr3_impedance") is not None:
        from dataclasses import asdict
        from robosuite_env.robots.fr3.impedance import ImpedanceConfig
        try:
            resolved["fr3_impedance"] = json.loads(json.dumps(
                asdict(ImpedanceConfig.from_mapping(parameters["fr3_impedance"]))
            ))
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc), path="simulation.parameters.fr3_impedance") from exc
        if resolved["physics"] != "dynamic":
            raise ConfigError("fr3_impedance requires physics: dynamic")
    if "fr3_joint_kp" in parameters:
        joint_kp = _number(
            parameters["fr3_joint_kp"],
            "simulation.parameters.fr3_joint_kp",
            minimum=0.0,
        )
        if joint_kp <= 0.0:
            raise ConfigError(
                "simulation.parameters.fr3_joint_kp must be > 0",
                path="simulation.parameters.fr3_joint_kp",
            )
        resolved["fr3_joint_kp"] = joint_kp
    if "fr3_joint_kd" in parameters:
        resolved["fr3_joint_kd"] = _number(
            parameters["fr3_joint_kd"],
            "simulation.parameters.fr3_joint_kd",
            minimum=0.0,
        )
    # Only the dynamic JOINT_POSITION controller reads the gains
    # (robosuite_env/rollout/build/model.py). A kinematic profile that sets
    # them would otherwise pass every check, carry them in its official argv,
    # and have them dropped in silence.
    if resolved["physics"] != "dynamic":
        for key in ("fr3_joint_kp", "fr3_joint_kd"):
            if key in resolved:
                raise ConfigError(
                    f"simulation.parameters.{key} requires physics: dynamic; "
                    f"physics: {resolved['physics']} has no joint controller "
                    "to apply it to",
                    path=f"simulation.parameters.{key}",
                )
    for key in (
        "success_contact_tol_mm",
        "success_contact_penetration_tol_mm",
        "success_place_contact_penetration_tol_mm",
        "success_plane_gap_tol_mm",
        "success_plane_penetration_tol_mm",
        "success_bottom_contact_tol_mm",
        "success_footprint_slack_mm",
        "success_upright_tol_deg",
    ):
        resolved[key] = _number(
            parameters[key], f"simulation.parameters.{key}", minimum=0.0
        )

    for key, maximum in _TASK_INDEX_MAXIMA.items():
        raw = parameters.get(key, _V2_OPTIONAL_PARAMETER_DEFAULTS[key])
        if raw is None:
            resolved[key] = None
            continue
        path = f"simulation.parameters.{key}"
        if key == "box_pack_task_index":
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ConfigError(f"{path} must be an integer", path=path)
            value = raw
            supported = list(range(maximum + 1))
            if value not in supported:
                raise ConfigError(
                    f"{path} supports only runnable into-box task indices "
                    f"{', '.join(str(index) for index in supported)}",
                    path=path,
                    allowed=supported,
                    got=value,
                )
            resolved[key] = value
        else:
            resolved[key] = _integer(
                raw, path, minimum=0, maximum=maximum
            )
    # Box packing may name one layout path per runner repeat; the batch
    # runner selects list[(NN - 1) % len(list)] for run_NN
    # (robosuite_env/tasks/layout_selection.py).
    _box_layout_raw = parameters.get("box_pack_layout_json", None)
    if isinstance(_box_layout_raw, list):
        if not _box_layout_raw:
            raise ConfigError(
                "simulation.parameters.box_pack_layout_json must be a non-empty list",
                path="simulation.parameters.box_pack_layout_json",
            )
        resolved["box_pack_layout_json"] = [
            str(
                _resolved_path(
                    _string(
                        item,
                        f"simulation.parameters.box_pack_layout_json[{index}]",
                    ),
                    f"simulation.parameters.box_pack_layout_json[{index}]",
                    context,
                    config_dir,
                )
            )
            for index, item in enumerate(_box_layout_raw)
        ]
    else:
        _box_layout_single = _optional_string(
            _box_layout_raw, "simulation.parameters.box_pack_layout_json"
        )
        resolved["box_pack_layout_json"] = (
            str(
                _resolved_path(
                    _box_layout_single,
                    "simulation.parameters.box_pack_layout_json",
                    context,
                    config_dir,
                )
            )
            if _box_layout_single
            else None
        )
    for key in (
        "success_stable_sec",
        "success_linear_speed_mm_s",
        "success_angular_speed_rad_s",
        "dynamic_object_settle_max_sec",
        "dynamic_object_settle_min_sec",
        "dynamic_object_settle_stable_sec",
        "dynamic_object_settle_linear_speed_mm_s",
        "dynamic_object_settle_angular_speed_rad_s",
        "dynamic_object_settle_linear_acceleration_m_s2",
        "dynamic_object_settle_angular_acceleration_rad_s2",
        "dynamic_object_settle_position_drift_mm",
        "dynamic_object_settle_orientation_drift_deg",
    ):
        raw = parameters.get(key, _V2_OPTIONAL_PARAMETER_DEFAULTS[key])
        resolved[key] = (
            None
            if raw is None
            else _number(raw, f"simulation.parameters.{key}", minimum=0.0)
        )

    _cup_test = parameters.get("cup_above_plate_test", None)
    if _cup_test is not None:
        _cup_test = _string(_cup_test, "simulation.parameters.cup_above_plate_test")
        if _cup_test not in ("middle", "lowest"):
            raise ConfigError(
                "simulation.parameters.cup_above_plate_test must be 'middle' or "
                f"'lowest', got {_cup_test!r}",
                path="simulation.parameters.cup_above_plate_test",
            )
    resolved["cup_above_plate_test"] = _cup_test

    _guard = parameters.get("start_state_guard", None)
    if _guard is not None:
        _guard = _string(_guard, "simulation.parameters.start_state_guard")
        if _guard not in ("off", "warn", "fail"):
            raise ConfigError(
                "simulation.parameters.start_state_guard must be 'off', 'warn' "
                f"or 'fail', got {_guard!r}",
                path="simulation.parameters.start_state_guard",
            )
    resolved["start_state_guard"] = _guard

    initial_name = _optional_string(
        parameters.get(
            "initial_arm_joints",
            _V2_OPTIONAL_PARAMETER_DEFAULTS["initial_arm_joints"],
        ),
        "simulation.parameters.initial_arm_joints",
    )
    initial_raw = _optional_string(
        parameters.get(
            "initial_arm_joints_json",
            _V2_OPTIONAL_PARAMETER_DEFAULTS["initial_arm_joints_json"],
        ),
        "simulation.parameters.initial_arm_joints_json",
    )
    initial_path = (
        str(
            _resolved_path(
                initial_raw,
                "simulation.parameters.initial_arm_joints_json",
                context,
                config_dir,
            )
        )
        if initial_raw is not None
        else None
    )
    resolved["initial_arm_joints"] = initial_name
    # ``initial_arm_joints`` is launcher-only: both spellings collapse to the
    # one absolute path the worker already accepts, so the rendered argument
    # vector and the settled init-state contract are unchanged.
    resolved["initial_arm_joints_json"] = resolve_initial_arm_joint_pose(
        initial_name, initial_path
    )
    check_pose_signature(
        resolved["initial_arm_joints_json"], str(context.get("robot", "fr3"))
    )
    base_raw = _optional_string(
        parameters.get(
            "base_placement",
            _V2_OPTIONAL_PARAMETER_DEFAULTS["base_placement"],
        ),
        "simulation.parameters.base_placement",
    )
    resolved["base_placement"] = (
        str(
            _resolved_path(
                base_raw,
                "simulation.parameters.base_placement",
                context,
                config_dir,
            )
        )
        if base_raw is not None
        else None
    )
    if resolved["no_obs_gs"] and resolved["obs_gs_only"]:
        raise ConfigError(
            "simulation.parameters.no_obs_gs and obs_gs_only are mutually exclusive"
        )
    if resolved["table_z"] != 0.0:
        raise ConfigError(
            "simulation.parameters.table_z must be 0.0 for schema-v2 "
            "mujoco_world table collision"
        )
    if resolved["gs_calibration_world_offset"] != [0.0, 0.0, 0.0]:
        raise ConfigError(
            "schema_version 2 requires "
            "simulation.parameters.gs_calibration_world_offset: [0, 0, 0]; "
            "the validated scene world-to-GS transform is authoritative"
        )
    if resolved["dynamic_subframes"] != 1:
        raise ConfigError(
            "schema_version 2 requires simulation.parameters.dynamic_subframes: 1; "
            "the required grid video must contain exactly runner.frames frames"
        )
    desk_raw = _optional_string(parameters["desk"], "simulation.parameters.desk")
    if desk_raw is None or desk_raw in {"robosuite", "scene"}:
        resolved["desk"] = desk_raw
    else:
        resolved["desk"] = str(
            _resolved_path(desk_raw, "simulation.parameters.desk", context, config_dir)
        )
    if resolved["desk"] != "scene":
        raise ConfigError(
            "schema_version 2 requires simulation.parameters.desk: scene; "
            "the scene contract owns the table collision"
        )
    if not resolved["desk_world_aligned"]:
        raise ConfigError(
            "schema_version 2 requires simulation.parameters.desk_world_aligned: true"
        )
    cli: list[str] = []
    cli.extend(("--policy-yubi-render-profile", resolved["policy_yubi_render_profile"]))
    cli.extend(("--policy-wrist-backend", resolved["policy_wrist_backend"]))
    # Not a profile parameter, but always in the worker argv: a rerender of a run that stopped before
    # recording gs_kb4_near_cull in sim_summary.json or a policy observation keeps the mode it rendered with.
    cli.extend(("--gs-kb4-near-cull", DEFAULT_KB4_NEAR_CULL))

    def option(key: str, name: str | None = None) -> None:
        value = resolved[key]
        if value is not None:
            # A box-layout list is selected per run by the batch runner. It is
            # carried in BOX_PACK_LAYOUT_JSON_LIST rather than stringifying a
            # Python list into one unusable worker flag.
            if key == "box_pack_layout_json" and isinstance(value, list):
                return
            cli.extend((name or "--" + key.replace("_", "-"), str(value)))

    for key in (
        "physics",
        "policy_anchor",
        "initial_arm_joints_json",
        "base_placement",
        "policy_image_source",
        "policy_image_resize",
        "policy_wrist_contrast",
        "policy_delta_anchor",
        "policy_pose_source",
        "policy_image_index",
        "desk",
        "dynamic_control_hz",
        "frame_hz",
        "dynamic_ready_sec",
        "joint_ik_max_velocity",
        "joint_ik_max_acceleration",
        "joint_ik_max_jerk",
        "joint_ik_nullspace_gain",
        "joint_ik_ready_position_tol_mm",
        "joint_ik_ready_orientation_tol_deg",
        "dynamic_subframes",
        "substeps",
        "mujoco_integrator",
        "torque_scale",
        "grip_priority",
        "rubber_priority",
        "cup_mass",
        "rule_grasp_hand",
        "rule_grasp_effort_eps",
        "rule_grasp_policy_eps_rad",
        "rule_grasp_release_eps_rad",
        "grasp_contact_tol_mm",
        "gs_render_timeout_sec",
        "gs_render_attempts",
        "gs_render_retry_delay_sec",
        "gs_composite_mode",
        "success_contact_tol_mm",
        "success_contact_penetration_tol_mm",
        "success_place_contact_penetration_tol_mm",
        "success_plane_gap_tol_mm",
        "success_plane_penetration_tol_mm",
        "success_bottom_contact_tol_mm",
        "success_footprint_slack_mm",
        "success_upright_tol_deg",
        "success_mode",
        *_V2_TASK_SCOPED_PARAMETER_KEYS,
        "table_z",
        "ik_jump_reject",
        "ik_jump_retries",
    ):
        option(key)
    # These options are absent from both the resolved config and argv unless
    # explicitly configured, preserving every pre-existing profile exactly.
    if "fr3_impedance" in resolved:
        cli.extend(("--fr3-impedance", json.dumps(resolved["fr3_impedance"], sort_keys=True)))
    for key in ("fr3_joint_kp", "fr3_joint_kd", "mujoco_timestep"):
        if key in resolved:
            option(key)
    cli.extend(
        (
            "--gs-color-gain-rgb",
            *(str(float(item)) for item in resolved["gs_color_gain_rgb"]),
        )
    )
    cli.extend(
        (
            "--gs-color-gamma-rgb",
            *(str(float(item)) for item in resolved["gs_color_gamma_rgb"]),
        )
    )
    for key in (
        "grip_friction",
        "rubber_friction",
        "solref",
        "rubber_solref",
        "rubber_solimp",
    ):
        value = resolved[key]
        rendered = (
            value
            if isinstance(value, str)
            else " ".join(str(item) for item in value)
        )
        cli.extend(("--" + key.replace("_", "-"), rendered))
    cli.extend(("--world-offset", *(str(item) for item in resolved["world_offset"])))
    cli.extend(
        (
            "--gs-calibration-world-offset",
            *(str(item) for item in resolved["gs_calibration_world_offset"]),
        )
    )
    for key, flag in (
        ("policy_obs_inset", "--policy-obs-inset"),
        ("policy_image_loop", "--policy-image-loop"),
        ("no_obs_gs", "--no-obs-gs"),
        ("obs_gs_only", "--obs-gs-only"),
        ("desk_world_aligned", "--desk-world-aligned"),
        ("debug_grasp_contact", "--debug-grasp-contact"),
    ):
        if resolved[key]:
            cli.append(flag)
    cli.append(
        "--policy-action-clamp"
        if resolved["policy_action_clamp"]
        else "--no-policy-action-clamp"
    )
    return resolved, cli
