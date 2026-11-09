# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Schema key sets and configuration environment contracts."""

from __future__ import annotations

from real2sim.wrist_profiles import DEFAULT_POLICY_WRIST_RENDER_PROFILE


RETIRED_GPU_ENV_NAME = "REAL2SIM_GPU_IDS"
# The portable WebSocket policy bridge was removed. A caller that still sets its
# endpoint asked for a remote policy; starting the local checkpoint instead would
# report another policy's results, so a nonempty value fails closed.
RETIRED_POLICY_TRANSPORT_ENV_NAME = "PORTABLE_POLICY_SERVER"

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "robot",
        "name",
        "description",
        "runner",
        "policy",
        "task",
        "scenes",
        "simulation",
        "render",
        "runtime",
        "output",
        "reproducibility",
    }
)

_SECTION_KEYS = {
    "runner": frozenset(
        {
            "mode",
            "repeats",
            "parallel_repeats",
            "policy_batch_size",
            "frames",
            "max_wall_sec",
            "expected_rollouts",
            "shards",
        }
    ),
    "policy": frozenset(
        {
            "python",
            "checkpoint",
            "config_name",
            "config_source",
            "action_horizon",
            "exec_steps",
            "noise_seed",
            "deterministic_xla",
        }
    ),
    "task": frozenset(
        {
            "objects",
            "instruction",
            "goal_condition",
            "score_condition",
            "policy_image_episode",
        }
    ),
    "simulation": frozenset(
        {
            "deterministic",
            "seed",
            "init_state",
            "rf_overlap",
            "rf_overlap_sampler",
            "rf_overlap_prev_weight",
            "rf_overlap_step_shift",
            "grip_torque_mult",
            "grasp_mode",
            "gripper_calibration",
            "extra_args",
            "parameters",
            "fr3_controller",
            "fr3_nullspace",
            "yubi_controller",
        }
    ),
    "render": frozenset(
        {
            "desk_texture",
            "view_azimuth",
            "view_elevation",
            "view_distance",
            "view_lookat",
            "wrist",
            "wrist_gs",
            "origin",
            "analysis",
            "clean_video",
            "enabled",
            "third_person",
        }
    ),
    "runtime": frozenset(
        {
            "server_host",
            "server_port",
            "gs_port_base",
            "gs_cuda_devices",
            "gs_servers_per_gpu",
            "policy_cuda_device",
            "sim_cuda_devices",
            "gs_start_timeout_sec",
            "gs_background_color",
            "sim_num_threads",
            "render_lock_root",
            "gaussian_wrapping_repo",
        }
    ),
    "output": frozenset({"root", "run_name"}),
}

_RETIRED_SECTION_KEYS = {
    "render": {
        "blender_export": "render.blender_export is retired; Blender export was removed",
        "build_blend": "render.build_blend is retired; Blender export was removed",
    },
}

_REQUIRED_SECTION_KEYS = {
    "runner": _SECTION_KEYS["runner"] - frozenset({"shards"}),
    "policy": _SECTION_KEYS["policy"],
    "task": frozenset({"objects", "instruction", "goal_condition", "score_condition"}),
    "simulation": _SECTION_KEYS["simulation"]
    - frozenset({
        "fr3_controller",
        "fr3_nullspace",
        "yubi_controller",
        "grasp_mode",
        "gripper_calibration",
        "parameters",
    }),
    "render": _SECTION_KEYS["render"] - frozenset({"enabled", "third_person"}),
    "runtime": _SECTION_KEYS["runtime"]
    - frozenset(
        {
            "gs_cuda_devices",
            # omitting it keeps one GS server per GPU, as every earlier run had
            "gs_servers_per_gpu",
            "policy_cuda_device",
            "sim_cuda_devices",
            "render_lock_root",
            # omitting it keeps the black far field every earlier run used
            "gs_background_color",
        }
    ),
    "output": _SECTION_KEYS["output"],
}

_SCENE_KEYS = frozenset({"tag", "path"})

_REPRODUCIBILITY_KEYS = frozenset(
    {
        "mode",
        "artifact_manifest",
        "artifacts",
        "lockfiles",
        "source_dependencies",
        "support_matrix",
        "accelerator_profile",
        "environment",
    }
)

_REPRO_ENVIRONMENT_KEYS = frozenset({"allowlist", "required", "values"})

_CONFIG_SOURCES = frozenset({"deploy", "training"})

_ROBOT_CLI_NAMES = {"fr3": "FR3", "fr3_duo": "FR3_DUO"}

_V2_PARAMETER_KEYS = frozenset(
    {
        "physics",
        "grasp_mode",
        "policy_anchor",
        # Two spellings of one selection: a registry name
        # (real2sim.config.initial_arm_joints) or the historical explicit
        # path. Setting both is rejected.
        "initial_arm_joints",
        "initial_arm_joints_json",
        "base_placement",
        "policy_image_source",
        "policy_image_resize",
        "policy_yubi_render_profile",
        "policy_wrist_contrast",
        "policy_wrist_backend",
        "policy_obs_inset",
        "policy_delta_anchor",
        # Sign convention of the relative-hand state quaternion: auto (from the
        # checkpoint norm stats), positive or negative. Only a resolved
        # negative reaches the worker argv.
        "policy_state_quat_hemisphere",
        "policy_action_clamp",
        "policy_pose_source",
        "policy_image_index",
        "policy_image_loop",
        "no_obs_gs",
        "obs_gs_only",
        "desk",
        "desk_world_aligned",
        "dynamic_control_hz",
        "frame_hz",
        "dynamic_ready_sec",
        "fr3_impedance",
        "fr3_joint_kp",
        "fr3_joint_kd",
        "joint_ik_max_velocity",
        "joint_ik_max_acceleration",
        "joint_ik_max_jerk",
        "joint_ik_nullspace_gain",
        "joint_ik_ready_position_tol_mm",
        "joint_ik_ready_orientation_tol_deg",
        "dynamic_subframes",
        "substeps",
        "mujoco_integrator",
        "mujoco_timestep",
        "torque_scale",
        "grip_friction",
        "rubber_friction",
        "solref",
        "grip_priority",
        "rubber_solref",
        "rubber_solimp",
        "rubber_priority",
        "cup_mass",
        "rule_grasp_hand",
        "rule_grasp_effort_eps",
        "rule_grasp_policy_eps_rad",
        "rule_grasp_release_eps_rad",
        "grasp_contact_tol_mm",
        "debug_grasp_contact",
        "gs_render_timeout_sec",
        "gs_render_attempts",
        "gs_render_retry_delay_sec",
        "gs_calibration_world_offset",
        "gs_composite_mode",
        "gs_color_gain_rgb",
        "gs_color_gamma_rgb",
        "success_contact_tol_mm",
        "success_contact_penetration_tol_mm",
        "success_place_contact_penetration_tol_mm",
        "success_plane_gap_tol_mm",
        "success_plane_penetration_tol_mm",
        "success_bottom_contact_tol_mm",
        "success_footprint_slack_mm",
        "success_upright_tol_deg",
        "success_mode",
        # Optional task-scoped stability contract. These are rendered into the
        # worker argv only when a config sets them, so the cup_plate profiles
        # keep their exact historical argument vector.
        "success_stable_sec",
        "success_linear_speed_mm_s",
        "success_angular_speed_rad_s",
        # cup_on_plate: which point of the cup must be over the plate --
        # "middle" (the cup's mid-height, as Duo scored) or "lowest"
        # (the cup's lowest collision point, Flat's stricter test). Absent
        # keeps the criterion's own default, "middle".
        "cup_above_plate_test",
        # What the first-frame interpenetration check does: "warn" records
        # the overlaps, "fail" refuses the run. Absent keeps the worker's
        # default (START_STATE_GUARD, else "warn").
        "start_state_guard",
        # Box packing, scoped to its own task.
        "box_pack_task_index",
        # Which layout places the box, tape, pen and LAN cable. Empty or
        # absent keeps the shared hand-authored layout, so the tape and pen
        # variants stand on the same arrangement unless a profile says
        # otherwise.
        "box_pack_layout_json",
        "dynamic_object_settle_max_sec",
        "dynamic_object_settle_min_sec",
        "dynamic_object_settle_stable_sec",
        "dynamic_object_settle_linear_speed_mm_s",
        "dynamic_object_settle_angular_speed_rad_s",
        "dynamic_object_settle_linear_acceleration_m_s2",
        "dynamic_object_settle_angular_acceleration_rad_s2",
        "dynamic_object_settle_position_drift_mm",
        "dynamic_object_settle_orientation_drift_deg",
        "table_z",
        "world_offset",
        "ik_jump_reject",
        "ik_jump_retries",
    }
)

# Task-scoped parameters that are omitted by every profile that does not need
# them. ``None`` means "do not render a worker argument for this key".
_V2_TASK_SCOPED_PARAMETER_KEYS = (
    "success_stable_sec",
    "success_linear_speed_mm_s",
    "success_angular_speed_rad_s",
    "cup_above_plate_test",
    "start_state_guard",
    "box_pack_task_index",
    "box_pack_layout_json",
    "dynamic_object_settle_max_sec",
    "dynamic_object_settle_min_sec",
    "dynamic_object_settle_stable_sec",
    "dynamic_object_settle_linear_speed_mm_s",
    "dynamic_object_settle_angular_speed_rad_s",
    "dynamic_object_settle_linear_acceleration_m_s2",
    "dynamic_object_settle_angular_acceleration_rad_s2",
    "dynamic_object_settle_position_drift_mm",
    "dynamic_object_settle_orientation_drift_deg",
)

_V2_OPTIONAL_PARAMETER_DEFAULTS = {
    "policy_yubi_render_profile": DEFAULT_POLICY_WRIST_RENDER_PROFILE,
    "policy_wrist_backend": "direct_kb4",
    "policy_action_clamp": True,
    "obs_gs_only": False,
    # Omitted gains must not enter the resolved config or worker argv: the
    # repo-local robosuite controller JSON remains the source of truth.
    "fr3_impedance": None,
    "mujoco_timestep": None,
    "fr3_joint_kp": None,
    "fr3_joint_kd": None,
    # Where this rig's mounted bases stand. Omitted selects the rig's own
    # registered default; a file belonging to another rig is refused.
    "base_placement": None,
    # A profile selects the initial arm joint pose with exactly one of these.
    # Both default to "unset" so a name-only profile does not have to carry a
    # redundant ``initial_arm_joints_json: null`` line.
    "initial_arm_joints": None,
    "initial_arm_joints_json": None,
    # Resolved from the checkpoint norm stats at plan resolution. Every
    # profile that resolves positive keeps its exact worker argument vector.
    "policy_state_quat_hemisphere": "auto",
    "gs_composite_mode": "depth",
    "gs_color_gain_rgb": [1.0, 1.0, 1.0],
    "gs_color_gamma_rgb": [1.0, 1.0, 1.0],
    **{key: None for key in _V2_TASK_SCOPED_PARAMETER_KEYS},
}

_BEHAVIOR_ENV_NAMES = frozenset(
    {
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "CUP_MASS",
        "DETERMINISTIC_SEED",
        "DYNAMIC_SUBFRAMES",
        "FR3_BASE",
        "GRASP_CONTACT_TOL_MM",
        "GRASP_MODE",
        "GRIP_CTRL",
        "GRIP_FRICTION",
        "GRIP_PRIORITY",
        "GRIP_TORQUE_MULT",
        "GS_RENDER_ATTEMPTS",
        "GS_RENDER_RETRY_DELAY_SEC",
        "GS_RENDER_TIMEOUT_SEC",
        "INIT_STATE_JSON",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "POLICY_DELTA_ANCHOR",
        "REPLAY_DELTA",
        "REPLAY_OBJECT_SOURCE",
        "RUBBER_FRICTION",
        "RUBBER_PRIORITY",
        "RUBBER_SOLIMP",
        "RUBBER_SOLREF",
        "RULE_GRASP_EFFORT_EPS",
        "RULE_GRASP_HAND",
        "RULE_GRASP_POLICY_EPS_RAD",
        "RULE_GRASP_RELEASE_EPS_RAD",
        "SOLREF",
        "SUBSTEPS",
        "SUCCESS_ANGULAR_SPEED_RAD_S",
        "SUCCESS_BOTTOM_CONTACT_TOL_MM",
        "SUCCESS_CONTACT_PENETRATION_TOL_MM",
        "SUCCESS_CONTACT_TOL_MM",
        "SUCCESS_FOOTPRINT_SLACK_MM",
        "SUCCESS_LINEAR_SPEED_MM_S",
        "SUCCESS_MODE",
        "SUCCESS_PLACE_CONTACT_PENETRATION_TOL_MM",
        "SUCCESS_PLANE_GAP_TOL_MM",
        "SUCCESS_PLANE_PENETRATION_TOL_MM",
        "SUCCESS_STABLE_SEC",
        "SUCCESS_UPRIGHT_TOL_DEG",
        "TORQUE_SCALE",
        "XLA_FLAGS",
        "YUBI_CTRL",
        "YUBI_GRIPPER_CALIBRATION",
    }
)

_SECRET_WORDS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "password",
        "passwd",
        "secret",
        "token",
    }
)
