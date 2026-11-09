# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Argument validation, runtime preflight, and rollout configuration banner."""

import atexit
import functools
import os
import random
import signal
import sys
import threading

import cv2 as cv
import numpy as np

from real2sim.config.task_objects import validate_task_objects
from robosuite_env.robots.fr3.controllers import (
    FR3_IMPEDANCE_HOOK_CONTROLLERS,
    fr3_uses_cmd_twist_torque,
    fr3_uses_impedance_hook,
    validate_fr3_controller,
)
from robosuite_env.policy import policy_io as pio
from robosuite_env.robots.legacy_pose_paths import redirect_legacy_pose_path
from robosuite_env.robots.fr3.runtime_ik import effective_nullspace_gain
from robosuite_env.robots.profiles import get_robot_profile
from robosuite_env.robots.yubi.calibration import (
    GripperCalibrationError,
    load_gripper_calibration,
)
from robosuite_env.rollout.cli import DEFAULT_GRIPPER_CALIBRATION, parse
from robosuite_env.rollout.preflight import reported_gs_w2gs, run_runtime_preflight
from robosuite_env.rollout.replay.recorded_wrist import (
    RecordedWristError,
    resolve_state_log_wrist_options,
)
from robosuite_env.rollout.resource_lifecycle import (
    GlobalResourceCleanup,
    cleanup_rollout_resources_at_exit,
    mujoco_render_lock,
    terminate_rollout,
)
from robosuite_env.rollout.setup.frames import build_frame_constants
from robosuite_env.rollout.timing import (
    nearest_control_ticks_for_source_action,
    resolve_dynamic_timing,
)

def _parse_int_set(text):
    out=set()
    for token in str(text or "").split(","):
        token=token.strip()
        if not token:
            continue
        try:
            out.add(int(token))
        except ValueError as exc:
            raise SystemExit(f"[sim] invalid --state-dump-frames token: {token}") from exc
    return out


def _validate_success_stability_arguments(arguments):
    names = []
    if arguments.objects in {"box_pack"}:
        names.append(("success_stable_sec", True))
    if arguments.objects in {"box_pack"}:
        names.extend(
            (
                ("success_linear_speed_mm_s", False),
                ("success_angular_speed_rad_s", False),
            )
        )
    for name, positive in names:
        value = float(getattr(arguments, name))
        invalid = not np.isfinite(value) or (
            value <= 0.0 if positive else value < 0.0
        )
        if invalid:
            relation = "positive" if positive else "non-negative"
            raise SystemExit(
                f"[sim] --{name.replace('_', '-')} must be finite and "
                f"{relation}, got {value:g}"
            )


def run(state):
    """Parse CLI arguments and enforce cross-flag compatibility rules."""
    A = parse()
    state.A = A
    # Before anything reads them: a state-log replay renders the run's own wrist camera.
    try:
        resolve_state_log_wrist_options(state.A)
    except RecordedWristError as exc:
        raise SystemExit(f"[sim] --state-log-replay: {exc}") from exc
    # A command line from before the initial poses moved names the old path.
    state.A.initial_arm_joints_json = redirect_legacy_pose_path(
        state.A.initial_arm_joints_json
    )
    if state.A.no_obs_gs and state.A.obs_gs_only:
        raise SystemExit(
            "[sim] --no-obs-gs and --obs-gs-only are mutually exclusive"
        )
    try:
        validate_task_objects(state.A.objects, require_file=True)
    except ValueError as exc:
        raise SystemExit(f"[sim] {exc}") from exc
    if bool(state.A.strict_assets) != bool(state.A.official_worker_contract):
        raise SystemExit(
            "[sim] --strict-assets and --official-worker-contract must be used together"
        )
    if (
        len(state.A.gs_color_gain_rgb) != 3
        or not np.all(np.isfinite(np.asarray(state.A.gs_color_gain_rgb, dtype=float)))
        or np.any(np.asarray(state.A.gs_color_gain_rgb, dtype=float) < 0.0)
    ):
        raise SystemExit(
            "[sim] --gs-color-gain-rgb requires three finite non-negative values"
        )
    if (
        len(state.A.gs_color_gamma_rgb) != 3
        or not np.all(np.isfinite(np.asarray(state.A.gs_color_gamma_rgb, dtype=float)))
        or np.any(np.asarray(state.A.gs_color_gamma_rgb, dtype=float) <= 0.0)
    ):
        raise SystemExit(
            "[sim] --gs-color-gamma-rgb requires three finite positive values"
        )
    if state.A.dynamic_grip_cbpc and state.A.grip_ctrl != "cbpc":
        raise SystemExit(
            "[sim] --dynamic-grip-cbpc requires --grip-ctrl cbpc"
        )
    for _name in (
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
        _value = float(getattr(state.A, _name))
        state._value = _value
        if not np.isfinite(_value) or _value < 0.0:
            raise SystemExit(
                f"[sim] --{_name.replace('_', '-')} must be finite and non-negative, "
                f"got {_value:g}"
            )
    if float(state.A.dynamic_object_settle_max_sec) > 0.0:
        if state.A.physics != "dynamic":
            raise SystemExit(
                "[sim] --dynamic-object-settle-max-sec requires --physics dynamic"
            )
        if float(state.A.dynamic_object_settle_min_sec) > float(
            state.A.dynamic_object_settle_max_sec
        ):
            raise SystemExit(
                "[sim] --dynamic-object-settle-min-sec cannot exceed "
                "--dynamic-object-settle-max-sec"
            )
        if float(state.A.dynamic_object_settle_stable_sec) <= 0.0:
            raise SystemExit(
                "[sim] --dynamic-object-settle-stable-sec must be positive when "
                "settling is enabled"
            )
        if float(state.A.dynamic_object_settle_stable_sec) > float(
            state.A.dynamic_object_settle_max_sec
        ):
            raise SystemExit(
                "[sim] --dynamic-object-settle-stable-sec cannot exceed "
                "--dynamic-object-settle-max-sec"
            )
    _validate_success_stability_arguments(state.A)
    for _name, _allow_zero in (
        ("fr3_joint_kp", False),
        ("fr3_joint_kd", True),
    ):
        _raw_value = getattr(state.A, _name)
        if _raw_value is None:
            continue
        if state.A.physics != "dynamic":
            raise SystemExit(
                f"[sim] --{_name.replace('_', '-')} applies to the dynamic "
                f"joint controller only, got --physics {state.A.physics}"
            )
        _value = float(_raw_value)
        _invalid = not np.isfinite(_value) or (
            _value < 0.0 if _allow_zero else _value <= 0.0
        )
        if _invalid:
            _relation = "non-negative" if _allow_zero else "positive"
            raise SystemExit(
                f"[sim] --{_name.replace('_', '-')} must be finite and "
                f"{_relation}, got {_value:g}"
            )
    _nullspace_gain = float(state.A.joint_ik_nullspace_gain)
    state._nullspace_gain = _nullspace_gain
    if not np.isfinite(_nullspace_gain) or _nullspace_gain < 0.0:
        raise SystemExit(
            "[sim] --joint-ik-nullspace-gain must be finite and non-negative"
        )
    if state.A.fr3_nullspace == "home" and _nullspace_gain == 0.0:
        raise SystemExit(
            "[sim] --fr3-nullspace home requires a positive "
            "--joint-ik-nullspace-gain; select --fr3-nullspace off to "
            "disable the posture objective explicitly"
        )
    for _name in (
        "joint_ik_max_velocity",
        "joint_ik_max_acceleration",
        "joint_ik_max_jerk",
        "joint_ik_ready_position_tol_mm",
        "joint_ik_ready_orientation_tol_deg",
        "initial_arm_joints_tol_deg",
    ):
        _value = float(getattr(state.A, _name))
        state._value = _value
        if not np.isfinite(_value) or _value <= 0.0:
            raise SystemExit(
                f"[sim] --{_name.replace('_', '-')} must be positive, got {_value:g}"
            )
    if state.A.fr3_controller == "joint_ik":
        _desk_clearance_mm = float(state.A.joint_ik_desk_collision_clearance_mm)
        state._desk_clearance_mm = _desk_clearance_mm
        if not np.isfinite(_desk_clearance_mm) or _desk_clearance_mm < 0.0:
            raise SystemExit(
                "[sim] --joint-ik-desk-collision-clearance-mm must be finite "
                f"and non-negative, got {_desk_clearance_mm:g}"
            )
        if (
            state.A.physics != "dynamic"
            and (
                bool(state.A.joint_ik_fail_hold)
                or _desk_clearance_mm != 0.0
            )
        ):
            raise SystemExit(
                "[sim] joint_ik fail-hold / desk collision options require "
                "--physics dynamic"
            )
    elif (
        bool(state.A.joint_ik_fail_hold)
        or float(state.A.joint_ik_desk_collision_clearance_mm) != 0.0
    ):
            raise SystemExit(
                "[sim] joint_ik fail-hold / desk collision options require "
                "--fr3-controller joint_ik"
            )
    if state.A.initial_arm_joints_json and state.A.physics != "dynamic":
        raise SystemExit("[sim] --initial-arm-joints-json requires --physics dynamic")
    try:
        validate_fr3_controller(state.A.fr3_controller)
    except ValueError as exc:
        raise SystemExit(f"[sim] {exc}") from exc
    if fr3_uses_cmd_twist_torque(state.A.fr3_controller):
        if state.A.mujoco_timestep is None:
            state.A.mujoco_timestep = .001
        if state.A.physics != "dynamic" or get_robot_profile(state.A.robot).embodiment != "fr3":
            raise SystemExit("[sim] Cartesian impedance requires a dynamic FR3 rig")
        if not state.A.state_log_replay:
            if not (state.A.policy or state.A.policy_batch_server) or state.A.policy_pose_repr != "delta":
                raise SystemExit("[sim] Cartesian impedance requires per-step delta policy actions")
            if state.A.policy_action_clamp or state.A.rf_overlap or any(state.A.world_offset):
                raise SystemExit("[sim] deployed cmd_twist requires --no-policy-action-clamp, zero overlap and world offset")
        if state.A.ik_replay:
            raise SystemExit("[sim] joint-pinning IK replay cannot evaluate Cartesian impedance")
    if fr3_uses_impedance_hook(state.A.fr3_controller):
        from robosuite_env.robots.fr3.impedance import ImpedanceConfig
        import json
        try:
            state.FR3_IMPEDANCE_CONFIG = ImpedanceConfig.from_mapping(json.loads(state.A.fr3_impedance))
        except (ValueError, TypeError) as exc:
            raise SystemExit(f"[sim] invalid --fr3-impedance: {exc}") from exc
    elif state.A.fr3_impedance != "{}":
        raise SystemExit(
            f"[sim] --fr3-impedance requires --fr3-controller {' or '.join(sorted(FR3_IMPEDANCE_HOOK_CONTROLLERS))}"
        )
    _RENDER_LOCK_PATH = (
        os.path.abspath(os.path.expanduser(state.A.render_lock)) if str(state.A.render_lock).strip() else ""
    )
    state._RENDER_LOCK_PATH = _RENDER_LOCK_PATH
    _RENDER_LOCK_STATE = threading.local()
    state._RENDER_LOCK_STATE = _RENDER_LOCK_STATE
    state.render_lock = functools.partial(mujoco_render_lock, state)

    GS_SERVER_PROCESS=GS_SERVER_LOG=None
    state.GS_SERVER_PROCESS = GS_SERVER_PROCESS
    state.GS_SERVER_LOG = GS_SERVER_LOG
    state._ROLLOUT_CLEANUP = GlobalResourceCleanup(
        state.cleanup_resources(), render_lock=state.render_lock
    ).cleanup
    state.at_exit_cleanup = functools.partial(
        cleanup_rollout_resources_at_exit, state
    )
    atexit.register(state.at_exit_cleanup)
    signal.signal(signal.SIGTERM, terminate_rollout)

    PROFILE = get_robot_profile(state.A.robot)
    state.PROFILE = PROFILE
    try:
        # Calibration is mandatory for every supported YUBI rollout.  Normalize
        # an empty CLI value back to the checked-in asset instead of allowing an
        # accidental empty environment variable to disable it.
        _calibration_path = str(state.A.gripper_calibration).strip() or DEFAULT_GRIPPER_CALIBRATION
        state._calibration_path = _calibration_path
        GRIPPER_CALIBRATION = load_gripper_calibration(_calibration_path)
        state.GRIPPER_CALIBRATION = GRIPPER_CALIBRATION
    except GripperCalibrationError as exc:
        raise SystemExit(f"[sim] invalid --gripper-calibration: {exc}") from exc
    if (
        state.A.policy_image_source == "sim"
        and GRIPPER_CALIBRATION is None
    ):
        raise SystemExit(
            f"[sim] {state.A.policy_yubi_render_profile} requires --gripper-calibration so "
            "rendering, contacts, control, and observations share one calibrated "
            "YUBI model"
        )
    if (
        not np.isfinite(float(state.A.policy_wrist_contrast))
        or float(state.A.policy_wrist_contrast) < 0.0
    ):
        raise SystemExit(
            "[sim] --policy-wrist-contrast must be finite and non-negative"
        )
    STATE_DUMP_FRAMES=_parse_int_set(state.A.state_dump_frames)
    state.STATE_DUMP_FRAMES = STATE_DUMP_FRAMES
    _STATE_DUMPED=set()
    state._STATE_DUMPED = _STATE_DUMPED
    state._STATE_SUCCESS_DUMPED=False
    DYNAMIC_CONTROL_REPEATS_USER_REQUESTED=int(state.A.dynamic_control_repeats)
    state.DYNAMIC_CONTROL_REPEATS_USER_REQUESTED = DYNAMIC_CONTROL_REPEATS_USER_REQUESTED
    _FR3_DYNAMIC_SOURCE_HZ=None
    state._FR3_DYNAMIC_SOURCE_HZ = _FR3_DYNAMIC_SOURCE_HZ
    _policy_action_hz=float(state.A.policy_action_hz)
    state._policy_action_hz = _policy_action_hz
    if not np.isfinite(_policy_action_hz) or _policy_action_hz < 0.0:
        raise SystemExit(
            "[sim] --policy-action-hz must be a finite non-negative number"
        )
    if _policy_action_hz > 0.0:
        if state.A.robot!="fr3" or state.A.physics!="dynamic" or not (state.A.policy or state.A.policy_batch_server):
            raise SystemExit(
                "[sim] --policy-action-hz requires an FR3 dynamic policy rollout"
            )
        if float(state.A.frame_hz)>0.0:
            raise SystemExit(
                "[sim] --policy-action-hz is mutually exclusive with --frame-hz; "
                "one clock must own both source actions and playback"
            )
        if state.DYNAMIC_CONTROL_REPEATS_USER_REQUESTED!=0:
            raise SystemExit(
                "[sim] --policy-action-hz requires --dynamic-control-repeats 0 so "
                "the phase scheduler owns every controller tick"
            )
        if int(state.A.dynamic_subframes)!=1:
            raise SystemExit(
                "[sim] --policy-action-hz currently requires --dynamic-subframes 1; "
                "intermediate rendering cannot change the source-action clock"
            )
        if not np.isclose(_policy_action_hz, round(_policy_action_hz), atol=1e-12, rtol=0.0):
            raise SystemExit(
                "[sim] --policy-action-hz must be an integer rate so one encoded "
                "playback frame represents exactly one source action"
            )
        try:
            # Validate that every source action can receive at least one complete
            # controller tick before any model, renderer, or output is created.
            nearest_control_ticks_for_source_action(
                0,
                _policy_action_hz,
                control_frequency_hz=float(state.A.dynamic_control_hz),
            )
        except ValueError as exc:
            raise SystemExit(f"[sim] invalid --policy-action-hz: {exc}") from exc
        _FR3_DYNAMIC_SOURCE_HZ=_policy_action_hz
        state._FR3_DYNAMIC_SOURCE_HZ = _FR3_DYNAMIC_SOURCE_HZ
    try:
        _DYNAMIC_TIMING = resolve_dynamic_timing(
            state.A.exec,
            state.A.dynamic_control_repeats,
            state.A.dynamic_subframes,
            # FR3's explicit source clock supports non-divisor rates through a
            # per-action cumulative-nearest schedule below. The legacy frame clock
            # retains its historical integer-ratio behavior.
            action_hz=(0.0 if state._FR3_DYNAMIC_SOURCE_HZ is not None else state.A.frame_hz),
            control_hz=state.A.dynamic_control_hz,
        )
        state._DYNAMIC_TIMING = _DYNAMIC_TIMING
    except ValueError as exc:
        raise SystemExit(f"[sim] {exc}") from exc
    if int(state.A.deterministic_seed) >= 0:
        random.seed(int(state.A.deterministic_seed))
        np.random.seed(int(state.A.deterministic_seed))
        try:
            cv.setRNGSeed(int(state.A.deterministic_seed))
            cv.setNumThreads(1)
        except Exception:
            pass
        print(f"[sim] deterministic seed={int(state.A.deterministic_seed)}", flush=True)
    _PREFLIGHT = run_runtime_preflight(
        state.A,
        worker_argv=sys.argv[1:],
        package_root=state._DIR,
    )
    state._PREFLIGHT = _PREFLIGHT
    ASSET_CONTRACT_REPORT = _PREFLIGHT.asset_contract_report
    state.ASSET_CONTRACT_REPORT = ASSET_CONTRACT_REPORT
    SCENE_CONTRACT_REPORT = _PREFLIGHT.scene_contract_report
    state.SCENE_CONTRACT_REPORT = SCENE_CONTRACT_REPORT
    if state.SCENE_CONTRACT_REPORT is None and state.A.desk == "scene":
        # Diagnostic executions (the post-hoc third-person replay drops
        # --strict-assets and the official attestations on purpose) still
        # need the schema-v2 table collision that `--desk scene` binds.
        # Validate the scene directly from --gs-background; the strict
        # preflight above remains the only path that attests the digest
        # against the launcher's run manifest.
        if not state.A.gs_background:
            raise SystemExit(
                "[sim] --desk scene requires a schema-v2 --gs-background directory"
            )
        from robosuite_env.contracts.scene import (
            SceneContractError,
            validate_scene_contract,
        )
        try:
            SCENE_CONTRACT_REPORT = validate_scene_contract(state.A.gs_background, strict=True)
            state.SCENE_CONTRACT_REPORT = SCENE_CONTRACT_REPORT
        except SceneContractError as exc:
            raise SystemExit(
                f"[sim] --desk scene requires a validated schema-v2 --gs-background: {exc}"
            ) from exc
    OFFICIAL_RUN_MANIFEST_PATH = _PREFLIGHT.official_run_manifest_path
    state.OFFICIAL_RUN_MANIFEST_PATH = OFFICIAL_RUN_MANIFEST_PATH
    OFFICIAL_WORKER_CONTRACT_PATH = _PREFLIGHT.official_worker_contract_path
    state.OFFICIAL_WORKER_CONTRACT_PATH = OFFICIAL_WORKER_CONTRACT_PATH
    EXECUTION_CLASS = _PREFLIGHT.execution_class
    state.EXECUTION_CLASS = EXECUTION_CLASS
    POLICY_SOURCE = bool(state.A.policy or state.A.policy_batch_server)
    state.POLICY_SOURCE = POLICY_SOURCE
    if state.POLICY_SOURCE:
        try:
            pio.validate_policy_pose_execution_contract(
                state.A.policy_pose_repr,
                robot=state.A.robot,
                policy_delta_anchor=state.A.policy_delta_anchor,
                rf_overlap=state.A.rf_overlap,
                action_clamp=state.A.policy_action_clamp,
            )
        except ValueError as exc:
            raise SystemExit(f"[sim] invalid policy pose execution contract: {exc}") from exc
    if state.A.state_replay_policy_obs_dir and not state.A.state_log_replay:
        raise SystemExit("[sim] --state-replay-policy-obs-dir requires --state-log-replay")
    if state.A.state_replay_policy_obs_display != "aspect-fit" and not state.A.state_replay_policy_obs_dir:
        raise SystemExit(
            "[sim] --state-replay-policy-obs-display requires "
            "--state-replay-policy-obs-dir"
        )
    if state.A.rf_overlap and not POLICY_SOURCE:
        raise SystemExit("[sim] --rf-overlap requires --policy or --policy-batch-server")
    if state.A.rf_overlap and state.A.policy_freeze_left:
        raise SystemExit(
            "[sim] --rf-overlap cannot be combined with --policy-freeze-left: "
            "left-reference rebasing requires the policy's left local deltas to be executed"
        )
    if int(state.A.rf_overlap_step_shift) < 0:
        raise SystemExit(
            f"[sim] --rf-overlap-step-shift must be non-negative, got {state.A.rf_overlap_step_shift}"
        )
    if int(state.A.rf_overlap_step_shift) != 0:
        raise SystemExit(
            "[sim] --rf-overlap-step-shift is only valid with a paired RF overlap "
            "sampler, which is no longer supported; leave it at 0"
        )
    try:
        pio.validate_rf_overlap(state.A.exec, state.A.rf_overlap)
    except ValueError as exc:
        raise SystemExit(f"[sim] {exc}") from exc
    if state.A.grasp_test:
        _valid_grasp = state.A.physics == "dynamic"
        state._valid_grasp = _valid_grasp
        if not _valid_grasp:
            raise SystemExit("[sim] --grasp-test requires --physics dynamic")
    DEFAULT_POLICY_OBJECTS = os.path.join(state._DIR, "obj_calib", "scene_positions_sam3.json")
    state.DEFAULT_POLICY_OBJECTS = DEFAULT_POLICY_OBJECTS
    DEFAULT_POLICY_EPISODE = os.environ.get("REAL2SIM_POLICY_EPISODE", "")
    state.DEFAULT_POLICY_EPISODE = DEFAULT_POLICY_EPISODE
    if state.A.policy_image_source != "sim" and not POLICY_SOURCE:
        raise SystemExit("[sim] --policy-image-source is only valid with --policy")
    if state.A.policy_pose_source != "sim" and not POLICY_SOURCE:
        raise SystemExit("[sim] --policy-pose-source is only valid with --policy")
    if (state.A.policy_image_source == "lerobot" or state.A.policy_pose_source == "lerobot") and not state.A.policy_image_episode:
        if not DEFAULT_POLICY_EPISODE:
            raise SystemExit(
                "[sim] lerobot policy image/pose input requires "
                "--policy-image-episode or REAL2SIM_POLICY_EPISODE"
            )
        state.A.policy_image_episode = DEFAULT_POLICY_EPISODE
    if POLICY_SOURCE and state.A.objects == "cup_plate" and os.path.isfile(DEFAULT_POLICY_OBJECTS):
        state.A.objects = DEFAULT_POLICY_OBJECTS
        print(f"[sim] standard objects -> {state.A.objects}", flush=True)
    (
        T,
        EE_OFF,
        ROOT_TCP_OFF,
        R_CANON_TO_FRANKA,
        Q_CANON_TO_FRANKA,
        Q_CANON_TO_ROBOT,
        Q_ROBOT_TO_CANON,
    ) = build_frame_constants(state)
    state.T = T
    LOGDIR = os.path.abspath(state.A.logdir) if state.A.logdir else state._DIR
    state.LOGDIR = LOGDIR
    os.makedirs(state.LOGDIR, exist_ok=True)
    state.EE_OFF = EE_OFF
    state.ROOT_TCP_OFF = ROOT_TCP_OFF
    state.R_CANON_TO_FRANKA = R_CANON_TO_FRANKA
    state.Q_CANON_TO_FRANKA = Q_CANON_TO_FRANKA
    state.Q_CANON_TO_ROBOT = Q_CANON_TO_ROBOT
    state.Q_ROBOT_TO_CANON = Q_ROBOT_TO_CANON

    _POLICY_WRIST_PREPROCESS_ORDER = (
        (
            "contrast_then_resize_with_pad"
            if state.A.policy_image_resize == "pad"
            else "contrast_then_resize_stretch"
        )
        if state.A.policy_image_source == "sim"
        else "resize_only"
    )
    state._POLICY_WRIST_PREPROCESS_ORDER = _POLICY_WRIST_PREPROCESS_ORDER

def banner(state):
    print("="*70)
    print("  UNIFIED SIM  (robosuite Environment base)")
    print("="*70)
    print(f"  robot          : {state.A.robot}")
    _dynamic_label=f"FR3 robosuite {state.A.fr3_controller.upper()} arm control, full dynamics"
    _kinematic_label="IK qpos pin + physical object contacts"
    print(f"  physics        : {state.A.physics}  ({_kinematic_label if state.A.physics=='kinematic' else _dynamic_label})")
    if get_robot_profile(state.A.robot).embodiment=="fr3":
        if state.A.fr3_controller=="joint_ik":
            print(f"  FR3 nullspace  : {state.A.fr3_nullspace}  (effective gain "
                  f"{effective_nullspace_gain(state.A.fr3_nullspace, state.A.joint_ik_nullspace_gain):g})")
    print(f"  grip ctrl      : {state.A.grip_ctrl}  substeps={state.A.substeps}  solref={state.A.solref!r} priority={state.A.grip_priority} "
          f"torque_scale={state.A.torque_scale} grip_torque_mult={state.A.grip_torque_mult}")
    print(f"  grip friction  : finger={state.A.grip_friction!r} rubber={state.A.rubber_friction!r}")
    print(f"  source         : {'GRASP-TEST' if state.A.grasp_test else ('REPLAY '+state.A.replay if state.A.replay else ('IK-REPLAY '+state.A.ik_replay if state.A.ik_replay else ('POLICY-BATCH '+str(state.A.policy_batch_server) if state.A.policy_batch_server else 'POLICY '+str(state.A.policy))))}")
    print(f"  instruction    : {state.A.instruction!r}")
    if state.POLICY_SOURCE:
        extra = f" ({state.A.policy_image_episode}, index={state.A.policy_image_index}{', loop' if state.A.policy_image_loop else ', clamp'})" if state.A.policy_image_source == "lerobot" else ""
        print(f"  policy images  : {state.A.policy_image_source}{extra}")
        print(f"  YUBI rendering : {state.A.policy_yubi_render_profile}")
        pose_extra = f" ({state.A.policy_image_episode}, index={state.A.policy_image_index}{', loop' if state.A.policy_image_loop else ', clamp'})" if state.A.policy_pose_source == "lerobot" else ""
        print(f"  policy pose    : {state.A.policy_pose_source}{pose_extra}")
        print(f"  policy actions : pose_repr={state.A.policy_pose_repr} delta_anchor={state.A.policy_delta_anchor} "
              f"clamp={bool(state.A.policy_action_clamp) if state.A.robot=='fr3' else 'n/a'}")
        print(f"  policy img rsz : {state.A.policy_image_resize}{'  (resize_with_pad = matches training)' if state.A.policy_image_resize=='pad' else '  (cv.resize stretch = legacy)'}")
        print(f"  policy img flow: {state._POLICY_WRIST_PREPROCESS_ORDER}")
        if state.A.policy_batch_server:
            print(f"  policy batch   : server={state.A.policy_batch_server}")
    if state.A.gripper_xml:
        print(f"  gripper XML    : {state.A.gripper_xml}")
    if state.A.gripper_xml_left:
        print(f"  gripper XML L  : {state.A.gripper_xml_left}")
    print(
        "  gripper calib  : "
        + (
            f"{state.GRIPPER_CALIBRATION.path} "
            f"(left_offset={state.GRIPPER_CALIBRATION.left_offset_rad:+.12f} rad, "
            f"sha256={state.GRIPPER_CALIBRATION.sha256[:12]})"
            if state.GRIPPER_CALIBRATION is not None
            else "(disabled)"
        )
    )
    print(f"  desk/scene     : {state.A.desk}")
    print(f"  objects        : {state.A.objects}")
    _galign = reported_gs_w2gs(state.A) or "-"
    print(
        f"  background 3DGS : {state.A.gs or '(none)'}  "
        f"src={state.A.gs_background_source}  "
        f"rot={state.A.gs_rot or ('matrix' if state.A.gs_rot_R is not None else '-')}  "
        f"port={state.A.gs_port} composite={state.A.gs_composite_mode}"
    )
    if state.A.gs:
        print(f"  GS align       : {_galign}")
        print(
            "  GS color gain  : "
            + " ".join(f"{float(value):.4g}" for value in state.A.gs_color_gain_rgb)
            + " (RGB, GS-only)"
        )
        print(
            "  GS color gamma : "
            + " ".join(f"{float(value):.4g}" for value in state.A.gs_color_gamma_rgb)
            + " (RGB, GS-only, before gain)"
        )
        print(
            "  GS calib offset: "
            + " ".join(f"{float(value):+.6f}" for value in state.A.gs_calibration_world_offset)
            + " m (added before world->GS)"
        )
    print(f"  output video   : {state.A.out}")
    print(f"  log dir        : {state.LOGDIR}  (sim_log.jsonl = per-frame IK+penetration, sim_summary.json)")
    print(f"  goal condition : {state.A.success_criterion}")
    print(f"  score condition: {state.A.score_condition}")
    print("="*70, flush=True)


__all__ = ["banner", "run"]
