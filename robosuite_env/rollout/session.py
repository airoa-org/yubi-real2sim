# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Mutable per-run rollout state shared by the worker stages.

Field names match the worker's module globals. The container is
behavior-free.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RendererSlots:
    """Named renderer handles with deterministic iteration order."""

    ren: Any = None
    wfL: Any = None
    wfR: Any = None
    wfL_obs: Any = None
    wfR_obs: Any = None

    def close(self, names: tuple[str, ...]) -> None:
        closed: set[int] = set()
        for name in names:
            renderer = getattr(self, name)
            if renderer is None:
                continue
            if id(renderer) not in closed:
                renderer.close()
                closed.add(id(renderer))
            setattr(self, name, None)

    def live_names(self, order: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(name for name in order if getattr(self, name) is not None)


class _CleanupResourceMapping(MutableMapping[str, Any]):
    """MutableMapping view used by GlobalResourceCleanup.

    The cleanup contract deliberately sees "env" as None. The environment close
    is therefore a no-op here, which keeps the EGL/GS release order.
    """

    _NAMES = (
        "logf",
        "_OVERVIEW_VIDEO_FRAMES",
        "_WRIST_VIDEO_FRAMES",
        "realcap",
        "wfL_obs",
        "wfR_obs",
        "wfL",
        "wfR",
        "ren",
        "env",
        "GS_SERVER_PROCESS",
        "GS_SERVER_LOG",
    )
    _RENDERER_NAMES = frozenset(
        ("wfL_obs", "wfR_obs", "wfL", "wfR", "ren")
    )

    def __init__(self, session: "RolloutSession") -> None:
        self._session = session

    def __getitem__(self, key: str) -> Any:
        if key == "env":
            return None
        if key in self._RENDERER_NAMES:
            return getattr(self._session.renderers, key)
        try:
            return getattr(self._session, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "env":
            return
        if key in self._RENDERER_NAMES:
            setattr(self._session.renderers, key, value)
            return
        setattr(self._session, key, value)

    def __delitem__(self, key: str) -> None:
        self[key] = None

    def __iter__(self) -> Iterator[str]:
        return iter(self._NAMES)

    def __len__(self) -> int:
        return len(self._NAMES)


@dataclass
class RolloutSession:
    """State rebound during one worker process."""

    # Handles and state already shared by the pre-decomposition stages.
    m: Any = None
    d: Any = None
    env: Any = None
    cup: Any = None
    POLICY_INIT_GRIP: Any = None
    YUBI: Any = None
    _STATE_SUCCESS_DUMPED: bool = False
    _SIM_TIME: float = 0.0
    ANCHOR_TABLE_ORIGIN: Any = None
    POLICY_ANCHOR_EFFECTIVE: Any = None
    POLICY_ANCHOR_EPISODE: Any = None
    _CAPTURE_INDEX: int = 0
    N: Any = None
    WALL_STOPPED: Any = None
    SUCCESS_STOPPED: Any = None
    _SUCCESS_FIRST_FRAME: Any = None
    maxpen: Any = None
    nfail: Any = None
    njump: Any = None
    npen: Any = None
    _JOINT_IK_FAIL_COUNT: Any = None
    _JOINT_IK_SOLVE_COUNT: Any = None
    _JOINT_IK_TARGET_REJECT_COUNT: Any = None
    _STATE_RENDER_MAX_APPLY_ERROR: Any = None
    _STATE_RENDER_ROWS_APPLIED: Any = None
    _VIDEO_NEEDS_WRISTS: Any = None
    _render_count: Any = None
    # What the first frame looked like before step 0: run_frames writes it,
    # finalize_outputs puts it in the summary.
    _start_state_report: Any = None
    _dynamic_grip_samples: Any = None
    _frame_capture_reps: Any = None
    _frame_render_reps: Any = None
    _frame_telemetry_reps: Any = None
    t: Any = None
    TGT: Any = None
    RULE_GRASP: Any = None
    _dump_state_once: Any = None
    DYNAMIC_YUBI_HOOK: Any = None
    gsc: Any = None
    _OVERVIEW_STATE_FRAMES: Any = None
    _POLICY_CHUNK_RECORDS: Any = None
    _RECS: Any = None
    _pstate: Any = None
    FR3_STRATEGY: Any = None

    # Paths and callable backends are explicit session ownership points.
    REPO: Any = None
    _DIR: Any = None
    _ROLLOUT_DIR: Any = None
    renderers: RendererSlots = field(default_factory=RendererSlots)
    render_lock: Any = None
    at_exit_cleanup: Any = None
    gs_api: Any = None
    policy_backend: Any = None
    _cid: Any = None
    _POLICY_WRIST_EPHEMERAL: bool = False
    _FR3_DYNAMIC_CONTROL_TICKS_PER_ACTION: Any = None
    _FR3_DYNAMIC_ENV_STEP_COUNT: Any = None
    _joint_ik_ready_gate_report: Any = None

    # Worker-stage globals, named as in the worker module.
    A: Any = None
    ALLARM: Any = None
    ARM: Any = None
    ARM_GEOMS: Any = None
    ASSET_CONTRACT_REPORT: Any = None
    BatchPolicyClient: Any = None
    CUP_GEOMS: Any = None
    GRASPABLE_GEOMS: Any = None
    CUPa: Any = None
    CUP_JOINT_NAME: Any = None
    DEFAULT_POLICY_EPISODE: Any = None
    DEFAULT_POLICY_OBJECTS: Any = None
    DYNAMIC_CONTROL_REPEATS: Any = None
    DYNAMIC_CONTROL_REPEATS_USER_REQUESTED: Any = None
    DYNAMIC_SUBFRAMES: Any = None
    DYNAMIC_SUBFRAMES_REQUESTED: Any = None
    DynamicYubiQfrcHook: Any = None
    EE_OFF: Any = None
    EXECUTION_CLASS: Any = None
    EnvCls: Any = None
    FINGER_SIDE_GEOMS: Any = None
    FRAME_DT_S: Any = None
    FRAME_HZ_EFFECTIVE: Any = None
    GH: Any = None
    GH_OTHER: Any = None
    GI_L: Any = None
    GI_R: Any = None
    GRASP_N: Any = None
    GRIPPER_CALIBRATION: Any = None
    GRIPPER_CALIBRATION_REPORT: Any = None
    GRIP_CTRL_EFFECTIVE: Any = None
    GRIP_GEOMS: Any = None
    GSBG: Any = None
    GSBackgroundClient: Any = None
    GS_DEPTH: Any = None
    GS_EFFECTIVE_MAT_W2GS: Any = None
    GS_OVERVIEW_ATTEMPTS: Any = None
    GS_RAW_MAT_W2GS: Any = None
    GS_SERVER_IDENTITY: Any = None
    GS_SERVER_LOG: Any = None
    GS_SERVER_PROCESS: Any = None
    GS_URL: Any = None
    IKGRIP: Any = None
    IK_JL: Any = None
    IK_JR: Any = None
    INNER_PAD_SIDE_GEOMS: Any = None
    LOGDIR: Any = None
    MIMICGEN_INIT_REPORT: Any = None
    MODEL_SIGNATURE: Any = None
    MUJOCO_INTEGRATOR: Any = None
    MUJOCO_INTEGRATOR_SELECTION: Any = None
    OBJECT_SETTLE_REPORT: Any = None
    OFFICIAL_RUN_MANIFEST_PATH: Any = None
    OFFICIAL_WORKER_CONTRACT_PATH: Any = None
    OUTPUT_FPS: Any = None
    OVFOV: Any = None
    PLATEa: Any = None
    PLATEd: Any = None
    POLICY: Any = None
    POLICY_LEROBOT_IMAGES: Any = None
    POLICY_LEROBOT_POSES: Any = None
    POLICY_OBS_DISPLAY: Any = None
    POLICY_OBS_DISPLAY_TRANSFORM: Any = None
    POLICY_OBS_LOGDIR: Any = None
    POLICY_OBS_LOGGED: Any = None
    POLICY_OBS_RAW: Any = None
    POLICY_SOURCE: Any = None
    PROFILE: Any = None
    Q_CANON_TO_FRANKA: Any = None
    Q_CANON_TO_ROBOT: Any = None
    Q_ROBOT_TO_CANON: Any = None
    R: Any = None
    READY: Any = None
    REPLAY_OBJ: Any = None
    #: Rig identity (fr3 / fr3_duo). Recorded, and bound to a cached
    #: init state and the compiled model signature.
    ROBOT: Any = None
    #: Mechanism family behind ROBOT. Guards FR3-only machinery so the
    #: Duo rig, which is two FR3s, keeps reaching it.
    EMBODIMENT: Any = None
    #: Validated RobotBasePlacement this run stands its bases with.
    BASE_PLACEMENT: Any = None
    ROOT_TCP_OFF: Any = None
    R_CANON_TO_FRANKA: Any = None
    SCENE_CONTRACT_REPORT: Any = None
    SKY: Any = None
    STATE_DUMP_FRAMES: Any = None
    STATE_ROWS: Any = None
    T: Any = None
    WristFisheye: Any = None
    _A0L: Any = None
    _A0R: Any = None
    _ABSOLUTE_JOINT_IK: Any = None
    _ARM_DESK_COLLISION_GATE: Any = None
    _ARM_LINK_COLLISION_GEOMS: Any = None
    _AV: Any = None
    _BGOPT: Any = None
    _CALIBRATION_PROVENANCE_KEYS: Any = None
    _CUP_MASS_EFFECTIVE: Any = None
    _CUP_MASS_XML: Any = None
    _DATA: Any = None
    _DEFER_OVERVIEW_RENDER: Any = None
    _DESK_COLLISION_GATE_GEOMS: Any = None
    _DROT_Q: Any = None
    _DYNAMIC_READY_DRIVEN: Any = None
    _DYNAMIC_READY_STEPS: Any = None
    _DYNAMIC_TIMING: Any = None
    _FINGER_DOFS: Any = None
    _FR3_DYNAMIC_SOURCE_HZ: Any = None
    _FRAME_METRICS: Any = None
    _FRAME_SPOOL_DIR: Any = None
    _FRAME_SPOOL_SAMPLES: Any = None
    _GQ: Any = None
    _INITIAL_ARM_JOINT_POSE: Any = None
    _INITIAL_ARM_JOINT_Q: Any = None
    _INITIAL_GRIP_Q: Any = None
    _INTEGRATOR_ENUMS: Any = None
    _INTEGRATOR_NAMES: Any = None
    _JOINT_IK_COMMAND: Any = None
    _JOINT_IK_COMMAND_ACCELERATION: Any = None
    _JOINT_IK_COMMAND_VELOCITY: Any = None
    _JOINT_IK_INITIAL_SAFETY: Any = None
    _JOINT_IK_LAST_RESULT: Any = None
    _JOINT_IK_LAST_SAFETY_DECISION: Any = None
    _JOINT_IK_LAST_SAMPLE: Any = None
    _JOINT_IK_LAST_TARGET_UPDATE: Any = None
    _JOINT_IK_SAFETY_STATS: Any = None
    _JOINT_IK_TARGET_ATTEMPT_COUNT: Any = None
    _JOINT_IK_TARGET_REJECT_REASONS: Any = None
    _JOINT_IK_TRAJECTORY: Any = None
    _JS: Any = None
    _LOADED_INIT_STATE: Any = None
    _LOCKED_POLICY_OBS_VIDEO_INSETS: Any = None
    _OBJECT_GEOM_IDS: Any = None
    _OBJECT_GEOM_OWNER: Any = None
    _OBJECT_ROOT_BODY_IDS: Any = None
    _OBJECT_SETTLE_SUPPORT_FORCE_N: Any = None
    _OBJECT_SETTLE_SUPPORT_NORMAL_Z: Any = None
    _OVERVIEW_RENDER_DATA: Any = None
    _OVERVIEW_SPOOL_RESERVE: Any = None
    _OVERVIEW_VIDEO_FRAMES: Any = None
    _PASS1_OVERVIEW_RENDERER_RELEASED: Any = None
    _PB: Any = None
    _PH: Any = None
    _POLICY_INSET_INPUT: Any = None
    _POLICY_PLAN_CONFIG: Any = None
    _POLICY_WRIST_PREPROCESS_ORDER: Any = None
    _POLICY_WRIST_RENDERER_STATS: Any = None
    _PREFLIGHT: Any = None
    _GRASP_SPAN: Any = None
    _P_BY_HAND: Any = None
    _P_GRASP: Any = None
    _P_PARK: Any = None
    _P_PRE: Any = None
    _P_TOP: Any = None
    _QPREV: Any = None
    _Q_FC: Any = None
    _READY_EE: Any = None
    _READY_GATE_REPORT: Any = None
    _READY_GRIP_CMD: Any = None
    _READY_HOLD_STEPS: Any = None
    _READY_JOINT_IK: Any = None
    _READY_JOINT_SEGMENTS: Any = None
    _READY_MOVE_STEPS: Any = None
    _READY_OBJECT_QPOS: Any = None
    _READY_PRIMARY_QUAT: Any = None
    _RENDER_LOCK_PATH: Any = None
    _RENDER_LOCK_STATE: Any = None
    _RENDER_STATE_CONTRACT: Any = None
    _ROLLOUT_CLEANUP: Any = None
    _SIM_LOG_ENABLED: Any = None
    _STATE_DUMPED: Any = None
    _STATE_LOG_SOURCE_SUMMARY: Any = None
    _STATE_LOG_SOURCE_SUMMARY_PATH: Any = None
    _STATE_RENDER_QPOS: Any = None
    _STATE_RENDER_ROWS_VALIDATED: Any = None
    _STATE_RENDER_VALIDATION: Any = None
    _STATE_REPLAY_POLICY_INSETS: Any = None
    _STATE_REPLAY_TARGET_BASELINE_SOURCE: Any = None
    _SUCCESS_CRITERION: Any = None
    _SUCCESS_OPTIONS: Any = None
    _SUPPORT_GEOM_IDS: Any = None
    _TASK_BINDING: Any = None
    _TASK_OBJECT_BINDINGS: Any = None
    _TASK_OBJECT_NAMES: Any = None
    _TASK_OBJECT_RENDER_INVENTORY: Any = None
    _TASK_OBJECT_TARGET: Any = None
    _VIDEO_WRIST_EPHEMERAL: Any = None
    _VIDEO_WRIST_RENDERER_STATS: Any = None
    _VIEW_LOOKAT: Any = None
    _WF: Any = None
    _WILL_CREATE_ANY_WRIST_RENDERER: Any = None
    _WILL_CREATE_POLICY_WRIST_RENDERER: Any = None
    _WILL_CREATE_VIDEO_WRIST_RENDERER: Any = None
    _WRIST_SPOOL_RESERVE: Any = None
    _WRIST_VIDEO_FRAMES: Any = None
    _acc: Any = None
    _anchor_dir: Any = None
    _anchor_json: Any = None
    _anchor_origin_geom: Any = None
    _anchor_payload: Any = None
    _arm_cfg: Any = None
    _batch_rgb: Any = None
    _batch_rgbd: Any = None
    _black_obs: Any = None
    _black_wrist: Any = None
    _body_name: Any = None
    _calibration_path: Any = None
    _cam0: Any = None
    _cbid: Any = None
    _cj: Any = None
    _close_error: Any = None
    _cup_init: Any = None
    _cur: Any = None
    _deploy_ready: Any = None
    _desk_clearance_mm: Any = None
    _expected_controller_action_dim: Any = None
    _expected_link_bodies: Any = None
    _finger_jids: Any = None
    _geom: Any = None
    _geom_id: Any = None
    _grasp_quat_side: Any = None
    _grasp_quat_topdown: Any = None
    _grip: Any = None
    _gs_calibration_world_offset: Any = None
    _gs_overview: Any = None
    _gs_overview_payload: Any = None
    _gs_server_command: Any = None
    _gx: Any = None
    _gy: Any = None
    _img0: Any = None
    _initial_joint_command: Any = None
    _initial_safety: Any = None
    _jids: Any = None
    _joint_cfg: Any = None
    _joint_ik_source_hz: Any = None
    _left: Any = None
    _left_finger_jids: Any = None
    _legacy_target_pos: Any = None
    _len: Any = None
    _lerp: Any = None
    _link_index: Any = None
    _loaded_validation: Any = None
    _lpre: Any = None
    _m: Any = None
    _missing_links: Any = None
    _name: Any = None
    _new_policy_wrist_pair: Any = None
    _new_video_wrist_pair: Any = None
    _nullspace_gain: Any = None
    _observed_link_bodies: Any = None
    _ogs: Any = None
    _overview_close_errors: Any = None
    _p: Any = None
    _pd: Any = None
    _phase_of: Any = None
    _plate_jid: Any = None
    _policy_action_hz: Any = None
    _policy_batch_infer: Any = None
    _q: Any = None
    _ready_goal: Any = None
    _ready_index: Any = None
    _ready_move_duration: Any = None
    _ready_result: Any = None
    _ready_seed: Any = None
    _ready_step: Any = None
    _ren0: Any = None
    _required_batch_available: Any = None
    _right: Any = None
    _right_finger_jids: Any = None
    _robot_index: Any = None
    _rpre: Any = None
    _runtime_gripper_types: Any = None
    _sc: Any = None
    _scene0: Any = None
    _scene_module: Any = None
    _source_settle: Any = None
    _source_settle_objects: Any = None
    _source_target_qpos: Any = None
    _source_target_state: Any = None
    _state_act: Any = None
    _state_ctrl: Any = None
    _state_qpos: Any = None
    _state_qvel: Any = None
    _state_replay_clock: Any = None
    _state_time: Any = None
    _stream: Any = None
    _task_joint_ids: Any = None
    _ticks_ratio: Any = None
    _timeout: Any = None
    _tloop: Any = None
    _ts: Any = None
    _ur: Any = None
    _valid_grasp: Any = None
    _validate_ephemeral_policy_capture: Any = None
    _validate_ephemeral_video_wrist_capture: Any = None
    _value: Any = None
    _video_batch_available: Any = None
    _video_batch_rgb: Any = None
    _video_batch_rgbd: Any = None
    _view_lookat: Any = None
    _view_raw: Any = None
    _vopt0: Any = None
    _wrist_profile: Any = None
    _yubi_model: Any = None
    aa: Any = None
    absL: Any = None
    absR: Any = None
    an: Any = None
    apply_delta_local: Any = None
    arg: Any = None
    arm_profile: Any = None
    ax: Any = None
    batch_render: Any = None
    cdof: Any = None
    compose_world_to_gs: Any = None
    composite_background_depth: Any = None
    composite_background_mask: Any = None
    ctrl: Any = None
    dg: Any = None
    e: Any = None
    ee_now: Any = None
    fr3_robot: Any = None
    geom_id: Any = None
    geom_ids: Any = None
    gid_: Any = None
    grasp_script: Any = None
    grasp_scripts: Any = None
    grip: Any = None
    j: Any = None
    joint_id: Any = None
    load_lerobot_policy_images: Any = None
    load_lerobot_policy_poses: Any = None
    logf: Any = None
    missing: Any = None
    names: Any = None
    nm: Any = None
    oj: Any = None
    ov: Any = None
    pL: Any = None
    pR: Any = None
    path: Any = None
    payloads: Any = None
    phi: Any = None
    pol_apply: Any = None
    policy: Any = None
    qL: Any = None
    qR: Any = None
    q_top: Any = None
    q_yaw: Any = None
    realcap: Any = None
    relL: Any = None
    relR: Any = None
    rel_state: Any = None
    ren: Any = None
    render_depth_and_foreground_mask: Any = None
    render_foreground_mask: Any = None
    render_source_frames: Any = None
    root_id: Any = None
    secrets: Any = None
    spec: Any = None
    step_action: Any = None
    value: Any = None
    vopt: Any = None
    wfL: Any = None
    wfL_obs: Any = None
    wfR: Any = None
    wfR_obs: Any = None
    wgs: Any = None
    xc: Any = None
    yc: Any = None
    zc: Any = None

    def cleanup_resources(self) -> MutableMapping[str, Any]:
        return _CleanupResourceMapping(self)


__all__ = ["RendererSlots", "RolloutSession"]
