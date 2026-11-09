# Configuration

## Frequently changed settings

| YAML key path | Effect | Details |
| --- | --- | --- |
| `policy.checkpoint` + `policy.config_name` | Select the checkpoint and the OpenPI config that supplies its normalization contract. | [Policy checkpoint and config binding](#policy-checkpoint-and-config-binding) |
| `scenes[].tag` + `scenes[].path` | Select a scene directory; the tag must match its `background.json` name. | [Scenes](#scenes) |
| `runner.repeats` + `runner.expected_rollouts` | Set repeats per scene and the required total, which must equal scenes times repeats. | [Scenes](#scenes) |
| `simulation.parameters.initial_arm_joints` + `simulation.parameters.base_placement` | Select the arms' start pose and the rig's base placement; a tilted shared mount needs `rpy_deg`. | [Initial arm joint poses](#initial-arm-joint-poses) and [Base placement](#base-placement) |
| `runner.shards` | Optionally split absolute repeat ranges across existing Slurm allocations. | [Rollout shards](rollout_shards.md) |
| `policy.exec_steps` + `policy.action_horizon` | Set how many actions run before re-inference and the model chunk length that must contain them. | [FR3 inverse kinematics](#fr3-inverse-kinematics) |
| `runtime.*_cuda_devices` + `runtime.server_port` + `runtime.gs_port_base` | Choose visible GPU slots and collision-free local ports for policy and 3DGS services. | [Configuration ownership](#configuration-ownership) and [parallelism and GPUs](#parallelism-and-gpus) |
| `simulation.parameters.gs_color_gain_rgb` | Apply per-channel gain to 3DGS background pixels without changing MuJoCo foreground colors. | [GS background color transform](#gs-background-color-transform) |
| `simulation.parameters.grasp_mode` | Choose contact/friction-only `physics` grasping or the explicit `rule` latch. | [Parallelism and GPUs](#parallelism-and-gpus) |

## Configuration ownership

Profiles use `.env` for machine-local inputs:

- `REAL2SIM_INPUTS_ROOT`: the single input root (see
  [inputs root](#inputs-root) below). Checkpoints, scenes, and external
  table-collision bundles are all addressed relative to it;
- optional `CUDA_HOME`, `REAL2SIM_POLICY_EPISODE`, and the OpenPI
  cache override listed in `.env.example`.

GPU visibility belongs to the shell that invokes `real2sim-run`, not `.env`.
Obtain GPU allocations before running this repository. The default launcher
path uses the current shell only; optional [`runner.shards`](rollout_shards.md)
starts `srun` steps in named, already-existing Slurm allocations. The
repository does not submit or cancel allocations and does not discover job
IDs.

### Inputs root

`REAL2SIM_INPUTS_ROOT` is the only asset-input root in `.env`. It must be an
absolute, symlink-free directory (the scene contract expands it in several
separate processes, and artifact verification compares exact paths); the
launcher normalizes `.`/`..` once and forwards that one string to the runner,
the run manifest, and artifact resolution. The canonical layout under it is:

```text
$REAL2SIM_INPUTS_ROOT/
├── checkpoints/<name>/           OpenPI checkpoint directories
├── scenes/<scene>/               schema-v2 scene directories (named by scene, not by tag)
└── registered-world/<bundle>/    external table-collision bundles (coacd/)
```

Profiles and `artifacts/manifest.yaml` reach it through the `{inputs_root}`
template field:

```yaml
policy:
  checkpoint: "{inputs_root}/checkpoints/pi05_cup"
scenes:
  - tag: franka_flat
    path: "{inputs_root}/scenes/flat_reading_room"
reproducibility:
  environment:
    allowlist: [..., CUDA_VISIBLE_DEVICES, REAL2SIM_INPUTS_ROOT]
    required: [HOME, PATH, REAL2SIM_INPUTS_ROOT]
runtime:
  gs_cuda_devices: "0"
  policy_cuda_device: "0"
  sim_cuda_devices: "0"
```

A profile that uses `{inputs_root}` must list `REAL2SIM_INPUTS_ROOT` in
`reproducibility.environment.allowlist` (and should list it in `required`);
that is what propagates the value to `run_policy_batch.sh`, the worker, and
artifact verification. A profile that never uses the field (for example
`config/release/fixture_v2.yaml`) needs no root. `output.*` must not use it.
Scene `background.json` files name an external collision directory with the
environment spelling, `${REAL2SIM_INPUTS_ROOT}/registered-world/<bundle>/coacd`,
because the scene contract expands environment references rather than
template fields; both spellings resolve to the same absolute string.

`REAL2SIM_INPUTS_ROOT` is the only input-path variable; a profile, `${...}`
reference, or manifest `local_path` that names another input-path variable is
rejected. Limit the host GPUs exposed to the process with
`CUDA_VISIBLE_DEVICES` in the invoking shell, then select zero-based visible
slots with `runtime.gs_cuda_devices`, `runtime.policy_cuda_device`, and
`runtime.sim_cuda_devices` in YAML. The release profiles select slot `0` for
the policy, 3DGS, and MuJoCo, so a two-GPU host runs one 3DGS server and all
MuJoCo on slot `0` by default. To use both GPUs, set the 3DGS and MuJoCo fields
to `"0,1"` in a local copy of the profile under `config/`, such as the
git-ignored `config/local_run_two_gpu.yaml`; the launcher only accepts profiles
under `config/`. Unsupported variables such as `REAL2SIM_GPU_IDS` and
`PORTABLE_POLICY_SERVER` fail closed when they are set; unset them.

## Policy checkpoint and config binding

Every policy rollout profile must bind a checkpoint to an explicit OpenPI
config. There is no implicit default config:

```yaml
policy:
  checkpoint: "{inputs_root}/checkpoints/pi05_cup"
  config_name: pi05_cup
  config_source: deploy
```

`pi05_cup` pins the normalization asset ID to `pi05_cup`.
The loader reads only
`<checkpoint>/assets/<asset_id>/norm_stats.json`; it does not search the
checkpoint tree. A missing configured file is fatal even if another
`norm_stats.json` exists elsewhere. The resolved asset ID, absolute path, and
SHA-256 are written to policy-server metadata for each new rollout.

The normal entry point is a parent simulation profile containing both fields:

```bash
# Inspect the resolved checkpoint/config pair first.
real2sim-run \
  --config config/release/fr3/flat/a_room_cup_cup1trim.yaml \
  --dry-run

# Run with that same pair.
real2sim-run \
  --config config/release/fr3/flat/a_room_cup_cup1trim.yaml
```

The supported public entry point is the profile-driven `real2sim-run` command
above. The launcher owns `scripts/run_policy_batch.sh`; that script only
orchestrates processes on the current host and never obtains GPU resources.
It is not a substitute for entering a provisioned interactive shell or for
the YAML GPU-role contract.

`CKPT` and `CONFIG` are developer environment overrides. Both must
be non-empty. Direct `sim.py --policy` and
`robosuite_env.policy.policy_batch_server --policy` calls likewise require
`--config`; a checkpoint-only invocation is rejected before inference.

Use YAML for reproducible experiment choices: robot, task, scene list,
controller, timing, render settings, repeats, ports, and output name. For a
release run, migrate to schema v2 and declare the permitted machine-local
variables under `reproducibility.environment`; the launcher does not inherit
the ambient process environment.

Temporary machine-specific run snapshots named
`config/local_run_*.yaml` are ignored by Git. Keep reusable experiment
semantics in the public profiles and move their paths back into `.env`.

The command examples below assume `source .venv/bin/activate` was run after
`scripts/bootstrap.sh`.

The launcher supports `${ENV_VAR}` plus the safe built-ins `{repo_root}`,
`{config_dir}`, `{timestamp}`, `{robot}`, and `{inputs_root}` (the normalized
`REAL2SIM_INPUTS_ROOT`, see [inputs root](#inputs-root)). An unset variable
or unknown field is a configuration error. Secrets are forbidden in YAML.
The integer `runtime.server_port` and `runtime.gs_port_base` fields may also
use `${ENV_VAR}` strings; after expansion they must resolve to ports in
`1..65535`, and the referenced variables must be declared in the
reproducibility environment allowlist.

## Schema versions

Schema v2 is the only rollout-config schema this repository accepts.
`load_rollout_config` rejects any `schema_version` other than `2` with a
`ConfigError` before any other validation runs.

Schema v2 is the fail-closed release contract. It:

- uses the complete typed `simulation.parameters` mapping instead of raw
  CLI-style arguments;
- rejects unknown and missing keys before launching;
- passes only explicitly allowlisted environment variables to children and
  rejects undeclared behavior-changing variables; path expansion uses this
  same resolved allowlist, so ambient values cannot override YAML `values`;
- binds each checkpoint, scene/table-collision pair, episode, robot payload, and runtime
  bundle to a stable artifact ID in `artifacts/manifest.yaml`;
- verifies artifact bytes rather than merely checking path existence; and
- records code/source commits, lockfile hashes, artifact digests, resolved
  environment, host, GPU/driver, and accelerator profile in
  `run_manifest.json`.

Schema v2 requires `simulation.parameters.desk: scene` and
`desk_world_aligned: true`. The exact scene `background.json` owns the table
collision; there is no independent runtime desk selector and no table visual
mesh in MuJoCo. See [simulation operations](simulation_operations.md).

`config/release/fixture_v2.yaml` is the canonical structural example. It is
safe for clean CI but its tiny inputs are intentionally unable to perform a
rollout. The shipped profiles are the two FR3 Flat cup-on-plate release
profiles under `config/release/fr3/flat/`; each binds its runtime inputs to
`artifacts/manifest.yaml` and selects a visible GPU slot for every GPU role.

`config/release/fr3/flat/a_room_cup_cup1trim.yaml` is the environment check that a
new host runs first, and the only release profile in production mode. The other
release profile, `reading_room_cup_cup1trim.yaml`, uses `mode: fixture`. The
environment check's reproducibility section:

```yaml
reproducibility:
  mode: production
  artifact_manifest: '{repo_root}/artifacts/manifest.yaml'
  artifacts:
    checkpoint: pi05-cup
    scene.franka_flat: public-scene-flat-a-room-v2
    runtime-assets: public-cup-plate-visual-v1
    robot-assets: fr3-collision-franka-description-2fd6aaa8
  lockfiles:
  - '{repo_root}/uv.lock'
  source_dependencies:
  - '{repo_root}/third_party/openpi'
  - '{repo_root}/third_party/GaussianWrapping'
  - '{repo_root}/third_party/robosuite'
  - '{repo_root}/third_party/franka_description'
  support_matrix: '{repo_root}/config/support-matrix.yaml'
  accelerator_profile: cuda12-x86_64
  environment:
    allowlist:
    - HOME
    - PATH
    - LANG
    - LC_ALL
    - TZ
    - TMPDIR
    - LD_LIBRARY_PATH
    - CUDA_HOME
    - SSL_CERT_FILE
    - SSL_CERT_DIR
    - REAL2SIM_INPUTS_ROOT
    - CUDA_VISIBLE_DEVICES
    - XLA_PYTHON_CLIENT_MEM_FRACTION
    - XLA_PYTHON_CLIENT_ALLOCATOR
    required:
    - HOME
    - PATH
    - REAL2SIM_INPUTS_ROOT
    values:
      XLA_PYTHON_CLIENT_MEM_FRACTION: '0.45'
      XLA_PYTHON_CLIENT_ALLOCATOR: cuda_async
```

Production runs also require clean Git worktrees and complete provenance. The
checkout in `runtime.gaussian_wrapping_repo` must also be at the
`third_party/GaussianWrapping` gitlink.
Missing commits, locks, artifact metadata, or CUDA GPU/driver detection aborts
before execution.

The clean-tree check covers untracked files, in the repository and in every
source dependency. The production preflight and doctor also require the locked
`uv.lock` environment from `scripts/bootstrap.sh` with `uv` on `PATH`, the
Gaussian CUDA extensions that bootstrap builds, FFmpeg, and readable GPU and
driver details, and they verify each artifact's size and digest. They run
before `--check-runtime` reports `ok` and before a run starts. The doctor
report is stored in the run's `run_manifest.json` under `preflight`.

To experiment, copy the profile to a git-ignored `config/local_run_*.yaml` and
set `mode: fixture`, or start from a fixture-mode profile.

## Complete schema-v2 key reference

`src/real2sim/config/keys.py` is the executable key set and
`config/release/fixture_v2.yaml` is the complete copyable example. Unknown keys
and missing required keys fail before launch. In the tables below, “optional”
means the key may be omitted; a required key may still deliberately contain
`null` where its value description permits it.

### Top level and runner

| Key | Requirement and value |
| --- | --- |
| `schema_version` | Required; exactly `2`. |
| `robot` | Optional; `fr3` (default) or `fr3_duo` (`_ROBOT_CLI_NAMES` in `src/real2sim/config/keys.py`). |
| `name` | Required; 3–64 lowercase letters, digits, `_`, or `-`. |
| `description` | Optional free text; defaults to an empty string. |
| `runner`, `policy`, `task`, `scenes`, `simulation`, `render`, `runtime`, `output`, `reproducibility` | Required mappings/list described below. |
| `runner.mode` | Required; `fast_parallel` or `sequential_waves`. |
| `runner.repeats` | Required positive integer per scene. |
| `runner.policy_batch_size` | Required positive integer; divisible by scene count. Per-scene concurrency is this value divided by scene count and must divide `repeats`. |
| `runner.frames` | Required positive policy-action count. Every release profile under `config/release/<robot>/<rig>/` sets `300`; `fixture_v2.yaml` sets `1`. |
| `runner.max_wall_sec` | Required positive worker/policy-barrier timeout. |
| `runner.expected_rollouts` | Required; exactly `len(scenes) * repeats`. |

Every release profile under `config/release/<robot>/<rig>/` sets `runner.frames: 300`, 30 s
at `frame_hz: 10` (`fixture_v2.yaml`, the structural example, sets `1`).

`runner.parallel_repeats` is a derived resolved-plan field, not a legal YAML
key. `sequential_waves` additionally requires one concurrent repeat per scene.

### Policy, task, and scenes

| Key | Requirement and value |
| --- | --- |
| `policy.python` | Required path to the policy Python executable. |
| `policy.checkpoint` | Required checkpoint directory. |
| `policy.config_name` | Required OpenPI config name. |
| `policy.config_source` | Required; `deploy` or `training`. |
| `policy.action_horizon` | Required positive action-chunk length; at least `exec_steps + rf_overlap`. |
| `policy.exec_steps` | Required positive number of rows executed before re-inference. |
| `policy.noise_seed` | Required; `null` or a non-negative integer. |
| `policy.deterministic_xla` | Required boolean. |
| `task.objects` | Required object preset or positions-JSON path; built-in cup/plate usage is described under [Adding a task](#adding-a-task). |
| `task.instruction` | Required policy instruction. |
| `task.goal_condition` | Required registered final-success condition. |
| `task.score_condition` | Required registered score condition. |
| `task.policy_image_episode` | Optional path or `null` for recorded policy images. |
| `scenes[].tag` | Required unique lowercase runner tag. |
| `scenes[].path` | Required unique scene directory; its contract `name` must equal the tag. |

### Simulation wrapper

| Key | Requirement and value |
| --- | --- |
| `simulation.deterministic` | Required boolean. |
| `simulation.seed` | Required non-negative integer. |
| `simulation.init_state` | Required; `null`, a state-JSON path, or a mapping containing exactly `fr3`. |
| `simulation.rf_overlap` | Required non-negative integer, no greater than `policy.exec_steps`. |
| `simulation.rf_overlap_sampler` | Required; exactly `single`. |
| `simulation.rf_overlap_prev_weight` | Required; exactly `null` with the single sampler. |
| `simulation.rf_overlap_step_shift` | Required; exactly `0` with the single sampler. |
| `simulation.grip_torque_mult` | Required non-negative number. |
| `simulation.gripper_calibration` | Optional path or `null`. |
| `simulation.fr3_controller` | Optional; `joint_ik` or `cartesian_impedance` (default `joint_ik`; see [FR3 Cartesian impedance](fr3_impedance.md)). |
| `simulation.fr3_nullspace` | Optional; `home` or `off` (default `home`). |
| `simulation.yubi_controller` | Optional; `cbpc` or `kinematic` (default `cbpc`). |
| `simulation.parameters` | Required mapping containing every parameter below; a key documented as optional may be omitted and takes its default from `_V2_OPTIONAL_PARAMETER_DEFAULTS` in `src/real2sim/config/keys.py`. |

`simulation.extra_args` and a wrapper-level `simulation.grasp_mode` are
forbidden; use typed parameters.

### All 96 `simulation.parameters` keys

| Area | Keys and constraints |
| --- | --- |
| Execution and policy input | `simulation.parameters.physics` (`kinematic`/`dynamic`); `simulation.parameters.grasp_mode` (`physics`/`rule`); `simulation.parameters.policy_anchor` (`deploy`/`ready`); `simulation.parameters.initial_arm_joints` (registered pose name or `null`) and `simulation.parameters.initial_arm_joints_json` (path or `null`) -- two spellings of one selection, at most one of which may be set (see [Initial arm joint poses](#initial-arm-joint-poses)); `simulation.parameters.policy_image_source` and `simulation.parameters.policy_pose_source` (`sim`/`lerobot`); `simulation.parameters.policy_image_resize` (`stretch`/`pad`); optional `simulation.parameters.policy_yubi_render_profile` (`elp-mean-wrist-rgb`/`franka-pinhole-wrist-rgb`, default `elp-mean-wrist-rgb`; see [Wrist camera model](#wrist-camera-model)); non-negative `simulation.parameters.policy_wrist_contrast`; booleans `simulation.parameters.policy_obs_inset`, `simulation.parameters.policy_image_loop`, and `simulation.parameters.no_obs_gs`; optional diagnostic `simulation.parameters.obs_gs_only` (boolean, default `false`, mutually exclusive with `no_obs_gs`) removes the MuJoCo foreground while preserving the wrist observation projection; `simulation.parameters.policy_delta_anchor` (`observation`/`live`); `simulation.parameters.policy_image_index` (`infer`/`frame`); optional `simulation.parameters.policy_action_clamp` (boolean, default `true`); optional `simulation.parameters.policy_state_quat_hemisphere` (`auto`/`positive`/`negative`, default `auto`; see [Policy state quaternion hemisphere](#policy-state-quaternion-hemisphere)). |
| Scene and timing | `simulation.parameters.desk` (required value `scene`); `simulation.parameters.desk_world_aligned` (required `true`); optional `simulation.parameters.base_placement` (path to a schema-2 base placement, `robosuite_env/robots/base_placement.py`: `frame: table`, the `robot` it was calibrated for, which must be this run's rig, and per base slot a `pos` plus exactly one of `yaw_deg`/`rpy_deg`; omitted means the rig's default in `DEFAULT_BASE_PLACEMENTS` -- `fr3_base_from_blend.json` or `fr3_base_duo.json` -- see [Base placement](#base-placement)); positive integer `simulation.parameters.dynamic_control_hz`; non-negative `simulation.parameters.frame_hz` and `simulation.parameters.dynamic_ready_sec`; `simulation.parameters.dynamic_subframes` (required value `1`); positive integer `simulation.parameters.substeps`; `simulation.parameters.mujoco_integrator` (`keep`, `euler`, `rk4`, `implicit`, or `implicitfast`). |
| FR3 IK | Non-negative `simulation.parameters.joint_ik_max_velocity`, `simulation.parameters.joint_ik_max_acceleration`, `simulation.parameters.joint_ik_max_jerk`, `simulation.parameters.joint_ik_nullspace_gain`, `simulation.parameters.joint_ik_ready_position_tol_mm`, and `simulation.parameters.joint_ik_ready_orientation_tol_deg`; `simulation.parameters.ik_jump_reject` (`off`, `hold`, or `retry-hold`); non-negative integer `simulation.parameters.ik_jump_retries`; optional positive `simulation.parameters.fr3_joint_kp` and non-negative `simulation.parameters.fr3_joint_kd`, which override the JOINT_POSITION gains in `robosuite_env/controllers/default_fr3_joint_ik.json` and require `physics: dynamic` -- omitted, the controller JSON is used byte-for-byte. |
| Contact and mass | Non-negative `simulation.parameters.torque_scale`; `simulation.parameters.grip_friction` and `simulation.parameters.rubber_friction` (`keep` or three non-negative numbers); two-number `simulation.parameters.solref` and `simulation.parameters.rubber_solref`; three-number `simulation.parameters.rubber_solimp`; non-negative integers `simulation.parameters.grip_priority` and `simulation.parameters.rubber_priority`; non-negative `simulation.parameters.cup_mass`. |
| Rule grasp | `simulation.parameters.rule_grasp_hand` (`auto`, `R`, or `L`); non-negative `simulation.parameters.rule_grasp_effort_eps`, `simulation.parameters.rule_grasp_policy_eps_rad`, `simulation.parameters.rule_grasp_release_eps_rad`, and `simulation.parameters.grasp_contact_tol_mm`; boolean `simulation.parameters.debug_grasp_contact`. |
| 3DGS | Positive `simulation.parameters.gs_render_timeout_sec`; positive integer `simulation.parameters.gs_render_attempts`; non-negative `simulation.parameters.gs_render_retry_delay_sec`; `simulation.parameters.gs_calibration_world_offset` (required `[0, 0, 0]`); optional `simulation.parameters.gs_composite_mode` (`depth`/`foreground`, default `depth`); optional `simulation.parameters.gs_color_gain_rgb` and `simulation.parameters.gs_color_gamma_rgb` (three values in `(0, 4]`, default `[1, 1, 1]`). |
| Wrist backend | Optional `simulation.parameters.policy_wrist_backend` (only `direct_kb4`, the default; see [Direct KB4 wrist rendering](direct_kb4_development.md)). The `direct_kb4` near cull is not a profile parameter: the launcher always passes `--gs-kb4-near-cull znear` (0.01 m), and a run records it as `gs_kb4_near_cull` and `gs_kb4_near_m` in `sim_summary.json` (they describe the `direct_kb4` render only); a replay takes the recorded value. |
| Success | Non-negative `simulation.parameters.success_contact_tol_mm`, `simulation.parameters.success_contact_penetration_tol_mm`, `simulation.parameters.success_place_contact_penetration_tol_mm`, `simulation.parameters.success_plane_gap_tol_mm`, `simulation.parameters.success_plane_penetration_tol_mm`, `simulation.parameters.success_bottom_contact_tol_mm`, `simulation.parameters.success_footprint_slack_mm`, and `simulation.parameters.success_upright_tol_deg`; `simulation.parameters.success_mode` (only `bbox`). |
| Task-scoped (optional) | Omitted by every profile that does not need them; a key that is absent renders no worker argument. Integer 0..5 `simulation.parameters.box_pack_task_index`; either a path or one-path-per-repeat list for `simulation.parameters.box_pack_layout_json`, each refused for any other `task.objects` (see [Task layout selection](#task-layout-selection)); non-negative `simulation.parameters.success_stable_sec`, `simulation.parameters.success_linear_speed_mm_s`, and `simulation.parameters.success_angular_speed_rad_s`; non-negative `simulation.parameters.dynamic_object_settle_max_sec`, `simulation.parameters.dynamic_object_settle_min_sec`, `simulation.parameters.dynamic_object_settle_stable_sec`, `simulation.parameters.dynamic_object_settle_linear_speed_mm_s`, `simulation.parameters.dynamic_object_settle_angular_speed_rad_s`, `simulation.parameters.dynamic_object_settle_linear_acceleration_m_s2`, `simulation.parameters.dynamic_object_settle_angular_acceleration_rad_s2`, `simulation.parameters.dynamic_object_settle_position_drift_mm`, and `simulation.parameters.dynamic_object_settle_orientation_drift_deg`. |
| World | `simulation.parameters.table_z` (required `0.0`: the schema-v2 reference plane, not the height of the table collision top; see [Success heights](#success-heights)); `simulation.parameters.world_offset` (three numbers). |

### Initial arm joint poses

`simulation.parameters.initial_arm_joints` selects one of the checked-in start
poses by name. `simulation.parameters.initial_arm_joints_json` takes an
explicit path instead. A profile may set either one or neither; setting both
is ambiguous and is rejected. Both spellings resolve to the same
`--initial-arm-joints-json <path>` worker argument, so the joint ready gate and the
settled init-state contract behave identically either way. An unregistered name is an error, never a
silently ignored value.

The registry lives in `src/real2sim/config/initial_arm_joints.py`. List it with:

```bash
real2sim-run --list-initial-arm-joints
```

| Name | Older name | Schema | Gripper |
| --- | --- | --- | --- |
| `fr3/duo/init_high` | `duo_init_high` | 2 | `0.7313` / `0.7466` rad (left / right) |
| `fr3/duo/home` | – | 2 | `0.7` rad, both hands |
| `fr3/duo/home_20260906` | – | 2 | `0.7` rad, both hands |
| `fr3/flat/grip07` | `grip07` | 2 | `0.7` rad, both hands |
| `fr3/flat/registered_cup_calibrated` | `registered_cup_calibrated` | 1 | none |

Each name is the file's path under `assets/initial_poses/` without `.json`;
the file's `signature` block names the same robot and rig (see
`assets/initial_poses/README.md`).

#### Pose file schema

`schema_version: 1` is the arms-only contract: `units: rad`, `joint_order`
`[j1..j7]`, and seven-number `left_arm` / `right_arm` vectors. Unknown keys are
rejected.

`schema_version: 2` adds exactly one optional `gripper` block and changes
nothing else, so every existing schema-1 file stays valid unchanged and keeps
letting the rollout settle the gripper from the model state:

```yaml
gripper:
  coordinate: raw
  joint_order: [hand_left_motor_joint, hand_right_motor_joint]
  left: 0.7
  right: 0.7
```

`coordinate` must be `raw` and `joint_order` must be those two real-robot motor
names. `robosuite_env/assets/gripper_calib_rad.json` defines
`calibrated_left_angle_rad = raw_left_angle_rad + left_offset_rad`, and
`robosuite_env/robots/yubi/calibration.py` applies it by rotating the compiled
finger *body frames* while asserting that `qpos0`, the joint ranges, and the
actuator `ctrlrange` are unchanged (`policy_coordinate: "raw"`). Finger `qpos`
and every finger command therefore stay in the raw motor coordinate the real
robot publishes on its gripper joint-trajectory topic and that LeRobot
stores as `observation.joint_states[:2]`, so a recorded motor angle is used
verbatim. Declaring `coordinate: calibrated` is rejected rather than converted;
converting would move the angle by 5.852 deg. Angles must lie in
`[0, 0.7854]` rad, the interval the runtime clamps finger commands to.

When a pose declares a gripper angle, it replaces the ready drive's
fully-open command and is forced onto both hands once the drive finishes, so
the object settle and the first policy observation start from it exactly. A
MimicGen replay supplies its own initial gripper angle; combining the two is
rejected.

### Base placement

`simulation.parameters.base_placement` names a schema-2 base placement file
(`robosuite_env/robots/base_placement.py`). A profile that omits the key gets
its rig's default from `DEFAULT_BASE_PLACEMENTS`: `fr3_base_from_blend.json`
for `fr3` and `fr3_base_duo.json` for `fr3_duo`, both under
`robosuite_env/obj_calib/`. A file whose `robot` field names another rig is
refused.

Each base slot sets **either** `yaw_deg` **or** `rpy_deg`, and
`base_placement.py` refuses a slot that sets both or neither. A tilted mount,
such as the shared FR3 Duo mount, needs `rpy_deg`, because a yaw cannot
describe it.

`rpy_deg` is quoted the way URDF composes it, `Rz(y) @ Ry(p) @ Rx(r)`.
robosuite's `euler2mat` composes the other way, `Rx(r) @ Ry(p) @ Rz(y)`, so
`place_robots_at_calibrated_bases` (`robosuite_env/tasks/scene.py`) re-solves
the triple through `robosuite_base_euler` before calling `set_base_ori`. **Do
not pass rpy from a calibration file to `set_base_ori` yourself.** A yaw-only
pose is a fixed point of that conversion, so the difference only shows for
poses with roll or pitch.

### Task layout selection

`box_pack` places its objects from a hand-authored
layout (see [Where the objects start](#where-the-objects-start)), and a layout
is placed for one scene and, optionally, particular rigs. The optional
`simulation.parameters.box_pack_layout_json`
names the file; box packing may instead name a non-empty list
of files. The paths accept
`{repo_root}` and `{inputs_root}` and are made absolute at load. The preflight
and the worker refuse each one for any other `task.objects`.

The launcher preflight, the worker, the environment, the init-state identity,
and the run summary all resolve the layout through `select_task_layout` in
`robosuite_env/tasks/layout_selection.py`, in one order:

1. the profile parameter, rendered as the worker flag of the same name
   (`--box-pack-layout-json`);
2. the harness-only `RS_BOX_PACK_LAYOUT_JSON`.
   A profile cannot reach this step: naming it in
   `reproducibility.environment.allowlist` or `required` is a configuration
   error, and an ambient `RS_*` variable outside the allowlist is already
   refused. When a flag and its override are both set they must resolve to the
   same file (the worker exports the profile's file under the override name for
   the environment); two different files are refused, never ranked;
3. the task default: `obj_calib/layout_box_pack_duo_manual.json`.

For box packing, a layout list must contain exactly `runner.repeats` entries
and `simulation.init_state` must be `null`. Rollout directory/client `run_NN`
selects zero-based list index `(NN - 1) % len(list)`: `run_01` takes the first
entry and the last rollout takes the last entry. The runner passes the selected
path and index to each worker and prepares or reuses a separate
content-addressed deterministic init state for each layout.

`--check-runtime` and launch (`_validate_selected_layout` in
`src/real2sim/preflight.py`) check the resolved file, and the environment checks
it again when it loads it:

- it exists and validates as a manual layout for the task's objects;
- its `source.scene` equals every `scenes[].tag` (in the worker, the tag of the
  validated scene contract; an environment without one is refused);
- if it names rigs, the run's `robot` is one of them. Rigs come from the optional
  `source.robots` list (non-empty, no duplicates). A layout that names no rig is
  accepted on any rig of its scene.

`sim_summary.json` records `task_layout`: `task`, `parameter`, `configured`,
`selected_by` (`profile`, `harness_override`, or `default`), `path`, `sha256`,
`scene`, and `running_scene`. Box-pack layout-list rollouts additionally record
zero-based `index`, and `pen_offset_mm` / `box_offset_mm` as `{dx, dy}` when
those values occur under `source.sweep` (otherwise `null`). Invalid offset
metadata is refused by launcher preflight; non-launcher diagnostics record a
null offset and its validation reason rather than failing summary finalization.
The summary is written only after the selection is
re-resolved and found to name the file the environment loaded, with unchanged
bytes. The layout's content hash is also part of the init-state identity (see
[fail-closed checks](simulation_operations.md#fail-closed-checks)).

### Policy state quaternion hemisphere

The YUBI policy state is the left-to-right hand-root pose
`[dx, dy, dz, qx, qy, qz, qw]` followed by both gripper angles. `q` and `-q` are
the same rotation, but the policy consumes the raw components, so the sign sent
must match the checkpoint's training state. The optional
`simulation.parameters.policy_state_quat_hemisphere` (default `auto`) sets it
(`robosuite_env/policy/policy_io.py`); actions are unaffected.

| Value | Quaternion sent |
| --- | --- |
| `positive` | `qw >= 0` on every observation. |
| `negative` | `qw <= 0` at the rollout's first observation; afterwards the sign that keeps `dot(q, q_previous) >= 0`, so the sign never jumps within a rollout. |
| `auto` | Resolved with the plan from the checkpoint's single `assets/**/norm_stats.json`: `q99[qw] < 0` gives `negative`, `q01[qw] > 0` gives `positive`, and mixed signs give `positive`. |

`auto` refuses more than one `norm_stats.json`, and statistics whose qw index
cannot be confirmed: a state shorter than nine components, non-zero dimensions
beyond nine, or dimensions 3..6 that are not unit-quaternion components. Only a
resolved `negative` renders `--policy-state-quat-hemisphere negative`;
`positive` renders nothing, so a profile that resolves `positive` keeps its
worker argument vector. The resolved plan records the result at
`simulation.policy_state_quat_hemisphere`, for `auto` with the norm-stats path,
SHA-256, and qw quantiles.

The value is checked three times:

- `--check-runtime` and launch re-resolve it against `policy.checkpoint`
  (`validate_policy_state_quat_hemisphere`). A rendered flag that disagrees with
  the record, an explicit value the statistics contradict, and `auto` statistics
  whose SHA-256 changed since resolution are refused.
- A checkpoint with no norm stats at all cannot verify the value, and leaves
  `auto` unresolved with no flag. The preflight refuses that run whatever
  `reproducibility.mode` says, unless `reproducibility.artifacts.checkpoint` is a
  placeholder fixture (manifest `access: fixture` with a
  `fixture://tests/fixtures/artifacts/` URI), which runs unverified. Statistics
  that exist but cannot be read are always refused, and so are statistics that
  appear after an unresolved plan.
- The policy resolves the hemisphere again from the norm stats it actually
  loaded and publishes `policy_state_quat_hemisphere` and
  `policy_state_quat_hemisphere_resolution` (a batch server in its hello
  metadata and `policy_batch_server_ready.json`). The worker refuses a different
  value or a policy that publishes none.

With `policy_pose_source: lerobot` the dataset quaternion is replayed with its
own sign, and the record says `applied_to_observation: false`.
`sim_summary.json` records `policy_state_quat_hemisphere` and
`policy_state_quat_hemisphere_provenance` (what it was verified against, the
norm-stats SHA-256 and qw quantiles, and `sent_observations`: counts, the sent qw
range, observations in the opposite hemisphere, and sign jumps between
consecutive observations). Each `logs/policy_chunks.jsonl` record carries the
hemisphere, whether it was verified, and the norm-stats SHA-256 and qw
quantiles.

### Render, runtime, and output

| Key | Requirement and value |
| --- | --- |
| `render.view_azimuth`, `render.view_elevation` | Required numeric free-camera angles. |
| `render.view_distance` | Required positive distance. |
| `render.view_lookat` | Required list of three numbers. |
| `render.wrist`, `render.wrist_gs`, `render.origin`, `render.analysis`, `render.clean_video` | Required booleans; `wrist_gs` requires `wrist`, while `clean_video` requires `wrist: false`. |
| `render.enabled` | Optional boolean, default `true`; requests final video, though generation with a policy batch of 10 or more is deferred. |
| `render.third_person` | Optional boolean, default `true`; controls required post-hoc third-person finalization. |
| `runtime.server_host` | Required policy-server host string. |
| `runtime.server_port`, `runtime.gs_port_base` | Required port integer or a whole `${VAR}` reference resolving to `1..65535`. |
| `runtime.gs_cuda_devices` | Optional comma-separated non-negative visible GPU slots; required explicitly in production. |
| `runtime.policy_cuda_device` | Optional one non-negative visible GPU slot; otherwise the first GS slot is used. Checked-in production profiles set it explicitly. |
| `runtime.sim_cuda_devices` | Optional comma-separated non-negative visible GPU slots; required explicitly in production. |
| `runtime.gs_start_timeout_sec`, `runtime.sim_num_threads` | Required positive integers. |
| `runtime.render_lock_root` | Optional lock-directory path. |
| `runtime.gaussian_wrapping_repo` | Required GaussianWrapping repository path. |
| `output.root` | Required output directory; may not use `{inputs_root}`. |
| `output.run_name` | Required template rendering to one 3–128 character safe path component. |

`render.desk_texture` is forbidden because scene appearance is embedded in
3DGS. `render.blender_export` and `render.build_blend` are rejected.

### Reproducibility

Every key in this section is required:

| Key | Value |
| --- | --- |
| `reproducibility.mode` | `fixture` or `production`; production requires a CUDA support profile. |
| `reproducibility.artifact_manifest` | Artifact-manifest path. |
| `reproducibility.artifacts` | Non-empty role-to-artifact-id mapping; role names use lowercase letters, digits, `.`, `:`, `_`, or `-`. |
| `reproducibility.lockfiles` | List of lockfile paths. |
| `reproducibility.source_dependencies` | List of source-tree paths. OpenPI and robosuite install from `uv.lock`, so `{repo_root}/third_party/openpi` and `{repo_root}/third_party/robosuite` entries are recorded without hashing a checkout. |
| `reproducibility.support_matrix` | Support-matrix YAML path. |
| `reproducibility.accelerator_profile` | Profile name from that support matrix. |
| `reproducibility.environment.allowlist` | Environment names children may receive; must include `HOME` and `PATH`. |
| `reproducibility.environment.required` | Required subset of the allowlist. |
| `reproducibility.environment.values` | Explicit string overrides whose keys are in the allowlist. |

Secret-like keys, undeclared `${VAR}` references, ambient `RS_*` behavior
overrides, and input-path variables other than `REAL2SIM_INPUTS_ROOT` are rejected. Paths may use
`${VAR}` and the built-ins `{repo_root}`, `{config_dir}`, `{timestamp}`,
`{robot}`, and `{inputs_root}` subject to those environment rules.

## Integrated preflight

`real2sim-doctor` checks the Python dependency graph, Git state, hash locks,
artifact digests, output permissions/free space, runtime tools, architecture,
GPU/driver, and the exact CUDA 12.8 reported by `torch.version.cuda` against
one support profile. A production preflight also checks that the current
environment still matches the root `uv.lock`, that the Gaussian CUDA extensions'
build records name the profile's compiler and toolkit, and that a kernel of each
of the three extensions runs on every GPU that the run's roles select. The run
host needs no compiler or `nvcc`. The standalone `real2sim-doctor` has no roles,
so it checks every visible GPU.

The checked-in `config/support-matrix.yaml` is the source for CUDA major/minor,
the PyTorch index, the toolkit fallback path, the minimum driver, and compiler
selection. It uses JSON syntax (valid YAML), which lets the read-only host gate
parse it with system Python before PyYAML is installed. The static uv index in
`pyproject.toml` is checked against it by the distribution tests; update both and
regenerate `uv.lock` when migrating the toolchain. Runtime callers can still
supply custom YAML matrices.

Host check and the extension build select `nvcc` in the same order: explicit
`CUDA_HOME`, then `PATH`, then the profile's `cuda.toolkit_path`. An invalid
explicit `CUDA_HOME` fails instead of silently selecting another toolkit.
Bootstrap exports the selected toolkit as `CUDA_HOME` for the extension build;
a plain `uv sync` uses torch's order, which ends at `/usr/local/cuda`.
Compiler overrides `CC` and `CXX` accept executable names or absolute paths;
they must still report the pinned major version.

The redistributable CPU profile used
in clean CI can be checked with:

```bash
real2sim-doctor --repo-root "$PWD" \
  --profile cpu-fixture \
  --artifact tiny-checkpoint-v1 \
  --artifact tiny-scene-v2 \
  --lockfile requirements-lock.ci.txt
```

For an acceptance rollout, select `cuda12-x86_64`, list every production
artifact and lock, and add `--production`. A production check rejects a dirty
or unsupported Git source, unpinned submodule, unhashed lock, fixture artifact,
missing or unavailable configured GPU slots, GPU/driver/CUDA/compiler mismatch,
or artifact mismatch. The successful production preflight report is embedded
in `run_manifest.json` alongside the resolved environment and input digests.
The `origin` remote must be an HTTPS or GitHub SSH URL.

## Commands

```bash
# Resolve schema and variables. Does not inspect runtime files.
real2sim-run \
  --config config/release/fr3/flat/a_room_cup_cup1trim.yaml \
  --dry-run

# Print the operator-facing projection as a compact vertical table.
real2sim-run \
  --config config/release/fr3/flat/a_room_cup_cup1trim.yaml \
  --dry-run --format table

# Pin the run name/timestamp for a reproducible dry-run or execution.
real2sim-run \
  --config config/release/fr3/flat/a_room_cup_cup1trim.yaml \
  --timestamp 20260823_010203 --dry-run

# Verify checkpoint, Python, scene payloads, assets, and dependency paths.
real2sim-run \
  --config config/release/fr3/flat/a_room_cup_cup1trim.yaml \
  --check-runtime

# Execute and summarize.
real2sim-run \
  --config config/release/fr3/flat/a_room_cup_cup1trim.yaml

# Use a non-default environment file.
real2sim-run \
  --env-file ../real2sim-run.env \
  --config config/release/fr3/flat/a_room_cup_cup1trim.yaml \
  --dry-run
```

`--dry-run` emits stable JSON by default. Use `--format table` when inspecting
checkpoint, scenes, controller axes, rollout sizing, runtime ports, GPU
assignments, render state, and `run_root` interactively. `--timestamp` accepts
only `YYYYMMDD_HHMMSS`; without it, the launcher uses the current time.

## Production cup/plate `.env`

The production environment check, `config/release/fr3/flat/a_room_cup_cup1trim.yaml`,
installs its inputs with
`real2sim-download-release --profile config/release/fr3/flat/a_room_cup_cup1trim.yaml`
after `.venv/bin/hf auth login` and accepting the terms of both release repositories on
Hugging Face. The installer places the scene and the
checkpoint under one root, installs the cup/plate and FR3 collision
meshes below `robosuite_env/assets/`, and writes only `REAL2SIM_INPUTS_ROOT`
as an asset path. This is the only required `.env` entry:

| Variable | Set it to |
| --- | --- |
| `REAL2SIM_INPUTS_ROOT` | The absolute input root holding `scenes/flat_a_room/` (the public A Room registered to the FR3 Flat rig; schema-v2 `background.json` whose `name` equals the tag `franka_flat`; artifact `public-scene-flat-a-room-v2`) and `checkpoints/pi05_cup/` (artifact `pi05-cup`) |

The scene artifact `public-scene-flat-a-room-v2` is a public CC BY 4.0
scene (see `assets/public_scenes/`): it is downloaded from the `scenes/`
folder of the dataset repository `airoa-org/yubi-real2sim-assets`, and its manifest
entry records the room's licence together with the Apache License 2.0 of
AIRoA's `background.json` and `table_collision/` (`CC-BY-4.0 AND Apache-2.0`).

Use an absolute path. The scene directory must contain `background.json`;
that JSON binds the point cloud, complete world-to-GS transform, and
invisible table collision. The table appearance is part of 3DGS, so schema v2
has no table visual path. The `flat_a_room` table collision is one box that the
scene itself defines. When a scene's collision lives outside its directory, the
named bundle must contain the referenced `coacd/*.obj` parts.

A schema-v2 `background.json` must also declare
`calibration_table_z_m` (with `calibration_table_z_method:
"ply-plane-fit-v1"`): the height, in MuJoCo world metres, of the 3DGS
table-top after `inv(mat_w2gs)`. Runtime `table_z`, the schema-v2 reference
plane, is pinned to `0.0` (it does not fix the height of the table collision
top; see [Success heights](#success-heights)), so the value must be within
0.05 m of zero, and every strict validation re-measures it from the point
cloud and rejects the scene when the two disagree by more than 0.05 m. A
scene whose table top is not near z = 0 therefore fails at config load
with a `SceneContractError` instead of rendering a floating table. A scene
without the field fails with `schema v2 scene must declare
calibration_table_z_m`; upgrade a separate staging copy as described in
`docs/simulation_operations.md` ("Table height check"). The public scene
archives already contain the field, and the installer never rewrites a
scene manifest.

The downloads deliberately omit the PaliGemma tokenizer. OpenPI fetches
`gs://big_vision/paligemma_tokenizer.model` anonymously on first use and caches
it below `OPENPI_DATA_HOME`; no tokenizer path belongs in `.env`.

After editing `.env`, resolve the values without executing anything, then
validate every runtime input:

```bash
real2sim-run \
  --config config/release/fr3/flat/a_room_cup_cup1trim.yaml \
  --dry-run
real2sim-run \
  --config config/release/fr3/flat/a_room_cup_cup1trim.yaml \
  --check-runtime
```

## Scenes

Each scene entry has a stable lowercase tag and one directory:

```yaml
scenes:
  - tag: franka_flat
    path: "{inputs_root}/scenes/flat_reading_room"
```

The normalized `name` in `background.json` must equal the tag. The independent
`id` identifies scene content and must be unique within a comparison. To run
five scenes ten times, list five entries, set `runner.repeats: 10`, and set
`runner.expected_rollouts: 50`.

### GS composite mode

`robosuite_env/rollout/cli.py`'s `--gs-composite-mode` controls how the MuJoCo
foreground (robot, gripper, task objects) is combined with the rendered 3DGS
background for both the third-person and policy wrist observations:

- `depth` (default) z-tests MuJoCo geometry against a per-pixel depth estimate
  of the GS scene, so GS geometry can occlude MuJoCo geometry that is behind
  it.
- `foreground` always pastes visible MuJoCo geometry over the GS background,
  regardless of relative depth (mask-style compositing).

Schema v2 exposes the same switch as the optional string
`simulation.parameters.gs_composite_mode`, which defaults to `"depth"` for
backward compatibility (mirroring the optional
`simulation.parameters.policy_action_clamp` pattern above). The launcher emits
the resolved mode explicitly, so this YAML value is authoritative for an
official run. The scene contract's fallback is consulted only by a direct
diagnostic worker invocation that omits the option.

`depth` can let a GS scene's static geometry (for example, the table
underframe baked into the Gaussian splat) depth-occlude a task object in the
policy's cropped 224x224 wrist observation even though the object is not
occluded from the third-person view. If a policy's rule-grasp trigger never
fires and the arm never approaches the object, check whether the wrist
observation shows the object incorrectly hidden behind GS geometry; setting
`gs_composite_mode: foreground` removes that occlusion by always compositing
MuJoCo geometry on top.

### GS background color transform

Schema v2 exposes the optional `simulation.parameters.gs_color_gain_rgb` and
`simulation.parameters.gs_color_gamma_rgb` lists. Each contains three finite
positive RGB values in `(0, 4]`, and both default to `[1.0, 1.0, 1.0]`.
The gamma exponents are applied per channel to decoded GS pixels in sRGB,
followed by the per-channel gain, before either GS compositing path. The
transform affects GS background pixels only; MuJoCo foreground geometry is
unchanged. The launcher always emits both explicit three-token options,
including when they resolve to the identity defaults.

Post-hoc state-log rerendering inherits these values from the recorded worker
argv, so it uses the same GS color transform as the original rollout. The
resulting `sim_summary.json` records the resolved lists in the top-level
`gs_color_gain_rgb` and `gs_color_gamma_rgb` keys.

### Wrist camera model

`fisheye_projection.FX`..`K4` describe a 640x480 calibration raster, not the
policy wrist image. The `franka-pinhole-wrist-rgb` profile renders them on a
719x539 raster and then resizes and crops each wrist differently, so each side
writes its own focal lengths and principal point: the written image follows
neither `FX`..`K4` nor a real camera unit.
`wrist_image_intrinsics(profile, side)` in
`robosuite_env/rendering/wrist_render_profiles.py` returns the KB4 model of the
written image, and `FisheyeIntrinsics.resized` maps it to a resized copy. When
the policy sees rendered wrists, `sim_summary.json` records both sides under
`policy_wrist_intrinsics` (`"model": "kb4"`), before the 224x224 policy resize.

- The model has the parameters of `cv2.fisheye`, but it takes the full angle
  `atan2(|xy|, z)` to the optical axis. The wrist images reach 100 degrees, and
  `cv2.fisheye` agrees with the model only in front of the camera (`z > 0`), so
  project with `project_points` from the same module.
- The direct renderer draws the source raster with the model's ray at each pixel
  centre. The profile crop resizes it bilinearly; `tests/unit/rendering/test_wrist_intrinsics.py`
  checks that drawn dots stay within 0.35 px of the written model.

The default `elp-mean-wrist-rgb` profile writes the board-calibration mean of the
9 ELP wrist units at 640x480 (`fx = fy = 270.47`, `cx = 320.5`,
`cy = 242.5`) for both wrists without a crop. Pixels whose ray is more than
100 degrees off the optical axis stay black, and this includes the image
corners. `simulation.parameters.policy_yubi_render_profile` selects the same
profile for policy observations and wrist video. The worker accepts
`--policy-yubi-render-profile` with the same values. Both default to
`elp-mean-wrist-rgb`; `franka-pinhole-wrist-rgb` selects the alternative
pinhole camera model.

The profile applies to every profile and command, including those that omit the
key or flag: simulated policy wrist observations and wrist video (also in runs
with `policy_image_source: lerobot` or without a policy). With
`elp-mean-wrist-rgb` the renderer's native raster is 640x480; policy images are
224x224 and wrist video insets 180x135. Recorded LeRobot observations are
unaffected.

Every run records its profile in `sim_summary.json` (`policy_yubi_render_profile`).
Rerendering a run without the flag takes the profile from that key, or from the
state log's `policy.obs_log` when the run has no summary, so the replay keeps the
recorded camera. When neither records a profile, rerendering uses
`franka-pinhole-wrist-rgb`. A summary with `policy_wrist_intrinsics` but no
profile is refused. A direct
`python -m robosuite_env.sim --state-log-replay` resolves the profile the same way
(`robosuite_env/rollout/replay/recorded_wrist.py`, with the `sim_summary.json`
beside the state log), and refuses a `--policy-yubi-render-profile` that
contradicts it.

## Parallelism and GPUs

Acquire GPU resources outside this repository and invoke `real2sim-run` from
the resulting interactive shell. Without optional
[`runner.shards`](rollout_shards.md), the launcher does not invoke a workload
scheduler; shard mode only attaches `srun` steps to job IDs supplied by the
operator and never submits allocations or discovers job IDs. If the shell sets
`CUDA_VISIBLE_DEVICES`, the runtime GPU
fields are zero-based slots in that ordered list: with
`CUDA_VISIBLE_DEVICES=GPU-a,GPU-b`, slot `0` selects `GPU-a` and slot `1`
selects `GPU-b`. Numeric and full GPU-UUID entries are supported. If the
variable is unset, the same slot values are host ordinals. Keep
`CUDA_VISIBLE_DEVICES` optional in `reproducibility.environment.allowlist`; do
not add it to `required`, because a directly attached GPU host need not set it.

Schema v2 uses `runner.policy_batch_size` as the total number of observations
passed to one policy inference. It must divide evenly across the configured
scenes; per-scene concurrency is derived and must divide `runner.repeats`.
The resolved plan also exposes the derived per-scene concurrency as
`runner.parallel_repeats` for the shell runner to consume. 3DGS workers
are assigned round-robin over
`runtime.gs_cuda_devices`. `runtime.policy_cuda_device` optionally pins the
shared OpenPI server; otherwise it uses the first 3DGS GPU.
Both release profiles use `policy_batch_size: 2`, so a one-scene evaluation
runs its ten repeats in five waves. Multi-scene evaluations divide the
configured concurrency evenly across scenes. The direct shell runner uses a
policy batch of 10 when `REPEATS=10` and no batch override is supplied; other
repeat counts run one worker per scene. Policy batches of 10 or larger always
omit live third-person video during rollout generation: both
the launcher and the direct shell runner force `--no-render`, even if
`render.enabled: true` was requested. Policy-observation rendering needed for
inference is unchanged. When canonical video was requested, the runner stops
the generation services, renders every third-person video sequentially from
the complete saved state logs, validates it, records its derived provenance,
and builds a checked-in grid before the launcher can report success:

```bash
real2sim-run --config config/<profile>.yaml
```

`render.third_person` (optional boolean, default `true`) turns this automatic
post-hoc finalization on or off: `false` exports `POSTHOC_RENDER_REQUIRED=0`,
so the batch completes without third-person videos or a grid. Batch runs use
the policy-eval template for a ten-repeat finalization with a policy batch of
10 or more
(`config/render_templates/rollout_grid_policy_eval_4x3.json`). Its ten cells
keep dedicated headers that never obscure rollout or OBS pixels, with a
bottom-right info panel for checkpoint, task, instruction, success rate, and
score statistics. Every template's panel also carries a per-frame adopted
action chunk line, read from each rollout's `logs/policy_chunks.jsonl`:

```
adopted action chunks 2/2 | chunk 1 step 5/16 | frames 16-31 | ah=32 exec=16
```

It reports how many chunks the rollouts have adopted so far out of the total
they recorded, which `chunk_index` is executing on this frame, how far into
that chunk's adopted slice the frame is, the global frame span of the slice,
and the configured action horizon and adopted step count. When the tiled
rollouts disagree at a frame, each field prints the `min~max` span instead of
a single value. Thirty- and 100-repeat finalizations use the corresponding
clear-header templates listed below. Rollout and grid MP4s are transcoded to
H.264 (yuv420p) with the pinned ffmpeg. Each transcode runs FFmpeg's decoder,
filter graph and libx264 with one thread each (`FFMPEG_THREADS` in
`robosuite_env/rollout/outputs.py`), which keeps a transcode at six threads in
total with the pinned FFmpeg 7.0.2 on any host, so its bytes do not depend on
the host's core count. One exception: in an inline-rendered run with wrist insets, the saved
policy OBS republish (`robosuite_env/tools/rewrite_saved_policy_obs_video.py`)
replaces each rollout video with an OpenCV mp4v encode, which
`FFMPEG_THREADS` does not cover. The grid is still transcoded to H.264.

The lower-level rerender tool remains available to retry an interrupted
finalization or render a diagnostic no-video run. It accepts an explicit grid
template and a third-person viewpoint override that replaces the recorded
free-camera flags in every replayed run:

```bash
.venv/bin/python -m robosuite_env.tools.rerender_third_person \
  outputs/<run_root> --gpu 0 \
  --grid-template config/render_templates/rollout_grid_policy_eval_4x3.json \
  --view-azimuth 130 --view-elevation -22 --view-distance 2.0 \
  --view-lookat "0 0 0.15"
```

The runner and tool enforce one post-hoc process per GPU and render each run's
MP4 sequentially on that GPU. They build the checked-in grid when a scene has
exactly 10, 30, or 100 runs; other per-scene counts keep their individual
videos and skip the grid. The default `--wrist-insets saved` mode requires the
complete recorded policy wrist sequence and fails before rendering if its
frame, side, source geometry, or image contract is inconsistent. There is no
fallback to a newly rendered wrist stream; `--wrist-insets off` is the only
alternative. The saved 224x224 policy tensors remain unchanged. Video insets
remove only the exact OpenPI letterbox rectangle computed from recorded camera
geometry, then resize with the original aspect ratio. After encoding, every
left and right ROI is compared with its scheduled source on every frame,
including chunk boundaries. The default `--saved-wrist-display raw224`
preserves the complete 224x224 policy tensor in the video inset, including its
real black letterbox. Select `aspect-fit` only when a diagnostic video should
remove that recorded letterbox.

Separating these phases preserves policy and simulator throughput at a policy
batch of 10 or more, removes live third-person composition and video encoding
from the inference wave, and reduces EGL/3DGS GPU contention that can produce
black, striped, or corrupted frames. Policy batches of 1 through 9 keep their
configured render behavior.
`runtime.sim_cuda_devices` explicitly assigns MuJoCo EGL workers and defaults
to the 3DGS list when omitted. The runner maps workers round-robin over this
list and records both the selected visible slot and resolved CUDA/EGL binding
in `RUN_COMMAND.md`.

`simulation.parameters.grasp_mode` accepts `physics` or `rule` and is a
required typed parameter, as is `policy_wrist_contrast`. The runner emits
exactly one `--grasp-mode` selector derived from this value.
With `physics`, the policy angle drives the dynamic CbPC finger controller,
and the object is retained only by MuJoCo contacts and friction. `rule`
retains the explicit same-object, both-inner-pad contact-angle latch. Every
release profile uses `rule`.

## Grid inputs must embed the exact saved policy OBS

A rollout grid is evidence of what the policy saw, so the wrist OBS insets
burned into every rollout video it tiles must be the exact saved 224x224
policy observations from `RUN/logs/policy_obs`, scaled as a complete canvas to
the 180x135 slot so the real OpenPI letterbox stays visible. Re-rendering the
wrist camera for presentation, or cropping the letterbox off the policy
tensor, produces pixels the policy never received and is forbidden.

`make_rollout_grid_video.py` enforces this fail-closed before it encodes
anything. For each input it reads the summary of the *published* video —
`RUN/saved_policy_obs/sim_summary.json`, else `RUN/rerender/sim_summary.json`,
else `RUN/sim_summary.json` — and requires either `video_wrist_insets_enabled:
false` (no OBS pixels at all) or the exact-saved contract:

```
video_wrist_inset_source     = "saved"
video_wrist_display_contract = "raw_policy_tensor_full_canvas_resize_v1"
video_wrist_inset_layout     = {"version": 3, ...}
```

It then re-decodes the video and compares both ROIs against the saved PNGs
with `validate_saved_wrist_insets`, recording the result under
`saved_policy_obs_validation` in the grid JSON.

The post-hoc renderer already publishes this contract. A run that rendered
inline burns a presentation crop instead, so `run_policy_batch.sh` republishes
each rollout video from its own saved OBS before the grid. In both cases the
runner stops the policy and 3DGS servers after the last rollout, before any
of these finalization tools runs:

```bash
.venv/bin/python -m robosuite_env.tools.rewrite_saved_policy_obs_video \
  RUN_ROOT --scene SCENE_TAG
```

For an inline-rendered run this never edits the generation `sim_summary.json`.
It republishes `SCENE_TAG_run_NN.mp4` and describes the result in a new
`RUN/saved_policy_obs/` sidecar (`sim_summary.json`, `video_validation.json`,
`provenance.json`), which records the generation source under
`video_wrist_inset_source_generated` and the superseded video under
`raw224_policy_obs_overlay.base_video_sha256`. `RUN/sim_summary.json` and
`RUN/video_validation.json` keep describing the pre-overlay generation video,
so read the sidecar for anything about the published one. The command is
idempotent.

## Template-only grid rendering

The count-based defaults used by manual rerendering and live-render batches
support 10, 30, or 100 repeats per scene. Ten repeats use
`rollout_grid_clear_headers_4x3.json`; 30 repeats use
`rollout_grid_clear_headers_6x6.json`; 100-repeat runs use
`rollout_grid_clear_headers_10x12.json`. Multi-scene comparisons create one
grid per scene so a 100-by-100 two-scene comparison produces two clear
100-rollout grids. Template path and SHA-256 are recorded in each grid JSON,
and missing/corrupt required grids make completion validation fail. The
official ten-repeat post-hoc path with a policy batch of 10 or more overrides the count-based
4x3 default with `rollout_grid_policy_eval_4x3.json`, as described above.

Manual regeneration uses the same mandatory path:

```bash
.venv/bin/python -m robosuite_env.tools.make_rollout_grid_video RUN_ROOT \
  --output RUN_ROOT/SCENE_TAG/rollout_grid_10.mp4 \
  --scene SCENE_TAG \
  --title SCENE_TAG \
  --template config/render_templates/rollout_grid_clear_headers_4x3.json \
  --expected-frames 300
```

For a single-scene run, write the output directly below `RUN_ROOT` instead.
The renderer also defaults the title to `SCENE_TAG` when `--title` is omitted.

Manual 30-rollout regeneration uses the same command with
`rollout_grid_clear_headers_6x6.json` and an output named
`RUN_ROOT/rollout_grid_30.mp4` for a single-scene run.

There is no free-form layout renderer. Options such as `--columns`,
`--cell-width`, `--source-crop`, `--fps`, and `--poster-frame` are rejected.
A new layout requires a versioned JSON template passed with `--template`.
Rendered schema-v2 profiles fail during config loading when their per-scene
repeat count has no checked-in template. Manual 100-rollout regeneration uses
the same command with `rollout_grid_clear_headers_10x12.json` and an output
named `RUN_ROOT/SCENE_TAG/rollout_grid_100.mp4` (or directly under `RUN_ROOT`
for a single scene).

## FR3 inverse kinematics

FR3 `joint_ik` is fully in-process and does not require ROS. It uses MuJoCo's
Jacobian with damped least squares for the 6D TCP task and selects the redundant
joint solution with a projected canonical-home objective:

`Δq = J⁺ e + k N (q_home − q)`

The task inverse and the structural projector `N` come from the same SVD, and
feasible posture updates are backtracked if they would move the TCP outside its
position or orientation tolerance. In dynamic mode the selected joint goal is
then passed through the existing stateful Ruckig velocity, acceleration, and
jerk limits. In kinematic mode it is pinned directly. The gain is controlled by
`--joint-ik-nullspace-gain` (default `0.2`). Whether the posture objective is
applied at all is a separate, explicit switch: `--fr3-nullspace home|off`
(YAML `simulation.fr3_nullspace`). `off` solves the pure whole-pose IK with an
effective gain of 0 regardless of the configured gain; `home` with a zero gain
is rejected as a silent no-op. `sim_summary.json` records `fr3_ik_nullspace`,
the effective `fr3_ik_nullspace_gain`, and `fr3_ik_nullspace_gain_configured`,
and `arm_controller` reads `fr3_absolute_joint_ik` when the null space is off.

FR3 policy deltas retain the deploy translation and rotation clamp by default.
Use `--no-policy-action-clamp` for an unclamped ablation; raw deltas still must
be finite and contain nonzero quaternions. Schema v2 exposes the same switch as
the optional boolean `simulation.parameters.policy_action_clamp`, which defaults
to `true`. Every FR3 policy summary records the
effective `policy_action_clamp` value.

Policies that declare a training cadence use `--policy-action-hz`
rather than coupling the policy rate to `--frame-hz`. For example, a 30 Hz policy on the
validated 50 Hz FR3 controller receives a deterministic `2,1,2` repeating
controller-tick schedule. Cumulative-nearest, half-up boundary quantization
keeps the phase error below 10 ms while preserving exactly 30 policy rows and
50 complete controller ticks per simulated second.

`runner.frames` counts policy actions. Therefore 300 actions at 30 Hz are a
10-second evaluation, and `exec_steps: 16` replans every 16 / 30 ≈ 0.533
seconds. One encoded frame represents one source action, so playback remains
30 fps; do not encode all 30 Hz states at 10 fps because that silently makes
the rollout three times slower.

## Adding a task

`task.objects` selects the task. Two values are built in.

| `task.objects` | objects | goal / score condition | what else the profile must set |
|---|---|---|---|
| `cup_plate` | cup, plate | `cup_on_plate` | optionally a positions JSON with `cup_xy` / `plate_xy` |
| `box_pack` | box, tape, pen, LAN cable | `object_in_box` / `object_in_box_containment` | `box_pack_task_index` (0-5, the runnable into-box prompts; registered out-of-box prompts 6-11 are refused until their start layout and direction are supported); `box_pack_layout_json` off the Duo scene |

Anything else is read as a path to a cup/plate positions JSON. No release
profile uses `box_pack` in this release, and the release downloads do not
include its checkpoints or object meshes. For the into-box prompts, success
latches at the first output frame where the object lies in the box cavity and
no hand holds it; later frames do not change it
(`robosuite_env/tasks/success/box_pack.py`).

Every one of them pins its instruction. The worker refuses to start when
`task.instruction` does not match what the task's own contract says the
selected index is called, so a profile cannot quietly prompt a
policy with words it was not trained on. For `box_pack` the index fixes the
object *and* the hand together.

Success identifiers are validated by the simulator's success-criteria
registry. Keep the instruction selected for a rollout and its success
condition in YAML, not code. Dataset-grounded multi-stage presets additionally
pin their exact source instruction mapping in the task contract, and the worker
refuses to start when `task.instruction` does not match the selected stage.

### Where the objects start

Only `cup_plate` has a layout measured from a capture. The box
layouts are hand-authored per scene and rig; the one in this repository is the
task default `robosuite_env/obj_calib/layout_box_pack_duo_manual.json` for the
Duo rig. A profile selects one as described in
[Task layout selection](#task-layout-selection). They follow
an honesty rule: they must name who placed
them, with what, when and why, and they must not carry capture evidence.

An authored z does not decide where an object starts. Layouts write z against
a flat plane at z = 0, but the compiled table is not that plane: the Duo slab's
top is +34 mm, and the parts of a CoACD-decomposed table differ by millimetres. At every reset
that is not a deterministic restore, `CupPlateLiftGW._reset_internal`
(`robosuite_env/tasks/envs/cup_plate_registered.py`) seats each object placed
from an exact pose (box-packing objects) and
the cup/plate pair. It translates the object vertically until its collision
hulls touch the upper surface of the table collision (`gw_desk_coll_*`), leaving
XY and orientation as authored. Each object is seated directly on the table;
an object whose XY is not over the table is refused. The lift is exact for
convex hulls: `seat_lift` in
`robosuite_env/tasks/layouts.py` solves a linear program per hull pair, poses it
only for pairs whose footprints overlap under a 2D separating-axis test, and
accepts only an optimal solution. The result (`env.layout_seating_report`,
printed as `[layout] seated ...`) is part of the init-state identity. During the
ready drive, the box-packing targets are
held only in x, y, and heading so they stay on the table; cup/plate keeps
the full-pose hold.

### Success heights

`object_in_box` measures heights from the
table collision compiled into the model, not from the schema-v2 reference plane
(`TableSurface` and `TableDatum` in `robosuite_env/tasks/success/_geometry.py`).
The datum resolves in this order:

1. an explicit `table_top_z_m` criterion option, a flat table everywhere (no
   profile key sets it);
2. the upper envelope of the table collision hulls: the highest surface under a
   5 x 5 grid over the box's collision footprint, latched at the
   first evaluation;
3. the table's highest collision vertex, when the footprint is not over the
   table.

The `table_z_m` option the runtime always passes (0.0) is only reported, as
`reference_plane_z_m`. Criterion descriptions publish `table_top_z_m` and `table_datum`
(value, source, reference plane, table geoms), and per-frame metrics carry
`table_top_z_mm` (box). `sim_summary.json` adds the
table-relative `task_lift_mm` and `task_lift`; `cup_lift_mm` stays, described by
`cup_lift_mm_definition`.

`object_in_box` finds each
hand in the body tree: the wrist is the body of the rig profile's last arm joint,
the hand is the wrist's child that leads to both fingers, and the hand's geoms
are the collision geoms below it. Construction fails if no rig profile fits the
model, the fitting profiles disagree, a hand has no geoms, or the two hands share
one.
