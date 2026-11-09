#!/usr/bin/env bash
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
# Run FR3 YUBI policy rollouts for the configured 3DGS scenes.
#
# One OpenPI policy server is shared by all simultaneously running sim workers.
# Each wave launches PARALLEL_REPEATS workers per GS scene. POLICY_BATCH_SIZE
# is the total number of policy inputs across all scenes in that wave. Workers for
# the same scene and GS GPU also share one resident GS server.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"

REAL2SIM_STRICT_LAUNCH="${REAL2SIM_STRICT_LAUNCH:-0}"
PY="${PY:-${ROOT}/.venv/bin/python}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${ROOT}/runtime-cache/openpi}"
export OPENPI_DATA_HOME
CKPT="${CKPT:-}"
ROBOT="${ROBOT:-FR3}"
CONFIG="${CONFIG:-}"
CONFIG_SRC="${CONFIG_SRC:-deploy}"
OBJECTS="${OBJECTS:-cup_plate}"
INSTRUCTION="${INSTRUCTION:-Pick up the cup with your right hand and set it on the plate}"
POLICY_IMAGE_EPISODE="${POLICY_IMAGE_EPISODE:-}"
GOAL_CONDITION="${GOAL_CONDITION:-cup_on_plate}"
SCORE_CONDITION="${SCORE_CONDITION:-cup_on_plate_contact_points}"
# Schema v2 gets table appearance from 3DGS and has no repository desk texture.
WHITE_DESK_TEX="${WHITE_DESK_TEX-}"
EXPORTS_ROOT="${EXPORTS_ROOT:-${ROOT}/outputs}"
GAUSSIAN_WRAPPING_REPO="${GAUSSIAN_WRAPPING_REPO:-${ROOT}/third_party/GaussianWrapping}"
GS_SCENE_DIRS="${GS_SCENE_DIRS:-}"
RUN_ROOT="${RUN_ROOT:-}"
REPEATS="${REPEATS:-10}"
REPEAT_START="${REPEAT_START:-1}"
REPEAT_STOP_DEFAULTED=0
if [ -z "${REPEAT_STOP+x}" ] || [ -z "${REPEAT_STOP}" ]; then
  REPEAT_STOP_DEFAULTED=1
fi
REPEAT_STOP="${REPEAT_STOP:-${REPEATS}}"
PARALLEL_REPEATS="${PARALLEL_REPEATS:-}"
POLICY_BATCH_SIZE="${POLICY_BATCH_SIZE:-}"
POSTHOC_RENDER_BATCH_THRESHOLD=10
POSTHOC_RENDER_REQUIRED="${POSTHOC_RENDER_REQUIRED:-auto}"
FRAMES="${FRAMES:-300}"
MAX_WALL_SEC="${MAX_WALL_SEC:-3600}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8765}"
GS_PORT_BASE="${GS_PORT_BASE:-8321}"
GS_CUDA_DEVICES="${GS_CUDA_DEVICES:-}"
# How many GS servers to start on each GPU. One server holds its own copy of
# the scene (about 10 GB for a 14 M gaussian cloud) and renders for whichever
# workers are assigned to it, so this trades GPU memory for concurrency.
GS_SERVERS_PER_GPU="${GS_SERVERS_PER_GPU:-1}"
POLICY_CUDA_DEVICE="${POLICY_CUDA_DEVICE:-}"
SIM_CUDA_DEVICES="${SIM_CUDA_DEVICES:-}"
GS_START_TIMEOUT_SEC="${GS_START_TIMEOUT_SEC:-600}"
GS_BG_COLOR="${GS_BG_COLOR:-0 0 0}"
RENDER_LOCK_ROOT="${RENDER_LOCK_ROOT:-/tmp/real2sim-render-locks-${UID}}"
GS_NUM_THREADS="${GS_NUM_THREADS:-1}"
GRIP_TORQUE_MULT="${GRIP_TORQUE_MULT:-0.5}"
GRASP_MODE="${GRASP_MODE:-rule}"
GRIPPER_CALIBRATION="${GRIPPER_CALIBRATION:-}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
EXEC_STEPS="${EXEC_STEPS:-16}"
RF_OVERLAP="${RF_OVERLAP:-0}"
RF_OVERLAP_SAMPLER="${RF_OVERLAP_SAMPLER:-single}"
RF_OVERLAP_STEP_SHIFT="${RF_OVERLAP_STEP_SHIFT:-0}"
RF_OVERLAP_PREV_WEIGHT="${RF_OVERLAP_PREV_WEIGHT:-}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
CLEAN_VIDEO="${CLEAN_VIDEO:-0}"
GRID_TEMPLATE="${GRID_TEMPLATE:-}"
VIEW_AZIMUTH="${VIEW_AZIMUTH:-270}"
VIEW_ELEVATION="${VIEW_ELEVATION:--20}"
VIEW_DISTANCE="${VIEW_DISTANCE:-1.35}"
VIEW_LOOKAT="${VIEW_LOOKAT:-0.08 0.18 0.10}"
RENDER_WRIST="${RENDER_WRIST:-0}"
WRIST_GS="${WRIST_GS:-0}"
SHOW_ORIGIN="${SHOW_ORIGIN:-0}"
EXTRA_SIM_ARGS="${EXTRA_SIM_ARGS:-}"
EXTRA_SIM_ARGS_JSON="${EXTRA_SIM_ARGS_JSON:-}"
BOX_PACK_LAYOUT_JSON_LIST="${BOX_PACK_LAYOUT_JSON_LIST:-}"
DETERMINISTIC_SIM="${DETERMINISTIC_SIM:-1}"
DETERMINISTIC_SEED="${DETERMINISTIC_SEED:-0}"
INIT_STATE_JSON="${INIT_STATE_JSON:-}"
INIT_STATE_ROOT="${INIT_STATE_ROOT:-}"
# Bump this whenever the semantic contract of a post-ready state changes.
# v4 adds atomic publication, provenance, and compiled-model validation to the
# physical cup / plate settle contract. Keeping the schema in the cache key
# prevents a legacy zero-velocity-but-free-falling state from being reused.
INIT_STATE_CACHE_SCHEMA="${INIT_STATE_CACHE_SCHEMA:-initv5_named_task_object_settle}"
MIMICGEN_INIT_DATASET_ROOT="${MIMICGEN_INIT_DATASET_ROOT:-}"
MIMICGEN_INIT_STATE_LOG_ROOT="${MIMICGEN_INIT_STATE_LOG_ROOT:-}"
MIMICGEN_INIT_EPISODE_OFFSET="${MIMICGEN_INIT_EPISODE_OFFSET:-0}"
POLICY_NOISE_SEED="${POLICY_NOISE_SEED:-}"
POLICY_NOISE_BANK_DIR="${POLICY_NOISE_BANK_DIR:-}"
POLICY_INFERENCE_MANIFEST="${POLICY_INFERENCE_MANIFEST:-}"
POLICY_DETERMINISTIC_XLA="${POLICY_DETERMINISTIC_XLA:-0}"
BATCH_CLIENT_ORDER="${BATCH_CLIENT_ORDER:-}"
SIM_NUM_THREADS="${SIM_NUM_THREADS:-}"
REAL2SIM_OFFICIAL_RUN_MANIFEST="${REAL2SIM_OFFICIAL_RUN_MANIFEST:-}"
REAL2SIM_RUNTIME_SOURCE_MANIFEST="${REAL2SIM_RUNTIME_SOURCE_MANIFEST:-}"
REAL2SIM_RUNTIME_SOURCE_UNSEALED="${REAL2SIM_RUNTIME_SOURCE_UNSEALED:-}"
REAL2SIM_RUNTIME_SOURCE_MANIFEST_SHA256="${REAL2SIM_RUNTIME_SOURCE_MANIFEST_SHA256:-}"
REAL2SIM_GPU_BINDINGS_JSON="${REAL2SIM_GPU_BINDINGS_JSON:-}"
REAL2SIM_SHARD_NAME="${REAL2SIM_SHARD_NAME:-}"
REAL2SIM_AGGREGATE_ONLY="${REAL2SIM_AGGREGATE_ONLY:-0}"
DRY_RUN=0
CLI_ARGUMENT_COUNT="$#"

usage() {
  cat <<'EOF'
Usage: scripts/run_policy_batch.sh [options]

Options:
  --run-root DIR       Output root directory.
  --scene-dir DIR      Use one GS background directory or background.json. Can be repeated.
                       Alias: --scene
  --checkpoint PATH    OpenPI checkpoint directory. Required unless CKPT is set. Alias: --ckpt
  --robot NAME         Simulation robot: FR3 or FR3_DUO. Default: FR3
  --config NAME        OpenPI config name matching --checkpoint. Required unless CONFIG is set.
  --config-src SOURCE  OpenPI config source: deploy or training. Default: deploy
  --objects SPEC       sim.py object preset or positions JSON. Default: cup_plate
  --instruction TEXT   Policy language instruction. Default: cup-on-plate prompt
  --policy-image-episode PATH
                       LeRobot parquet used for deploy-ready pose / optional real observations.
  --goal-condition ID  Rollout success condition. Default: cup_on_plate
  --score-condition ID Rollout score condition. Default: cup_on_plate_contact_points
  --repeats N          Repeats per GS scene. Default: 10
  --repeat-start N     First absolute repeat number to execute. Default: 1
  --repeat-stop N      Last absolute repeat number to execute (inclusive). Default: REPEATS
  --parallel-repeats N Repeats per scene launched in each policy batch wave.
  --policy-batch-size N Total policy inputs per inference wave. Default: 10
                       when repeats=10; must divide evenly across scenes.
  --frames N           Source/policy frames. Default: 300
  --action-horizon N   Policy action horizon for the server and sim. Default: 32. Alias: --ah
  --exec-steps N       Actions executed per policy chunk. Default: 16. Alias: --exec
  --rf-overlap N       RF action-prefix overlap. Default: 0 (legacy).
                       Requires overlap <= exec. single needs AH >= exec + overlap;
                       paired/paired_mean/paired_weighted need AH >= max(exec, 2 * overlap).
  --rf-overlap-sampler MODE
                       RF overlap sampler: single, paired (hard replace), paired_mean (50/50), or paired_weighted. Default: single
  --rf-overlap-prev-weight W
                       Frame-0 previous-OBS contribution for paired_weighted; it linearly decays by prefix
                       frame distance and current uses the complement. Range [0,1]. Default: 0.5.
  --rf-overlap-step-shift N
                       Initial paired/paired_mean/paired_weighted ODE blend delay; must be less than policy num_steps. Default: 0
  --max-wall-sec N     Per-worker sim loop wall limit. Default: 3600
  --server-port N      Policy batch server TCP port. Default: 8765
  --gs-port-base N     First GS server port to try. Default: 8321
  --render-lock-root DIR
                       Node-local GPU render-lock directory shared across runner invocations.
  --with-wrist         Render wrist-view insets into the output mp4. Default: off
  --no-wrist           Do not render wrist-view insets.
  --wrist-gs           Composite GS into displayed wrist insets.
  --no-wrist-gs        Do not force GS in displayed wrist insets.
  --show-origin        Render world origin marker.
  --no-origin          Hide world origin marker. Default.
  --no-analysis        Skip gripper and trajectory post-processing.
  --dry-run            Print commands only.
  -h, --help           Show help.

Supported public entry point:
  real2sim-run \
    --config config/release/fr3/flat/a_room_cup_cup1trim.yaml

This script is internal current-host process orchestration; it never obtains
GPU resources. Direct developer calls must run in an already provisioned
interactive shell and explicitly set GS_CUDA_DEVICES, POLICY_CUDA_DEVICE, and
SIM_CUDA_DEVICES to the intended current-host visible slots.

Environment overrides:
  PY, OPENPI_DATA_HOME,
  CKPT, ROBOT (FR3), CONFIG, CONFIG_SRC,
  OBJECTS, INSTRUCTION, POLICY_IMAGE_EPISODE, GOAL_CONDITION, SCORE_CONDITION,
  WHITE_DESK_TEX, EXPORTS_ROOT,
  GS_SCENE_DIRS, RUN_ROOT, REPEATS, REPEAT_START, REPEAT_STOP,
  PARALLEL_REPEATS, POLICY_BATCH_SIZE,
  FRAMES, MAX_WALL_SEC, SERVER_HOST, SERVER_PORT, GS_PORT_BASE, GS_CUDA_DEVICES,
  GS_SERVERS_PER_GPU,
  POLICY_CUDA_DEVICE,
  SIM_CUDA_DEVICES,
  RENDER_LOCK_ROOT (node-local absolute directory shared by all invocations from this user),
  GAUSSIAN_WRAPPING_REPO, GRIP_TORQUE_MULT (default 0.5),
  GRASP_MODE (physics or rule; default rule), GRIPPER_CALIBRATION,
  RUN_ANALYSIS,
  CLEAN_VIDEO (default 0),
  GRID_TEMPLATE (required template for automatic 10/30/100-rollout grid rendering),
  ACTION_HORIZON (default 32), EXEC_STEPS (default 16), RF_OVERLAP (default 0),
  RF_OVERLAP_SAMPLER (default single), RF_OVERLAP_PREV_WEIGHT (paired_weighted default 0.5), RF_OVERLAP_STEP_SHIFT (default 0),
  VIEW_AZIMUTH, VIEW_ELEVATION, VIEW_DISTANCE, VIEW_LOOKAT,
  RENDER_WRIST, WRIST_GS, SHOW_ORIGIN,
  DETERMINISTIC_SIM (default 1), DETERMINISTIC_SEED (default 0),
  INIT_STATE_JSON, INIT_STATE_ROOT, POLICY_NOISE_SEED, POLICY_NOISE_BANK_DIR,
  MIMICGEN_INIT_DATASET_ROOT, MIMICGEN_INIT_STATE_LOG_ROOT,
  MIMICGEN_INIT_EPISODE_OFFSET (default 0),
  POLICY_INFERENCE_MANIFEST, POLICY_DETERMINISTIC_XLA (default 0),
  PORTABLE_POLICY_SERVER (retired; a nonempty value is refused),
  BATCH_CLIENT_ORDER,
  SIM_NUM_THREADS (default 1 in deterministic mode, otherwise 4; OMP/MKL/OpenBLAS/OpenCV threads
    per worker, per worker helper and for the policy server),
  GS_NUM_THREADS (default 1; OMP/MKL/OpenBLAS/NumExpr threads per GS server),
  POSTHOC_RENDER_REQUIRED (auto, 0, or launcher-owned 1),
  EXTRA_SIM_ARGS, EXTRA_SIM_ARGS_JSON (JSON array; mutually exclusive),
  BOX_PACK_LAYOUT_JSON_LIST (launcher-owned JSON array; empty in scalar mode),
  REAL2SIM_STRICT_LAUNCH (launcher-owned; disables all legacy simulation defaults)
  REAL2SIM_OFFICIAL_RUN_MANIFEST (launcher-written, required for official runs)
  REAL2SIM_RUNTIME_SOURCE_MANIFEST and REAL2SIM_RUNTIME_SOURCE_MANIFEST_SHA256
    (launcher-written fail-closed source/config generation seal)
EOF
}

SCENE_DIR_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --scene-dir|--scene) SCENE_DIR_ARGS+=("$2"); shift 2 ;;
    --checkpoint|--ckpt) CKPT="$2"; shift 2 ;;
    --robot) ROBOT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --config-src) CONFIG_SRC="$2"; shift 2 ;;
    --objects) OBJECTS="$2"; shift 2 ;;
    --instruction) INSTRUCTION="$2"; shift 2 ;;
    --policy-image-episode) POLICY_IMAGE_EPISODE="$2"; shift 2 ;;
    --goal-condition) GOAL_CONDITION="$2"; shift 2 ;;
    --score-condition) SCORE_CONDITION="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --repeat-start) REPEAT_START="$2"; shift 2 ;;
    --repeat-stop) REPEAT_STOP="$2"; REPEAT_STOP_DEFAULTED=0; shift 2 ;;
    --parallel-repeats) PARALLEL_REPEATS="$2"; shift 2 ;;
    --policy-batch-size) POLICY_BATCH_SIZE="$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --action-horizon|--ah) ACTION_HORIZON="$2"; shift 2 ;;
    --exec-steps|--exec) EXEC_STEPS="$2"; shift 2 ;;
    --rf-overlap) RF_OVERLAP="$2"; shift 2 ;;
    --rf-overlap-sampler) RF_OVERLAP_SAMPLER="$2"; shift 2 ;;
    --rf-overlap-prev-weight) RF_OVERLAP_PREV_WEIGHT="$2"; shift 2 ;;
    --rf-overlap-step-shift) RF_OVERLAP_STEP_SHIFT="$2"; shift 2 ;;
    --max-wall-sec) MAX_WALL_SEC="$2"; shift 2 ;;
    --server-port) SERVER_PORT="$2"; shift 2 ;;
    --gs-port-base) GS_PORT_BASE="$2"; shift 2 ;;
    --gs-servers-per-gpu) GS_SERVERS_PER_GPU="$2"; shift 2 ;;
    --render-lock-root) RENDER_LOCK_ROOT="$2"; shift 2 ;;
    --with-wrist) RENDER_WRIST=1; shift ;;
    --no-wrist) RENDER_WRIST=0; shift ;;
    --wrist-gs) WRIST_GS=1; shift ;;
    --no-wrist-gs) WRIST_GS=0; shift ;;
    --show-origin) SHOW_ORIGIN=1; shift ;;
    --no-origin) SHOW_ORIGIN=0; shift ;;
    --no-analysis) RUN_ANALYSIS=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[real2sim-batch] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# The portable WebSocket policy bridge was removed. A caller that still exports
# its endpoint asked for a remote policy; starting the local checkpoint instead
# would report another policy's results, so refuse before anything starts.
# real2sim-run refuses the same variable when it loads the profile.
if [ -n "${PORTABLE_POLICY_SERVER:-}" ]; then
  echo "[real2sim-batch] PORTABLE_POLICY_SERVER is retired: the portable WebSocket policy bridge was removed, and this run would start the local checkpoint (CKPT/--checkpoint) instead of the requested remote policy; unset it" >&2
  exit 2
fi

if [ "${REPEAT_STOP_DEFAULTED}" = "1" ]; then
  REPEAT_STOP="${REPEATS}"
fi

# Mirrors SUPPORTED_ROBOT_NAMES in robosuite_env/robots/profiles.py, spelled
# in the launcher's uppercase display form.
case "${ROBOT}" in
  FR3|FR3_DUO) ;;
  *) echo "[real2sim-batch] ROBOT/--robot must be FR3 or FR3_DUO, got ${ROBOT}" >&2; exit 2 ;;
esac
ROBOT_TAG="${ROBOT,,}"

if [ -n "${MIMICGEN_INIT_DATASET_ROOT}" ] || [ -n "${MIMICGEN_INIT_STATE_LOG_ROOT}" ]; then
  if [ -z "${MIMICGEN_INIT_DATASET_ROOT}" ] || [ -z "${MIMICGEN_INIT_STATE_LOG_ROOT}" ]; then
    echo "[real2sim-batch] MIMICGEN_INIT_DATASET_ROOT and MIMICGEN_INIT_STATE_LOG_ROOT must be set together" >&2
    exit 2
  fi
  [[ "${MIMICGEN_INIT_EPISODE_OFFSET}" =~ ^[0-9]+$ ]] || {
    echo "[real2sim-batch] MIMICGEN_INIT_EPISODE_OFFSET must be non-negative, got ${MIMICGEN_INIT_EPISODE_OFFSET}" >&2
    exit 2
  }
  [ -d "${MIMICGEN_INIT_DATASET_ROOT}/data" ] || {
    echo "[real2sim-batch] MimicGen dataset data directory not found: ${MIMICGEN_INIT_DATASET_ROOT}/data" >&2
    exit 2
  }
  [ -d "${MIMICGEN_INIT_STATE_LOG_ROOT}" ] || {
    echo "[real2sim-batch] MimicGen state-log root not found: ${MIMICGEN_INIT_STATE_LOG_ROOT}" >&2
    exit 2
  }
fi

EXTRA_SIM_ARGS_ARR=()
if [ -n "${EXTRA_SIM_ARGS}" ] && [ -n "${EXTRA_SIM_ARGS_JSON}" ]; then
  echo "[real2sim-batch] set only one of EXTRA_SIM_ARGS or EXTRA_SIM_ARGS_JSON" >&2
  exit 2
fi
if [ -n "${EXTRA_SIM_ARGS_JSON}" ]; then
  extra_sim_args_lines="$(
    "${PY}" - "${EXTRA_SIM_ARGS_JSON}" <<'PY'
import json
import sys

try:
    value = json.loads(sys.argv[1])
except json.JSONDecodeError as exc:
    raise SystemExit(f"EXTRA_SIM_ARGS_JSON is not valid JSON: {exc}") from exc
if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
    raise SystemExit("EXTRA_SIM_ARGS_JSON must be a JSON array of strings")
if any(not item or "\0" in item or "\n" in item or "\r" in item for item in value):
    raise SystemExit("EXTRA_SIM_ARGS_JSON items must be non-empty and contain no NUL/newline")
for item in value:
    print(item)
PY
  )" || exit $?
  if [ -n "${extra_sim_args_lines}" ]; then
    mapfile -t EXTRA_SIM_ARGS_ARR <<< "${extra_sim_args_lines}"
  fi
elif [ -n "${EXTRA_SIM_ARGS}" ]; then
  read -r -a EXTRA_SIM_ARGS_ARR <<< "${EXTRA_SIM_ARGS}"
fi
BOX_PACK_LAYOUTS=()
if [ -n "${BOX_PACK_LAYOUT_JSON_LIST}" ]; then
  box_pack_layout_lines="$(
    "${PY}" - "${BOX_PACK_LAYOUT_JSON_LIST}" "${REPEATS}" <<'PY'
import json
import sys

try:
    value = json.loads(sys.argv[1])
except json.JSONDecodeError as exc:
    raise SystemExit(f"BOX_PACK_LAYOUT_JSON_LIST is not valid JSON: {exc}") from exc
if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
    raise SystemExit("BOX_PACK_LAYOUT_JSON_LIST must be a non-empty JSON array of strings")
if len(value) != int(sys.argv[2]):
    raise SystemExit(
        f"BOX_PACK_LAYOUT_JSON_LIST length must equal REPEATS ({sys.argv[2]}); got {len(value)}"
    )
if any(not item or "\0" in item or "\n" in item or "\r" in item for item in value):
    raise SystemExit("BOX_PACK_LAYOUT_JSON_LIST items must be non-empty and contain no NUL/newline")
for item in value:
    print(item)
PY
  )" || exit $?
  mapfile -t BOX_PACK_LAYOUTS <<< "${box_pack_layout_lines}"
  if [ "${OBJECTS}" != "box_pack" ]; then
    echo "[real2sim-batch] BOX_PACK_LAYOUT_JSON_LIST requires OBJECTS=box_pack" >&2
    exit 2
  fi
  if [ -n "${INIT_STATE_JSON}" ]; then
    echo "[real2sim-batch] BOX_PACK_LAYOUT_JSON_LIST requires an unset INIT_STATE_JSON" >&2
    exit 2
  fi
fi
for extra_arg in "${EXTRA_SIM_ARGS_ARR[@]}"; do
  case "${extra_arg}" in
    --robot|--robot=*|--franka|--franka=*)
      echo "[real2sim-batch] set the simulation robot with ROBOT/--robot; do not pass ${extra_arg} in EXTRA_SIM_ARGS_JSON/EXTRA_SIM_ARGS" >&2
      exit 2
      ;;
    --ah|--ah=*|--action-horizon|--action-horizon=*|--exec|--exec=*|--exec-steps|--exec-steps=*|--adopted-action-chunks|--adopted-action-chunks=*|--rf-overlap|--rf-overlap=*|--action-overlap|--action-overlap=*|--rf-overlap-sampler|--rf-overlap-sampler=*|--rf-overlap-prev-weight|--rf-overlap-prev-weight=*|--rf-overlap-step-shift|--rf-overlap-step-shift=*)
      echo "[real2sim-batch] set policy timing with ACTION_HORIZON/--action-horizon, EXEC_STEPS/--exec-steps, RF_OVERLAP/--rf-overlap, RF_OVERLAP_SAMPLER/--rf-overlap-sampler, RF_OVERLAP_PREV_WEIGHT/--rf-overlap-prev-weight, and RF_OVERLAP_STEP_SHIFT/--rf-overlap-step-shift; do not pass ${extra_arg} in EXTRA_SIM_ARGS_JSON/EXTRA_SIM_ARGS" >&2
      exit 2
      ;;
    --grasp-mode|--grasp-mode=*)
      echo "[real2sim-batch] set grasp semantics with GRASP_MODE; do not pass ${extra_arg} in EXTRA_SIM_ARGS_JSON/EXTRA_SIM_ARGS" >&2
      exit 2
      ;;
    --render-lock|--render-lock=*)
      echo "[real2sim-batch] render locking is GPU-scoped and runner-owned; do not pass ${extra_arg} in EXTRA_SIM_ARGS_JSON/EXTRA_SIM_ARGS" >&2
      exit 2
      ;;
    --strict-assets|--strict-assets=*|--official-run-manifest|--official-run-manifest=*|--official-worker-contract|--official-worker-contract=*)
      echo "[real2sim-batch] official execution classification is launcher-owned; do not pass ${extra_arg} in EXTRA_SIM_ARGS_JSON/EXTRA_SIM_ARGS" >&2
      exit 2
      ;;
  esac
done
RENDER_ENABLED=1
for extra_arg in "${EXTRA_SIM_ARGS_ARR[@]}"; do
  if [ "${extra_arg}" = "--no-render" ]; then
    RENDER_ENABLED=0
  fi
done

if [ "${#SCENE_DIR_ARGS[@]}" -gt 0 ]; then
  GS_SCENE_DIRS="$(IFS=:; echo "${SCENE_DIR_ARGS[*]}")"
fi
[ -n "${GS_SCENE_DIRS}" ] || {
  echo "[real2sim-batch] provide at least one --scene-dir or set GS_SCENE_DIRS" >&2
  exit 2
}
IFS=: read -r -a REQUESTED_SCENES <<< "${GS_SCENE_DIRS}"
REQUESTED_SCENE_COUNT=0
for requested_scene in "${REQUESTED_SCENES[@]}"; do
  if [ -n "${requested_scene}" ]; then
    REQUESTED_SCENE_COUNT=$((REQUESTED_SCENE_COUNT + 1))
  fi
done
if [ "${REQUESTED_SCENE_COUNT}" -lt 1 ]; then
  echo "[real2sim-batch] GS_SCENE_DIRS contains no non-empty scene path" >&2
  exit 2
fi
[[ "${REPEATS}" =~ ^[0-9]+$ ]] && [ "${REPEATS}" -ge 1 ] || {
  echo "[real2sim-batch] REPEATS/--repeats must be a positive integer, got ${REPEATS}" >&2
  exit 2
}
case "${REAL2SIM_AGGREGATE_ONLY}" in
  0|1) ;;
  *) echo "[real2sim-batch] REAL2SIM_AGGREGATE_ONLY must be 0 or 1" >&2; exit 2 ;;
esac
if [ -n "${REAL2SIM_SHARD_NAME}" ] \
  && ! [[ "${REAL2SIM_SHARD_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "[real2sim-batch] REAL2SIM_SHARD_NAME must be a safe path component" >&2
  exit 2
fi
[[ "${REPEAT_START}" =~ ^[0-9]+$ ]] && [ "${REPEAT_START}" -ge 1 ] || {
  echo "[real2sim-batch] REPEAT_START/--repeat-start must be a positive integer, got ${REPEAT_START}" >&2
  exit 2
}
[[ "${REPEAT_STOP}" =~ ^[0-9]+$ ]] && [ "${REPEAT_STOP}" -ge 1 ] || {
  echo "[real2sim-batch] REPEAT_STOP/--repeat-stop must be a positive integer, got ${REPEAT_STOP}" >&2
  exit 2
}
REPEATS="$((10#${REPEATS}))"
REPEAT_START="$((10#${REPEAT_START}))"
REPEAT_STOP="$((10#${REPEAT_STOP}))"
if [ "${REPEAT_START}" -gt "${REPEAT_STOP}" ] \
  || [ "${REPEAT_STOP}" -gt "${REPEATS}" ]; then
  echo "[real2sim-batch] repeat range must satisfy 1 <= start <= stop <= REPEATS (start=${REPEAT_START}, stop=${REPEAT_STOP}, repeats=${REPEATS})" >&2
  exit 2
fi
REPEAT_RANGE_LENGTH=$((REPEAT_STOP - REPEAT_START + 1))
PARTIAL_REPEAT_RANGE=0
if [ "${REPEAT_START}" -ne 1 ] || [ "${REPEAT_STOP}" -ne "${REPEATS}" ]; then
  PARTIAL_REPEAT_RANGE=1
fi
if [ "${REAL2SIM_AGGREGATE_ONLY}" = "1" ] \
  && [ "${PARTIAL_REPEAT_RANGE}" = "1" ]; then
  echo "[real2sim-batch] aggregate-only mode requires the full repeat range" >&2
  exit 2
fi
if [ -z "${GRID_TEMPLATE}" ]; then
  case "${REPEATS}" in
    20)
      GRID_TEMPLATE="${ROOT}/config/render_templates/rollout_grid_clear_headers_5x5.json"
      ;;
    30)
      GRID_TEMPLATE="${ROOT}/config/render_templates/rollout_grid_clear_headers_6x6.json"
      ;;
    100)
      GRID_TEMPLATE="${ROOT}/config/render_templates/rollout_grid_clear_headers_10x12.json"
      ;;
    *)
      GRID_TEMPLATE="${ROOT}/config/render_templates/rollout_grid_clear_headers_4x3.json"
      ;;
  esac
fi
if [ -n "${POLICY_BATCH_SIZE}" ] \
  && { ! [[ "${POLICY_BATCH_SIZE}" =~ ^[0-9]+$ ]] || [ "${POLICY_BATCH_SIZE}" -lt 1 ]; }; then
  echo "[real2sim-batch] POLICY_BATCH_SIZE/--policy-batch-size must be a positive integer, got ${POLICY_BATCH_SIZE}" >&2
  exit 2
fi
if [ -n "${PARALLEL_REPEATS}" ] \
  && { ! [[ "${PARALLEL_REPEATS}" =~ ^[0-9]+$ ]] || [ "${PARALLEL_REPEATS}" -lt 1 ]; }; then
  echo "[real2sim-batch] PARALLEL_REPEATS/--parallel-repeats must be a positive integer, got ${PARALLEL_REPEATS}" >&2
  exit 2
fi
if [ -z "${POLICY_BATCH_SIZE}" ] && [ -z "${PARALLEL_REPEATS}" ]; then
  if [ "${REPEATS}" = "10" ]; then
    POLICY_BATCH_SIZE=10
  else
    POLICY_BATCH_SIZE="${REQUESTED_SCENE_COUNT}"
  fi
fi
if [ -z "${PARALLEL_REPEATS}" ]; then
  if [ $((POLICY_BATCH_SIZE % REQUESTED_SCENE_COUNT)) -ne 0 ]; then
    echo "[real2sim-batch] POLICY_BATCH_SIZE must divide evenly across scenes (batch=${POLICY_BATCH_SIZE}, scenes=${REQUESTED_SCENE_COUNT})" >&2
    exit 2
  fi
  PARALLEL_REPEATS=$((POLICY_BATCH_SIZE / REQUESTED_SCENE_COUNT))
elif [ -z "${POLICY_BATCH_SIZE}" ]; then
  POLICY_BATCH_SIZE=$((REQUESTED_SCENE_COUNT * PARALLEL_REPEATS))
elif [ "${POLICY_BATCH_SIZE}" -ne $((REQUESTED_SCENE_COUNT * PARALLEL_REPEATS)) ]; then
  echo "[real2sim-batch] POLICY_BATCH_SIZE must equal scenes * PARALLEL_REPEATS (batch=${POLICY_BATCH_SIZE}, scenes=${REQUESTED_SCENE_COUNT}, parallel=${PARALLEL_REPEATS})" >&2
  exit 2
fi
case "${POSTHOC_RENDER_REQUIRED}" in
  auto|0|1) ;;
  *)
    echo "[real2sim-batch] POSTHOC_RENDER_REQUIRED must be auto, 0, or 1" >&2
    exit 2
    ;;
esac
if [ "${POLICY_BATCH_SIZE}" -ge "${POSTHOC_RENDER_BATCH_THRESHOLD}" ]; then
  if [ "${POSTHOC_RENDER_REQUIRED}" = "auto" ]; then
    POSTHOC_RENDER_REQUIRED="${RENDER_ENABLED}"
  elif [ "${POSTHOC_RENDER_REQUIRED}" = "0" ] \
    && [ "${RENDER_ENABLED}" = "1" ]; then
    echo "[real2sim-batch] launcher contract is inconsistent: B${POLICY_BATCH_SIZE} requested video without post-hoc finalization" >&2
    exit 2
  fi
  if [ "${POSTHOC_RENDER_REQUIRED}" = "1" ]; then
    echo "[real2sim-batch] B${POLICY_BATCH_SIZE} forces --no-render during generation and automatically renders third-person video afterward"
  fi
  RENDER_ENABLED=0
else
  if [ "${POSTHOC_RENDER_REQUIRED}" = "1" ] \
    && [ -z "${REAL2SIM_SHARD_NAME}" ]; then
    echo "[real2sim-batch] POSTHOC_RENDER_REQUIRED=1 requires policy batch size >= ${POSTHOC_RENDER_BATCH_THRESHOLD}" >&2
    exit 2
  fi
  if [ -z "${REAL2SIM_SHARD_NAME}" ] \
    || [ "${POSTHOC_RENDER_REQUIRED}" != "1" ]; then
    POSTHOC_RENDER_REQUIRED=0
  else
    RENDER_ENABLED=0
  fi
fi
# Normalize the runner-owned render selector so worker and official-contract
# commands contain exactly one --no-render when rendering is disabled.
NORMALIZED_EXTRA_SIM_ARGS_ARR=()
for extra_arg in "${EXTRA_SIM_ARGS_ARR[@]}"; do
  if [ "${extra_arg}" != "--no-render" ]; then
    NORMALIZED_EXTRA_SIM_ARGS_ARR+=("${extra_arg}")
  fi
done
if [ "${RENDER_ENABLED}" = "0" ]; then
  NORMALIZED_EXTRA_SIM_ARGS_ARR+=(--no-render)
fi
EXTRA_SIM_ARGS_ARR=("${NORMALIZED_EXTRA_SIM_ARGS_ARR[@]}")
if [ "${REAL2SIM_STRICT_LAUNCH}" = "1" ] \
  && { [ "${RENDER_ENABLED}" = "1" ] || [ "${POSTHOC_RENDER_REQUIRED}" = "1" ]; } \
  && [ "${REPEATS}" != "10" ] \
  && [ "${REPEATS}" != "20" ] \
  && [ "${REPEATS}" != "30" ] \
  && [ "${REPEATS}" != "50" ] \
  && [ "${REPEATS}" != "100" ]; then
  echo "[real2sim-batch] rendered official runs require a checked-in 10-, 20-, 30-, 50-, or 100-rollout grid template" >&2
  exit 2
fi
if [ -z "${RUN_ROOT}" ]; then
  RUN_ROOT="${EXPORTS_ROOT}/${ROBOT_TAG}_rollout_$(date +%Y%m%d_%H%M%S)"
fi
case "${RENDER_LOCK_ROOT}" in
  /*) ;;
  *) RENDER_LOCK_ROOT="${ROOT}/${RENDER_LOCK_ROOT}" ;;
esac
STATE_ROOT="${INIT_STATE_ROOT:-${EXPORTS_ROOT}/deterministic-init-state}"
STATE_DIR_MODE=750
STATE_FILE_MODE=640
case "${CONFIG_SRC}" in
  deploy|training) ;;
  *) echo "[real2sim-batch] --config-src must be deploy or training, got ${CONFIG_SRC}" >&2; exit 2 ;;
esac
case "${GRASP_MODE}" in
  physics|rule) ;;
  *) echo "[real2sim-batch] GRASP_MODE must be physics or rule, got ${GRASP_MODE}" >&2; exit 2 ;;
esac
for value_name in CKPT CONFIG OBJECTS INSTRUCTION GOAL_CONDITION SCORE_CONDITION; do
  [ -n "${!value_name}" ] || { echo "[real2sim-batch] ${value_name} must not be empty" >&2; exit 2; }
done
[ -d "${CKPT}" ] || { echo "[real2sim-batch] checkpoint directory not found: ${CKPT}" >&2; exit 2; }
if [ -n "${POLICY_IMAGE_EPISODE}" ] && [ ! -f "${POLICY_IMAGE_EPISODE}" ]; then
  echo "[real2sim-batch] policy image episode not found: ${POLICY_IMAGE_EPISODE}" >&2
  exit 2
fi
if [ ! -x "${PY}" ]; then echo "[real2sim-batch] python not executable: ${PY}" >&2; exit 2; fi
[ -f "${GRID_TEMPLATE}" ] || { echo "[real2sim-batch] grid template not found: ${GRID_TEMPLATE}" >&2; exit 2; }
for n in "${REPEATS}" "${PARALLEL_REPEATS}" "${POLICY_BATCH_SIZE}" "${FRAMES}" "${GS_PORT_BASE}" "${GS_SERVERS_PER_GPU}"; do
  [[ "${n}" =~ ^[0-9]+$ ]] && [ "${n}" -ge 1 ] || { echo "[real2sim-batch] positive integer expected, got ${n}" >&2; exit 2; }
done
if [ "${REAL2SIM_AGGREGATE_ONLY}" = "0" ] \
  && { [ "${PARALLEL_REPEATS}" -gt "${REPEAT_RANGE_LENGTH}" ] \
    || [ $((REPEAT_RANGE_LENGTH % PARALLEL_REPEATS)) -ne 0 ]; }; then
  if [ "${PARTIAL_REPEAT_RANGE}" = "0" ]; then
    echo "[real2sim-batch] REPEATS must be divisible by and >= PARALLEL_REPEATS (repeats=${REPEATS}, parallel_repeats=${PARALLEL_REPEATS})" >&2
  else
    echo "[real2sim-batch] repeat-range length must be divisible by and >= PARALLEL_REPEATS (range=${REPEAT_START}-${REPEAT_STOP}, parallel_repeats=${PARALLEL_REPEATS})" >&2
  fi
  exit 2
fi
[[ "${ACTION_HORIZON}" =~ ^[0-9]+$ ]] && [ "${ACTION_HORIZON}" -ge 1 ] || {
  echo "[real2sim-batch] ACTION_HORIZON/--action-horizon must be a positive integer, got ${ACTION_HORIZON}" >&2
  exit 2
}
[[ "${EXEC_STEPS}" =~ ^[0-9]+$ ]] && [ "${EXEC_STEPS}" -ge 1 ] || {
  echo "[real2sim-batch] EXEC_STEPS/--exec-steps must be a positive integer, got ${EXEC_STEPS}" >&2
  exit 2
}
[[ "${RF_OVERLAP}" =~ ^[0-9]+$ ]] || {
  echo "[real2sim-batch] RF_OVERLAP/--rf-overlap must be a non-negative integer, got ${RF_OVERLAP}" >&2
  exit 2
}
case "${RF_OVERLAP_SAMPLER}" in
  single|paired|paired_mean|paired_weighted) ;;
  *)
    echo "[real2sim-batch] RF_OVERLAP_SAMPLER/--rf-overlap-sampler must be single, paired, paired_mean, or paired_weighted, got ${RF_OVERLAP_SAMPLER}" >&2
    exit 2
    ;;
esac
[[ "${RF_OVERLAP_STEP_SHIFT}" =~ ^[0-9]+$ ]] || {
  echo "[real2sim-batch] RF_OVERLAP_STEP_SHIFT/--rf-overlap-step-shift must be a non-negative integer, got ${RF_OVERLAP_STEP_SHIFT}" >&2
  exit 2
}
ACTION_HORIZON="$((10#${ACTION_HORIZON}))"
EXEC_STEPS="$((10#${EXEC_STEPS}))"
RF_OVERLAP="$((10#${RF_OVERLAP}))"
RF_OVERLAP_STEP_SHIFT="$((10#${RF_OVERLAP_STEP_SHIFT}))"
if [ "${RF_OVERLAP}" -gt "${EXEC_STEPS}" ]; then
  echo "[real2sim-batch] RF_OVERLAP/--rf-overlap (${RF_OVERLAP}) must not exceed EXEC_STEPS/--exec-steps (${EXEC_STEPS})" >&2
  exit 2
fi
if [ "${RF_OVERLAP_SAMPLER}" != "single" ]; then
  required_action_horizon=$((2 * RF_OVERLAP))
  if [ "${EXEC_STEPS}" -gt "${required_action_horizon}" ]; then
    required_action_horizon="${EXEC_STEPS}"
  fi
  required_horizon_reason="max(EXEC_STEPS, 2 * RF_OVERLAP)"
else
  required_action_horizon=$((EXEC_STEPS + RF_OVERLAP))
  required_horizon_reason="EXEC_STEPS + RF_OVERLAP"
fi
if [ "${ACTION_HORIZON}" -lt "${required_action_horizon}" ]; then
  echo "[real2sim-batch] ACTION_HORIZON/--action-horizon must be >= ${required_horizon_reason} (${required_action_horizon}), got ${ACTION_HORIZON}" >&2
  exit 2
fi
if [ "${RF_OVERLAP_SAMPLER}" != "single" ] && [ "${RF_OVERLAP}" -eq 0 ]; then
  echo "[real2sim-batch] ${RF_OVERLAP_SAMPLER} RF overlap sampling requires RF_OVERLAP/--rf-overlap > 0" >&2
  exit 2
fi
if [ "${RF_OVERLAP_STEP_SHIFT}" -ne 0 ] && [ "${RF_OVERLAP_SAMPLER}" = "single" ]; then
  echo "[real2sim-batch] nonzero RF_OVERLAP_STEP_SHIFT/--rf-overlap-step-shift requires --rf-overlap-sampler paired, paired_mean, or paired_weighted" >&2
  exit 2
fi
if [ "${RF_OVERLAP_SAMPLER}" = "paired_weighted" ] && [ -z "${RF_OVERLAP_PREV_WEIGHT}" ]; then
  RF_OVERLAP_PREV_WEIGHT="0.5"
fi
if [ -n "${RF_OVERLAP_PREV_WEIGHT}" ]; then
  if ! awk -v weight="${RF_OVERLAP_PREV_WEIGHT}" 'BEGIN { exit !(weight == weight && weight >= 0 && weight <= 1) }'; then
    echo "[real2sim-batch] RF_OVERLAP_PREV_WEIGHT/--rf-overlap-prev-weight must be finite and in [0,1], got ${RF_OVERLAP_PREV_WEIGHT}" >&2
    exit 2
  fi
  if [ "${RF_OVERLAP_SAMPLER}" = "single" ]; then
    echo "[real2sim-batch] RF_OVERLAP_PREV_WEIGHT is only valid with paired, paired_mean, or paired_weighted" >&2
    exit 2
  fi
fi
case "${POLICY_DETERMINISTIC_XLA}" in
  0|1) ;;
  *) echo "[real2sim-batch] POLICY_DETERMINISTIC_XLA must be 0 or 1, got ${POLICY_DETERMINISTIC_XLA}" >&2; exit 2 ;;
esac
case "${CLEAN_VIDEO}" in
  0|1) ;;
  *) echo "[real2sim-batch] CLEAN_VIDEO must be 0 or 1, got ${CLEAN_VIDEO}" >&2; exit 2 ;;
esac
case "${REAL2SIM_STRICT_LAUNCH}" in
  0|1) ;;
  *) echo "[real2sim-batch] REAL2SIM_STRICT_LAUNCH must be 0 or 1" >&2; exit 2 ;;
esac
if [ "${REAL2SIM_STRICT_LAUNCH}" = "1" ]; then
  # Schema-v2 launch manifests predate this runner-local tuning knob. Keep the
  # official execution contract fixed without requiring a launcher schema
  # migration; diagnostic/direct launches may still override it explicitly.
  GS_NUM_THREADS=1
  if [ "${CLI_ARGUMENT_COUNT}" -ne 0 ]; then
    echo "[real2sim-batch] official schema-v2 runs do not accept direct batch CLI overrides" >&2
    exit 2
  fi
  [ -f "${REAL2SIM_OFFICIAL_RUN_MANIFEST}" ] || {
    echo "[real2sim-batch] official schema-v2 runs must start through real2sim-run; missing REAL2SIM_OFFICIAL_RUN_MANIFEST" >&2
    exit 2
  }
  "${PY}" - "${REAL2SIM_OFFICIAL_RUN_MANIFEST}" "${RUN_ROOT}" \
    "${REPEAT_START}" "${REPEAT_STOP}" "${REAL2SIM_SHARD_NAME}" \
    "${REAL2SIM_AGGREGATE_ONLY}" <<'PY'
import json
import hashlib
import os
import re
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).expanduser().resolve()
expected_path = (Path(sys.argv[2]).expanduser().resolve() / "run_manifest.json")
repeat_start, repeat_stop, shard_name, aggregate_only = sys.argv[3:]
if manifest_path != expected_path:
    raise SystemExit(
        f"official run manifest must be {expected_path}, got {manifest_path}"
    )
try:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    contracts = payload["preflight"]["contracts"]
    asset_digest = contracts["assets"]["contract_digest"]
    scenes = contracts["scenes"]
    base_environment = payload["resolved_environment"]
    expected_resolved_digest = payload["config"]["resolved_sha256"]
except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
    raise SystemExit(f"official run manifest has no validated contracts: {exc}") from exc
if not isinstance(asset_digest, str) or re.fullmatch(r"[0-9a-f]{64}", asset_digest) is None:
    raise SystemExit("official run manifest has an invalid asset contract digest")
if not isinstance(scenes, dict) or not scenes:
    raise SystemExit("official run manifest has no validated scene contract")
for tag, scene in scenes.items():
    digest = scene.get("contract_digest") if isinstance(scene, dict) else None
    if (
        not isinstance(tag, str)
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise SystemExit("official run manifest has an invalid scene contract digest")

resolved_path = expected_path.parent / "resolved_rollout_config.json"
try:
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"official resolved config is unreadable: {resolved_path}: {exc}") from exc
resolved_digest = hashlib.sha256(
    json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
if resolved_digest != expected_resolved_digest:
    raise SystemExit("official resolved config changed after launcher preflight")
if resolved.get("schema_version") != 2:
    raise SystemExit("official run requires a schema-version 2 resolved config")
if Path(resolved.get("output", {}).get("run_root", "")).resolve() != expected_path.parent:
    raise SystemExit("official resolved config belongs to a different run root")

shard_execution = payload.get("shard_execution")
if shard_execution is None:
    if shard_name or aggregate_only != "0" or repeat_start != "1" or repeat_stop != str(
        resolved.get("runner", {}).get("repeats", "")
    ):
        raise SystemExit("unsharded official run cannot select a repeat subset or shard phase")
    expected_environment = base_environment
else:
    if (
        not isinstance(shard_execution, dict)
        or shard_execution.get("schema_version") != 1
        or not isinstance(shard_execution.get("shards"), list)
        or not shard_name
    ):
        raise SystemExit("official shard execution authority is invalid")
    if aggregate_only == "1":
        if shard_name != "aggregate":
            raise SystemExit("official aggregation must use the aggregate shard identity")
        expected_environment = shard_execution.get("aggregation_environment")
    elif aggregate_only == "0":
        matches = [
            item
            for item in shard_execution["shards"]
            if isinstance(item, dict) and item.get("name") == shard_name
        ]
        if len(matches) != 1:
            raise SystemExit(f"official shard name is not configured: {shard_name!r}")
        expected_environment = matches[0].get("environment")
    else:
        raise SystemExit("REAL2SIM_AGGREGATE_ONLY must be 0 or 1")
    if not isinstance(expected_environment, dict):
        raise SystemExit("official shard environment is invalid")

launcher_owned_environment = {
    "ROBOT", "PY", "CKPT", "CONFIG", "CONFIG_SRC", "OBJECTS",
    "INSTRUCTION", "POLICY_IMAGE_EPISODE", "GOAL_CONDITION",
    "SCORE_CONDITION", "RUN_ROOT", "REPEATS", "PARALLEL_REPEATS",
    "POLICY_BATCH_SIZE",
    "FRAMES", "MAX_WALL_SEC", "ACTION_HORIZON", "EXEC_STEPS",
    "POLICY_NOISE_SEED", "POLICY_DETERMINISTIC_XLA",
    "DETERMINISTIC_SIM", "DETERMINISTIC_SEED", "INIT_STATE_JSON",
    "RF_OVERLAP", "RF_OVERLAP_SAMPLER", "RF_OVERLAP_PREV_WEIGHT",
    "RF_OVERLAP_STEP_SHIFT", "GRIP_TORQUE_MULT", "GRASP_MODE",
    "GRIPPER_CALIBRATION", "EXTRA_SIM_ARGS", "EXTRA_SIM_ARGS_JSON",
    "WHITE_DESK_TEX", "VIEW_AZIMUTH", "VIEW_ELEVATION",
    "VIEW_DISTANCE", "VIEW_LOOKAT", "RENDER_WRIST", "WRIST_GS",
    "SHOW_ORIGIN", "RUN_ANALYSIS",
    "CLEAN_VIDEO", "SERVER_HOST", "SERVER_PORT", "GS_PORT_BASE",
    "GS_START_TIMEOUT_SEC", "GS_BG_COLOR", "SIM_NUM_THREADS", "GAUSSIAN_WRAPPING_REPO",
    "GS_SCENE_DIRS", "BATCH_CLIENT_ORDER", "GS_CUDA_DEVICES", "GS_SERVERS_PER_GPU",
    "POLICY_CUDA_DEVICE", "SIM_CUDA_DEVICES", "RENDER_LOCK_ROOT",
    "REAL2SIM_GPU_BINDINGS_JSON",
    "OPENPI_DATA_HOME",
    "REAL2SIM_STRICT_LAUNCH",
    "REAL2SIM_OFFICIAL_RUN_MANIFEST",
    "POSTHOC_RENDER_REQUIRED",
    # Official launch does not expose these legacy/direct override knobs.
    "EXPORTS_ROOT", "INIT_STATE_ROOT", "INIT_STATE_CACHE_SCHEMA",
    "MIMICGEN_INIT_DATASET_ROOT", "MIMICGEN_INIT_STATE_LOG_ROOT",
    "MIMICGEN_INIT_EPISODE_OFFSET", "POLICY_NOISE_BANK_DIR",
    "POLICY_INFERENCE_MANIFEST", "GRID_TEMPLATE",
}
if not isinstance(base_environment, dict) or not isinstance(expected_environment, dict):
    raise SystemExit("official run manifest has no resolved launcher environment")
# A manifest recorded before a launcher variable was retired still lists it;
# this release neither exports nor reads it, so it is history, not drift.
from real2sim.runner.environment import without_retired_environment
expected_environment = without_retired_environment(expected_environment)
launcher_owned_environment.update(str(key) for key in expected_environment)
drift = {
    key: (expected_environment.get(key, ""), os.environ.get(key, ""))
    for key in sorted(launcher_owned_environment)
    if str(expected_environment.get(key, "")) != os.environ.get(key, "")
}
if drift:
    keys = ", ".join(drift)
    raise SystemExit(f"official runner environment differs from launcher manifest: {keys}")
PY
fi
if [ -z "${SIM_NUM_THREADS}" ]; then
  if [ "${DETERMINISTIC_SIM}" = "1" ]; then
    SIM_NUM_THREADS=1
  else
    SIM_NUM_THREADS=4
  fi
fi
[[ "${SIM_NUM_THREADS}" =~ ^[0-9]+$ ]] && [ "${SIM_NUM_THREADS}" -ge 1 ] || {
  echo "[real2sim-batch] SIM_NUM_THREADS must be a positive integer, got ${SIM_NUM_THREADS}" >&2
  exit 2
}
# CPU thread caps for the simulation side: the policy server, every worker,
# and every helper Python process launched for a worker. Without them each
# process starts one OpenMP/BLAS thread per host core, which can exhaust a
# container's task limit when several workers start at once.
SIM_THREAD_VARIABLES=(
  OMP_NUM_THREADS
  MKL_NUM_THREADS
  OPENBLAS_NUM_THREADS
  OPENCV_FOR_THREADS_NUM
)
export_sim_thread_caps() {
  local name
  for name in "${SIM_THREAD_VARIABLES[@]}"; do
    export "${name}=${SIM_NUM_THREADS}"
  done
}
# Run a helper Python process for a worker in a subshell with the caps.
sim_helper_python() {
  (
    export_sim_thread_caps
    exec "${PY}" "$@"
  )
}
[[ "${GS_NUM_THREADS}" =~ ^[0-9]+$ ]] && [ "${GS_NUM_THREADS}" -ge 1 ] || {
  echo "[real2sim-batch] GS_NUM_THREADS must be a positive integer, got ${GS_NUM_THREADS}" >&2
  exit 2
}
# Resolve and validate the complete selector/CUDA/EGL contract before creating
# the render-lock directory, run root, or any child process. Actual launches
# re-prove sealed and direct bindings against nvidia-smi here.
# A shard always seals its target-step boundary. Fixture/direct manifests do
# not carry a launcher-time binding, so seal that runtime boundary here too;
# workers need the physical EGL identity rather than only allocation slots.
if [ -n "${REAL2SIM_SHARD_NAME}" ] || [ -z "${REAL2SIM_GPU_BINDINGS_JSON}" ]; then
  REAL2SIM_GPU_BINDINGS_JSON="$(
    PYTHONNOUSERSITE=1 \
      REAL2SIM_GPU_BINDINGS_JSON= \
      "${PY}" -m real2sim.gpu_binding json
  )" || {
    echo "[real2sim-batch] failed to seal runtime GPU bindings" >&2
    exit 2
  }
fi
export REAL2SIM_GPU_BINDINGS_JSON
GPU_BINDING_ARGS=(shell)
if [ "${DRY_RUN}" = "1" ] && [ -n "${REAL2SIM_GPU_BINDINGS_JSON}" ]; then
  # A dry-run starts no GPU consumer. Keep sealed fixture tests host-neutral
  # while still applying the complete structural/CVD/role validation.
  GPU_BINDING_ARGS+=(--structural-only)
fi
GPU_BINDING_ROWS="$(
  PYTHONNOUSERSITE=1 \
    "${PY}" -m real2sim.gpu_binding "${GPU_BINDING_ARGS[@]}"
)" || {
  echo "[real2sim-batch] GPU binding validation failed before process launch" >&2
  exit 2
}
declare -A GPU_CUDA_TOKEN_BY_SELECTOR=()
declare -A GPU_EGL_INDEX_BY_SELECTOR=()
gpu_roles_seen=0
while IFS=$'\t' read -r record first second third; do
  case "${record}" in
    roles)
      GS_CUDA_DEVICES="${first}"
      POLICY_CUDA_DEVICE="${second}"
      SIM_CUDA_DEVICES="${third}"
      gpu_roles_seen=$((gpu_roles_seen + 1))
      ;;
    binding)
      [[ "${first}" =~ ^(0|[1-9][0-9]*)$ ]] \
        && [ -n "${second}" ] \
        && [[ "${third}" =~ ^(0|[1-9][0-9]*)$ ]] || {
        echo "[real2sim-batch] invalid GPU binding record from resolver" >&2
        exit 2
      }
      [ -z "${GPU_CUDA_TOKEN_BY_SELECTOR[${first}]+set}" ] || {
        echo "[real2sim-batch] duplicate GPU selector ${first} from resolver" >&2
        exit 2
      }
      GPU_CUDA_TOKEN_BY_SELECTOR["${first}"]="${second}"
      GPU_EGL_INDEX_BY_SELECTOR["${first}"]="${third}"
      ;;
    "") ;;
    *)
      echo "[real2sim-batch] unknown GPU binding record from resolver: ${record}" >&2
      exit 2
      ;;
  esac
done <<< "${GPU_BINDING_ROWS}"
[ "${gpu_roles_seen}" -eq 1 ] || {
  echo "[real2sim-batch] GPU resolver did not return exactly one role record" >&2
  exit 2
}
IFS=, read -r -a GS_GPU_IDS <<< "${GS_CUDA_DEVICES}"
IFS=, read -r -a SIM_GPU_IDS <<< "${SIM_CUDA_DEVICES}"
for gpu_id in "${GS_GPU_IDS[@]}" "${POLICY_CUDA_DEVICE}" "${SIM_GPU_IDS[@]}"; do
  [ -n "${GPU_CUDA_TOKEN_BY_SELECTOR[${gpu_id}]+set}" ] \
    && [ -n "${GPU_EGL_INDEX_BY_SELECTOR[${gpu_id}]+set}" ] || {
    echo "[real2sim-batch] GPU selector ${gpu_id} has no validated CUDA/EGL binding" >&2
    exit 2
  }
done

# How many render requests one 3DGS server must hold at once.
#
# Each worker has at most one wrist render outstanding, including across the
# left-to-right transition, so a server's fan-in is the concurrent workers
# divided over the servers they round-robin across. The server refuses beyond
# its admission capacity with HTTP 503 (RenderQueueFull). The capacity is
# derived from the layout, so the limit tracks it instead of being a constant.
# It is an admission limit, not a thread count or a queue of renders waiting
# on the JPEG encoder, which is what the three neighbouring knobs are and
# which this deliberately leaves alone.
#
# The floor is 8, and the +2 is headroom for a retry arriving before its
# predecessor's slot clears.
GS_SERVER_COUNT=$(( ${#GS_GPU_IDS[@]} * GS_SERVERS_PER_GPU ))
[ "${GS_SERVER_COUNT}" -ge 1 ] || GS_SERVER_COUNT=1
GS_FANIN_PER_SERVER=$(( (POLICY_BATCH_SIZE + GS_SERVER_COUNT - 1) / GS_SERVER_COUNT ))
GS_ADMISSION_CAPACITY=$(( GS_FANIN_PER_SERVER + 2 ))
[ "${GS_ADMISSION_CAPACITY}" -ge 8 ] || GS_ADMISSION_CAPACITY=8
echo "[real2sim-batch] gs admission capacity=${GS_ADMISSION_CAPACITY} (batch=${POLICY_BATCH_SIZE} over ${GS_SERVER_COUNT} gs server(s) = ${GS_FANIN_PER_SERVER}/server + 2)"

install -d -m 700 "${RENDER_LOCK_ROOT}"
RENDER_LOCK_ROOT="$(cd "${RENDER_LOCK_ROOT}" && pwd -P)"
SHARD_RUNTIME_ROOT="${RUN_ROOT}"
if [ -n "${REAL2SIM_SHARD_NAME}" ]; then
  SHARD_RUNTIME_ROOT="${RUN_ROOT}/.shards/${REAL2SIM_SHARD_NAME}"
elif [ "${PARTIAL_REPEAT_RANGE}" = "1" ]; then
  SHARD_RUNTIME_ROOT="${RUN_ROOT}/.ranges/${REPEAT_START}-${REPEAT_STOP}"
fi
SERVER_LOG_ROOT="${SHARD_RUNTIME_ROOT}/server_logs"
install -d -m 750 "${RUN_ROOT}" "${SHARD_RUNTIME_ROOT}" "${SERVER_LOG_ROOT}"
if [ "${REAL2SIM_STRICT_LAUNCH}" = "1" ]; then
  EXECUTION_CLASS="official"
else
  EXECUTION_CLASS="diagnostic"
fi
printf '%s\n' "${EXECUTION_CLASS}" > "${RUN_ROOT}/execution_class.txt"
SCENES_RAW="${SHARD_RUNTIME_ROOT}/gs_scenes_raw.tsv"
SCENES_RESOLVED="${SHARD_RUNTIME_ROOT}/gs_scenes.tsv"
GS_WORKERS_RESOLVED="${SHARD_RUNTIME_ROOT}/gs_workers.tsv"
GS_SERVERS_RESOLVED="${SHARD_RUNTIME_ROOT}/gs_servers.tsv"
SERVER_READY="${SHARD_RUNTIME_ROOT}/policy_batch_server_ready.json"
SERVER_LOG="${SHARD_RUNTIME_ROOT}/policy_batch_server.log"
PHYSICS_CONTRACT_FILE="${SHARD_RUNTIME_ROOT}/physics_contract.sha256"
RUNTIME_SOURCE_MANIFEST_EXPECTED="${RUN_ROOT}/runtime_source_manifest.json"
RUNTIME_SOURCE_VERIFICATION_LOG="${SHARD_RUNTIME_ROOT}/runtime_source_verifications.jsonl"
OUTPUTS_CSV="${RUN_ROOT}/outputs.csv"
if [ "${PARTIAL_REPEAT_RANGE}" = "1" ]; then
  OUTPUTS_CSV="${SHARD_RUNTIME_ROOT}/outputs.csv"
fi
SERVER_PID=""
GS_PIDS=()
PIDS=()
TERMINATION_GRACE_SEC=5

verify_runtime_source_manifest() {
  local label="$1"
  local report
  local rc
  if [ "${REAL2SIM_RUNTIME_SOURCE_UNSEALED}" = "1" ]; then
    # The launcher could not seal this run and said so. There is nothing to
    # verify against, and re-deriving a manifest here would seal the already
    # changed tree and call it verified.
    echo "[real2sim-batch] runtime source verification skipped at ${label}: launched unsealed" >&2
    printf '%s\n' "{\"label\":\"${label}\",\"pid\":$$,\"verification\":{\"status\":\"unsealed\"}}" \
      >> "${RUNTIME_SOURCE_VERIFICATION_LOG}" || return 1
    return 0
  fi
  # set -e is on: a bare assignment from a failing command substitution would
  # exit the script before the exit code could be read, which is the whole
  # point of this function now.
  rc=0
  report="$(
    "${PY}" -m real2sim.runtime_manifest verify \
      --manifest "${REAL2SIM_RUNTIME_SOURCE_MANIFEST}" \
      --expected-sha256 "${REAL2SIM_RUNTIME_SOURCE_MANIFEST_SHA256}"
  )" || rc=$?
  if [ "${rc}" -eq 3 ]; then
    # Exit 3 is source drift: the manifest is intact and the working tree moved
    # under a running batch. Warn and record it; finished rollouts are kept.
    # The event below is the record; if it cannot be written there is no
    # record, so that is a failure rather than something to swallow.
    echo "[real2sim-batch] WARNING: runtime source/config generation changed at ${label}; continuing anyway" >&2
    printf '%s\n' "{\"label\":\"${label}\",\"pid\":$$,\"verification\":{\"status\":\"changed\"}}" \
      >> "${RUNTIME_SOURCE_VERIFICATION_LOG}" || return 1
    return 0
  fi
  if [ "${rc}" -ne 0 ]; then
    # Any other non-zero exit means the manifest is missing, unreadable,
    # malformed or self-inconsistent. That is not evidence the rollouts are
    # fine, so it refuses.
    echo "[real2sim-batch] runtime source/config manifest is unusable at ${label} (exit ${rc}); refusing to launch or accept another worker wave" >&2
    return 1
  fi
  "${PY}" - "${label}" "${RUNTIME_SOURCE_VERIFICATION_LOG}" "${report}" <<'PY'
import json
import os
import sys
from pathlib import Path

label, output_path, raw_report = sys.argv[1:]
report = json.loads(raw_report)
event = {
    "label": label,
    "pid": os.getpid(),
    "verification": report,
}
with Path(output_path).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
PY
  echo "[real2sim-batch] runtime generation verified label=${label} files=$("${PY}" -c 'import json,sys; print(json.loads(sys.argv[1])["file_count"])' "${report}")"
}

initialize_runtime_source_manifest() {
  if { [ -n "${REAL2SIM_RUNTIME_SOURCE_MANIFEST}" ] && [ -z "${REAL2SIM_RUNTIME_SOURCE_MANIFEST_SHA256}" ]; } \
    || { [ -z "${REAL2SIM_RUNTIME_SOURCE_MANIFEST}" ] && [ -n "${REAL2SIM_RUNTIME_SOURCE_MANIFEST_SHA256}" ]; }; then
    echo "[real2sim-batch] REAL2SIM_RUNTIME_SOURCE_MANIFEST and REAL2SIM_RUNTIME_SOURCE_MANIFEST_SHA256 must be set together" >&2
    return 2
  fi
  if [ "${REAL2SIM_RUNTIME_SOURCE_UNSEALED}" = "1" ]; then
    # The launcher tried to seal, found the tree had already moved, and chose
    # to run anyway. Honour the launcher's decision not to seal; never re-seal
    # a changed checkout.
    if [ -n "${REAL2SIM_RUNTIME_SOURCE_MANIFEST}" ]; then
      echo "[real2sim-batch] REAL2SIM_RUNTIME_SOURCE_UNSEALED=1 must not be combined with a manifest" >&2
      return 2
    fi
    echo "[real2sim-batch] launched unsealed: no runtime source manifest for this run" >&2
    return 0
  fi
  if [ -z "${REAL2SIM_RUNTIME_SOURCE_MANIFEST}" ]; then
    local create_args=(
      --repo-root "${ROOT}"
      --output "${RUNTIME_SOURCE_MANIFEST_EXPECTED}"
    )
    local provenance_file
    for provenance_file in \
      "${RUN_ROOT}/rollout_config.yaml" \
      "${RUN_ROOT}/resolved_rollout_config.json" \
      "${RUN_ROOT}/run_manifest.json"; do
      if [ -f "${provenance_file}" ]; then
        create_args+=(--extra-file "${provenance_file}")
      fi
    done
    local create_report
    create_report="$(
      "${PY}" -m real2sim.runtime_manifest create "${create_args[@]}"
    )"
    REAL2SIM_RUNTIME_SOURCE_MANIFEST="${RUNTIME_SOURCE_MANIFEST_EXPECTED}"
    REAL2SIM_RUNTIME_SOURCE_MANIFEST_SHA256="$(
      "${PY}" -c 'import json,sys; print(json.loads(sys.argv[1])["manifest_sha256"])' "${create_report}"
    )"
    export REAL2SIM_RUNTIME_SOURCE_MANIFEST REAL2SIM_RUNTIME_SOURCE_MANIFEST_SHA256
    echo "[real2sim-batch] diagnostic direct launch self-sealed runtime generation"
  fi

  [ -f "${REAL2SIM_RUNTIME_SOURCE_MANIFEST}" ] || {
    echo "[real2sim-batch] runtime source manifest not found: ${REAL2SIM_RUNTIME_SOURCE_MANIFEST}" >&2
    return 2
  }
  local actual_manifest
  actual_manifest="$(cd "$(dirname "${REAL2SIM_RUNTIME_SOURCE_MANIFEST}")" && pwd -P)/$(basename "${REAL2SIM_RUNTIME_SOURCE_MANIFEST}")"
  local expected_manifest
  expected_manifest="$(cd "$(dirname "${RUNTIME_SOURCE_MANIFEST_EXPECTED}")" && pwd -P)/$(basename "${RUNTIME_SOURCE_MANIFEST_EXPECTED}")"
  if [ "${actual_manifest}" != "${expected_manifest}" ]; then
    echo "[real2sim-batch] runtime source manifest must belong to this run root: expected ${expected_manifest}, got ${actual_manifest}" >&2
    return 2
  fi
  REAL2SIM_RUNTIME_SOURCE_MANIFEST="${actual_manifest}"
  export REAL2SIM_RUNTIME_SOURCE_MANIFEST REAL2SIM_RUNTIME_SOURCE_MANIFEST_SHA256
  : > "${RUNTIME_SOURCE_VERIFICATION_LOG}"
  verify_runtime_source_manifest "pre-services"
}

collect_process_tree() {
  local root_pid="$1"
  local child_pid
  while read -r child_pid; do
    [ -n "${child_pid}" ] || continue
    collect_process_tree "${child_pid}"
  done < <(ps -o pid= --ppid "${root_pid}" 2>/dev/null)
  printf '%s\n' "${root_pid}"
}

process_is_running() {
  local pid="$1"
  local state
  kill -0 "${pid}" 2>/dev/null || return 1
  state="$(ps -o stat= -p "${pid}" 2>/dev/null)" || return 1
  [ -n "${state}" ] && [[ "${state}" != Z* ]]
}

terminate_pids() {
  local label="$1"
  shift
  local root_pids=("$@")
  local target_pids=()
  local process_tree=()
  local root_pid pid grace_tick any_running
  if [ "${#root_pids[@]}" -eq 0 ]; then
    return 0
  fi

  # Background worker functions own a simulator subshell and tee process.
  # Snapshot every descendant before signaling so those children cannot be
  # orphaned when their wrapper exits first. collect_process_tree emits leaves
  # first, preserving a child-before-parent shutdown order.
  for root_pid in "${root_pids[@]}"; do
    mapfile -t process_tree < <(collect_process_tree "${root_pid}")
    target_pids+=("${process_tree[@]}")
  done

  echo "[real2sim-batch] stopping ${label}: root_pids=${root_pids[*]}"
  for pid in "${target_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done

  # Poll the whole saved tree, not just the wrapper roots: a wrapper may exit
  # immediately while a simulator child is still draining CUDA work. All
  # batch processes share this one grace interval rather than waiting
  # sequentially.
  for ((grace_tick = 0; grace_tick < TERMINATION_GRACE_SEC * 10; grace_tick++)); do
    any_running=0
    for pid in "${target_pids[@]}"; do
      if process_is_running "${pid}"; then
        any_running=1
        break
      fi
    done
    [ "${any_running}" -eq 1 ] || break
    sleep 0.1
  done

  for pid in "${target_pids[@]}"; do
    if process_is_running "${pid}"; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done

  for root_pid in "${root_pids[@]}"; do
    wait "${root_pid}" 2>/dev/null || true
  done
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  if [ "${#PIDS[@]}" -gt 0 ]; then
    terminate_pids "workers" "${PIDS[@]}"
    PIDS=()
  fi
  stop_generation_services
  exit "${rc}"
}

stop_generation_services() {
  if [ -n "${SERVER_PID}" ]; then
    terminate_pids "policy server" "${SERVER_PID}"
    SERVER_PID=""
  fi
  if [ "${#GS_PIDS[@]}" -gt 0 ]; then
    terminate_pids "GS servers" "${GS_PIDS[@]}"
    GS_PIDS=()
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

port_up() {
  local port="$1"
  curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1
}

gs_server_matches() {
  local port="$1" token="$2" ply="$3" ply_sha256="$4"
  "${PY}" - "${port}" "${token}" "${ply}" "${ply_sha256}" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request
from pathlib import Path

port, token, ply, digest = sys.argv[1:]
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/health",
    headers={"X-Real2Sim-GS-Token": token} if token else {},
)
with urllib.request.urlopen(request, timeout=2) as response:
    identity = json.load(response)
if identity != {
    "status": "ok",
    "ready_token": token,
    "ply_path": str(Path(ply).resolve()),
    "ply_sha256": digest,
}:
    raise SystemExit(1)
PY
}

discover_scenes() {
  "${PY}" - "${GS_SCENE_DIRS}" "${SCENES_RAW}" "${REAL2SIM_STRICT_LAUNCH}" "${PHYSICS_CONTRACT_FILE}" "${REAL2SIM_OFFICIAL_RUN_MANIFEST}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from robosuite_env.contracts.scene import (
    SceneContractError,
    scene_tag,
    validate_scene_contract,
)
from robosuite_env.contracts.assets import validate_repository_assets

scene_entries = [p for p in sys.argv[1].split(os.pathsep) if p.strip()]
out = Path(sys.argv[2])
strict = sys.argv[3] == "1"
rows = []
seen = set()
seen_ids = set()
reports = []

for entry in scene_entries:
    try:
        report = validate_scene_contract(entry, strict=strict)
    except (OSError, ValueError, SceneContractError) as exc:
        raise SystemExit(f"invalid explicit GS scene {entry!r}: {exc}") from exc
    base = report["background_dir"]
    ply_path = report["ply_path"]
    name = str(report["name"])
    scene_id = str(report["id"])
    tag = scene_tag(name)
    while tag in seen:
        raise SystemExit(
            f"duplicate scene tag {tag!r}; give each explicit background.json "
            "a distinct name"
        )
    seen.add(tag)
    if strict and scene_id in seen_ids:
        raise SystemExit(
            f"duplicate official scene id {scene_id!r}; each comparison scene "
            "must have a unique id"
        )
    seen_ids.add(scene_id)
    rows.append((tag, str(base), str(ply_path), name, str(report["ply_sha256"])))
    reports.append(report)

if not rows:
    raise SystemExit(f"no explicit GS background.json found in GS_SCENE_DIRS={sys.argv[1]}")

with out.open("w") as f:
    for row in rows:
        f.write("\t".join(row) + "\n")
print(f"[discover] scenes={len(rows)} file={out}")
for row in rows:
    print("[discover]", row)

if strict:
    collision_hashes = {report["collision_tree_sha256"] for report in reports}
    if len(collision_hashes) != 1:
        raise SystemExit(
            "all scenes in one official comparison must use the same table "
            "collision contract"
        )
    launch_manifest = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
    expected_contracts = launch_manifest["preflight"]["contracts"]
    bundle_ids = expected_contracts["assets"]["bundles"]
    assets = validate_repository_assets(
        Path("robosuite_env/assets"), bundle_ids=bundle_ids
    )
    if assets["contract_digest"] != expected_contracts["assets"]["contract_digest"]:
        raise SystemExit("repository assets changed after launcher preflight")
    expected_scenes = {
        tag: {
            "contract_digest": scene["contract_digest"],
            "manifest_path": str(Path(scene["manifest_path"]).resolve()),
        }
        for tag, scene in expected_contracts["scenes"].items()
    }
    actual_scenes = {
        scene_tag(report["name"]): {
            "contract_digest": report["contract_digest"],
            "manifest_path": str(Path(report["manifest_path"]).resolve()),
        }
        for report in reports
    }
    if actual_scenes != expected_scenes:
        raise SystemExit(
            "3DGS scene path, tag, or content changed after launcher preflight"
        )
    payload = {
        "asset_contract_sha256": assets["contract_digest"],
        "table_collision_sha256": next(iter(collision_hashes)),
    }
    physics_digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
else:
    physics_digest = "legacy-unverified"
Path(sys.argv[4]).write_text(physics_digest + "\n", encoding="utf-8")
PY
}

resolve_ports() {
  : > "${SCENES_RESOLVED}"
  : > "${GS_WORKERS_RESOLVED}"
  : > "${GS_SERVERS_RESOLVED}"
  local next_port="${GS_PORT_BASE}"
  local tag bg_dir ply name ply_sha256 port slot gpu_idx gpu_id sim_gpu_idx sim_gpu_id server_key token
  local worker_ordinal=0
  local -A server_ports=()
  local -A server_tokens=()
  while IFS=$'\t' read -r tag bg_dir ply name ply_sha256; do
    for slot in $(seq 1 "${PARALLEL_REPEATS}"); do
      gpu_idx=$((worker_ordinal % ${#GS_GPU_IDS[@]}))
      gpu_id="${GS_GPU_IDS[$gpu_idx]}"
      sim_gpu_idx=$((worker_ordinal % ${#SIM_GPU_IDS[@]}))
      sim_gpu_id="${SIM_GPU_IDS[$sim_gpu_idx]}"
      server_slot=$(((worker_ordinal / ${#GS_GPU_IDS[@]}) % GS_SERVERS_PER_GPU))
      server_key="${tag}|${gpu_id}|${server_slot}"
      if [ -n "${server_ports[${server_key}]+set}" ]; then
        port="${server_ports[${server_key}]}"
        token="${server_tokens[${server_key}]}"
      else
        port="${next_port}"
        while port_up "${port}"; do
          port=$((port + 1))
        done
        if [ "${port}" -gt 65535 ]; then
          echo "[real2sim-batch] no free GS server port at or above ${GS_PORT_BASE}" >&2
          return 1
        fi
        token="$("${PY}" -c 'import secrets; print(secrets.token_hex(32))')"
        server_ports["${server_key}"]="${port}"
        server_tokens["${server_key}"]="${token}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${tag}" "${bg_dir}" "${ply}" "${port}" "${gpu_id}" "${name}" "${token}" "${ply_sha256}" >> "${GS_SERVERS_RESOLVED}"
        next_port=$((port + 1))
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${tag}" "${slot}" "${bg_dir}" "${ply}" "${port}" "${gpu_id}" "${sim_gpu_id}" "${name}" "${token}" "${ply_sha256}" >> "${GS_WORKERS_RESOLVED}"
      if [ "${slot}" = "1" ]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}" "${bg_dir}" "${ply}" "${port}" "${name}" "${token}" "${ply_sha256}" >> "${SCENES_RESOLVED}"
      fi
      worker_ordinal=$((worker_ordinal + 1))
    done
  done < "${SCENES_RAW}"
}

read_scenes() {
  TAGS=()
  BG_DIRS=()
  PLYS=()
  PORTS=()
  NAMES=()
  GS_TOKENS=()
  PLY_SHA256S=()
  local tag bg_dir ply port name token ply_sha256
  while IFS=$'\t' read -r tag bg_dir ply port name token ply_sha256; do
    TAGS+=("${tag}")
    BG_DIRS+=("${bg_dir}")
    PLYS+=("${ply}")
    PORTS+=("${port}")
    NAMES+=("${name}")
    GS_TOKENS+=("${token}")
    PLY_SHA256S+=("${ply_sha256}")
  done < "${SCENES_RESOLVED}"
  SCENE_COUNT="${#TAGS[@]}"
  if [ "${SCENE_COUNT}" -lt 1 ]; then
    echo "[real2sim-batch] no scenes after port resolution" >&2
    exit 2
  fi
  WORKER_TAGS=()
  WORKER_SLOTS=()
  WORKER_BG_DIRS=()
  WORKER_PLYS=()
  WORKER_PORTS=()
  WORKER_GPUS=()
  WORKER_SIM_GPUS=()
  WORKER_GS_TOKENS=()
  WORKER_PLY_SHA256S=()
  local slot gpu_id sim_gpu_id
  while IFS=$'\t' read -r tag slot bg_dir ply port gpu_id sim_gpu_id name token ply_sha256; do
    WORKER_TAGS+=("${tag}")
    WORKER_SLOTS+=("${slot}")
    WORKER_BG_DIRS+=("${bg_dir}")
    WORKER_PLYS+=("${ply}")
    WORKER_PORTS+=("${port}")
    WORKER_GPUS+=("${gpu_id}")
    WORKER_SIM_GPUS+=("${sim_gpu_id}")
    WORKER_GS_TOKENS+=("${token}")
    WORKER_PLY_SHA256S+=("${ply_sha256}")
  done < "${GS_WORKERS_RESOLVED}"
  local worker_count="${#WORKER_TAGS[@]}"
  if [ "${worker_count}" -ne "${POLICY_BATCH_SIZE}" ]; then
    echo "[real2sim-batch] internal worker expansion mismatch: workers=${worker_count} policy_batch_size=${POLICY_BATCH_SIZE} scenes=${SCENE_COUNT} parallel_repeats=${PARALLEL_REPEATS}" >&2
    exit 2
  fi
  if [ -z "${BATCH_CLIENT_ORDER}" ]; then
    BATCH_CLIENT_ORDER="$(IFS=,; printf '%s' "${TAGS[*]}")"
  fi
}

start_gs_servers() {
  local tag bg_dir ply port gpu_id name token ply_sha256 pid waited cuda_token egl_index
  while IFS=$'\t' read -r tag bg_dir ply port gpu_id name token ply_sha256; do
    cuda_token="${GPU_CUDA_TOKEN_BY_SELECTOR[${gpu_id}]}"
    egl_index="${GPU_EGL_INDEX_BY_SELECTOR[${gpu_id}]}"
    echo "[real2sim-batch] start GS ${tag}/gpu_${gpu_id} port=${port} selector=${gpu_id} cuda=${cuda_token} egl=${egl_index} cpu_threads=${GS_NUM_THREADS} ply=${ply}"
    if [ "${DRY_RUN}" = "1" ]; then
      continue
    fi
    (
      export CUDA_VISIBLE_DEVICES="${cuda_token}"
      export OMP_NUM_THREADS="${GS_NUM_THREADS}"
      export MKL_NUM_THREADS="${GS_NUM_THREADS}"
      export OPENBLAS_NUM_THREADS="${GS_NUM_THREADS}"
      export NUMEXPR_NUM_THREADS="${GS_NUM_THREADS}"
      export VECLIB_MAXIMUM_THREADS="${GS_NUM_THREADS}"
      export BLIS_NUM_THREADS="${GS_NUM_THREADS}"
      exec "${PY}" -u scene/gs_background_server.py \
        --ply "${ply}" \
        --gw-repo "${GAUSSIAN_WRAPPING_REPO}" \
        --host 127.0.0.1 \
        --port "${port}" \
        --ready-token "${token}" \
        --render-lock "${RENDER_LOCK_ROOT}/mujoco_render_faces_gpu_${egl_index}.lock" \
        --width 640 \
        --height 480 \
        --jpeg-backlog-capacity "${GS_ADMISSION_CAPACITY}" \
        --bg-color "${GS_BG_COLOR}"
    ) > "${SERVER_LOG_ROOT}/gs_${tag}_gpu_${gpu_id}_${port}.log" 2>&1 &
    pid="$!"
    GS_PIDS+=("${pid}")
    waited=0
    while [ "${waited}" -lt "${GS_START_TIMEOUT_SEC}" ]; do
      if gs_server_matches "${port}" "${token}" "${ply}" "${ply_sha256}"; then
        echo "[real2sim-batch] GS ${tag}/gpu_${gpu_id} UP pid=${pid} port=${port} gpu=${gpu_id}"
        break
      fi
      if ! kill -0 "${pid}" 2>/dev/null; then
        echo "[real2sim-batch] GS ${tag}/gpu_${gpu_id} exited early; see ${SERVER_LOG_ROOT}/gs_${tag}_gpu_${gpu_id}_${port}.log" >&2
        return 1
      fi
      sleep 2
      waited=$((waited + 2))
    done
    if ! gs_server_matches "${port}" "${token}" "${ply}" "${ply_sha256}"; then
      echo "[real2sim-batch] GS ${tag}/gpu_${gpu_id} did not publish the expected token/PLY identity after ${GS_START_TIMEOUT_SEC}s" >&2
      return 1
    fi
  done < "${GS_SERVERS_RESOLVED}"
}

start_policy_server() {
  rm -f "${SERVER_READY}"
  SERVER_CMD=(
    "${PY}" -m robosuite_env.policy.policy_batch_server
  )
  SERVER_CMD+=(--policy "${CKPT}")
  SERVER_CMD+=(
    --config "${CONFIG}"
    --config-src "${CONFIG_SRC}"
    --ah "${ACTION_HORIZON}"
    --host "${SERVER_HOST}"
    --port "${SERVER_PORT}"
    --batch-size "${POLICY_BATCH_SIZE}"
    --ready-file "${SERVER_READY}"
  )
  if [ -n "${BATCH_CLIENT_ORDER:-}" ]; then
    SERVER_CMD+=(--client-order "${BATCH_CLIENT_ORDER}")
  fi
  if [ -n "${POLICY_NOISE_SEED}" ]; then
    if [ -z "${POLICY_NOISE_BANK_DIR}" ]; then
      POLICY_NOISE_BANK_DIR="${RUN_ROOT}/policy_noise_bank"
    fi
    if [ -z "${POLICY_INFERENCE_MANIFEST}" ]; then
      POLICY_INFERENCE_MANIFEST="${RUN_ROOT}/policy_inference_manifest.jsonl"
    fi
    SERVER_CMD+=(
      --noise-seed "${POLICY_NOISE_SEED}"
      --noise-bank-dir "${POLICY_NOISE_BANK_DIR}"
    )
  fi
  if [ -n "${POLICY_INFERENCE_MANIFEST}" ]; then
    SERVER_CMD+=(--inference-manifest "${POLICY_INFERENCE_MANIFEST}")
  fi
  local policy_cuda_token="${GPU_CUDA_TOKEN_BY_SELECTOR[${POLICY_CUDA_DEVICE}]}"
  local policy_egl_index="${GPU_EGL_INDEX_BY_SELECTOR[${POLICY_CUDA_DEVICE}]}"
  echo "[real2sim-batch] start policy batch server ${SERVER_HOST}:${SERVER_PORT} batch_size=${POLICY_BATCH_SIZE} selector=${POLICY_CUDA_DEVICE} cuda=${policy_cuda_token} egl=${policy_egl_index}"
  if [ "${DRY_RUN}" = "1" ]; then
    printf '[real2sim-batch] server command:'
    printf ' %q' "${SERVER_CMD[@]}"
    printf '\n'
    return 0
  fi
  (
    export CUDA_VISIBLE_DEVICES="${policy_cuda_token}"
    export PYTHONUNBUFFERED=1
    export PYTHONNOUSERSITE=1
    export PYTHONHASHSEED="${DETERMINISTIC_SEED}"
    export DETERMINISTIC_SEED
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export_sim_thread_caps
    if [ "${POLICY_DETERMINISTIC_XLA}" = "1" ]; then
      export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_deterministic_ops=true --xla_gpu_exclude_nondeterministic_ops=true"
      export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
    fi
    exec "${SERVER_CMD[@]}"
  ) > "${SERVER_LOG}" 2>&1 &
  SERVER_PID="$!"
  for _ in $(seq 1 360); do
    if [ -s "${SERVER_READY}" ]; then
      echo "[real2sim-batch] policy batch server ready: $(cat "${SERVER_READY}")"
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[real2sim-batch] policy batch server exited early; see ${SERVER_LOG}" >&2
      return 1
    fi
    sleep 1
  done
  echo "[real2sim-batch] policy batch server did not become ready; see ${SERVER_LOG}" >&2
  return 1
}

# The worker runs as a module of the installed robosuite_env package. Every
# worker command is "${PY}" "${COMMON_ARGS[@]}" ...; the worker contract and the
# init-state identity take the arguments after this head.
SIM_WORKER=(-m robosuite_env.sim)
COMMON_ARGS=(
  "${SIM_WORKER[@]}"
  --robot "${ROBOT}"
  --policy-batch-server "${SERVER_HOST}:${SERVER_PORT}"
  --config "${CONFIG}"
  --config-src "${CONFIG_SRC}"
  --objects "${OBJECTS}"
  --instruction "${INSTRUCTION}"
  --ah "${ACTION_HORIZON}"
  --exec "${EXEC_STEPS}"
  --rf-overlap "${RF_OVERLAP}"
  --rf-overlap-sampler "${RF_OVERLAP_SAMPLER}"
  --rf-overlap-step-shift "${RF_OVERLAP_STEP_SHIFT}"
  --grip-torque-mult "${GRIP_TORQUE_MULT}"
  --grasp-mode "${GRASP_MODE}"
  --deterministic-seed "${DETERMINISTIC_SEED}"
  --frames "${FRAMES}"
  --max-wall-sec "${MAX_WALL_SEC}"
  --policy-batch-timeout-sec "${MAX_WALL_SEC}"
  --goal-condition "${GOAL_CONDITION}"
  --score-condition "${SCORE_CONDITION}"
  --view-azimuth "${VIEW_AZIMUTH}"
  --view-elevation "${VIEW_ELEVATION}"
  --view-distance "${VIEW_DISTANCE}"
)
if [ "${REAL2SIM_STRICT_LAUNCH}" = "0" ]; then
  # Compatibility defaults for direct callers that bypass the YAML launcher.
  # The canonical schema-v2 launcher sets every one through typed parameters.
  COMMON_ARGS+=(
    --desk-tex "${WHITE_DESK_TEX}"
    --physics dynamic
    --dynamic-subframes 4
    --grip-friction "0.25 0.025 0.0001"
    --rubber-friction "1.0 0.05 0.0001"
    --policy-image-resize pad
    --policy-obs-inset
  )
else
  # Schema-v2 runs are official. The simulator repeats the same manifest and
  # scene validation immediately before MuJoCo construction so this shell
  # runner cannot become an asset-contract bypass.
  COMMON_ARGS+=(
    --strict-assets
    --official-run-manifest "${REAL2SIM_OFFICIAL_RUN_MANIFEST}"
  )
fi
if [ -n "${RF_OVERLAP_PREV_WEIGHT}" ]; then
  COMMON_ARGS+=(--rf-overlap-prev-weight "${RF_OVERLAP_PREV_WEIGHT}")
fi
if [ -n "${POLICY_IMAGE_EPISODE}" ]; then
  COMMON_ARGS+=(--policy-image-episode "${POLICY_IMAGE_EPISODE}")
fi
if [ -n "${GRIPPER_CALIBRATION}" ]; then
  COMMON_ARGS+=(--gripper-calibration "${GRIPPER_CALIBRATION}")
fi
if [ -n "${VIEW_LOOKAT}" ]; then
  COMMON_ARGS+=(--view-lookat "${VIEW_LOOKAT}")
fi
if [ "${RENDER_WRIST}" != "1" ]; then
  COMMON_ARGS+=(--no-wrist)
fi
if [ "${WRIST_GS}" = "1" ]; then
  COMMON_ARGS+=(--wrist-gs)
fi
if [ "${SHOW_ORIGIN}" != "1" ]; then
  COMMON_ARGS+=(--no-origin)
fi
if [ "${CLEAN_VIDEO}" = "1" ]; then
  COMMON_ARGS+=(--clean-video)
fi
if [ "${#EXTRA_SIM_ARGS_ARR[@]}" -gt 0 ]; then
  COMMON_ARGS+=("${EXTRA_SIM_ARGS_ARR[@]}")
fi

write_run_command() {
  local outdir="$1"
  local gpu_id="$2"
  local cuda_token="${GPU_CUDA_TOKEN_BY_SELECTOR[${gpu_id}]}"
  local egl_index="${GPU_EGL_INDEX_BY_SELECTOR[${gpu_id}]}"
  local name
  shift 2
  {
    echo "# Run Command"
    echo
    printf 'GPU binding: selector `%s`, CUDA token `%s`, MuJoCo EGL physical index `%s`.\n\n' \
      "${gpu_id}" "${cuda_token}" "${egl_index}"
    echo '```bash'
    printf 'MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=%q CUDA_VISIBLE_DEVICES=%q PYTHONNOUSERSITE=1 PYTHONHASHSEED=%q XLA_PYTHON_CLIENT_PREALLOCATE=false' \
      "${egl_index}" "" "${DETERMINISTIC_SEED}"
    for name in "${SIM_THREAD_VARIABLES[@]}"; do
      printf ' %s=%q' "${name}" "${SIM_NUM_THREADS}"
    done
    printf ' %q' "$@"
    echo
    echo '```'
  } > "${outdir}/RUN_COMMAND.md"
}

write_official_worker_contract() {
  local contract_path="$1"
  shift
  if [ "${REAL2SIM_STRICT_LAUNCH}" != "1" ]; then
    return 0
  fi
  sim_helper_python -m robosuite_env.contracts.worker write \
    --contract "${contract_path}" \
    --run-manifest "${REAL2SIM_OFFICIAL_RUN_MANIFEST}" \
    -- "$@"
}

cache_tag() {
  local raw="$1"
  local fallback="$2"
  local max_chars="$3"
  local slug digest
  slug="$(printf '%s' "${raw}" | LC_ALL=C tr '[:upper:]' '[:lower:]' | LC_ALL=C sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//' | cut -c1-"${max_chars}")"
  [ -n "${slug}" ] || slug="${fallback}"
  digest="$(printf '%s' "${raw}" | sha256sum | awk '{print substr($1, 1, 8)}')"
  printf '%s_%s' "${slug}" "${digest}"
}

prepare_deterministic_init_state() {
  local selected_layout="${1:-}"
  local selected_layout_index="${2:-0}"
  local append_common="${3:-1}"
  local selected_layout_args=()
  if [ -n "${selected_layout}" ]; then
    selected_layout_args=(
      --box-pack-layout-json "${selected_layout}"
      --box-pack-layout-index "${selected_layout_index}"
    )
  fi
  if [ "${DETERMINISTIC_SIM}" != "1" ]; then
    return 0
  fi
  if [ -z "${INIT_STATE_JSON}" ]; then
    local torque_tag="${GRIP_TORQUE_MULT//./p}"
    local objects_tag task_tag anchor_tag sim_tag objects_sha source_sha
    local calibration_tag init_identity init_tag
    objects_tag="$(cache_tag "${OBJECTS}" objects 32)"
    task_tag="$(cache_tag "${INSTRUCTION}" task 48)"
    anchor_tag="$(cache_tag "${POLICY_IMAGE_EPISODE:-default}" anchor 24)"
    if [ -f "${OBJECTS}" ]; then
      objects_sha="$(sha256sum "${OBJECTS}" | awk '{print $1}')"
    else
      objects_sha="$(printf '%s' "${OBJECTS}" | sha256sum | awk '{print $1}')"
    fi
    source_sha="$(
      {
        git -C "${ROOT}" rev-parse HEAD 2>/dev/null || true
        sha256sum \
          "${ROOT}/robosuite_env/sim.py" \
          "${ROOT}/robosuite_env/paths.py"
        find \
          "${ROOT}/robosuite_env/contracts" \
          "${ROOT}/robosuite_env/policy" \
          "${ROOT}/robosuite_env/rendering" \
          "${ROOT}/robosuite_env/robots" \
          "${ROOT}/robosuite_env/rollout" \
          "${ROOT}/robosuite_env/tasks" \
          -type f \( -name '*.py' -o -name '*.yaml' \) -print0 \
          | LC_ALL=C sort -z \
          | xargs -0 -r sha256sum
        sha256sum \
          "${ROOT}/robosuite_env/assets/manifest.json" \
          "${ROOT}/robosuite_env/controllers/default_fr3_joint_ik.json"
      } | sha256sum | awk '{print $1}'
    )"
    # Layouts, pose contracts, the initial arm pose and the base calibration
    # are data the source hash does not cover, and the named tasks (
    # box_pack) select their layout without naming a file in
    # OBJECTS. Hash the exact files this worker argument vector resolves: the
    # identity the worker records in a state and checks when it loads one.
    local scene_inputs_sha
    if ! scene_inputs_sha="$(
      sim_helper_python -m robosuite_env.rollout.state.init_identity -- "${COMMON_ARGS[@]:${#SIM_WORKER[@]}}" \
        "${selected_layout_args[@]}" \
        | tail -n 1
    )" || [[ ! "${scene_inputs_sha}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "[real2sim-batch] cannot resolve the deterministic init-state scene inputs" >&2
      return 1
    fi
    # Controller, cadence, integrator, and physical overrides can all change the
    # post-ready deterministic state. Include source and object contents in
    # addition to the exact extra-argument vector; sim.py also validates a
    # compiled physical-model fingerprint before accepting a cached state.
    local physics_contract
    physics_contract="$(<"${PHYSICS_CONTRACT_FILE}")"
    sim_tag="$(cache_tag "schema=${INIT_STATE_CACHE_SCHEMA};physics_contract=${physics_contract};objects_sha=${objects_sha};scene_inputs_sha=${scene_inputs_sha};source_sha=${source_sha};grasp=${GRASP_MODE};args=${EXTRA_SIM_ARGS_JSON:-${EXTRA_SIM_ARGS:-default}}" sim 24)"
    calibration_tag="disabled"
    if [ -n "${GRIPPER_CALIBRATION}" ]; then
      if [ ! -f "${GRIPPER_CALIBRATION}" ]; then
        echo "[real2sim-batch] GRIPPER_CALIBRATION is not a file: ${GRIPPER_CALIBRATION}" >&2
        return 1
      fi
      local right_gripper_xml="${ROOT}/robosuite_env/assets/yubi_right.xml"
      local left_gripper_xml="${ROOT}/robosuite_env/assets/yubi_left.xml"
      if [ ! -f "${right_gripper_xml}" ]; then
        echo "[real2sim-batch] calibrated right-gripper MJCF is not a file: ${right_gripper_xml}" >&2
        return 1
      fi
      if [ ! -f "${left_gripper_xml}" ]; then
        echo "[real2sim-batch] calibrated left-gripper MJCF is not a file: ${left_gripper_xml}" >&2
        return 1
      fi
      calibration_tag="$(
        {
          sha256sum "${GRIPPER_CALIBRATION}"
          sha256sum "${right_gripper_xml}"
          sha256sum "${left_gripper_xml}"
        } | sha256sum | awk '{print substr($1, 1, 16)}'
      )"
    fi
    init_identity="objects=${objects_tag};task=${task_tag};anchor=${anchor_tag};sim=${sim_tag};torque=${torque_tag};calibration=${calibration_tag};seed=${DETERMINISTIC_SEED}"
    init_tag="$(cache_tag "${init_identity}" init 48)"
    INIT_STATE_JSON="${STATE_ROOT}/${ROBOT_TAG}_yubi_policy_deterministic_init_state_${init_tag}.json"
  fi
  install -d -m "${STATE_DIR_MODE}" "$(dirname "${INIT_STATE_JSON}")"
  local init_state_lock_fd=""
  if [ "${DRY_RUN}" != "1" ]; then
    local init_state_lock="${INIT_STATE_JSON}.lock"
    exec {init_state_lock_fd}>"${init_state_lock}"
    chmod "${STATE_FILE_MODE}" "${init_state_lock}"
    flock "${init_state_lock_fd}"
  fi
  if [ -s "${INIT_STATE_JSON}" ]; then
    chmod "${STATE_FILE_MODE}" "${INIT_STATE_JSON}"
    echo "[real2sim-batch] deterministic init-state reuse=${INIT_STATE_JSON}"
    PREPARED_INIT_STATE_JSON="${INIT_STATE_JSON}"
    if [ "${append_common}" = "1" ]; then
      COMMON_ARGS+=(--init-state-in "${INIT_STATE_JSON}")
    fi
    if [ "${DRY_RUN}" != "1" ]; then
      flock -u "${init_state_lock_fd}"
      exec {init_state_lock_fd}>&-
    fi
    return 0
  fi

  local prep_dir="${SHARD_RUNTIME_ROOT}/deterministic_init_state_prep"
  if [ -n "${selected_layout}" ]; then
    prep_dir="${prep_dir}/$(printf 'layout_%02d' "${selected_layout_index}")"
  fi
  local prep_logdir="${prep_dir}/logs"
  local prep_summary="${prep_dir}/sim_summary.json"
  local prep_video="${prep_dir}/init_state_only.mp4"
  mkdir -p "${prep_logdir}"
  local prep_no_render_args=()
  if [ "${RENDER_ENABLED}" = "1" ]; then
    prep_no_render_args=(--no-render)
  fi
  local prep_cmd=("${COMMON_ARGS[@]}"
    "${selected_layout_args[@]}"
    --gs-background "${BG_DIRS[0]}"
    --official-scene-tag "${TAGS[0]}"
    --gs-port "${PORTS[0]}"
    --gs-server-token "${GS_TOKENS[0]}"
    "${prep_no_render_args[@]}"
    # INIT_STATE_JSON is exported for child workers.  While creating a new
    # cache, explicitly override sim.py's environment-derived input default so
    # the not-yet-created output is not also opened as --init-state-in.
    --init-state-in ""
    --init-state-out "${INIT_STATE_JSON}"
    --init-state-only
    --out "${prep_video}"
    --logdir "${prep_logdir}"
    --summary-path "${prep_summary}"
  )
  if [ "${REAL2SIM_STRICT_LAUNCH}" = "1" ]; then
    local prep_worker_contract="${prep_dir}/official_worker_contract.json"
    write_official_worker_contract \
      "${prep_worker_contract}" "${prep_cmd[@]:${#SIM_WORKER[@]}}"
    prep_cmd+=(--official-worker-contract "${prep_worker_contract}")
  fi
  echo "[real2sim-batch] deterministic init-state create=${INIT_STATE_JSON}"
  if [ "${DRY_RUN}" = "1" ]; then
    printf '[real2sim-batch] init-state command:'
    printf ' %q' "${PY}" "${prep_cmd[@]}"
    printf '\n'
  else
    local prep_gpu_id="${WORKER_SIM_GPUS[0]}"
    local prep_egl_index="${GPU_EGL_INDEX_BY_SELECTOR[${prep_gpu_id}]}"
    (
      set -euo pipefail
      umask 027
      export MUJOCO_GL=egl
      export CUDA_VISIBLE_DEVICES=""
      export MUJOCO_EGL_DEVICE_ID="${prep_egl_index}"
      export PYTHONNOUSERSITE=1
      export PYTHONUNBUFFERED=1
      export PYTHONHASHSEED="${DETERMINISTIC_SEED}"
      export DETERMINISTIC_SEED
      export XLA_PYTHON_CLIENT_PREALLOCATE=false
      export_sim_thread_caps
      "${PY}" "${prep_cmd[@]}"
    ) > "${prep_dir}/sim_stdout.log" 2>&1
    chmod "${STATE_FILE_MODE}" "${INIT_STATE_JSON}"
  fi
  PREPARED_INIT_STATE_JSON="${INIT_STATE_JSON}"
  if [ "${append_common}" = "1" ]; then
    COMMON_ARGS+=(--init-state-in "${INIT_STATE_JSON}")
  fi
  if [ "${DRY_RUN}" != "1" ]; then
    flock -u "${init_state_lock_fd}"
    exec {init_state_lock_fd}>&-
  fi
}

run_worker() {
  local idx="$1" repeat="$2"
  local slot=$(((repeat - 1) % PARALLEL_REPEATS))
  local worker_idx=$((idx * PARALLEL_REPEATS + slot))
  local tag="${TAGS[$idx]}"
  local bg_dir="${BG_DIRS[$idx]}"
  local port="${WORKER_PORTS[$worker_idx]}"
  local gs_token="${WORKER_GS_TOKENS[$worker_idx]}"
  local gpu_id="${WORKER_SIM_GPUS[$worker_idx]}"
  local cuda_token="${GPU_CUDA_TOKEN_BY_SELECTOR[${gpu_id}]}"
  local egl_index="${GPU_EGL_INDEX_BY_SELECTOR[${gpu_id}]}"
  local run_tag
  run_tag="$(printf 'run_%02d' "${repeat}")"
  local outdir="${RUN_ROOT}/${tag}/${run_tag}"
  local logdir="${outdir}/logs"
  local video="${outdir}/${tag}_${run_tag}.mp4"
  local summary="${outdir}/sim_summary.json"
  local stdout_log="${outdir}/sim_stdout.log"
  local exit_code="${outdir}/exit_code"
  local video_validation="${outdir}/video_validation.json"
  mkdir -p "${outdir}" "${logdir}"

  local mimicgen_init_args=()
  local selected_layout_args=()
  local selected_init_state_args=()
  if [ "${#BOX_PACK_LAYOUTS[@]}" -gt 0 ]; then
    local layout_index=$(((repeat - 1) % ${#BOX_PACK_LAYOUTS[@]}))
    selected_layout_args=(
      --box-pack-layout-json "${BOX_PACK_LAYOUTS[${layout_index}]}"
      --box-pack-layout-index "${layout_index}"
    )
    if [ "${DETERMINISTIC_SIM}" = "1" ]; then
      selected_init_state_args=(--init-state-in "${INIT_STATE_JSON_BY_LAYOUT[${layout_index}]}")
    fi
  fi
  if [ -n "${MIMICGEN_INIT_DATASET_ROOT}" ]; then
    local episode_index episode_chunk episode_path
    episode_index=$((10#${MIMICGEN_INIT_EPISODE_OFFSET} + repeat - 1))
    episode_chunk=$((episode_index / 1000))
    episode_path="$(printf '%s/data/chunk-%03d/episode_%06d.parquet' \
      "${MIMICGEN_INIT_DATASET_ROOT}" "${episode_chunk}" "${episode_index}")"
    if [ ! -s "${episode_path}" ]; then
      echo "[real2sim-batch] MimicGen init episode not found for repeat ${repeat}: ${episode_path}" >&2
      return 2
    fi
    mimicgen_init_args=(
      --mimicgen-init-episode "${episode_path}"
      --mimicgen-init-state-log-root "${MIMICGEN_INIT_STATE_LOG_ROOT}"
    )
  fi

  local cmd=("${COMMON_ARGS[@]}"
    "${selected_init_state_args[@]}"
    "${selected_layout_args[@]}"
    "${mimicgen_init_args[@]}"
    --policy-batch-client-id "${tag}_${run_tag}"
    --gs-background "${bg_dir}"
    --official-scene-tag "${tag}"
    --gs-port "${port}"
    --gs-server-token "${gs_token}"
    --render-lock "${RENDER_LOCK_ROOT}/mujoco_render_faces_gpu_${egl_index}.lock"
    --out "${video}"
    --logdir "${logdir}"
    --summary-path "${summary}"
  )
  if [ "${REAL2SIM_STRICT_LAUNCH}" = "1" ]; then
    local worker_contract="${outdir}/official_worker_contract.json"
    write_official_worker_contract "${worker_contract}" "${cmd[@]:${#SIM_WORKER[@]}}"
    cmd+=(--official-worker-contract "${worker_contract}")
  fi
  write_run_command "${outdir}" "${gpu_id}" "${PY}" "${cmd[@]}"

  echo "[real2sim-batch] start ${tag}/${run_tag} frame=0/${FRAMES}"
  if [ "${DRY_RUN}" = "1" ]; then
    printf '[real2sim-batch] worker command:'
    printf ' %q' "${PY}" "${cmd[@]}"
    printf '\n'
    return 0
  fi

  set +e
  local pipeline_status=()
  (
    set -euo pipefail
    export MUJOCO_GL=egl
    # The simulation child needs EGL, not CUDA. Hiding CUDA prevents it from
    # widening or reinterpreting the parent visibility list; EGL receives the
    # exact host physical index proven for this visible selector.
    export CUDA_VISIBLE_DEVICES=""
    export MUJOCO_EGL_DEVICE_ID="${egl_index}"
    export PYTHONNOUSERSITE=1
    export PYTHONUNBUFFERED=1
    export PYTHONHASHSEED="${DETERMINISTIC_SEED}"
    export DETERMINISTIC_SEED
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export_sim_thread_caps
    echo "[real2sim-batch:${tag}/${run_tag}] node=$(hostname) selector=${gpu_id} cuda_token=${cuda_token} egl=${MUJOCO_EGL_DEVICE_ID} start=$(date -Is)"
    "${PY}" "${cmd[@]}"
    if "${PY}" - "${summary}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    print(
        f"worker summary is unreadable after a successful simulator exit: "
        f"{summary_path}: {type(exc).__name__}: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc
if not isinstance(summary, dict):
    print(
        f"worker summary must contain a JSON object: {summary_path}",
        file=sys.stderr,
    )
    raise SystemExit(2)
if summary.get("wall_stopped_early") is True:
    raise SystemExit(124)
PY
    then
      :
    else
      summary_rc="$?"
      if [ "${summary_rc}" = "124" ]; then
        echo "[real2sim-batch:${tag}/${run_tag}] max wall time stopped this worker early; failing the wave to break the policy batch barrier" >&2
      else
        echo "[real2sim-batch:${tag}/${run_tag}] simulator exited successfully without a valid completion summary" >&2
      fi
      exit "${summary_rc}"
    fi
    echo "[real2sim-batch:${tag}/${run_tag}] sim done $(date -Is)"
    if [ "${RENDER_ENABLED}" = "1" ]; then
      "${PY}" robosuite_env/tools/validate_rollout_video.py \
        "${video}" --summary "${summary}" --output "${video_validation}"
      echo "[real2sim-batch:${tag}/${run_tag}] video validated $(date -Is)"
    else
      echo "[real2sim-batch:${tag}/${run_tag}] video validation skipped (--no-render)"
    fi
    if [ "${RUN_ANALYSIS}" = "1" ]; then
      "${PY}" robosuite_env/tools/plot_sim_trajectories.py "${logdir}" \
        --out-dir "${outdir}" \
        --prefix trajectory_analysis \
        > "${outdir}/trajectory_analysis_stdout.json"
      echo "[real2sim-batch:${tag}/${run_tag}] analysis done $(date -Is)"
    fi
  ) 2>&1 | tee "${stdout_log}"
  pipeline_status=("${PIPESTATUS[@]}")
  local rc="${pipeline_status[0]}"
  if [ "${pipeline_status[1]}" != "0" ]; then
    rc="${pipeline_status[1]}"
  fi
  set -e
  echo "${rc}" > "${exit_code}"
  echo "[real2sim-batch] done ${tag}/${run_tag} rc=${rc}"
  return "${rc}"
}

wait_for_worker_pids() {
  local fail_message="$1"
  local progress_prefix="${2:-}"
  local progress_mod="${3:-0}"
  local done_pid rc pid
  while [ "${#PIDS[@]}" -gt 0 ]; do
    set +e
    done_pid=""
    wait -n -p done_pid "${PIDS[@]}"
    rc=$?
    if [ -z "${done_pid:-}" ]; then  # wait -n cannot see a worker whose job entry bash already removed; wait <pid> still returns its saved status.
      done_pid="${PIDS[0]}"
      wait "${done_pid}"
      rc=$?
    fi
    set -e

    local next_pids=()
    for pid in "${PIDS[@]}"; do
      if [ "${pid}" != "${done_pid}" ]; then
        next_pids+=("${pid}")
      fi
    done
    PIDS=("${next_pids[@]}")

    if [ "${rc}" = "0" ]; then
      if [ "${progress_mod}" -gt 0 ] && [ $(( ${#PIDS[@]} % progress_mod )) -eq 0 ]; then
        echo "${progress_prefix} remaining_workers=${#PIDS[@]}"
        write_outputs_csv
      fi
      continue
    fi

    overall_rc=1
    echo "${fail_message}" >&2
    terminate_pids "remaining workers after wave failure" "${PIDS[@]}"
    PIDS=()
    return 1
  done
  return 0
}

write_outputs_csv() {
  {
    echo "scene,run,exit_code,video,summary,video_validation,trajectory_png,trajectory_10tracks_png,trajectory_mp4"
    local tag run_tag outdir rc video_validation
    local trajectory_png trajectory_10tracks_png trajectory_mp4
    for tag in "${TAGS[@]}"; do
      for repeat in $(seq "${REPEAT_START}" "${REPEAT_STOP}"); do
        run_tag="$(printf 'run_%02d' "${repeat}")"
        outdir="${RUN_ROOT}/${tag}/${run_tag}"
        rc="missing"
        [ -f "${outdir}/exit_code" ] && rc="$(cat "${outdir}/exit_code")"
        video_validation="${outdir}/video_validation.json"
        if [ "${POSTHOC_RENDER_REQUIRED}" = "1" ]; then
          video_validation="${outdir}/rerender/video_validation.json"
        fi
        trajectory_png=""
        trajectory_10tracks_png=""
        trajectory_mp4=""
        if [ "${RUN_ANALYSIS}" = "1" ]; then
          trajectory_png="${outdir}/trajectory_analysis.png"
          trajectory_10tracks_png="${outdir}/trajectory_analysis_10tracks.png"
          trajectory_mp4="${outdir}/trajectory_analysis.mp4"
        fi
        echo "${tag},${run_tag},${rc},${outdir}/${tag}_${run_tag}.mp4,${outdir}/sim_summary.json,${video_validation},${trajectory_png},${trajectory_10tracks_png},${trajectory_mp4}"
      done
    done
  } > "${OUTPUTS_CSV}"
}

publish_saved_policy_obs_videos() {
  # Wrist OBS insets are evidence of what the policy actually saw, so a
  # rollout video that anything tiles into a grid must carry the exact saved
  # 224px policy observation. Generation burns a presentation crop of that
  # tensor, so republish every rollout video from its own logs/policy_obs
  # before the grid renderer, which rejects any other wrist provenance.
  if [ "${RENDER_ENABLED}" != "1" ]; then
    echo "[real2sim-batch] saved policy OBS republish skipped: no inline render"
    return 0
  fi
  if [ "${POSTHOC_RENDER_REQUIRED}" = "1" ]; then
    echo "[real2sim-batch] saved policy OBS published by the post-hoc renderer"
    return 0
  fi
  echo "[real2sim-batch] republish rollout videos with the exact saved 224px policy OBS"
  local tag
  for tag in "${TAGS[@]}"; do
    if ! "${PY}" -m robosuite_env.tools.rewrite_saved_policy_obs_video \
      "${RUN_ROOT}" \
      --scene "${tag}" \
      > "${RUN_ROOT}/saved_policy_obs_${tag}.json"; then
      echo "[real2sim-batch] saved policy OBS republish failed for ${tag}" >&2
      return 1
    fi
  done
}

render_rollout_grid_template() {
  local total_rollouts=$((SCENE_COUNT * REPEATS))
  if [ "${RENDER_ENABLED}" != "1" ] \
    && [ "${POSTHOC_RENDER_REQUIRED}" != "1" ]; then
    echo "[real2sim-batch] grid skipped: simulation explicitly selected --no-render"
    return 0
  fi
  if [ "${POSTHOC_RENDER_REQUIRED}" = "1" ]; then
    # rerender_third_person builds and validates the same canonical grid after
    # it has finalized every individual video.
    echo "[real2sim-batch] grid already finalized by post-hoc renderer"
    return 0
  fi
  if [ "${REPEATS}" -ne 10 ] \
    && [ "${REPEATS}" -ne 20 ] \
    && [ "${REPEATS}" -ne 30 ] \
    && [ "${REPEATS}" -ne 100 ]; then
    echo "[real2sim-batch] grid skipped: no checked-in template for ${REPEATS} repeats per scene"
    return 0
  fi
  echo "[real2sim-batch] render grid template=${GRID_TEMPLATE}"
  local tag grid_root
  for tag in "${TAGS[@]}"; do
    grid_root="${RUN_ROOT}"
    if [ "${SCENE_COUNT}" -gt 1 ]; then
      grid_root="${RUN_ROOT}/${tag}"
    fi
    "${PY}" -m robosuite_env.tools.make_rollout_grid_video \
      "${RUN_ROOT}" \
      --scene "${tag}" \
      --title "${tag}" \
      --output "${grid_root}/rollout_grid_${REPEATS}.mp4" \
      --template "${GRID_TEMPLATE}" \
      --expected-frames "${FRAMES}" \
      > "${grid_root}/rollout_grid_${REPEATS}_stdout.json"
  done
}

initialize_runtime_source_manifest
discover_scenes
resolve_ports
read_scenes

echo "[real2sim-batch] root=${ROOT}"
echo "[real2sim-batch] run_root=${RUN_ROOT}"
echo "[real2sim-batch] robot=${ROBOT} scenes=${SCENE_COUNT} repeats=${REPEATS} parallel_repeats=${PARALLEL_REPEATS} policy_batch_size=${POLICY_BATCH_SIZE} frames=${FRAMES}"
if [ "${PARTIAL_REPEAT_RANGE}" = "1" ]; then
  echo "[real2sim-batch] repeat_range=${REPEAT_START}-${REPEAT_STOP}/${REPEATS} (absolute run numbering)"
fi
if [ "${REAL2SIM_AGGREGATE_ONLY}" = "1" ]; then
  echo "[real2sim-batch] aggregate_only=1 (no rollout services or workers)"
fi
if [ "${POSTHOC_RENDER_REQUIRED}" = "1" ]; then
  echo "[real2sim-batch] render_mode=posthoc generation=no-render"
  echo "[real2sim-batch] posthoc_command=${PY} -m robosuite_env.tools.rerender_third_person ${RUN_ROOT} --gpu ${SIM_GPU_IDS[0]}"
fi
echo "[real2sim-batch] gs_scene_dirs=${GS_SCENE_DIRS}"
echo "[real2sim-batch] policy config=${CONFIG}/${CONFIG_SRC} checkpoint=${CKPT}"
echo "[real2sim-batch] objects=${OBJECTS} instruction=${INSTRUCTION@Q}"
echo "[real2sim-batch] policy_image_episode=${POLICY_IMAGE_EPISODE:-default} goal=${GOAL_CONDITION} score=${SCORE_CONDITION}"
echo "[real2sim-batch] action_horizon=${ACTION_HORIZON} exec_steps=${EXEC_STEPS} rf_overlap=${RF_OVERLAP} rf_overlap_sampler=${RF_OVERLAP_SAMPLER} rf_overlap_prev_weight=${RF_OVERLAP_PREV_WEIGHT:-default} rf_overlap_step_shift=${RF_OVERLAP_STEP_SHIFT}"
echo "[real2sim-batch] grip_torque_mult=${GRIP_TORQUE_MULT} grasp_mode=${GRASP_MODE}"
echo "[real2sim-batch] gripper_calibration=${GRIPPER_CALIBRATION:-disabled}"
echo "[real2sim-batch] view az=${VIEW_AZIMUTH} el=${VIEW_ELEVATION} dist=${VIEW_DISTANCE} lookat=${VIEW_LOOKAT:-auto}"
echo "[real2sim-batch] render_wrist=${RENDER_WRIST} wrist_gs=${WRIST_GS} show_origin=${SHOW_ORIGIN}"
echo "[real2sim-batch] grid_template=${GRID_TEMPLATE} (mandatory for rendered 10/30/100-rollout runs)"
echo "[real2sim-batch] deterministic_sim=${DETERMINISTIC_SIM} deterministic_seed=${DETERMINISTIC_SEED} init_state_json=${INIT_STATE_JSON:-auto}"
echo "[real2sim-batch] mimicgen_init_dataset=${MIMICGEN_INIT_DATASET_ROOT:-disabled} state_logs=${MIMICGEN_INIT_STATE_LOG_ROOT:-disabled} episode_offset=${MIMICGEN_INIT_EPISODE_OFFSET}"
echo "[real2sim-batch] sim_num_threads=${SIM_NUM_THREADS} (OMP/MKL/OpenBLAS) gs_num_threads=${GS_NUM_THREADS} (OMP/MKL/OpenBLAS/NumExpr) gs_cuda_devices=${GS_CUDA_DEVICES} gs_servers_per_gpu=${GS_SERVERS_PER_GPU} policy_cuda_device=${POLICY_CUDA_DEVICE} sim_cuda_devices=${SIM_CUDA_DEVICES} render_lock_root=${RENDER_LOCK_ROOT}"
echo "[real2sim-batch] policy_noise_seed=${POLICY_NOISE_SEED:-disabled} policy_noise_bank=${POLICY_NOISE_BANK_DIR:-none} policy_inference_manifest=${POLICY_INFERENCE_MANIFEST:-none} deterministic_xla=${POLICY_DETERMINISTIC_XLA}"
echo "[real2sim-batch] scene manifest=${SCENES_RESOLVED}"
cat "${SCENES_RESOLVED}"
echo "[real2sim-batch] physics_contract=$(<"${PHYSICS_CONTRACT_FILE}")"
echo "[real2sim-batch] GS worker manifest=${GS_WORKERS_RESOLVED}"
cat "${GS_WORKERS_RESOLVED}"
echo "[real2sim-batch] GS server manifest=${GS_SERVERS_RESOLVED}"
cat "${GS_SERVERS_RESOLVED}"

overall_rc=0
if [ "${REAL2SIM_AGGREGATE_ONLY}" = "0" ]; then
  verify_runtime_source_manifest "service-launch"
  start_gs_servers
  INIT_STATE_JSON_BY_LAYOUT=()
  if [ "${#BOX_PACK_LAYOUTS[@]}" -gt 0 ]; then
    for layout_index in "${!BOX_PACK_LAYOUTS[@]}"; do
      INIT_STATE_JSON=""
      PREPARED_INIT_STATE_JSON=""
      prepare_deterministic_init_state \
        "${BOX_PACK_LAYOUTS[${layout_index}]}" "${layout_index}" 0
      INIT_STATE_JSON_BY_LAYOUT+=("${PREPARED_INIT_STATE_JSON}")
    done
    INIT_STATE_JSON=""
  else
    prepare_deterministic_init_state
  fi
  start_policy_server

  for wave_start in $(seq "${REPEAT_START}" "${PARALLEL_REPEATS}" "${REPEAT_STOP}"); do
    wave_end=$((wave_start + PARALLEL_REPEATS - 1))
    if [ "${wave_end}" -gt "${REPEAT_STOP}" ]; then
      wave_end="${REPEAT_STOP}"
    fi
    echo "[real2sim-batch] wave repeats=${wave_start}-${wave_end}/${REPEATS}"
    if ! verify_runtime_source_manifest "wave-${wave_start}-${wave_end}-before"; then
      overall_rc=1
      break
    fi
    PIDS=()
    for idx in "${!TAGS[@]}"; do
      for repeat in $(seq "${wave_start}" "${wave_end}"); do
        run_worker "${idx}" "${repeat}" &
        PIDS+=("$!")
      done
    done
    remaining="${#PIDS[@]}"
    wait_for_worker_pids "[real2sim-batch] worker failed; terminating wave to break policy batch barrier" || true
    for pid in "${PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
    if ! verify_runtime_source_manifest "wave-${wave_start}-${wave_end}-after"; then
      overall_rc=1
    fi
    write_outputs_csv
    if [ "${overall_rc}" != "0" ]; then
      break
    fi
  done
else
  write_outputs_csv
fi

# Every rollout has ended. Stop the policy and 3DGS servers before the
# finalization tools run (post-hoc render, saved policy OBS republish, grid):
# none of them uses the servers, the post-hoc renderer needs their GPU memory,
# and the servers' threads would otherwise share a container's task limit with
# the video tools. The EXIT trap still stops them on an earlier exit.
stop_generation_services

if [ "${overall_rc}" = "0" ] \
  && ! verify_runtime_source_manifest "post-rollouts"; then
  overall_rc=1
fi

if [ "${DRY_RUN}" = "0" ] && [ "${overall_rc}" = "0" ] \
  && [ "${RUN_ANALYSIS}" = "1" ] \
  && [ "${PARTIAL_REPEAT_RANGE}" = "0" ]; then
  "${PY}" robosuite_env/tools/plot_repeat_background_summary.py "${RUN_ROOT}" --out-dir "${RUN_ROOT}" \
    > "${RUN_ROOT}/repeat_background_summary_stdout.json" || true
fi

if [ "${DRY_RUN}" = "0" ] \
  && [ "${overall_rc}" = "0" ] \
  && [ "${PARTIAL_REPEAT_RANGE}" = "0" ] \
  && [ "${POSTHOC_RENDER_REQUIRED}" = "1" ]; then
  # The generation services were stopped above, so this serialized renderer
  # never competes with them for GPU memory.
  # The launcher owns the template: 10-rollout policy batches get the
  # policy-eval layout (checkpoint/instruction info panel), 30/100-rollout
  # batches get the matching clear-headers grid.
  if ! "${PY}" -m robosuite_env.tools.rerender_third_person \
    "${RUN_ROOT}" --gpu "${SIM_GPU_IDS[0]}" \
    --grid-template "${GRID_TEMPLATE}"; then
    echo "[real2sim-batch] required post-hoc video finalization failed" >&2
    overall_rc=1
  fi
fi

if [ "${DRY_RUN}" = "0" ] && [ "${overall_rc}" = "0" ] \
  && [ "${PARTIAL_REPEAT_RANGE}" = "0" ]; then
  if ! publish_saved_policy_obs_videos; then
    echo "[real2sim-batch] required saved policy OBS republish failed" >&2
    overall_rc=1
  fi
fi

if [ "${DRY_RUN}" = "0" ] && [ "${overall_rc}" = "0" ] \
  && [ "${PARTIAL_REPEAT_RANGE}" = "0" ]; then
  if ! verify_runtime_source_manifest "pre-grid"; then
    overall_rc=1
  fi
fi

if [ "${DRY_RUN}" = "0" ] && [ "${overall_rc}" = "0" ] \
  && [ "${PARTIAL_REPEAT_RANGE}" = "0" ]; then
  if ! render_rollout_grid_template; then
    echo "[real2sim-batch] required template grid render failed" >&2
    overall_rc=1
  fi
fi

if [ "${PARTIAL_REPEAT_RANGE}" = "1" ]; then
  echo "[real2sim-batch] partial repeat range: aggregate summary, post-hoc rendering, grid, poster, and grid manifest are skipped; whole-run finalization is deferred"
fi

if [ "${overall_rc}" = "0" ] \
  && ! verify_runtime_source_manifest "final"; then
  overall_rc=1
fi

write_outputs_csv
echo "[real2sim-batch] outputs=${OUTPUTS_CSV}"
exit "${overall_rc}"
