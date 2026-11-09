#!/usr/bin/env bash
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
# Verify one exact Real2Sim commit from a real, isolated Git clone.

set -Eeuo pipefail

MODE=""
SOURCE=""
COMMIT=""
WORK_ROOT=""
TIMESTAMP=""
CONFIRM_FULL=0
MIN_SUCCESSES=""
RUN_ROOT=""
REPO=""
EVIDENCE=""
CURRENT_STEP="argument validation"
CURRENT_LOG=""

usage() {
  cat <<'EOF'
Usage:
  scripts/verify_clean_clone.sh \
    --source <git-url-or-path> \
    --commit <40-lowercase-hex-commit> \
    --work-root <absolute-existing-directory> \
    --mode fixture|full [options]

Modes:
  fixture  Real clone, CPU-only bootstrap, clean-checkout tests, fixture
           artifacts, CPU doctor, and fixture dry-run. No large assets or GPU.
  full     Everything in fixture mode plus the real host gate, full CUDA
           bootstrap, the environment-check profile's input install and its
           four artifacts, production doctor, profile dry-run, runtime check,
           and rollout.

Full-mode options:
  --confirm-full                 Required acknowledgement for download/rollout.
  --min-successes N              Required acceptance floor, from 1 through 10.

Common options:
  --timestamp YYYYMMDD_HHMMSS    Fix the resolved run name (default: now).
  -h, --help                     Show this help.

Run this inside an already allocated interactive shell. The verifier never
acquires resources. The caller's CUDA_VISIBLE_DEVICES is retained and recorded
in acceptance-result.json. GPU-free fixture and common CPU checks run with that
variable removed only from their child environment. Full host/bootstrap,
production doctor/profile/runtime checks, and the rollout inherit it unchanged.
EOF
}

on_error() {
  local rc=$?
  trap - ERR
  set +e
  printf '[clean-clone] FAILED step=%s rc=%s\n' "${CURRENT_STEP}" "${rc}" >&2
  if [ -n "${CURRENT_LOG}" ]; then
    printf '[clean-clone] command log: %s\n' "${CURRENT_LOG}" >&2
  fi
  if [ -n "${EVIDENCE}" ] && [ -d "${EVIDENCE}" ]; then
    {
      printf 'status=error\n'
      printf 'step=%s\n' "${CURRENT_STEP}"
      printf 'exit_code=%s\n' "${rc}"
      printf 'log=%s\n' "${CURRENT_LOG}"
    } > "${EVIDENCE}/failure.txt"
    printf '[clean-clone] retained run root: %s\n' "${RUN_ROOT}" >&2
    printf '[clean-clone] retained evidence: %s\n' "${EVIDENCE}" >&2
  fi
  exit "${rc}"
}
trap on_error ERR

die() {
  printf '[clean-clone] ERROR: %s\n' "$1" >&2
  return 2
}

run_logged() {
  local step="$1"
  local log_name="$2"
  local rc=0
  shift 2
  CURRENT_STEP="${step}"
  CURRENT_LOG="${EVIDENCE}/${log_name}"
  printf '[clean-clone] %s\n' "${step}"
  if "$@" >"${CURRENT_LOG}" 2>&1; then
    return 0
  else
    rc=$?
    printf '[clean-clone] command failed rc=%s (see %s)\n' \
      "${rc}" "${CURRENT_LOG}" >&2
    return "${rc}"
  fi
}

run_cpu_logged() {
  local step="$1"
  local log_name="$2"
  shift 2
  run_logged "${step}" "${log_name}" env -u CUDA_VISIBLE_DEVICES "$@"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source)
      [ "$#" -ge 2 ] || die "--source requires a value"
      SOURCE="$2"
      shift 2
      ;;
    --commit)
      [ "$#" -ge 2 ] || die "--commit requires a value"
      COMMIT="$2"
      shift 2
      ;;
    --work-root)
      [ "$#" -ge 2 ] || die "--work-root requires a value"
      WORK_ROOT="$2"
      shift 2
      ;;
    --mode)
      [ "$#" -ge 2 ] || die "--mode requires a value"
      MODE="$2"
      shift 2
      ;;
    --timestamp)
      [ "$#" -ge 2 ] || die "--timestamp requires a value"
      TIMESTAMP="$2"
      shift 2
      ;;
    --confirm-full)
      CONFIRM_FULL=1
      shift
      ;;
    --min-successes)
      [ "$#" -ge 2 ] || die "--min-successes requires a value"
      MIN_SUCCESSES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
done

[ -n "${SOURCE}" ] || die "--source is required"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || die "--commit must be exactly 40 lowercase hexadecimal characters"
[ "${MODE}" = "fixture" ] || [ "${MODE}" = "full" ] \
  || die "--mode must be fixture or full"
[ -n "${WORK_ROOT}" ] || die "--work-root is required"
[[ "${WORK_ROOT}" = /* ]] || die "--work-root must be absolute"
[ -d "${WORK_ROOT}" ] || die "--work-root must already exist"
[ -w "${WORK_ROOT}" ] && [ -x "${WORK_ROOT}" ] \
  || die "--work-root must be writable and searchable"
if [ "${MODE}" = "full" ]; then
  [ "${CONFIRM_FULL}" = "1" ] || die "full mode requires --confirm-full"
  [ -n "${MIN_SUCCESSES}" ] || die "full mode requires --min-successes N"
  [[ "${MIN_SUCCESSES}" =~ ^([1-9]|10)$ ]] \
    || die "--min-successes must be an integer from 1 through 10"
elif [ -n "${MIN_SUCCESSES}" ]; then
  die "--min-successes is valid only in full mode"
fi
if [ -z "${TIMESTAMP}" ]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
fi
[[ "${TIMESTAMP}" =~ ^[0-9]{8}_[0-9]{6}$ ]] \
  || die "--timestamp must match YYYYMMDD_HHMMSS"

WORK_ROOT="$(cd "${WORK_ROOT}" && pwd -P)"
RUN_ROOT="$(mktemp -d "${WORK_ROOT%/}/real2sim-acceptance.XXXXXXXX")"
REPO="${RUN_ROOT}/repo"
EVIDENCE="${RUN_ROOT}/evidence"
INPUTS="${RUN_ROOT}/inputs"
ISOLATED_HOME="${RUN_ROOT}/home"
mkdir -p "${EVIDENCE}" "${ISOLATED_HOME}"
CURRENT_LOG="${EVIDENCE}/argument-validation.log"

if [ -v CUDA_VISIBLE_DEVICES ]; then
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}" \
    > "${EVIDENCE}/accelerator-environment.txt"
else
  printf 'CUDA_VISIBLE_DEVICES=<unset>\n' \
    > "${EVIDENCE}/accelerator-environment.txt"
fi

# A parent Python environment must never select code or packages for the clone.
# The shell's accelerator visibility remains intact. run_cpu_logged removes it
# only for GPU-free child processes; production commands use run_logged.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
unset GIT_DIR GIT_WORK_TREE CDPATH
export PYTHONNOUSERSITE=1

run_logged "clone source repository" "clone.log" \
  git clone --no-local --no-checkout -- "${SOURCE}" "${REPO}"
run_logged "checkout exact detached commit" "checkout.log" \
  git -C "${REPO}" checkout --detach "${COMMIT}"
run_logged "read clone commit" "commit.log" \
  git -C "${REPO}" rev-parse HEAD
IFS= read -r OBSERVED_COMMIT < "${EVIDENCE}/commit.log"
[ "${OBSERVED_COMMIT}" = "${COMMIT}" ] \
  || die "clone resolved ${OBSERVED_COMMIT}, expected ${COMMIT}"
run_logged "check target commit patch whitespace" "commit-diff-check.log" \
  git -C "${REPO}" diff-tree --check --root -m --no-commit-id -r "${COMMIT}"
run_logged "verify initial clone cleanliness" "initial-git-status.log" \
  git -C "${REPO}" status --porcelain=v1 --untracked-files=all
[ ! -s "${EVIDENCE}/initial-git-status.log" ] \
  || die "fresh clone has tracked or untracked changes"
[ -e "${REPO}/.git" ] || die "clone has no Git metadata"
[ -f "${REPO}/scripts/bootstrap.sh" ] || die "clone is missing scripts/bootstrap.sh"
[ -f "${REPO}/scripts/validate_clean_clone_acceptance.py" ] \
  || die "selected commit does not contain the acceptance validator"

# Isolate all bootstrap/runtime caches after the source clone. This happens
# after cloning so an explicitly supplied authenticated source transport may
# still use the caller's normal transport configuration.
export HOME="${ISOLATED_HOME}"
cd "${REPO}"

if [ "${MODE}" = "fixture" ]; then
  run_cpu_logged "bootstrap CPU fixture environment" "bootstrap.log" \
    "${REPO}/scripts/bootstrap.sh" --skip-cuda-extensions
else
  run_logged "verify GPU host prerequisites" "host-check.log" \
    "${REPO}/scripts/check_host.sh"
  run_logged "bootstrap full locked runtime" "bootstrap.log" \
    "${REPO}/scripts/bootstrap.sh"
fi

PY="${REPO}/.venv/bin/python"
ARTIFACTS="${REPO}/.venv/bin/real2sim-artifacts"
DOCTOR="${REPO}/.venv/bin/real2sim-doctor"
RUN="${REPO}/.venv/bin/real2sim-run"
DOWNLOAD="${REPO}/.venv/bin/real2sim-download-release"
VALIDATOR="${REPO}/scripts/validate_clean_clone_acceptance.py"
for executable in "${PY}" "${ARTIFACTS}" "${DOCTOR}" "${RUN}" "${DOWNLOAD}"; do
  [ -x "${executable}" ] || die "bootstrap did not create clone-local executable: ${executable}"
done

run_cpu_logged "verify clone-local Python and imports" "python-identity.log" \
  "${PY}" "${VALIDATOR}" identity --repo-root "${REPO}"

# This is the clean-checkout CPU contract. Every command uses the clone's
# interpreter, and every log stays in the sibling evidence directory.
run_cpu_logged "compile source tree" "compileall.log" \
  "${PY}" -m compileall -q src scripts robosuite_env scene
run_cpu_logged "build source distribution and wheel" "build.log" \
  "${PY}" -m build --no-isolation
run_cpu_logged "run CPU clean-checkout tests" "pytest.log" \
  "${PY}" -m pytest -q -m "not gpu and not heavy and not assets" -p no:cacheprovider
run_cpu_logged "check shell syntax" "shell-syntax.log" \
  bash -n scripts/*.sh

FIXTURE_IDS=(
  tiny-checkpoint-v1
  tiny-scene-v2
  tiny-registered-desk-v1
  tiny-policy-episode-v1
  tiny-repository-assets-v1
)
fixture_artifact_args=()
fixture_expected_args=()
for artifact_id in "${FIXTURE_IDS[@]}"; do
  fixture_artifact_args+=(--artifact "${artifact_id}")
  fixture_expected_args+=(--expected "${artifact_id}")
done
run_cpu_logged "verify fixture artifacts" "fixture-artifacts.json" \
  "${ARTIFACTS}" --manifest artifacts/manifest.yaml --repo-root "${REPO}" \
    "${fixture_artifact_args[@]}" --json
run_cpu_logged "validate fixture artifact JSON" "fixture-artifacts-validation.log" \
  "${PY}" "${VALIDATOR}" artifacts \
    --input "${EVIDENCE}/fixture-artifacts.json" "${fixture_expected_args[@]}"
run_cpu_logged "run CPU fixture doctor" "fixture-doctor.json" \
  "${DOCTOR}" --repo-root "${REPO}" \
    --manifest "${REPO}/artifacts/manifest.yaml" \
    --support-matrix "${REPO}/config/support-matrix.yaml" \
    --profile cpu-fixture \
    --output-path "${RUN_ROOT}/fixture-doctor-output" \
    --min-free-gib 0.01 \
    --lockfile "${REPO}/requirements-lock.ci.txt" \
    "${fixture_artifact_args[@]}" --json
run_cpu_logged "validate fixture doctor JSON" "fixture-doctor-validation.log" \
  "${PY}" "${VALIDATOR}" status \
    --input "${EVIDENCE}/fixture-doctor.json" --kind doctor
run_cpu_logged "resolve fixture profile" "fixture-dry-run.json" \
  "${RUN}" --config config/release/fixture_v2.yaml \
    --timestamp "${TIMESTAMP}" --dry-run
run_cpu_logged "validate fixture dry-run JSON" "fixture-dry-run-validation.log" \
  "${PY}" "${VALIDATOR}" dry-run \
    --input "${EVIDENCE}/fixture-dry-run.json" \
    --repo-root "${REPO}" --mode fixture --expected-rollouts 1 \
    --run-root-output "${EVIDENCE}/fixture-run-root.txt"
run_logged "verify post-bootstrap Git cleanliness" "post-bootstrap-git-status.log" \
  git status --porcelain=v1 --untracked-files=all
[ ! -s "${EVIDENCE}/post-bootstrap-git-status.log" ] \
  || die "bootstrap or CPU checks dirtied the clone"

FINAL_RUN_ROOT=""
if [ "${MODE}" = "full" ]; then
  # The environment-check profile is the production profile a clean clone
  # runs. Its installer fetches exactly the inputs that profile binds.
  RELEASE_CONFIG="config/release/fr3/flat/a_room_cup_cup1trim.yaml"
  run_cpu_logged "preview the profile's input installation" "archive-plan.log" \
    "${DOWNLOAD}" --profile "${RELEASE_CONFIG}" \
      --repo-root "${REPO}" --inputs-root "${INPUTS}" --no-env --dry-run
  run_cpu_logged "download, verify, and install the profile's inputs" "archive-install.log" \
    "${DOWNLOAD}" --profile "${RELEASE_CONFIG}" \
      --repo-root "${REPO}" --inputs-root "${INPUTS}" --no-env
  export REAL2SIM_INPUTS_ROOT="${INPUTS}"

  RELEASE_IDS=(
    pi05-cup
    public-scene-flat-a-room-v2
    public-cup-plate-visual-v1
    fr3-collision-franka-description-2fd6aaa8
  )
  release_artifact_args=()
  release_expected_args=()
  for artifact_id in "${RELEASE_IDS[@]}"; do
    release_artifact_args+=(--artifact "${artifact_id}")
    release_expected_args+=(--expected "${artifact_id}")
  done
  run_cpu_logged "verify the profile's four artifacts" "release-artifacts.json" \
    "${ARTIFACTS}" --manifest artifacts/manifest.yaml --repo-root "${REPO}" \
      "${release_artifact_args[@]}" --json
  run_cpu_logged "validate release artifact JSON" "release-artifacts-validation.log" \
    "${PY}" "${VALIDATOR}" artifacts \
      --input "${EVIDENCE}/release-artifacts.json" "${release_expected_args[@]}"
  run_logged "verify post-install Git cleanliness" "post-install-git-status.log" \
    git status --porcelain=v1 --untracked-files=all
  [ ! -s "${EVIDENCE}/post-install-git-status.log" ] \
    || die "public asset installation dirtied the clone"

  run_logged "run production doctor" "production-doctor.json" \
    "${DOCTOR}" --repo-root "${REPO}" \
      --manifest "${REPO}/artifacts/manifest.yaml" \
      --support-matrix "${REPO}/config/support-matrix.yaml" \
      --profile cuda12-x86_64 --production \
      --output-path "${RUN_ROOT}/production-doctor-output" \
      --min-free-gib 0.01 --lockfile "${REPO}/uv.lock" \
      "${release_artifact_args[@]}" --json
  run_cpu_logged "validate production doctor JSON" "production-doctor-validation.log" \
    "${PY}" "${VALIDATOR}" status \
      --input "${EVIDENCE}/production-doctor.json" --kind doctor

  run_logged "resolve release profile" "release-dry-run.json" \
    "${RUN}" --config "${RELEASE_CONFIG}" \
      --timestamp "${TIMESTAMP}" --dry-run
  run_cpu_logged "validate release dry-run JSON" "release-dry-run-validation.log" \
    "${PY}" "${VALIDATOR}" dry-run \
      --input "${EVIDENCE}/release-dry-run.json" \
      --repo-root "${REPO}" --mode production --expected-rollouts 10 \
      --run-root-output "${EVIDENCE}/release-run-root.txt"
  IFS= read -r FINAL_RUN_ROOT < "${EVIDENCE}/release-run-root.txt"

  run_logged "check every release runtime input" "runtime-check.json" \
    "${RUN}" --config "${RELEASE_CONFIG}" \
      --timestamp "${TIMESTAMP}" --check-runtime
  run_cpu_logged "validate runtime-check JSON" "runtime-check-validation.log" \
    "${PY}" "${VALIDATOR}" status \
      --input "${EVIDENCE}/runtime-check.json" --kind runtime \
      --repo-root "${REPO}"
  run_logged "execute accepted release rollout" "rollout.log" \
    "${RUN}" --config "${RELEASE_CONFIG}" --timestamp "${TIMESTAMP}"
  run_cpu_logged "validate and copy completed-run evidence" "run-evidence-validation.log" \
    "${PY}" "${VALIDATOR}" summary \
      --run-root "${FINAL_RUN_ROOT}" --expected-rollouts 10 \
      --min-successes "${MIN_SUCCESSES}" --evidence-output "${EVIDENCE}"
fi

result_args=(
  result
  --output "${EVIDENCE}/acceptance-result.json"
  --mode "${MODE}"
  --commit "${COMMIT}"
  --repo-root "${REPO}"
)
if [ -n "${FINAL_RUN_ROOT}" ]; then
  result_args+=(
    --run-root "${FINAL_RUN_ROOT}"
    --min-successes "${MIN_SUCCESSES}"
  )
fi
run_logged "write final acceptance result" "result-validation.log" \
  "${PY}" "${VALIDATOR}" "${result_args[@]}"

CURRENT_STEP="complete"
CURRENT_LOG=""
printf '[clean-clone] PASS mode=%s commit=%s\n' "${MODE}" "${COMMIT}"
printf '[clean-clone] retained run root: %s\n' "${RUN_ROOT}"
printf '[clean-clone] evidence: %s\n' "${EVIDENCE}"
