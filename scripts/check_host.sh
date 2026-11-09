#!/usr/bin/env bash
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
# Real2Sim GPU host preflight. Run before bootstrap downloads dependencies.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PROFILE_HELPER="${ROOT}/src/real2sim/support_matrix.py"
EXPECTED_UV="0.12.3"
MIN_GLIBC="2.31"
failures=0
declare -a blockers=()
declare -a actions=()

ok() {
  printf '[host-check] ok: %s\n' "$1"
}

fail() {
  local message="$1"
  local action="${2:-Correct this requirement and rerun ./scripts/check_host.sh.}"
  printf '[host-check] ERROR: %s\n' "${message}" >&2
  blockers+=("${message}")
  actions+=("${action}")
  failures=$((failures + 1))
}

blocked_report() {
  local index=0
  printf '%s\n' \
    "[host-check] DECISION: STOP" \
    "[host-check] NEXT_COMMAND: NONE" \
    "[host-check] EXIT_CODE: 1" \
    "[host-check] RETURN_TO: HOST_ADMIN_OR_SETUP_REQUESTER" \
    "[host-check] RETURN_WHAT: COPY_REPORT_BEGIN..COPY_REPORT_END" \
    "[host-check] COPY_REPORT_BEGIN" \
    "Real2Sim environment check: BLOCKED" \
    "Kernel: $(uname -r)" \
    "Do not run ./scripts/bootstrap.sh yet." \
    >&2
  for index in "${!blockers[@]}"; do
    printf '  %d. BLOCKER: %s\n' "$((index + 1))" "${blockers[index]}" >&2
    printf '     REQUIRED_ACTION: %s\n' "${actions[index]}" >&2
  done
  printf '%s\n' \
    "After every blocker is corrected, rerun: ./scripts/check_host.sh" \
    "Continue only when it prints DECISION: PROCEED." \
    "[host-check] COPY_REPORT_END" \
    >&2
}

version_at_least() {
  [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)" = "$2" ]
}

if ! profile_settings="$(python3 "${PROFILE_HELPER}" shell)"; then
  fail "could not read the support matrix" "Install system Python 3.9+ and restore config/support-matrix.yaml from Git."
  blocked_report
  exit 1
fi
eval "${profile_settings}"
EXPECTED_ARCH="${R2S_ARCHITECTURE}"
EXPECTED_CUDA="${R2S_CUDA_VERSION}"
EXPECTED_COMPILER_MAJOR="${R2S_COMPILER_MAJOR}"
MIN_DRIVER="${R2S_MIN_DRIVER}"

actual_arch="$(uname -m)"
if [ "${actual_arch}" = "${EXPECTED_ARCH}" ]; then
  ok "architecture ${actual_arch}"
else
  fail \
    "expected ${EXPECTED_ARCH}, found ${actual_arch}" \
    "Use an x86_64 Linux host; this repository does not support the detected architecture."
fi

glibc_version="$(ldd --version 2>&1 | head -n 1 | awk '{print $NF}')"
if [ -n "${glibc_version}" ] && version_at_least "${glibc_version}" "${MIN_GLIBC}"; then
  ok "glibc ${glibc_version} (minimum ${MIN_GLIBC})"
else
  fail \
    "expected glibc ${MIN_GLIBC} or newer, found ${glibc_version:-unreadable}" \
    "Use a supported Linux distribution with glibc ${MIN_GLIBC} or newer."
fi

if command -v git >/dev/null 2>&1; then
  ok "git $(git --version | awk '{print $3}')"
else
  fail "git is required" "Install Git using the host package manager."
fi
if command -v curl >/dev/null 2>&1; then
  ok "curl is available"
else
  fail "curl is required" "Install curl using the host package manager."
fi
if command -v uv >/dev/null 2>&1; then
  actual_uv="$(uv --version 2>/dev/null || true)"
  read -r uv_name actual_uv_version _ <<< "${actual_uv}"
  if [ "${uv_name:-}" = "uv" ] && [ "${actual_uv_version:-}" = "${EXPECTED_UV}" ]; then
    ok "${actual_uv}"
  else
    fail \
      "expected uv ${EXPECTED_UV}, found ${actual_uv:-unreadable}" \
      "Install the pinned uv ${EXPECTED_UV} release using the verified runbook command."
  fi
else
  fail \
    "uv ${EXPECTED_UV} is required" \
    "Install the pinned uv ${EXPECTED_UV} release using the verified runbook command."
fi

nvidia_output="$(nvidia-smi 2>&1)"
nvidia_status=$?
if [ "${nvidia_status}" -ne 0 ]; then
  if [[ "${nvidia_output}" == *"Driver/library version mismatch"* ]]; then
    kernel_driver="$(cat /sys/module/nvidia/version 2>/dev/null || true)"
    nvml_library="$(printf '%s\n' "${nvidia_output}" \
      | sed -n 's/^NVML library version:[[:space:]]*//p' \
      | head -n 1)"
    installed_driver="$(modinfo -F version nvidia 2>/dev/null | head -n 1 || true)"
    fail \
      "NVIDIA driver/NVML mismatch: loaded kernel=${kernel_driver:-unknown}, userspace NVML=${nvml_library:-unknown}, installed module=${installed_driver:-unknown}" \
      "Ask the administrator to align the NVIDIA kernel module and userspace packages. If the installed module matches NVML after a recent upgrade, schedule a safe reboot; otherwise repair the driver/DKMS installation."
  else
    fail \
      "nvidia-smi is unusable: ${nvidia_output//$'\n'/; }" \
      "Ask the administrator to make nvidia-smi operational before bootstrapping Real2Sim."
  fi
else
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits \
    2>/dev/null | head -n 1 | tr -d '[:space:]')"
  if [ -n "${driver_version}" ] && version_at_least "${driver_version}" "${MIN_DRIVER}"; then
    ok "NVIDIA driver ${driver_version} (minimum ${MIN_DRIVER})"
  else
    fail \
      "NVIDIA driver ${driver_version:-unknown} is older than ${MIN_DRIVER}" \
      "Ask the administrator to install a supported NVIDIA driver (${MIN_DRIVER} or newer) and reboot safely."
  fi
  # Report contention without guessing a universal policy/scene memory budget.
  printf '[host-check] GPU memory: index, name, total MiB, used MiB, free MiB (all host GPUs)\n'
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits || true
  printf '[host-check] GPU compute processes: GPU UUID, PID, name, used MiB\n'
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits || true
fi

nvcc="$(python3 "${PROFILE_HELPER}" nvcc 2>/dev/null || true)"
if [ -z "${nvcc}" ]; then
  fail \
    "CUDA toolkit ${EXPECTED_CUDA} nvcc is missing" \
    "Install/select NVIDIA's toolkit-only cuda-toolkit-${EXPECTED_CUDA/./-} package and correct CUDA_HOME if set; do not use a driver-installing cuda or cuda-runtime meta-package."
else
  nvcc_release="$("${nvcc}" --version 2>/dev/null \
    | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
    | tail -n 1)"
  if [ "${nvcc_release}" = "${EXPECTED_CUDA}" ]; then
    ok "CUDA toolkit ${nvcc_release} (${nvcc})"
  else
    fail \
      "expected CUDA toolkit ${EXPECTED_CUDA}, found ${nvcc_release:-unreadable} at ${nvcc}" \
      "Install or select CUDA toolkit ${EXPECTED_CUDA}; set CUDA_HOME to that toolkit before rerunning the check."
  fi
fi

for compiler_kind in "${R2S_CC}" "${R2S_CXX}"; do
  compiler_selection="${CC:-${R2S_CC}}"
  [ "${compiler_kind}" != "${R2S_CXX}" ] || compiler_selection="${CXX:-${R2S_CXX}}"
  compiler="$(command -v "${compiler_selection}" || true)"
  if [ -z "${compiler}" ]; then
    fail \
      "${compiler_kind} major ${EXPECTED_COMPILER_MAJOR} is required" \
      "Install ${compiler_kind} major ${EXPECTED_COMPILER_MAJOR} with the host package manager."
    continue
  fi
  compiler_major="$("${compiler}" -dumpversion | cut -d. -f1)"
  if [ "${compiler_major}" = "${EXPECTED_COMPILER_MAJOR}" ]; then
    ok "${compiler_kind} $("${compiler}" -dumpfullversion) (${compiler})"
  else
    fail \
      "expected ${compiler_kind} major ${EXPECTED_COMPILER_MAJOR}, found $("${compiler}" -dumpfullversion) at ${compiler}" \
      "Install ${compiler_kind} major ${EXPECTED_COMPILER_MAJOR} and ensure it is on PATH."
  fi
done

if [ "${failures}" -ne 0 ]; then
  printf '[host-check] FAILED: %d blocking host requirement(s)\n' "${failures}" >&2
  blocked_report
  exit 1
fi
printf '%s\n' \
  "[host-check] READY: host can bootstrap the locked GPU environment" \
  "[host-check] DECISION: PROCEED" \
  "[host-check] NEXT_COMMAND: ./scripts/bootstrap.sh" \
  "[host-check] EXIT_CODE: 0"
