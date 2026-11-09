#!/usr/bin/env bash
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
# Reproduce the reviewed, hash-locked Python dependency profiles.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
UV_VERSION="0.12.3"
PYTHON_VERSION="$(tr -d '[:space:]' < "${ROOT}/.python-version")"
CUTOFF="2026-08-02T00:00:00Z"
PLATFORM="x86_64-manylinux_2_31"

[[ "${PYTHON_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
  echo "[locks] .python-version must contain exactly major.minor.patch" >&2
  exit 1
}

cd "${ROOT}"

command -v uv >/dev/null 2>&1 || {
  echo "[locks] uv ${UV_VERSION} is required" >&2
  exit 1
}
actual_uv="$(uv --version 2>/dev/null || true)"
read -r uv_name actual_uv_version _ <<< "${actual_uv}"
[ "${uv_name:-}" = "uv" ] && [ "${actual_uv_version:-}" = "${UV_VERSION}" ] || {
  echo "[locks] expected uv ${UV_VERSION}, got ${actual_uv:-unreadable}" >&2
  exit 1
}

compile_profile() {
  local input="$1"
  local output="$2"
  local temporary="${output}.tmp"
  local pass
  [ -f "${output}" ] || install -m 0644 /dev/null "${output}"
  for pass in 1 2 3; do
    uv pip compile "${input}" \
      --quiet \
      --constraint "${output}" \
      --python-version "${PYTHON_VERSION}" \
      --python-platform "${PLATFORM}" \
      --exclude-newer "${CUTOFF}" \
      --generate-hashes \
      --output-file "${temporary}"
    if cmp -s "${temporary}" "${output}"; then
      rm -f -- "${temporary}"
      echo "[locks] ${output} fixed on pass ${pass}"
      return 0
    fi
    mv -- "${temporary}" "${output}"
  done
  echo "${output} did not reach a fixed point after 3 passes" >&2
  return 1
}

# uv.lock is the complete runtime graph. Existing resolutions remain pinned;
# pyproject changes must be reviewed together with the generated lock diff.
uv lock --python "${PYTHON_VERSION}"
uv lock --check

# The small CI profile stays requirements-based so parent clean-checkout CI can
# test release tooling without downloading the multi-gigabyte CUDA graph.
compile_profile \
  "requirements-ci.in" \
  "requirements-lock.ci.txt"

echo "[locks] verified runtime uv.lock and regenerated the CI profile"
