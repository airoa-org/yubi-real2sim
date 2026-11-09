#!/usr/bin/env bash
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
# Bootstrap pinned public source dependencies and the Python environment.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
THIRD_PARTY="${ROOT}/third_party"
UV_VERSION="0.12.3"
PYTHON_VERSION="$(tr -d '[:space:]' < "${ROOT}/.python-version")"
PROFILE_HELPER="${ROOT}/src/real2sim/support_matrix.py"

[[ "${PYTHON_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
  echo "[bootstrap] .python-version must contain exactly major.minor.patch" >&2
  exit 1
}

find_exact_python() {
  local interpreter=""

  if [ "${DRY_RUN}" = "1" ]; then
    printf '<exact-CPython-%s>\n' "${PYTHON_VERSION}"
    return
  fi
  uv python install "${PYTHON_VERSION}" --no-bin
  interpreter="$(uv python find "${PYTHON_VERSION}" \
    --managed-python --resolve-links --no-project 2>/dev/null || true)"
  [ -n "${interpreter}" ] || {
    echo "[bootstrap] exact CPython ${PYTHON_VERSION} is unavailable" >&2
    exit 1
  }
  "${interpreter}" - "${PYTHON_VERSION}" <<'PY'
import platform
import sys

expected = sys.argv[1]
actual = platform.python_version()
raise SystemExit(0 if actual == expected else f"expected Python {expected}, got {actual}")
PY
  printf '%s\n' "${interpreter}"
}

pin_venv_base_python() {
  local interpreter="$1"
  local interpreter_home=""
  local metadata="${ROOT}/.venv/pyvenv.cfg"

  interpreter="$(readlink -f "${interpreter}")"
  interpreter_home="$(cd "$(dirname "${interpreter}")" && pwd -P)"
  [ -f "${metadata}" ] || {
    echo "[bootstrap] uv did not create ${metadata}" >&2
    exit 1
  }
  ln -sfn "${interpreter}" "${ROOT}/.venv/bin/python"
  sed -i "s#^home = .*#home = ${interpreter_home}#" "${metadata}"
  grep -Fxq "home = ${interpreter_home}" "${metadata}" || {
    echo "[bootstrap] could not pin the venv base interpreter" >&2
    exit 1
  }
}

FRANKA_DESCRIPTION_URL="${FRANKA_DESCRIPTION_URL:-https://github.com/frankarobotics/franka_description.git}"
FRANKA_DESCRIPTION_REF="${FRANKA_DESCRIPTION_REF:-2fd6aaa8c5e3bbfe857b8fb19978694cdc41e6ea}"
[ -z "${GAUSSIAN_WRAPPING_URL:-}${GAUSSIAN_WRAPPING_REF:-}" ] || {
  echo "[bootstrap] GAUSSIAN_WRAPPING_URL/REF are not read: the third_party/GaussianWrapping gitlink is the pin" >&2
  exit 1
}

DRY_RUN=0
UPDATE=0
SKIP_PYTHON=0
SKIP_CUDA_EXTENSIONS=0

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap.sh [options]

Options:
  --skip-python       Fetch sources and public robot assets only.
  --skip-cuda-extensions
                       Sync without the cuda-extensions dependency group.
  --update            Fetch pinned refs from configured URLs in existing clones.
  --dry-run           Print operations without changing the workspace.
  -h, --help          Show this help.

Environment overrides:
  FRANKA_DESCRIPTION_URL, FRANKA_DESCRIPTION_REF
  CUDA_HOME, TORCH_CUDA_ARCH_LIST, MAX_JOBS, CC, CXX
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip-python) SKIP_PYTHON=1; shift ;;
    --skip-cuda-extensions) SKIP_CUDA_EXTENSIONS=1; shift ;;
    --update) UPDATE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[bootstrap] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

run() {
  if [ "${DRY_RUN}" = "1" ]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [ "${DRY_RUN}" = "0" ] \
  && [ "${SKIP_PYTHON}" = "0" ] \
  && [ "${SKIP_CUDA_EXTENSIONS}" = "0" ]; then
  if ! "${ROOT}/scripts/check_host.sh"; then
    echo "[bootstrap] stopped before .venv and portable tools were installed" >&2
    echo "[bootstrap] fix every host-check error, then rerun scripts/bootstrap.sh" >&2
    exit 1
  fi
fi

profile_settings="$(python3 "${PROFILE_HELPER}" shell)"
eval "${profile_settings}"
export CC="${CC:-${R2S_CC}}" CXX="${CXX:-${R2S_CXX}}"  # the compilers check_host.sh accepted
if [ "${SKIP_CUDA_EXTENSIONS}" = "1" ]; then
  export UV_NO_GROUP=cuda-extensions
elif [ "${DRY_RUN}" = "0" ] && [ "${SKIP_PYTHON}" = "0" ]; then
  nvcc="$(python3 "${PROFILE_HELPER}" nvcc)" || exit 1
  CUDA_HOME="$(cd "$(dirname "${nvcc}")/.." && pwd -P)" || exit 1
  export CUDA_HOME  # torch would otherwise fall back to /usr/local/cuda, not the toolkit check_host.sh accepted
  [ ! -L "${THIRD_PARTY}/GaussianWrapping" ] || {
    echo "[bootstrap] third_party/GaussianWrapping must be a directory, not a symlink: the extension build finds scripts/ from its resolved path" >&2
    exit 1
  }
fi

# These are source-only checkouts; runtime assets come from the verified release downloads.
# LeRobot carries LFS media that is not needed to build its wheel. Disable the
# filter in child Git processes (including uv's), even on hosts with a stale
# global LFS filter but no git-lfs binary. Preserve existing command-scope config.
git_config_count="${GIT_CONFIG_COUNT:-0}"
for git_lfs_setting in process= smudge=cat clean=cat required=false; do
  export "GIT_CONFIG_KEY_${git_config_count}=filter.lfs.${git_lfs_setting%%=*}"
  export "GIT_CONFIG_VALUE_${git_config_count}=${git_lfs_setting#*=}"
  git_config_count=$((git_config_count + 1))
done
export GIT_CONFIG_COUNT="${git_config_count}"

check_source_clean() {
  local path="$1"
  local name="$2"
  git -C "${path}" diff --quiet --ignore-submodules -- \
    && git -C "${path}" diff --cached --quiet --ignore-submodules -- || {
      echo "[bootstrap] ${name} has tracked local changes" >&2
      exit 1
    }
}

# The superproject gitlink is the only GaussianWrapping pin. Its nested
# submodules (nvdiffrast, Depth-Anything-V2, Geometry-Grounded-Gaussian-Splatting,
# TNTUniScanEvals) are not fetched: the CUDA build and the GS server use only
# directories tracked in GaussianWrapping itself (THIRD_PARTY_NOTICES B.1).
checkout_gaussian_wrapping() {
  local path="${THIRD_PARTY}/GaussianWrapping"
  local pin=""

  if [ "${DRY_RUN}" = "0" ] && [ -e "${path}/.git" ]; then
    check_source_clean "${path}" "GaussianWrapping"
  fi
  run git -C "${ROOT}" submodule init -- third_party/GaussianWrapping
  run git -C "${ROOT}" submodule sync -- third_party/GaussianWrapping  # repoints a clone made before the submodule
  run git -C "${ROOT}" submodule update --init -- third_party/GaussianWrapping
  [ "${DRY_RUN}" = "0" ] || return 0
  pin="$(git -C "${ROOT}" rev-parse ":third_party/GaussianWrapping")"
  [ "$(git -C "${path}" rev-parse HEAD)" = "${pin}" ] || {
    echo "[bootstrap] GaussianWrapping did not resolve to its gitlink ${pin}" >&2
    exit 1
  }
  check_source_clean "${path}" "GaussianWrapping"
}

clone_checkout() {
  local name="$1"
  local url="$2"
  local ref="$3"
  local path="${THIRD_PARTY}/${name}"

  if [ -L "${path}" ]; then
    if [ "${DRY_RUN}" = "1" ]; then
      echo "[dry-run] replace legacy symlink before cloning: ${path}"
      run git clone "${url}" "${path}"
      run git -C "${path}" checkout --detach "${ref}"
      return
    else
      echo "[bootstrap] ${path} is a legacy symlink; move it away before bootstrapping" >&2
      exit 1
    fi
  fi
  # rev-parse alone walks up to the parent repository for an empty directory
  # (including an uninitialized submodule). Require this checkout's own .git.
  if [ -e "${path}" ] && [ ! -e "${path}/.git" ]; then
    echo "[bootstrap] ${path} exists but is not a Git clone" >&2
    exit 1
  fi
  if [ "${DRY_RUN}" = "0" ] && [ -e "${path}/.git" ]; then
    check_source_clean "${path}" "${name}"
  fi
  if [ ! -e "${path}/.git" ]; then
    run git clone "${url}" "${path}"
  elif [ "${UPDATE}" = "1" ]; then
    run git -C "${path}" fetch --no-tags "${url}" "${ref}"
  fi
  run git -C "${path}" checkout --detach "${ref}"
  if [ "${DRY_RUN}" = "0" ]; then
    [ "$(git -C "${path}" rev-parse HEAD)" = "${ref}" ] || {
      echo "[bootstrap] ${name} did not resolve to ${ref}" >&2
      exit 1
    }
    check_source_clean "${path}" "${name}"
  fi
}

python_environment_is_exact() {
  "${ROOT}/.venv/bin/python" - "${PYTHON_VERSION}" "${ROOT}/.venv" <<'PY'
import platform
from pathlib import Path
import sys

expected = sys.argv[1]
expected_prefix = Path(sys.argv[2]).resolve()
actual = platform.python_version()
actual_prefix = Path(sys.prefix).resolve()
base_prefix = Path(sys.base_prefix).resolve()
ok = (
    actual == expected
    and actual_prefix == expected_prefix
    and actual_prefix != base_prefix
)
raise SystemExit(
    0
    if ok
    else (
        f"expected Python {expected} virtual environment at {expected_prefix}, "
        f"got Python {actual} with prefix {actual_prefix} and base {base_prefix}"
    )
)
PY
}

run mkdir -p "${THIRD_PARTY}"
checkout_gaussian_wrapping
clone_checkout "franka_description" "${FRANKA_DESCRIPTION_URL}" "${FRANKA_DESCRIPTION_REF}"

if [ "${SKIP_PYTHON}" = "0" ]; then
  exact_python=""
  command -v uv >/dev/null 2>&1 || {
    echo "[bootstrap] uv ${UV_VERSION} is required: https://docs.astral.sh/uv/" >&2
    exit 1
  }
  actual_uv="$(uv --version 2>/dev/null || true)"
  read -r uv_name actual_uv_version _ <<< "${actual_uv}"
  [ "${uv_name:-}" = "uv" ] && [ "${actual_uv_version:-}" = "${UV_VERSION}" ] || {
    echo "[bootstrap] expected uv ${UV_VERSION}, got ${actual_uv:-unreadable}" >&2
    exit 1
  }
  exact_python="$(find_exact_python)"
  if [ ! -x "${ROOT}/.venv/bin/python" ]; then
    run uv venv "${ROOT}/.venv" --python "${exact_python}"
    if [ "${DRY_RUN}" = "0" ]; then
      pin_venv_base_python "${exact_python}"
    fi
  elif [ "${DRY_RUN}" = "0" ]; then
    python_environment_is_exact || {
      echo "[bootstrap] move the existing .venv aside and rerun bootstrap" >&2
      exit 1
    }
  fi
  run env UV_PROJECT_ENVIRONMENT="${ROOT}/.venv" \
    uv sync --project "${ROOT}" --frozen --python "${ROOT}/.venv/bin/python" \
      --no-install-project \
      --no-install-package openpi \
      --no-install-package openpi-client \
      --no-install-package lerobot \
      --no-install-package robosuite \
      --no-install-package diff-gaussian-rasterization \
      --no-install-package diff-gaussian-rasterization-ours \
      --no-install-package simple-knn \
      --inexact  # keeps the installed extensions; reinstalling them needs uv's cached wheels or a rebuild
  reinstall_cuda_extensions=()
  if [ "${SKIP_CUDA_EXTENSIONS}" = "0" ] && [ "${DRY_RUN}" = "0" ]; then
    if ! "${ROOT}/.venv/bin/python" "${ROOT}/scripts/cuda_extension_build/inputs.py" "${ROOT}" --check; then
      reinstall_cuda_extensions=(
        --reinstall-package diff-gaussian-rasterization
        --reinstall-package diff-gaussian-rasterization-ours
        --reinstall-package simple-knn
      )
    fi
  fi
  # With ninja, each extension build runs MAX_JOBS compilers, so concurrent builds can exhaust host memory.
  run env UV_PROJECT_ENVIRONMENT="${ROOT}/.venv" UV_CONCURRENT_BUILDS=1 MAX_JOBS="${MAX_JOBS:-4}" \
    uv sync --project "${ROOT}" --frozen --python "${ROOT}/.venv/bin/python" \
      --no-build-isolation \
      "${reinstall_cuda_extensions[@]}"
  run "${ROOT}/.venv/bin/python" -I -m real2sim.portable_tools \
    --environment-bin "${ROOT}/.venv/bin"
  run uv pip check --python "${ROOT}/.venv/bin/python"
  run "${ROOT}/.venv/bin/python" "${PROFILE_HELPER}" check-torch
  if [ "${SKIP_CUDA_EXTENSIONS}" = "0" ] && [ "${DRY_RUN}" = "0" ]; then
    "${ROOT}/.venv/bin/python" "${ROOT}/scripts/cuda_extension_build/inputs.py" "${ROOT}" --check
  fi
fi

echo "[bootstrap] install one profile's runtime inputs (README step 5), for example:"
echo "[bootstrap] .venv/bin/real2sim-download-release --profile config/release/fr3/flat/a_room_cup_cup1trim.yaml"
echo "[bootstrap] complete"
