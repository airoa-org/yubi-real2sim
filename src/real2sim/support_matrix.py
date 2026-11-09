# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Read the bootstrap toolchain without third-party Python dependencies.

The checked-in support-matrix.yaml uses JSON syntax (a subset of YAML), so
the host gate can read it with system Python before PyYAML is installed.
Runtime callers may still load custom YAML matrices through PyYAML.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
from typing import Any, Mapping
import warnings


DEFAULT_PROFILE = "cuda12-x86_64"
DEFAULT_MATRIX = Path(__file__).resolve().parents[2] / "config/support-matrix.yaml"


def load_bootstrap_profile(path: Path = DEFAULT_MATRIX) -> dict[str, Any]:
    matrix = json.loads(path.read_text(encoding="utf-8"))
    if matrix.get("schema_version") != 1:
        raise ValueError("support matrix must use schema_version: 1")
    profile = matrix["profiles"][DEFAULT_PROFILE]
    if profile["kind"] != "cuda" or profile["cuda"]["source"] != "torch":
        raise ValueError("bootstrap requires a CUDA profile with source: torch")
    for key in ("major", "minor"):
        value = profile["cuda"][key]
        if type(value) is not int or value < (1 if key == "major" else 0):
            raise ValueError(f"cuda.{key} must be a valid version integer")
    cuda = profile["cuda"]
    expected_index = f"https://download.pytorch.org/whl/cu{cuda['major']}{cuda['minor']}"
    if cuda["pytorch_index"] != expected_index:
        raise ValueError("pytorch_index must match the profile's CUDA version")
    if not Path(cuda["toolkit_path"]).is_absolute():
        raise ValueError("cuda.toolkit_path must be absolute")
    return profile


def profile_fingerprint(profile: Mapping[str, Any]) -> str:
    # Runtime loaders add a display name; it is not part of the build inputs.
    payload = {key: value for key, value in profile.items() if key != "name"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def find_nvcc(
    profile: Mapping[str, Any], environment: Mapping[str, str] | None = None
) -> str | None:
    """CUDA_HOME wins, then PATH, then the configured toolkit directory.

    An explicitly set but invalid CUDA_HOME is an error, not a reason to
    silently compile with a different toolkit.
    """
    env = os.environ if environment is None else environment
    if env.get("CUDA_HOME"):
        candidate = Path(env["CUDA_HOME"]) / "bin/nvcc"
    else:
        on_path = shutil.which("nvcc", path=env.get("PATH", os.defpath))
        if on_path is not None:
            return str(Path(on_path).resolve())
        toolkit_path = profile["cuda"].get("toolkit_path")
        if not toolkit_path:
            return None
        candidate = Path(toolkit_path) / "bin/nvcc"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate.resolve())
    return None


def shell_settings(profile: Mapping[str, Any]) -> str:
    cuda = profile["cuda"]
    values = {
        "R2S_ARCHITECTURE": profile["architecture"],
        "R2S_CUDA_VERSION": f"{cuda['major']}.{cuda['minor']}",
        "R2S_MIN_DRIVER": profile["driver"]["minimum"],
        "R2S_CC": profile["compiler"]["command"],
        "R2S_CXX": profile["compiler"]["cxx_command"],
        "R2S_COMPILER_MAJOR": profile["compiler"]["major"],
        "R2S_PROFILE_SHA256": profile_fingerprint(profile),
    }
    return "\n".join(f"{key}={shlex.quote(str(value))}" for key, value in values.items())


def check_torch(profile: Mapping[str, Any]) -> None:
    import torch
    import torchvision

    cuda = profile["cuda"]
    expected = f"{cuda['major']}.{cuda['minor']}"
    suffix = f"+cu{cuda['major']}{cuda['minor']}"
    if (
        torch.version.cuda != expected
        or not torch.__version__.endswith(suffix)
        or not torchvision.__version__.endswith(suffix)
    ):
        raise ValueError(
            f"expected torch/torchvision {suffix} and CUDA {expected}; "
            f"got torch={torch.__version__}, torchvision={torchvision.__version__}, "
            f"CUDA={torch.version.cuda}"
        )
    print(f"torch={torch.__version__} torchvision={torchvision.__version__} CUDA={expected}")


def cuda_arch_flags() -> str:
    """Return the -gencode flags that torch's extension build derives from
    TORCH_CUDA_ARCH_LIST, or from the visible GPUs when the list is unset."""
    import torch
    from torch.utils.cpp_extension import _get_cuda_arch_flags

    if not os.environ.get("TORCH_CUDA_ARCH_LIST") and torch.cuda.device_count() == 0:
        raise ValueError("no visible CUDA GPU; set TORCH_CUDA_ARCH_LIST")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch warns on every unset list
        return " ".join(_get_cuda_arch_flags())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("shell", "nvcc", "check-torch", "cuda-arch-flags"))
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    args = parser.parse_args()
    try:
        profile = load_bootstrap_profile(args.matrix)
        if args.action == "shell":
            print(shell_settings(profile))
        elif args.action == "nvcc":
            nvcc = find_nvcc(profile)
            if nvcc is None:
                raise ValueError("nvcc not found in CUDA_HOME, PATH, or the support matrix toolkit_path")
            print(nvcc)
        elif args.action == "cuda-arch-flags":
            print(cuda_arch_flags())
        else:
            check_torch(profile)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"[support-matrix] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
