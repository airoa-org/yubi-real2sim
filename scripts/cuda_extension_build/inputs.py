# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Build records for the unpackageable GaussianWrapping extension sources.

This adapter is shared by setuptools, bootstrap, and doctor. It runs as a
script outside the application package because uv builds that package alongside
these extensions before either is installed.
"""

import argparse
import hashlib
import importlib.metadata
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib


PACKAGES = {
    "diff-gaussian-rasterization": "diff_gaussian_rasterization",
    "diff-gaussian-rasterization-ours": "diff_gaussian_rasterization_ours",
    "simple-knn": "simple_knn",
}
RECORD = "_real2sim_build.json"
COMPILER_FLAGS = ("CPPFLAGS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "CFLAGS", "CXXFLAGS", "LDFLAGS")


def command(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_revision(root):
    return command("git", "-C", str(root), "rev-parse", ":third_party/GaussianWrapping")  # the staged gitlink


def static_inputs(root):
    lock = tomllib.loads((root / "uv.lock").read_text())
    files = [root / "config/support-matrix.yaml",
             root / "scripts/gcc13_libstdcxx_compat.h", root / "scripts/cuda_extension_build.cfg",
             root / "src/real2sim/support_matrix.py",
             *sorted((root / "scripts/cuda_extension_build").glob("*.py"))]
    return {
        "files": {str(path.relative_to(root)): sha256(path) for path in files},
        "torch": [p["version"] for p in lock["package"] if p["name"] == "torch"],
        "gaussian_wrapping": expected_revision(root),
    }


def build_digest(root):
    return hashlib.sha256(json.dumps(static_inputs(root), sort_keys=True).encode()).hexdigest()[:16]


def check_compiler(profile, variable, name, version):
    if version.split(".")[0] != str(profile["compiler"]["major"]):
        raise ValueError(f"{variable}={name} is GCC {version}; expected GCC {profile['compiler']['major']}; export CC and CXX before uv sync")


def check_toolkit(profile, version):
    cuda = profile["cuda"]
    if not re.search(rf"release {cuda['major']}\.{cuda['minor']}(?:\D|$)", version):
        raise ValueError(f"nvcc does not match CUDA {cuda['major']}.{cuda['minor']}: {version}")


def load_profile(root):
    return json.loads((root / "config/support-matrix.yaml").read_text())["profiles"]["cuda12-x86_64"]


def build_inputs(root, *, building=False, runtime=False):
    if not runtime and "PYTORCH_NVCC" in os.environ:  # torch selects even an empty override instead of CUDA_HOME/bin/nvcc.
        raise ValueError("PYTORCH_NVCC is unsupported; unset it and select the toolkit with CUDA_HOME before uv sync")
    source = root / "third_party/GaussianWrapping"
    actual = command("git", f"--git-dir={source / '.git'}", "rev-parse", "HEAD")
    expected = expected_revision(root)
    if actual != expected:
        raise ValueError(f"third_party/GaussianWrapping is at {actual}, but the build expects {expected}; run scripts/bootstrap.sh")
    profile = load_profile(root)
    compilers = {}
    for variable, key in () if runtime else (("CC", "command"), ("CXX", "cxx_command")):
        name = os.environ.get(variable) or profile["compiler"][key]
        path = shutil.which(name)
        if path is None:
            raise ValueError(f"{variable}={name} is unavailable; select the support-matrix compiler before uv sync")
        version = command(path, "-dumpfullversion")
        check_compiler(profile, variable, name, version)
        compilers[variable] = {"path": str(Path(path).resolve()), "version": version}
    helper = root / "src/real2sim/support_matrix.py"
    toolchain = {}
    if not runtime:  # a run host checks the recorded toolchain and needs no build tools
        nvcc = command(sys.executable, str(helper), "nvcc")
        toolkit_version = command(nvcc, "--version")
        check_toolkit(profile, toolkit_version)
        toolchain = {"compilers": compilers, "nvcc": {"path": str(Path(nvcc).resolve()), "version": toolkit_version}}
    if building:
        import torch.utils.cpp_extension as cpp_extension
        if not cpp_extension.CUDA_HOME or Path(cpp_extension.CUDA_HOME).resolve() != Path(nvcc).parent.parent.resolve():
            raise ValueError(f"torch selected CUDA_HOME={cpp_extension.CUDA_HOME}; export CUDA_HOME={Path(nvcc).parent.parent} before uv sync")
        os.environ.setdefault("MAX_JOBS", "4")
    tracked = command("git", "-C", str(source), "ls-files", "--", "submodules").splitlines()
    source_hash = hashlib.sha256()
    for name in tracked:
        path = source / name
        if path.is_file():
            source_hash.update(name.encode() + b"\0" + path.read_bytes())
    project = tomllib.loads((root / "pyproject.toml").read_text())
    return {
        "schema_version": 1,
        "gaussian_wrapping_commit": actual,
        "source_sha256": source_hash.hexdigest(),
        "static": static_inputs(root),
        "torch": importlib.metadata.version("torch"),
        **({"architectures": command(sys.executable, str(helper), "cuda-arch-flags")} if not runtime else {}),
        **toolchain,
        "build_variables": project["tool"]["uv"]["extra-build-variables"],
        "environment": {key: os.environ.get(key, "") for key in ("CFLAGS", "CXXFLAGS", "LDFLAGS", "NVCC_APPEND_FLAGS")},
    }


def verify_installed(root, *, runtime=False):
    expected = build_inputs(root, runtime=runtime)
    profile = load_profile(root)
    records = {}
    for name, package in PACKAGES.items():
        distribution = importlib.metadata.distribution(name)
        path = distribution.locate_file(f"{package}/{RECORD}")
        record = json.loads(path.read_text())
        observed = record["inputs"]
        recorded_flags = record.get("compiler_flags", {})
        ambient = {key: os.environ.get(key, "") for key in COMPILER_FLAGS}
        if runtime:  # Ambient flags select only the next build, and release profiles do not pass them to doctor.
            ambient = {key: observed.get("environment", {}).get(key) for key in COMPILER_FLAGS}
        wanted = {**expected, "environment": {key: ambient[key] for key in expected["environment"]}}
        # A runtime may expose fewer GPUs than the build; doctor launches kernels on its selected GPUs.
        changed = sorted(key for key in wanted if observed.get(key) != wanted[key])
        if changed:
            raise ValueError(f"{name} build inputs changed: {', '.join(changed)}; rebuild with uv sync --locked --reinstall-package {name}")
        if runtime:
            if sorted(observed.get("compilers", {})) != ["CC", "CXX"]:
                raise ValueError(f"{name} build record has no CC/CXX compilers; rebuild the extension")
            try:
                for variable, compiler in observed["compilers"].items():
                    check_compiler(profile, variable, compiler["path"], compiler["version"])
                check_toolkit(profile, observed.get("nvcc", {}).get("version", ""))
            except ValueError as exc:
                raise ValueError(f"{name} was built with an unsupported toolchain: {exc}") from exc
        variables = expected["build_variables"][name]
        expected_flags = {key: variables.get(key, ambient[key]) for key in COMPILER_FLAGS}
        if recorded_flags != expected_flags:
            raise ValueError(f"{name} was compiled with different flags; rebuild the extension")
        if not record["binaries"]:
            raise ValueError(f"{name} has no recorded native binaries")
        for relative, digest in record["binaries"].items():
            binary = distribution.locate_file(relative)
            if sha256(binary) != digest:
                raise ValueError(f"{name} native binary changed: {binary}; rebuild the extension")
        package_spec = PathFinder.find_spec(package)
        native_spec = (PathFinder.find_spec(f"{package}._C", package_spec.submodule_search_locations)
                       if package_spec is not None else None)
        installed_binaries = {Path(distribution.locate_file(relative)).resolve()
                              for relative in record["binaries"]}
        if native_spec is None or Path(native_spec.origin).resolve() not in installed_binaries:
            raise ValueError(f"{name} import does not resolve to its recorded native binary")
        records[name] = record
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--digest", action="store_true")
    args = parser.parse_args()
    try:
        if args.digest:
            print(build_digest(args.root))
        elif args.check:
            print(json.dumps(verify_installed(args.root, runtime=args.runtime), sort_keys=True))
        else:
            print(json.dumps(build_inputs(args.root), sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError,
            importlib.metadata.PackageNotFoundError) as exc:
        print(f"[cuda-build] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
