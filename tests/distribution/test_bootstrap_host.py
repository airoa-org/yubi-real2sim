# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import configparser
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import yaml

from real2sim import portable_tools


REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap.sh"
COMPILE_LOCKS = REPO_ROOT / "scripts" / "compile_locks.sh"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PYTHON_VERSION_FILE = REPO_ROOT / ".python-version"
UV_LOCK = REPO_ROOT / "uv.lock"
SUPPORT_MATRIX = REPO_ROOT / "config" / "support-matrix.yaml"
HOST_CHECK = REPO_ROOT / "scripts" / "check_host.sh"
PYPROJECT = REPO_ROOT / "pyproject.toml"
COMPAT_HEADER = REPO_ROOT / "scripts" / "gcc13_libstdcxx_compat.h"
CUDA_EXTENSION_BUILD_CFG = REPO_ROOT / "scripts" / "cuda_extension_build.cfg"


def test_bootstrap_uses_pinned_public_sources():
    text = BOOTSTRAP.read_text(encoding="utf-8")
    for source in (
        "https://github.com/frankarobotics/franka_description.git",
    ):
        assert source in text
    # OpenPI and robosuite install from uv.lock; bootstrap clones neither.
    assert "Physical-Intelligence/openpi" not in text
    assert "ARISE-Initiative/robosuite" not in text
    assert "git@" + "github.com" not in text
    assert "/" + "home/" not in text
    assert "/" + "groups/" not in text
    assert "url = https://github.com/airoa-org/GaussianWrapping.git" in (REPO_ROOT / ".gitmodules").read_text(encoding="utf-8")


def test_python_patch_version_has_one_canonical_source():
    python_version = PYTHON_VERSION_FILE.read_text(encoding="utf-8").strip()
    assert python_version == "3.11.15"
    jobs = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    setup_steps = [
        step for job in jobs.values() for step in job["steps"]
        if step.get("uses", "").startswith("actions/setup-python@")
    ]
    assert setup_steps
    assert all(step["with"]["python-version-file"] == ".python-version" for step in setup_steps)
    for script in (BOOTSTRAP, COMPILE_LOCKS):
        text = script.read_text(encoding="utf-8")
        assert 'PYTHON_VERSION="$(tr -d \'[:space:]\' < "${ROOT}/.python-version")"' in text
    assert '--python-version "${PYTHON_VERSION}"' in COMPILE_LOCKS.read_text(
        encoding="utf-8"
    )
    profiles = yaml.safe_load(SUPPORT_MATRIX.read_text(encoding="utf-8"))["profiles"]
    assert {profile["python"] for profile in profiles.values()} == {python_version}
    profile = profiles["cuda12-x86_64"]
    assert profile["python"] == "3.11.15"
    assert profile["cuda"]["major"] == 12
    assert profile["cuda"]["minor"] == 8
    assert profile["cuda"]["source"] == "torch"
    assert profile["driver"] == {"minimum": "570.26"}
    assert profile["compiler"] == {"command": "gcc", "cxx_command": "g++", "major": 13}


def test_portable_ffmpeg_is_version_pinned():
    assert portable_tools.FFMPEG_VERSION


def test_root_uv_lock_hashes_every_registry_artifact():
    lock = tomllib.loads(UV_LOCK.read_text(encoding="utf-8"))
    registry_packages = [
        package
        for package in lock["package"]
        if "registry" in package.get("source", {})
    ]
    assert registry_packages
    for package in registry_packages:
        artifacts = [*package.get("wheels", [])]
        if "sdist" in package:
            artifacts.append(package["sdist"])
        assert artifacts, package["name"]
        assert all(item["hash"].startswith("sha256:") for item in artifacts)


def _cuda_extension_build_inputs():
    return subprocess.check_output(
        [sys.executable, str(REPO_ROOT / "scripts/cuda_extension_build/inputs.py"),
         str(REPO_ROOT), "--digest"], text=True,
    ).strip()


def test_gaussian_cuda_extensions_build_with_the_compat_header():
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    uv = project["tool"]["uv"]
    extensions = project["dependency-groups"]["cuda-extensions"]
    assert "cuda-extensions" in uv["default-groups"]
    assert uv["concurrent-builds"] == 1
    assert {entry["name"] for entry in uv["dependency-metadata"]} == set(extensions)
    build_cfg = configparser.ConfigParser()
    build_cfg.read_string(CUDA_EXTENSION_BUILD_CFG.read_text(encoding="utf-8"))
    assert build_cfg["aliases"]["bdist_wheel"] == "check_gaussian_wrapping clean --all build record_gaussian_build bdist_wheel"
    for name in extensions:
        assert name in uv["no-build-isolation-package"]
        source = REPO_ROOT / uv["sources"][name]["path"]
        assert source.parent == REPO_ROOT / "third_party/GaussianWrapping/submodules"
        variables = uv["extra-build-variables"][name]
        build_cfg_path = Path(os.path.normpath(source / variables["DIST_EXTRA_CONFIG"]))
        assert build_cfg_path == CUDA_EXTENSION_BUILD_CFG
        command_package = Path(os.path.normpath(source / variables["PYTHONPATH"])) / build_cfg[
            "global"
        ]["command_packages"]
        assert (command_package / "check_gaussian_wrapping.py").is_file()
        include_dir = Path(os.path.normpath(source / build_cfg["build_ext"]["include_dirs"]))
        assert include_dir / COMPAT_HEADER.name == COMPAT_HEADER
        for flag in ("CPPFLAGS", "NVCC_PREPEND_FLAGS"):
            assert variables[flag].endswith(f"-include {COMPAT_HEADER.name}")
        assert variables["REAL2SIM_BUILD_INPUTS"] == _cuda_extension_build_inputs(), (
            "set REAL2SIM_BUILD_INPUTS to the new digest so uv rebuilds the extensions"
        )


def test_bootstrap_dry_run_is_non_mutating_and_complete():
    completed = subprocess.run(
        [str(BOOTSTRAP), "--dry-run", "--skip-python"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "[dry-run]" in completed.stdout
    assert "[bootstrap] complete" in completed.stdout

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_CHECK = REPO_ROOT / "scripts" / "check_host.sh"
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap.sh"


def _executable(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _host_environment(
    tmp_path: Path,
    *,
    broken_nvml: bool = False,
    include_nvcc: bool = True,
    driver_version: str = "580.173",
    cuda_version: str = "12.8",
) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _executable(
        fake_bin,
        "uv",
        "printf 'uv 0.12.3 (x86_64-unknown-linux-gnu)\\n'\n",
    )
    if include_nvcc:
        _executable(
            fake_bin,
            "nvcc",
            f"printf 'Cuda compilation tools, release {cuda_version}, V{cuda_version}.20\\n'\n",
        )
    for compiler in ("gcc", "g++"):
        _executable(
            fake_bin,
            compiler,
            "if [ \"${1:-}\" = -dumpversion ]; then printf '13\\n'; "
            "else printf '13.3.0\\n'; fi\n",
        )
    if broken_nvml:
        _executable(
            fake_bin,
            "nvidia-smi",
            "printf 'Failed to initialize NVML: Driver/library version mismatch\\n' >&2\n"
            "printf 'NVML library version: 580.173\\n' >&2\n"
            "exit 1\n",
        )
    else:
        _executable(
            fake_bin,
            "nvidia-smi",
            "if [ \"${1:-}\" = --query-gpu=driver_version ]; then "
            f"printf '{driver_version}\\n'; else printf 'NVIDIA-SMI {driver_version}\\n'; fi\n",
        )
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join((str(fake_bin), "/usr/bin", "/bin"))
    for name in ("CUDA_HOME", "CC", "CXX"):
        environment.pop(name, None)
    if not include_nvcc:
        environment["CUDA_HOME"] = str(tmp_path / "missing-cuda")
    return environment


def test_host_check_accepts_the_locked_gpu_toolchain(tmp_path: Path) -> None:
    completed = subprocess.run(
        [str(HOST_CHECK)],
        cwd=REPO_ROOT,
        env=_host_environment(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "CUDA toolkit 12.8" in completed.stdout
    assert "DECISION: PROCEED" in completed.stdout
    assert "NEXT_COMMAND: ./scripts/bootstrap.sh" in completed.stdout
    assert "EXIT_CODE: 0" in completed.stdout
    assert "COPY_REPORT_BEGIN" not in completed.stdout


def test_host_check_explains_nvml_driver_mismatch(tmp_path: Path) -> None:
    completed = subprocess.run(
        [str(HOST_CHECK)],
        cwd=REPO_ROOT,
        env=_host_environment(tmp_path, broken_nvml=True, include_nvcc=False),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "NVIDIA driver/NVML mismatch" in completed.stderr
    assert "userspace NVML=580.173" in completed.stderr
    assert "CUDA toolkit 12.8 nvcc is missing" in completed.stderr
    assert "DECISION: STOP" in completed.stderr
    assert "NEXT_COMMAND: NONE" in completed.stderr
    assert "EXIT_CODE: 1" in completed.stderr
    assert "RETURN_TO: HOST_ADMIN_OR_SETUP_REQUESTER" in completed.stderr
    assert "RETURN_WHAT: COPY_REPORT_BEGIN..COPY_REPORT_END" in completed.stderr
    assert "COPY_REPORT_BEGIN" in completed.stderr
    assert "COPY_REPORT_END" in completed.stderr
    assert "Do not run ./scripts/bootstrap.sh yet" in completed.stderr
    assert "REQUIRED_ACTION:" in completed.stderr
    assert "schedule a safe reboot" in completed.stderr
    assert "cuda-toolkit-12-8" in completed.stderr
    assert "[host-check] DECISION: PROCEED" not in completed.stderr


def test_bootstrap_does_not_continue_after_a_stop_decision(tmp_path: Path) -> None:
    completed = subprocess.run(
        [str(BOOTSTRAP)],
        cwd=REPO_ROOT,
        env=_host_environment(tmp_path, broken_nvml=True, include_nvcc=False),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "DECISION: STOP" in completed.stderr
    assert "stopped before .venv and portable tools were installed" in completed.stderr
    assert "[bootstrap] complete" not in completed.stdout


def test_host_check_rejects_the_retired_toolkit_and_old_driver(tmp_path):
    result = subprocess.run(
        [str(HOST_CHECK)], capture_output=True, text=True,
        env=_host_environment(tmp_path, driver_version="560.28.03", cuda_version="12.6"),
    )
    assert result.returncode == 1
    assert "older than 570.26" in result.stderr
    assert "expected CUDA toolkit 12.8, found 12.6" in result.stderr
    assert "DECISION: STOP" in result.stderr


def test_host_check_accepts_the_minimum_driver_and_reports_memory(tmp_path):
    result = subprocess.run(
        [str(HOST_CHECK)], capture_output=True, text=True,
        env=_host_environment(tmp_path, driver_version="570.26"),
    )
    assert result.returncode == 0, result.stderr
    assert "GPU memory:" in result.stdout
    assert "GPU compute processes:" in result.stdout


def test_host_check_uses_profile_toolkit_fallback_and_compiler_overrides(tmp_path):
    import json
    import shutil

    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "src/real2sim").mkdir(parents=True)
    (root / "config").mkdir()
    shutil.copy(HOST_CHECK, root / "scripts/check_host.sh")
    shutil.copy(REPO_ROOT / "src/real2sim/support_matrix.py", root / "src/real2sim")
    profile = json.loads(SUPPORT_MATRIX.read_text())
    toolkit = tmp_path / "toolkit"
    (toolkit / "bin").mkdir(parents=True)
    _executable(toolkit / "bin", "nvcc", "printf 'release 12.8\\n'\n")
    profile["profiles"]["cuda12-x86_64"]["cuda"]["toolkit_path"] = str(toolkit)
    (root / "config/support-matrix.yaml").write_text(json.dumps(profile))
    environment = _host_environment(tmp_path, include_nvcc=False)
    environment.pop("CUDA_HOME")
    for variable, name in (("CC", "gcc"), ("CXX", "g++")):
        original = tmp_path / "bin" / name
        selected = original.with_name(f"selected-{name}")
        original.rename(selected)
        _executable(original.parent, name, "printf '11\\n'\n")
        environment[variable] = str(selected)
    result = subprocess.run(
        [str(root / "scripts/check_host.sh")], env=environment,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert str(toolkit / "bin/nvcc") in result.stdout
    assert "selected-gcc" in result.stdout
    assert "selected-g++" in result.stdout


def test_bootstrap_builds_the_cuda_extensions_one_at_a_time(tmp_path):
    environment = _host_environment(tmp_path)
    environment.pop("MAX_JOBS", None)

    def syncs(**overrides):
        completed = subprocess.run(
            [str(BOOTSTRAP), "--dry-run"], cwd=REPO_ROOT, env={**environment, **overrides},
            capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        return [line.split() for line in completed.stdout.splitlines() if " uv sync " in line]

    first, second = syncs()
    assert {"UV_CONCURRENT_BUILDS=1", "MAX_JOBS=4"} <= set(second)
    assert "MAX_JOBS=2" in syncs(MAX_JOBS="2")[1]
    assert "--inexact" in first
    for name in ("diff-gaussian-rasterization", "diff-gaussian-rasterization-ours", "simple-knn"):
        assert first[first.index(name) - 1] == "--no-install-package"
