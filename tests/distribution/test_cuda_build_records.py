# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Exercise the wheel hook and stale-build checks with real CPU native wheels.

Only torch's CUDA discovery and nvcc are stand-ins; setuptools compiles the
fixture extension and packages the production build record in each wheel.
"""

import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[2]
PACKAGES = {
    "diff-gaussian-rasterization": "diff_gaussian_rasterization",
    "diff-gaussian-rasterization_ours": "diff_gaussian_rasterization_ours",
    "simple-knn": "simple_knn",
}


def run(*args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, **kwargs)


def pin(root, commit):
    git(root, "update-index", "--add", "--cacheinfo", f"160000,{commit},third_party/GaussianWrapping")


def git(path, *args):
    result = run("git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "-c", "commit.gpgsign=false", "-C", str(path), *args)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture()
def fixture(tmp_path):
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    shutil.copytree(ROOT / "scripts/cuda_extension_build", root / "scripts/cuda_extension_build",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("cuda_extension_build.cfg", "gcc13_libstdcxx_compat.h"):
        shutil.copy(ROOT / "scripts" / name, root / "scripts" / name)
    (root / "src/real2sim").mkdir(parents=True)
    shutil.copy(ROOT / "src/real2sim/support_matrix.py", root / "src/real2sim")
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy(ROOT / name, root / name)
    (root / "config").mkdir()
    profile = json.loads((ROOT / "config/support-matrix.yaml").read_text())
    version = subprocess.check_output(["gcc", "-dumpfullversion"], text=True).strip()
    profile["profiles"]["cuda12-x86_64"]["compiler"]["major"] = int(version.split(".")[0])
    (root / "config/support-matrix.yaml").write_text(json.dumps(profile))
    toolkit = tmp_path / "cuda"
    (toolkit / "bin").mkdir(parents=True)
    nvcc = toolkit / "bin/nvcc"
    nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 12.8, V12.8.1'\n")
    nvcc.chmod(0o755)
    site = tmp_path / "site"
    (site / "torch/utils").mkdir(parents=True)
    (site / "torch/__init__.py").write_text(
        "from types import SimpleNamespace\ncuda=SimpleNamespace(device_count=lambda: 0)\n"
    )
    (site / "torch/utils/__init__.py").write_text("")
    (site / "torch/utils/cpp_extension.py").write_text(
        "import os\nCUDA_HOME=os.environ.get('CUDA_HOME')\n"
        "def _get_cuda_arch_flags(): return [os.environ['TORCH_CUDA_ARCH_LIST']]\n"
    )
    (site / "torch-2.7.1.dist-info").mkdir()
    (site / "torch-2.7.1.dist-info/METADATA").write_text("Name: torch\nVersion: 2.7.1\n")
    source = root / "third_party/GaussianWrapping"
    for directory, package in PACKAGES.items():
        extension = source / "submodules" / directory
        (extension / package).mkdir(parents=True)
        (extension / package / "__init__.py").write_text("")
        (extension / "module.cpp").write_text(
            '#include <Python.h>\nstatic PyModuleDef module = {PyModuleDef_HEAD_INIT,"_C",0,-1,0};\n'
            'PyMODINIT_FUNC PyInit__C(){return PyModule_Create(&module);}\n'
        )
        (extension / "setup.py").write_text(
            "import os\nfrom setuptools import setup, Extension\n"
            "from setuptools.command.build_ext import build_ext\n"
            "class BuildExt(build_ext):\n"
            "    def build_extensions(self):\n"
            "        assert os.environ['MAX_JOBS'] == os.environ.get('EXPECTED_JOBS', '4')\n"
            "        super().build_extensions()\n"
            f"setup(name={package!r},version='0.0.0',packages=[{package!r}],"
            f"ext_modules=[Extension('{package}._C',['module.cpp'])],cmdclass={{'build_ext':BuildExt}})\n"
        )
    git(source, "init", "-q")
    git(source, "add", ".")
    git(source, "commit", "-qm", "fixture")
    head = git(source, "rev-parse", "HEAD")
    git(root, "init", "-q")
    pin(root, head)
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"CC", "CXX", "MAX_JOBS",
                                  "CFLAGS", "CXXFLAGS", "LDFLAGS", "CUDA_VISIBLE_DEVICES",
                                  "NVCC_APPEND_FLAGS", "PYTORCH_NVCC"}}
    environment.update(CUDA_HOME=str(toolkit), TORCH_CUDA_ARCH_LIST="12.0", PYTHONPATH=str(site))
    variables = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["uv"]["extra-build-variables"]
    wheels = tmp_path / "wheels"
    wheels.mkdir()

    def build(directory="simple-knn", **overrides):
        name = directory.replace("_", "-")
        values = variables[name]
        return run(sys.executable, "-c", "import setuptools.build_meta as b; "
                   f"b.__legacy__.build_wheel({str(wheels)!r})",
                   cwd=source / "submodules" / directory,
                   env={**environment, **values,
                        "PYTHONPATH": os.pathsep.join((str(root / "scripts"), str(site))), **overrides})

    def install():
        for directory in PACKAGES:
            result = build(directory)
            assert result.returncode == 0, result.stdout + result.stderr
        result = run("uv", "pip", "install", "--python", sys.executable,
                     "--no-deps", "--target", str(site),
                     *(str(path) for path in wheels.glob("*.whl")))
        assert result.returncode == 0, result.stdout + result.stderr

    def check(*args, **overrides):
        return run(sys.executable, str(root / "scripts/cuda_extension_build/inputs.py"),
                   str(root), "--check", *args, cwd=tmp_path,
                   env={**environment, **overrides})

    return root, source, site, environment, build, install, check


def test_wheels_record_actual_native_artifacts_without_bootstrap(fixture):
    root, source, site, environment, build, install, check = fixture
    assert not (root / "scripts/bootstrap.sh").exists()
    install()
    result = check()
    assert result.returncode == 0, result.stderr
    records = json.loads(result.stdout)
    assert len(records) == 3
    for record in records.values():
        assert record["inputs"]["gaussian_wrapping_commit"] == git(source, "rev-parse", "HEAD")
        assert record["inputs"]["architectures"] == "12.0"
        assert len(record["binaries"]) == 1
    loaded = run(sys.executable, "-c", "import simple_knn._C; print(simple_knn._C.__file__)",
                 cwd=root.parent, env=environment)
    assert loaded.returncode == 0, loaded.stderr
    assert str(site) in loaded.stdout


def test_build_rejects_unpinned_checkout_and_wrong_compiler_before_compiling(fixture):
    root, source, site, environment, build, install, check = fixture
    pin(root, "1" * 40)
    rejected = build()
    assert rejected.returncode != 0
    assert "but the build expects " + "1" * 40 in rejected.stderr
    compiler = root.parent / "old-gcc"
    compiler.write_text("#!/bin/sh\necho 1.0.0\n")
    compiler.chmod(0o755)
    pin(root, git(source, "rev-parse", "HEAD"))
    rejected = build(CC=str(compiler))
    assert rejected.returncode != 0
    assert "export CC and CXX before uv sync" in rejected.stderr
    assert not list(source.rglob("*.so"))


@pytest.mark.parametrize("override", ["", "/uninvoked/alternate-nvcc"])
def test_build_rejects_pytorch_nvcc_override_before_compiling(fixture, override):
    root, source, site, environment, build, install, check = fixture
    rejected = build(PYTORCH_NVCC=override)
    assert rejected.returncode != 0
    assert "PYTORCH_NVCC is unsupported; unset it" in rejected.stderr
    assert not list(source.rglob("*.so"))
    assert not list(source.rglob("_real2sim_build.json"))


def test_records_track_appended_flags_without_requiring_runtime_build_overrides(fixture):
    root, source, site, environment, build, install, check = fixture
    environment["NVCC_APPEND_FLAGS"] = "--fmad=false"
    install()
    checked = check()
    assert checked.returncode == 0, checked.stderr
    for record in json.loads(checked.stdout).values():
        assert record["inputs"]["environment"]["NVCC_APPEND_FLAGS"] == "--fmad=false"
        assert record["compiler_flags"]["NVCC_APPEND_FLAGS"] == "--fmad=false"
    for flags in ("--fmad=true", ""):
        rejected = check(NVCC_APPEND_FLAGS=flags)
        assert rejected.returncode != 0
        assert "build inputs changed: environment" in rejected.stderr
        # The release allowlist drops these flags, and they do not change the installed binary.
        assert check("--runtime", NVCC_APPEND_FLAGS=flags).returncode == 0
    for override in ("", "/uninvoked/alternate-nvcc"):
        rejected = check(PYTORCH_NVCC=override)
        assert rejected.returncode != 0
        assert "PYTORCH_NVCC is unsupported; unset it" in rejected.stderr
        checked = check("--runtime", PYTORCH_NVCC=override)
        assert checked.returncode == 0, checked.stderr
    record_path = site / "diff_gaussian_rasterization/_real2sim_build.json"
    original = record_path.read_text()
    for field, message in (("environment", "build inputs changed: environment"),
                           ("compiler_flags", "was compiled with different flags")):
        record = json.loads(original)
        values = record["inputs"][field] if field == "environment" else record[field]
        del values["NVCC_APPEND_FLAGS"]
        record_path.write_text(json.dumps(record))
        for args in ((), ("--runtime",)):
            rejected = check(*args)
            assert rejected.returncode != 0
            assert message in rejected.stderr


def test_doctor_and_bootstrap_reject_a_stale_wheel_after_checkout_change(fixture):
    root, source, site, environment, build, install, check = fixture
    install()
    path = source / "submodules/simple-knn/module.cpp"
    path.write_text(path.read_text() + "\n// fixture change\n")
    git(source, "add", ".")
    git(source, "commit", "-qm", "another revision")
    head = git(source, "rev-parse", "HEAD")
    result = check()
    assert result.returncode != 0
    assert "but the build expects" in result.stderr
    pin(root, head)
    for args in ((), ("--runtime",)):
        result = check(*args)
        assert result.returncode != 0
        assert "gaussian_wrapping_commit" in result.stderr


def test_records_detect_architecture_toolchain_adapter_and_binary_changes(fixture):
    root, source, site, environment, build, install, check = fixture
    install()
    assert check(TORCH_CUDA_ARCH_LIST="8.9").returncode != 0
    assert "architectures" in check(TORCH_CUDA_ARCH_LIST="8.9").stderr
    assert check(TORCH_CUDA_ARCH_LIST="").returncode != 0
    header = root / "scripts/gcc13_libstdcxx_compat.h"
    before = header.read_text()
    header.write_text(before + "\n// fixture change\n")
    assert "static" in check().stderr
    header.write_text(before)
    compiler = root.parent / "other-gcc"
    compiler.symlink_to(shutil.which("gcc"))
    assert check(CC=str(compiler)).returncode == 0  # the actual executable is unchanged
    nvcc = Path(environment["CUDA_HOME"]) / "bin/nvcc"
    before = nvcc.read_text()
    nvcc.write_text(before.replace("V12.8.1", "V12.8.2"))
    assert "nvcc" in check().stderr
    nvcc.write_text(before)
    binary = next((site / "simple_knn").glob("*.so"))
    binary.write_bytes(binary.read_bytes() + b"altered")
    assert "native binary changed" in check().stderr


def test_runtime_check_verifies_the_recorded_toolchain_without_build_tools(fixture, tmp_path):
    # A run from another shell, without the build environment, must not ask for a rebuild.
    root, source, site, environment, build, install, check = fixture
    install()
    tools = tmp_path / "runtime-bin"
    tools.mkdir()
    (tools / "git").symlink_to(shutil.which("git"))
    runtime = {"PATH": str(tools), "CUDA_HOME": str(tmp_path / "no-toolkit")}
    assert "CC=gcc is unavailable" in check(**runtime).stderr
    checked = check("--runtime", **runtime)
    assert checked.returncode == 0, checked.stderr
    for architectures in ("", "invalid-build-architecture"):
        checked = check("--runtime", **runtime, TORCH_CUDA_ARCH_LIST=architectures,
                        CC="/missing/gcc", CXX="/missing/g++", PYTORCH_NVCC="/missing/nvcc",
                        NVCC_APPEND_FLAGS="-ccbin /missing/gcc", NVCC_PREPEND_FLAGS="--options-file missing")
        assert checked.returncode == 0, checked.stderr
    other = tmp_path / "other-cuda/bin/nvcc"
    other.parent.mkdir(parents=True)
    other.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 12.8, V12.8.93'\n")
    other.chmod(0o755)
    assert check("--runtime", CUDA_HOME=str(other.parents[1])).returncode == 0
    record_path = site / "simple_knn/_real2sim_build.json"
    original = record_path.read_text()
    for field, value, message in (("compilers", {"CC": {"path": "/usr/bin/gcc-11", "version": "11.5.0"}},
                                   "CC=/usr/bin/gcc-11 is GCC 11.5.0"),
                                  ("nvcc", {"path": "/usr/local/cuda-12.6/bin/nvcc",
                                            "version": "Cuda compilation tools, release 12.6, V12.6.85"},
                                   "nvcc does not match CUDA 12.8")):
        record = json.loads(original)
        record["inputs"][field] = {**record["inputs"][field], **value} if field == "compilers" else value
        record_path.write_text(json.dumps(record))
        rejected = check("--runtime", **runtime)
        assert rejected.returncode != 0
        assert message in rejected.stderr
    record = json.loads(original)
    del record["inputs"]["compilers"]["CXX"]
    record_path.write_text(json.dumps(record))
    assert "compilers" in check("--runtime", **runtime).stderr


def test_production_preflight_checks_installed_native_wheels_without_build_environment(fixture):
    root, source, site, environment, build, install, check = fixture
    environment.update(CC=shutil.which("gcc"), CXX=shutil.which("g++"), NVCC_APPEND_FLAGS="--fmad=false")
    install()
    runtime = root.parent / "runtime-env"
    python = runtime / "bin/python"
    created = run("uv", "venv", str(runtime), "--python", sys.executable)
    assert created.returncode == 0, created.stderr
    dependencies = run("uv", "pip", "install", "--python", str(python),
                       "setuptools==80.9.0", "wheel==0.46.3", "packaging==25.0")
    assert dependencies.returncode == 0, dependencies.stderr
    torch = root.parent / "runtime-torch"
    shutil.copytree(site / "torch", torch / "torch")
    (torch / "torch/__init__.py").write_text(
        "from types import SimpleNamespace\nversion=SimpleNamespace(cuda='12.8')\n"
    )
    (torch / "pyproject.toml").write_text(
        '[build-system]\nrequires=["setuptools==80.9.0"]\nbuild-backend="setuptools.build_meta"\n'
        '[project]\nname="torch"\nversion="2.7.1"\n'
    )
    installed = run("uv", "pip", "install", "--python", str(python), "--no-deps",
                    "PyYAML==6.0.3", "numpy==1.26.4", "--editable", str(ROOT), str(torch),
                    *(str(path) for path in (root.parent / "wheels").glob("*.whl")))
    assert installed.returncode == 0, installed.stderr
    tools = root.parent / "runtime-tools"
    tools.mkdir()
    (tools / "git").symlink_to(shutil.which("git"))
    ffmpeg = tools / "ffmpeg"
    ffmpeg.write_text("#!/bin/sh\nexit 99\n")  # video encoding is outside the native-build check
    ffmpeg.chmod(0o755)
    # The run host's default compiler is not the recorded one, as on a host whose gcc is GCC 11 while
    # the wheels were built with CC=gcc-13. The release allowlist drops CC and CXX, so a runtime check
    # that looked up a compiler would find this one and fail.
    major = json.loads((root / "config/support-matrix.yaml").read_text())["profiles"]["cuda12-x86_64"]["compiler"]["major"]
    for name in ("gcc", "g++"):
        (tools / name).write_text(f"#!/bin/sh\necho {major - 2}.5.0\n")
        (tools / name).chmod(0o755)
    runtime_environment = {key: value for key, value in environment.items() if key != "PYTHONPATH"}
    runtime_environment.update(PATH=str(tools), CC="/missing/gcc", CXX="/missing/g++",
                               CUDA_HOME="/missing/cuda", PYTORCH_NVCC="/missing/nvcc",
                               TORCH_CUDA_ARCH_LIST="invalid-build-architecture")
    checked = run(str(python), str(ROOT / "tests/fixtures/native_build_preflight.py"),
                  str(root), str(ROOT), cwd=root.parent, env=runtime_environment)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert json.loads(checked.stdout) == {"records": 3, "rejections": ["source_sha256", "static", "native binary changed", "import does not resolve"]}


def test_plain_uv_sync_uses_the_same_wheel_hook_and_build_limits(fixture):
    root, source, site, environment, build, install, check = fixture
    uv = shutil.which("uv")
    assert uv, "uv is required for the supported sync-path regression"
    project = tomllib.loads((root / "pyproject.toml").read_text())
    settings = project["tool"]["uv"]
    names = project["dependency-groups"]["cuda-extensions"]
    build_dependencies = ["setuptools==80.9.0", "wheel==0.46.3", "packaging==25.0"]
    lines = [
        '[project]', 'name = "cuda-build-fixture"', 'version = "0.0.0"',
        'requires-python = ">=3.11"', 'dependencies = ' + json.dumps(build_dependencies),
        '[dependency-groups]',
        'cuda-extensions = ' + json.dumps(names), '[tool.uv]', 'package = false',
        'default-groups = ["cuda-extensions"]',
        'concurrent-builds = ' + str(settings["concurrent-builds"]),
        'no-build-isolation-package = ' + json.dumps(names), '[tool.uv.sources]',
    ]
    lines += [f'{name} = {{ path = {json.dumps(settings["sources"][name]["path"])} }}' for name in names]
    for name in names:
        lines += ['[[tool.uv.dependency-metadata]]', f'name = "{name}"',
                  'version = "0.0.0"', 'requires-dist = []',
                  f'[tool.uv.extra-build-variables.{name}]']
        values = {**settings["extra-build-variables"][name],
                  "PYTHONPATH": os.pathsep.join((str(root / "scripts"), str(site)))}
        lines += [f'{key} = {json.dumps(value)}' for key, value in values.items()]
    (root / "pyproject.toml").write_text("\n".join(lines) + "\n")
    (root / "uv.lock").unlink()
    venv = root.parent / "sync-env"
    created = run(uv, "venv", str(venv), "--python", sys.executable)
    assert created.returncode == 0, created.stderr
    installed = run(uv, "pip", "install", "--python", str(venv / "bin/python"),
                    *build_dependencies)
    assert installed.returncode == 0, installed.stderr
    result = run(uv, "sync", "--project", str(root), "-v",
                 env={**environment, "UV_PROJECT_ENVIRONMENT": str(venv),
                      "UV_CACHE_DIR": str(root.parent / "uv-cache")})
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr.count("running record_gaussian_build") == 3
    def verified(env):
        return run(str(venv / "bin/python"), str(root / "scripts/cuda_extension_build/inputs.py"),
                   str(root), "--check", cwd=root.parent, env=env)

    checked = verified(environment)
    assert checked.returncode == 0, checked.stderr
    changed = {**environment, "TORCH_CUDA_ARCH_LIST": "8.9", "NVCC_APPEND_FLAGS": "--fmad=false",
               "UV_PROJECT_ENVIRONMENT": str(venv), "UV_CACHE_DIR": str(root.parent / "uv-cache")}
    ordinary = run(uv, "sync", "--locked", "--project", str(root), env=changed)
    assert ordinary.returncode == 0, ordinary.stderr
    assert "architectures, environment" in verified(changed).stderr
    reinstall = [arg for name in names for arg in ("--reinstall-package", name)]
    rebuilt = run(uv, "sync", "--locked", "--project", str(root), *reinstall, env=changed)
    assert rebuilt.returncode == 0, rebuilt.stderr
    checked = verified(changed)
    assert checked.returncode == 0, checked.stderr
    for record in json.loads(checked.stdout).values():
        assert record["compiler_flags"]["NVCC_APPEND_FLAGS"] == "--fmad=false"
    rejected = run(uv, "sync", "--locked", "--project", str(root), *reinstall,
                   env={**changed, "PYTORCH_NVCC": "/uninvoked/alternate-nvcc"})
    assert rejected.returncode != 0
    assert "PYTORCH_NVCC is unsupported; unset it" in rejected.stderr


def test_bootstrap_selects_reinstall_from_installed_records(fixture):
    root, source, site, environment, build, install, check = fixture
    install()
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/python").symlink_to(sys.executable)
    bootstrap = (ROOT / "scripts/bootstrap.sh").read_text()
    decision = re.search(r"  reinstall_cuda_extensions=\(\)\n.*?(?=  # With ninja,)",
                         bootstrap, re.S).group(0)
    script = decision + '\nprintf "reinstall:%s\n" "${reinstall_cuda_extensions[*]}"\n'
    env = {**environment, "ROOT": str(root), "SKIP_CUDA_EXTENSIONS": "0", "DRY_RUN": "0"}
    unchanged = run("bash", "-eu", "-c", script, env=env)
    assert unchanged.returncode == 0, unchanged.stderr
    assert unchanged.stdout.splitlines()[-1] == "reinstall:"
    for overrides in ({"TORCH_CUDA_ARCH_LIST": "8.9"}, {"NVCC_APPEND_FLAGS": "--fmad=false"}):
        changed = run("bash", "-eu", "-c", script, env={**env, **overrides})
        assert changed.returncode == 0, changed.stderr
        assert changed.stdout.splitlines()[-1] == (
            "reinstall:--reinstall-package diff-gaussian-rasterization "
            "--reinstall-package diff-gaussian-rasterization-ours --reinstall-package simple-knn"
        )
