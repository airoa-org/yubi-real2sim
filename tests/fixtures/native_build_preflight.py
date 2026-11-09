# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Run production preflight against installed CPU fixture wheels outside the checkout."""

from contextlib import ExitStack
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from real2sim import doctor, preflight
from real2sim.config.schema import load_rollout_config


root, checkout = map(Path, sys.argv[1:])
assert "PYTHONPATH" not in os.environ
plan = load_rollout_config(checkout / "config/release/fixture_v2.yaml", timestamp="20260802_000000")
plan["repo_root"] = str(root)
plan["run_root"] = str(root.parent / "run")
plan["resolved"]["policy"]["python"] = sys.executable
reproducibility = plan["resolved"]["reproducibility"]
assert not {"CC", "CXX", "NVCC_APPEND_FLAGS", "CUDA_HOME", "TORCH_CUDA_ARCH_LIST", "PYTHONPATH"} & reproducibility["environment"].keys()
reproducibility.update(production=True, artifacts={}, support_matrix=str(root / "config/support-matrix.yaml"),
                       accelerator_profile={**reproducibility["accelerator_profile"], "name": "cuda12-x86_64"},
                       lockfiles=[str(root / "uv.lock")])
for key in ("gs_cuda_devices", "policy_cuda_device", "sim_cuda_devices"):
    plan["resolved"]["runtime"][key] = "0"

unrelated = {
    "collect_git_repository": lambda _path, **_kwargs: {"commit": "0" * 40},
    "collect_lockfiles": lambda _root, _locks, **_kwargs: [{"path": str(root / "uv.lock")}],
    "collect_nvidia": lambda _env: {"gpus": [
        {"index": "0", "uuid": "GPU-a", "driver_version": "999.0", "compute_capability": "12.0"}]},
    "_check_locked_uv_environment": lambda _root, _env: (True, "CPU fixture dependencies"),
    "_check_python_dependencies": lambda _env: (True, "CPU fixture dependencies"),
    "verify_artifacts": lambda *_args, **_kwargs: [],
}
kernels = []
unrelated["_check_gaussian_cuda_extensions"] = lambda _env, devices: (kernels.append(devices) is None, "CPU fixture")
rejections = []


def reject(message):
    try:
        preflight.validate_production_preflight(plan)
    except ValueError as exc:
        assert "gaussian_build_records" in str(exc), str(exc)
        assert message in str(exc), str(exc)
        rejections.append(message)
    else:
        raise AssertionError(f"preflight accepted {message}")


with ExitStack() as stack:
    for name, replacement in unrelated.items():
        stack.enter_context(patch.object(doctor, name, replacement))
    report = preflight.validate_production_preflight(plan)
    records_check = next(item for item in report["checks"] if item["name"] == "gaussian_build_records")
    assert records_check["status"] == "ok", records_check
    records = json.loads(records_check["detail"])
    assert kernels == [[0]]
    for path, message in ((root / "third_party/GaussianWrapping/submodules/simple-knn/module.cpp", "source_sha256"),
                          (root / "scripts/gcc13_libstdcxx_compat.h", "static")):
        original = path.read_bytes()
        path.write_bytes(original + b"\n// altered\n")
        reject(message)
        path.write_bytes(original)
    distribution = importlib.metadata.distribution("simple-knn")
    [relative] = records["simple-knn"]["binaries"]
    binary = Path(distribution.locate_file(relative))
    original = binary.read_bytes()
    binary.write_bytes(original + b"altered")
    reject("native binary changed")
    binary.write_bytes(original)
    other = binary.with_name("other.so")
    other.write_bytes(original)
    record_path = Path(distribution.locate_file("simple_knn/_real2sim_build.json"))
    record = records["simple-knn"]
    record["binaries"] = {str(Path(relative).with_name(other.name)): record["binaries"][relative]}
    record_path.write_text(json.dumps(record))
    reject("import does not resolve")
    assert kernels == [[0]]
print(json.dumps({"records": len(records), "rejections": rejections}))
