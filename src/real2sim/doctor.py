# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""One preflight for dependencies, artifacts, GPU/driver, disk, and permissions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

from real2sim._bootstrap import yaml

from .artifacts import (
    ArtifactVerification,
    ManifestError,
    load_manifest,
    verify_artifacts,
    verify_artifacts_with_snapshot,
)
from .gpu_binding import GpuBindingError, resolve_gpu_contract
from .provenance import ProvenanceError, collect_git_repository, collect_lockfiles, collect_nvidia


class DoctorError(ValueError):
    """Raised when a support profile or host preflight is invalid."""


def _load_profile(path: Path, name: str) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DoctorError(f"could not read support matrix {path}: {exc}") from exc
    if not isinstance(loaded, dict) or loaded.get("schema_version") != 1:
        raise DoctorError("support matrix must use schema_version: 1")
    profiles = loaded.get("profiles")
    if not isinstance(profiles, dict) or name not in profiles:
        raise DoctorError(f"unknown accelerator profile: {name}")
    profile = profiles[name]
    if not isinstance(profile, dict):
        raise DoctorError(f"support profile {name} must be a mapping")
    required = {"kind", "architecture", "python", "cuda", "driver", "compiler"}
    if set(profile) != required:
        raise DoctorError(
            f"support profile {name} keys must be exactly: {', '.join(sorted(required))}"
        )
    if profile["kind"] not in {"cpu", "cuda"}:
        raise DoctorError(f"support profile {name}.kind must be cpu or cuda")
    if re.fullmatch(r"\d+\.\d+\.\d+", str(profile["python"])) is None:
        raise DoctorError(
            f"support profile {name}.python must be major.minor.patch"
        )
    compiler = profile["compiler"]
    if (
        not isinstance(compiler, dict)
        or not {"command", "major"} <= set(compiler)
        or set(compiler) - {"command", "major", "cxx_command"}
    ):
        raise DoctorError(f"support profile {name}.compiler must contain command and major")
    if profile["kind"] == "cuda":
        if not isinstance(profile["driver"], dict) or set(profile["driver"]) != {"minimum"}:
            raise DoctorError(f"support profile {name}.driver must contain minimum")
        if (
            not isinstance(profile["cuda"], dict)
            or not {"major", "minor", "source"} <= set(profile["cuda"])
            or set(profile["cuda"]) - {"major", "minor", "source", "pytorch_index", "toolkit_path"}
        ):
            raise DoctorError(
                f"support profile {name}.cuda must contain major, minor, and source"
            )
        cuda_major = profile["cuda"]["major"]
        cuda_minor = profile["cuda"]["minor"]
        if isinstance(cuda_major, bool) or not isinstance(cuda_major, int) or cuda_major < 1:
            raise DoctorError(f"support profile {name}.cuda.major must be a positive integer")
        if isinstance(cuda_minor, bool) or not isinstance(cuda_minor, int) or cuda_minor < 0:
            raise DoctorError(
                f"support profile {name}.cuda.minor must be a non-negative integer"
            )
        if profile["cuda"]["source"] != "torch":
            raise DoctorError(f"support profile {name}.cuda.source must be torch")
    elif profile["driver"] is not None or profile["cuda"] is not None:
        raise DoctorError(f"CPU support profile {name} must use null cuda and driver")
    return {"name": name, **profile}


def _run_probe(
    command: Sequence[str], environment: Mapping[str, str] | None = None
) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env=None if environment is None else dict(environment),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    detail = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, detail


def _check_python_dependencies(
    environment: Mapping[str, str] | None,
) -> tuple[bool, str]:
    """Check installed distributions without source-only package metadata."""

    # Clean-checkout CI invokes doctor with PYTHONPATH=src. A preceding package
    # build leaves src/*.egg-info behind, which pip would otherwise mistake for
    # an installed full runtime while checking the deliberately small CI venv.
    probe_environment = dict(os.environ if environment is None else environment)
    probe_environment.pop("PYTHONPATH", None)
    search_path = probe_environment.get("PATH")
    if importlib.util.find_spec("pip") is not None:
        return _run_probe(
            (sys.executable, "-m", "pip", "check"), probe_environment
        )
    uv = shutil.which("uv", path=search_path)
    if uv is not None:
        return _run_probe(
            (uv, "pip", "check", "--python", sys.executable),
            probe_environment,
        )
    return False, "neither pip nor uv is available"


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.search(r"\d+(?:\.\d+)+", value)
    if match is None:
        raise DoctorError(f"could not parse version: {value!r}")
    return tuple(int(part) for part in match.group(0).split("."))


def _check_locked_uv_environment(
    repo_root: Path,
    environment: Mapping[str, str] | None,
) -> tuple[bool, str]:
    search_path = environment.get("PATH") if environment is not None else None
    uv = shutil.which("uv", path=search_path)
    if uv is None:
        return False, "uv is required to verify the production environment against uv.lock"
    probe_environment = dict(os.environ if environment is None else environment)
    probe_environment["UV_PROJECT_ENVIRONMENT"] = str(Path(sys.prefix).resolve())
    return _run_probe(
        (
            uv,
            "sync",
            "--project",
            str(repo_root),
            "--frozen",
            "--check",
            "--inexact",
            "--python",
            sys.executable,
        ),
        probe_environment,
    )


# A kernel built for another GPU architecture imports fine and fails only at launch.
_GAUSSIAN_KERNEL_PROBE = """
import sys
import torch
import diff_gaussian_rasterization._C as rasterizer
import diff_gaussian_rasterization_ours._C as rasterizer_ours
import simple_knn._C as knn

if torch.cuda.device_count() < 1:
    raise SystemExit("no visible CUDA device")
for index in [int(arg) for arg in sys.argv[1:]] or range(torch.cuda.device_count()):
    device = f"cuda:{index} compute capability %d.%d" % torch.cuda.get_device_capability(index)
    try:
        with torch.cuda.device(index):  # the extensions have no device guard
            points = torch.rand((64, 3), device=index) + torch.tensor([0.0, 0.0, 1.0], device=index)
            eye = torch.eye(4, device=index)
            ran = [
                bool((knn.distCUDA2(points) > 0).all()),
                bool(rasterizer.mark_visible(points, eye, eye).all()),
                bool(rasterizer_ours.mark_visible(points, eye, eye).all()),
            ]
            torch.cuda.synchronize(index)
    except Exception as exc:  # a missing kernel image surfaces as an unrelated CUDA error
        raise SystemExit(f"{device}: {type(exc).__name__}: {exc}")
    if not all(ran):
        raise SystemExit(f"{device}: kernels ran {ran}")
    print(f"{device}: ok")
"""


def _check_gaussian_cuda_extensions(
    environment: Mapping[str, str] | None,
    devices: Sequence[int] = (),
) -> tuple[bool, Any]:
    """Run the kernels on ``devices``, or on every visible GPU when it is empty."""

    probe_environment = dict(os.environ if environment is None else environment)
    probe_environment.pop("LD_LIBRARY_PATH", None)
    return _run_probe(
        (sys.executable, "-c", _GAUSSIAN_KERNEL_PROBE, *(str(index) for index in devices)),
        probe_environment,
    )


def _check_gaussian_build_records(repo_root, environment):
    return _run_probe(
        (sys.executable, str(repo_root / "scripts/cuda_extension_build/inputs.py"),
         str(repo_root), "--check", "--runtime"),
        environment,
    )


def run_doctor(
    *,
    repo_root: Path,
    manifest_path: Path,
    artifact_ids: Sequence[str],
    support_matrix_path: Path,
    profile_name: str,
    output_path: Path,
    min_free_gib: float,
    production: bool,
    lockfiles: Sequence[str],
    environment: Mapping[str, str] | None = None,
    capture_artifact_verification: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    artifact_verification: ArtifactVerification | None = None

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "status": "ok" if ok else "error", "detail": detail})

    profile = _load_profile(support_matrix_path, profile_name)
    if production:
        check(
            "production_profile",
            profile["kind"] == "cuda",
            {"required_kind": "cuda", "actual_kind": profile["kind"]},
        )
    expected_arch = profile["architecture"]
    actual_arch = platform.machine()
    check("architecture", expected_arch in {"any", actual_arch}, {"expected": expected_arch, "actual": actual_arch})
    expected_python = str(profile["python"])
    actual_python = platform.python_version()
    check("python", actual_python == expected_python, {"expected": expected_python, "actual": actual_python})

    requested_output = output_path.expanduser().resolve()
    output_anchor = requested_output
    while not output_anchor.exists() and output_anchor != output_anchor.parent:
        output_anchor = output_anchor.parent
    output_ok = output_anchor.is_dir() and os.access(
        output_anchor, os.W_OK | os.X_OK
    )
    check(
        "output_writable",
        output_ok,
        {"requested": str(requested_output), "existing_parent": str(output_anchor)},
    )
    usage = shutil.disk_usage(output_anchor)
    free_gib = usage.free / (1024 ** 3)
    check("disk", free_gib >= min_free_gib, {"free_gib": round(free_gib, 3), "required_gib": min_free_gib})

    try:
        git_record = collect_git_repository(repo_root, production=production)
        check("git", git_record["commit"] is not None, git_record)
    except ProvenanceError as exc:
        check("git", False, str(exc))

    lock_records: list[dict[str, Any]] = []
    try:
        lock_records = collect_lockfiles(repo_root, lockfiles, production=production)
        check("lockfiles", True, lock_records)
    except ProvenanceError as exc:
        check("lockfiles", False, str(exc))

    nvidia = collect_nvidia(environment) if profile["kind"] == "cuda" else None
    launch_gpu_roles = environment is not None and any(
        name in environment
        for name in ("GS_CUDA_DEVICES", "POLICY_CUDA_DEVICE", "SIM_CUDA_DEVICES")
    )
    gpu_bindings: dict[str, Any] | None = None
    if profile["kind"] == "cuda" and launch_gpu_roles:
        if nvidia is None:
            check(
                "gpu_binding",
                False,
                "nvidia-smi returned no GPU mapping for the configured selectors",
            )
        else:
            try:
                gpu_bindings = resolve_gpu_contract(
                    os.environ if environment is None else environment,
                    nvidia=nvidia,
                )
            except GpuBindingError as exc:
                check("gpu_binding", False, str(exc))
            else:
                check("gpu_binding", True, gpu_bindings)
    root_uv_lock = (repo_root / "uv.lock").resolve()
    if production and any(
        Path(record["path"]).resolve() == root_uv_lock for record in lock_records
    ):
        locked_ok, locked_detail = _check_locked_uv_environment(repo_root, environment)
        check("locked_environment", locked_ok, locked_detail)
        run_devices: list[int] = []
        if gpu_bindings is not None:
            # The probe inherits this environment, so an allocation slot is its CUDA device index.
            run_devices = sorted(
                {
                    entry["allocation_slot"]
                    for entries in gpu_bindings["roles"].values()
                    for entry in entries
                }
            )
        records_ok, records_detail = _check_gaussian_build_records(repo_root, environment)
        check("gaussian_build_records", records_ok, records_detail)
        if records_ok:
            extensions_ok, extensions_detail = _check_gaussian_cuda_extensions(
                environment, run_devices
            )
            check("gaussian_cuda_extensions", extensions_ok, extensions_detail)

    search_path = environment.get("PATH") if environment is not None else None
    dependencies_ok, dependencies_detail = _check_python_dependencies(environment)
    check("python_dependencies", dependencies_ok, dependencies_detail)

    runtime_tools = {}
    for name in ("git", "ffmpeg"):
        resolved = shutil.which(name, path=search_path)
        environment_tool = Path(sys.prefix) / "bin" / name
        if resolved is None and environment_tool.is_file() and os.access(environment_tool, os.X_OK):
            resolved = str(environment_tool)
        runtime_tools[name] = resolved
    tools_ok = runtime_tools["git"] is not None and (
        not production or runtime_tools["ffmpeg"] is not None
    )
    check("runtime_tools", tools_ok, runtime_tools)

    try:
        artifact_manifest = load_manifest(manifest_path)
        if capture_artifact_verification:
            candidate_verification = verify_artifacts_with_snapshot(
                artifact_manifest,
                artifact_ids,
                repo_root=repo_root,
                environment=environment,
            )
            artifact_records = candidate_verification.records
        else:
            candidate_verification = None
            artifact_records = verify_artifacts(
                artifact_manifest,
                artifact_ids,
                repo_root=repo_root,
                environment=environment,
            )
        if production:
            fixture_ids = sorted(
                record.artifact_id
                for record in artifact_records
                if record.access == "fixture"
            )
            if fixture_ids:
                raise ManifestError(
                    "production doctor cannot use fixture artifacts: "
                    + ", ".join(fixture_ids)
                )
        artifact_verification = candidate_verification
        check("artifacts", True, [record.as_dict() for record in artifact_records])
    except ManifestError as exc:
        check("artifacts", False, str(exc))

    accelerator_ok = profile["kind"] == "cpu" or nvidia is not None
    check("accelerator", accelerator_ok, {"profile": profile, "detected": nvidia})

    if profile["kind"] == "cuda" and nvidia is not None:
        minimum_driver = _version_tuple(str(profile["driver"]["minimum"]))
        observed = [gpu["driver_version"] for gpu in nvidia["gpus"]]
        drivers_ok = all(_version_tuple(value) >= minimum_driver for value in observed)
        check(
            "nvidia_driver",
            drivers_ok,
            {"minimum": profile["driver"]["minimum"], "observed": observed},
        )

        cuda_probe_ok, cuda_version = _run_probe(
            (
                sys.executable,
                "-c",
                "import torch; print(torch.version.cuda or '')",
            ),
            environment,
        )
        expected_cuda = (
            int(profile["cuda"]["major"]),
            int(profile["cuda"]["minor"]),
        )
        observed_cuda: tuple[int, ...] | None = None
        if cuda_probe_ok:
            try:
                observed_cuda = _version_tuple(cuda_version)[:2]
            except DoctorError:
                cuda_probe_ok = False
        check(
            "cuda_runtime",
            cuda_probe_ok and observed_cuda == expected_cuda,
            {
                "source": profile["cuda"]["source"],
                "expected": ".".join(str(part) for part in expected_cuda),
                "observed": cuda_version or None,
            },
        )

    report = {
        "schema_version": 1,
        "profile": profile_name,
        "production": production,
        "status": "ok" if all(item["status"] == "ok" for item in checks) else "error",
        "checks": checks,
    }
    if gpu_bindings is not None:
        report["gpu_bindings"] = gpu_bindings
    if capture_artifact_verification and artifact_verification is not None:
        report["_artifact_verification"] = artifact_verification
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.yaml"))
    parser.add_argument("--artifact", action="append", default=[], dest="artifacts")
    parser.add_argument("--support-matrix", type=Path, default=Path("config/support-matrix.yaml"))
    parser.add_argument("--profile", default="cpu-fixture")
    parser.add_argument("--output-path", type=Path, default=Path("outputs"))
    parser.add_argument("--min-free-gib", type=float, default=0.01)
    parser.add_argument("--lockfile", action="append", default=[], dest="lockfiles")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = run_doctor(
            repo_root=args.repo_root.resolve(),
            manifest_path=args.manifest,
            artifact_ids=args.artifacts,
            support_matrix_path=args.support_matrix,
            profile_name=args.profile,
            output_path=args.output_path,
            min_free_gib=args.min_free_gib,
            production=args.production,
            lockfiles=args.lockfiles,
        )
    except (DoctorError, ProvenanceError) as exc:
        print(f"doctor error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for check in report["checks"]:
            print(f"{check['status']:5} {check['name']}: {check['detail']}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
