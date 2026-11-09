# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pytest
import yaml

from tests._helpers.launcher import base_config as _base_config


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_SCRIPT_PATH = REPO_ROOT / "scripts" / "run_policy_batch.sh"
# How a scene's background.json names an external table collision: a
# registered-desk bundle under the single input root.
EXTERNAL_COLLISION_DIRECTORY = (
    "${REAL2SIM_INPUTS_ROOT}/registered-world/table/coacd"
)

from real2sim import launcher  # noqa: E402
from real2sim import artifacts as real2sim_artifacts  # noqa: E402
from real2sim import doctor  # noqa: E402
from real2sim import doctor as real2sim_doctor  # noqa: E402
from real2sim import preflight as real2sim_preflight  # noqa: E402
from real2sim import artifacts_binding  # noqa: E402
from real2sim import cli as real2sim_cli  # noqa: E402
from real2sim import launch_provenance  # noqa: E402
from real2sim import preflight  # noqa: E402
from real2sim.gpu_binding import (  # noqa: E402
    GPU_BINDINGS_ENV,
    resolve_gpu_contract,
)
from real2sim.runner import execute as runner_execute  # noqa: E402
from real2sim.artifacts import (  # noqa: E402
    load_manifest,
)


@pytest.fixture(autouse=True)
def _isolate_launcher_environment(monkeypatch, isolated_environ, request):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    if request.instance is not None:
        request.instance.isolated_environ = isolated_environ



class RunPolicyFromConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        # CPU fixture/config tests must not inherit the scheduler's GPU
        # visibility boundary. Tests of the GPU contract set it explicitly.
        environment = dict(os.environ)
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        patcher = mock.patch.dict(os.environ, environment, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write_config(self, root: Path, config: dict) -> Path:
        config_dir = root / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        path = config_dir / "rollout.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def _load_temp(self, config: dict):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        path = self._write_config(root, config)
        return launcher.load_rollout_config(
            path,
            repo_root=root,
            timestamp="20260730_120000",
        )


    def test_schema_v2_completion_rejects_diagnostic_summary(self):
        plan = launcher.load_rollout_config(
            REPO_ROOT / "config" / "release" / "fixture_v2.yaml",
            timestamp="20990101_000001",
        )
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        run_root = Path(temporary.name) / "official-run"
        plan["run_root"] = run_root
        plan["resolved"]["output"]["run_root"] = str(run_root)
        run_root.mkdir(parents=True)
        scene = plan["resolved"]["scenes"][0]["tag"]
        outdir = run_root / scene / "run_01"
        outdir.mkdir(parents=True)
        summary = outdir / "sim_summary.json"
        video = outdir / f"{scene}_run_01.mp4"
        summary.write_text(
            json.dumps(
                {
                    "frames": 1,
                    "rendered_frames": 1,
                    "wall_stopped_early": False,
                    "execution_class": "diagnostic",
                }
            ),
            encoding="utf-8",
        )
        video.write_bytes(b"video")
        with (run_root / "outputs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["scene", "run", "exit_code", "video", "summary"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "scene": scene,
                    "run": "run_01",
                    "exit_code": "0",
                    "video": str(video),
                    "summary": str(summary),
                }
            )

        errors = launcher.rollout_completion_errors(plan)

        self.assertTrue(any("expected official" in error for error in errors))
        self.assertTrue(any("asset contract" in error for error in errors))
        self.assertTrue(any("scene contract" in error for error in errors))

    def test_ruckig_preflight_follows_the_runtime_path_not_the_rig_name(self):
        # The probe follows the Ruckig path the worker executes: fr3_duo shares
        # the FR3 joint_ik trajectory.
        for physics, robot in (
            ("dynamic", "fr3"),
            ("dynamic", "fr3_duo"),
        ):
            config = _base_config()
            config["simulation"]["fr3_controller"] = "joint_ik"
            config["simulation"]["parameters"]["physics"] = physics
            plan = self._load_temp(config)
            plan["resolved"]["robot"] = robot
            with self.subTest(physics=physics, robot=robot):
                with mock.patch.object(
                    launcher.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(
                        [], 1, "", "wrong version"
                    ),
                ):
                    with self.assertRaisesRegex(
                        launcher.ConfigError, "ruckig==0.19.4"
                    ):
                        launcher.validate_joint_trajectory_runtime(plan)

        for physics, robot in (("kinematic", "fr3"), ("kinematic", "fr3_duo")):
            config = _base_config()
            config["simulation"]["fr3_controller"] = "joint_ik"
            config["simulation"]["parameters"]["physics"] = physics
            plan = self._load_temp(config)
            plan["resolved"]["robot"] = robot
            with self.subTest(physics=physics, robot=robot):
                with mock.patch.object(launcher.subprocess, "run") as run:
                    launcher.validate_joint_trajectory_runtime(plan)
                run.assert_not_called()

        # A rig with no known Ruckig rule is refused, never silently skipped.
        plan = self._load_temp(_base_config())
        plan["resolved"]["robot"] = "other_robot"
        with mock.patch.object(launcher.subprocess, "run") as run:
            with self.assertRaisesRegex(
                launcher.ConfigError, "not a supported simulation rig"
            ):
                launcher.validate_joint_trajectory_runtime(plan)
        run.assert_not_called()

    def test_schema_v2_runtime_uses_exact_common_asset_and_scene_contracts(self):
        plan = launcher.load_rollout_config(
            REPO_ROOT / "config" / "release" / "fixture_v2.yaml",
            timestamp="20990101_000000",
        )
        scene = plan["resolved"]["scenes"][0]
        collision_directory = (
            REPO_ROOT / "tests" / "fixtures" / "artifacts" / "registered-desk" / "coacd"
        )
        scene_report = {
            "id": "tiny-fixture-scene-v2",
            "name": "tiny_fixture_scene",
            "manifest_path": Path(scene["path"]) / "background.json",
            "ply_sha256": "d" * 64,
            "w2gs_matrix": np.eye(4),
            "collision_directory": collision_directory,
            "collision_files": (collision_directory / "part.obj",),
            "contract_digest": "b" * 64,
            "collision_tree_sha256": "c" * 64,
        }
        asset_report = {
            "manifest_path": REPO_ROOT / "robosuite_env" / "assets" / "manifest.json",
            "contract_digest": "a" * 64,
        }

        with mock.patch.object(
            artifacts_binding,
            "validate_repository_assets",
            return_value=asset_report,
        ) as validate_assets, mock.patch.object(
            artifacts_binding,
            "validate_scene_contract",
            return_value=scene_report,
        ) as validate_scene:
            report = launcher.validate_runtime_paths(plan)
            artifact_roles = launcher._artifact_role_paths(plan, report)

        validate_assets.assert_called_once_with(
            REPO_ROOT / "robosuite_env" / "assets",
            bundle_ids=("yubi_right", "yubi_left", "fr3", "cup", "plate"),
        )
        validate_scene.assert_called_once_with(scene["path"], strict=True)
        self.assertEqual(report["assets"]["contract_digest"], "a" * 64)
        self.assertEqual(
            report["scenes"]["tiny_fixture_scene"]["collision_files"],
            [str(collision_directory / "part.obj")],
        )
        self.assertEqual(
            artifact_roles["registered-desk"],
            (collision_directory.parent, True),
        )

        without_binding = copy.deepcopy(plan)
        del without_binding["resolved"]["reproducibility"]["artifacts"][
            "registered-desk"
        ]
        with self.assertRaisesRegex(
            launcher.ConfigError,
            "outside the scene artifact requires.*registered-desk",
        ):
            launcher._artifact_role_paths(without_binding, report)

        internal_report = copy.deepcopy(report)
        internal_collision = Path(scene["path"]) / "table_collision"
        internal_report["scenes"]["tiny_fixture_scene"][
            "collision_directory"
        ] = str(internal_collision)
        internal_roles = launcher._artifact_role_paths(
            without_binding,
            internal_report,
        )
        self.assertNotIn("registered-desk", internal_roles)


    def test_check_runtime_and_execute_apply_the_same_joint_ik_gate(self):
        config = _base_config()
        config["simulation"]["fr3_controller"] = "joint_ik"
        plan = self._load_temp(config)
        contract_report = {"assets": {"contract_digest": "a" * 64}, "scenes": {}}
        doctor_report = {"status": "ok", "checks": []}

        with mock.patch.object(
            real2sim_cli, "load_rollout_config", return_value=plan
        ), mock.patch.object(
            real2sim_cli, "capture_launch_runtime_generation"
        ), mock.patch.object(
            real2sim_cli, "capture_runtime_selection"
        ), mock.patch.object(
            preflight, "validate_runtime_paths", return_value=contract_report
        ), mock.patch.object(
            preflight, "validate_locked_imports"
        ) as check_runtime_locked, mock.patch.object(
            preflight, "validate_joint_trajectory_runtime"
        ) as check_runtime_joint, mock.patch.object(
            preflight, "validate_production_preflight", return_value=doctor_report
        ), mock.patch.object(
            real2sim_cli, "prepare_run_manifest", return_value=None
        ) as prepare_check:
            self.assertEqual(
                launcher.main(
                    [
                        "--config",
                        str(plan["config_path"]),
                        "--check-runtime",
                        "--env-file",
                        str(self.isolated_environ),
                    ]
                ),
                0,
            )
        check_runtime_locked.assert_called_once_with(plan)
        check_runtime_joint.assert_called_once_with(plan)
        self.assertEqual(
            prepare_check.call_args.args[1],
            {
                "contracts": contract_report,
                "doctor": doctor_report,
                "_locked_imports": check_runtime_locked.return_value,
            },
        )

        with mock.patch.object(
            preflight, "validate_runtime_paths", return_value=contract_report
        ), mock.patch.object(
            preflight, "validate_locked_imports"
        ) as execute_locked, mock.patch.object(
            preflight, "validate_joint_trajectory_runtime"
        ) as execute_joint, mock.patch.object(
            preflight, "validate_production_preflight", return_value=doctor_report
        ), mock.patch.object(
            runner_execute, "prepare_run_manifest", return_value=None
        ) as prepare_execute, mock.patch.object(
            runner_execute, "persist_config_provenance"
        ), mock.patch.object(
            launcher.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0),
        ), mock.patch.object(
            runner_execute,
            "summarize_run",
            return_value=(0, {"total_runs": plan["expected_rollouts"]}, {}),
        ):
            self.assertEqual(launcher.execute(plan), 0)
        execute_locked.assert_called_once_with(plan)
        execute_joint.assert_called_once_with(plan)
        self.assertEqual(
            prepare_execute.call_args.args[1],
            {
                "contracts": contract_report,
                "doctor": doctor_report,
                "_locked_imports": execute_locked.return_value,
            },
        )

    def test_execute_passes_and_rechecks_runtime_source_seal(self):
        plan = self._load_temp(_base_config())
        manifest_path = Path(plan["run_root"]) / "runtime_source_manifest.json"
        manifest_digest = "a" * 64
        baseline = {"generation": "captured-before-config-parse"}

        with mock.patch.object(
            runner_execute,
            "validate_launch_preflight",
            return_value={"contracts": None, "doctor": None},
        ), mock.patch.object(
            runner_execute, "prepare_run_manifest", return_value=None
        ), mock.patch.object(
            runner_execute, "persist_config_provenance"
        ), mock.patch.object(
            runner_execute,
            "seal_runtime_source_manifest",
            return_value=(manifest_path, manifest_digest),
        ) as seal, mock.patch.object(
            runner_execute, "verify_runtime_manifest"
        ) as verify, mock.patch.object(
            launcher.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run, mock.patch.object(
            runner_execute,
            "summarize_run",
            return_value=(0, {"total_runs": plan["expected_rollouts"]}, {}),
        ):
            self.assertEqual(
                launcher.execute(plan, runtime_source_baseline=baseline), 0
            )

        seal.assert_called_once_with(plan, baseline)
        verify.assert_called_once_with(
            manifest_path,
            expected_sha256=manifest_digest,
        )
        runner_environment = run.call_args.kwargs["env"]
        self.assertEqual(
            runner_environment[launcher.RUNTIME_SOURCE_MANIFEST_ENV],
            str(manifest_path),
        )
        self.assertEqual(
            runner_environment[launcher.RUNTIME_SOURCE_MANIFEST_SHA256_ENV],
            manifest_digest,
        )

    def test_production_manifest_preparation_uses_common_launch_preflight(self):
        plan = launcher.load_rollout_config(
            REPO_ROOT / "config" / "release" / "fixture_v2.yaml",
            timestamp="20990101_000003",
        )
        plan["resolved"]["reproducibility"]["production"] = True
        with mock.patch.object(
            launch_provenance,
            "validate_launch_preflight",
            side_effect=launcher.ConfigError("common preflight sentinel"),
        ) as validate_preflight, self.assertRaisesRegex(
            launcher.ConfigError,
            "common preflight sentinel",
        ):
            launcher.prepare_run_manifest(plan)
        validate_preflight.assert_called_once_with(plan)


def test_launcher_facade_preserves_doctor_exports():
    assert launcher.DoctorError is doctor.DoctorError
    assert launcher.run_doctor is preflight.run_doctor


def test_production_doctor_rejects_cpu_profile_and_fixture_artifact(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        real2sim_doctor,
        "collect_git_repository",
        lambda _path, *, production: {
            "path": str(REPO_ROOT),
            "commit": "0" * 40,
            "dirty": False,
            "remote": "https://example.invalid/real2sim.git",
            "submodules": [],
        },
    )
    report = real2sim_doctor.run_doctor(
        repo_root=REPO_ROOT,
        manifest_path=REPO_ROOT / "artifacts" / "manifest.yaml",
        artifact_ids=["tiny-checkpoint-v1"],
        support_matrix_path=REPO_ROOT / "config" / "support-matrix.yaml",
        profile_name="cpu-fixture",
        output_path=tmp_path,
        min_free_gib=0.0,
        production=True,
        lockfiles=["requirements-lock.ci.txt"],
    )
    checks = {item["name"]: item for item in report["checks"]}

    assert report["status"] == "error"
    assert checks["production_profile"]["status"] == "error"
    assert checks["artifacts"]["status"] == "error"
    assert "fixture artifacts" in checks["artifacts"]["detail"]

def test_cuda_doctor_without_launch_roles_does_not_invent_host_gpu_zero(
    monkeypatch, tmp_path: Path
):
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "GS_CUDA_DEVICES",
        "POLICY_CUDA_DEVICE",
        "SIM_CUDA_DEVICES",
    ):
        monkeypatch.delenv(name, raising=False)
    nvidia = {
        "gpus": [
            {
                "index": "3",
                "name": "fixture GPU",
                "uuid": "GPU-three",
                "driver_version": "999.0",
                "compute_capability": "8.9",
            }
        ],
        "nvidia_smi_available": True,
        "tool_version": "fixture",
    }
    monkeypatch.setattr(real2sim_doctor, "collect_nvidia", lambda _env: nvidia)
    monkeypatch.setattr(
        real2sim_doctor,
        "collect_git_repository",
        lambda _path, *, production: {
            "path": str(REPO_ROOT),
            "commit": "0" * 40,
            "dirty": False,
            "remote": "https://example.invalid/real2sim.git",
            "submodules": [],
        },
    )
    monkeypatch.setattr(
        real2sim_doctor,
        "collect_lockfiles",
        lambda _root, _lockfiles, *, production: [],
    )
    monkeypatch.setattr(
        real2sim_doctor,
        "_check_python_dependencies",
        lambda _environment: (True, {}),
    )
    monkeypatch.setattr(
        real2sim_doctor,
        "_run_probe",
        lambda _command, _environment: (True, "12.8"),
    )

    def run(environment):
        return real2sim_doctor.run_doctor(
            repo_root=REPO_ROOT,
            manifest_path=REPO_ROOT / "artifacts" / "manifest.yaml",
            artifact_ids=[],
            support_matrix_path=REPO_ROOT / "config" / "support-matrix.yaml",
            profile_name="cuda12-x86_64",
            output_path=tmp_path,
            min_free_gib=0.0,
            production=False,
            lockfiles=[],
            environment=environment,
        )

    standalone = run(None)
    standalone_checks = {item["name"]: item for item in standalone["checks"]}
    assert standalone_checks["accelerator"]["status"] == "ok"
    assert "gpu_binding" not in standalone_checks
    assert "gpu_bindings" not in standalone

    launch = run(
        {
            "GS_CUDA_DEVICES": "3",
            "POLICY_CUDA_DEVICE": "3",
            "SIM_CUDA_DEVICES": "3",
        }
    )
    launch_checks = {item["name"]: item for item in launch["checks"]}
    assert launch_checks["gpu_binding"]["status"] == "ok"
    assert launch["gpu_bindings"]["selection_mode"] == "host_ordinals"
    assert launch["gpu_bindings"]["roles"]["policy"] == [
        {
            "allocation_slot": 3,
            "cuda_token": "3",
            "egl_physical_index": 3,
        }
    ]

def test_cuda_doctor_needs_no_compiler_or_nvcc_on_the_run_host(monkeypatch, tmp_path: Path):
    # A run from another shell, without the build environment, must not ask for a rebuild.
    monkeypatch.setattr(
        real2sim_doctor,
        "collect_git_repository",
        lambda _path, *, production: {"path": str(REPO_ROOT), "commit": "0" * 40, "dirty": False,
                                      "remote": "https://example.invalid/real2sim.git", "submodules": []},
    )
    monkeypatch.setattr(real2sim_doctor, "_run_probe", lambda _command, _environment: (True, "12.8"))
    report = real2sim_doctor.run_doctor(
        repo_root=REPO_ROOT,
        manifest_path=REPO_ROOT / "artifacts" / "manifest.yaml",
        artifact_ids=[],
        support_matrix_path=REPO_ROOT / "config" / "support-matrix.yaml",
        profile_name="cuda12-x86_64",
        output_path=tmp_path,
        min_free_gib=0.0,
        production=False,
        lockfiles=[],
        environment={"PATH": str(tmp_path)},
    )
    assert not {"compiler", "cxx_compiler", "cuda_toolkit"} & {item["name"] for item in report["checks"]}


def test_support_profiles_require_an_exact_python_patch(tmp_path):
    source = yaml.safe_load(
        (REPO_ROOT / "config" / "support-matrix.yaml").read_text(
            encoding="utf-8"
        )
    )
    source["profiles"]["cpu-fixture"]["python"] = "3.11"
    support_matrix = tmp_path / "support-matrix.yaml"
    support_matrix.write_text(
        yaml.safe_dump(source, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(real2sim_doctor.DoctorError, match="major.minor.patch"):
        real2sim_doctor._load_profile(support_matrix, "cpu-fixture")
    with pytest.raises(launcher.ConfigError, match="major.minor.patch"):
        launcher._load_support_profile(support_matrix, "cpu-fixture")

def test_locked_uv_environment_uses_current_interpreter_and_inexact_check(
    monkeypatch, tmp_path
):
    captured: dict[str, object] = {}

    monkeypatch.setattr(real2sim_doctor.shutil, "which", lambda *_args, **_kwargs: "/bin/uv")

    def fake_probe(command, environment):
        captured["command"] = command
        captured["environment"] = environment
        return True, "environment is synchronized"

    monkeypatch.setattr(real2sim_doctor, "_run_probe", fake_probe)
    ok, detail = real2sim_doctor._check_locked_uv_environment(
        tmp_path, {"PATH": "/bin", "LANG": "C.UTF-8"}
    )

    assert ok is True
    assert detail == "environment is synchronized"
    assert captured["command"] == (
        "/bin/uv",
        "sync",
        "--project",
        str(tmp_path),
        "--frozen",
        "--check",
        "--inexact",
        "--python",
        sys.executable,
    )
    assert captured["environment"] == {
        "PATH": "/bin",
        "LANG": "C.UTF-8",
        "UV_PROJECT_ENVIRONMENT": str(Path(sys.prefix).resolve()),
    }

def test_doctor_dependency_check_ignores_source_only_pythonpath(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        real2sim_doctor.importlib.util,
        "find_spec",
        lambda name: object() if name == "pip" else None,
    )

    def fake_probe(command, environment):
        captured["command"] = command
        captured["environment"] = environment
        return True, "No broken requirements found."

    monkeypatch.setattr(real2sim_doctor, "_run_probe", fake_probe)

    ok, detail = real2sim_doctor._check_python_dependencies(
        {
            "PATH": "/bin",
            "LANG": "C.UTF-8",
            "PYTHONPATH": "/checkout/src",
        }
    )

    assert ok is True
    assert detail == "No broken requirements found."
    assert captured["command"] == (sys.executable, "-m", "pip", "check")
    assert captured["environment"] == {
        "PATH": "/bin",
        "LANG": "C.UTF-8",
    }

def test_gaussian_cuda_extension_check_runs_kernels_without_ld_library_path(monkeypatch):
    calls = []

    def fake_probe(command, environment):
        calls.append((command, environment))
        return False, "ImportError: simple_knn._C"

    monkeypatch.setattr(real2sim_doctor, "_run_probe", fake_probe)
    ok, detail = real2sim_doctor._check_gaussian_cuda_extensions(
        {"PATH": "/bin", "LD_LIBRARY_PATH": "/compiler/lib", "GS_CUDA_DEVICES": "0"}
    )

    assert (ok, detail) == (False, "ImportError: simple_knn._C")
    assert calls == [
        (
            (sys.executable, "-c", real2sim_doctor._GAUSSIAN_KERNEL_PROBE),
            {"PATH": "/bin", "GS_CUDA_DEVICES": "0"},
        )
    ]


@pytest.mark.parametrize(
    ("knn_kernel", "expected"),
    [
        ("distCUDA2 = lambda points: Result(True)", (True, "cuda:0 compute capability 12.0: ok")),
        (
            "distCUDA2 = lambda points: Result(False)",
            (False, "cuda:0 compute capability 12.0: kernels ran [False, True, True]"),
        ),
        (
            "def distCUDA2(points): raise MemoryError('cudaErrorMemoryAllocation')",
            (False, "cuda:0 compute capability 12.0: MemoryError: cudaErrorMemoryAllocation"),
        ),
    ],
)
def test_gaussian_kernel_probe_fails_when_a_kernel_did_not_run(tmp_path, knn_kernel, expected):
    (tmp_path / "torch.py").write_text(
        "import contextlib\n"
        "from types import SimpleNamespace\n"
        "class Result:\n"
        "    def __init__(self, ran=True): self.ran = ran\n"
        "    def __add__(self, other): return self\n"
        "    def __gt__(self, other): return self\n"
        "    def all(self): return self.ran\n"
        "rand = tensor = eye = lambda *args, **kwargs: Result()\n"
        "cuda = SimpleNamespace(device_count=lambda: 1, synchronize=lambda index: None,\n"
        "                       device=lambda index: contextlib.nullcontext(),\n"
        "                       get_device_capability=lambda index: (12, 0))\n",
        encoding="utf-8",
    )
    for package, kernel in (
        ("diff_gaussian_rasterization", "mark_visible = lambda *args: Result()"),
        ("diff_gaussian_rasterization_ours", "mark_visible = lambda *args: Result()"),
        ("simple_knn", knn_kernel),
    ):
        (tmp_path / package).mkdir()
        (tmp_path / package / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / package / "_C.py").write_text(
            f"from torch import Result\n{kernel}\n", encoding="utf-8"
        )

    assert real2sim_doctor._check_gaussian_cuda_extensions(
        {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(tmp_path)}
    ) == expected


def _two_gpu_extensions(tmp_path):
    """Fake extensions that fail like raw CUDA launches when the current device is wrong."""
    (tmp_path / "torch.py").write_text(
        "import contextlib\n"
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "LOG = Path(__file__).with_name('events.log')\n"
        "current = [0]\n"
        "class Tensor:\n"
        "    def __init__(self, device): self.device = device\n"
        "    def __add__(self, other): return self\n"
        "    def __gt__(self, other): return self\n"
        "    def all(self): return True\n"
        "def record(event):\n"
        "    with LOG.open('a') as log: log.write(event + '\\n')\n"
        "def allocate(*args, device, **kwargs):\n"
        "    record(f'alloc {device}')\n"
        "    return Tensor(device)\n"
        "def launch(name, tensor):\n"
        "    record(f'launch {name} current={current[0]} input={tensor.device}')\n"
        "    if current[0] != tensor.device:\n"
        "        raise RuntimeError('an illegal memory access was encountered')\n"
        "    return Tensor(tensor.device)\n"
        "@contextlib.contextmanager\n"
        "def device(index):\n"
        "    previous, current[0] = current[0], index\n"
        "    yield\n"
        "    current[0] = previous\n"
        "rand = tensor = eye = allocate\n"
        "cuda = SimpleNamespace(device_count=lambda: 2, device=device, synchronize=lambda index: None,\n"
        "                       get_device_capability=lambda index: ((12, 0), (8, 9))[index])\n",
        encoding="utf-8",
    )
    for package, kernel in (
        ("diff_gaussian_rasterization", "mark_visible = lambda points, *_: launch('mark_visible', points)"),
        ("diff_gaussian_rasterization_ours", "mark_visible = lambda points, *_: launch('mark_visible', points)"),
        ("simple_knn", "distCUDA2 = lambda points: launch('distCUDA2', points)"),
    ):
        (tmp_path / package).mkdir()
        (tmp_path / package / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / package / "_C.py").write_text(f"from torch import launch\n{kernel}\n", encoding="utf-8")
    return {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(tmp_path)}


def _probe_events(devices):
    events = []
    for index in devices:
        events += [f"alloc {index}"] * 3
        events += [
            f"launch {name} current={index} input={index}"
            for name in ("distCUDA2", "mark_visible", "mark_visible")
        ]
    return events


def test_gaussian_kernel_probe_launches_on_the_device_of_its_inputs(tmp_path):
    environment = _two_gpu_extensions(tmp_path)

    assert real2sim_doctor._check_gaussian_cuda_extensions(environment) == (
        True,
        "cuda:0 compute capability 12.0: ok\ncuda:1 compute capability 8.9: ok",
    )
    assert (tmp_path / "events.log").read_text().splitlines() == _probe_events((0, 1))


def test_gaussian_kernel_probe_runs_only_on_the_given_devices(tmp_path):
    environment = _two_gpu_extensions(tmp_path)

    assert real2sim_doctor._check_gaussian_cuda_extensions(environment, [1]) == (
        True,
        "cuda:1 compute capability 8.9: ok",
    )
    assert (tmp_path / "events.log").read_text().splitlines() == _probe_events((1,))


@pytest.mark.parametrize(
    ("roles", "expected_devices"),
    [
        ({"GS_CUDA_DEVICES": "1", "POLICY_CUDA_DEVICE": "1", "SIM_CUDA_DEVICES": "1"}, [1]),
        ({"GS_CUDA_DEVICES": "0", "POLICY_CUDA_DEVICE": "1", "SIM_CUDA_DEVICES": "0"}, [0, 1]),
    ],
)
@pytest.mark.parametrize("records_ok", [True, False])
def test_production_doctor_runs_the_kernels_only_on_the_bound_gpus(
    monkeypatch, tmp_path, roles, expected_devices, records_ok
):
    nvidia = {
        "gpus": [
            {"index": "0", "uuid": "GPU-a", "driver_version": "999.0", "compute_capability": "12.0"},
            {"index": "1", "uuid": "GPU-b", "driver_version": "999.0", "compute_capability": "12.0"},
        ],
    }
    monkeypatch.setattr(real2sim_doctor, "collect_nvidia", lambda _env: nvidia)
    monkeypatch.setattr(
        real2sim_doctor,
        "collect_git_repository",
        lambda _path, *, production: {"path": str(REPO_ROOT), "commit": "0" * 40},
    )
    monkeypatch.setattr(
        real2sim_doctor,
        "collect_lockfiles",
        lambda root, _lockfiles, *, production: [{"path": str(root / "uv.lock")}],
    )
    monkeypatch.setattr(
        real2sim_doctor, "_check_locked_uv_environment", lambda _root, _env: (True, "")
    )
    monkeypatch.setattr(real2sim_doctor, "_check_python_dependencies", lambda _env: (True, ""))
    monkeypatch.setattr(real2sim_doctor, "_run_probe", lambda _command, _env: (True, "12.8"))
    monkeypatch.setattr(
        real2sim_doctor, "_check_gaussian_build_records", lambda _root, _env: (records_ok, "fixture records")
    )
    probed = []

    def fake_kernel_check(environment, devices=()):
        probed.append(list(devices))
        return True, "ok"

    monkeypatch.setattr(real2sim_doctor, "_check_gaussian_cuda_extensions", fake_kernel_check)

    report = real2sim_doctor.run_doctor(
        repo_root=REPO_ROOT,
        manifest_path=REPO_ROOT / "artifacts" / "manifest.yaml",
        artifact_ids=[],
        support_matrix_path=REPO_ROOT / "config" / "support-matrix.yaml",
        profile_name="cuda12-x86_64",
        output_path=tmp_path,
        min_free_gib=0.0,
        production=True,
        lockfiles=["uv.lock"],
        environment={"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "GPU-b,GPU-a", **roles},
    )

    assert report["gpu_bindings"]["selection_mode"] == "visible_slots"
    assert probed == ([expected_devices] if records_ok else [])
    record_check = next(check for check in report["checks"] if check["name"] == "gaussian_build_records")
    assert record_check["status"] == ("ok" if records_ok else "error")

def test_prepare_run_manifest_reuses_doctor_artifact_verification(monkeypatch):
    plan = launcher.load_rollout_config(
        REPO_ROOT / "config" / "release" / "fixture_v2.yaml",
        timestamp="20260802_000000",
    )
    reproducibility = plan["resolved"]["reproducibility"]
    selected_ids = list(dict.fromkeys(reproducibility["artifacts"].values()))
    manifest = load_manifest(reproducibility["artifact_manifest"])
    verification = real2sim_artifacts.verify_artifacts_with_snapshot(
        manifest,
        selected_ids,
        repo_root=Path(plan["repo_root"]),
        environment=reproducibility["environment"],
    )
    captured: dict[str, object] = {}

    def fake_production_preflight(_plan, *, capture_artifact_verification):
        captured["capture_artifact_verification"] = capture_artifact_verification
        return {
            "status": "ok",
            "_artifact_verification": verification,
        }

    monkeypatch.setattr(
        real2sim_preflight,
        "validate_production_preflight",
        fake_production_preflight,
    )
    monkeypatch.setattr(
        real2sim_preflight, "validate_runtime_paths", lambda _plan: None
    )
    monkeypatch.setattr(
        real2sim_preflight,
        "validate_joint_trajectory_runtime",
        lambda _plan: None,
    )
    preflight = launcher.validate_launch_preflight(plan)

    def unexpected_verification(*_args, **_kwargs):
        raise AssertionError("prepare_run_manifest must not hash artifacts twice")

    monkeypatch.setattr(
        artifacts_binding, "verify_artifacts", unexpected_verification
    )
    run_manifest = launcher.prepare_run_manifest(plan, preflight)

    assert captured["capture_artifact_verification"] is True
    assert run_manifest is not None
    assert run_manifest["preflight"] == {
        "contracts": None,
        "doctor": {"status": "ok"},
    }
    assert run_manifest["artifacts"] == [
        record.as_dict() for record in verification.records
    ]


@pytest.mark.parametrize("visible_devices", ("5", "3,5"))
def test_prepare_run_manifest_persists_launcher_gpu_binding(visible_devices):
    plan = launcher.load_rollout_config(
        REPO_ROOT / "config" / "release" / "fixture_v2.yaml",
        timestamp="20260802_000001",
    )
    reproducibility = plan["resolved"]["reproducibility"]
    reproducibility["environment"]["CUDA_VISIBLE_DEVICES"] = visible_devices
    plan["resolved"]["runtime"].update(
        {
            "gs_cuda_devices": "0",
            "policy_cuda_device": "0",
            "sim_cuda_devices": "0",
        }
    )
    environment = launcher.build_runner_environment(plan)
    gpu_bindings = resolve_gpu_contract(
        environment,
        nvidia={
            "gpus": [
                {"index": str(index), "uuid": f"GPU-{index}"}
                for index in range(8)
            ]
        },
    )

    run_manifest = launcher.prepare_run_manifest(
        plan,
        {
            "contracts": None,
            "doctor": None,
            "gpu_bindings": gpu_bindings,
        },
    )

    assert run_manifest is not None
    assert json.loads(
        run_manifest["resolved_environment"][GPU_BINDINGS_ENV]
    ) == gpu_bindings


def test_production_schema_v2_requires_explicit_worker_gpu_assignments(tmp_path):
    source = yaml.safe_load(
        (
            REPO_ROOT
            / "config" / "release" / "fr3" / "flat" / "reading_room_cup_cup1trim.yaml"
        ).read_text(encoding="utf-8")
    )
    source["reproducibility"]["mode"] = "production"
    del source["runtime"]["gs_cuda_devices"]
    config_path = tmp_path / "missing-gpu-assignment.yaml"
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    fixture_root = REPO_ROOT / "portable-fixtures"
    environment = {
        "REAL2SIM_INPUTS_ROOT": str(fixture_root),
    }

    with mock.patch.dict(os.environ, environment, clear=False):
        with pytest.raises(launcher.ConfigError, match="gs_cuda_devices"):
            launcher.load_rollout_config(config_path)

def test_production_preflight_rejects_failed_gpu_binding(monkeypatch):
    plan = launcher.load_rollout_config(
        REPO_ROOT / "config" / "release" / "fixture_v2.yaml",
        timestamp="20260802_000000",
    )
    plan["resolved"]["reproducibility"]["production"] = True
    plan["resolved"]["runtime"].update(
        {
            "gs_cuda_devices": "7",
            "policy_cuda_device": "0",
            "sim_cuda_devices": "4",
        }
    )
    monkeypatch.setattr(
        real2sim_preflight,
        "run_doctor",
        lambda **_kwargs: {
            "status": "error",
            "checks": [
                {
                    "name": "gpu_binding",
                    "status": "error",
                    "detail": "configured CUDA device '7' is not a host GPU",
                }
            ],
        },
    )
    monkeypatch.setattr(
        real2sim_preflight,
        "_check_policy_python_runtime",
        lambda *_args: (True, {"version": "3.11.15"}),
    )

    with pytest.raises(launcher.ConfigError, match="gpu_binding.*7.*not a host GPU"):
        launcher.validate_production_preflight(plan)

def test_production_preflight_checks_derived_policy_gpu(monkeypatch):
    plan = launcher.load_rollout_config(
        REPO_ROOT / "config" / "release" / "fixture_v2.yaml",
        timestamp="20260802_000000",
    )
    plan["resolved"]["reproducibility"]["production"] = True
    plan["resolved"]["runtime"].update(
        {
            "gs_cuda_devices": "2,4",
            "policy_cuda_device": None,
            "sim_cuda_devices": "4",
        }
    )
    captured = {}

    def fake_doctor(**kwargs):
        captured.update(kwargs["environment"])
        gpu_bindings = launcher.resolve_gpu_contract(
            kwargs["environment"],
            nvidia={
                "gpus": [
                    {"index": "2", "uuid": "GPU-two"},
                    {"index": "4", "uuid": "GPU-four"},
                ]
            },
        )
        return {
            "status": "ok",
            "checks": [
                {
                    "name": "gpu_binding",
                    "status": "ok",
                    "detail": gpu_bindings,
                }
            ],
            "gpu_bindings": gpu_bindings,
        }

    monkeypatch.setattr(real2sim_preflight, "run_doctor", fake_doctor)
    monkeypatch.setattr(
        real2sim_preflight,
        "_check_policy_python_runtime",
        lambda *_args: (True, {"version": "3.11.15"}),
    )

    assert launcher.validate_production_preflight(plan)["status"] == "ok"
    assert captured["POLICY_CUDA_DEVICE"] == "2"

def test_production_preflight_rejects_policy_python_environment_drift(monkeypatch):
    plan = launcher.load_rollout_config(
        REPO_ROOT / "config" / "release" / "fixture_v2.yaml",
        timestamp="20260802_000000",
    )
    plan["resolved"]["reproducibility"]["production"] = True

    def doctor_must_not_run(**_kwargs):
        raise AssertionError("doctor must not hash artifacts for a wrong Python")

    monkeypatch.setattr(real2sim_preflight, "run_doctor", doctor_must_not_run)
    monkeypatch.setattr(
        real2sim_preflight,
        "_check_policy_python_runtime",
        lambda *_args: (
            False,
            {
                "expected_version": "3.11.15",
                "observed": {"version": "3.11.14"},
            },
        ),
    )

    with pytest.raises(launcher.ConfigError, match="policy_python.*3.11.14"):
        launcher.validate_production_preflight(plan)

def test_policy_python_runtime_requires_exact_patch_and_same_venv(
    monkeypatch, tmp_path
):
    environment_root = tmp_path / "venv"
    python_path = environment_root / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("placeholder\n", encoding="utf-8")
    plan = launcher.load_rollout_config(
        REPO_ROOT / "config" / "release" / "fixture_v2.yaml",
        timestamp="20260802_000000",
    )
    plan["resolved"]["policy"]["python"] = str(python_path)
    plan["resolved"]["reproducibility"]["accelerator_profile"]["python"] = (
        "3.11.15"
    )
    monkeypatch.setattr(launcher.sys, "prefix", str(environment_root))

    def probe(version, prefix, base_prefix):
        return subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps(
                {
                    "version": version,
                    "executable": str(python_path),
                    "prefix": str(prefix),
                    "base_prefix": str(base_prefix),
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: probe(
            "3.11.15", environment_root, tmp_path / "cpython"
        ),
    )
    ok, detail = launcher._check_policy_python_runtime(plan, {})
    assert ok is True
    assert all(detail["checks"].values())

    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: probe(
            "3.11.14", tmp_path / "shared-runtime", tmp_path / "shared-runtime"
        ),
    )
    ok, detail = launcher._check_policy_python_runtime(plan, {})
    assert ok is False
    assert detail["checks"] == {
        "version_matches": False,
        "is_virtual_environment": False,
        "path_matches_prefix": False,
        "matches_launcher_environment": False,
    }


def test_official_preflight_keeps_file_form_registration_identity(
    tmp_path: Path,
) -> None:
    from robosuite_env.rollout.preflight import (
        bind_validated_gs_registration,
        reported_gs_w2gs,
    )

    matrix = np.diag([1.0, 2.0, 3.0, 1.0])
    w2gs = tmp_path / "gs_w2gs.npy"
    np.save(w2gs, matrix)
    arguments = SimpleNamespace(gs_mat_w2gs=None, gs_w2gs="")

    bind_validated_gs_registration(
        arguments, {"w2gs_path": w2gs, "w2gs_matrix": matrix}
    )
    assert np.array_equal(arguments.gs_mat_w2gs, matrix)
    assert arguments.gs_w2gs == str(w2gs.resolve())
    assert reported_gs_w2gs(arguments) == str(w2gs.resolve())

    bind_validated_gs_registration(
        arguments, {"w2gs_path": None, "w2gs_matrix": matrix}
    )
    assert np.array_equal(arguments.gs_mat_w2gs, matrix)
    assert arguments.gs_w2gs == ""
    assert reported_gs_w2gs(arguments) == "inline-matrix"

    assert reported_gs_w2gs(
        SimpleNamespace(gs_mat_w2gs=None, gs_w2gs="")
    ) is None

if __name__ == "__main__":
    unittest.main()


def test_runtime_preflight_validates_assets_and_strict_scene_contract(tmp_path):
    from robosuite_env.rollout import preflight as rollout_preflight

    manifest_path = tmp_path / "run_manifest.json"
    scene_manifest = tmp_path / "background.json"
    worker_contract = tmp_path / "official_worker_contract.json"
    ply_path = tmp_path / "scene.ply"
    for path in (scene_manifest, worker_contract, ply_path):
        path.write_text("{}", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "preflight": {
                    "contracts": {
                        "assets": {
                            "contract_digest": "a" * 64,
                            "bundles": ["runtime-bundle"],
                        },
                        "scenes": {
                            "scene": {
                                "id": "scene-id",
                                "manifest_path": str(scene_manifest),
                                "contract_digest": "b" * 64,
                                "collision_tree_sha256": "c" * 64,
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    arguments = SimpleNamespace(
        robot="fr3",
        objects="cup,plate",
        strict_assets=False,
        gs_calibration_world_offset=[0.0, 0.0, 0.0],
        gs_rot="",
        gs_rot_R=None,
        official_run_manifest=str(manifest_path),
        official_worker_contract=str(worker_contract),
        official_scene_tag="scene",
        gs_background=str(scene_manifest),
        gs_server_token="launch-token",
        gs=str(ply_path),
        desk="scene",
        desk_world_aligned=False,
        gs_w2gs="",
        gs_mat_w2gs=None,
    )
    asset_report = {
        "contract_digest": "a" * 64,
        "bundles": ("runtime-bundle",),
    }
    scene_report = {
        "id": "scene-id",
        "manifest_path": scene_manifest,
        "contract_digest": "b" * 64,
        "collision_tree_sha256": "c" * 64,
        "ply_path": ply_path,
        "w2gs_path": None,
        "w2gs_matrix": np.eye(4),
    }
    calls: list[str] = []

    def record(name, result=None):
        def wrapped(*_args, **_kwargs):
            calls.append(name)
            return result

        return wrapped

    patches = (
        mock.patch.object(rollout_preflight, "resolve_gs_background", record("resolve")),
        mock.patch.object(
            rollout_preflight,
            "runtime_bundle_ids",
            return_value=("runtime-bundle",),
        ),
        mock.patch.object(
            rollout_preflight,
            "validate_repository_assets",
            record("assets", asset_report),
        ),
        mock.patch.object(
            rollout_preflight,
            "validate_worker_contract",
            record("worker"),
        ),
        mock.patch.object(
            rollout_preflight,
            "validate_scene_contract",
            record("scene", scene_report),
        ),
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        diagnostic = rollout_preflight.run_runtime_preflight(
            arguments,
            worker_argv=[],
            package_root=tmp_path,
        )
        assert diagnostic.execution_class == "diagnostic"
        assert calls == ["resolve", "assets"]

        calls.clear()
        arguments.strict_assets = True
        worker_argv = ["--official-worker-contract", str(worker_contract)]
        official = rollout_preflight.run_runtime_preflight(
            arguments,
            worker_argv=worker_argv,
            package_root=tmp_path,
        )
        assert official.execution_class == "official"
        assert calls == ["resolve", "assets", "worker", "scene"]
        assert arguments.desk_world_aligned is True

        arguments.desk = "plain"
        with pytest.raises(SystemExit, match="requires --desk scene"):
            rollout_preflight.run_runtime_preflight(
                arguments,
                worker_argv=worker_argv,
                package_root=tmp_path,
            )

    arguments.strict_assets = False
    with mock.patch.object(rollout_preflight, "resolve_gs_background"), mock.patch.object(
        rollout_preflight,
        "runtime_bundle_ids",
        return_value=("runtime-bundle",),
    ), mock.patch.object(
        rollout_preflight,
        "validate_repository_assets",
        side_effect=ValueError("missing bundle"),
    ):
        with pytest.raises(SystemExit, match="repository asset preflight failed"):
            rollout_preflight.run_runtime_preflight(
                arguments,
                worker_argv=[],
                package_root=tmp_path,
            )


# --------------------------------------------------------------------------
# Host-path and task-object gates that run before any contract validation.
# --------------------------------------------------------------------------


def _preflight_plan(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "rollout.yaml"
    path.write_text(yaml.safe_dump(_base_config(), sort_keys=False), encoding="utf-8")
    return launcher.load_rollout_config(
        path, repo_root=tmp_path, timestamp="20260730_120000"
    )


def test_task_objects_accepts_the_named_cup_plate_task(tmp_path):
    plan = _preflight_plan(tmp_path)

    assert real2sim_preflight._validate_task_objects(plan) is None


@pytest.mark.parametrize(
    ("objects", "message"),
    (
        ("positions.txt", "task.objects must be cup_plate, box_pack or a positions JSON"),
        ("missing_positions.json", "task.objects positions JSON not found"),
    ),
)
def test_task_objects_rejects_unusable_selectors(tmp_path, objects, message):
    plan = _preflight_plan(tmp_path)
    plan["resolved"]["task"]["objects"] = objects

    with pytest.raises(launcher.ConfigError, match=message):
        real2sim_preflight._validate_task_objects(plan)


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        ("[]", "task.objects positions JSON must contain an object"),
        ("{not json", "task.objects positions JSON is unreadable"),
        (
            json.dumps({"cup_xy": [0.0], "plate_xy": [0.0, 0.1]}),
            "task.objects cup_xy must be a two-element array of finite numbers",
        ),
        (
            json.dumps({"cup_xy": [0.0, 0.1], "plate_xy": [True, 0.1]}),
            "task.objects plate_xy must be a two-element array of finite numbers",
        ),
        (
            '{"cup_xy": [0.0, 0.1], "plate_xy": [1e400, 0.1]}',
            "task.objects plate_xy must be a two-element array of finite numbers",
        ),
    ),
)
def test_task_objects_positions_json_contract(tmp_path, payload, message):
    plan = _preflight_plan(tmp_path)
    positions = tmp_path / "positions.json"
    positions.write_text(payload, encoding="utf-8")
    plan["resolved"]["task"]["objects"] = str(positions)

    with pytest.raises(launcher.ConfigError, match=message):
        real2sim_preflight._validate_task_objects(plan)


def test_task_objects_accepts_a_finite_positions_json(tmp_path):
    plan = _preflight_plan(tmp_path)
    positions = tmp_path / "positions.json"
    positions.write_text(
        json.dumps({"cup_xy": [0.0, 0.1], "plate_xy": [0.2, -0.3]}), encoding="utf-8"
    )
    plan["resolved"]["task"]["objects"] = str(positions)

    assert real2sim_preflight._validate_task_objects(plan) is None


def test_runtime_paths_reject_a_malformed_base_calibration(tmp_path):
    """A half-specified mount pose must stop the launch, not mis-place the arms."""

    calibration = tmp_path / "base.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "robot": "fr3",
                "base": {"R": {"pos": [0.0, 0.0, 0.0], "yaw_deg": 45.0}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(launcher.ConfigError, match="is not usable for fr3"):
        real2sim_preflight._validate_base_placement("fr3", str(calibration))


def test_runtime_paths_require_the_base_calibration_to_exist(tmp_path):
    with pytest.raises(
        launcher.ConfigError,
        match="simulation.parameters.base_placement is not a file",
    ):
        real2sim_preflight._validate_base_placement(
            "fr3", str(tmp_path / "absent.json")
        )


def test_a_rig_is_refused_another_rigs_calibration(tmp_path):
    """The tilted Duo mount and the flat pedestals are not interchangeable.

    Both files parse and both name R and L, so nothing downstream would notice:
    the arms would simply stand at the wrong angle on a scene registered for
    the other layout.
    """

    duo = (
        Path(real2sim_preflight.__file__).resolve().parents[2]
        / "robosuite_env"
        / "obj_calib"
        / "fr3_base_duo.json"
    )
    with pytest.raises(launcher.ConfigError, match="is not usable for fr3"):
        real2sim_preflight._validate_base_placement("fr3", str(duo))


def test_an_unset_base_calibration_uses_the_rigs_registered_default(tmp_path):
    for robot in ("fr3", "fr3_duo"):
        assert real2sim_preflight._validate_base_placement(robot, None) is None


def test_runtime_paths_refuse_to_reuse_an_existing_run_root(tmp_path):
    plan = _preflight_plan(tmp_path)
    Path(plan["run_root"]).mkdir(parents=True)

    with pytest.raises(
        launcher.ConfigError, match="output.run_name already exists under output.root"
    ):
        real2sim_preflight.validate_runtime_paths(plan)


def test_runtime_paths_require_an_executable_policy_python(tmp_path):
    plan = _preflight_plan(tmp_path)
    python_path = Path(plan["resolved"]["policy"]["python"])
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(launcher.ConfigError, match="policy.python is not executable"):
        real2sim_preflight.validate_runtime_paths(plan)


def test_runtime_paths_require_a_checkpoint_directory(tmp_path):
    plan = _preflight_plan(tmp_path)
    python_path = Path(plan["resolved"]["policy"]["python"])
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("#!/bin/sh\n", encoding="utf-8")
    python_path.chmod(0o755)
    checkpoint = Path(plan["resolved"]["policy"]["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("not a directory", encoding="utf-8")

    with pytest.raises(
        launcher.ConfigError, match="policy.checkpoint is not a directory"
    ):
        real2sim_preflight.validate_runtime_paths(plan)
