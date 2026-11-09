# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
from typing import Iterator

import pytest
import yaml

from real2sim import launcher
from real2sim.gpu_binding import (
    GPU_BINDINGS_ENV,
    contract_json,
    resolve_gpu_contract,
)
from robosuite_env.contracts.worker import (
    WorkerContractError,
    validate_worker_contract,
    write_worker_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_CONFIG = REPO_ROOT / "config" / "release" / "fixture_v2.yaml"
FIXTURE_SCENE = REPO_ROOT / "tests" / "fixtures" / "artifacts" / "scene"
TOKEN = "a" * 64
EIGHT_GPU_NVIDIA = {
    "gpus": [
        {"index": str(index), "uuid": f"GPU-{index}"}
        for index in range(8)
    ]
}


@pytest.fixture(autouse=True)
def _isolate_cpu_fixture_from_ambient_cuda(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("REAL2SIM_GPU_BINDINGS_JSON", raising=False)


def _digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@contextmanager
def _authority(
    output_root: Path,
    *,
    repeats: int = 1,
    scene_tags: tuple[str, ...] = ("tiny_fixture_scene",),
    sealed_gpu_bindings: bool = False,
    sharded: bool = False,
    gpu_roles: tuple[str, str, str] = ("0", "0", "0"),
    allow_cuda_visible_devices: bool = False,
    policy_batch_size: int | None = None,
    box_pack_layouts: list[str] | None = None,
    impedance: bool = False,
) -> Iterator[tuple[Path, dict[str, object], dict[str, object]]]:
    config_directory = Path(
        tempfile.mkdtemp(prefix=".worker-contract-test-", dir=REPO_ROOT / "config")
    )
    try:
        source = yaml.safe_load(FIXTURE_CONFIG.read_text(encoding="utf-8"))
        source["runner"].update(
            {
                "mode": "fast_parallel" if repeats > 1 else "sequential_waves",
                "repeats": repeats,
                "policy_batch_size": (
                    policy_batch_size
                    if policy_batch_size is not None
                    else 10 if repeats >= 10 else len(scene_tags)
                ),
                "expected_rollouts": repeats * len(scene_tags),
            }
        )
        if impedance:
            source["simulation"]["fr3_controller"] = "cartesian_impedance"
            source["simulation"]["parameters"].update(
                fr3_impedance={}, policy_action_clamp=False)
        if box_pack_layouts is not None:
            source["simulation"]["parameters"]["box_pack_layout_json"] = box_pack_layouts
        scene_paths = {
            tag: (
                FIXTURE_SCENE
                if len(scene_tags) == 1
                else output_root / "scene-inputs" / tag
            )
            for tag in scene_tags
        }
        source["scenes"] = [
            {"tag": tag, "path": str(scene_paths[tag])} for tag in scene_tags
        ]
        artifacts = source["reproducibility"]["artifacts"]
        for role in tuple(artifacts):
            if role.startswith("scene."):
                del artifacts[role]
        for tag in scene_tags:
            artifacts[f"scene.{tag}"] = "tiny-scene-v2"
        source["output"] = {
            "root": str(output_root),
            "run_name": "worker_contract_{timestamp}",
        }
        # This fixture exercises an exact render-lock contract, so give it an
        # explicit host-ordinal selector without changing the shared CPU YAML.
        gs_devices, policy_device, sim_devices = gpu_roles
        source["runtime"].update(
            {
                "gs_cuda_devices": gs_devices,
                "policy_cuda_device": policy_device,
                "sim_cuda_devices": sim_devices,
            }
        )
        if allow_cuda_visible_devices:
            source["reproducibility"]["environment"]["allowlist"].append(
                "CUDA_VISIBLE_DEVICES"
            )
        if sharded:
            if repeats != 4 or len(scene_tags) != 1:
                raise ValueError(
                    "the sharded worker fixture requires one scene and four repeats"
                )
            source["runner"]["shards"] = [
                {
                    "name": "a",
                    "repeats": [1, 2],
                    "slurm_job_id": "101",
                    "gs_cuda_devices": "0",
                    "sim_cuda_devices": "0",
                    "policy_cuda_device": "0",
                    "policy_batch_size": 1,
                },
                {
                    "name": "b",
                    "repeats": [3, 4],
                    "slurm_job_id": "102",
                    "gs_cuda_devices": "0",
                    "sim_cuda_devices": "0",
                    "policy_cuda_device": "0",
                    "policy_batch_size": 1,
                },
            ]
        config_path = config_directory / "rollout.yaml"
        config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")

        timestamp = "20990101_000000"
        plan = launcher.load_rollout_config(
            config_path,
            repo_root=REPO_ROOT,
            timestamp=timestamp,
        )
        run_root = Path(plan["run_root"])
        run_root.mkdir(parents=True)
        resolved = plan["resolved"]
        environment = launcher.build_runner_environment(plan)
        if sealed_gpu_bindings:
            gpu_bindings = launcher.resolve_gpu_contract(
                environment,
                nvidia=EIGHT_GPU_NVIDIA,
            )
            environment = launcher.build_runner_environment(
                plan, gpu_bindings=gpu_bindings
            )
        scene_reports = {
            tag: {
                "id": f"{tag}-id",
                "manifest_path": str(scene_paths[tag] / "background.json"),
                "contract_digest": "b" * 64,
                "ply_sha256": "c" * 64,
            }
            for tag in scene_tags
        }
        worker_contract: dict[str, object] = {
            "schema_version": 1,
            "static_argv": launcher._official_static_sim_argv(environment),
        }
        shard_fields: dict[str, object] = {}
        if sharded:
            shard_environments, aggregate_environment = (
                launcher.build_all_shard_environments(plan)
            )
            records = []
            static_by_shard = {}
            for shard in resolved["runner"]["shards"]:
                name = str(shard["name"])
                shard_environment = shard_environments[name]
                static = launcher._official_static_sim_argv(shard_environment)
                static_by_shard[name] = static
                records.append(
                    {
                        "name": name,
                        "repeat_start": shard["repeat_start"],
                        "repeat_stop": shard["repeat_stop"],
                        "environment": shard_environment,
                        "static_argv": static,
                    }
                )
            worker_contract = {
                "schema_version": 2,
                "static_argv_by_shard": static_by_shard,
            }
            shard_fields = {
                "shard_execution": {
                    "schema_version": 1,
                    "shards": records,
                    "aggregation_environment": aggregate_environment,
                },
                "launcher_context": {
                    "shards_override": None,
                    "environment": {},
                },
            }
        manifest: dict[str, object] = {
            "config": {
                "path": plan["config_repo_path"],
                "sha256": plan["config_sha256"],
                "resolved_sha256": _digest_json(resolved),
                "timestamp": timestamp,
            },
            "resolved_environment": environment,
            "worker_contract": worker_contract,
            "preflight": {
                "contracts": {
                    "assets": {"contract_digest": "a" * 64},
                    "scenes": scene_reports,
                }
            },
            **shard_fields,
        }
        (run_root / "resolved_rollout_config.json").write_text(
            json.dumps(resolved, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_path = run_root / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        yield manifest_path, manifest, resolved
    finally:
        shutil.rmtree(config_directory)


def _state_path(environment: dict[str, object]) -> Path:
    return (
        Path(str(environment["EXPORTS_ROOT"]))
        / "deterministic-init-state"
        / "fr3_yubi_policy_deterministic_init_state_test_1234abcd.json"
    )


def _rollout_argv(
    manifest: dict[str, object],
    contract: Path,
    *,
    state_path: Path | None = None,
    physical_gpu: int = 0,
) -> list[str]:
    environment = manifest["resolved_environment"]
    worker = manifest["worker_contract"]
    scenes = manifest["preflight"]["contracts"]["scenes"]
    assert isinstance(environment, dict)
    assert isinstance(worker, dict)
    assert isinstance(scenes, dict)
    tag = contract.parent.parent.name
    run = contract.parent.name
    output = contract.parent
    arguments = list(worker["static_argv"])
    if state_path is not None:
        arguments.extend(("--init-state-in", str(state_path)))
    arguments.extend(
        (
            "--policy-batch-client-id",
            f"{tag}_{run}",
            "--gs-background",
            str(Path(scenes[tag]["manifest_path"]).parent),
            "--official-scene-tag",
            tag,
            "--gs-port",
            "8321",
            "--gs-server-token",
            TOKEN,
            "--render-lock",
            str(
                Path(environment["RENDER_LOCK_ROOT"])
                / f"mujoco_render_faces_gpu_{physical_gpu}.lock"
            ),
            "--out",
            str(output / f"{tag}_{run}.mp4"),
            "--logdir",
            str(output / "logs"),
            "--summary-path",
            str(output / "sim_summary.json"),
        )
    )
    return arguments


def _write_completed_rollout(
    manifest_path: Path,
    manifest: dict[str, object],
    resolved: dict[str, object],
    worker_contract: Path,
) -> dict[str, object]:
    environment = manifest["resolved_environment"]
    contracts = manifest["preflight"]
    assert isinstance(environment, dict)
    assert isinstance(contracts, dict)
    contract_reports = contracts["contracts"]
    assert isinstance(contract_reports, dict)
    scene_reports = contract_reports["scenes"]
    assert isinstance(scene_reports, dict)
    scene = worker_contract.parent.parent.name
    run = worker_contract.parent.name
    scene_report = scene_reports[scene]
    assert isinstance(scene_report, dict)
    summary_path = worker_contract.parent / "sim_summary.json"
    video_path = worker_contract.parent / f"{scene}_{run}.mp4"
    summary_path.write_text(
        json.dumps(
            {
                "frames": resolved["runner"]["frames"],
                "rendered_frames": 0,
                "wall_stopped_early": False,
                "execution_class": "official",
                "success": False,
                "new_final_success": False,
                "success_criterion": {"name": resolved["task"]["goal_condition"]},
                "asset_contract_sha256": "a" * 64,
                "official_run_manifest": str(manifest_path),
                "official_worker_contract": str(worker_contract),
                "scene_contract": {
                    "id": scene_report["id"],
                    "sha256": scene_report["contract_digest"],
                    "manifest": scene_report["manifest_path"],
                    "table_appearance": "embedded_in_3dgs",
                    "table_collision_parts": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    with (manifest_path.parent / "outputs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("scene", "run", "exit_code", "video", "summary"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "scene": scene,
                "run": run,
                "exit_code": "0",
                "video": str(video_path),
                "summary": str(summary_path),
            }
        )
    return {
        "run_root": manifest_path.parent,
        "repo_root": REPO_ROOT,
        "resolved": resolved,
    }


def _prep_argv(manifest: dict[str, object], contract: Path, state_path: Path) -> list[str]:
    environment = manifest["resolved_environment"]
    worker = manifest["worker_contract"]
    scenes = manifest["preflight"]["contracts"]["scenes"]
    assert isinstance(environment, dict)
    assert isinstance(worker, dict)
    assert isinstance(scenes, dict)
    first_tag = next(iter(scenes))
    output = contract.parent
    return list(worker["static_argv"]) + [
        "--gs-background",
        str(Path(scenes[first_tag]["manifest_path"]).parent),
        "--official-scene-tag",
        first_tag,
        "--gs-port",
        "8321",
        "--gs-server-token",
        TOKEN,
        "--no-render",
        "--init-state-in",
        "",
        "--init-state-out",
        str(state_path),
        "--init-state-only",
        "--out",
        str(output / "init_state_only.mp4"),
        "--logdir",
        str(output / "logs"),
        "--summary-path",
        str(output / "sim_summary.json"),
    ]


def test_static_argv_omits_retired_controller_options(tmp_path: Path):
    with _authority(tmp_path) as (_manifest_path, manifest, _resolved):
        worker = manifest["worker_contract"]
        assert isinstance(worker, dict)
        static = worker["static_argv"]
        assert isinstance(static, list)
        assert "--fr3-controller" in static
        assert static[static.index("--fr3-controller") + 1] == "joint_ik"
        assert "--fr3-contact-control" not in static
        assert "--ik-url" not in static


def test_layout_list_worker_contract_binds_run_number_to_selected_entry(
    tmp_path: Path,
) -> None:
    layouts = [str(tmp_path / "layout-0.json"), str(tmp_path / "layout-1.json")]
    with _authority(tmp_path, repeats=2, box_pack_layouts=layouts) as (
        manifest_path,
        manifest,
        _resolved,
    ):
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_02"
            / "official_worker_contract.json"
        )
        argv = _rollout_argv(
            manifest, contract, state_path=_state_path(manifest["resolved_environment"])
        )
        static_size = len(manifest["worker_contract"]["static_argv"])
        argv[static_size + 2:static_size + 2] = [
            "--box-pack-layout-json",
            layouts[1],
            "--box-pack-layout-index",
            "1",
        ]
        write_worker_contract(contract, manifest_path, argv)

        bad = list(argv)
        bad[static_size + 3] = layouts[0]
        with pytest.raises(WorkerContractError, match="differs from the resolved layout list"):
            write_worker_contract(contract, manifest_path, bad)

        prep_contract = (
            manifest_path.parent
            / "deterministic_init_state_prep"
            / "layout_01"
            / "official_worker_contract.json"
        )
        prep_argv = _prep_argv(
            manifest, prep_contract, _state_path(manifest["resolved_environment"])
        )
        prep_static_size = len(manifest["worker_contract"]["static_argv"])
        prep_argv[prep_static_size:prep_static_size] = [
            "--box-pack-layout-json",
            layouts[1],
            "--box-pack-layout-index",
            "1",
        ]
        write_worker_contract(prep_contract, manifest_path, prep_argv)


def test_layout_list_worker_contract_accepts_three_digit_run_and_prep_indices(
    tmp_path: Path,
) -> None:
    layouts = [str(tmp_path / f"layout-{index}.json") for index in range(101)]
    with _authority(
        tmp_path, repeats=101, policy_batch_size=1, box_pack_layouts=layouts
    ) as (
        manifest_path,
        manifest,
        _resolved,
    ):
        static_size = len(manifest["worker_contract"]["static_argv"])
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_101"
            / "official_worker_contract.json"
        )
        argv = _rollout_argv(
            manifest, contract, state_path=_state_path(manifest["resolved_environment"])
        )
        argv[static_size + 2:static_size + 2] = [
            "--box-pack-layout-json",
            layouts[100],
            "--box-pack-layout-index",
            "100",
        ]
        write_worker_contract(contract, manifest_path, argv)

        prep_contract = (
            manifest_path.parent
            / "deterministic_init_state_prep"
            / "layout_100"
            / "official_worker_contract.json"
        )
        prep_argv = _prep_argv(
            manifest, prep_contract, _state_path(manifest["resolved_environment"])
        )
        prep_argv[static_size:static_size] = [
            "--box-pack-layout-json",
            layouts[100],
            "--box-pack-layout-index",
            "100",
        ]
        write_worker_contract(prep_contract, manifest_path, prep_argv)


def test_worker_authority_rebuilds_sealed_gpu_bindings(tmp_path: Path):
    with _authority(tmp_path, sealed_gpu_bindings=True) as (
        manifest_path,
        manifest,
        _resolved,
    ):
        environment = manifest["resolved_environment"]
        assert isinstance(environment, dict)
        assert "REAL2SIM_GPU_BINDINGS_JSON" in environment
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_01"
            / "official_worker_contract.json"
        )
        argv = _rollout_argv(
            manifest,
            contract,
            state_path=_state_path(environment),
        )

        write_worker_contract(contract, manifest_path, argv)
        validate_worker_contract(contract, manifest_path, argv)

@pytest.mark.parametrize(
    ("visible_devices", "gpu_roles", "physical_indices"),
    (
        ("0", ("0", "0", "0"), (0,)),
        ("1", ("0", "0", "0"), (1,)),
        ("3,5", ("0", "0", "0"), (3,)),
        ("3,5", ("0,1", "0", "0,1"), (3, 5)),
        ("GPU-3,GPU-5", ("0,1", "0", "0,1"), (3, 5)),
    ),
)
def test_worker_render_lock_uses_dynamic_physical_gpu_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    visible_devices: str,
    gpu_roles: tuple[str, str, str],
    physical_indices: tuple[int, ...],
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    with _authority(
        tmp_path,
        repeats=len(physical_indices),
        gpu_roles=gpu_roles,
        allow_cuda_visible_devices=True,
        policy_batch_size=len(physical_indices),
    ) as (manifest_path, manifest, _resolved):
        environment = manifest["resolved_environment"]
        assert isinstance(environment, dict)
        assert GPU_BINDINGS_ENV not in environment
        dynamic_gpu_bindings = resolve_gpu_contract(
            environment,
            nvidia=EIGHT_GPU_NVIDIA,
        )
        monkeypatch.setenv(
            GPU_BINDINGS_ENV,
            contract_json(dynamic_gpu_bindings),
        )
        # Simulator children intentionally hide CUDA and select MuJoCo by the
        # proven physical EGL index. The manifest remains the CVD authority.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

        for repeat, physical_index in enumerate(physical_indices, start=1):
            contract = (
                manifest_path.parent
                / "tiny_fixture_scene"
                / f"run_{repeat:02d}"
                / "official_worker_contract.json"
            )
            argv = _rollout_argv(
                manifest,
                contract,
                state_path=_state_path(environment),
            )
            lock_index = argv.index("--render-lock") + 1
            argv[lock_index] = str(
                Path(environment["RENDER_LOCK_ROOT"])
                / f"mujoco_render_faces_gpu_{physical_index}.lock"
            )

            write_worker_contract(contract, manifest_path, argv)
            validate_worker_contract(contract, manifest_path, argv)


@pytest.mark.parametrize(
    ("visible_devices", "physical_gpu"),
    (("5", 5), ("3,5", 3)),
)
def test_legacy_completed_fixture_loses_shell_only_gpu_binding_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    visible_devices: str,
    physical_gpu: int,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    with _authority(
        tmp_path,
        allow_cuda_visible_devices=True,
    ) as (manifest_path, manifest, resolved):
        environment = manifest["resolved_environment"]
        assert isinstance(environment, dict)
        dynamic_gpu_bindings = resolve_gpu_contract(
            environment,
            nvidia=EIGHT_GPU_NVIDIA,
        )
        monkeypatch.setenv(GPU_BINDINGS_ENV, contract_json(dynamic_gpu_bindings))
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_01"
            / "official_worker_contract.json"
        )
        argv = _rollout_argv(
            manifest,
            contract,
            state_path=_state_path(environment),
            physical_gpu=physical_gpu,
        )

        write_worker_contract(contract, manifest_path, argv)
        validate_worker_contract(contract, manifest_path, argv)

        legacy_payload = json.loads(contract.read_text(encoding="utf-8"))
        legacy_payload["schema_version"] = 1
        legacy_payload.pop("gpu_bindings")
        contract.write_text(
            json.dumps(legacy_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        plan = _write_completed_rollout(
            manifest_path,
            manifest,
            resolved,
            contract,
        )
        monkeypatch.delenv(GPU_BINDINGS_ENV)

        assert launcher.rollout_completion_errors(plan) == [
            "tiny_fixture_scene/run_01 worker contract invalid: "
            "worker render lock GPU is not configured"
        ]


@pytest.mark.parametrize(
    ("visible_devices", "physical_gpu", "conflicting_visible_devices"),
    (("5", 5, "4"), ("3,5", 3, "1")),
)
def test_completed_fixture_uses_launcher_sealed_gpu_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    visible_devices: str,
    physical_gpu: int,
    conflicting_visible_devices: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    with _authority(
        tmp_path,
        sealed_gpu_bindings=True,
        allow_cuda_visible_devices=True,
    ) as (manifest_path, manifest, resolved):
        environment = manifest["resolved_environment"]
        assert isinstance(environment, dict)
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_01"
            / "official_worker_contract.json"
        )
        argv = _rollout_argv(
            manifest,
            contract,
            state_path=_state_path(environment),
            physical_gpu=physical_gpu,
        )

        write_worker_contract(contract, manifest_path, argv)
        plan = _write_completed_rollout(
            manifest_path,
            manifest,
            resolved,
            contract,
        )
        conflicting_environment = dict(environment)
        conflicting_environment["CUDA_VISIBLE_DEVICES"] = conflicting_visible_devices
        conflicting_gpu_bindings = resolve_gpu_contract(
            conflicting_environment,
            nvidia=EIGHT_GPU_NVIDIA,
        )
        monkeypatch.setenv(
            GPU_BINDINGS_ENV,
            contract_json(conflicting_gpu_bindings),
        )

        assert launcher.rollout_completion_errors(plan) == []


@pytest.mark.parametrize("sealed_gpu_bindings", (False, True))
def test_worker_render_lock_refuses_visible_gpu_not_owned_by_sim_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sealed_gpu_bindings: bool,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    with _authority(
        tmp_path,
        sealed_gpu_bindings=sealed_gpu_bindings,
        gpu_roles=("0,1", "0", "0"),
        allow_cuda_visible_devices=True,
    ) as (manifest_path, manifest, _resolved):
        environment = manifest["resolved_environment"]
        assert isinstance(environment, dict)
        dynamic_gpu_bindings = resolve_gpu_contract(
            environment,
            nvidia=EIGHT_GPU_NVIDIA,
        )
        monkeypatch.setenv(
            GPU_BINDINGS_ENV,
            contract_json(dynamic_gpu_bindings),
        )
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_01"
            / "official_worker_contract.json"
        )
        argv = _rollout_argv(
            manifest,
            contract,
            state_path=_state_path(environment),
        )
        lock_index = argv.index("--render-lock") + 1
        argv[lock_index] = str(
            Path(environment["RENDER_LOCK_ROOT"])
            / "mujoco_render_faces_gpu_5.lock"
        )

        with pytest.raises(WorkerContractError, match="render lock GPU"):
            write_worker_contract(contract, manifest_path, argv)


def test_worker_dynamic_gpu_binding_cannot_replace_recorded_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    with _authority(
        tmp_path,
        allow_cuda_visible_devices=True,
    ) as (manifest_path, manifest, _resolved):
        environment = manifest["resolved_environment"]
        assert isinstance(environment, dict)
        other_environment = dict(environment)
        other_environment["CUDA_VISIBLE_DEVICES"] = "1"
        dynamic_gpu_bindings = resolve_gpu_contract(
            other_environment,
            nvidia=EIGHT_GPU_NVIDIA,
        )
        monkeypatch.setenv(
            GPU_BINDINGS_ENV,
            contract_json(dynamic_gpu_bindings),
        )
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_01"
            / "official_worker_contract.json"
        )
        argv = _rollout_argv(
            manifest,
            contract,
            state_path=_state_path(environment),
        )
        lock_index = argv.index("--render-lock") + 1
        argv[lock_index] = str(
            Path(environment["RENDER_LOCK_ROOT"])
            / "mujoco_render_faces_gpu_1.lock"
        )

        with pytest.raises(
            WorkerContractError,
            match="differs from the current CUDA_VISIBLE_DEVICES boundary",
        ):
            write_worker_contract(contract, manifest_path, argv)


def test_worker_manifest_gpu_binding_must_match_recorded_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    with _authority(
        tmp_path,
        allow_cuda_visible_devices=True,
    ) as (manifest_path, manifest, _resolved):
        environment = manifest["resolved_environment"]
        assert isinstance(environment, dict)
        other_environment = dict(environment)
        other_environment["CUDA_VISIBLE_DEVICES"] = "1"
        wrong_gpu_bindings = resolve_gpu_contract(
            other_environment,
            nvidia=EIGHT_GPU_NVIDIA,
        )
        environment[GPU_BINDINGS_ENV] = contract_json(wrong_gpu_bindings)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_01"
            / "official_worker_contract.json"
        )
        argv = _rollout_argv(
            manifest,
            contract,
            state_path=_state_path(environment),
            physical_gpu=1,
        )

        with pytest.raises(
            WorkerContractError,
            match="differs from the current CUDA_VISIBLE_DEVICES boundary",
        ):
            write_worker_contract(contract, manifest_path, argv)


def test_worker_recorded_gpu_binding_cannot_replace_launcher_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    with _authority(
        tmp_path,
        sealed_gpu_bindings=True,
        allow_cuda_visible_devices=True,
    ) as (manifest_path, manifest, _resolved):
        environment = manifest["resolved_environment"]
        assert isinstance(environment, dict)
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_01"
            / "official_worker_contract.json"
        )
        argv = _rollout_argv(
            manifest,
            contract,
            state_path=_state_path(environment),
            physical_gpu=3,
        )
        write_worker_contract(contract, manifest_path, argv)
        payload = json.loads(contract.read_text(encoding="utf-8"))
        payload["schema_version"] = 2
        payload["gpu_bindings"] = json.loads(environment[GPU_BINDINGS_ENV])
        contract.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(
            WorkerContractError,
            match="cannot replace launcher GPU authority",
        ):
            validate_worker_contract(contract, manifest_path, argv)


def test_worker_recorded_gpu_binding_must_match_manifest_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    with _authority(
        tmp_path,
        allow_cuda_visible_devices=True,
    ) as (manifest_path, manifest, _resolved):
        environment = manifest["resolved_environment"]
        assert isinstance(environment, dict)
        gpu_bindings = resolve_gpu_contract(
            environment,
            nvidia=EIGHT_GPU_NVIDIA,
        )
        monkeypatch.setenv(GPU_BINDINGS_ENV, contract_json(gpu_bindings))
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_01"
            / "official_worker_contract.json"
        )
        argv = _rollout_argv(
            manifest,
            contract,
            state_path=_state_path(environment),
            physical_gpu=3,
        )
        write_worker_contract(contract, manifest_path, argv)
        payload = json.loads(contract.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 2

        other_environment = dict(environment)
        other_environment["CUDA_VISIBLE_DEVICES"] = "1"
        payload["gpu_bindings"] = resolve_gpu_contract(
            other_environment,
            nvidia=EIGHT_GPU_NVIDIA,
        )
        contract.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        monkeypatch.delenv(GPU_BINDINGS_ENV)

        with pytest.raises(
            WorkerContractError,
            match="differs from the current CUDA_VISIBLE_DEVICES boundary",
        ):
            validate_worker_contract(contract, manifest_path, argv)


def test_exact_worker_arguments_are_required(tmp_path: Path):
    with _authority(tmp_path) as (manifest_path, manifest, _resolved):
        environment = manifest["resolved_environment"]
        assert isinstance(environment, dict)
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_01"
            / "official_worker_contract.json"
        )
        argv = _rollout_argv(
            manifest,
            contract,
            state_path=_state_path(environment),
        )

        write_worker_contract(contract, manifest_path, argv)
        validate_worker_contract(contract, manifest_path, argv)
        with pytest.raises(WorkerContractError, match="arguments differ"):
            validate_worker_contract(
                contract,
                manifest_path,
                argv[:-1] + [str(contract.parent / "other-summary.json")],
            )


def test_sharded_worker_authority_selects_the_absolute_repeat_owner(
    tmp_path: Path,
):
    with _authority(tmp_path, repeats=4, sharded=True) as (
        manifest_path,
        manifest,
        _resolved,
    ):
        shard_execution = manifest["shard_execution"]
        assert isinstance(shard_execution, dict)
        records = shard_execution["shards"]
        assert isinstance(records, list)
        shard_b = records[1]
        assert isinstance(shard_b, dict)
        selected_manifest = dict(manifest)
        selected_manifest["resolved_environment"] = shard_b["environment"]
        selected_manifest["worker_contract"] = {
            "schema_version": 1,
            "static_argv": shard_b["static_argv"],
        }
        environment = selected_manifest["resolved_environment"]
        assert isinstance(environment, dict)
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_03"
            / "official_worker_contract.json"
        )
        argv = _rollout_argv(
            selected_manifest,
            contract,
            state_path=_state_path(environment),
        )

        write_worker_contract(contract, manifest_path, argv)
        validate_worker_contract(contract, manifest_path, argv)

        shard_a = records[0]
        assert isinstance(shard_a, dict)
        wrong_argv = list(shard_a["static_argv"]) + argv[
            len(shard_b["static_argv"]) :
        ]
        with pytest.raises(WorkerContractError, match="static arguments differ"):
            write_worker_contract(contract, manifest_path, wrong_argv)

        shard_a_manifest = dict(manifest)
        shard_a_manifest["resolved_environment"] = shard_a["environment"]
        shard_a_manifest["worker_contract"] = {
            "schema_version": 1,
            "static_argv": shard_a["static_argv"],
        }
        shard_a_environment = shard_a_manifest["resolved_environment"]
        assert isinstance(shard_a_environment, dict)
        prep_contract = (
            manifest_path.parent
            / ".shards"
            / "a"
            / "deterministic_init_state_prep"
            / "official_worker_contract.json"
        )
        prep_argv = _prep_argv(
            shard_a_manifest,
            prep_contract,
            _state_path(shard_a_environment),
        )
        write_worker_contract(prep_contract, manifest_path, prep_argv)
        validate_worker_contract(prep_contract, manifest_path, prep_argv)


def test_sharded_layout_list_prep_contract_selects_its_shard_authority(
    tmp_path: Path,
) -> None:
    layouts = [str(tmp_path / f"layout-{index}.json") for index in range(4)]
    with _authority(
        tmp_path, repeats=4, sharded=True, box_pack_layouts=layouts
    ) as (manifest_path, manifest, _resolved):
        records = manifest["shard_execution"]["shards"]
        shard_b = next(record for record in records if record["name"] == "b")
        selected_manifest = dict(manifest)
        selected_manifest["resolved_environment"] = shard_b["environment"]
        selected_manifest["worker_contract"] = {
            "schema_version": 1,
            "static_argv": shard_b["static_argv"],
        }
        contract = (
            manifest_path.parent
            / ".shards"
            / "b"
            / "deterministic_init_state_prep"
            / "layout_02"
            / "official_worker_contract.json"
        )
        argv = _prep_argv(
            selected_manifest,
            contract,
            _state_path(shard_b["environment"]),
        )
        static_size = len(shard_b["static_argv"])
        argv[static_size:static_size] = [
            "--box-pack-layout-json",
            layouts[2],
            "--box-pack-layout-index",
            "2",
        ]

        write_worker_contract(contract, manifest_path, argv)
        validate_worker_contract(contract, manifest_path, argv)

        wrong_contract = (
            manifest_path.parent
            / ".shards"
            / "b"
            / "not_deterministic_init_state_prep"
            / "layout_02"
            / "official_worker_contract.json"
        )
        with pytest.raises(
            WorkerContractError,
            match="worker path does not select one configured shard",
        ):
            write_worker_contract(wrong_contract, manifest_path, argv)


def test_sharded_worker_persists_target_gpu_binding_for_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    with _authority(
        tmp_path,
        repeats=4,
        sharded=True,
        allow_cuda_visible_devices=True,
    ) as (
        manifest_path,
        manifest,
        _resolved,
    ):
        shard_execution = manifest["shard_execution"]
        assert isinstance(shard_execution, dict)
        records = shard_execution["shards"]
        assert isinstance(records, list)
        shard = records[0]
        assert isinstance(shard, dict)
        selected_manifest = dict(manifest)
        selected_manifest["resolved_environment"] = shard["environment"]
        selected_manifest["worker_contract"] = {
            "schema_version": 1,
            "static_argv": shard["static_argv"],
        }
        environment = selected_manifest["resolved_environment"]
        assert isinstance(environment, dict)
        runtime_environment = dict(environment)
        runtime_environment["CUDA_VISIBLE_DEVICES"] = "5"
        gpu_bindings = resolve_gpu_contract(
            runtime_environment,
            nvidia=EIGHT_GPU_NVIDIA,
        )
        monkeypatch.setenv(GPU_BINDINGS_ENV, contract_json(gpu_bindings))
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_01"
            / "official_worker_contract.json"
        )
        argv = _rollout_argv(
            selected_manifest,
            contract,
            state_path=_state_path(environment),
            physical_gpu=5,
        )

        write_worker_contract(contract, manifest_path, argv)
        validate_worker_contract(contract, manifest_path, argv)
        payload = json.loads(contract.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 2
        assert payload["gpu_bindings"] == gpu_bindings

        monkeypatch.delenv(GPU_BINDINGS_ENV)
        validate_worker_contract(contract, manifest_path, argv)

        wrong_argv = list(argv)
        lock_index = wrong_argv.index("--render-lock") + 1
        wrong_argv[lock_index] = str(
            Path(environment["RENDER_LOCK_ROOT"])
            / "mujoco_render_faces_gpu_7.lock"
        )
        payload["argv"] = wrong_argv
        contract.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(WorkerContractError, match="render lock GPU"):
            validate_worker_contract(contract, manifest_path, wrong_argv)

        payload["argv"] = argv
        payload["gpu_bindings"] = None
        contract.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(WorkerContractError, match="worker GPU binding is invalid"):
            validate_worker_contract(contract, manifest_path, argv)


def test_manifest_mutation_and_out_of_root_contract_are_rejected(tmp_path: Path):
    with _authority(tmp_path) as (manifest_path, manifest, _resolved):
        environment = manifest["resolved_environment"]
        assert isinstance(environment, dict)
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_01"
            / "official_worker_contract.json"
        )
        argv = _rollout_argv(
            manifest,
            contract,
            state_path=_state_path(environment),
        )
        write_worker_contract(contract, manifest_path, argv)

        manifest["unrelated_note"] = "mutated"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(WorkerContractError, match="manifest changed"):
            validate_worker_contract(contract, manifest_path, argv)

        outside = tmp_path.parent / "outside-worker.json"
        with pytest.raises(WorkerContractError, match="inside official run root"):
            write_worker_contract(outside, manifest_path, argv)


def test_run_100_is_canonical_and_bounded(tmp_path: Path):
    with _authority(tmp_path, repeats=100) as (manifest_path, manifest, _resolved):
        environment = manifest["resolved_environment"]
        assert isinstance(environment, dict)
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_100"
            / "official_worker_contract.json"
        )
        argv = _rollout_argv(
            manifest,
            contract,
            state_path=_state_path(environment),
        )
        write_worker_contract(contract, manifest_path, argv)
        validate_worker_contract(contract, manifest_path, argv)

        outside_contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_101"
            / "official_worker_contract.json"
        )
        with pytest.raises(WorkerContractError, match="configured range"):
            write_worker_contract(
                outside_contract,
                manifest_path,
                _rollout_argv(
                    manifest,
                    outside_contract,
                    state_path=_state_path(environment),
                ),
            )


def test_prep_uses_yaml_scene_order_and_allows_empty_input(tmp_path: Path):
    with _authority(tmp_path, scene_tags=("z_scene", "a_scene")) as (
        manifest_path,
        manifest,
        _resolved,
    ):
        environment = manifest["resolved_environment"]
        assert isinstance(environment, dict)
        contract = (
            manifest_path.parent
            / "deterministic_init_state_prep"
            / "official_worker_contract.json"
        )
        argv = _prep_argv(manifest, contract, _state_path(environment))

        write_worker_contract(contract, manifest_path, argv)
        validate_worker_contract(contract, manifest_path, argv)

        nested_contract = (
            manifest_path.parent
            / "nested"
            / "deterministic_init_state_prep"
            / "official_worker_contract.json"
        )
        with pytest.raises(WorkerContractError, match="path is invalid"):
            write_worker_contract(
                nested_contract,
                manifest_path,
                _prep_argv(manifest, nested_contract, _state_path(environment)),
            )

        # A policy batch of 10 or more carries --no-render in the
        # launcher-owned prefix. Remove the runner's suffix copy and prove the
        # resulting contract is canonical.
        worker = manifest["worker_contract"]
        assert isinstance(worker, dict)
        static = list(worker["static_argv"])
        if "--no-render" not in static:
            static.append("--no-render")
        worker["static_argv"] = static
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        argv = _prep_argv(manifest, contract, _state_path(environment))
        suffix_copies = [
            index
            for index, argument in enumerate(argv[len(static) :], len(static))
            if argument == "--no-render"
        ]
        assert len(suffix_copies) == 1
        del argv[suffix_copies[0]]
        assert argv.count("--no-render") == static.count("--no-render")
        write_worker_contract(contract, manifest_path, argv)
        validate_worker_contract(contract, manifest_path, argv)


def test_deterministic_state_and_exact_gpu_lock_are_required(tmp_path: Path):
    with _authority(tmp_path) as (manifest_path, manifest, _resolved):
        environment = manifest["resolved_environment"]
        assert isinstance(environment, dict)
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_01"
            / "official_worker_contract.json"
        )
        with pytest.raises(WorkerContractError, match="requires --init-state-in"):
            write_worker_contract(
                contract,
                manifest_path,
                _rollout_argv(manifest, contract),
            )

        argv = _rollout_argv(
            manifest,
            contract,
            state_path=_state_path(environment),
        )
        lock_index = argv.index("--render-lock") + 1
        argv[lock_index] = str(
            Path(environment["RENDER_LOCK_ROOT"])
            / "nested"
            / "mujoco_render_faces_gpu_0.lock"
        )
        with pytest.raises(WorkerContractError, match="render lock GPU"):
            write_worker_contract(contract, manifest_path, argv)


def test_resolved_controller_or_obs_tampering_is_rejected(tmp_path: Path):
    with _authority(tmp_path) as (manifest_path, manifest, resolved):
        resolved["simulation"]["cli_args"].extend(
            ("--policy-wrist-contrast", "0.25")
        )
        manifest["config"]["resolved_sha256"] = _digest_json(resolved)
        (manifest_path.parent / "resolved_rollout_config.json").write_text(
            json.dumps(resolved, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_01"
            / "official_worker_contract.json"
        )
        with pytest.raises(WorkerContractError, match="checked-in YAML"):
            write_worker_contract(contract, manifest_path, [])



def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    # The child imports the installed package; no checkout path is added.
    environment.pop("PYTHONPATH", None)
    return environment

def test_launcher_entrypoints_work_without_repository_root_on_pythonpath(tmp_path):
    environment = _subprocess_environment()
    import_code = (
        f"import sys; sys.path[0] = {str(tmp_path)!r}; "
        "assert sys.path[0] == "
        f"{str(tmp_path)!r}; import real2sim.launcher"
    )
    completed = subprocess.run(
        [sys.executable, "-c", import_code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    completed = subprocess.run(
        [sys.executable, "-m", "real2sim.cli", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout

    completed = subprocess.run(
        [sys.executable, "-m", "real2sim.launcher", "--help"],
        cwd=REPO_ROOT / "scripts",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout

def test_pyyaml_guard_runs_before_launcher_imports(tmp_path):
    environment = _subprocess_environment()
    code = """
import sys

class BlockYaml:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "yaml":
            raise ModuleNotFoundError("blocked yaml")
        return None

sys.meta_path.insert(0, BlockYaml())
try:
    import real2sim.launcher
except SystemExit as exc:
    assert str(exc) == (
        "PyYAML is required; run scripts/bootstrap.sh to sync the locked environment"
    )
else:
    raise AssertionError("launcher import unexpectedly succeeded")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

def test_launcher_submodules_import_in_fresh_processes():
    modules = (
        "real2sim._bootstrap",
        "real2sim.env_file",
        "real2sim.gpu_binding",
        "real2sim.config._types",
        "real2sim.config.keys",
        "real2sim.config.paths",
        "real2sim.config.controllers",
        "real2sim.config.parameters",
        "real2sim.config.reproducibility",
        "real2sim.config.schema",
        "real2sim.runner.argv",
        "real2sim.runner.environment",
        "real2sim.runner.execute",
        "real2sim.artifacts_binding",
        "real2sim.preflight",
        "real2sim.launch_provenance",
        "real2sim.reporting.posthoc_provenance",
        "real2sim.reporting.completion",
        "real2sim.reporting.summary",
        "real2sim.reporting.dry_run",
        "real2sim.cli",
        "real2sim.launcher",
    )
    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.pop("PYTHONPATH", None)
    for module in modules:
        completed = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, (
            f"{module} failed to import:\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )

from tests._helpers.worker_ast import *  # noqa: F403

def test_cli_owns_all_six_strict_runtime_options() -> None:
    cli_arguments = _argument_calls(CLI_TREE)
    worker_arguments = _argument_calls(*_WORKER_TREES.values())

    for option in STRICT_RUNTIME_OPTIONS:
        assert len(cli_arguments.get(option, [])) == 1, option
        assert option not in worker_arguments

    strict_call = cli_arguments["--strict-assets"][0]
    assert ast.literal_eval(_keyword(strict_call, "action")) == "store_true"

    offset_call = cli_arguments["--gs-calibration-world-offset"][0]
    assert ast.literal_eval(_keyword(offset_call, "nargs")) == 3
    assert ast.literal_eval(_keyword(offset_call, "default")) == [0.0, 0.0, 0.0]

    token_call = cli_arguments["--gs-server-token"][0]
    assert ast.literal_eval(_keyword(token_call, "default")) == ""

def test_official_preflight_finishes_before_environment_construction() -> None:
    ordered_preflight_calls = (
        "resolve_gs_background",
        "validate_repository_assets",
        "validate_worker_contract",
        "validate_scene_contract",
    )
    lines: list[int] = []
    for name in ordered_preflight_calls:
        occurrences = _call_lines(PREFLIGHT_TREE, name)
        assert len(occurrences) == 1, (name, occurrences)
        lines.append(occurrences[0])
    assert lines == sorted(lines)

    preflight_calls = _call_lines(SETUP_ARGS_TREE, "run_runtime_preflight")
    environment_calls = _call_lines(BUILD_MODEL_TREE, "EnvCls")
    assert len(preflight_calls) == 1
    assert len(environment_calls) == 1

    validate_stage_calls = _call_lines(WORKER_TREE, "validate_args")
    model_stage_calls = _call_lines(WORKER_TREE, "build_model_env")
    assert len(validate_stage_calls) == 1
    assert len(model_stage_calls) == 1
    assert validate_stage_calls[0] < model_stage_calls[0]

def test_rollout_summary_preserves_official_provenance_contract() -> None:
    required = {
        "execution_class",
        "official_run_manifest",
        "official_worker_contract",
        "asset_contract_sha256",
        "scene_contract",
    }
    candidates: list[dict[str, ast.expr]] = []
    for tree in _WORKER_TREES.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                fields = _string_keyed_dict(node)
                if required <= fields.keys():
                    candidates.append(fields)
    assert len(candidates) == 1
    summary = candidates[0]

    assert ast.unparse(summary["obs_gs_only"]) == "bool(state.A.obs_gs_only)"
    assert ast.unparse(summary["execution_class"]) == "state.EXECUTION_CLASS"
    assert (
        ast.unparse(summary["official_run_manifest"])
        == "state.OFFICIAL_RUN_MANIFEST_PATH"
    )
    assert (
        ast.unparse(summary["official_worker_contract"])
        == "state.OFFICIAL_WORKER_CONTRACT_PATH"
    )
    assert "contract_digest" in ast.unparse(summary["asset_contract_sha256"])

    scene_value = summary["scene_contract"]
    assert isinstance(scene_value, ast.IfExp)
    assert isinstance(scene_value.body, ast.Dict)
    scene = _string_keyed_dict(scene_value.body)
    assert {
        "schema_version",
        "id",
        "name",
        "tag",
        "sha256",
        "manifest",
        "table_appearance",
        "table_collision_parts",
    } <= scene.keys()
    assert ast.literal_eval(scene["table_appearance"]) == "embedded_in_3dgs"

    consumed_summary_fields = {
        ast.literal_eval(node.args[0])
        for node in ast.walk(LAUNCHER_TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "summary"
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert required <= consumed_summary_fields

def test_worker_imports_extracted_responsibilities_from_canonical_packages() -> None:
    imports: set[str] = set()
    for tree in _WORKER_TREES.values():
        imports |= _imported_modules(tree)
    required = {
        "robosuite_env.contracts.assets",
        "robosuite_env.contracts.scene",
        "robosuite_env.contracts.worker",
        "robosuite_env.policy",
        "robosuite_env.policy.rel2abs",
        "robosuite_env.rendering.wrist_fisheye",
        "robosuite_env.robots.profiles",
        "robosuite_env.robots.fr3.runtime_ik",
        "robosuite_env.rollout.cli",
        "robosuite_env.rollout.control.grasp_intent",
        "robosuite_env.rollout.control.joint_trajectory",
        "robosuite_env.rollout.control.pose_math",
        "robosuite_env.rollout.preflight",
        "robosuite_env.rollout.sources.lerobot",
        "robosuite_env.rollout.state.cache",
        "robosuite_env.rollout.state.mujoco",
        "robosuite_env.rollout.state.object_settle",
        # The plain TableArena envs are no longer imported by the worker: the
        # native --desk robosuite path was removed and only the registered
        # (GW) environments remain constructible.
        "robosuite_env.tasks.envs.cup_plate_registered",
        "robosuite_env.tasks.success",
    }
    assert required <= imports

    # Runtime calibration must use the same imported canonical YUBI model.
    combined_source = "\n".join(_WORKER_SOURCES.values())
    assert "_yubi_model.YUBI_XML" in combined_source
    assert "_yubi_model.YUBI_XML_LEFT" in combined_source

def test_worker_has_no_cli_blender_or_gs_resolver_implementation() -> None:
    imports: set[str] = set()
    definitions: set[str] = set()
    for path, tree in _WORKER_TREES.items():
        imports |= _imported_modules(tree)
        if path == PREFLIGHT_PATH:
            continue
        definitions |= {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
    assert "argparse" not in imports
    assert "bpy" not in imports

    assert not {
        "_argv_has",
        "_gs_is_off",
        "_gs_path",
        "_gs_pick",
        "_load_gs_index",
        "_builtin_gs_preset",
        "_find_gs_preset",
        "_gs_matrix_or_path",
        "_extract_gs_alignment",
        "_extract_gs_rotation",
        "_extract_gs_pivot",
        "_apply_gs_definition",
        "_resolve_gs_background",
    } & definitions

    preflight_definitions = {
        node.name
        for node in ast.walk(PREFLIGHT_TREE)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert {"argv_has", "resolve_gs_background"} <= preflight_definitions


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda s: s.pop("success"), "has no boolean success verdict"),
        (lambda s: s.update(success="yes"), "success is not a boolean"),
        (
            lambda s: s.update(success_criterion={"name": "unrelated_criterion"}),
            "was scored by 'unrelated_criterion'",
        ),
        (
            lambda s: s.update(
                start_state={"overlaps": [{"geom_a": "cup", "geom_b": "plate"}]}
            ),
            "penetrating contact",
        ),
    ),
    ids=("no-verdict", "non-boolean", "wrong-criterion", "start-penetration"),
)
def test_completion_refuses_a_run_it_cannot_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    message: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    with _authority(
        tmp_path,
        sealed_gpu_bindings=True,
        allow_cuda_visible_devices=True,
    ) as (manifest_path, manifest, resolved):
        environment = manifest["resolved_environment"]
        assert isinstance(environment, dict)
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_01"
            / "official_worker_contract.json"
        )
        argv = _rollout_argv(
            manifest, contract, state_path=_state_path(environment), physical_gpu=5
        )
        write_worker_contract(contract, manifest_path, argv)
        plan = _write_completed_rollout(manifest_path, manifest, resolved, contract)
        assert launcher.rollout_completion_errors(plan) == []

        summary_path = contract.parent / "sim_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        mutate(summary)
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        errors = launcher.rollout_completion_errors(plan)
        assert any(message in error for error in errors), errors


def test_completion_refuses_a_json_null_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    with _authority(
        tmp_path,
        sealed_gpu_bindings=True,
        allow_cuda_visible_devices=True,
    ) as (manifest_path, manifest, resolved):
        environment = manifest["resolved_environment"]
        assert isinstance(environment, dict)
        contract = (
            manifest_path.parent
            / "tiny_fixture_scene"
            / "run_01"
            / "official_worker_contract.json"
        )
        argv = _rollout_argv(
            manifest, contract, state_path=_state_path(environment), physical_gpu=5
        )
        write_worker_contract(contract, manifest_path, argv)
        plan = _write_completed_rollout(manifest_path, manifest, resolved, contract)
        (contract.parent / "sim_summary.json").write_text("null", encoding="utf-8")

        errors = launcher.rollout_completion_errors(plan)
        assert any("summary must contain a JSON object" in error for error in errors), errors


def test_impedance_arrays_survive_official_manifest_roundtrip(tmp_path: Path):
    with _authority(tmp_path, impedance=True) as (manifest_path, manifest, _resolved):
        contract = manifest_path.parent / "tiny_fixture_scene/run_01/official_worker_contract.json"
        argv = _rollout_argv(manifest, contract, state_path=_state_path(manifest["resolved_environment"]))
        write_worker_contract(contract, manifest_path, argv)
        validate_worker_contract(contract, manifest_path, argv)
        assert argv[argv.index("--mujoco-timestep") + 1] == "0.001"


# Fields a launcher exported before OpenPI moved to the locked distribution.
# Official manifests recorded then still carry them.
RETIRED_KEY = "OPENPI_SRC"
RETIRED_MANIFEST_FIELDS = {
    RETIRED_KEY: "/recorded/checkout/third_party/openpi",
}


def test_retired_keys_match_by_their_exact_name():
    from real2sim.runner import environment

    assert environment.RETIRED_RUNNER_ENVIRONMENT == {RETIRED_KEY}
    for key in RETIRED_MANIFEST_FIELDS:
        assert environment.is_retired_runner_environment_key(key)
    # Only the exact upper-case variable names are history; anything else is a
    # field the manifest still has to account for.
    for key in (
        RETIRED_KEY.lower(),
        RETIRED_KEY + "_EXTRA",
        "CKPT",
        1,
    ):
        assert not environment.is_retired_runner_environment_key(key)
    assert environment.without_retired_environment(
        {**RETIRED_MANIFEST_FIELDS, "CKPT": "x", RETIRED_KEY.lower(): "y"}
    ) == {"CKPT": "x", RETIRED_KEY.lower(): "y"}


# Valid upper-case variable names that differ from the retired one only in its
# underscores. They were never exported by any launcher, so a manifest that
# records one has a field the current release must still account for.
UNDERSCORE_VARIANTS = {
    "absent": RETIRED_KEY.replace("_", ""),
    "repeated": RETIRED_KEY.replace("_", "__"),
    "trailing": RETIRED_KEY + "_",
    "moved": RETIRED_KEY.replace("_", "") + "_",
}


@pytest.mark.parametrize("variant", sorted(UNDERSCORE_VARIANTS))
def test_underscore_variants_of_a_retired_key_are_not_retired(variant):
    from real2sim.runner import environment

    key = UNDERSCORE_VARIANTS[variant]
    assert key != RETIRED_KEY
    # Still a well-formed variable name, so only the exact match tells it apart.
    assert environment._ENVIRONMENT_NAME.fullmatch(key)
    assert not environment.is_retired_runner_environment_key(key)
    assert environment.without_retired_environment(
        {**RETIRED_MANIFEST_FIELDS, "CKPT": "x", key: "y"}
    ) == {"CKPT": "x", key: "y"}


@pytest.mark.parametrize("variant", sorted(UNDERSCORE_VARIANTS))
def test_worker_contract_reports_an_underscore_variant_as_drift(tmp_path: Path, variant):
    key = UNDERSCORE_VARIANTS[variant]
    with _authority(tmp_path) as (manifest_path, manifest, _resolved):
        _as_pre_change(manifest_path, manifest)
        environment = manifest["resolved_environment"]
        assert isinstance(environment, dict)
        contract = (
            manifest_path.parent / "tiny_fixture_scene" / "run_01" / "official_worker_contract.json"
        )
        argv = _rollout_argv(manifest, contract, state_path=_state_path(environment))
        write_worker_contract(contract, manifest_path, argv)

        manifest["resolved_environment"] = {**environment, key: "/recorded/elsewhere"}
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with pytest.raises(WorkerContractError, match="environment differs"):
            write_worker_contract(contract, manifest_path, argv)


def _as_pre_change(manifest_path: Path, manifest: dict[str, object]) -> None:
    """Rewrite a fresh manifest the way a launcher recorded it before OpenPI was locked."""

    manifest["resolved_environment"] = {
        **manifest["resolved_environment"],  # type: ignore[dict-item]
        **RETIRED_MANIFEST_FIELDS,
    }
    shard_execution = manifest.get("shard_execution")
    if isinstance(shard_execution, dict):
        for record in shard_execution["shards"]:
            record["environment"] = {**record["environment"], **RETIRED_MANIFEST_FIELDS}
        shard_execution["aggregation_environment"] = {
            **shard_execution["aggregation_environment"],
            **RETIRED_MANIFEST_FIELDS,
        }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_pre_change_manifest_still_authorizes_its_worker(tmp_path: Path):
    with _authority(tmp_path) as (manifest_path, manifest, _resolved):
        _as_pre_change(manifest_path, manifest)
        environment = manifest["resolved_environment"]
        assert isinstance(environment, dict)
        contract = (
            manifest_path.parent / "tiny_fixture_scene" / "run_01" / "official_worker_contract.json"
        )
        argv = _rollout_argv(manifest, contract, state_path=_state_path(environment))
        write_worker_contract(contract, manifest_path, argv)
        validate_worker_contract(contract, manifest_path, argv)

        # Only the retired fields are history; any other recorded difference
        # is still drift.
        manifest["resolved_environment"] = {**environment, "CKPT": "/another/checkpoint"}
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with pytest.raises(WorkerContractError, match="environment differs"):
            write_worker_contract(contract, manifest_path, argv)


def test_pre_change_sharded_manifest_still_authorizes_its_worker(tmp_path: Path):
    with _authority(tmp_path, repeats=4, sharded=True) as (manifest_path, manifest, _resolved):
        _as_pre_change(manifest_path, manifest)
        shard_execution = manifest["shard_execution"]
        assert isinstance(shard_execution, dict)
        shard_b = shard_execution["shards"][1]
        selected = dict(manifest)
        selected["resolved_environment"] = shard_b["environment"]
        selected["worker_contract"] = {"schema_version": 1, "static_argv": shard_b["static_argv"]}
        contract = (
            manifest_path.parent / "tiny_fixture_scene" / "run_03" / "official_worker_contract.json"
        )
        argv = _rollout_argv(
            selected, contract, state_path=_state_path(shard_b["environment"])
        )
        write_worker_contract(contract, manifest_path, argv)
        validate_worker_contract(contract, manifest_path, argv)


def _official_drift_check() -> str:
    """The manifest check scripts/run_policy_batch.sh runs before a strict launch."""

    text = (REPO_ROOT / "scripts" / "run_policy_batch.sh").read_text(encoding="utf-8")
    start = text.index("<<'PY'\nimport json\nimport hashlib\n") + len("<<'PY'\n")
    return text[start : text.index("\nPY\n", start)]


def _run_drift_check(manifest_path: Path, environment: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-", str(manifest_path), str(manifest_path.parent), "1", "1", "", "0"],
        input=_official_drift_check(),
        env={"PATH": os.environ.get("PATH", ""), **environment},
        capture_output=True,
        text=True,
        check=False,
    )


def test_runner_replays_a_pre_change_manifest_without_reporting_drift(tmp_path: Path):
    with _authority(tmp_path) as (manifest_path, manifest, _resolved):
        current = {key: str(value) for key, value in manifest["resolved_environment"].items()}  # type: ignore[union-attr]
        _as_pre_change(manifest_path, manifest)

        # The runner exports today's environment, which no longer has the
        # retired fields the manifest recorded.
        replay = _run_drift_check(manifest_path, current)
        assert replay.returncode == 0, replay.stderr

        drifted = _run_drift_check(manifest_path, {**current, "CKPT": "/another/checkpoint"})
        assert drifted.returncode != 0
        assert "differs from launcher manifest: CKPT" in drifted.stderr


@pytest.mark.parametrize("variant", sorted(UNDERSCORE_VARIANTS))
def test_runner_reports_an_underscore_variant_as_drift(tmp_path: Path, variant):
    key = UNDERSCORE_VARIANTS[variant]
    with _authority(tmp_path) as (manifest_path, manifest, _resolved):
        current = {name: str(value) for name, value in manifest["resolved_environment"].items()}  # type: ignore[union-attr]
        manifest["resolved_environment"] = {
            **manifest["resolved_environment"],  # type: ignore[dict-item]
            key: "/recorded/elsewhere",
        }
        _as_pre_change(manifest_path, manifest)

        drifted = _run_drift_check(manifest_path, current)
        assert drifted.returncode != 0
        assert f"differs from launcher manifest: {key}" in drifted.stderr
