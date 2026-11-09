# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from real2sim.gpu_binding import (
    GPU_BINDINGS_ENV,
    GpuBindingError,
    contract_from_environment,
    contract_json,
    resolve_gpu_binding,
    resolve_gpu_contract,
    validate_gpu_contract,
)


NVIDIA = {
    "gpus": [
        {"index": "2", "uuid": "GPU-a"},
        {"index": "4", "uuid": "GPU-b"},
        {"index": "7", "uuid": "GPU-c"},
    ]
}
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "run_policy_batch.sh"
FIXTURE_SCENE = REPO_ROOT / "tests" / "fixtures" / "artifacts" / "scene"


def _roles(**overrides: str) -> dict[str, str]:
    environment = {
        "GS_CUDA_DEVICES": "0,1",
        "POLICY_CUDA_DEVICE": "0",
        "SIM_CUDA_DEVICES": "1",
    }
    environment.update(overrides)
    return environment


def test_visible_slots_resolve_exact_cuda_tokens_and_physical_egl_indices() -> None:
    environment = _roles(CUDA_VISIBLE_DEVICES="GPU-a,GPU-c")

    contract = resolve_gpu_contract(environment, nvidia=NVIDIA)

    assert contract == {
        "schema_version": 1,
        "selection_mode": "visible_slots",
        "cuda_visible_devices": "GPU-a,GPU-c",
        "roles": {
            "gs": [
                {
                    "allocation_slot": 0,
                    "cuda_token": "GPU-a",
                    "egl_physical_index": 2,
                },
                {
                    "allocation_slot": 1,
                    "cuda_token": "GPU-c",
                    "egl_physical_index": 7,
                },
            ],
            "policy": [
                {
                    "allocation_slot": 0,
                    "cuda_token": "GPU-a",
                    "egl_physical_index": 2,
                }
            ],
            "sim": [
                {
                    "allocation_slot": 1,
                    "cuda_token": "GPU-c",
                    "egl_physical_index": 7,
                }
            ],
        },
    }


def test_numeric_visibility_is_not_replaced_by_bare_local_slot() -> None:
    environment = _roles(
        CUDA_VISIBLE_DEVICES="4,7",
        GS_CUDA_DEVICES="1",
        POLICY_CUDA_DEVICE="1",
        SIM_CUDA_DEVICES="1",
    )

    binding = resolve_gpu_contract(environment, nvidia=NVIDIA)["roles"]["gs"][0]

    assert binding == {
        "allocation_slot": 1,
        "cuda_token": "7",
        "egl_physical_index": 7,
    }


def test_selectors_are_host_ordinals_when_visibility_is_absent() -> None:
    environment = _roles(
        GS_CUDA_DEVICES="4",
        POLICY_CUDA_DEVICE="4",
        SIM_CUDA_DEVICES="4",
    )

    contract = resolve_gpu_contract(environment, nvidia=NVIDIA)

    assert contract["selection_mode"] == "host_ordinals"
    assert contract["cuda_visible_devices"] is None
    assert contract["roles"]["sim"][0]["cuda_token"] == "4"
    assert contract["roles"]["sim"][0]["egl_physical_index"] == 4


@pytest.mark.parametrize(
    ("visible", "message"),
    (
        ("", "set but empty"),
        ("GPU-a,,GPU-c", "malformed"),
        ("GPU-a, GPU-c", "malformed"),
        ("GPU-a,GPU-a", "duplicate"),
        ("MIG-GPU-a/1/0", "MIG CUDA device tokens"),
        ("not-a-device", "unsupported CUDA device token"),
    ),
)
def test_invalid_visibility_boundaries_fail_closed(
    visible: str, message: str
) -> None:
    environment = _roles(
        CUDA_VISIBLE_DEVICES=visible,
        GS_CUDA_DEVICES="0",
        POLICY_CUDA_DEVICE="0",
        SIM_CUDA_DEVICES="0",
    )

    with pytest.raises(GpuBindingError, match=message):
        resolve_gpu_contract(environment, nvidia=NVIDIA)


def test_out_of_range_visible_slot_fails_closed() -> None:
    environment = _roles(
        CUDA_VISIBLE_DEVICES="GPU-a",
        GS_CUDA_DEVICES="1",
        POLICY_CUDA_DEVICE="1",
        SIM_CUDA_DEVICES="1",
    )

    with pytest.raises(GpuBindingError, match="outside CUDA_VISIBLE_DEVICES"):
        resolve_gpu_contract(environment, nvidia=NVIDIA)


@pytest.mark.parametrize(
    ("environment", "message"),
    (
        (
            _roles(
                CUDA_VISIBLE_DEVICES="GPU-unknown",
                GS_CUDA_DEVICES="0",
                POLICY_CUDA_DEVICE="0",
                SIM_CUDA_DEVICES="0",
            ),
            "no exact host EGL mapping",
        ),
        (
            _roles(
                GS_CUDA_DEVICES="9",
                POLICY_CUDA_DEVICE="9",
                SIM_CUDA_DEVICES="9",
            ),
            "not a host GPU",
        ),
        (
            _roles(GS_CUDA_DEVICES="0,0"),
            "duplicate GPU selectors",
        ),
    ),
)
def test_unknown_or_duplicate_role_selections_fail_closed(
    environment: dict[str, str], message: str
) -> None:
    with pytest.raises(GpuBindingError, match=message):
        resolve_gpu_contract(environment, nvidia=NVIDIA)


def test_unknown_unselected_visible_token_fails_the_atomic_boundary() -> None:
    environment = _roles(
        CUDA_VISIBLE_DEVICES="GPU-a,GPU-unknown",
        GS_CUDA_DEVICES="0",
        POLICY_CUDA_DEVICE="0",
        SIM_CUDA_DEVICES="0",
    )

    with pytest.raises(GpuBindingError, match="no exact host EGL mapping"):
        resolve_gpu_contract(environment, nvidia=NVIDIA)


def test_no_implicit_host_gpu_zero_without_visibility() -> None:
    with pytest.raises(GpuBindingError, match="no GPU selection is configured"):
        resolve_gpu_contract({}, nvidia=NVIDIA)


def test_sealed_contract_rejects_visibility_or_role_drift() -> None:
    environment = _roles(CUDA_VISIBLE_DEVICES="GPU-a,GPU-c")
    contract = resolve_gpu_contract(environment, nvidia=NVIDIA)

    assert validate_gpu_contract(contract, environment) == contract
    with pytest.raises(GpuBindingError, match="visibility|CUDA_VISIBLE_DEVICES"):
        validate_gpu_contract(
            contract,
            _roles(CUDA_VISIBLE_DEVICES="GPU-b,GPU-c"),
        )
    with pytest.raises(GpuBindingError, match="changed the gs selector|invalid gs"):
        validate_gpu_contract(
            contract,
            _roles(
                CUDA_VISIBLE_DEVICES="GPU-a,GPU-c",
                GS_CUDA_DEVICES="1",
            ),
        )


def test_structural_fixture_contract_never_requeries_nvidia(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _roles(CUDA_VISIBLE_DEVICES="GPU-a,GPU-c")
    contract = resolve_gpu_contract(environment, nvidia=NVIDIA)
    environment[GPU_BINDINGS_ENV] = contract_json(contract)
    monkeypatch.setattr(
        "real2sim.gpu_binding._query_nvidia_gpu_records",
        lambda: (_ for _ in ()).throw(AssertionError("must not query")),
    )

    assert contract_from_environment(
        environment, verify_topology=False
    ) == contract
    assert json.loads(environment[GPU_BINDINGS_ENV]) == contract


def test_sealed_environment_contract_requeries_current_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _roles(CUDA_VISIBLE_DEVICES="GPU-a,GPU-c")
    contract = resolve_gpu_contract(environment, nvidia=NVIDIA)
    environment[GPU_BINDINGS_ENV] = contract_json(contract)
    calls = 0

    def query():
        nonlocal calls
        calls += 1
        return list(NVIDIA["gpus"])

    monkeypatch.setattr("real2sim.gpu_binding._query_nvidia_gpu_records", query)

    assert contract_from_environment(environment) == contract
    assert calls == 1


def test_sealed_contract_rejects_forged_uuid_egl_mapping() -> None:
    environment = _roles(CUDA_VISIBLE_DEVICES="GPU-a,GPU-c")
    forged = resolve_gpu_contract(environment, nvidia=NVIDIA)
    for entries in forged["roles"].values():
        for entry in entries:
            if entry["allocation_slot"] == 0:
                entry["egl_physical_index"] = 4
    environment[GPU_BINDINGS_ENV] = contract_json(forged)

    with pytest.raises(GpuBindingError, match="current NVIDIA topology"):
        contract_from_environment(environment, nvidia=NVIDIA)


def test_sealed_contract_rejects_unknown_uuid_even_when_internally_consistent() -> None:
    environment = _roles(CUDA_VISIBLE_DEVICES="GPU-a,GPU-c")
    forged = resolve_gpu_contract(environment, nvidia=NVIDIA)
    forged["cuda_visible_devices"] = "GPU-unknown,GPU-c"
    for entries in forged["roles"].values():
        for entry in entries:
            if entry["allocation_slot"] == 0:
                entry["cuda_token"] = "GPU-unknown"
    environment["CUDA_VISIBLE_DEVICES"] = "GPU-unknown,GPU-c"
    environment[GPU_BINDINGS_ENV] = contract_json(forged)

    with pytest.raises(GpuBindingError, match="no exact host EGL mapping"):
        contract_from_environment(environment, nvidia=NVIDIA)


def test_sealed_contract_rejects_out_of_bound_visible_slot() -> None:
    environment = _roles(
        CUDA_VISIBLE_DEVICES="GPU-a",
        GS_CUDA_DEVICES="1",
        POLICY_CUDA_DEVICE="1",
        SIM_CUDA_DEVICES="1",
    )
    entry = {
        "allocation_slot": 1,
        "cuda_token": "GPU-c",
        "egl_physical_index": 7,
    }
    forged = {
        "schema_version": 1,
        "selection_mode": "visible_slots",
        "cuda_visible_devices": "GPU-a",
        "roles": {
            "gs": [dict(entry)],
            "policy": [dict(entry)],
            "sim": [dict(entry)],
        },
    }
    environment[GPU_BINDINGS_ENV] = contract_json(forged)

    with pytest.raises(GpuBindingError, match="outside CUDA_VISIBLE_DEVICES"):
        contract_from_environment(environment, nvidia=NVIDIA)


def test_sealed_contract_rejects_tampered_physical_mapping() -> None:
    environment = _roles(CUDA_VISIBLE_DEVICES="GPU-a,GPU-c")
    contract = resolve_gpu_contract(environment, nvidia=NVIDIA)
    contract["roles"]["gs"][0]["egl_physical_index"] = 7

    with pytest.raises(GpuBindingError, match="distinct CUDA devices|inconsistent"):
        validate_gpu_contract(contract, environment)


def test_single_binding_query_rejects_unknown_numeric_host_device() -> None:
    with pytest.raises(GpuBindingError, match="not a host GPU"):
        resolve_gpu_binding(9, environment={}, nvidia=NVIDIA)


def _batch_runner_dry_run_environment(
    tmp_path: Path,
    *,
    visible_devices: str,
    gs_devices: str,
    policy_device: str,
    sim_devices: str,
    parallel_repeats: int,
) -> tuple[dict[str, str], Path]:
    """Return a complete ``--dry-run`` environment and its binding capture file."""

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/bin/sh\n"
        + "".join(
            f"printf '%s\\n' '{index}, GPU-{index}'\n"
            for index in range(8)
        ),
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)
    bindings_seen = tmp_path / "bindings-seen"
    python_wrapper = tmp_path / "python-wrapper"
    python_wrapper.write_text(
        "#!/bin/sh\n"
        'if [ -n "${REAL2SIM_GPU_BINDINGS_JSON:-}" ]; then\n'
        '  printf "seen\\n" >> "${GPU_BINDINGS_CAPTURE}"\n'
        "fi\n"
        'exec "${GPU_BINDINGS_PYTHON}" "$@"\n',
        encoding="utf-8",
    )
    python_wrapper.chmod(0o755)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    run_root = tmp_path / "run"
    environment = {
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', os.defpath)}",
        "PYTHONNOUSERSITE": "1",
    }
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": visible_devices,
            GPU_BINDINGS_ENV: "",
            "REAL2SIM_SHARD_NAME": "",
            "REAL2SIM_AGGREGATE_ONLY": "0",
            "REAL2SIM_STRICT_LAUNCH": "0",
            "REAL2SIM_OFFICIAL_RUN_MANIFEST": "",
            "REAL2SIM_RUNTIME_SOURCE_MANIFEST": "",
            "REAL2SIM_RUNTIME_SOURCE_MANIFEST_SHA256": "",
            "PY": str(python_wrapper),
            "GPU_BINDINGS_CAPTURE": str(bindings_seen),
            "GPU_BINDINGS_PYTHON": sys.executable,
            "CKPT": str(checkpoint),
            "CONFIG": "fixture",
            "OBJECTS": "cup_plate",
            "INSTRUCTION": "Place the cup on the plate.",
            "GOAL_CONDITION": "cup_on_plate",
            "SCORE_CONDITION": "cup_on_plate_contact_points",
            "GS_SCENE_DIRS": str(FIXTURE_SCENE),
            "RUN_ROOT": str(run_root),
            "EXPORTS_ROOT": str(tmp_path / "outputs"),
            "REPEATS": str(parallel_repeats),
            "REPEAT_START": "1",
            "REPEAT_STOP": str(parallel_repeats),
            "PARALLEL_REPEATS": str(parallel_repeats),
            "POLICY_BATCH_SIZE": str(parallel_repeats),
            "FRAMES": "1",
            "DETERMINISTIC_SIM": "0",
            "EXTRA_SIM_ARGS_JSON": '["--no-render"]',
            "EXTRA_SIM_ARGS": "",
            "INIT_STATE_JSON": "",
            "POSTHOC_RENDER_REQUIRED": "0",
            "RUN_ANALYSIS": "0",
            "GS_CUDA_DEVICES": gs_devices,
            "GS_SERVERS_PER_GPU": "1",
            "POLICY_CUDA_DEVICE": policy_device,
            "SIM_CUDA_DEVICES": sim_devices,
            "RENDER_LOCK_ROOT": str(tmp_path / "render-locks"),
            "GAUSSIAN_WRAPPING_REPO": str(FIXTURE_SCENE),
            "GRID_TEMPLATE": str(
                REPO_ROOT
                / "config"
                / "render_templates"
                / "rollout_grid_clear_headers_4x3.json"
            ),
        }
    )
    return environment, bindings_seen


@pytest.mark.parametrize(
    (
        "visible_devices",
        "gs_devices",
        "policy_device",
        "sim_devices",
        "parallel_repeats",
        "expected_lock_indices",
    ),
    (
        ("0", "0", "0", "0", 1, [0]),
        ("1", "0", "0", "0", 1, [1]),
        ("3,5", "0", "0", "0", 1, [3]),
        ("3,5", "0,1", "0", "0,1", 2, [3, 5]),
        ("GPU-3,GPU-5", "0,1", "0", "0,1", 2, [3, 5]),
    ),
)
def test_batch_runner_render_lock_uses_physical_egl_index(
    tmp_path: Path,
    visible_devices: str,
    gs_devices: str,
    policy_device: str,
    sim_devices: str,
    parallel_repeats: int,
    expected_lock_indices: list[int],
) -> None:
    environment, bindings_seen = _batch_runner_dry_run_environment(
        tmp_path,
        visible_devices=visible_devices,
        gs_devices=gs_devices,
        policy_device=policy_device,
        sim_devices=sim_devices,
        parallel_repeats=parallel_repeats,
    )

    completed = subprocess.run(
        [str(RUNNER_SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert bindings_seen.read_text(encoding="utf-8").splitlines()
    visible_tokens = visible_devices.split(",")
    for selector in gs_devices.split(","):
        token = visible_tokens[int(selector)]
        physical_index = int(token.removeprefix("GPU-"))
        assert (
            f"selector={selector} cuda={token} egl={physical_index}"
            in completed.stdout
        )
    policy_token = visible_tokens[int(policy_device)]
    policy_physical_index = int(policy_token.removeprefix("GPU-"))
    assert (
        f"selector={policy_device} cuda={policy_token} egl={policy_physical_index}"
        in completed.stdout
    )
    actual_lock_indices = [
        int(value)
        for value in re.findall(
            r"mujoco_render_faces_gpu_([0-9]+)\.lock",
            completed.stdout,
        )
    ]
    assert sorted(actual_lock_indices) == sorted(expected_lock_indices)


def test_batch_runner_refuses_the_retired_portable_policy_endpoint(
    tmp_path: Path,
) -> None:
    environment, bindings_seen = _batch_runner_dry_run_environment(
        tmp_path,
        visible_devices="0",
        gs_devices="0",
        policy_device="0",
        sim_devices="0",
        parallel_repeats=1,
    )

    def dry_run(endpoint: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(RUNNER_SCRIPT), "--dry-run"],
            cwd=REPO_ROOT,
            env={**environment, "PORTABLE_POLICY_SERVER": endpoint},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    # Before the bridge was removed an empty value meant the local checkpoint,
    # and the identical environment still plans the checkpoint server.
    local = dry_run("")
    assert local.returncode == 0, local.stdout + local.stderr
    assert "PORTABLE_POLICY_SERVER" not in local.stderr
    server_lines = [
        line for line in local.stdout.splitlines() if "server command:" in line
    ]
    assert len(server_lines) == 1
    assert f"--policy {environment['CKPT']}" in server_lines[0]
    assert bindings_seen.read_text(encoding="utf-8").splitlines()
    bindings_seen.unlink()

    # An endpoint asked for a remote policy. Starting the checkpoint instead
    # would report another policy's results, so nothing may be planned.
    refused = dry_run("127.0.0.1:8000")
    assert refused.returncode == 2, refused.stdout + refused.stderr
    assert "PORTABLE_POLICY_SERVER is retired" in refused.stderr
    assert "server command:" not in refused.stdout
    assert not bindings_seen.exists()


@pytest.mark.parametrize("robot", ["UR5", "fr3_duo_x"])
def test_batch_runner_refuses_an_unsupported_robot(
    tmp_path: Path, robot: str
) -> None:
    environment, bindings_seen = _batch_runner_dry_run_environment(
        tmp_path,
        visible_devices="0",
        gs_devices="0",
        policy_device="0",
        sim_devices="0",
        parallel_repeats=1,
    )
    refused = subprocess.run(
        [str(RUNNER_SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        env={**environment, "ROBOT": robot},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert refused.returncode == 2, refused.stdout + refused.stderr
    assert f"ROBOT/--robot must be FR3 or FR3_DUO, got {robot}" in refused.stderr
    assert "server command:" not in refused.stdout
    assert not bindings_seen.exists()


def test_batch_runner_selects_first_and_tenth_box_layout_arguments(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\nprintf '%s\\n' '0, GPU-0'\n", encoding="utf-8")
    nvidia_smi.chmod(0o755)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    layouts = [str(tmp_path / f"layout-{index}.json") for index in range(10)]
    environment = {
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', os.defpath)}",
        "PY": sys.executable,
        "CKPT": str(checkpoint),
        "CONFIG": "fixture",
        "OBJECTS": "box_pack",
        "INSTRUCTION": "Put the pen in the box with the right gripper.",
        "GOAL_CONDITION": "box_pack_task",
        "SCORE_CONDITION": "box_pack_task",
        "GS_SCENE_DIRS": str(FIXTURE_SCENE),
        "RUN_ROOT": str(tmp_path / "run"),
        "EXPORTS_ROOT": str(tmp_path / "outputs"),
        "REPEATS": "10",
        "PARALLEL_REPEATS": "1",
        "POLICY_BATCH_SIZE": "1",
        "FRAMES": "1",
        "DETERMINISTIC_SIM": "0",
        "EXTRA_SIM_ARGS_JSON": '["--no-render"]',
        "EXTRA_SIM_ARGS": "",
        "INIT_STATE_JSON": "",
        "BOX_PACK_LAYOUT_JSON_LIST": json.dumps(layouts),
        "POSTHOC_RENDER_REQUIRED": "0",
        "RUN_ANALYSIS": "0",
        "GS_CUDA_DEVICES": "0",
        "GS_SERVERS_PER_GPU": "1",
        "POLICY_CUDA_DEVICE": "0",
        "SIM_CUDA_DEVICES": "0",
        "RENDER_LOCK_ROOT": str(tmp_path / "render-locks"),
        "GAUSSIAN_WRAPPING_REPO": str(FIXTURE_SCENE),
        "GRID_TEMPLATE": str(
            REPO_ROOT
            / "config"
            / "render_templates"
            / "rollout_grid_clear_headers_4x3.json"
        ),
        "REAL2SIM_STRICT_LAUNCH": "0",
        "REAL2SIM_OFFICIAL_RUN_MANIFEST": "",
        "REAL2SIM_RUNTIME_SOURCE_MANIFEST": "",
        "REAL2SIM_RUNTIME_SOURCE_MANIFEST_SHA256": "",
        "REAL2SIM_GPU_BINDINGS_JSON": "",
    }
    completed = subprocess.run(
        [str(RUNNER_SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    commands = [
        line for line in completed.stdout.splitlines()
        if line.startswith("[real2sim-batch] worker command:")
    ]
    assert len(commands) == 10
    run_01 = next(line for line in commands if "_run_01" in line)
    run_10 = next(line for line in commands if "_run_10" in line)
    assert f"--box-pack-layout-json {layouts[0]} --box-pack-layout-index 0" in run_01
    assert f"--box-pack-layout-json {layouts[9]} --box-pack-layout-index 9" in run_10
