# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml

from real2sim import launcher
from real2sim import cli as real2sim_cli
from real2sim.runner import execute as runner_execute
from tests._helpers.launcher import base_config
from tests._helpers.repo import REPO_ROOT


@pytest.fixture(autouse=True)
def _clean_behavior_environment(monkeypatch, isolated_environ):
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)
    return isolated_environ


def _shard(
    name: str,
    start: int,
    stop: int,
    *,
    job_id: str,
    **overrides,
) -> dict:
    value = {
        "name": name,
        "repeats": [start, stop],
        "slurm_job_id": job_id,
        "gs_cuda_devices": "0,1",
        "sim_cuda_devices": "0,1",
        "policy_cuda_device": "0",
        "policy_batch_size": 2,
    }
    value.update(overrides)
    return value


def _source(*, shards=None) -> dict:
    source = base_config()
    source["runner"].update(
        mode="fast_parallel",
        repeats=4,
        policy_batch_size=2,
        expected_rollouts=4,
    )
    if shards is not None:
        source["runner"]["shards"] = shards
    return source


def _load(tmp_path: Path, source: dict):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "shards.yaml"
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return launcher.load_rollout_config(
        config_path,
        repo_root=tmp_path,
        timestamp="20990101_000000",
    )


def test_omitting_shards_preserves_the_existing_runner_environment(tmp_path: Path):
    plan = _load(tmp_path, _source())
    environment = launcher.build_runner_environment(plan)

    assert set(plan["resolved"]["runner"]) == {
        "mode",
        "repeats",
        "parallel_repeats",
        "policy_batch_size",
        "frames",
        "max_wall_sec",
        "expected_rollouts",
    }
    assert environment["REPEATS"] == "4"
    assert environment["PARALLEL_REPEATS"] == "2"
    assert environment["POLICY_BATCH_SIZE"] == "2"
    assert not {
        "REPEAT_START",
        "REPEAT_STOP",
        "REAL2SIM_SHARD_NAME",
        "REAL2SIM_AGGREGATE_ONLY",
    }.intersection(environment)


def test_omitting_shards_keeps_the_direct_runner_argv_and_environment(
    tmp_path: Path, monkeypatch
):
    plan = _load(tmp_path, _source())
    expected_environment = launcher.build_runner_environment(plan)
    run = mock.Mock(return_value=SimpleNamespace(returncode=0))
    popen = mock.Mock(side_effect=AssertionError("unsharded launch used srun"))
    monkeypatch.setattr(
        runner_execute,
        "validate_launch_preflight",
        lambda plan: {"gpu_bindings": None},
    )
    monkeypatch.setattr(
        runner_execute, "prepare_run_manifest", lambda plan, report: None
    )
    monkeypatch.setattr(
        runner_execute, "persist_config_provenance", lambda plan, manifest: None
    )
    monkeypatch.setattr(
        runner_execute,
        "summarize_run",
        lambda plan: (0, {"total_runs": plan["expected_rollouts"]}, {}),
    )
    monkeypatch.setattr(runner_execute.subprocess, "run", run)
    monkeypatch.setattr(runner_execute.subprocess, "Popen", popen)

    assert runner_execute.execute(plan) == 0
    run.assert_called_once_with(
        [str(plan["runner_script"])],
        cwd=runner_execute.REPO_ROOT,
        env=expected_environment,
        check=False,
    )
    popen.assert_not_called()


@pytest.mark.parametrize(
    ("shards", "message"),
    (
        ([], "must not be empty"),
        (
            [
                _shard("a", 1, 3, job_id="101", policy_batch_size=1),
                _shard("b", 3, 4, job_id="102", policy_batch_size=1),
            ],
            "overlap",
        ),
        (
            [
                _shard("a", 1, 1, job_id="101", policy_batch_size=1),
                _shard("b", 3, 4, job_id="102", policy_batch_size=1),
            ],
            "gap",
        ),
    ),
)
def test_shard_coverage_is_fail_closed(tmp_path: Path, shards, message: str):
    with pytest.raises(launcher.ConfigError, match=message):
        _load(tmp_path, _source(shards=shards))


def test_shard_policy_batch_must_fit_its_range(tmp_path: Path):
    shards = [
        _shard("a", 1, 1, job_id="101", policy_batch_size=2),
        _shard("b", 2, 4, job_id="102", policy_batch_size=1),
    ]
    with pytest.raises(launcher.ConfigError, match="must not exceed"):
        _load(tmp_path, _source(shards=shards))


def test_omitted_shard_job_id_uses_the_launcher_slurm_job(
    tmp_path: Path, monkeypatch
):
    shards = [
        _shard(
            "a", 1, 2, job_id="unused",
            gs_cuda_devices="0", sim_cuda_devices="0", policy_cuda_device="0",
        ),
        _shard(
            "b", 3, 4, job_id="unused",
            gs_cuda_devices="1", sim_cuda_devices="1", policy_cuda_device="1",
        ),
    ]
    for shard in shards:
        del shard["slurm_job_id"]
    monkeypatch.setenv("SLURM_JOB_ID", "777")

    plan = _load(tmp_path, _source(shards=shards))

    assert [
        shard["slurm_job_id"] for shard in plan["resolved"]["runner"]["shards"]
    ] == ["777", "777"]
    assert plan["shard_resolution_environment"] == {"SLURM_JOB_ID": "777"}


def test_automatic_port_assignment_avoids_every_shard_port_range(tmp_path: Path):
    source = _source(
        shards=[
            _shard("a", 1, 2, job_id="101"),
            _shard("b", 3, 4, job_id="102"),
        ]
    )
    # Exercise auto-allocation when the two default port sequences begin at the
    # same value; policy and GS ports must still be globally disjoint.
    source["runtime"]["server_port"] = 8321
    source["runtime"]["gs_port_base"] = 8321
    shards = _load(tmp_path, source)["resolved"]["runner"]["shards"]

    # Two selectors, two parallel repeats, one servers a GPU: two servers a
    # shard, so two ports each. Spelled out rather than recomputed, so the
    # assertion cannot follow production into the same mistake.
    _assert_shard_port_ranges_are_disjoint(shards, gs_ports_per_shard=2)


def _assert_shard_port_ranges_are_disjoint(shards, *, gs_ports_per_shard):
    """Every shard's policy port and whole GS range must be its own.

    ``gs_ports_per_shard`` is stated by the caller, not derived from the
    production formula: an assertion that recomputed the span the same way
    would pass for any span the code chose, including a wrong one.
    """

    occupied: set[int] = set()
    for shard in shards:
        base = int(shard["gs_port_base"])
        ports = {
            int(shard["server_port"]),
            *range(base, base + gs_ports_per_shard),
        }
        assert occupied.isdisjoint(ports)
        occupied.update(ports)


def test_two_servers_a_gpu_widens_every_shard_gs_range(tmp_path: Path):
    """A shard that starts four GS servers must be reserved four ports.

    The shell starts ``min(parallel_repeats, selectors * servers_a_gpu)``
    servers per scene, so the batch size is raised to four here: with two
    selectors and two servers a GPU it is the servers that bind, and the
    shard really does open four ports. At the fixture's default batch of two
    the repeats would bind instead, and the reservation would look correct
    even without the fix.
    """

    source = _source(
        shards=[
            _shard("a", 1, 4, job_id="101", policy_batch_size=4),
            _shard("b", 5, 8, job_id="102", policy_batch_size=4),
        ]
    )
    source["runner"]["repeats"] = 8
    source["runner"]["policy_batch_size"] = 4
    source["runner"]["expected_rollouts"] = 8
    source["runtime"]["gs_servers_per_gpu"] = 2
    source["runtime"]["server_port"] = 8321
    source["runtime"]["gs_port_base"] = 8321
    shards = _load(tmp_path, source)["resolved"]["runner"]["shards"]

    first, second = (int(shard["gs_port_base"]) for shard in shards)
    assert abs(second - first) >= 4
    _assert_shard_port_ranges_are_disjoint(shards, gs_ports_per_shard=4)


def test_explicit_shard_port_collision_is_rejected(tmp_path: Path):
    shards = [
        _shard("a", 1, 2, job_id="101", server_port=9000),
        _shard("b", 3, 4, job_id="102", gs_port_base=9000),
    ]
    with pytest.raises(launcher.ConfigError, match="ports collide"):
        _load(tmp_path, _source(shards=shards))


def test_cli_shards_override_the_yaml(tmp_path: Path, capsys, isolated_environ):
    source = _source(
        shards=[
            _shard("yaml-a", 1, 2, job_id="901"),
            _shard("yaml-b", 3, 4, job_id="902"),
        ]
    )
    with tempfile.TemporaryDirectory(dir=REPO_ROOT / "config") as temporary:
        config_path = Path(temporary) / "shards.yaml"
        config_path.write_text(
            yaml.safe_dump(source, sort_keys=False), encoding="utf-8"
        )
        exit_code = real2sim_cli.main(
            [
                "--config",
                str(config_path),
                "--dry-run",
                "--env-file",
                str(isolated_environ),
                "--timestamp",
                "20990101_000000",
                "--shards",
                "111:0,1:1-2 222:2,3:3-4",
            ]
        )

    assert exit_code == 0
    view = json.loads(capsys.readouterr().out)
    assert [shard["slurm_job_id"] for shard in view["shards"]] == ["111", "222"]
    assert [shard["repeats"] for shard in view["shards"]] == [[1, 2], [3, 4]]
    assert [shard["name"] for shard in view["shards"]] == ["01", "02"]
    assert all("yaml" not in shard["name"] for shard in view["shards"])


class _FakeProcess:
    def __init__(self, returncode):
        self.returncode = returncode
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise runner_execute.subprocess.TimeoutExpired("srun", timeout)
        return self.returncode


def test_two_shards_may_not_claim_the_same_gpu_of_one_job(tmp_path: Path):
    source = _source(
        shards=[
            _shard("a", 1, 2, job_id="101", gs_cuda_devices="0", sim_cuda_devices="0"),
            _shard("b", 3, 4, job_id="101", gs_cuda_devices="0", sim_cuda_devices="0"),
        ]
    )
    with pytest.raises(Exception) as excinfo:
        _load(tmp_path, source)
    assert "both claim GPU 0 of Slurm job 101" in str(excinfo.value)


def test_the_same_gpu_index_in_two_jobs_is_two_distinct_devices(tmp_path: Path):
    source = _source(
        shards=[
            _shard("a", 1, 2, job_id="101", gs_cuda_devices="0", sim_cuda_devices="0"),
            _shard("b", 3, 4, job_id="102", gs_cuda_devices="0", sim_cuda_devices="0"),
        ]
    )
    plan = _load(tmp_path, source)
    shards = plan["resolved"]["runner"]["shards"]
    assert [(item["slurm_job_id"], item["gs_cuda_devices"]) for item in shards] == [
        ("101", "0"),
        ("102", "0"),
    ]


def test_two_shards_may_not_claim_the_same_cpu_of_one_job(tmp_path: Path):
    source = _source(
        shards=[
            _shard(
                "a", 1, 2, job_id="101",
                gs_cuda_devices="0", sim_cuda_devices="0", policy_cuda_device="0",
                cpus="4,5,6",
            ),
            _shard(
                "b", 3, 4, job_id="101",
                gs_cuda_devices="1", sim_cuda_devices="1", policy_cuda_device="1",
                cpus="6,7,8",
            ),
        ]
    )
    with pytest.raises(Exception) as excinfo:
        _load(tmp_path, source)
    assert "both claim CPU 6 of Slurm job 101" in str(excinfo.value)


def test_a_shard_may_not_set_both_cpus_and_cpus_per_task(tmp_path: Path):
    source = _source(
        shards=[
            _shard("a", 1, 4, job_id="101", cpus="0,1", cpus_per_task=2),
        ]
    )
    source["runner"]["shards"][0]["repeats"] = [1, 4]
    with pytest.raises(Exception) as excinfo:
        _load(tmp_path, source)
    assert "sets both cpus and cpus_per_task" in str(excinfo.value)


def test_named_cpus_pin_the_runner_and_drop_the_shared_cpu_count(tmp_path: Path):
    source = _source(
        shards=[
            _shard(
                "a", 1, 2, job_id="101",
                gs_cuda_devices="0", sim_cuda_devices="0", policy_cuda_device="0",
                cpus="1,3,5",
            ),
            _shard(
                "b", 3, 4, job_id="101",
                gs_cuda_devices="1", sim_cuda_devices="1", policy_cuda_device="1",
                cpus="7,9,11",
            ),
        ]
    )
    plan = _load(tmp_path, source)
    shards = plan["resolved"]["runner"]["shards"]
    script = str(plan["runner_script"])

    assert runner_execute._srun_command(script, shards[0]) == [
        "srun",
        "--jobid=101",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        "--cpu-bind=none",
        "taskset",
        "-c",
        "1,3,5",
        script,
    ]
    assert "--cpus-per-task=3" not in runner_execute._srun_command(script, shards[1])
    # The aggregation step runs alone, so it keeps the whole allocation.
    assert runner_execute._srun_command(script, shards[0], aggregate=True) == [
        "srun",
        "--jobid=101",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        "--cpu-bind=none",
        script,
    ]


def test_launcher_starts_all_shards_with_srun_then_aggregates_once(
    tmp_path: Path, monkeypatch
):
    source = _source(
        shards=[
            _shard("a", 1, 2, job_id="101", cpus_per_task=8),
            _shard("b", 3, 4, job_id="102"),
        ]
    )
    plan = _load(tmp_path, source)
    Path(plan["run_root"]).mkdir(parents=True)
    processes = [_FakeProcess(0), _FakeProcess(0)]
    popen = mock.Mock(side_effect=processes)
    run = mock.Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(runner_execute.subprocess, "Popen", popen)
    monkeypatch.setattr(runner_execute.subprocess, "run", run)

    assert runner_execute._execute_shards(
        plan, runtime_manifest_path=None, runtime_manifest_digest=None
    ) == 0

    commands = [call.args[0] for call in popen.call_args_list]
    assert commands == [
        [
            "srun",
            "--jobid=101",
            "--overlap",
            "--nodes=1",
            "--ntasks=1",
            "--cpu-bind=none",
            "--cpus-per-task=8",
            str(plan["runner_script"]),
        ],
        [
            "srun",
            "--jobid=102",
            "--overlap",
            "--nodes=1",
            "--ntasks=1",
            "--cpu-bind=none",
            str(plan["runner_script"]),
        ],
    ]
    assert [call.kwargs["env"]["REPEAT_START"] for call in popen.call_args_list] == [
        "1",
        "3",
    ]
    assert [call.kwargs["env"]["REPEAT_STOP"] for call in popen.call_args_list] == [
        "2",
        "4",
    ]
    assert all(
        "CUDA_VISIBLE_DEVICES" not in call.kwargs["env"]
        and "REAL2SIM_GPU_BINDINGS_JSON" not in call.kwargs["env"]
        for call in popen.call_args_list
    )
    run.assert_called_once()
    assert run.call_args.args[0][:6] == [
        "srun",
        "--jobid=101",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        "--cpu-bind=none",
    ]
    assert run.call_args.kwargs["env"]["REAL2SIM_AGGREGATE_ONLY"] == "1"
    assert (Path(plan["run_root"]) / "shard_a.log").is_file()
    assert (Path(plan["run_root"]) / "shard_b.log").is_file()


def test_launcher_stops_other_shards_and_skips_aggregation_on_failure(
    tmp_path: Path, monkeypatch
):
    plan = _load(
        tmp_path,
        _source(
            shards=[
                _shard("a", 1, 2, job_id="101"),
                _shard("b", 3, 4, job_id="102"),
            ]
        ),
    )
    Path(plan["run_root"]).mkdir(parents=True)
    failed = _FakeProcess(7)
    running = _FakeProcess(None)
    monkeypatch.setattr(
        runner_execute.subprocess,
        "Popen",
        mock.Mock(side_effect=[failed, running]),
    )
    run = mock.Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(runner_execute.subprocess, "run", run)

    assert runner_execute._execute_shards(
        plan, runtime_manifest_path=None, runtime_manifest_digest=None
    ) == 7
    assert running.terminated is True
    run.assert_not_called()
