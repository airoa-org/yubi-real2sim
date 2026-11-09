# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import csv

import json

from pathlib import Path

import subprocess



import pytest

from real2sim.gpu_binding import contract_json, resolve_gpu_contract

from robosuite_env import gs_launch


import robosuite_env.tools.rerender_third_person as rerender

from robosuite_env.rerender import cli as rerender_cli



from robosuite_env.rerender import run as rerender_run

from robosuite_env.rerender import sources as rerender_sources


from tests._helpers.rollouts import (
    RUN_COMMAND,
    _configure_render_state_source,
    _source_registration_summary,
    run_dir,  # noqa: F401 -- pytest fixture
)

@pytest.mark.parametrize("selected_present", (True, False))
def test_update_outputs_csv_after_rerender_updates_selected_rows_atomically(
    tmp_path: Path,
    selected_present: bool,
) -> None:
    run_root = tmp_path / "batch"
    selected = run_root / "scene_a" / "run_01"
    untouched = run_root / "scene_a" / "run_02"
    (selected / "rerender").mkdir(parents=True)
    untouched.mkdir(parents=True)
    validation = selected / "rerender" / "video_validation.json"
    validation.write_text('{"ok": true}\n', encoding="utf-8")
    outputs = run_root / "outputs.csv"
    header = "scene,run,exit_code,video,summary,video_validation,trajectory_png\n"
    selected_row = (
        f"scene_a,run_01,0,{selected / 'old.mp4'},summary-a.json,"
        f"{selected / 'video_validation.json'},missing-a.png\n"
    )
    untouched_row = (
        f"scene_a,run_02,0,{untouched / 'old.mp4'},summary-b.json,"
        f"{untouched / 'video_validation.json'},missing-b.png\n"
    )
    outputs.write_text(
        header + (selected_row if selected_present else "") + untouched_row,
        encoding="utf-8",
    )

    if not selected_present:
        with pytest.raises(rerender.RerenderError, match="has no row"):
            rerender.update_outputs_csv_after_rerender(run_root, [selected])
        return

    assert rerender.update_outputs_csv_after_rerender(run_root, [selected]) is True
    with outputs.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["video"] == str(
        (selected / "scene_a_run_01.mp4").resolve()
    )
    assert rows[0]["video_validation"] == str(validation.resolve())
    assert rows[0]["trajectory_png"] == "missing-a.png"
    assert rows[1]["video"] == str(untouched / "old.mp4")
    assert rows[1]["video_validation"] == str(
        untouched / "video_validation.json"
    )

def _mock_gpu_inventory(monkeypatch):
    from real2sim import gpu_binding
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("REAL2SIM_GPU_BINDINGS_JSON", raising=False)
    monkeypatch.setattr(gpu_binding, "_query_nvidia_gpu_records",
                        lambda: [{"index": "0", "uuid": "GPU-test-0"}])


@pytest.mark.parametrize("uuid_binding", (False, True))
def test_posthoc_gpu_lock_rejects_a_second_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uuid_binding: bool,
) -> None:
    if not uuid_binding:
        _mock_gpu_inventory(monkeypatch)
    first_owner: int | rerender.GpuBinding = 0
    competing_owner = 0
    physical_gpu = 0
    if uuid_binding:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("REAL2SIM_GPU_BINDINGS_JSON", raising=False)
        monkeypatch.setattr(
            rerender.subprocess,
            "run",
            lambda command, **_kwargs: rerender.subprocess.CompletedProcess(
                command,
                0,
                stdout="5, GPU-selected\n",
                stderr="",
            ),
        )
        first_owner = rerender.GpuBinding(
            allocation_slot=0,
            cuda_token="GPU-selected",
            egl_physical_index=5,
        )
        competing_owner = physical_gpu = 5

    first = rerender._acquire_posthoc_gpu_lock(first_owner, lock_root=tmp_path)
    try:
        with pytest.raises(
            rerender.RerenderError,
            match=f"owns physical GPU {physical_gpu}",
        ):
            rerender._acquire_posthoc_gpu_lock(
                competing_owner,
                lock_root=tmp_path,
            )
    finally:
        first.close()

    second = rerender._acquire_posthoc_gpu_lock(
        competing_owner,
        lock_root=tmp_path,
    )
    second.close()

def test_main_closes_gpu_lifecycle_lock_when_rerender_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_gpu_inventory(monkeypatch)

    class FakeLock:
        closed = False

        def close(self) -> None:
            self.closed = True

    lock = FakeLock()
    monkeypatch.setattr(
        rerender_cli,
        "_acquire_posthoc_gpu_lock",
        lambda _gpu: lock,
    )
    monkeypatch.setattr(
        rerender_cli,
        "prepare_selected_sources",
        lambda _arguments: ([], False),
    )

    def fail(_arguments, **_kwargs) -> int:
        raise rerender.RerenderError("failed after acquiring GPU")

    monkeypatch.setattr(rerender_cli, "_run_rerender", fail)
    monkeypatch.setattr(
        rerender.sys,
        "argv",
        ["rerender_third_person.py", str(tmp_path)],
    )

    with pytest.raises(rerender.RerenderError, match="failed after acquiring"):
        rerender.main()

    assert lock.closed is True

@pytest.mark.parametrize("invalid_render_state", (False, True))
def test_main_validates_every_source_before_acquiring_gpu(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_render_state: bool,
) -> None:
    if invalid_render_state:
        _configure_render_state_source(run_dir, frames=1)
        summary_path = run_dir / "sim_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        del summary["render_state_log"]
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        expected_error = "without a summary contract"
    else:
        second = run_dir.parent / "run_02"
        (second / "logs").mkdir(parents=True)
        (second / "RUN_COMMAND.md").write_text(
            RUN_COMMAND.replace("run_01", "run_02"), encoding="utf-8"
        )
        (second / "sim_summary.json").write_text(
            json.dumps(
                {
                    **_source_registration_summary(run_dir),
                    "execution_class": "diagnostic",
                    "frames": 1,
                    "wall_stopped_early": False,
                }
            ),
            encoding="utf-8",
        )
        (second / "logs" / "sim_log.jsonl").write_text(
            "{}\n",
            encoding="utf-8",
        )
        expected_error = "frame=None, expected 0"

    monkeypatch.setattr(
        rerender_sources,
        "_git_code_identity",
        lambda: {"commit": "a" * 40, "dirty": True},
    )
    touched: list[str] = []

    def unexpected_gpu(*_args, **_kwargs):
        touched.append("gpu")
        raise AssertionError("GPU setup must happen after source validation")

    def unexpected_renderer(*_args, **_kwargs):
        touched.append("renderer")
        raise AssertionError("renderer must not start for an invalid source")

    monkeypatch.setattr(rerender_cli, "resolve_gpu_binding", unexpected_gpu)
    monkeypatch.setattr(rerender_cli, "_acquire_posthoc_gpu_lock", unexpected_gpu)
    monkeypatch.setattr(rerender_run, "_start_gs_server", unexpected_renderer)
    monkeypatch.setattr(
        rerender.sys,
        "argv",
        ["rerender_third_person.py", str(run_dir.parents[1]), "--skip-grid"],
    )

    with pytest.raises(rerender.RerenderError, match=expected_error):
        rerender.main()

    assert touched == []

def test_recorded_scene_dir_resolves_relative_path_from_repository(
    run_dir: Path,
) -> None:
    _, argv = rerender.parse_run_command(run_dir)
    argv[argv.index("--gs-background") + 1] = "config/scenes/example"

    assert rerender._recorded_scene_dir(run_dir, argv) == (
        rerender.REPO_ROOT / "config" / "scenes" / "example"
    ).resolve()

@pytest.mark.parametrize("equals_form", (False, True))
def test_recorded_scene_dir_uses_the_scene_the_worker_loaded(
    run_dir: Path,
    equals_form: bool,
) -> None:
    # The batch runner appends the per-scene --gs-background after any extra
    # simulation arguments, and the worker's argparse keeps the last value.
    _, argv = rerender.parse_run_command(run_dir)
    index = argv.index("--gs-background")
    argv[index + 1] = "config/scenes/worker_loaded"
    final = (
        ["--gs-background=config/scenes/worker_loaded"]
        if equals_form
        else argv[index : index + 2]
    )
    del argv[index : index + 2]
    argv[2:2] = ["--gs-background", "config/scenes/overridden"]
    argv += final

    assert rerender._recorded_scene_dir(run_dir, argv) == (
        rerender.REPO_ROOT / "config" / "scenes" / "worker_loaded"
    ).resolve()


def test_recorded_scene_dir_rejects_a_command_without_a_scene(
    run_dir: Path,
) -> None:
    _, argv = rerender.parse_run_command(run_dir)
    index = argv.index("--gs-background")
    del argv[index : index + 2]

    with pytest.raises(rerender.RerenderError, match="no --gs-background value"):
        rerender._recorded_scene_dir(run_dir, argv)


def test_canonical_render_lock_matches_batch_runner_namespace() -> None:
    assert rerender.canonical_render_lock(
        rerender.GpuBinding(
            allocation_slot=3,
            cuda_token="3",
            egl_physical_index=3,
        )
    ) == (
        f"/tmp/real2sim-render-locks-{rerender.os.getuid()}/"
        "mujoco_render_faces_gpu_3.lock"
    )
    assert rerender.canonical_render_lock(
        rerender.GpuBinding(
            allocation_slot=0,
            cuda_token="GPU-selected",
            egl_physical_index=5,
        )
    ).endswith("/mujoco_render_faces_gpu_5.lock")

def test_gs_server_environment_caps_inherited_cpu_thread_pools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    monkeypatch.setenv("OMP_NUM_THREADS", "192")
    environment = rerender.gs_server_environment(1)

    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-b"
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        assert environment[name] == "1"

def test_resolve_gpu_binding_proves_numeric_physical_device_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setattr(
        rerender.subprocess,
        "run",
        lambda command, **_kwargs: rerender.subprocess.CompletedProcess(
            command,
            0,
            stdout="".join(f"{index}, GPU-{index}\n" for index in range(8)),
            stderr="",
        ),
    )

    assert rerender.resolve_gpu_binding(5) == rerender.GpuBinding(
        allocation_slot=5,
        cuda_token="5",
        egl_physical_index=5,
    )

def test_resolve_gpu_binding_maps_allocated_uuid_to_physical_egl_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c")

    def fake_run(command, **kwargs):
        assert command == [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ]
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        return rerender.subprocess.CompletedProcess(
            command,
            0,
            stdout="5, GPU-a\n2, GPU-b\n7, GPU-c\n",
            stderr="",
        )

    monkeypatch.setattr(rerender.subprocess, "run", fake_run)
    assert rerender.resolve_gpu_binding(2) == rerender.GpuBinding(
        allocation_slot=2,
        cuda_token="GPU-c",
        egl_physical_index=7,
    )

def test_resolve_gpu_binding_revalidates_sealed_contract_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b",
        "GS_CUDA_DEVICES": "0,1",
        "POLICY_CUDA_DEVICE": "0",
        "SIM_CUDA_DEVICES": "1",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    contract = resolve_gpu_contract(
        environment,
        nvidia={
            "gpus": [
                {"index": "5", "uuid": "GPU-a"},
                {"index": "2", "uuid": "GPU-b"},
            ]
        },
    )
    monkeypatch.setenv("REAL2SIM_GPU_BINDINGS_JSON", contract_json(contract))

    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        assert command == [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ]
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        return rerender.subprocess.CompletedProcess(
            command,
            0,
            stdout="5, GPU-a\n2, GPU-b\n",
            stderr="",
        )

    monkeypatch.setattr(
        rerender.subprocess,
        "run",
        fake_run,
    )

    assert rerender.resolve_gpu_binding(1) == rerender.GpuBinding(
        allocation_slot=1,
        cuda_token="GPU-b",
        egl_physical_index=2,
    )
    assert calls == 1

@pytest.mark.parametrize(
    ("visible_devices", "gpu", "query_output", "message"),
    (
        ("GPU-a,GPU-b", 2, None, "outside CUDA_VISIBLE_DEVICES"),
        ("GPU-a,,GPU-b", 0, None, "malformed CUDA_VISIBLE_DEVICES"),
        ("GPU-a,GPU-a", 0, None, "duplicate entries"),
        ("MIG-GPU-a/1/0", 0, None, "MIG CUDA device tokens"),
        ("GPU-a", 0, "5, GPU-other\n", "no exact host EGL mapping"),
        ("GPU-a", 0, "5, GPU-a\n5, GPU-b\n", "ambiguous NVIDIA GPU"),
        ("GPU-a", 0, "not-an-index, GPU-a\n", "invalid NVIDIA physical GPU index"),
        ("GPU-a", 0, "0, MIG-GPU-a/1/0\n", "unsupported NVIDIA GPU UUID"),
    ),
)
def test_resolve_gpu_binding_rejects_unsafe_allocation_tokens(
    monkeypatch: pytest.MonkeyPatch,
    visible_devices: str,
    gpu: int,
    query_output: str | None,
    message: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    if query_output is not None:
        monkeypatch.setattr(
            rerender.subprocess,
            "run",
            lambda command, **_kwargs: rerender.subprocess.CompletedProcess(
                command,
                0,
                stdout=query_output,
                stderr="",
            ),
        )
    with pytest.raises(rerender.RerenderError, match=message):
        rerender.resolve_gpu_binding(gpu)

def test_replay_worker_environment_splits_cuda_server_from_physical_egl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-parent")
    binding = rerender.GpuBinding(
        allocation_slot=0,
        cuda_token="GPU-parent",
        egl_physical_index=5,
    )

    environment = rerender.replay_worker_environment(
        binding,
        recorded={
            "CUDA_VISIBLE_DEVICES": "stale-recorded-device",
            "MUJOCO_EGL_DEVICE_ID": "7",
            "REAL2SIM_DEFERRED_MUJOCO_EGL_DEVICE_ID": "6",
            "MUJOCO_GL": "glx",
            "PYOPENGL_PLATFORM": "glx",
            "RECORDED_SENTINEL": "preserved",
        },
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["MUJOCO_EGL_DEVICE_ID"] == "5"
    assert environment["MUJOCO_GL"] == "egl"
    assert environment["PYOPENGL_PLATFORM"] == "egl"
    assert environment["OPENCV_FOR_THREADS_NUM"] == "1"
    assert "REAL2SIM_DEFERRED_MUJOCO_EGL_DEVICE_ID" not in environment
    assert environment["RECORDED_SENTINEL"] == "preserved"

class _FakeGsProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.waited = False

    def poll(self):
        return None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        raise AssertionError("cooperative fake process must not be killed")

    def wait(self, timeout=None):
        self.waited = True
        return 0

class _FakeHealthResponse:
    def __init__(self, payload: dict) -> None:
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, _size: int) -> bytes:
        return self.body

def test_start_gs_server_uses_authenticated_health_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ply = tmp_path / "scene.ply"
    ply.write_bytes(b"ply")
    token = "secret-token"
    digest = "a" * 64
    process = _FakeGsProcess()
    requests = []
    monkeypatch.setattr(rerender.subprocess, "Popen", lambda *_a, **_k: process)

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return _FakeHealthResponse(
            {
                "status": "ok",
                "ready_token": token,
                "ply_path": str(ply.resolve()),
                "ply_sha256": digest,
            }
        )

    monkeypatch.setattr(gs_launch, "_open_loopback", fake_urlopen)

    observed_process, identity = rerender._start_gs_server(
        "/venv/bin/python",
        ply,
        8123,
        token,
        tmp_path / "server.log",
        0,
    )

    assert observed_process is process
    assert identity["ply_sha256"] == digest
    request, timeout = requests[0]
    headers = {key.lower(): value for key, value in request.header_items()}
    assert headers["x-real2sim-gs-token"] == token
    assert timeout == 5.0
    assert process.terminated is False

def test_start_gs_server_retries_transient_bad_status_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ply = tmp_path / "scene.ply"
    ply.write_bytes(b"ply")
    token = "secret-token"
    process = _FakeGsProcess()
    calls = 0
    monkeypatch.setattr(rerender.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(rerender.time, "sleep", lambda _seconds: None)

    def fake_urlopen(_request, timeout):
        assert timeout == 5.0
        nonlocal calls
        calls += 1
        if calls == 1:
            raise rerender.http.client.BadStatusLine("GET /health HTTP/1.1")
        return _FakeHealthResponse(
            {
                "status": "ok",
                "ready_token": token,
                "ply_path": str(ply.resolve()),
                "ply_sha256": "a" * 64,
            }
        )

    monkeypatch.setattr(gs_launch, "_open_loopback", fake_urlopen)

    observed_process, _identity = rerender._start_gs_server(
        "/venv/bin/python",
        ply,
        8123,
        token,
        tmp_path / "server.log",
        0,
    )

    assert observed_process is process
    assert calls == 2
    assert process.terminated is False

def test_start_gs_server_stops_process_on_health_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ply = tmp_path / "scene.ply"
    ply.write_bytes(b"ply")
    process = _FakeGsProcess()
    monkeypatch.setattr(rerender.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(
        gs_launch,
        "_open_loopback",
        lambda *_a, **_k: _FakeHealthResponse(
            {
                "status": "ok",
                "ply_path": str(ply.resolve()),
                "ply_sha256": "a" * 64,
            }
        ),
    )

    with pytest.raises(rerender.RerenderError, match="token mismatch"):
        rerender._start_gs_server(
            "/venv/bin/python",
            ply,
            8123,
            "secret-token",
            tmp_path / "server.log",
            0,
        )

    assert process.terminated is True
    assert process.waited is True

@pytest.mark.parametrize("already_exited", (False, True))
def test_stop_process_escalates_from_term_to_kill(already_exited: bool) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.wait_timeouts: list[float | None] = []

        def poll(self):
            return 0 if already_exited else None

        def terminate(self) -> None:
            if already_exited:
                raise AssertionError("must not terminate an exited process")
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            if timeout is not None and not already_exited:
                raise subprocess.TimeoutExpired("gs", timeout)
            return 0 if already_exited else -9

    process = FakeProcess()

    rerender._stop_process(process, timeout=0.01)

    assert process.terminated is not already_exited
    assert process.killed is not already_exited
    assert process.wait_timeouts == ([None] if already_exited else [0.01, None])

@pytest.mark.parametrize(
    ("repeat_count", "template_name"),
    [
        (10, "rollout_grid_clear_headers_4x3.json"),
        (30, "rollout_grid_clear_headers_6x6.json"),
        (100, "rollout_grid_clear_headers_10x12.json"),
    ],
)
def test_build_scene_grids_selects_template_and_single_scene_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repeat_count: int,
    template_name: str,
) -> None:
    run_root = tmp_path / "batch"
    runs = [
        run_root / "scene_a" / f"run_{index:02d}"
        for index in range(1, repeat_count + 1)
    ]
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(rerender.subprocess, "run", fake_run)

    failures = rerender.build_scene_grids(
        run_root, runs, {"scene_a": 300}
    )

    assert failures == []
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[command.index("--scene") + 1] == "scene_a"
    assert command[command.index("--title") + 1] == "scene_a"
    assert command[command.index("--runs") + 1] == ",".join(
        f"run_{index:02d}" for index in range(1, repeat_count + 1)
    )
    assert command[command.index("--output") + 1] == str(
        run_root / f"rollout_grid_{repeat_count}.mp4"
    )
    assert Path(command[command.index("--template") + 1]).name == template_name
    assert command[command.index("--expected-frames") + 1] == "300"
    assert kwargs["cwd"] == rerender.REPO_ROOT
    assert kwargs["stdout"].name == str(
        run_root / f"rollout_grid_{repeat_count}_stdout.json"
    )
    assert kwargs["stderr"] is subprocess.STDOUT

def test_build_scene_grids_skips_a_scene_with_per_rollout_lengths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A chained recorded scene publishes its videos and no composite grid."""

    run_root = tmp_path / "batch"
    runs = [run_root / "scene_a" / f"run_{index:02d}" for index in range(1, 11)]

    def fake_run(command, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("the grid tool must not run for a ragged scene")

    monkeypatch.setattr(rerender.subprocess, "run", fake_run)

    assert (
        rerender.build_scene_grids(
            run_root, runs, {}, ragged_scenes={"scene_a"}
        )
        == []
    )
    assert "per-rollout lengths" in capsys.readouterr().out


def test_build_scene_grids_puts_multi_scene_outputs_under_scene_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "batch"
    runs = [
        run_root / scene / f"run_{index:02d}"
        for scene in ("scene_a", "scene_b")
        for index in range(1, 11)
    ]
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(rerender.subprocess, "run", fake_run)

    assert rerender.build_scene_grids(
        run_root, runs, {"scene_a": 300, "scene_b": 300}
    ) == []
    outputs = {
        command[command.index("--scene") + 1]: Path(
            command[command.index("--output") + 1]
        )
        for command in commands
    }
    assert outputs == {
        "scene_a": run_root / "scene_a" / "rollout_grid_10.mp4",
        "scene_b": run_root / "scene_b" / "rollout_grid_10.mp4",
    }
    for command in commands:
        assert command[command.index("--runs") + 1] == ",".join(
            f"run_{index:02d}" for index in range(1, 11)
        )

@pytest.mark.parametrize("explicit_template", (False, True))
def test_run_rerender_forwards_explicit_grid_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit_template: bool,
) -> None:
    _mock_gpu_inventory(monkeypatch)
    calls = []

    def fake_build(
        run_root, runs, expected_frames, grid_template=None, ragged_scenes=()
    ):
        calls.append((run_root, runs, expected_frames, grid_template))
        return []

    monkeypatch.setattr(rerender_run, "build_scene_grids", fake_build)
    template = tmp_path / "custom_grid.json" if explicit_template else None
    if template is not None:
        template.write_text("{}", encoding="utf-8")
    arguments = rerender.argparse.Namespace(
        run_root=tmp_path,
        gpu=0,
        wrist_insets="off",
        skip_grid=False,
        grid_template=template,
    )

    assert rerender._run_rerender(
        arguments,
        sources=[],
        partial_selection=True,
    ) == 0
    assert calls == [(tmp_path.resolve(), [], {}, template)]

def test_build_scene_grids_skips_unsupported_run_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_root = tmp_path / "batch"
    runs = [run_root / "scene_a" / f"run_{index:02d}" for index in range(1, 12)]

    def unexpected_run(*args, **kwargs):
        raise AssertionError("unsupported grids must not invoke the grid tool")

    monkeypatch.setattr(rerender.subprocess, "run", unexpected_run)

    assert rerender.build_scene_grids(
        run_root, runs, {"scene_a": 300}
    ) == []
    output = capsys.readouterr().out
    assert "grid skipped for scene_a" in output
    assert "no checked-in template for 11 runs" in output

@pytest.mark.parametrize("template_exists", (True, False))
def test_build_scene_grids_explicit_template_bypasses_count_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    template_exists: bool,
) -> None:
    run_root = tmp_path / "batch"
    run_count = 7 if template_exists else 10
    runs = [
        run_root / "scene_a" / f"run_{index:02d}"
        for index in range(1, run_count + 1)
    ]
    template = tmp_path / ("custom_grid.json" if template_exists else "missing_grid.json")
    if template_exists:
        template.write_text("{}", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(rerender.subprocess, "run", fake_run)

    if not template_exists:
        with pytest.raises(rerender.RerenderError, match="does not exist"):
            rerender.build_scene_grids(
                run_root,
                runs,
                {"scene_a": 300},
                grid_template=template,
            )
        assert calls == []
        return

    assert rerender.build_scene_grids(run_root, runs, {"scene_a": 300}, grid_template=template) == []
    assert len(calls) == 1
    command = calls[0]
    assert command[command.index("--template") + 1] == str(template.resolve())
    assert command[command.index("--output") + 1] == str(
        run_root / f"rollout_grid_{run_count}.mp4"
    )
