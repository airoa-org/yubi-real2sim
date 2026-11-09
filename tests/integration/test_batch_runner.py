# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Integration contracts for scripts/run_policy_batch.sh."""

import json
import tempfile
import unittest
from unittest import mock

import pytest

from real2sim.gpu_binding import (
    GPU_BINDINGS_ENV,
    contract_json as gpu_contract_json,
    resolve_gpu_contract,
)
from real2sim.runner.argv import SIM_WORKER_MODULE, worker_command_head_length
from robosuite_env.rerender.replay_argv import parse_run_command
from tests._helpers.worker_ast import *  # noqa: F403

RUNNER_SCRIPT_PATH = RUNNER_PATH  # noqa: F405

def test_init_state_source_hash_covers_structured_runtime_deterministically() -> None:
    function = _shell_function(RUNNER_SOURCE, "prepare_deterministic_init_state")
    start = function.index('source_sha="$(')
    end = function.index("\n    )\"", start)
    source_hash = function[start:end]

    for directory in STRUCTURED_RUNTIME_DIRECTORIES:
        assert (PACKAGE_ROOT / directory).is_dir()
        assert f'"${{ROOT}}/robosuite_env/{directory}"' in source_hash

    # The entry point and shared resource roots live outside the recursive
    # packages, so dirty changes to either must invalidate deterministic state.
    assert '"${ROOT}/robosuite_env/sim.py"' in source_hash
    assert '"${ROOT}/robosuite_env/paths.py"' in source_hash
    assert "-print0" in source_hash
    assert "LC_ALL=C sort -z" in source_hash
    assert "xargs -0 -r sha256sum" in source_hash
    assert '"${ROOT}/robosuite_env/assets/manifest.json"' in source_hash
    assert (
        '"${ROOT}/robosuite_env/controllers/default_fr3_joint_ik.json"'
        in source_hash
    )

def test_runner_env_tripwire() -> None:
    assert 'GS_NUM_THREADS="${GS_NUM_THREADS:-1}"' in RUNNER_SOURCE
    assert "GS_NUM_THREADS must be a positive integer" in RUNNER_SOURCE
    strict_case = RUNNER_SOURCE.index('case "${REAL2SIM_STRICT_LAUNCH}" in')
    strict_start = RUNNER_SOURCE.index(
        'if [ "${REAL2SIM_STRICT_LAUNCH}" = "1" ]; then', strict_case
    )
    strict_end = RUNNER_SOURCE.index("\nfi\n", strict_start)
    assert "GS_NUM_THREADS=1" in RUNNER_SOURCE[strict_start:strict_end]
    authority_start = RUNNER_SOURCE.index("launcher_owned_environment = {")
    authority_end = RUNNER_SOURCE.index("\n}", authority_start)
    assert '"GS_NUM_THREADS"' not in RUNNER_SOURCE[authority_start:authority_end]

    function = _shell_function(RUNNER_SOURCE, "start_gs_servers")
    server_start = function.index('exec "${PY}" -u scene/gs_background_server.py')
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        export = f'export {variable}="${{GS_NUM_THREADS}}"'
        assert export in function
        assert function.index(export) < server_start

    assert "cpu_threads=${GS_NUM_THREADS}" in function
    assert "gs_num_threads=${GS_NUM_THREADS}" in RUNNER_SOURCE
    assert 'exec "${SERVER_CMD[@]}"' in _shell_function(
        RUNNER_SOURCE, "start_policy_server"
    )
    # The simulation thread caps have one definition; the server, the prep and
    # worker subshells and the recorded run command all use it.
    for function_name, launch in (
        ("start_policy_server", 'exec "${SERVER_CMD[@]}"'),
        ("prepare_deterministic_init_state", '"${PY}" "${prep_cmd[@]}"\n'),
        ("run_worker", '"${PY}" "${cmd[@]}"\n'),
    ):
        function = _shell_function(RUNNER_SOURCE, function_name)
        assert function.index("export_sim_thread_caps") < function.rindex(launch)
        assert 'export OPENBLAS_NUM_THREADS="${SIM_NUM_THREADS}"' not in function
    command_writer = _shell_function(RUNNER_SOURCE, "write_run_command")
    assert '"${SIM_THREAD_VARIABLES[@]}"' in command_writer
    # Helpers launched for a worker outside those subshells use the same caps.
    assert (
        "sim_helper_python -m robosuite_env.rollout.state.init_identity"
        in _shell_function(RUNNER_SOURCE, "prepare_deterministic_init_state")
    )
    assert "sim_helper_python -m robosuite_env.contracts.worker write" in (
        _shell_function(RUNNER_SOURCE, "write_official_worker_contract")
    )

    cleanup = _shell_function(RUNNER_SOURCE, "cleanup")
    assert cleanup.index('terminate_pids "workers"') < cleanup.index(
        "stop_generation_services"
    )
    service_cleanup = _shell_function(RUNNER_SOURCE, "stop_generation_services")
    assert service_cleanup.index('terminate_pids "policy server"') < (
        service_cleanup.index('terminate_pids "GS servers"')
    )
    assert "trap - EXIT INT TERM" in cleanup
    for trap in ("trap cleanup EXIT", "trap 'exit 130' INT", "trap 'exit 143' TERM"):
        assert trap in RUNNER_SOURCE
    wave_wait = _shell_function(RUNNER_SOURCE, "wait_for_worker_pids")
    assert 'terminate_pids "remaining workers after wave failure"' in wave_wait

    for contract in (
        'export CUDA_VISIBLE_DEVICES=""',
        'export MUJOCO_EGL_DEVICE_ID="${prep_egl_index}"',
        '--gs-server-token "${gs_token}"',
        'if [ -f "${OBJECTS}" ]; then',
        'sha256sum "${OBJECTS}"',
        '--template "${GRID_TEMPLATE}"',
        "required template grid render failed",
    ):
        assert contract in RUNNER_SOURCE


SIM_THREAD_CAP_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
)


def test_worker_contract_writer_gets_the_worker_thread_caps(tmp_path) -> None:
    """The per-worker contract writer must not start a BLAS thread per core.

    Every worker launch first runs the contract writer beside the policy and
    GS servers. Without the worker's caps each writer would start one OpenBLAS
    thread per host core, and a wave could exhaust a container's task limit.
    """

    record = tmp_path / "record.txt"
    stub = tmp_path / "python"
    names = " ".join(SIM_THREAD_CAP_NAMES)
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" > "${RECORD}"\n'
        f"for name in {names}; do\n"
        '  printf "%s=%s\\n" "${name}" "${!name-unset}" >> "${RECORD}"\n'
        "done\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    script = f"""
set -euo pipefail
{_sim_thread_caps_definitions()}
{_shell_function(RUNNER_SOURCE, "write_official_worker_contract")}
write_official_worker_contract contract.json --robot FR3
for name in {names}; do
  echo "parent ${{name}}=${{!name-unset}}"
done
"""
    environment = {
        key: value
        for key, value in os.environ.items()  # noqa: F405
        if key not in SIM_THREAD_CAP_NAMES
    }
    environment.update(
        {
            "PY": os.fspath(stub),  # noqa: F405
            "RECORD": os.fspath(record),  # noqa: F405
            "REAL2SIM_STRICT_LAUNCH": "1",
            "REAL2SIM_OFFICIAL_RUN_MANIFEST": "manifest.json",
            "SIM_NUM_THREADS": "3",
        }
    )
    completed = subprocess.run(  # noqa: F405
        ["bash", "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    recorded = record.read_text(encoding="utf-8").splitlines()
    assert recorded[0] == (
        "-m robosuite_env.contracts.worker write --contract contract.json "
        "--run-manifest manifest.json -- --robot FR3"
    )
    assert recorded[1:] == [f"{name}=3" for name in SIM_THREAD_CAP_NAMES]
    # The caps apply to the helper only; run-level tools keep their environment.
    assert completed.stdout.splitlines() == [
        f"parent {name}=unset" for name in SIM_THREAD_CAP_NAMES
    ]


def test_runner_shutdown_escalates_a_term_ignoring_child(tmp_path) -> None:
    """The cleanup helper must kill a wedged wrapper and its child."""

    collect = _shell_function(RUNNER_SOURCE, "collect_process_tree")
    running = _shell_function(RUNNER_SOURCE, "process_is_running")
    terminate = _shell_function(RUNNER_SOURCE, "terminate_pids")
    ready_file = tmp_path / "ready"
    script = f"""
set -euo pipefail
TERMINATION_GRACE_SEC=1
{collect}
{running}
{terminate}
"${{TEST_PYTHON}}" -c "${{TREE_PARENT_CODE}}" &
root_pid="$!"
for _ in {{1..100}}; do
  [ -f "${{READY_FILE}}" ] && break
  sleep 0.01
done
[ -f "${{READY_FILE}}" ]
child_pid="$(cat "${{READY_FILE}}")"
terminate_pids "test process tree" "${{root_pid}}"
for _ in {{1..100}}; do
  ! kill -0 "${{child_pid}}" 2>/dev/null && break
  sleep 0.01
done
! kill -0 "${{root_pid}}" 2>/dev/null
! kill -0 "${{child_pid}}" 2>/dev/null
"""
    child_code = (
        "import os, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(os.environ['READY_FILE'], 'w').write(str(os.getpid())); "
        "time.sleep(60)"
    )
    parent_code = (
        "import os, subprocess, sys; "
        "subprocess.run([sys.executable, '-c', os.environ['TREE_CHILD_CODE']])"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "READY_FILE": os.fspath(ready_file),
            "TEST_PYTHON": sys.executable,
            "TREE_CHILD_CODE": child_code,
            "TREE_PARENT_CODE": parent_code,
        }
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_wave_wait_fails_a_worker_that_bash_already_reaped() -> None:
    """A worker killed before ``wait -n`` must fail the wave, not trip ``set -u``."""

    wave_wait = _shell_function(RUNNER_SOURCE, "wait_for_worker_pids")
    script = f"""
set -euo pipefail
terminate_pids() {{ :; }}
write_outputs_csv() {{ :; }}
wait() {{ echo "trace: wait $*" >&2; builtin wait "$@"; }}
{wave_wait}
overall_rc=0
(kill -KILL "${{BASHPID}}") &
PIDS=("$!")
while kill -0 "${{PIDS[0]}}" 2>/dev/null; do sleep 0.01; done
jobs >/dev/null  # drops the reaped worker from the job table, so wait -n cannot see it
worker="${{PIDS[0]}}"
status=0
wait_for_worker_pids "worker failed" || status=$?
echo "worker=${{worker}} status=${{status}} overall_rc=${{overall_rc}} remaining=${{#PIDS[@]}}"
"""
    completed = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    worker = completed.stdout.split("worker=")[1].split()[0]
    assert "status=1 overall_rc=1 remaining=0" in completed.stdout
    assert "worker failed" in completed.stderr
    trace = [line for line in completed.stderr.splitlines() if line.startswith("trace: ")]
    assert trace == [f"trace: wait -n -p done_pid {worker}", f"trace: wait {worker}"]


INLINE_FINALIZERS = (
    "python -m robosuite_env.tools.rewrite_saved_policy_obs_video",
    "python -m robosuite_env.tools.make_rollout_grid_video",
)


@pytest.mark.parametrize(
    ("posthoc", "failing_check", "finalizers"),
    (
        pytest.param("0", "", INLINE_FINALIZERS, id="inline"),
        pytest.param(
            "1", "", ("python -m robosuite_env.tools.rerender_third_person",), id="posthoc"
        ),
        pytest.param("0", "wave-1-10-after", (), id="inline-post-wave-check-fails"),
    ),
)
def test_runner_stops_the_servers_before_the_finalization_tools(
    tmp_path, posthoc, failing_check, finalizers
) -> None:
    """The finalization tools must not run beside the policy and GS servers.

    Republishing videos and rendering the grid while both servers still hold
    their threads can push the grid encode over the container's task limit.
    Run the runner's real main flow, from the
    rollout waves to its exit, under the runner's ``set -euo pipefail`` and its
    real EXIT trap, with stub services, a stub source check that can fail at
    one stage, and a stub Python that logs each tool it is asked to run.
    """

    assert "\nset -euo pipefail\n" in RUNNER_SOURCE
    assert "\ntrap cleanup EXIT\n" in RUNNER_SOURCE
    log = tmp_path / "events.log"
    stub_python = tmp_path / "python"
    stub_python.write_text(
        '#!/usr/bin/env bash\nprintf "python %s %s\\n" "$1" "$2" >> "${EVENTS}"\n',
        encoding="utf-8",
    )
    stub_python.chmod(0o755)
    main_flow = RUNNER_SOURCE[RUNNER_SOURCE.index("\noverall_rc=0\n") :]
    script = "\n".join(
        (
            "set -euo pipefail",
            _shell_function(RUNNER_SOURCE, "cleanup"),
            _shell_function(RUNNER_SOURCE, "stop_generation_services"),
            # Log every call, so the trap's second call is visible in the log.
            'eval "real_$(declare -f stop_generation_services)"',
            "stop_generation_services() {",
            '  echo "stop_generation_services" >> "${EVENTS}"',
            "  real_stop_generation_services",
            "}",
            _shell_function(RUNNER_SOURCE, "publish_saved_policy_obs_videos"),
            _shell_function(RUNNER_SOURCE, "render_rollout_grid_template"),
            'terminate_pids() { echo "stop $1" >> "${EVENTS}"; }',
            "verify_runtime_source_manifest() {",
            '  echo "check $1" >> "${EVENTS}"',
            '  [ "$1" != "${FAILING_CHECK}" ]',
            "}",
            'start_gs_servers() { GS_PIDS=(101); echo "start GS servers" >> "${EVENTS}"; }',
            "prepare_deterministic_init_state() { :; }",
            'start_policy_server() { SERVER_PID=102; echo "start policy server" >> "${EVENTS}"; }',
            'run_worker() { echo "worker $2" >> "${EVENTS}"; }',
            # Like the real wait: every worker is reaped and leaves PIDS.
            'wait_for_worker_pids() { local pid; for pid in "${PIDS[@]}"; do wait "${pid}"; done; PIDS=(); }',
            "write_outputs_csv() { :; }",
            f'PY="{stub_python}"',
            f'RUN_ROOT="{tmp_path}"',
            f'OUTPUTS_CSV="{tmp_path / "outputs.csv"}"',
            'GRID_TEMPLATE="template.json"',
            "TAGS=(scene)",
            "SCENE_COUNT=1",
            "SIM_GPU_IDS=(0)",
            "BOX_PACK_LAYOUTS=()",
            "REPEATS=10 REPEAT_START=1 REPEAT_STOP=10 PARALLEL_REPEATS=10 FRAMES=300",
            "REAL2SIM_AGGREGATE_ONLY=0 DRY_RUN=0 RUN_ANALYSIS=0 PARTIAL_REPEAT_RANGE=0",
            f"RENDER_ENABLED=1 POSTHOC_RENDER_REQUIRED={posthoc}",
            'SERVER_PID="" GS_PIDS=() PIDS=()',
            "trap cleanup EXIT",
            main_flow,
        )
    )
    completed = subprocess.run(  # noqa: F405
        ["bash", "-c", script],
        env={**os.environ, "EVENTS": os.fspath(log), "FAILING_CHECK": failing_check},  # noqa: F405
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == (1 if failing_check else 0), completed.stderr
    events = log.read_text(encoding="utf-8").splitlines()
    workers = [index for index, event in enumerate(events) if event.startswith("worker ")]
    assert sorted(events[index] for index in workers) == sorted(
        f"worker {repeat}" for repeat in range(1, 11)
    )
    stops = [index for index, event in enumerate(events) if event.startswith("stop ")]
    assert [events[index] for index in stops] == ["stop policy server", "stop GS servers"]
    tools = [index for index, event in enumerate(events) if event.startswith("python ")]
    assert tuple(events[index] for index in tools) == finalizers
    # The servers stop once, after the last worker and before any finalization tool.
    assert max(workers) < min(stops)
    assert all(max(stops) < index for index in tools)
    # The EXIT trap stops them a second time, which does nothing.
    assert events.count("stop_generation_services") == 2
    assert events[-1] == "stop_generation_services"
    assert max(stops) < len(events) - 1


class RunPolicyBatchIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        environment = dict(os.environ)  # noqa: F405
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        patcher = mock.patch.dict(os.environ, environment, clear=True)  # noqa: F405
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_shell_runner_dry_run_expands_parallel_workers_and_gpu_round_robin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = root / "test-scene"
            checkpoint = root / "checkpoint"
            run_root = root / "run"
            scene.mkdir()
            checkpoint.mkdir()
            (scene / "point_cloud.ply").write_text("ply\n", encoding="utf-8")
            (scene / "background.json").write_text(
                json.dumps(
                    {
                        "name": "test_scene",
                        "ply_path": "point_cloud.ply",
                        "mat_w2gs": [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                    }
                ),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "PY": sys.executable,
                    "CKPT": str(checkpoint),
                    "CONFIG": "pi05_yubi_pickplace",
                    "GS_SCENE_DIRS": str(scene),
                    "RUN_ROOT": str(run_root),
                    "ROBOT": "FR3",
                    "REPEATS": "4",
                    "PARALLEL_REPEATS": "2",
                    "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b",
                    "GS_CUDA_DEVICES": "0,1",
                    "POLICY_CUDA_DEVICE": "1",
                    "SIM_CUDA_DEVICES": "1,0",
                    "RENDER_LOCK_ROOT": str(root / "render-locks"),
                    "DETERMINISTIC_SIM": "0",
                    "GRASP_MODE": "physics",
                    "EXTRA_SIM_ARGS_JSON": "[]",
                }
            )
            environment[GPU_BINDINGS_ENV] = gpu_contract_json(
                resolve_gpu_contract(
                    environment,
                    nvidia={
                        "gpus": [
                            {"index": "5", "uuid": "GPU-a"},
                            {"index": "2", "uuid": "GPU-b"},
                        ]
                    },
                )
            )
            completed = subprocess.run(
                [str(RUNNER_SCRIPT_PATH), "--dry-run"],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = completed.stdout
            self.assertIn("policy_batch_size=2", output)
            self.assertIn("policy_cuda_device=1", output)
            self.assertIn("sim_cuda_devices=1,0", output)
            self.assertIn("test_scene/gpu_0", output)
            self.assertIn("test_scene/gpu_1", output)
            self.assertIn("selector=0 cuda=GPU-a egl=5", output)
            self.assertIn("selector=1 cuda=GPU-b egl=2", output)
            self.assertIn("--robot FR3", output)
            self.assertIn("wave repeats=1-2/4", output)
            self.assertIn("wave repeats=3-4/4", output)
            worker_rows = (run_root / "gs_workers.tsv").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(
                [row.split("\t")[5:7] for row in worker_rows],
                [["0", "1"], ["1", "0"]],
            )
            for repeat, (selector, cuda_token, egl_index) in enumerate(
                (
                    ("1", "GPU-b", "2"),
                    ("0", "GPU-a", "5"),
                    ("1", "GPU-b", "2"),
                    ("0", "GPU-a", "5"),
                ),
                start=1,
            ):
                command = (
                    run_root
                    / "test_scene"
                    / f"run_{repeat:02d}"
                    / "RUN_COMMAND.md"
                ).read_text(encoding="utf-8")
                self.assertIn(f"selector `{selector}`", command)
                self.assertIn(f"CUDA token `{cuda_token}`", command)
                self.assertIn(f"MUJOCO_EGL_DEVICE_ID={egl_index}", command)
                self.assertIn("CUDA_VISIBLE_DEVICES=''", command)
                self.assertIn(
                    "XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 "
                    "MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "
                    "OPENCV_FOR_THREADS_NUM=4 ",
                    command,
                )
                render_lock = (
                    root
                    / "render-locks"
                    / f"mujoco_render_faces_gpu_{egl_index}.lock"
                )
                self.assertIn(f"--render-lock {render_lock}", command)
                self.assertNotIn(
                    f"--render-lock {run_root}/mujoco_render_faces.lock",
                    command,
                )
                self.assertEqual(command.count("--grasp-mode physics"), 1)
                self.assertNotIn("--grasp-mode rule", command)
                # The head the shell runner records (its SIM_WORKER) must be
                # the module head the rerender parser and the official-source
                # check accept; the two are defined in bash and in Python.
                _, recorded = parse_run_command(
                    run_root / "test_scene" / f"run_{repeat:02d}"
                )
                self.assertEqual(recorded[0], sys.executable)
                self.assertEqual(recorded[1:3], ["-m", SIM_WORKER_MODULE])
                self.assertEqual(worker_command_head_length(recorded), 3)

            range_root = root / "range-run"
            environment.update(
                {
                    "RUN_ROOT": str(range_root),
                    "REPEAT_START": "3",
                    "REPEAT_STOP": "4",
                }
            )
            completed = subprocess.run(
                [str(RUNNER_SCRIPT_PATH), "--dry-run"],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "repeat_range=3-4/4 (absolute run numbering)", completed.stdout
            )
            self.assertIn("wave repeats=3-4/4", completed.stdout)
            self.assertNotIn("wave repeats=1-2/4", completed.stdout)
            self.assertIn(
                "aggregate summary, post-hoc rendering, grid, poster, and grid "
                "manifest are skipped; whole-run finalization is deferred",
                completed.stdout,
            )
            self.assertFalse((range_root / "test_scene" / "run_01").exists())
            self.assertFalse((range_root / "test_scene" / "run_02").exists())
            for repeat in (3, 4):
                run_dir = range_root / "test_scene" / f"run_{repeat:02d}"
                command = (run_dir / "RUN_COMMAND.md").read_text(encoding="utf-8")
                self.assertIn(
                    f"--policy-batch-client-id test_scene_run_{repeat:02d}",
                    command,
                )
            outputs = range_root / ".ranges" / "3-4" / "outputs.csv"
            self.assertTrue(outputs.is_file())
            output_rows = outputs.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(output_rows), 3)
            self.assertIn(",run_03,", output_rows[1])
            self.assertIn(",run_04,", output_rows[2])
            self.assertFalse((range_root / "outputs.csv").exists())

    def test_shell_runner_rejects_gpu_contract_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = root / "test-scene"
            checkpoint = root / "checkpoint"
            run_root = root / "run"
            render_lock_root = root / "render-locks"
            scene.mkdir()
            checkpoint.mkdir()
            environment = os.environ.copy()
            environment.update(
                {
                    "PY": sys.executable,
                    "CKPT": str(checkpoint),
                    "CONFIG": "pi05_yubi_pickplace",
                    "GS_SCENE_DIRS": str(scene),
                    "RUN_ROOT": str(run_root),
                    "RENDER_LOCK_ROOT": str(render_lock_root),
                    "CUDA_VISIBLE_DEVICES": "GPU-a",
                    "GS_CUDA_DEVICES": "0",
                    "POLICY_CUDA_DEVICE": "0",
                    "SIM_CUDA_DEVICES": "0",
                    GPU_BINDINGS_ENV: "{not-json",
                }
            )

            completed = subprocess.run(
                [str(RUNNER_SCRIPT_PATH), "--dry-run"],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("is not valid JSON", completed.stderr)
            self.assertFalse(run_root.exists())
            self.assertFalse(render_lock_root.exists())

    def test_official_batch_refuses_direct_cli_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            environment = os.environ.copy()
            environment.update(
                {
                    "PY": sys.executable,
                    "CKPT": str(checkpoint),
                    "CONFIG": "pi05_yubi_pickplace",
                    "GS_SCENE_DIRS": str(root / "unused-scene"),
                    "RUN_ROOT": str(root / "run"),
                    "RENDER_LOCK_ROOT": str(root / "render-locks"),
                    "REAL2SIM_STRICT_LAUNCH": "1",
                    "REAL2SIM_OFFICIAL_RUN_MANIFEST": str(
                        root / "run" / "run_manifest.json"
                    ),
                }
            )

            completed = subprocess.run(
                [str(RUNNER_SCRIPT_PATH), "--dry-run"],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "official schema-v2 runs do not accept direct batch CLI overrides",
            completed.stderr,
        )

    def test_batch_default_uses_b10_only_for_ten_rollouts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = root / "test-scene"
            checkpoint = root / "checkpoint"
            run_root = root / "run"
            lock_root = root / "node-render-locks"
            scene.mkdir()
            checkpoint.mkdir()
            (scene / "point_cloud.ply").write_text("ply\n", encoding="utf-8")
            (scene / "background.json").write_text(
                json.dumps(
                    {
                        "name": "test_scene",
                        "ply_path": "point_cloud.ply",
                        "mat_w2gs": [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                    }
                ),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment.pop("PARALLEL_REPEATS", None)
            environment.update(
                {
                    "PY": sys.executable,
                    "CKPT": str(checkpoint),
                    "CONFIG": "pi05_yubi_pickplace",
                    "GS_SCENE_DIRS": str(scene),
                    "RUN_ROOT": str(run_root),
                    "REPEATS": "10",
                    "GS_CUDA_DEVICES": "4",
                    "POLICY_CUDA_DEVICE": "4",
                    "SIM_CUDA_DEVICES": "4",
                    "RENDER_LOCK_ROOT": str(lock_root),
                    "DETERMINISTIC_SIM": "0",
                    "EXTRA_SIM_ARGS_JSON": "[]",
                }
            )
            environment[GPU_BINDINGS_ENV] = gpu_contract_json(
                resolve_gpu_contract(
                    environment,
                    nvidia={
                        "gpus": [{"index": "4", "uuid": "GPU-fixture-four"}]
                    },
                )
            )
            completed = subprocess.run(
                [str(RUNNER_SCRIPT_PATH), "--dry-run"],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = completed.stdout
            self.assertIn("policy_batch_size=10", output)
            self.assertIn("B10 forces --no-render", output)
            self.assertIn("render_mode=posthoc generation=no-render", output)
            self.assertIn("-m robosuite_env.tools.rerender_third_person", output)
            self.assertIn("wave repeats=1-10/10", output)
            self.assertEqual(output.count("[real2sim-batch] start GS "), 1)

            server_rows = (run_root / "gs_servers.tsv").read_text(
                encoding="utf-8"
            ).splitlines()
            worker_rows = (run_root / "gs_workers.tsv").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(server_rows), 1)
            self.assertEqual(len(worker_rows), 10)
            server_port = server_rows[0].split("\t")[3]
            self.assertEqual(
                {row.split("\t")[4] for row in worker_rows}, {server_port}
            )
            self.assertEqual({row.split("\t")[5] for row in worker_rows}, {"4"})
            self.assertEqual({row.split("\t")[6] for row in worker_rows}, {"4"})

            expected_lock = (
                lock_root.resolve() / "mujoco_render_faces_gpu_4.lock"
            )
            for repeat in range(1, 11):
                command = (
                    run_root
                    / "test_scene"
                    / f"run_{repeat:02d}"
                    / "RUN_COMMAND.md"
                ).read_text(encoding="utf-8")
                self.assertIn(f"--gs-port {server_port}", command)
                self.assertIn(f"--render-lock {expected_lock}", command)
                self.assertEqual(command.count("--no-render"), 1)
                self.assertEqual(command.count("--render-lock"), 1)

            fallback_run_root = root / "run-b1"
            environment.update(
                {
                    "RUN_ROOT": str(fallback_run_root),
                    "REPEATS": "6",
                }
            )
            completed = subprocess.run(
                [str(RUNNER_SCRIPT_PATH), "--dry-run"],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("policy_batch_size=1", completed.stdout)
            self.assertNotIn("render_mode=posthoc", completed.stdout)
            self.assertIn("wave repeats=1-1/6", completed.stdout)
            self.assertIn("wave repeats=6-6/6", completed.stdout)

            b11_run_root = root / "run-b11"
            environment.update(
                {
                    "RUN_ROOT": str(b11_run_root),
                    "REPEATS": "11",
                    "PARALLEL_REPEATS": "11",
                    "POLICY_BATCH_SIZE": "11",
                }
            )
            completed = subprocess.run(
                [str(RUNNER_SCRIPT_PATH), "--dry-run"],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("B11 forces --no-render", completed.stdout)
            self.assertIn("render_mode=posthoc", completed.stdout)
            for repeat in range(1, 12):
                command = (
                    b11_run_root
                    / "test_scene"
                    / f"run_{repeat:02d}"
                    / "RUN_COMMAND.md"
                ).read_text(encoding="utf-8")
                self.assertEqual(command.count("--no-render"), 1)

            dedupe_run_root = root / "run-b11-dedupe"
            environment.update(
                {
                    "RUN_ROOT": str(dedupe_run_root),
                    "DETERMINISTIC_SIM": "1",
                    "INIT_STATE_JSON": str(root / "dedupe-init-state.json"),
                    "EXTRA_SIM_ARGS_JSON": json.dumps(
                        ["--no-render", "--no-render"]
                    ),
                }
            )
            completed = subprocess.run(
                [str(RUNNER_SCRIPT_PATH), "--dry-run"],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            command = (
                dedupe_run_root
                / "test_scene"
                / "run_01"
                / "RUN_COMMAND.md"
            ).read_text(encoding="utf-8")
            self.assertEqual(command.count("--no-render"), 1)
            init_command = next(
                line
                for line in completed.stdout.splitlines()
                if "init-state command:" in line
            )
            self.assertEqual(init_command.count("--no-render"), 1)

    def test_batch_worker_wall_stop_fails_wave_before_next_barrier(self):
        worker_source = _shell_function(RUNNER_SOURCE, "run_worker")
        guard_start = worker_source.index("import json\n")
        guard_end = worker_source.index("\nPY\n", guard_start)
        guard_source = worker_source[guard_start:guard_end]
        with tempfile.TemporaryDirectory() as temporary:
            summary_path = Path(temporary) / "sim_summary.json"
            summary_path.write_text(
                json.dumps({"wall_stopped_early": True}),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [sys.executable, "-", str(summary_path)],
                input=guard_source,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 124, completed.stderr)
