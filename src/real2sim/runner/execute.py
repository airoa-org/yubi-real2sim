# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Execute a resolved rollout plan."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from real2sim._bootstrap import REPO_ROOT
from real2sim.config._types import ConfigError
from real2sim.gpu_binding import GpuBindingError, resolve_gpu_contract
from real2sim.launch_provenance import (
    persist_config_provenance,
    prepare_run_manifest,
    seal_runtime_source_manifest,
)
from real2sim.preflight import validate_launch_preflight
from real2sim.reporting.summary import summarize_run
from real2sim.runner.environment import (
    RUNTIME_SOURCE_MANIFEST_ENV,
    RUNTIME_SOURCE_MANIFEST_SHA256_ENV,
    RUNTIME_SOURCE_UNSEALED_ENV,
    build_all_shard_environments,
    build_runner_environment,
)
from real2sim.runtime_manifest import (
    RuntimeSourceDriftError,
    verify_manifest as verify_runtime_manifest,
)


def _srun_command(
    runner_script: str | Path,
    shard: Mapping[str, Any],
    *,
    aggregate: bool = False,
) -> list[str]:
    command = [
        "srun",
        f"--jobid={shard['slurm_job_id']}",
        "--overlap",
        # One runner process per shard. Without an explicit task count srun
        # derives it from the target allocation (free CPUs // cpus-per-task),
        # which silently starts N identical runners that fight over the same
        # GPU, policy-server port, and GS ports.
        "--nodes=1",
        "--ntasks=1",
        "--cpu-bind=none",
    ]
    cpus = None if aggregate else shard.get("cpus")
    if cpus is None:
        cpus_per_task = shard.get("cpus_per_task")
        if cpus_per_task is not None:
            command.append(f"--cpus-per-task={cpus_per_task}")
        command.append(str(runner_script))
        return command
    # --overlap lets steps of one allocation share resources, and Slurm then
    # hands every step the same leading --cpus-per-task CPUs instead of a
    # disjoint set. Ask for no count so the step inherits the whole allocation
    # cpuset, then pin the runner to the ids this shard owns. Children inherit
    # the affinity.
    command.extend(["taskset", "-c", str(cpus), str(runner_script)])
    return command


def _stop_shard_processes(processes: Sequence[subprocess.Popen[Any]]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
    deadline = time.monotonic() + 5.0
    for process in processes:
        if process.poll() is not None:
            continue
        timeout = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass
    for process in processes:
        try:
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _execute_shards(
    plan: Mapping[str, Any],
    *,
    runtime_manifest_path: Path | None,
    runtime_manifest_digest: str | None,
    runtime_source_unsealed: bool = False,
) -> int:
    shards = plan["resolved"]["runner"]["shards"]
    shard_environments, aggregate_environment = build_all_shard_environments(plan)
    if runtime_manifest_path is not None and runtime_manifest_digest is not None:
        for environment in [*shard_environments.values(), aggregate_environment]:
            environment[RUNTIME_SOURCE_MANIFEST_ENV] = str(runtime_manifest_path)
            environment[RUNTIME_SOURCE_MANIFEST_SHA256_ENV] = runtime_manifest_digest
    elif runtime_source_unsealed:
        # Every shard has to be told, or the ones without the flag re-derive a
        # manifest from the tree that already moved.
        for environment in [*shard_environments.values(), aggregate_environment]:
            environment[RUNTIME_SOURCE_UNSEALED_ENV] = "1"

    run_root = Path(plan["run_root"])
    active: dict[str, tuple[subprocess.Popen[Any], Any]] = {}
    try:
        for shard in shards:
            name = str(shard["name"])
            log_path = run_root / f"shard_{name}.log"
            try:
                log_stream = log_path.open("w", encoding="utf-8")
            except OSError as exc:
                print(
                    f"failed to open rollout shard log {log_path}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                _stop_shard_processes([item[0] for item in active.values()])
                return 127
            try:
                process = subprocess.Popen(
                    _srun_command(plan["runner_script"], shard),
                    cwd=REPO_ROOT,
                    env=shard_environments[name],
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:
                log_stream.close()
                print(
                    f"failed to start rollout shard {name}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                _stop_shard_processes([item[0] for item in active.values()])
                return 127
            active[name] = (process, log_stream)

        while active:
            completed_any = False
            for name, (process, log_stream) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                completed_any = True
                log_stream.close()
                del active[name]
                if returncode != 0:
                    print(
                        f"rollout shard {name} failed with exit code {returncode}; "
                        f"see {run_root / f'shard_{name}.log'}",
                        file=sys.stderr,
                    )
                    _stop_shard_processes([item[0] for item in active.values()])
                    return int(returncode) if returncode > 0 else 1
            if active and not completed_any:
                time.sleep(0.1)
    finally:
        # Also cover launcher interruption or an unexpected polling error: no
        # target-allocation step may outlive the coordinator.
        _stop_shard_processes([item[0] for item in active.values()])
        for _process, log_stream in active.values():
            if not log_stream.closed:
                log_stream.close()

    aggregate_shard = shards[0]
    aggregate_log = run_root / "aggregate.log"
    try:
        with aggregate_log.open("w", encoding="utf-8") as log_stream:
            completed = subprocess.run(
                _srun_command(
                    plan["runner_script"], aggregate_shard, aggregate=True
                ),
                cwd=REPO_ROOT,
                env=aggregate_environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
    except OSError as exc:
        print(
            "failed to start rollout shard aggregation: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 127
    if completed.returncode != 0:
        print(
            f"rollout shard aggregation failed with exit code {completed.returncode}; "
            f"see {aggregate_log}",
            file=sys.stderr,
        )
    return int(completed.returncode)


def _seal_unsharded_gpu_bindings(
    plan: Mapping[str, Any],
    preflight_report: dict[str, Any],
) -> None:
    """Persist a direct CUDA run's exact binding in launcher authority."""

    if preflight_report.get("gpu_bindings") is not None:
        return
    reproducibility = plan["resolved"]["reproducibility"]
    if reproducibility["accelerator_profile"]["kind"] != "cuda":
        return
    shards = plan["resolved"]["runner"].get("shards")
    if isinstance(shards, list) and shards:
        # Each target srun step owns a different CUDA visibility boundary and
        # seals it only after Slurm starts that step.
        return
    environment = build_runner_environment(plan)
    try:
        preflight_report["gpu_bindings"] = resolve_gpu_contract(environment)
    except GpuBindingError as exc:
        raise ConfigError(f"runtime GPU binding failed: {exc}") from exc


def execute(
    plan: Mapping[str, Any],
    *,
    runtime_source_baseline: Mapping[str, Any] | None = None,
) -> int:
    preflight_report = validate_launch_preflight(plan)
    _seal_unsharded_gpu_bindings(plan, preflight_report)
    try:
        run_manifest = prepare_run_manifest(plan, preflight_report)
    finally:
        # The content proof can contain one stat record per artifact path. It
        # is needed only while building the manifest and is never serialized
        # or retained for the rollout process.
        preflight_report.pop("_artifact_verification", None)
    persist_config_provenance(plan, run_manifest)
    runtime_manifest_path: Path | None = None
    runtime_manifest_digest: str | None = None
    runtime_source_unsealed = False
    if runtime_source_baseline is not None:
        try:
            runtime_manifest_path, runtime_manifest_digest = (
                seal_runtime_source_manifest(plan, runtime_source_baseline)
            )
        except RuntimeSourceDriftError as exc:
            # Only drift is survivable. A malformed baseline, an unreadable
            # config or a manifest that could not be written are sealing
            # failures, not a checkout change, and they still raise --
            # catching the base class here would start an unsealed run over a
            # broken launch and skip verification for the rest of it.
            #
            # The checkout changed between capture and seal. Run without a
            # sealed runtime manifest and say so.
            #
            # Dropping the two manifest variables is not enough on its own: the
            # runner re-derives a manifest when it finds none, which would seal
            # the already changed tree and report it verified.
            # The decision has to be stated, so the runner is told explicitly.
            #
            # Post-hoc rendering still works from the clean commit recorded in
            # run_manifest.json; the seal is the fallback that also allows a
            # moved or dirty checkout. So the video is at risk only if the
            # checkout later moves, which is said here rather than promised
            # either way.
            print(
                "WARNING: runtime source generation changed while starting; "
                "running without a sealed runtime manifest. The post-hoc video "
                "will still render from the commit recorded in "
                "run_manifest.json, but there is no seal to fall back on if "
                f"this checkout moves: {exc}",
                file=sys.stderr,
            )
            runtime_manifest_path = None
            runtime_manifest_digest = None
            runtime_source_unsealed = True
    shards = plan["resolved"]["runner"].get("shards")
    if isinstance(shards, list) and shards:
        runner_exit_code = _execute_shards(
            plan,
            runtime_manifest_path=runtime_manifest_path,
            runtime_manifest_digest=runtime_manifest_digest,
            runtime_source_unsealed=runtime_source_unsealed,
        )
    else:
        environment = build_runner_environment(
            plan,
            gpu_bindings=preflight_report.get("gpu_bindings"),
        )
        if runtime_manifest_path is not None and runtime_manifest_digest is not None:
            environment[RUNTIME_SOURCE_MANIFEST_ENV] = str(runtime_manifest_path)
            environment[RUNTIME_SOURCE_MANIFEST_SHA256_ENV] = runtime_manifest_digest
        elif runtime_source_unsealed:
            environment[RUNTIME_SOURCE_UNSEALED_ENV] = "1"
        runner_exit_code = 127
        try:
            completed = subprocess.run(
                [str(plan["runner_script"])],
                cwd=REPO_ROOT,
                env=environment,
                check=False,
            )
            runner_exit_code = int(completed.returncode)
        except OSError as exc:
            print(f"failed to start rollout runner: {type(exc).__name__}: {exc}", file=sys.stderr)

    if runtime_manifest_path is not None and runtime_manifest_digest is not None:
        try:
            verify_runtime_manifest(
                runtime_manifest_path,
                expected_sha256=runtime_manifest_digest,
            )
        except RuntimeSourceDriftError as exc:
            # Again only drift: a manifest that has become unreadable or
            # malformed between launch and finalization is not evidence the
            # rollouts are sound, and still fails.
            #
            # The run has finished. A checkout change only warns, so completed
            # results are kept.
            print(
                "WARNING: runtime source generation changed before launcher "
                f"finalization; keeping the run anyway: {exc}",
                file=sys.stderr,
            )

    if isinstance(shards, list) and shards and runner_exit_code != 0:
        return runner_exit_code
    summary_exit_code, summary, _summary_paths_by_format = summarize_run(plan)
    if runner_exit_code != 0:
        return runner_exit_code
    if summary_exit_code != 0:
        return summary_exit_code
    total_runs = int((summary or {}).get("total_runs", 0))
    if total_runs != int(plan["expected_rollouts"]):
        print(
            f"expected {plan['expected_rollouts']} rollouts but summarized {total_runs}",
            file=sys.stderr,
        )
        return 5
    return 0
