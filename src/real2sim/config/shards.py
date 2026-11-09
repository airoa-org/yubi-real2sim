# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Validation and CLI parsing for Slurm rollout shards."""

from __future__ import annotations

import argparse
import os
import re
from typing import Any, Mapping

from real2sim.config._types import ConfigError, _integer, _mapping, _string


SHARD_KEYS = frozenset(
    {
        "name",
        "repeats",
        "slurm_job_id",
        "gs_cuda_devices",
        "sim_cuda_devices",
        "policy_cuda_device",
        "cpus_per_task",
        "cpus",
        "policy_batch_size",
        "server_port",
        "gs_port_base",
    }
)

_GPU_LIST_RE = re.compile(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*")
_GPU_RE = re.compile(r"0|[1-9][0-9]*")
_JOB_ID_RE = re.compile(r"[1-9][0-9]*")
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_CLI_SHARD_RE = re.compile(
    r"(?P<job>[1-9][0-9]*):"
    r"(?P<gpus>(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*):"
    r"(?P<start>[1-9][0-9]*)-(?P<stop>[1-9][0-9]*)"
)


def parse_shards_cli(value: str) -> list[dict[str, Any]]:
    """Parse ``JOB:GPUS:START-STOP`` entries from one ``--shards`` value."""

    entries = value.split()
    if not entries:
        raise argparse.ArgumentTypeError("--shards must contain at least one shard")
    shards: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        match = _CLI_SHARD_RE.fullmatch(entry)
        if match is None:
            raise argparse.ArgumentTypeError(
                "--shards entries must use JOB:GPU[,GPU...]:START-STOP"
            )
        gpus = match.group("gpus")
        shards.append(
            {
                "name": f"{index:02d}",
                "repeats": [int(match.group("start")), int(match.group("stop"))],
                "slurm_job_id": match.group("job"),
                "gs_cuda_devices": gpus,
                "sim_cuda_devices": gpus,
                "policy_cuda_device": gpus.split(",", 1)[0],
            }
        )
    return shards


def _gpu_list(value: Any, path: str) -> str:
    result = _string(value, path)
    if _GPU_LIST_RE.fullmatch(result) is None:
        raise ConfigError(
            f"{path} must be a comma-separated list of canonical non-negative GPU selectors",
            path=path,
            got=result,
        )
    selectors = result.split(",")
    if len(selectors) != len(set(selectors)):
        raise ConfigError(f"{path} must not contain duplicate GPU selectors", path=path)
    return result


def _gpu(value: Any, path: str) -> str:
    result = _string(value, path)
    if _GPU_RE.fullmatch(result) is None:
        raise ConfigError(
            f"{path} must be one canonical non-negative GPU selector",
            path=path,
            got=result,
        )
    return result


_CPU_LIST_RE = re.compile(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*")


def _cpu_list(value: Any, path: str) -> str:
    result = _string(value, path)
    if _CPU_LIST_RE.fullmatch(result) is None:
        raise ConfigError(
            f"{path} must be a comma-separated list of canonical non-negative CPU ids",
            path=path,
            got=result,
        )
    ids = result.split(",")
    if len(ids) != len(set(ids)):
        raise ConfigError(f"{path} must not contain duplicate CPU ids", path=path)
    return result


def _optional_port(value: Any, path: str) -> int | None:
    if value is None:
        return None
    return _integer(value, path, minimum=1, maximum=65535)


def _find_free_port(
    start: int,
    count: int,
    occupied: Mapping[int, str],
) -> int | None:
    candidate = start
    while candidate + count - 1 <= 65535:
        if all(port not in occupied for port in range(candidate, candidate + count)):
            return candidate
        candidate += 1
    return None


def _reserve_ports(
    occupied: dict[int, str],
    *,
    start: int,
    count: int,
    owner: str,
) -> None:
    if start + count - 1 > 65535:
        raise ConfigError(f"{owner} exceeds TCP port 65535")
    for port in range(start, start + count):
        previous = occupied.get(port)
        if previous is not None:
            raise ConfigError(
                f"shard ports collide at {port}: {previous} and {owner}",
                path="runner.shards",
            )
        occupied[port] = owner


def resolve_shards(
    raw_shards: Any,
    *,
    repeats: int,
    scene_count: int,
    runner_mode: str,
    default_policy_batch_size: int,
    default_server_port: int,
    default_gs_port_base: int,
    gs_servers_per_gpu: int = 1,
    environment: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Validate, normalize, and assign collision-free ports to all shards."""

    if not isinstance(raw_shards, list):
        raise ConfigError("runner.shards must be a YAML list", path="runner.shards")
    if not raw_shards:
        raise ConfigError(
            "runner.shards must not be empty; omit it for an unsharded run",
            path="runner.shards",
        )
    source_environment = os.environ if environment is None else environment
    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, raw in enumerate(raw_shards):
        path = f"runner.shards[{index}]"
        shard = _mapping(raw, path)
        unknown = sorted(set(shard) - SHARD_KEYS)
        if unknown:
            raise ConfigError(
                f"{path} contains unknown keys: {', '.join(unknown)}",
                path=path,
                allowed=sorted(SHARD_KEYS),
                got=unknown,
            )
        required = {
            "repeats",
            "gs_cuda_devices",
            "sim_cuda_devices",
            "policy_cuda_device",
        }
        missing = sorted(required - set(shard))
        if missing:
            raise ConfigError(
                f"{path} is missing required keys: {', '.join(missing)}",
                path=path,
            )
        name = _string(shard.get("name", f"{index + 1:02d}"), f"{path}.name")
        if _NAME_RE.fullmatch(name) is None or name == "aggregate":
            raise ConfigError(
                f"{path}.name must be a safe path component other than 'aggregate'",
                path=f"{path}.name",
                got=name,
            )
        if name in seen_names:
            raise ConfigError(f"runner.shards contains duplicate name {name!r}")
        seen_names.add(name)

        raw_range = shard["repeats"]
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            raise ConfigError(
                f"{path}.repeats must be [start, stop] with inclusive endpoints",
                path=f"{path}.repeats",
            )
        repeat_start = _integer(raw_range[0], f"{path}.repeats[0]", minimum=1)
        repeat_stop = _integer(raw_range[1], f"{path}.repeats[1]", minimum=1)
        if repeat_start > repeat_stop or repeat_stop > repeats:
            raise ConfigError(
                f"{path}.repeats must satisfy 1 <= start <= stop <= runner.repeats",
                path=f"{path}.repeats",
            )
        range_length = repeat_stop - repeat_start + 1

        raw_job_id = shard.get("slurm_job_id")
        if raw_job_id is None:
            raw_job_id = source_environment.get("SLURM_JOB_ID")
            if raw_job_id is None:
                raise ConfigError(
                    f"{path}.slurm_job_id is omitted and SLURM_JOB_ID is unset",
                    path=f"{path}.slurm_job_id",
                )
        job_id = _string(raw_job_id, f"{path}.slurm_job_id")
        if _JOB_ID_RE.fullmatch(job_id) is None:
            raise ConfigError(
                f"{path}.slurm_job_id must be a positive numeric Slurm job ID",
                path=f"{path}.slurm_job_id",
                got=job_id,
            )

        policy_batch_size = _integer(
            shard.get("policy_batch_size", default_policy_batch_size),
            f"{path}.policy_batch_size",
            minimum=1,
        )
        if policy_batch_size > range_length:
            raise ConfigError(
                f"{path}.policy_batch_size must not exceed its repeat-range length",
                path=f"{path}.policy_batch_size",
            )
        if policy_batch_size % scene_count != 0:
            raise ConfigError(
                f"{path}.policy_batch_size must be divisible by len(scenes)",
                path=f"{path}.policy_batch_size",
            )
        parallel_repeats = policy_batch_size // scene_count
        if parallel_repeats > range_length or range_length % parallel_repeats != 0:
            raise ConfigError(
                f"{path} repeat-range length must be divisible by "
                "policy_batch_size / len(scenes)",
                path=f"{path}.repeats",
            )
        if runner_mode == "sequential_waves" and parallel_repeats != 1:
            raise ConfigError(
                f"{path}.policy_batch_size must equal len(scenes) for sequential_waves",
                path=f"{path}.policy_batch_size",
            )

        gs_cuda_devices = _gpu_list(
            shard["gs_cuda_devices"], f"{path}.gs_cuda_devices"
        )
        sim_cuda_devices = _gpu_list(
            shard["sim_cuda_devices"], f"{path}.sim_cuda_devices"
        )
        policy_cuda_device = _gpu(
            shard["policy_cuda_device"], f"{path}.policy_cuda_device"
        )
        cpus_per_task = (
            None
            if shard.get("cpus_per_task") is None
            else _integer(
                shard["cpus_per_task"], f"{path}.cpus_per_task", minimum=1
            )
        )
        cpus = (
            None
            if shard.get("cpus") is None
            else _cpu_list(shard["cpus"], f"{path}.cpus")
        )
        if cpus is not None and cpus_per_task is not None:
            # cpus_per_task asks Slurm for a count and lets every overlapping
            # step of the allocation land on the same CPUs; cpus names the ids
            # this shard owns. Asking for both would silently drop the naming.
            raise ConfigError(
                f"{path} sets both cpus and cpus_per_task; cpus names the CPU ids "
                "this shard owns and replaces the count",
                path=f"{path}.cpus",
            )
        # resolve_ports starts at gs_port_base and creates at most one server
        # per scene and configured GS selector used by a policy wave, times the
        # servers each selector carries, so it reserves
        # scenes x selectors x servers_per_gpu ports.
        # The shell starts min(parallel_repeats, selectors * servers_per_gpu)
        # per scene, so this is the conservative side: it can over-reserve, it
        # cannot under-reserve.
        gs_port_count = (
            scene_count
            * min(len(gs_cuda_devices.split(",")), parallel_repeats)
            * gs_servers_per_gpu
        )
        normalized.append(
            {
                "name": name,
                "repeats": [repeat_start, repeat_stop],
                "repeat_start": repeat_start,
                "repeat_stop": repeat_stop,
                "slurm_job_id": job_id,
                "gs_cuda_devices": gs_cuda_devices,
                "sim_cuda_devices": sim_cuda_devices,
                "policy_cuda_device": policy_cuda_device,
                "cpus_per_task": cpus_per_task,
                "cpus": cpus,
                "policy_batch_size": policy_batch_size,
                "parallel_repeats": parallel_repeats,
                "server_port": _optional_port(
                    shard.get("server_port"), f"{path}.server_port"
                ),
                "gs_port_base": _optional_port(
                    shard.get("gs_port_base"), f"{path}.gs_port_base"
                ),
                "_gs_port_count": gs_port_count,
            }
        )

    ordered_ranges = sorted(
        (item["repeat_start"], item["repeat_stop"], item["name"])
        for item in normalized
    )
    expected_start = 1
    for start, stop, name in ordered_ranges:
        if start != expected_start:
            relation = "overlap" if start < expected_start else "gap"
            raise ConfigError(
                f"runner.shards has a repeat-range {relation} before shard {name!r}; "
                f"expected next repeat {expected_start}, got {start}",
                path="runner.shards",
            )
        expected_start = stop + 1
    if expected_start != repeats + 1:
        raise ConfigError(
            "runner.shards does not cover every repeat through runner.repeats; "
            f"coverage stops at {expected_start - 1}, expected {repeats}",
            path="runner.shards",
        )

    # Two shards that name the same GPU inside the same allocation would each
    # start their own GS server, policy server and simulator workers on that
    # device. That is the double-booking the port resolver below already
    # rejects for TCP ports, so reject it for GPUs too: one shard owns one
    # GPU, and the config states which repeats run on it.
    gpu_owner: dict[tuple[str, str], str] = {}
    for item in normalized:
        selectors = {
            *item["gs_cuda_devices"].split(","),
            *item["sim_cuda_devices"].split(","),
            item["policy_cuda_device"],
        }
        for selector in sorted(selectors):
            key = (item["slurm_job_id"], selector)
            previous = gpu_owner.get(key)
            if previous is not None:
                raise ConfigError(
                    f"shards {previous!r} and {item['name']!r} both claim GPU "
                    f"{selector} of Slurm job {item['slurm_job_id']}",
                    path="runner.shards",
                )
            gpu_owner[key] = item["name"]

    # Same rule for CPUs. Overlapping steps of one allocation are not given
    # disjoint CPUs by Slurm, so a shard that names its ids owns them, and two
    # shards of one job may not name the same id.
    cpu_owner: dict[tuple[str, str], str] = {}
    for item in normalized:
        if item["cpus"] is None:
            continue
        for cpu in item["cpus"].split(","):
            key = (item["slurm_job_id"], cpu)
            previous = cpu_owner.get(key)
            if previous is not None:
                raise ConfigError(
                    f"shards {previous!r} and {item['name']!r} both claim CPU "
                    f"{cpu} of Slurm job {item['slurm_job_id']}",
                    path="runner.shards",
                )
            cpu_owner[key] = item["name"]

    occupied: dict[int, str] = {}
    for item in normalized:
        if item["server_port"] is not None:
            _reserve_ports(
                occupied,
                start=item["server_port"],
                count=1,
                owner=f"shard {item['name']} server_port",
            )
        if item["gs_port_base"] is not None:
            _reserve_ports(
                occupied,
                start=item["gs_port_base"],
                count=item["_gs_port_count"],
                owner=f"shard {item['name']} GS ports",
            )

    next_server_port = default_server_port
    next_gs_port = default_gs_port_base
    for item in normalized:
        if item["server_port"] is None:
            port = _find_free_port(next_server_port, 1, occupied)
            if port is None:
                raise ConfigError("could not auto-assign a shard policy server port")
            item["server_port"] = port
            _reserve_ports(
                occupied,
                start=port,
                count=1,
                owner=f"shard {item['name']} server_port",
            )
            next_server_port = port + 1
        if item["gs_port_base"] is None:
            port = _find_free_port(
                next_gs_port, item["_gs_port_count"], occupied
            )
            if port is None:
                raise ConfigError("could not auto-assign a shard GS port range")
            item["gs_port_base"] = port
            _reserve_ports(
                occupied,
                start=port,
                count=item["_gs_port_count"],
                owner=f"shard {item['name']} GS ports",
            )
            next_gs_port = port + item["_gs_port_count"]

    for item in normalized:
        del item["_gs_port_count"]
    return normalized


__all__ = ["SHARD_KEYS", "parse_shards_cli", "resolve_shards"]
