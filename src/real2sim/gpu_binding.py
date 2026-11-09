# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-neutral CUDA visibility and MuJoCo EGL device binding."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
import re
import subprocess
from typing import Any, Mapping, Sequence


CUDA_VISIBLE_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"
GPU_BINDINGS_ENV = "REAL2SIM_GPU_BINDINGS_JSON"

_ROLE_ENV_NAMES = {
    "gs": "GS_CUDA_DEVICES",
    "policy": "POLICY_CUDA_DEVICE",
    "sim": "SIM_CUDA_DEVICES",
}
_CUDA_UUID_RE = re.compile(r"GPU-[0-9A-Za-z-]+")
_CANONICAL_INDEX_RE = re.compile(r"0|[1-9][0-9]*")


class GpuBindingError(ValueError):
    """A requested GPU could not be bound inside the visible device set."""


@dataclass(frozen=True)
class GpuBinding:
    """One configured selector bound to its CUDA and host-EGL namespaces."""

    allocation_slot: int
    cuda_token: str
    egl_physical_index: int

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _parse_selector_list(raw: str, *, name: str) -> tuple[int, ...]:
    if not raw:
        return ()
    parts = raw.split(",")
    if any(_CANONICAL_INDEX_RE.fullmatch(part) is None for part in parts):
        raise GpuBindingError(
            f"{name} must be a comma-separated list of canonical non-negative integers, "
            f"got {raw!r}"
        )
    selectors = tuple(int(part) for part in parts)
    if len(set(selectors)) != len(selectors):
        raise GpuBindingError(f"{name} contains duplicate GPU selectors: {raw!r}")
    return selectors


def _visible_cuda_tokens(environment: Mapping[str, str]) -> tuple[str, ...] | None:
    if CUDA_VISIBLE_DEVICES_ENV not in environment:
        return None
    raw = environment[CUDA_VISIBLE_DEVICES_ENV]
    if not raw:
        raise GpuBindingError("CUDA_VISIBLE_DEVICES is set but empty")
    tokens = tuple(raw.split(","))
    if any(not token or token.strip() != token for token in tokens):
        raise GpuBindingError(f"malformed CUDA_VISIBLE_DEVICES: {raw!r}")
    for token in tokens:
        if token.startswith("MIG-") or "/" in token:
            raise GpuBindingError(
                f"MIG CUDA device tokens are not supported for MuJoCo EGL: {token!r}"
            )
        if _CANONICAL_INDEX_RE.fullmatch(token) is None and _CUDA_UUID_RE.fullmatch(
            token
        ) is None:
            raise GpuBindingError(f"unsupported CUDA device token: {token!r}")
    if len(set(tokens)) != len(tokens):
        raise GpuBindingError(
            f"CUDA_VISIBLE_DEVICES contains duplicate entries: {raw!r}"
        )
    return tokens


def resolve_cuda_token(
    selector: int,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Resolve a visible slot, or a host ordinal when visibility is absent."""

    if isinstance(selector, bool) or not isinstance(selector, int) or selector < 0:
        raise GpuBindingError(
            f"GPU selector must be a non-negative integer, got {selector!r}"
        )
    active_environment = os.environ if environment is None else environment
    visible = _visible_cuda_tokens(active_environment)
    if visible is None:
        return str(selector)
    if selector >= len(visible):
        raise GpuBindingError(
            f"GPU visible slot {selector} is outside CUDA_VISIBLE_DEVICES with "
            f"{len(visible)} entries"
        )
    return visible[selector]


def _query_nvidia_gpu_records() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GpuBindingError(
            "cannot resolve configured GPUs with nvidia-smi"
        ) from exc
    records: list[dict[str, str]] = []
    for line_number, raw_line in enumerate(completed.stdout.splitlines(), start=1):
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != 2:
            raise GpuBindingError(
                f"invalid nvidia-smi GPU mapping line {line_number}: {raw_line!r}"
            )
        records.append({"index": fields[0], "uuid": fields[1]})
    return records


def _nvidia_maps(
    nvidia: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, int], dict[int, str]]:
    if nvidia is None:
        records: Sequence[Mapping[str, Any]] = _query_nvidia_gpu_records()
    elif isinstance(nvidia, Mapping):
        raw_records = nvidia.get("gpus")
        if not isinstance(raw_records, Sequence) or isinstance(
            raw_records, (str, bytes, bytearray)
        ):
            raise GpuBindingError("NVIDIA GPU provenance has no GPU records")
        records = raw_records
    else:
        records = nvidia

    uuid_to_index: dict[str, int] = {}
    index_to_uuid: dict[int, str] = {}
    for position, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise GpuBindingError(f"invalid NVIDIA GPU record {position}")
        raw_index = str(record.get("index", ""))
        uuid = str(record.get("uuid", ""))
        if _CANONICAL_INDEX_RE.fullmatch(raw_index) is None:
            raise GpuBindingError(
                f"invalid NVIDIA physical GPU index in record {position}: {raw_index!r}"
            )
        if _CUDA_UUID_RE.fullmatch(uuid) is None:
            raise GpuBindingError(
                f"unsupported NVIDIA GPU UUID in record {position}: {uuid!r}"
            )
        physical_index = int(raw_index)
        if uuid in uuid_to_index or physical_index in index_to_uuid:
            raise GpuBindingError("ambiguous NVIDIA GPU index/UUID mapping")
        uuid_to_index[uuid] = physical_index
        index_to_uuid[physical_index] = uuid
    if not uuid_to_index:
        raise GpuBindingError("nvidia-smi returned no physical GPU mappings")
    return uuid_to_index, index_to_uuid


def resolve_gpu_binding(
    selector: int,
    *,
    environment: Mapping[str, str] | None = None,
    nvidia: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> GpuBinding:
    """Resolve one selector and prove its exact host EGL device exists."""

    active_environment = os.environ if environment is None else environment
    cuda_token = resolve_cuda_token(selector, environment=active_environment)
    uuid_to_index, index_to_uuid = _nvidia_maps(nvidia)
    if _CANONICAL_INDEX_RE.fullmatch(cuda_token) is not None:
        physical_index = int(cuda_token)
        if physical_index not in index_to_uuid:
            raise GpuBindingError(
                f"configured CUDA device {cuda_token!r} is not a host GPU reported by nvidia-smi"
            )
    elif _CUDA_UUID_RE.fullmatch(cuda_token) is not None:
        try:
            physical_index = uuid_to_index[cuda_token]
        except KeyError as exc:
            raise GpuBindingError(
                f"configured CUDA UUID has no exact host EGL mapping: {cuda_token!r}"
            ) from exc
    else:  # Defensive: resolve_cuda_token validates every visible token.
        raise GpuBindingError(f"unsupported CUDA device token: {cuda_token!r}")
    return GpuBinding(
        allocation_slot=selector,
        cuda_token=cuda_token,
        egl_physical_index=physical_index,
    )


def _role_selectors(
    environment: Mapping[str, str], visible: tuple[str, ...] | None
) -> dict[str, tuple[int, ...]]:
    gs = _parse_selector_list(
        environment.get(_ROLE_ENV_NAMES["gs"], ""),
        name=_ROLE_ENV_NAMES["gs"],
    )
    if not gs:
        if visible is None:
            raise GpuBindingError(
                "no GPU selection is configured; set CUDA_VISIBLE_DEVICES or "
                "GS_CUDA_DEVICES explicitly"
            )
        gs = tuple(range(len(visible)))
    policy = _parse_selector_list(
        environment.get(_ROLE_ENV_NAMES["policy"], ""),
        name=_ROLE_ENV_NAMES["policy"],
    )
    if len(policy) > 1:
        raise GpuBindingError("POLICY_CUDA_DEVICE must contain exactly one selector")
    if not policy:
        policy = (gs[0],)
    sim = _parse_selector_list(
        environment.get(_ROLE_ENV_NAMES["sim"], ""),
        name=_ROLE_ENV_NAMES["sim"],
    )
    if not sim:
        sim = gs
    return {"gs": gs, "policy": policy, "sim": sim}


def resolve_gpu_contract(
    environment: Mapping[str, str],
    *,
    nvidia: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve every role into one serializable, scheduler-neutral contract."""

    visible = _visible_cuda_tokens(environment)
    selectors = _role_selectors(environment, visible)
    uuid_to_index, index_to_uuid = _nvidia_maps(nvidia)
    nvidia_records = {
        "gpus": [
            {"index": str(index), "uuid": uuid}
            for index, uuid in sorted(index_to_uuid.items())
        ]
    }
    selectors_to_prove = {
        item for values in selectors.values() for item in values
    }
    # A CUDA visibility boundary is one atomic allocation contract. Prove
    # every exposed token, even when the configured roles use only a subset.
    if visible is not None:
        selectors_to_prove.update(range(len(visible)))
    bindings_by_selector = {
        selector: resolve_gpu_binding(
            selector,
            environment=environment,
            nvidia=nvidia_records,
        )
        for selector in sorted(selectors_to_prove)
    }
    # _nvidia_maps above also validates UUID uniqueness. Keep the local name so
    # static analysis makes it clear both namespaces were proven.
    del uuid_to_index
    return {
        "schema_version": 1,
        "selection_mode": "visible_slots" if visible is not None else "host_ordinals",
        "cuda_visible_devices": (
            environment[CUDA_VISIBLE_DEVICES_ENV] if visible is not None else None
        ),
        "roles": {
            role: [bindings_by_selector[selector].as_dict() for selector in role_selectors]
            for role, role_selectors in selectors.items()
        },
    }


def validate_gpu_contract(
    value: Mapping[str, Any], environment: Mapping[str, str]
) -> dict[str, Any]:
    """Validate a sealed binding contract against the current process boundary."""

    required = {
        "schema_version",
        "selection_mode",
        "cuda_visible_devices",
        "roles",
    }
    schema_version = value.get("schema_version")
    if (
        set(value) != required
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise GpuBindingError("GPU binding contract has an invalid schema")
    visible = _visible_cuda_tokens(environment)
    expected_visible = (
        environment[CUDA_VISIBLE_DEVICES_ENV] if visible is not None else None
    )
    expected_mode = "visible_slots" if visible is not None else "host_ordinals"
    if value.get("cuda_visible_devices") != expected_visible:
        raise GpuBindingError(
            "GPU binding contract differs from the current CUDA_VISIBLE_DEVICES boundary"
        )
    if value.get("selection_mode") != expected_mode:
        raise GpuBindingError("GPU binding contract has the wrong selection mode")
    expected_selectors = _role_selectors(environment, visible)
    roles = value.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != set(_ROLE_ENV_NAMES):
        raise GpuBindingError("GPU binding contract has invalid roles")
    canonical_roles: dict[str, list[dict[str, int | str]]] = {}
    binding_identity: dict[int, tuple[str, int]] = {}
    physical_identity: dict[int, tuple[int, str]] = {}
    for role, selectors in expected_selectors.items():
        entries = roles.get(role)
        if not isinstance(entries, list) or len(entries) != len(selectors):
            raise GpuBindingError(f"GPU binding contract has invalid {role} bindings")
        canonical_entries: list[dict[str, int | str]] = []
        for selector, entry in zip(selectors, entries, strict=True):
            if not isinstance(entry, Mapping) or set(entry) != {
                "allocation_slot",
                "cuda_token",
                "egl_physical_index",
            }:
                raise GpuBindingError(f"GPU binding contract has invalid {role} entry")
            raw_slot = entry.get("allocation_slot")
            token = entry.get("cuda_token")
            physical = entry.get("egl_physical_index")
            if (
                isinstance(raw_slot, bool)
                or not isinstance(raw_slot, int)
                or raw_slot != selector
                or not isinstance(token, str)
            ):
                raise GpuBindingError(f"GPU binding contract changed the {role} selector")
            expected_token = resolve_cuda_token(selector, environment=environment)
            if token != expected_token:
                raise GpuBindingError(
                    f"GPU binding contract changed the CUDA token for selector {selector}"
                )
            if isinstance(physical, bool) or not isinstance(physical, int) or physical < 0:
                raise GpuBindingError(f"GPU binding contract has invalid EGL index for {role}")
            if _CANONICAL_INDEX_RE.fullmatch(token) is not None and physical != int(
                token
            ):
                raise GpuBindingError(
                    f"GPU binding contract changed the physical index for CUDA device {token}"
                )
            identity = (token, physical)
            previous = binding_identity.setdefault(selector, identity)
            if previous != identity:
                raise GpuBindingError(
                    f"GPU selector {selector} has inconsistent role bindings"
                )
            physical_owner = physical_identity.setdefault(
                physical, (selector, token)
            )
            if physical_owner != (selector, token):
                raise GpuBindingError(
                    "GPU binding contract maps distinct CUDA devices to one EGL index"
                )
            canonical_entries.append(
                GpuBinding(selector, token, physical).as_dict()
            )
        canonical_roles[role] = canonical_entries
    return {
        "schema_version": 1,
        "selection_mode": expected_mode,
        "cuda_visible_devices": expected_visible,
        "roles": canonical_roles,
    }


def revalidate_gpu_contract(
    value: Mapping[str, Any],
    environment: Mapping[str, str],
    *,
    nvidia: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Re-prove a sealed contract against the current NVIDIA topology."""

    canonical = validate_gpu_contract(value, environment)
    current = resolve_gpu_contract(environment, nvidia=nvidia)
    if canonical != current:
        raise GpuBindingError(
            "sealed GPU binding contract does not match the current NVIDIA topology"
        )
    return canonical


def contract_json(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def contract_from_environment(
    environment: Mapping[str, str],
    *,
    verify_topology: bool = True,
    nvidia: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    raw = environment.get(GPU_BINDINGS_ENV, "")
    if not raw:
        return resolve_gpu_contract(environment, nvidia=nvidia)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GpuBindingError(f"{GPU_BINDINGS_ENV} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise GpuBindingError(f"{GPU_BINDINGS_ENV} must contain a JSON object")
    if verify_topology:
        return revalidate_gpu_contract(value, environment, nvidia=nvidia)
    return validate_gpu_contract(value, environment)


def _shell_rows(contract: Mapping[str, Any]) -> str:
    roles = contract["roles"]
    role_values = [
        ",".join(str(entry["allocation_slot"]) for entry in roles[role])
        for role in ("gs", "policy", "sim")
    ]
    lines = ["\t".join(("roles", *role_values))]
    unique: dict[int, Mapping[str, Any]] = {}
    for role in ("gs", "policy", "sim"):
        for entry in roles[role]:
            unique[int(entry["allocation_slot"])] = entry
    lines.extend(
        "\t".join(
            (
                "binding",
                str(selector),
                str(entry["cuda_token"]),
                str(entry["egl_physical_index"]),
            )
        )
        for selector, entry in sorted(unique.items())
    )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("shell", "json"),
        help="emit the current sealed/direct GPU binding contract",
    )
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="validate a sealed fixture contract without querying NVIDIA topology",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_arg_parser().parse_args(argv)
    try:
        contract = contract_from_environment(
            os.environ,
            verify_topology=not arguments.structural_only,
        )
    except GpuBindingError as exc:
        raise SystemExit(f"GPU binding failed: {exc}") from exc
    if arguments.command == "shell":
        print(_shell_rows(contract), end="")
    else:
        print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
