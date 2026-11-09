# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Pure render-replay state contracts for MuJoCo rollout logs.

The full-model ``qpos`` vector is the authoritative replay state.  A second,
named view of every task-object free joint is stored beside it so omissions,
legacy aliases, and address drift can be detected before a renderer consumes
the log.  This module deliberately has no MuJoCo dependency; callers derive
the inventory from a compiled model and pass ordinary mappings and vectors.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any


RENDER_STATE_SCHEMA = "real2sim-render-state-v1"
RENDER_STATE_CONTRACT_SCHEMA = "real2sim-render-state-contract-v1"
QUATERNION_NORM_ATOL = 1e-9

_INVENTORY_KEYS = frozenset(
    {"name", "role", "joint_name", "qpos_address", "dof_address"}
)
_CONTRACT_KEYS = frozenset(
    {
        "schema",
        "state_schema",
        "model_signature",
        "nq",
        "target",
        "inventory",
        "inventory_sha256",
    }
)
_SUMMARY_KEYS = _CONTRACT_KEYS | {"rows_written", "complete"}
_RENDER_STATE_KEYS = frozenset(
    {"schema", "model_signature", "nq", "inventory_sha256", "qpos"}
)
_TASK_OBJECT_STATE_KEYS = frozenset({"qpos"})


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(f"{label} keys do not match the schema: " + " ".join(details))


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer, not bool")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _vector(value: Any, length: int, label: str) -> list[float]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(
            f"{label} must have shape ({length},), got {type(value).__name__!r}"
        )
    try:
        value_length = len(value)
        items = list(value)
    except (TypeError, ValueError):
        value_length = type(value).__name__
        items = []
    if value_length != length or len(items) != length:
        shape = value_length
        raise ValueError(f"{label} must have shape ({length},), got {shape!r}")
    result: list[float] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"{label}[{index}] must be numeric, not bool")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{label}[{index}] must be finite")
        result.append(number)
    return result


def _canonical_inventory(
    inventory: Any, *, nq: int | None = None
) -> list[dict[str, Any]]:
    if (
        isinstance(inventory, (str, bytes, bytearray, Mapping))
        or not isinstance(inventory, Sequence)
        or not inventory
    ):
        raise ValueError("render-state inventory must be a non-empty sequence")

    canonical: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(inventory):
        entry = _mapping(raw_entry, f"inventory[{index}]")
        _exact_keys(entry, _INVENTORY_KEYS, f"inventory[{index}]")
        canonical.append(
            {
                "name": _nonempty_string(entry["name"], f"inventory[{index}].name"),
                "role": _nonempty_string(entry["role"], f"inventory[{index}].role"),
                "joint_name": _nonempty_string(
                    entry["joint_name"], f"inventory[{index}].joint_name"
                ),
                "qpos_address": _integer(
                    entry["qpos_address"], f"inventory[{index}].qpos_address"
                ),
                "dof_address": _integer(
                    entry["dof_address"], f"inventory[{index}].dof_address"
                ),
            }
        )

    canonical.sort(key=lambda item: item["name"])
    for field in ("name", "joint_name", "qpos_address", "dof_address"):
        values = [entry[field] for entry in canonical]
        if len(set(values)) != len(values):
            raise ValueError(f"render-state inventory contains duplicate {field}")

    occupied_qpos: set[int] = set()
    for entry in canonical:
        address = int(entry["qpos_address"])
        if nq is not None and address + 7 > nq:
            raise ValueError(
                f"task object {entry['name']!r} qpos slice [{address}:{address + 7}] "
                f"exceeds nq={nq}"
            )
        indices = set(range(address, address + 7))
        if occupied_qpos & indices:
            raise ValueError("render-state inventory contains overlapping qpos slices")
        occupied_qpos.update(indices)
    return canonical


def canonical_inventory_sha256(inventory: Any) -> str:
    """Return the stable SHA-256 of a validated, name-sorted inventory."""

    canonical = _canonical_inventory(inventory)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_render_state_contract(
    *,
    model_signature: str,
    nq: int,
    target: str,
    inventory: Any,
) -> dict[str, Any]:
    """Build a deterministic contract for render-state rows."""

    signature = _nonempty_string(model_signature, "model_signature")
    qpos_count = _integer(nq, "nq", minimum=1)
    target_name = _nonempty_string(target, "target")
    canonical = _canonical_inventory(inventory, nq=qpos_count)
    by_name = {entry["name"]: entry for entry in canonical}
    if target_name not in by_name:
        raise ValueError(f"target {target_name!r} is absent from the inventory")
    target_roles = [entry["name"] for entry in canonical if entry["role"] == "target"]
    if target_roles != [target_name]:
        raise ValueError(
            "inventory must assign role='target' to exactly the declared target; "
            f"declared={target_name!r} role_targets={target_roles}"
        )
    return {
        "schema": RENDER_STATE_CONTRACT_SCHEMA,
        "state_schema": RENDER_STATE_SCHEMA,
        "model_signature": signature,
        "nq": qpos_count,
        "target": target_name,
        "inventory": canonical,
        "inventory_sha256": canonical_inventory_sha256(canonical),
    }


def validate_render_state_contract(
    contract: Any,
    *,
    required: bool = True,
) -> dict[str, Any] | None:
    """Validate and return a canonical base contract.

    ``None`` is accepted only for an explicitly optional legacy path.
    """

    if contract is None:
        if required:
            raise ValueError("render-state contract is required")
        return None
    raw = _mapping(contract, "render-state contract")
    _exact_keys(raw, _CONTRACT_KEYS, "render-state contract")
    if raw["schema"] != RENDER_STATE_CONTRACT_SCHEMA:
        raise ValueError("render-state contract schema mismatch")
    if raw["state_schema"] != RENDER_STATE_SCHEMA:
        raise ValueError("render-state row schema mismatch in contract")
    expected = build_render_state_contract(
        model_signature=raw["model_signature"],
        nq=raw["nq"],
        target=raw["target"],
        inventory=raw["inventory"],
    )
    if dict(raw) != expected:
        if raw.get("inventory_sha256") != expected["inventory_sha256"]:
            raise ValueError("render-state inventory SHA-256 mismatch")
        raise ValueError("render-state contract is not canonical")
    return expected


def build_render_state_summary_contract(
    contract: Any,
    *,
    rows_written: int,
    complete: bool,
) -> dict[str, Any]:
    """Add publication status to a validated base contract for a run summary."""

    base = validate_render_state_contract(contract)
    assert base is not None
    if not isinstance(complete, bool):
        raise ValueError("complete must be bool")
    return {
        **base,
        "rows_written": _integer(rows_written, "rows_written"),
        "complete": complete,
    }


def encode_render_state_row(qpos: Any, contract: Any) -> dict[str, Any]:
    """Encode merge-ready ``render_state`` and ``task_objects`` row fields."""

    base = validate_render_state_contract(contract)
    assert base is not None
    full_qpos = _vector(qpos, int(base["nq"]), "render_state.qpos")
    task_objects = {
        entry["name"]: {
            "qpos": full_qpos[
                int(entry["qpos_address"]): int(entry["qpos_address"]) + 7
            ]
        }
        for entry in base["inventory"]
    }
    payload = {
        "render_state": {
            "schema": RENDER_STATE_SCHEMA,
            "model_signature": base["model_signature"],
            "nq": base["nq"],
            "inventory_sha256": base["inventory_sha256"],
            "qpos": full_qpos,
        },
        "task_objects": task_objects,
    }
    validate_render_state_row(payload, base)
    return payload


def validate_render_state_row(
    row: Any,
    contract: Any,
    *,
    required: bool = True,
) -> dict[str, Any] | None:
    """Strictly validate one possibly larger rollout-log row."""

    raw_row = _mapping(row, "render-state row")
    has_render_state = "render_state" in raw_row
    has_task_objects = "task_objects" in raw_row
    base = validate_render_state_contract(contract, required=required)
    if base is None:
        if has_render_state or has_task_objects:
            raise ValueError("render-state row is present without a contract")
        return None
    if not has_render_state and not has_task_objects and not required:
        return None
    if not has_render_state or not has_task_objects:
        missing = "render_state" if not has_render_state else "task_objects"
        raise ValueError(f"render-state row is missing {missing}")

    state = _mapping(raw_row["render_state"], "render_state")
    _exact_keys(state, _RENDER_STATE_KEYS, "render_state")
    if state["schema"] != RENDER_STATE_SCHEMA:
        raise ValueError("render_state schema mismatch")
    if state["model_signature"] != base["model_signature"]:
        raise ValueError("render_state model signature mismatch")
    if (
        isinstance(state["nq"], bool)
        or not isinstance(state["nq"], Integral)
        or int(state["nq"]) != base["nq"]
    ):
        raise ValueError("render_state nq mismatch")
    if state["inventory_sha256"] != base["inventory_sha256"]:
        raise ValueError("render_state inventory SHA-256 mismatch")
    full_qpos = _vector(state["qpos"], int(base["nq"]), "render_state.qpos")

    objects = _mapping(raw_row["task_objects"], "task_objects")
    expected_names = {entry["name"] for entry in base["inventory"]}
    actual_names = set(objects)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            "task_objects inventory mismatch: "
            f"missing={missing} extra={extra}"
        )
    for entry in base["inventory"]:
        name = entry["name"]
        object_state = _mapping(objects[name], f"task_objects[{name!r}]")
        _exact_keys(
            object_state, _TASK_OBJECT_STATE_KEYS, f"task_objects[{name!r}]"
        )
        object_qpos = _vector(
            object_state["qpos"], 7, f"task_objects[{name!r}].qpos"
        )
        quaternion = object_qpos[3:7]
        quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
        if quaternion_norm == 0.0:
            raise ValueError(f"task object {name!r} has a zero quaternion")
        if not math.isclose(
            quaternion_norm,
            1.0,
            rel_tol=0.0,
            abs_tol=QUATERNION_NORM_ATOL,
        ):
            raise ValueError(
                f"task object {name!r} has a non-unit quaternion: "
                f"norm={quaternion_norm:.17g}"
            )
        address = int(entry["qpos_address"])
        expected_qpos = full_qpos[address : address + 7]
        if object_qpos != expected_qpos:
            raise ValueError(
                f"task_objects[{name!r}].qpos does not exactly match "
                f"render_state.qpos[{address}:{address + 7}]"
            )

    return {
        "qpos": full_qpos,
        "task_objects": {
            name: list(objects[name]["qpos"]) for name in sorted(expected_names)
        },
    }


def validate_render_state_rows(
    rows: Any,
    summary_contract: Any,
    *,
    expected_contract: Any = None,
    required: bool = True,
) -> dict[str, Any] | None:
    """Validate a complete row collection against its published summary."""

    if (
        isinstance(rows, (str, bytes, bytearray, Mapping))
        or not isinstance(rows, Sequence)
    ):
        raise ValueError("render-state rows must be a sequence")
    if summary_contract is None:
        if required:
            raise ValueError("render-state summary contract is required")
        if any(
            isinstance(row, Mapping)
            and ("render_state" in row or "task_objects" in row)
            for row in rows
        ):
            raise ValueError("render-state rows are present without a summary contract")
        return None

    summary = _mapping(summary_contract, "render-state summary contract")
    _exact_keys(summary, _SUMMARY_KEYS, "render-state summary contract")
    base_payload = {key: summary[key] for key in _CONTRACT_KEYS}
    base = validate_render_state_contract(base_payload)
    assert base is not None
    if expected_contract is not None:
        expected = validate_render_state_contract(expected_contract)
        if base != expected:
            raise ValueError("render-state summary contract does not match expected contract")
    rows_written = _integer(summary["rows_written"], "rows_written")
    if rows_written != len(rows):
        raise ValueError(
            f"render-state rows_written mismatch: summary={rows_written} "
            f"actual={len(rows)}"
        )
    if not isinstance(summary["complete"], bool):
        raise ValueError("render-state summary complete must be bool")
    if summary["complete"] is not True:
        raise ValueError("render-state summary is incomplete")
    for index, row in enumerate(rows):
        try:
            validate_render_state_row(row, base)
        except ValueError as exc:
            raise ValueError(f"render-state row {index} is invalid: {exc}") from exc
    return {
        "rows_written": rows_written,
        "complete": True,
        "model_signature": base["model_signature"],
        "inventory_sha256": base["inventory_sha256"],
        "task_object_names": [entry["name"] for entry in base["inventory"]],
        "target": base["target"],
    }


__all__ = [
    "QUATERNION_NORM_ATOL",
    "RENDER_STATE_CONTRACT_SCHEMA",
    "RENDER_STATE_SCHEMA",
    "build_render_state_contract",
    "build_render_state_summary_contract",
    "canonical_inventory_sha256",
    "encode_render_state_row",
    "validate_render_state_contract",
    "validate_render_state_row",
    "validate_render_state_rows",
]
