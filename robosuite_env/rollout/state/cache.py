# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Atomic JSON publication and provenance checks for rollout init states."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


SETTLED_INIT_STATE_SCHEMA = "real2sim-task-object-physics-settle-v3"
SETTLED_INIT_STATE_OBJECTS = ("cup", "plate")


def atomic_write_json(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any],
    *,
    mode: int = 0o600,
) -> None:
    """Publish one complete JSON file with a same-directory atomic replace."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        os.fchmod(descriptor, mode)
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1
        with stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory_descriptor = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_saved_settled_init_payload(
    payload: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
) -> None:
    """Reject legacy, partial, or differently-built settled init states."""

    failures: list[str] = []
    if int(payload.get("version", -1)) < 3:
        failures.append("payload version is older than settled-state version 3")

    contract = payload.get("object_settle_contract")
    if contract != expected_contract:
        failures.append("object_settle_contract does not match the current model")

    expected_objects = tuple(expected_contract.get("expected_objects", ()))
    if (
        not expected_objects
        or any(not isinstance(name, str) or not name for name in expected_objects)
        or len(set(expected_objects)) != len(expected_objects)
    ):
        failures.append(
            "current contract expected_objects must be a non-empty ordered "
            "set of object names"
        )
    expected_object_set = set(expected_objects)

    report = payload.get("object_settle")
    if not isinstance(report, Mapping):
        failures.append("object_settle report is missing")
        report = {}
    if report.get("source") != "physics_settle":
        failures.append("object_settle source is not physics_settle")
    if report.get("validated") is not True:
        failures.append("object_settle report is not validated")
    if report.get("rule_grasp_unchanged") is not True:
        failures.append("object_settle did not preserve rule-grasp state")

    required_samples = int(expected_contract.get("stable_samples_required", -1))
    if int(report.get("stable_samples", -1)) < required_samples:
        failures.append(
            "object_settle stable sample count is below the current contract"
        )
    if report.get("thresholds") != expected_contract.get("thresholds"):
        failures.append("object_settle thresholds do not match the contract")

    top_objects = payload.get("objects")
    report_objects = report.get("objects")
    if not isinstance(top_objects, Mapping):
        failures.append("top-level object state is missing")
        top_objects = {}
    if set(top_objects) != expected_object_set:
        failures.append("top-level object set does not match the contract")
    if not isinstance(report_objects, Mapping):
        failures.append("object_settle final object state is missing")
    elif set(report_objects) != expected_object_set:
        failures.append("object_settle object set does not match the contract")
    elif report_objects != top_objects:
        failures.append(
            "object_settle final object state differs from the saved exact state"
        )

    support = report.get("support")
    if not isinstance(support, Mapping) or set(support) != expected_object_set:
        failures.append("object_settle support set does not match the contract")
    else:
        for name in expected_objects:
            state = support.get(name)
            if not isinstance(state, Mapping) or state.get("supported") is not True:
                failures.append(f"{name} lacks saved load-bearing support")

    rule_grasp = payload.get("rule_grasp")
    if not isinstance(rule_grasp, Mapping):
        failures.append("saved rule-grasp state is missing")
    elif (
        rule_grasp.get("active") is not False
        or rule_grasp.get("hand") is not None
        or rule_grasp.get("weld_eq_id") is not None
        or rule_grasp.get("events") != []
    ):
        failures.append("saved rule-grasp state is not pristine")

    if failures:
        raise ValueError("; ".join(failures))
