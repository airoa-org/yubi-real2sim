# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Pure builders for policy-chunk execution records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PolicyPlanConfig:
    execution_steps: int
    overlap_steps: int
    sampler_configured: str
    collapse_reason: str | None
    prev_weight_configured: float | None
    step_shift_configured: int
    sample_num_steps: int | None
    policy_delta_anchor: str
    policy_pose_repr: str


def build_policy_plan_record(
    config: PolicyPlanConfig,
    *,
    chunk_index: int,
    observation_frame: int,
    inference_frame: int,
    execution_start: int,
    sampler: str,
    prev_weight: float | None,
    dynamic_prev_weight: bool,
    prev_weight_schedule: list[float] | tuple[float, ...] | None,
    step_shift: int,
    conditioned: bool,
    execution_anchor: Mapping[str, Any],
    prefix_error: float | None,
    conditioning_prefix: np.ndarray | None,
    conditioning_rebase_meta: Mapping[str, Any] | None,
    chunk: np.ndarray,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable planner audit schema for one inferred chunk."""

    record = {
        "chunk_index": int(chunk_index),
        "observation_frame": int(observation_frame),
        "inference_frame": int(inference_frame),
        "execution_start": int(execution_start),
        "execution_stop": int(execution_start + config.execution_steps),
        "rf_overlap": int(config.overlap_steps),
        "rf_overlap_sampler": str(sampler),
        "rf_overlap_prev_weight": prev_weight,
        "rf_overlap_prev_weight_dynamic": bool(dynamic_prev_weight),
        "rf_overlap_prev_weight_schedule": (
            list(prev_weight_schedule)
            if prev_weight_schedule is not None
            else None
        ),
        "rf_overlap_step_shift": int(step_shift),
        "rf_overlap_sampler_configured": config.sampler_configured,
        "rf_overlap_sampler_effective": str(sampler),
        "rf_overlap_collapse_reason": config.collapse_reason,
        "rf_overlap_prev_weight_configured": config.prev_weight_configured,
        "rf_overlap_step_shift_configured": int(config.step_shift_configured),
        "rf_overlap_sample_num_steps": config.sample_num_steps,
        "conditioned": bool(conditioned),
        "policy_delta_anchor": config.policy_delta_anchor,
        "policy_pose_repr": config.policy_pose_repr,
        "execution_anchor": execution_anchor,
        "prefix_roundtrip_max_abs": prefix_error,
        "conditioning_prefix": (
            np.asarray(conditioning_prefix, dtype=float).tolist()
            if conditioning_prefix is not None
            else []
        ),
        "conditioning_left_reference_rebase": (
            dict(conditioning_rebase_meta)
            if conditioning_rebase_meta is not None
            else None
        ),
        "actions": np.asarray(chunk, dtype=float).tolist(),
    }
    _extend_without_duplicates(record, extra, record_name="policy plan")
    return record


def build_policy_chunk_result(
    *,
    chunk: np.ndarray,
    chunk_index: int,
    obs_meta: Mapping[str, Any],
    obs_log: Any,
    policy_result_meta: Mapping[str, Any],
    prefix_error: float | None,
    bridge_error: float | None,
    bridge_rebased_prev_distance: float | None,
    prev_weight: float | None,
    execution_start: int,
    sampler: str,
    conditioned: bool,
    inference_frame: int,
    conditioning_rebase_meta: Mapping[str, Any] | None,
    target_trajectory: Any,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the worker's common policy-chunk activation payload."""

    result = {
        "chunk": chunk,
        "chunk_index": int(chunk_index),
        "obs_meta": dict(obs_meta),
        "obs_log": obs_log,
        "policy_result_meta": policy_result_meta,
        "prefix_error": prefix_error,
        "bridge_error": bridge_error,
        "bridge_rebased_prev_distance": bridge_rebased_prev_distance,
        "prev_weight": prev_weight,
        "execution_start": int(execution_start),
        "sampler": str(sampler),
        "conditioned": bool(conditioned),
        "inference_frame": int(inference_frame),
        "conditioning_rebase_meta": conditioning_rebase_meta,
        "target_trajectory": target_trajectory,
    }
    _extend_without_duplicates(result, extra, record_name="policy chunk")
    return result


def _extend_without_duplicates(
    record: dict[str, Any],
    extra: Mapping[str, Any] | None,
    *,
    record_name: str,
) -> None:
    if not extra:
        return
    duplicate_keys = set(record).intersection(extra)
    if duplicate_keys:
        raise ValueError(
            f"{record_name} extra fields duplicate common fields: "
            f"{sorted(duplicate_keys)}"
        )
    record.update(extra)


__all__ = [
    "PolicyPlanConfig",
    "build_policy_chunk_result",
    "build_policy_plan_record",
]
