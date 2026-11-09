# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""FR3 JOINT_POSITION gain overrides and telemetry."""

from __future__ import annotations

import math
from typing import Any, Mapping, MutableMapping


def apply_fr3_joint_gain_overrides(
    controller_config: MutableMapping[str, Any],
    *,
    joint_kp: float | None,
    joint_kd: float | None,
) -> None:
    """Apply optional gains to a freshly loaded robosuite controller config.

    The pinned robosuite ``JointPositionController`` accepts a direct ``kd``
    config key through ``**kwargs`` and gives it precedence over
    ``damping_ratio``. When ``kd`` is absent, robosuite retains its historical
    conversion ``kd = 2 * sqrt(kp) * damping_ratio``. Consequently, this
    helper adds ``kd`` only for an explicit ``fr3_joint_kd`` override; leaving
    both overrides unset preserves the loaded controller JSON byte-for-byte.
    """

    arm_config = controller_config["body_parts"]["right"]
    if joint_kp is not None:
        arm_config["kp"] = float(joint_kp)
    if joint_kd is not None:
        arm_config["kd"] = float(joint_kd)


def effective_fr3_joint_kd(joint_config: Mapping[str, Any]) -> float:
    """Return the damping gain that robosuite will use for this FR3 config."""

    direct_kd = joint_config.get("kd")
    if direct_kd is not None:
        return float(direct_kd)
    return (
        2.0
        * math.sqrt(float(joint_config["kp"]))
        * float(joint_config["damping_ratio"])
    )


def fr3_joint_gain_summary(
    joint_config: Mapping[str, Any] | None,
    *,
    active: bool,
    joint_kp_override: float | None,
    joint_kd_override: float | None,
) -> dict[str, float | str | None]:
    """Build the effective FR3 joint-gain fields for ``sim_summary.json``."""

    if not active:
        return {
            "fr3_ik_joint_kp": None,
            "fr3_ik_joint_kd": None,
            "fr3_joint_gain_source": None,
        }
    if joint_config is None:
        raise RuntimeError("active FR3 joint control is missing its controller config")
    return {
        "fr3_ik_joint_kp": float(joint_config["kp"]),
        "fr3_ik_joint_kd": effective_fr3_joint_kd(joint_config),
        "fr3_joint_gain_source": (
            "config_override"
            if joint_kp_override is not None or joint_kd_override is not None
            else "controller_json"
        ),
    }
