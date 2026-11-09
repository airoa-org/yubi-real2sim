# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Tests for optional FR3 JOINT_POSITION gain overrides."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from robosuite_env.robots.fr3.joint_gains import (
    apply_fr3_joint_gain_overrides,
    fr3_joint_gain_summary,
)


CONTROLLER_PATH = (
    Path(__file__).resolve().parents[3]
    / "robosuite_env/controllers/default_fr3_joint_ik.json"
)


def _loaded_controller_config() -> dict:
    source = json.loads(CONTROLLER_PATH.read_text(encoding="utf-8"))
    return {
        "body_parts": {
            "right": deepcopy(source["body_parts"]["arms"]["right"]),
        }
    }


def _summary(config: dict, *, kp: float | None, kd: float | None) -> dict:
    return fr3_joint_gain_summary(
        config["body_parts"]["right"],
        active=True,
        joint_kp_override=kp,
        joint_kd_override=kd,
    )


def test_omitted_gains_preserve_the_controller_json_and_default_damping() -> None:
    config = _loaded_controller_config()
    original = deepcopy(config)

    apply_fr3_joint_gain_overrides(config, joint_kp=None, joint_kd=None)

    assert config == original
    assert config["body_parts"]["right"]["kp"] == 900
    assert config["body_parts"]["right"]["damping_ratio"] == 1
    assert "kd" not in config["body_parts"]["right"]
    assert _summary(config, kp=None, kd=None) == {
        "fr3_ik_joint_kp": 900.0,
        "fr3_ik_joint_kd": 60.0,
        "fr3_joint_gain_source": "controller_json",
    }


def test_kp_override_reaches_controller_and_uses_existing_damping_ratio() -> None:
    config = _loaded_controller_config()

    apply_fr3_joint_gain_overrides(config, joint_kp=400.0, joint_kd=None)

    arm_config = config["body_parts"]["right"]
    assert arm_config["kp"] == 400.0
    assert arm_config["damping_ratio"] == 1
    assert "kd" not in arm_config
    assert _summary(config, kp=400.0, kd=None) == {
        "fr3_ik_joint_kp": 400.0,
        "fr3_ik_joint_kd": 40.0,
        "fr3_joint_gain_source": "config_override",
    }


def test_kd_override_is_injected_directly_and_reported_without_conversion() -> None:
    config = _loaded_controller_config()

    apply_fr3_joint_gain_overrides(config, joint_kp=400.0, joint_kd=80.0)

    arm_config = config["body_parts"]["right"]
    assert arm_config["kp"] == 400.0
    assert arm_config["kd"] == 80.0
    assert arm_config["damping_ratio"] == 1
    assert _summary(config, kp=400.0, kd=80.0) == {
        "fr3_ik_joint_kp": 400.0,
        "fr3_ik_joint_kd": 80.0,
        "fr3_joint_gain_source": "config_override",
    }


def test_kinematic_summary_does_not_activate_joint_gain_telemetry() -> None:
    assert fr3_joint_gain_summary(
        None,
        active=False,
        joint_kp_override=400.0,
        joint_kd_override=80.0,
    ) == {
        "fr3_ik_joint_kp": None,
        "fr3_ik_joint_kd": None,
        "fr3_joint_gain_source": None,
    }
