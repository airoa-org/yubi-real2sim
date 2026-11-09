# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Worker validation for stability thresholds shared by placement tasks."""

from types import SimpleNamespace

import pytest

from robosuite_env.rollout.setup import args as setup_args


def test_worker_redirects_legacy_pose_before_validating_arguments(monkeypatch):
    arguments = SimpleNamespace(
        initial_arm_joints_json="config/fr3_grip07_initial_arm_joints.json",
        no_obs_gs=True,
        obs_gs_only=True,
    )
    monkeypatch.setattr(setup_args, "parse", lambda: arguments)
    with pytest.raises(SystemExit, match="mutually exclusive"):
        setup_args.run(SimpleNamespace())
    assert "assets/initial_poses/" in arguments.initial_arm_joints_json


def test_worker_refuses_unknown_task_before_its_runtime_preflight(monkeypatch):
    arguments = SimpleNamespace(
        initial_arm_joints_json=None,
        no_obs_gs=False,
        obs_gs_only=False,
        objects="unknown_task",
    )
    monkeypatch.setattr(setup_args, "parse", lambda: arguments)

    def _preflight(*_args, **_kwargs):
        raise AssertionError("an unknown task must not reach the runtime preflight")

    monkeypatch.setattr(setup_args, "run_runtime_preflight", _preflight)
    with pytest.raises(
        SystemExit,
        match=r"unknown task\.objects 'unknown_task'",
    ):
        setup_args.run(SimpleNamespace())


def _arguments(objects, **overrides):
    values = {
        "objects": objects,
        "success_stable_sec": 2.0,
        "success_linear_speed_mm_s": 1.0,
        "success_angular_speed_rad_s": 0.03,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("objects", ["box_pack"])
@pytest.mark.parametrize(
    "value",
    [
        0.0,
        float("nan"),
    ],
)
def test_stable_duration_is_validated_for_each_consumer(objects, value):
    with pytest.raises(SystemExit, match="--success-stable-sec.*(positive|finite)"):
        setup_args._validate_success_stability_arguments(
            _arguments(objects, success_stable_sec=value)
        )


@pytest.mark.parametrize("objects", ["box_pack"])
@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        (
            "success_linear_speed_mm_s",
            -0.1,
            "--success-linear-speed-mm-s.*non-negative",
        ),
        (
            "success_angular_speed_rad_s",
            float("nan"),
            "--success-angular-speed-rad-s.*finite",
        ),
    ],
)
def test_motion_thresholds_are_validated_for_each_consumer(
    objects, option, value, message
):
    with pytest.raises(SystemExit, match=message):
        setup_args._validate_success_stability_arguments(
            _arguments(objects, **{option: value})
        )


def test_cup_plate_does_not_consume_object_motion_thresholds():
    setup_args._validate_success_stability_arguments(
        _arguments(
            "cup_plate",
            success_linear_speed_mm_s=-1.0,
            success_angular_speed_rad_s=float("nan"),
        )
    )
