# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Launch preflight follows the Ruckig path the worker runs.

The pinned-Ruckig probe once keyed on the literal rig name ``fr3`` although
``fr3_duo`` executes Ruckig too.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest
import yaml

from tests._helpers.launcher import base_config

REPO_ROOT = Path(__file__).resolve().parents[3]

from real2sim import launcher  # noqa: E402
from real2sim import preflight  # noqa: E402
from real2sim.config._types import ConfigError  # noqa: E402


@pytest.fixture(autouse=True)
def _launcher_environment(monkeypatch, isolated_environ):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def _plan(root: Path, *, robot: str = "fr3", physics: str = "dynamic"):
    config = base_config()
    config["simulation"]["parameters"]["physics"] = physics
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "rollout.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    plan = launcher.load_rollout_config(
        path, repo_root=root, timestamp="20260730_120000"
    )
    plan["resolved"]["robot"] = robot
    return plan


def _pinned_ruckig_installed() -> bool:
    try:
        import ruckig
    except ImportError:
        return False
    return getattr(ruckig, "__version__", None) == preflight.REQUIRED_RUCKIG_VERSION


# --------------------------------------------------------------------------
# Ruckig: the probe follows the runtime path, not a rig name.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("robot", ("fr3", "fr3_duo"))
def test_every_rig_that_builds_ruckig_runs_the_pinned_ruckig_probe(tmp_path, robot):
    plan = _plan(tmp_path, robot=robot, physics="dynamic")
    with mock.patch.object(
        preflight.subprocess,
        "run",
        return_value=subprocess.CompletedProcess(
            [], 1, "", "expected ruckig==0.19.4, got '0.0.0'"
        ),
    ) as run:
        with pytest.raises(ConfigError, match=r"must provide ruckig==0\.19\.4 for"):
            preflight.validate_joint_trajectory_runtime(plan)
    argv = run.call_args.args[0]
    assert argv[-1] == preflight.REQUIRED_RUCKIG_VERSION


@pytest.mark.parametrize("robot", ("fr3", "fr3_duo"))
def test_fr3_rigs_without_dynamic_joint_ik_run_no_ruckig(tmp_path, robot):
    plan = _plan(tmp_path, robot=robot, physics="kinematic")
    with mock.patch.object(preflight.subprocess, "run") as run:
        preflight.validate_joint_trajectory_runtime(plan)
    run.assert_not_called()


def test_a_rig_without_a_ruckig_rule_is_refused_not_skipped(tmp_path):
    plan = _plan(tmp_path, robot="other_robot")
    with mock.patch.object(preflight.subprocess, "run") as run:
        with pytest.raises(ConfigError, match="not a supported simulation rig"):
            preflight.validate_joint_trajectory_runtime(plan)
    run.assert_not_called()


def test_every_supported_rig_has_a_ruckig_rule(tmp_path):
    from robosuite_env.robots.profiles import SUPPORTED_ROBOT_NAMES

    for robot in SUPPORTED_ROBOT_NAMES:
        plan = _plan(tmp_path / robot, robot=robot)
        assert preflight._ruckig_runtime_requirement(plan["resolved"]), robot


@pytest.mark.skipif(not _pinned_ruckig_installed(), reason="pinned ruckig is absent")
@pytest.mark.parametrize("robot", ("fr3_duo",))
def test_the_real_probe_accepts_the_pinned_ruckig(tmp_path, robot):
    plan = _plan(tmp_path, robot=robot)
    plan["resolved"]["policy"]["python"] = sys.executable
    preflight.validate_joint_trajectory_runtime(plan)


@pytest.mark.parametrize("robot", ("fr3_duo",))
def test_the_real_probe_rejects_a_different_ruckig(tmp_path, monkeypatch, robot):
    plan = _plan(tmp_path / "plan", robot=robot)
    plan["resolved"]["policy"]["python"] = sys.executable
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "ruckig.py").write_text('__version__ = "0.0.0"\n', encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(shadow))

    with pytest.raises(ConfigError, match=r"ruckig==0\.19\.4 for .*got '0\.0\.0'"):
        preflight.validate_joint_trajectory_runtime(plan)
