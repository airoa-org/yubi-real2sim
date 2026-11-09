# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

from real2sim.config._types import ConfigError
from real2sim.config.schema import load_rollout_config
from real2sim.runner.argv import _official_static_sim_argv
from real2sim.runner.environment import build_runner_environment
from real2sim.wrist_profiles import (
    DEFAULT_POLICY_WRIST_RENDER_PROFILE,
    ELP_MEAN_WRIST_RGB_PROFILE,
    FRANKA_PINHOLE_WRIST_RGB_PROFILE,
    WRIST_RENDER_PROFILE_NAMES,
)
from robosuite_env.rendering import wrist_render_profiles
from robosuite_env.rendering.wrist_render_profiles import get_wrist_render_profile
from robosuite_env.rollout.cli import parse
from tests._helpers.launcher import base_config


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("profile", (None, ELP_MEAN_WRIST_RGB_PROFILE, FRANKA_PINHOLE_WRIST_RGB_PROFILE))
def test_profile_reaches_official_worker_from_yaml(tmp_path, profile):
    config = base_config()
    if profile is not None:
        config["simulation"]["parameters"]["policy_yubi_render_profile"] = profile
    directory = tmp_path / "config"
    directory.mkdir()
    path = directory / "rollout.yaml"
    path.write_text(yaml.safe_dump(config))
    plan = load_rollout_config(path, repo_root=tmp_path, timestamp="20261027_063000")
    selected = profile or ELP_MEAN_WRIST_RGB_PROFILE
    assert plan["resolved"]["simulation"]["parameters"]["policy_yubi_render_profile"] == selected
    worker = parse(_official_static_sim_argv(build_runner_environment(plan)))
    assert worker.policy_yubi_render_profile == selected


def test_mean_is_the_renderer_and_worker_default():
    assert DEFAULT_POLICY_WRIST_RENDER_PROFILE == ELP_MEAN_WRIST_RGB_PROFILE
    profile = get_wrist_render_profile()
    assert profile.name == ELP_MEAN_WRIST_RGB_PROFILE
    assert profile.source_image_hw == (480, 640)
    assert profile.camera.fx == profile.camera.fy == 270.47
    assert (profile.camera.cx, profile.camera.cy) == (320.5, 242.5)
    assert parse(["--replay", "episode.parquet"]).policy_yubi_render_profile == profile.name


def test_profile_yaml_rejects_unknown_name(tmp_path):
    config = base_config()
    config["simulation"]["parameters"]["policy_yubi_render_profile"] = "unknown"
    path = tmp_path / "rollout.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ConfigError, match="policy_yubi_render_profile"):
        load_rollout_config(path, repo_root=tmp_path)


def test_launcher_names_every_renderer_profile_in_order():
    assert WRIST_RENDER_PROFILE_NAMES == tuple(wrist_render_profiles._PROFILES)
