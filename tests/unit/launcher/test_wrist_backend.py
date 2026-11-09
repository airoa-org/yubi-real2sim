# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml

from real2sim.config.keys import _V2_OPTIONAL_PARAMETER_DEFAULTS
from real2sim.config.schema import load_rollout_config
from real2sim.config._types import ConfigError
from real2sim.runner.environment import build_runner_environment
from real2sim.runner.argv import _official_static_sim_argv
from robosuite_env.rollout.cli import parse
from scene.gs_background_protocol import DEFAULT_KB4_NEAR_CULL
from tests._helpers.launcher import base_config


@pytest.mark.parametrize("backend", (None, "direct_kb4"))
@pytest.mark.parametrize("profile", (None, "elp-mean-wrist-rgb", "franka-pinhole-wrist-rgb"))
def test_backend_reaches_worker_from_typed_config(tmp_path, backend, profile):
    config = base_config()
    if backend is not None:
        config["simulation"]["parameters"]["policy_wrist_backend"] = backend
    if profile is not None:
        config["simulation"]["parameters"]["policy_yubi_render_profile"] = profile
    directory = tmp_path / "config"
    directory.mkdir()
    path = directory / "rollout.yaml"
    path.write_text(yaml.safe_dump(config))
    plan = load_rollout_config(path, repo_root=tmp_path, timestamp="20261027_024000")
    simulation = plan["resolved"]["simulation"]
    selected = backend or "direct_kb4"
    assert simulation["parameters"]["policy_wrist_backend"] == selected
    worker = parse(_official_static_sim_argv(build_runner_environment(plan)))
    assert worker.policy_wrist_backend == selected
    assert worker.policy_yubi_render_profile == (profile or "elp-mean-wrist-rgb")
    cli = simulation["cli_args"]
    assert cli[cli.index("--policy-wrist-backend") + 1] == selected


def test_launcher_and_worker_share_the_default_backend():
    assert _V2_OPTIONAL_PARAMETER_DEFAULTS["policy_wrist_backend"] == parse(["--replay", "episode.parquet"]).policy_wrist_backend == "direct_kb4"


def test_backend_config_rejects_unknown_projection(tmp_path):
    config = base_config()
    config["simulation"]["parameters"]["policy_wrist_backend"] = "fisheye_fallback"
    directory = tmp_path / "config"
    directory.mkdir()
    path = directory / "rollout.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ConfigError, match="policy_wrist_backend"):
        load_rollout_config(path, repo_root=tmp_path)


def test_launcher_always_passes_the_kb4_near_cull(tmp_path):
    # A rerender takes the mode from the recorded worker argv first, so a run that stops before it
    # records gs_kb4_near_cull still replays with the cull it rendered with, not the legacy fallback.
    config = base_config()
    directory = tmp_path / "config"
    directory.mkdir()
    path = directory / "rollout.yaml"
    path.write_text(yaml.safe_dump(config))
    plan = load_rollout_config(path, repo_root=tmp_path, timestamp="20261027_024000")
    cli = plan["resolved"]["simulation"]["cli_args"]
    assert [token for token in cli if token.partition("=")[0] == "--gs-kb4-near-cull"] == ["--gs-kb4-near-cull"]
    assert cli[cli.index("--gs-kb4-near-cull") + 1] == DEFAULT_KB4_NEAR_CULL == "znear"
    argv = _official_static_sim_argv(build_runner_environment(plan))
    assert argv[argv.index("--gs-kb4-near-cull") + 1] == "znear"
    assert parse(argv).gs_kb4_near_cull == parse(["--replay", "episode.parquet"]).gs_kb4_near_cull == "znear"


@pytest.mark.parametrize("flag", (["--policy-wrist-backend", "fisheye_fallback"], ["--policy-wrist-backend=fisheye_fallback"]))
def test_worker_rejects_an_unknown_backend(capsys, flag):
    with pytest.raises(SystemExit) as raised:
        parse(["--replay", "episode.parquet", *flag])
    assert raised.value.code == 2
    assert "argument --policy-wrist-backend: invalid choice: 'fisheye_fallback'" in capsys.readouterr().err
