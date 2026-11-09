# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Preflight re-resolution and batch-server publication of the state quaternion hemisphere.

The "no norm stats" exemption is keyed on the bound checkpoint artifact, not on
``reproducibility.mode``: every ten-rollout release profile is fixture mode yet
binds a real generated checkpoint, so only a placeholder fixture artifact may
run unresolved.
"""

from __future__ import annotations

import json
from pathlib import Path
import types
from unittest import mock

import pytest

from real2sim import preflight
from real2sim.config._types import ConfigError
from real2sim.config.parameters import (
    POLICY_STATE_QUAT_HEMISPHERE_FLAG as FLAG,
    resolve_policy_state_quat_hemisphere_parameter,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "artifacts" / "manifest.yaml"
PLACEHOLDER_ARTIFACT = "tiny-checkpoint-v1"
REAL_CHECKPOINT_ARTIFACT = "pi05-cup"
OPENPI_CONFIG = "pi05_cup"
NORM_STATS_ASSET = "pi05_cup"


def _write_norm_stats(
    checkpoint: Path,
    qw_q01: float,
    qw_q99: float,
    asset: str = NORM_STATS_ASSET,
) -> Path:
    q01 = [0.0] * 32
    q99 = [0.0] * 32
    for index in range(9):
        q01[index], q99[index] = -0.5, 0.5
    q01[6], q99[6] = qw_q01, qw_q99
    path = checkpoint / "assets" / asset / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"norm_stats": {"state": {"q01": q01, "q99": q99}, "actions": {}}}))
    return path


def _real_looking_checkpoint(root: Path) -> Path:
    """A checkpoint tree shaped like a staged one, but without assets/."""

    (root / "params").mkdir(parents=True)
    (root / "params" / "_METADATA").write_text("{}")
    (root / "_CHECKPOINT_METADATA").write_text("{}")
    return root


def _plan(
    checkpoint: Path,
    configured: str = "auto",
    *,
    artifact: str = REAL_CHECKPOINT_ARTIFACT,
    mode: str = "fixture",
) -> dict:
    cli_args, record = resolve_policy_state_quat_hemisphere_parameter(
        {"policy_state_quat_hemisphere": configured},
        ["--physics", "dynamic"],
        checkpoint=checkpoint,
        config_name=OPENPI_CONFIG,
        config_source="deploy",
        action_horizon=32,
        repo_root=REPO_ROOT,
    )
    return {
        "resolved": {
            "reproducibility": {
                "mode": mode,
                "artifact_manifest": str(MANIFEST),
                "artifacts": {"checkpoint": artifact},
            },
            "policy": {
                "checkpoint": str(checkpoint),
                "config_name": OPENPI_CONFIG,
                "config_source": "deploy",
                "action_horizon": 32,
            },
            "simulation": {
                "parameters": {"policy_state_quat_hemisphere": configured},
                "policy_state_quat_hemisphere": record,
                "cli_args": cli_args,
            },
        },
        "repo_root": REPO_ROOT,
    }


# ------------------------------------------------------------------ resolved checkpoints


def test_preflight_accepts_a_consistent_resolution(tmp_path):
    _write_norm_stats(tmp_path, -0.986, -0.150)
    report = preflight.validate_policy_state_quat_hemisphere(_plan(tmp_path))
    assert report["hemisphere"] == "negative"
    assert report["verified_against_norm_stats"] is True


def test_launcher_and_preflight_ignore_an_opposite_hemisphere_decoy(tmp_path):
    selected = _write_norm_stats(tmp_path, -0.986, -0.150)
    _write_norm_stats(tmp_path, 0.499, 0.965, asset="decoy/opposite")

    plan = _plan(tmp_path)
    record = plan["resolved"]["simulation"]["policy_state_quat_hemisphere"]
    assert record["hemisphere"] == "negative"
    assert record["norm_stats_path"] == str(selected)
    assert record["norm_stats_asset_id"] == NORM_STATS_ASSET
    report = preflight.validate_policy_state_quat_hemisphere(plan)
    assert report["hemisphere"] == "negative"
    assert report["norm_stats_path"] == str(selected)
    assert report["norm_stats_asset_id"] == NORM_STATS_ASSET


def test_launcher_refuses_a_decoy_when_the_configured_asset_is_missing(tmp_path):
    _write_norm_stats(tmp_path, 0.499, 0.965, asset="decoy/opposite")

    with pytest.raises(
        ConfigError,
        match="checkpoint contains other norm stats",
    ):
        _plan(tmp_path)


def test_preflight_refuses_norm_stats_changed_since_resolution(tmp_path):
    path = _write_norm_stats(tmp_path, -0.986, -0.150)
    plan = _plan(tmp_path)
    path.write_text(path.read_text().replace("-0.15", "0.97"))
    with pytest.raises(ConfigError, match="disagrees with checkpoint norm stats"):
        preflight.validate_policy_state_quat_hemisphere(plan)


def test_preflight_refuses_same_hemisphere_but_different_norm_stats_bytes(tmp_path):
    path = _write_norm_stats(tmp_path, -0.986, -0.150)
    plan = _plan(tmp_path)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ConfigError, match="changed since the plan was resolved"):
        preflight.validate_policy_state_quat_hemisphere(plan)


def test_preflight_refuses_a_worker_argument_that_drops_the_resolution(tmp_path):
    _write_norm_stats(tmp_path, -0.986, -0.150)
    plan = _plan(tmp_path)
    args = plan["resolved"]["simulation"]["cli_args"]
    del args[args.index(FLAG) : args.index(FLAG) + 2]
    with pytest.raises(ConfigError, match="worker argument"):
        preflight.validate_policy_state_quat_hemisphere(plan)


def test_preflight_refuses_an_explicit_value_the_checkpoint_contradicts(tmp_path):
    _write_norm_stats(tmp_path, -0.986, -0.150)
    with pytest.raises(ConfigError, match="disagrees with checkpoint norm stats"):
        preflight.validate_policy_state_quat_hemisphere(_plan(tmp_path, "positive"))


def test_preflight_refuses_a_plan_without_a_resolution(tmp_path):
    _write_norm_stats(tmp_path, 0.5, 0.9)
    plan = _plan(tmp_path)
    del plan["resolved"]["simulation"]["policy_state_quat_hemisphere"]
    with pytest.raises(ConfigError, match="no valid hemisphere resolution"):
        preflight.validate_policy_state_quat_hemisphere(plan)


# ------------------------------------------------------------------ checkpoints without norm stats


def test_plan_resolution_does_not_need_norm_stats_for_a_placeholder_checkpoint(tmp_path):
    plan = _plan(tmp_path / "placeholder", artifact=PLACEHOLDER_ARTIFACT)
    record = plan["resolved"]["simulation"]["policy_state_quat_hemisphere"]
    assert record["source"] == "unresolved" and record["hemisphere"] is None
    assert FLAG not in plan["resolved"]["simulation"]["cli_args"]


def test_a_placeholder_fixture_checkpoint_artifact_runs_unresolved_unverified(tmp_path):
    for mode in ("fixture", "production"):
        plan = _plan(tmp_path / "placeholder", artifact=PLACEHOLDER_ARTIFACT, mode=mode)
        report = preflight.validate_policy_state_quat_hemisphere(plan)
        assert report["hemisphere"] is None
        assert report["verified_against_norm_stats"] is False
        assert PLACEHOLDER_ARTIFACT in report["reason"]


@pytest.mark.parametrize("mode", ["fixture", "production"])
def test_a_real_checkpoint_artifact_without_norm_stats_is_refused_in_every_mode(tmp_path, mode):
    checkpoint = _real_looking_checkpoint(tmp_path / "checkpoint")
    plan = _plan(checkpoint, artifact=REAL_CHECKPOINT_ARTIFACT, mode=mode)
    assert plan["resolved"]["simulation"]["policy_state_quat_hemisphere"]["source"] == "unresolved"
    with pytest.raises(ConfigError, match="whatever reproducibility.mode says"):
        preflight.validate_policy_state_quat_hemisphere(plan)


def test_norm_stats_removed_after_an_auto_resolution_are_refused(tmp_path):
    checkpoint = _real_looking_checkpoint(tmp_path / "example")
    stats = _write_norm_stats(checkpoint, -0.986, -0.150)
    plan = _plan(checkpoint)
    stats.unlink()
    with pytest.raises(ConfigError, match="cannot re-read the checkpoint norm stats"):
        preflight.validate_policy_state_quat_hemisphere(plan)


@pytest.mark.parametrize(
    "reproducibility",
    [
        None,
        {"mode": "fixture", "artifact_manifest": str(MANIFEST), "artifacts": {}},
        {"mode": "fixture", "artifact_manifest": "/nonexistent/manifest.yaml",
         "artifacts": {"checkpoint": PLACEHOLDER_ARTIFACT}},
        {"mode": "fixture", "artifact_manifest": str(MANIFEST),
         "artifacts": {"checkpoint": "no-such-artifact-v1"}},
    ],
)
def test_an_unbound_or_unreadable_artifact_binding_is_not_a_placeholder(tmp_path, reproducibility):
    plan = _plan(tmp_path / "placeholder", artifact=PLACEHOLDER_ARTIFACT)
    plan["resolved"]["reproducibility"] = reproducibility
    with pytest.raises(ConfigError, match="whatever reproducibility.mode says"):
        preflight.validate_policy_state_quat_hemisphere(plan)


@pytest.mark.parametrize("configured", ["positive", "negative"])
def test_an_explicit_value_without_norm_stats_needs_a_placeholder_artifact(tmp_path, configured):
    placeholder = preflight.validate_policy_state_quat_hemisphere(
        _plan(tmp_path / "placeholder", configured, artifact=PLACEHOLDER_ARTIFACT)
    )
    assert placeholder["verified_against_norm_stats"] is False
    real = _real_looking_checkpoint(tmp_path / "real")
    with pytest.raises(ConfigError, match="whatever reproducibility.mode says"):
        preflight.validate_policy_state_quat_hemisphere(
            _plan(real, configured, artifact=REAL_CHECKPOINT_ARTIFACT)
        )


def test_norm_stats_that_appear_after_an_unresolved_resolution_are_refused(tmp_path):
    checkpoint = tmp_path / "late"
    plan = _plan(checkpoint, artifact=PLACEHOLDER_ARTIFACT)
    _write_norm_stats(checkpoint, -0.986, -0.150)
    with pytest.raises(ConfigError, match="appeared after the plan was resolved"):
        preflight.validate_policy_state_quat_hemisphere(plan)


def test_broken_norm_stats_fail_closed_at_resolution_even_for_a_placeholder(tmp_path):
    checkpoint = tmp_path / "broken"
    _write_norm_stats(checkpoint, -0.986, -0.150).write_text("{not json")
    with pytest.raises(ConfigError, match="auto cannot be resolved"):
        _plan(checkpoint, artifact=PLACEHOLDER_ARTIFACT)


# ------------------------------------------------------------------ server publication


def test_batch_server_ready_file_publishes_the_norm_stats_hemisphere(tmp_path):
    from robosuite_env.policy import policy_batch_server as pbs

    checkpoint = tmp_path / "checkpoint"
    asset_id = "dataset/example"
    norm_stats = _write_norm_stats(checkpoint, -0.986, -0.150, asset=asset_id)
    data_factory = types.SimpleNamespace(
        resolve_norm_stats_asset_id=lambda: asset_id,
    )
    config = types.SimpleNamespace(
        name="example", data=data_factory, assets_dirs=tmp_path / "a",
        model=types.SimpleNamespace(action_horizon=32),
    )
    policy_config = types.ModuleType("openpi.policies.policy_config")
    policy_config.create_trained_policy = mock.Mock(return_value=types.SimpleNamespace())
    policies = types.ModuleType("openpi.policies")
    policies.policy_config = policy_config
    openpi = types.ModuleType("openpi")
    openpi.policies = policies
    deploy = types.ModuleType("robosuite_env.policy.yubi_openpi.config_yubi")
    deploy.get_config = mock.Mock(return_value=config)
    ready_file = tmp_path / "policy_batch_server_ready.json"
    # The fake openpi above is not an installed distribution, so the shared
    # guard is stubbed; its records must reach the ready file unchanged.
    verified = [{"module": "openpi", "distribution": "openpi", "version": "0.1.0"}]
    fake_server = mock.Mock(server_address=("127.0.0.1", 8765))
    with (
        mock.patch.dict("sys.modules", {"openpi": openpi, "openpi.policies": policies,
                                        "openpi.policies.policy_config": policy_config,
                                        "robosuite_env.policy.yubi_openpi.config_yubi": deploy}),
        mock.patch.object(pbs.pbi, "ensure_locked_openpi", return_value=verified) as guard,
        mock.patch.object(pbs.pbi, "policy_sample_num_steps", return_value=10),
        mock.patch.object(pbs, "runtime_determinism_metadata", return_value={}),
        mock.patch.object(pbs, "BatchPolicyTCPServer", return_value=fake_server),
    ):
        assert pbs.main(["--policy", str(checkpoint), "--config", "example", "--batch-size", "1",
                         "--ready-file", str(ready_file)]) == 0
    metadata = json.loads(ready_file.read_text())["metadata"]
    guard.assert_called_once()
    assert metadata["locked_imports"] == verified
    assert metadata["policy_state_quat_hemisphere"] == "negative"
    resolution = metadata["policy_state_quat_hemisphere_resolution"]
    assert resolution["norm_stats_path"] == str(norm_stats)
    assert resolution["norm_stats_sha256"] == metadata["norm_stats_sha256"]
    assert (resolution["qw_q01"], resolution["qw_q99"]) == (-0.986, -0.150)
