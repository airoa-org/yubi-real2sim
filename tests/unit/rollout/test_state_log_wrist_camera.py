# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""A direct ``python -m robosuite_env.sim --state-log-replay`` renders the wrist camera its run recorded."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from real2sim.wrist_profiles import ELP_MEAN_WRIST_RGB_PROFILE, FRANKA_PINHOLE_WRIST_RGB_PROFILE
from robosuite_env.rerender.common import RerenderError
from robosuite_env.rerender.replay_argv import _recorded_gs_kb4_near_cull, _recorded_wrist_backend, _recorded_wrist_profile
from robosuite_env.rollout.replay.recorded_wrist import UNSUPPORTED_RECORDED_WRIST_BACKEND
from robosuite_env.rollout.setup import args as setup_args


class _Stop(Exception):
    pass


def _worker_arguments(monkeypatch, *argv):
    """The worker's arguments after ``setup.args.run`` has resolved them, stopping right after."""

    monkeypatch.setattr("sys.argv", ["robosuite_env/sim.py", *argv])

    def stop(path):
        raise _Stop

    monkeypatch.setattr(setup_args, "redirect_legacy_pose_path", stop)
    state = SimpleNamespace()
    with pytest.raises(_Stop):
        setup_args.run(state)
    return state.A


def _state_log(tmp_path, *, summary=None, obs_log=None, backend="direct_kb4"):
    """A state log whose second row records ``obs_log``; ``backend`` is the policy observation's wrist backend.

    A replay refuses a run that records no backend, so the tests of the other options record
    ``direct_kb4``.
    """

    logs = tmp_path / "logs"
    logs.mkdir()
    row = {"frame": 0}
    if backend is not None:
        obs_log = {"policy_wrist_backend": backend, **(obs_log or {})}
    if obs_log is not None:
        row["policy"] = {"obs_log": obs_log}
    (logs / "sim_log.jsonl").write_text(json.dumps({"frame": 0}) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    if summary is not None:
        (logs / "sim_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return logs / "sim_log.jsonl"


RECORDED = [
    pytest.param(None, None, FRANKA_PINHOLE_WRIST_RGB_PROFILE, id="nothing-recorded"),
    pytest.param({"policy_yubi_render_profile": None}, None, FRANKA_PINHOLE_WRIST_RGB_PROFILE, id="main-summary-without-policy"),
    pytest.param(None, {"policy_yubi_render_profile": FRANKA_PINHOLE_WRIST_RGB_PROFILE}, FRANKA_PINHOLE_WRIST_RGB_PROFILE, id="state-log"),
    pytest.param({"policy_yubi_render_profile": ELP_MEAN_WRIST_RGB_PROFILE}, None, ELP_MEAN_WRIST_RGB_PROFILE, id="summary"),
]


@pytest.mark.parametrize(("summary", "obs_log", "expected"), RECORDED)
def test_a_state_log_replay_without_the_flag_takes_the_recorded_profile(tmp_path, monkeypatch, summary, obs_log, expected):
    sim_log = _state_log(tmp_path, summary=summary, obs_log=obs_log)
    arguments = _worker_arguments(monkeypatch, "--state-log-replay", str(sim_log))
    assert arguments.policy_yubi_render_profile == expected
    # The rerender path resolves the same value from the same recorded outputs.
    assert _recorded_wrist_profile(summary, sim_log) == expected


def test_a_state_log_replay_refuses_a_profile_flag_that_contradicts_the_run(tmp_path, monkeypatch):
    sim_log = _state_log(tmp_path)
    with pytest.raises(SystemExit, match="contradicts the replayed run, which recorded franka-pinhole-wrist-rgb"):
        _worker_arguments(
            monkeypatch, "--state-log-replay", str(sim_log), "--policy-yubi-render-profile", ELP_MEAN_WRIST_RGB_PROFILE
        )


def test_a_state_log_replay_keeps_a_profile_flag_that_matches_the_run(tmp_path, monkeypatch):
    sim_log = _state_log(tmp_path, summary={"policy_yubi_render_profile": ELP_MEAN_WRIST_RGB_PROFILE})
    arguments = _worker_arguments(
        monkeypatch, "--state-log-replay", str(sim_log), f"--policy-yubi-render-profile={ELP_MEAN_WRIST_RGB_PROFILE}"
    )
    assert arguments.policy_yubi_render_profile == ELP_MEAN_WRIST_RGB_PROFILE


def test_a_summary_with_intrinsics_but_no_profile_is_refused(tmp_path, monkeypatch):
    sim_log = _state_log(tmp_path, summary={"policy_yubi_render_profile": None, "policy_wrist_intrinsics": None})
    with pytest.raises(SystemExit, match="wrist camera is unknown"):
        _worker_arguments(monkeypatch, "--state-log-replay", str(sim_log))


def test_a_run_that_replays_nothing_keeps_the_default(monkeypatch):
    assert _worker_arguments(monkeypatch, "--grasp-test").policy_yubi_render_profile == ELP_MEAN_WRIST_RGB_PROFILE


RECORDED_BACKENDS = [
    pytest.param(None, {"policy_wrist_backend": "direct_kb4"}, "direct_kb4", id="state-log"),
    pytest.param({"policy_wrist_backend": "direct_kb4"}, None, "direct_kb4", id="summary-direct"),
]


@pytest.mark.parametrize(("summary", "obs_log", "expected"), RECORDED_BACKENDS)
def test_a_state_log_replay_without_the_flag_takes_the_recorded_backend(tmp_path, monkeypatch, summary, obs_log, expected):
    sim_log = _state_log(tmp_path, summary=summary, obs_log=obs_log, backend=None)
    arguments = _worker_arguments(monkeypatch, "--state-log-replay", str(sim_log))
    assert arguments.policy_wrist_backend == expected
    # The rerender path resolves the same value from the same recorded outputs.
    assert _recorded_wrist_backend(summary, sim_log) == expected


UNSUPPORTED_BACKEND_RUNS = [
    pytest.param(None, None, id="nothing-recorded"),
    pytest.param({"policy_yubi_render_profile": None}, None, id="summary-without-the-backend"),
    pytest.param({"policy_wrist_backend": "retired_backend"}, None, id="summary-unsupported"),
    pytest.param(None, {"policy_wrist_backend": "retired_backend"}, id="state-log-unsupported"),
]


@pytest.mark.parametrize(("summary", "obs_log"), UNSUPPORTED_BACKEND_RUNS)
@pytest.mark.parametrize("flag", ((), ("--policy-wrist-backend", "direct_kb4")))
def test_a_state_log_replay_of_a_run_without_a_supported_backend_is_refused(tmp_path, monkeypatch, summary, obs_log, flag):
    # The replay neither guesses the run's backend nor switches it to direct_kb4.
    sim_log = _state_log(tmp_path, summary=summary, obs_log=obs_log, backend=None)
    with pytest.raises(SystemExit, match=re.escape(f"[sim] --state-log-replay: {UNSUPPORTED_RECORDED_WRIST_BACKEND}")):
        _worker_arguments(monkeypatch, "--state-log-replay", str(sim_log), *flag)
    with pytest.raises(RerenderError, match=re.escape(UNSUPPORTED_RECORDED_WRIST_BACKEND)):
        _recorded_wrist_backend(summary, sim_log)


def test_a_state_log_replay_rejects_an_unknown_backend_flag(tmp_path, monkeypatch, capsys):
    sim_log = _state_log(tmp_path)
    with pytest.raises(SystemExit):
        _worker_arguments(monkeypatch, "--state-log-replay", str(sim_log), "--policy-wrist-backend", "retired_backend")
    assert "argument --policy-wrist-backend: invalid choice: 'retired_backend'" in capsys.readouterr().err


def test_a_state_log_replay_keeps_a_backend_flag_that_matches_the_run(tmp_path, monkeypatch):
    sim_log = _state_log(tmp_path, summary={"policy_wrist_backend": "direct_kb4"})
    arguments = _worker_arguments(monkeypatch, "--state-log-replay", str(sim_log), "--policy-wrist-backend=direct_kb4")
    assert arguments.policy_wrist_backend == "direct_kb4"


def test_a_summary_with_a_null_backend_is_refused(tmp_path, monkeypatch):
    sim_log = _state_log(tmp_path, summary={"policy_yubi_render_profile": None, "policy_wrist_backend": None}, backend=None)
    with pytest.raises(SystemExit, match="null policy_wrist_backend"):
        _worker_arguments(monkeypatch, "--state-log-replay", str(sim_log))


def test_a_run_that_replays_nothing_keeps_the_default_backend(monkeypatch):
    assert _worker_arguments(monkeypatch, "--grasp-test").policy_wrist_backend == "direct_kb4"


RECORDED_NEAR_CULLS = [
    pytest.param(None, None, "legacy", id="nothing-recorded"),
    pytest.param({"policy_wrist_backend": "direct_kb4"}, None, "legacy", id="summary-before-the-option"),
    pytest.param(None, {"gs_kb4_near_cull": "znear"}, "znear", id="state-log"),
    pytest.param({"gs_kb4_near_cull": "znear"}, None, "znear", id="summary-znear"),
    pytest.param({"gs_kb4_near_cull": "legacy"}, None, "legacy", id="summary-legacy"),
]


@pytest.mark.parametrize(("summary", "obs_log", "expected"), RECORDED_NEAR_CULLS)
def test_a_state_log_replay_without_the_flag_takes_the_recorded_near_cull(tmp_path, monkeypatch, summary, obs_log, expected):
    # Every run recorded before --gs-kb4-near-cull rendered with GaussianWrapping's fixed 0.2 GS-unit cull.
    sim_log = _state_log(tmp_path, summary=summary, obs_log=obs_log)
    arguments = _worker_arguments(monkeypatch, "--state-log-replay", str(sim_log))
    assert arguments.gs_kb4_near_cull == expected
    assert _recorded_gs_kb4_near_cull(summary, sim_log) == expected


def test_a_state_log_replay_refuses_a_near_cull_flag_that_contradicts_the_run(tmp_path, monkeypatch):
    sim_log = _state_log(tmp_path)
    with pytest.raises(SystemExit, match="--gs-kb4-near-cull znear contradicts the replayed run, which recorded legacy"):
        _worker_arguments(monkeypatch, "--state-log-replay", str(sim_log), "--gs-kb4-near-cull", "znear")


def test_a_summary_with_a_null_near_cull_is_refused(tmp_path, monkeypatch):
    sim_log = _state_log(tmp_path, summary={"gs_kb4_near_cull": None})
    with pytest.raises(SystemExit, match="null gs_kb4_near_cull"):
        _worker_arguments(monkeypatch, "--state-log-replay", str(sim_log))


def test_a_new_run_defaults_to_the_metric_znear_cull(monkeypatch):
    assert _worker_arguments(monkeypatch, "--grasp-test").gs_kb4_near_cull == "znear"
