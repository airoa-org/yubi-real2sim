# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Recorded command compatibility through the production replay and worker parsers."""

from pathlib import Path

import pytest

from robosuite_env.rerender.common import RerenderError
from robosuite_env.rerender.replay_argv import build_replay_argv
from robosuite_env.rerender.sources import prepare_source_run
from robosuite_env.rollout.cli import parse
from tests._helpers.rollouts import replayed_worker_arguments, run_dir  # noqa: F401 -- pytest fixture


def replay_command(tmp_path: Path, options: list[str], *, task="cup_plate", summary=None):
    (tmp_path / "logs").mkdir(exist_ok=True)
    # A policy observation records the run's wrist backend; a replay refuses a run that records none.
    (tmp_path / "logs" / "sim_log.jsonl").write_text('{"policy": {"obs_log": {"policy_wrist_backend": "direct_kb4"}}}\n')
    recorded = ["python", "robosuite_env/sim.py", "--objects", task, *options]
    original = recorded.copy()
    replay = build_replay_argv(
        recorded,
        run_dir=tmp_path,
        gs_port=9999,
        gs_token="test",
        render_lock="/tmp/retired-option-test.lock",
        wrist_insets="off",
        source_summary=summary,
    )
    assert recorded == original
    return replay


QUIET_OPTIONS = ("--success-quiet-translation-mm", "--success-quiet-rotation-deg")


@pytest.mark.parametrize("task", ["cup_plate", "box_pack"])
@pytest.mark.parametrize("equals", [False, True])
def test_retired_quiet_options_replay_through_worker_parser(tmp_path, task, equals):
    options = []
    for option in QUIET_OPTIONS:
        options += [option, "1.5"]
        options += [f"{option}=2"] if equals else [option, "2"]
    replay = replay_command(tmp_path, options, task=task)
    parsed = parse(replayed_worker_arguments(replay))
    assert parsed.objects == task
    assert parsed.state_log_replay == str(tmp_path / "logs" / "sim_log.jsonl")
    assert not any(token.partition("=")[0] in QUIET_OPTIONS for token in replay)


@pytest.mark.parametrize("option", QUIET_OPTIONS)
@pytest.mark.parametrize("tail", [[], ["--frames", "1"], [""], ["oops"], ["nan"], ["inf"], ["-inf"]])
def test_retired_quiet_options_reject_missing_or_invalid_values(tmp_path, option, tail):
    with pytest.raises(RerenderError, match="missing its value|finite number"):
        replay_command(tmp_path, [option, *tail])
    assert not (tmp_path / "rerender").exists()


@pytest.mark.parametrize("option", QUIET_OPTIONS)
@pytest.mark.parametrize("value", ["", "oops", "nan", "inf", "-inf"])
def test_retired_quiet_options_validate_even_overridden_values(tmp_path, option, value):
    with pytest.raises(RerenderError, match="finite number"):
        replay_command(tmp_path, [f"{option}={value}", option, "2"])


@pytest.mark.parametrize("option", QUIET_OPTIONS)
def test_retired_quiet_options_remain_unsupported_for_live_worker(option):
    with pytest.raises(SystemExit) as error:
        parse([option, "2"])
    assert error.value.code == 2


def test_recorded_unknown_robot_is_refused_before_any_server_starts(run_dir):
    # A recorded robot outside the supported set gets the generic robot
    # error, and source preflight stops before the 3DGS server or a GPU is
    # touched.
    command_path = run_dir / "RUN_COMMAND.md"
    command_path.write_text(
        command_path.read_text().replace("--robot FR3", "--robot unknown")
    )
    with pytest.raises(RerenderError, match="recorded --robot: unsupported robot 'unknown'"):
        prepare_source_run(run_dir, renderer_code={"commit": "a" * 40, "dirty": True})
    assert not (run_dir / "rerender").exists()
