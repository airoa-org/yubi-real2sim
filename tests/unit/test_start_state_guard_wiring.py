# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The start-state check has to run before step 0, and reach the summary.

A guard nobody calls is a guard that does not exist, and a finding must reach
the run summary to be read. These pin the two connections rather than the check itself,
which ``test_start_state_guard.py`` covers.
"""

from __future__ import annotations

import types

import pytest

from real2sim.start_state_guard import Overlap, StartStateReport


def _state(**args):
    defaults = {"start_state_guard": "warn", "start_state_penetration_tol_mm": 2.0}
    defaults.update(args)
    return types.SimpleNamespace(
        A=types.SimpleNamespace(**defaults), m=object(), d=object())


def _patch(monkeypatch, report):
    calls = []

    def fake(model, data, **kwargs):
        calls.append(kwargs)
        return report

    monkeypatch.setattr(
        "robosuite_env.rollout.stages.run_frames.check_start_state", fake)
    return calls


def _clear():
    return StartStateReport(contacts=3)


def _dirty():
    return StartStateReport(
        contacts=4,
        overlaps=[Overlap("container_g", "desk_g", "container_main", "desk", 0.031)],
    )


def test_a_clear_start_state_is_recorded_and_does_not_stop_the_run(monkeypatch, capsys):
    from robosuite_env.rollout.stages import run_frames

    _patch(monkeypatch, _clear())
    state = _state()
    run_frames._check_start_state(state)
    assert state._start_state_report is not None
    assert "clear" in capsys.readouterr().out


def test_warn_mode_prints_the_overlap_and_keeps_going(monkeypatch, capsys):
    from robosuite_env.rollout.stages import run_frames

    _patch(monkeypatch, _dirty())
    state = _state(start_state_guard="warn")
    run_frames._check_start_state(state)          # must not raise
    out = capsys.readouterr().out
    assert "container_main" in out and "31." in out
    assert state._start_state_report.failed


def test_fail_mode_refuses_the_run(monkeypatch):
    from robosuite_env.rollout.stages import run_frames

    _patch(monkeypatch, _dirty())
    with pytest.raises(SystemExit, match="container_main"):
        run_frames._check_start_state(_state(start_state_guard="fail"))


def test_off_skips_the_check_entirely(monkeypatch):
    from robosuite_env.rollout.stages import run_frames

    calls = _patch(monkeypatch, _dirty())
    state = _state(start_state_guard="off")
    run_frames._check_start_state(state)
    assert calls == []
    assert state._start_state_report is None


def test_the_robot_is_excused_against_itself_but_not_against_an_object(monkeypatch):
    """Both sides must be the robot; a link into a cup still has to report."""
    from robosuite_env.rollout.stages import run_frames

    calls = _patch(monkeypatch, _clear())
    run_frames._check_start_state(_state())
    prefixes = calls[0]["ignore_body_prefixes"]
    assert set(prefixes) == {"robot0", "robot1", "gripper0", "gripper1"}


def test_the_tolerance_reaches_the_check_in_metres(monkeypatch):
    from robosuite_env.rollout.stages import run_frames

    calls = _patch(monkeypatch, _clear())
    run_frames._check_start_state(_state(start_state_penetration_tol_mm=0.5))
    assert calls[0]["penetration_tol_m"] == pytest.approx(0.0005)


def test_both_flags_are_on_the_rollout_command_line():
    from robosuite_env.rollout.cli import parse

    default = parse(["--robot", "FR3_DUO", "--grasp-test"])
    assert default.start_state_guard == "warn"
    assert default.start_state_penetration_tol_mm == pytest.approx(2.0)

    chosen = parse([
        "--robot", "FR3_DUO", "--grasp-test",
        "--start-state-guard", "fail",
        "--start-state-penetration-tol-mm", "0.5",
    ])
    assert chosen.start_state_guard == "fail"
    assert chosen.start_state_penetration_tol_mm == pytest.approx(0.5)


@pytest.mark.parametrize("replay", ("state_log_replay", "ik_replay", "replay"))
def test_a_replay_is_not_refused_for_its_recorded_first_frame(monkeypatch, replay):
    """The post-hoc render replays a finished run; the run passed the guard at its start."""
    from robosuite_env.rollout.stages import run_frames

    calls = _patch(monkeypatch, _dirty())
    state = _state(start_state_guard="fail", **{replay: "/runs/run_01/logs/sim_log.jsonl"})
    run_frames._check_start_state(state)          # must not raise
    assert calls == []
