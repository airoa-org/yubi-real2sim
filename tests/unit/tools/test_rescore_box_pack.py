# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Re-scoring a finished run from its state log, instead of running it again."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from robosuite_env.tools.rescore_box_pack import rescore


FRAME_HZ = 10.0
REQUIRED_SAMPLES = 100


def _row(frame, *, position, placed=True, clear=True, unwelded=True,
         hand=True, box_on_table=True, counter=0, velocity_quiet=False,
         success=False):
    return {
        "frame": frame,
        "cup_pos": list(position),
        "cup_quat": [1.0, 0.0, 0.0, 0.0],
        "success": {
            "success": success,
            "conditions": {
                "object_placed": placed,
                "object_clear_of_robot": clear,
                "object_rule_grasp_inactive": unwelded,
                "object_velocity_quiet": velocity_quiet,
                "moved_only_by_named_hand": hand,
                "box_on_table": box_on_table,
            },
            "metrics": {
                "stability": {
                    "consecutive_samples": counter,
                    "required_samples": REQUIRED_SAMPLES,
                }
            },
        },
    }


def _run(tmp_path, rows, name="run_01"):
    run_dir = tmp_path / name
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "sim_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (run_dir / "sim_summary.json").write_text(
        json.dumps({"frame_hz_effective": FRAME_HZ}), encoding="utf-8"
    )
    return run_dir


def _still(frames, *, drift_mm_per_frame=0.0, **kwargs):
    return [
        _row(
            frame,
            position=(0.0, 0.0, 0.05 + frame * drift_mm_per_frame / 1000.0),
            **kwargs,
        )
        for frame in range(frames)
    ]


def test_the_old_rule_is_replayed_from_what_the_run_recorded(tmp_path):
    # The counter reaching its mark is the whole of the old rest test, and the
    # run wrote the counter down, so this needs nothing recomputed.
    rows = _still(40)
    rows[-1] = _row(
        39, position=(0.0, 0.0, 0.05), counter=REQUIRED_SAMPLES, success=True
    )
    report = rescore(_run(tmp_path, rows), rule="old")

    assert report["reproduces_the_recorded_verdict"] is True
    assert report["success"] is True and report["recorded_success"] is True


def test_a_disagreement_with_the_recorded_verdict_is_reported(tmp_path):
    # If the log cannot reproduce what the run decided, the tool has to say so
    # rather than quietly re-score: the run is then the only authority.
    rows = _still(5)
    rows[-1] = _row(4, position=(0.0, 0.0, 0.05), counter=REQUIRED_SAMPLES)
    report = rescore(_run(tmp_path, rows), rule="old")

    assert report["reproduces_the_recorded_verdict"] is False
    assert report["disagreeing_frames"] == [4]


def test_the_new_rule_does_not_inherit_the_old_counter(tmp_path):
    # The rest test is recomputed from positions; the recorded counter is not
    # reused. That counter only advanced on frames the old speed test passed,
    # so it stays near zero through a perfectly still episode and would score
    # every such run a failure.
    rows = _still(40, counter=2)
    report = rescore(_run(tmp_path, rows), rule="new")

    assert report["success"] is True
    assert report["final_window_translation_mm"] == pytest.approx(0.0)


def test_an_object_that_keeps_creeping_is_not_still(tmp_path):
    rows = _still(40, drift_mm_per_frame=0.5, counter=2)  # 5 mm per window
    report = rescore(_run(tmp_path, rows), rule="new")

    assert report["success"] is False
    assert report["final_window_translation_mm"] > 2.0


def test_the_dwell_restarts_when_the_robot_comes_back(tmp_path):
    # Two seconds of stillness are two seconds since the last disturbance, not
    # two seconds since the object was last in place.
    rows = _still(25, counter=2)
    rows += [_row(25, position=(0.0, 0.0, 0.05), clear=False, counter=0)]
    rows += [
        _row(frame, position=(0.0, 0.0, 0.05), counter=2)
        for frame in range(26, 36)
    ]
    report = rescore(_run(tmp_path, rows), rule="new")

    assert report["success"] is False  # only 1.0 s since the hand let go
    assert report["success_ever"] is True  # it had settled before the nudge


def test_settling_and_then_creeping_is_a_failure_at_the_last_frame(tmp_path):
    # What the batch counts is the last frame (new_final_success), and a run
    # that settles and then slides away is not a packed object at the end.
    rows = _still(40, counter=2)
    rows += [
        _row(frame, position=(0.0, 0.0, 0.05 + (frame - 39) * 0.001), counter=2)
        for frame in range(40, 60)
    ]
    report = rescore(_run(tmp_path, rows), rule="new")

    assert report["success"] is False
    assert report["success_ever"] is True
    assert report["first_success_frame"] is not None


def test_a_batch_directory_is_expanded_to_its_runs(tmp_path):
    from robosuite_env.tools.rescore_box_pack import _run_dirs

    batch = tmp_path / "batch" / "franka_flat"
    for name in ("run_01", "run_02"):
        _run(batch, _still(3), name=name)
    assert [path.name for path in _run_dirs([str(tmp_path / "batch")])] == [
        "run_01",
        "run_02",
    ]


def _cli(*arguments):
    return subprocess.run(
        [sys.executable, "-m", "robosuite_env.tools.rescore_box_pack", *map(str, arguments)],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("recorded_success", [False, True])
def test_cli_verify_returns_the_recorded_verdict_gate(tmp_path, recorded_success):
    rows = _still(40, counter=2)
    rows[-1] = _row(
        39, position=(0.0, 0.0, 0.05),
        counter=REQUIRED_SAMPLES, success=recorded_success,
    )
    run = _run(tmp_path, rows)
    output = tmp_path / "report.json"

    result = _cli(run, "--verify", "--json", output)

    assert result.returncode == (0 if recorded_success else 1), result.stderr
    report, = json.loads(output.read_text())
    assert report["rule"] == "old"
    assert report["success"] is True
    assert report["recorded_success"] is recorded_success
    assert report["reproduces_the_recorded_verdict"] is recorded_success
    assert report["disagreeing_frames"] == ([] if recorded_success else [39])


def test_cli_rescores_finished_runs_and_requires_opt_in_for_unfinished(tmp_path):
    batch = tmp_path / "batch"
    _run(batch, _still(40, counter=2), name="run_still")
    _run(batch, _still(40, drift_mm_per_frame=0.5, counter=2), name="run_creeping")
    unfinished = _run(batch, _still(40, counter=2), name="run_unfinished")
    (unfinished / "sim_summary.json").unlink()
    output = tmp_path / "report.json"

    result = _cli(batch, "--json", output)

    assert result.returncode == 0, result.stderr
    reports = {Path(report["run"]).name: report for report in json.loads(output.read_text())}
    assert set(reports) == {"run_still", "run_creeping"}
    assert reports["run_still"]["rule"] == "new"
    assert reports["run_still"]["success"] is True
    assert reports["run_creeping"]["success"] is False
    assert reports["run_creeping"]["final_window_translation_mm"] > 2.0
    assert "1 of 3 runs never finished" in result.stdout

    rejected = _cli(unfinished)
    assert rejected.returncode == 1, rejected.stderr
    assert "nothing left to score" in rejected.stdout
    accepted = _cli(unfinished, "--include-unfinished", "--json", output)
    assert accepted.returncode == 0, accepted.stderr
    report, = json.loads(output.read_text())
    assert report["completed"] is False
    assert report["success"] is True
