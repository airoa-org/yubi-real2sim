# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robosuite_env.tools.evaluate_rollout_tree import evaluate_tree, render_markdown


def _write_stage(
    root: Path,
    stage: str,
    *,
    run: str,
    grasp: bool,
    lift_mm: float,
    place: bool,
) -> None:
    stage_root = root / stage
    run_dir = stage_root / "franka_flat" / run
    run_dir.mkdir(parents=True)
    summary = {
        "rule_grasp_events": (
            [{"event": "grasp", "frame": 7, "hand": "R"}] if grasp else []
        ),
        "cup_lift_mm": lift_mm,
        "max_pen_mm": 0.5,
        "frames_with_pen": 0,
        "init_state_in": str(stage_root / "init_state.json"),
        "success_final": {
            "conditions": {
                "cup_plate_contact": place,
                "cup_bottom_contact": place,
            },
            "metrics": {
                "bottom_center_offset_mm": 10.0 if place else 900.0,
                "height_from_plate_bottom_mm": 5.0 if place else 900.0,
                "cup_axis_plate_normal_dot": 0.99 if place else 0.0,
            },
        },
    }
    summary_path = run_dir / "sim_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (stage_root / "outputs.csv").write_text(
        "scene,run,exit_code,video,summary\n"
        f"franka_flat,{run},0,franka_flat/{run}/franka_flat_{run}.mp4,"
        f"franka_flat/{run}/sim_summary.json\n",
        encoding="utf-8",
    )


def test_episode_success_requires_grasp_lift_and_final_place(tmp_path: Path) -> None:
    _write_stage(
        tmp_path, "stage1_initial_b10", run="run_01", grasp=True, lift_mm=80.0, place=True
    )
    _write_stage(
        tmp_path,
        "stage2_from_grasp_b10",
        run="run_01",
        grasp=True,
        lift_mm=10.0,
        place=True,
    )

    result = evaluate_tree(tmp_path)

    assert list(result["stages"]) == ["stage1_initial_b10", "stage2_from_grasp_b10"]
    first = result["stages"]["stage1_initial_b10"]["rows"][0]
    assert first["grasp"] is True
    assert first["first_grasp_frame"] == 7
    assert first["first_grasp_hand"] == "R"
    assert first["grasp_lift_50"] is True
    assert first["place"] is True
    assert first["success"] is True
    # A 10 mm lift is below the 50 mm threshold, so a final-frame place alone
    # must not report episode success.
    second = result["stages"]["stage2_from_grasp_b10"]["rows"][0]
    assert second["place"] is True
    assert second["grasp_lift_50"] is False
    assert second["success"] is False
    assert result["stages"]["stage2_from_grasp_b10"]["counts"]["success"] == 0


def test_the_recorded_resting_verdict_decides_place(tmp_path: Path) -> None:
    # A cup that came to rest on its side: the criterion recorded a success,
    # while the seated recomputation below it says no bottom contact and no
    # upright axis. The recorded verdict is the one that counts.
    stage_root = tmp_path / "stage1_initial_b10"
    run_dir = stage_root / "franka_flat" / "run_01"
    run_dir.mkdir(parents=True)
    summary = {
        "rule_grasp_events": [{"event": "grasp", "frame": 7, "hand": "R"}],
        "cup_lift_mm": 80.0,
        "max_pen_mm": 0.5,
        "frames_with_pen": 0,
        "init_state_in": str(stage_root / "init_state.json"),
        "success_final": {
            "conditions": {
                "cup_resting_on_plate": True,
                "cup_plate_contact": True,
                "cup_bottom_contact": False,
                "cup_center_inside_plate_disc": False,
                "cup_bottom_in_plate_zrange": False,
                "cup_axis_aligned_with_plate_normal": False,
            },
            "metrics": {
                "bottom_center_offset_mm": 56.6,
                "height_from_plate_bottom_mm": 35.5,
                "cup_axis_plate_normal_dot": 0.0,
            },
        },
    }
    (run_dir / "sim_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (stage_root / "outputs.csv").write_text(
        "scene,run,exit_code,video,summary\n"
        "franka_flat,run_01,0,franka_flat/run_01/franka_flat_run_01.mp4,"
        "franka_flat/run_01/sim_summary.json\n",
        encoding="utf-8",
    )

    result = evaluate_tree(tmp_path)

    row = result["stages"]["stage1_initial_b10"]["rows"][0]
    assert row["place"] is True
    assert row["bbox"]["cup_resting_on_plate"] is True
    assert row["success"] is True


def test_bbox_metrics_decide_place_when_conditions_are_absent(tmp_path: Path) -> None:
    _write_stage(
        tmp_path, "stage1_initial_b10", run="run_01", grasp=True, lift_mm=80.0, place=True
    )
    summary_path = tmp_path / "stage1_initial_b10/franka_flat/run_01/sim_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["success_final"]["metrics"]["bottom_center_offset_mm"] = 500.0
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    row = evaluate_tree(tmp_path)["stages"]["stage1_initial_b10"]["rows"][0]

    assert row["bbox"]["cup_center_inside_plate_disc"] is False
    assert row["place"] is False
    assert row["success"] is False


def test_upright_tolerance_controls_the_normal_dot_threshold(tmp_path: Path) -> None:
    _write_stage(
        tmp_path, "stage1_initial_b10", run="run_01", grasp=True, lift_mm=80.0, place=True
    )

    strict = evaluate_tree(tmp_path, upright_tol_deg=1.0)
    tolerant = evaluate_tree(tmp_path, upright_tol_deg=35.0)

    assert strict["stages"]["stage1_initial_b10"]["rows"][0]["place"] is False
    assert tolerant["stages"]["stage1_initial_b10"]["rows"][0]["place"] is True


def test_missing_outputs_csv_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "stage1_initial_b10").mkdir()

    with pytest.raises(RuntimeError, match="no stage directories with outputs.csv"):
        evaluate_tree(tmp_path)


def test_markdown_report_lists_every_stage(tmp_path: Path) -> None:
    _write_stage(
        tmp_path, "stage1_initial_b10", run="run_01", grasp=True, lift_mm=80.0, place=True
    )

    markdown = render_markdown(evaluate_tree(tmp_path))

    assert "# Rollout Tree Evaluation" in markdown
    assert "## stage1_initial_b10" in markdown
    assert "1/1 = 100%" in markdown
