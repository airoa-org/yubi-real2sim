# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import pytest

try:
    from robosuite_env.tools import summarize_policy_rollouts as summary_tool
except ImportError:  # pragma: no cover - matplotlib absent from the CI lock
    summary_tool = None


def _write_run(
    root: Path,
    run: str,
    *,
    grasp: bool,
    held_heights_mm: list[float],
    place: bool = False,
) -> None:
    run_dir = root / "img1239_standard" / run
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True)
    events = [{"event": "grasp", "frame": 1}] if grasp else []
    summary = {
        "new_final_success": place,
        "success": place,
        "rule_grasp_events": events,
        "cup_lift_mm": max(held_heights_mm, default=0.0),
    }
    (run_dir / "sim_summary.json").write_text(json.dumps(summary) + "\n")
    (run_dir / "exit_code").write_text("0\n")
    (run_dir / f"{run}.mp4").touch()
    with (log_dir / "sim_log.jsonl").open("w") as stream:
        stream.write(json.dumps({"frame": 0, "cup_pos": [0, 0, 0.1]}) + "\n")
        for frame, height_mm in enumerate(held_heights_mm, start=1):
            stream.write(
                json.dumps(
                    {
                        "frame": frame,
                        "cup_pos": [0, 0, 0.1 + height_mm / 1000.0],
                        "rule_grasp": {"active": grasp},
                    }
                )
                + "\n"
            )


@pytest.mark.usefixtures("matplotlib")
class SummarizePolicyRolloutsTest(unittest.TestCase):
    def test_stage_score_uses_highest_achieved_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root, "run_01", grasp=False, held_heights_mm=[])
            _write_run(root, "run_02", grasp=True, held_heights_mm=[5.0, 19.9])
            _write_run(root, "run_03", grasp=True, held_heights_mm=[5.0, 20.0])
            _write_run(root, "run_04", grasp=True, held_heights_mm=[4.0], place=True)

            records = summary_tool.load_rollouts(root, lift_threshold_mm=20.0)

            self.assertEqual([record["stage"] for record in records], ["none", "grasp", "lift", "place"])
            self.assertEqual([record["stage_score"] for record in records], [0, 1, 2, 3])
            aggregate = summary_tool.aggregate(records, lift_threshold_mm=20.0)
            scene = aggregate["scenes"]["img1239_standard"]
            self.assertEqual(scene["grasp_count"], 3)
            self.assertEqual(scene["lift_count"], 1)
            self.assertEqual(scene["place_count"], 1)
            self.assertEqual(scene["stage_score_total"], 6)
            self.assertEqual(scene["stage_score_max_total"], 12)

    def test_lift_requires_grasp_even_if_cup_moves_up(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root, "run_01", grasp=False, held_heights_mm=[50.0])

            record = summary_tool.load_rollouts(root, lift_threshold_mm=20.0)[0]

            self.assertFalse(record["lift_ever"])
            self.assertEqual(record["stage_score"], 0)

    def test_missing_frame_log_does_not_fallback_to_global_cup_lift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root, "run_01", grasp=True, held_heights_mm=[50.0])
            (root / "img1239_standard" / "run_01" / "logs" / "sim_log.jsonl").unlink()

            record = summary_tool.load_rollouts(root, lift_threshold_mm=20.0)[0]

            self.assertFalse(record["lift_log_valid"])
            self.assertFalse(record["lift_ever"])
            self.assertEqual(record["stage_score"], 1)

    def test_csv_contains_stage_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root, "run_01", grasp=True, held_heights_mm=[25.0])
            records = summary_tool.load_rollouts(root)
            output = root / "summary.csv"

            summary_tool.write_csv(records, output)

            with output.open(newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["stage"], "lift")
            self.assertEqual(row["stage_score"], "2")

    def test_cup_plate_uses_cup_contract_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root, "run_01", grasp=True, held_heights_mm=[5.0, 25.0])

            record = summary_tool.load_rollouts(root)[0]

            self.assertEqual(record["stage"], "lift")
            self.assertEqual(record["stage_score"], 2)
            self.assertAlmostEqual(record["held_cup_lift_mm"], 25.0)
            self.assertEqual(record["cup_lift_mm"], 25.0)

            output = root / "summary.csv"
            summary_tool.write_csv([record], output)
            with output.open(newline="") as stream:
                reader = csv.DictReader(stream)
                csv_row = next(reader)
                fieldnames = reader.fieldnames or []
            self.assertEqual(csv_row["stage"], "lift")
            self.assertIn("cup_lift_mm", fieldnames)
            self.assertIn("held_cup_lift_mm", fieldnames)
            self.assertIn("cup_displacement_mm", fieldnames)

    def test_run_100_sorts_after_run_99(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for run in ("run_100", "run_11", "run_99", "run_10"):
                _write_run(root, run, grasp=False, held_heights_mm=[])

            records = summary_tool.load_rollouts(root)

            self.assertEqual(
                [record["run"] for record in records],
                ["run_10", "run_11", "run_99", "run_100"],
            )


    def test_two_scene_aggregate_reports_exact_success_rate_comparison(self):
        records = []
        for index in range(10):
            for scene, success in (("pinhole", True), ("panorama", False)):
                records.append(
                    {
                        "scene": scene,
                        "run": f"run_{index + 1:02d}",
                        "new_final_success": success,
                        "success_ever": success,
                    }
                )

        comparison = summary_tool.aggregate(records)[
            "two_scene_success_comparison"
        ]

        self.assertEqual(comparison["scene_order"], ["panorama", "pinhole"])
        self.assertEqual(comparison["success_counts"], [0, 10])
        self.assertEqual(
            comparison["success_rate_difference_second_minus_first"], 1.0
        )
        self.assertEqual(
            comparison["primary_test"]["method"], "fisher_exact_two_sided"
        )
        self.assertLess(comparison["primary_test"]["p_value"], 0.001)
        self.assertTrue(
            comparison["primary_test"]["statistically_significant"]
        )
        self.assertEqual(
            comparison["fisher_exact_two_sided"]["odds_ratio_status"],
            "finite",
        )
        self.assertEqual(
            comparison["fisher_exact_two_sided"]["odds_ratio_first_over_second"],
            0.0,
        )

    def test_fisher_exact_keeps_extreme_hundred_rollout_tail(self):
        expected = 2.0 / math.comb(200, 100)

        actual = summary_tool._fisher_exact_two_sided(0, 100, 100, 100)

        self.assertTrue(math.isclose(actual, expected, rel_tol=1e-12))


if __name__ == "__main__":
    unittest.main()


import unittest
from pathlib import Path

import cv2 as cv
import pytest

try:
    from robosuite_env.tools import plot_sim_trajectories as plot_tool
except ImportError:  # pragma: no cover - matplotlib absent from the CI lock
    plot_tool = None


def _arm_metrics(distance_field: str, distance: float) -> dict:
    return {
        side: {
            "tcp_pos": [0.1, 0.2, 0.3],
            "eef_pos": [0.11, 0.21, 0.31],
            distance_field: distance + index,
        }
        for index, side in enumerate(("R", "L"))
    }


@pytest.mark.usefixtures("matplotlib")
class PlotSimTrajectoriesTest(unittest.TestCase):
    def test_cup_plate_collect_and_outputs_use_cup_contract(self):
        records = []
        for frame, height in enumerate((0.1, 0.125)):
            records.append(
                {
                    "frame": frame,
                    "cup_pos": [0.2, 0.3, height],
                    "plate_pos": [0.4, 0.5, 0.02],
                    "task_objects": {
                        "cup": {"qpos": [0.0] * 7},
                        "plate": {"qpos": [0.0] * 7},
                    },
                    "arms_metric": _arm_metrics(
                        "tcp_cup_dist_mm", 100.0 + frame
                    ),
                    "grip": {"R": 0.5, "L": 0.6},
                    "grip_actual": {"R": 0.51, "L": 0.61},
                    "max_pen_mm": 0.0,
                    "success": {"success": False, "score": {"points": 0}},
                }
            )

        rows = plot_tool._collect(records, {"objects": "cup_plate"})

        self.assertEqual(rows[0]["cup_x"], 0.2)
        self.assertEqual(rows[1]["cup_z"], 0.125)
        self.assertEqual(rows[0]["plate_y"], 0.5)
        self.assertEqual(rows[0]["R_tcp_cup_dist_mm"], 100.0)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "trajectory.csv"
            png_path = root / "trajectory.png"
            tracks_path = root / "trajectory_10tracks.png"
            video_path = root / "trajectory.mp4"
            plot_tool.write_csv(rows, csv_path)
            plot_tool.plot_png(
                rows,
                {
                    "objects": "cup_plate",
                    "frames": 2,
                    "rendered_frames": 2,
                    "min_R_tcp_cup_dist_mm": 100.0,
                },
                png_path,
            )
            plot_tool.plot_10tracks_png(
                rows, {"objects": "cup_plate"}, tracks_path
            )
            plot_tool.render_mp4(rows, video_path, fps=10.0)

            with csv_path.open(newline="") as stream:
                fieldnames = csv.DictReader(stream).fieldnames or []
            self.assertIn("cup_x", fieldnames)
            self.assertIn("plate_x", fieldnames)
            self.assertIn("R_tcp_cup_dist_mm", fieldnames)
            self.assertGreater(png_path.stat().st_size, 0)
            self.assertGreater(tracks_path.stat().st_size, 0)
            capture = cv.VideoCapture(str(video_path))
            try:
                self.assertEqual(int(capture.get(cv.CAP_PROP_FRAME_COUNT)), 2)
            finally:
                capture.release()

    def test_cup_plate_keeps_legacy_cup_fields(self):
        record = {
            "frame": 0,
            "cup_pos": [0.1, 0.2, 0.3],
            "plate_pos": [0.4, 0.5, 0.6],
            "task_objects": {
                "cup": {"qpos": [0.0] * 7},
                "plate": {"qpos": [0.0] * 7},
            },
            "arms_metric": _arm_metrics("tcp_cup_dist_mm", 15.0),
            "grip": {"R": 0.5, "L": 0.6},
            "grip_actual": {"R": 0.51, "L": 0.61},
        }

        row = plot_tool._collect([record], {"objects": "cup_plate"})[0]

        self.assertEqual(row["cup_x"], 0.1)
        self.assertEqual(row["plate_z"], 0.6)
        self.assertEqual(row["R_tcp_cup_dist_mm"], 15.0)


if __name__ == "__main__":
    unittest.main()
