#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Summarize FR3/YUBI rollout results using new_final_success."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_FIELDS = [
    "scene",
    "run",
    "exit_code",
    "new_final_success",
    "success",
    "success_ever",
    "strict_success_ever",
    "partial_success_ever",
    "score_final",
    "score_best",
    "score_total",
    "score_max_total",
    "score_mean",
    "score_condition",
    "max_pen_mm",
    "frames_with_pen",
    "cup_lift_mm",
    "held_cup_lift_mm",
    "lift_log_valid",
    "cup_displacement_mm",
    "grasp_ever",
    "lift_ever",
    "place_success",
    "stage",
    "stage_score",
    "loop_sec",
    "frames",
    "rendered_frames",
    "deterministic_seed",
    "policy_batch_client_id",
    "wall_stopped_early",
    "video",
    "summary_path",
]

DEFAULT_LIFT_THRESHOLD_MM = 20.0


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _num(value: Any) -> float:
    if value in (None, "", "nan"):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _read_outputs_csv(run_root: Path) -> list[dict[str, str]]:
    outputs = run_root / "outputs.csv"
    if not outputs.exists():
        return []
    with outputs.open(newline="") as f:
        return list(csv.DictReader(f))


def _discover_rows(run_root: Path) -> list[dict[str, str]]:
    rows = _read_outputs_csv(run_root)
    if rows:
        return rows
    discovered: list[dict[str, str]] = []
    for summary_path in sorted(run_root.glob("*/run_*/sim_summary.json")):
        run_dir = summary_path.parent
        scene_dir = run_dir.parent
        video = next(run_dir.glob("*.mp4"), None)
        exit_path = run_dir / "exit_code"
        discovered.append(
            {
                "scene": scene_dir.name,
                "run": run_dir.name,
                "exit_code": exit_path.read_text().strip() if exit_path.exists() else "",
                "video": str(video) if video else "",
                "summary": str(summary_path),
            }
        )
    return discovered


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"summary_read_error": str(exc)}


def _grasp_ever(summary: dict[str, Any]) -> bool:
    return any(
        isinstance(event, dict) and event.get("event") == "grasp"
        for event in (summary.get("rule_grasp_events") or [])
    )


def _held_object_lift_mm(
    summary_path: Path,
    summary: dict[str, Any],
) -> tuple[float | None, bool]:
    """Return the peak held cup lift and whether the log was usable."""

    log_path = summary_path.parent / "logs" / "sim_log.jsonl"
    if not log_path.exists():
        return None, False

    records: list[dict[str, Any]] = []
    with log_path.open() as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)

    position_field = "cup_pos"

    initial_z = None
    max_held_z = None
    valid_records = 0
    for record in records:
        try:
            position = record.get(position_field)
            z = float(position[2])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        valid_records += 1
        if initial_z is None:
            initial_z = z
        grasp = record.get("rule_grasp") or {}
        if grasp.get("active"):
            max_held_z = z if max_held_z is None else max(max_held_z, z)

    if initial_z is None or valid_records == 0:
        return None, False
    if max_held_z is None:
        return None, True
    return (max_held_z - initial_z) * 1000.0, True


def load_rollouts(
    run_root: Path,
    *,
    lift_threshold_mm: float = DEFAULT_LIFT_THRESHOLD_MM,
) -> list[dict[str, Any]]:
    rows = _discover_rows(run_root)
    out: list[dict[str, Any]] = []
    for row in rows:
        summary_path = Path(row.get("summary") or row.get("summary_path") or "")
        summary = _load_summary(summary_path) if str(summary_path) else {}
        official_success = summary.get("new_final_success", summary.get("success", False))
        grasp_ever = _grasp_ever(summary)
        held_object_lift_mm, lift_log_valid = _held_object_lift_mm(
            summary_path,
            summary,
        )
        lift_ever = bool(
            grasp_ever
            and held_object_lift_mm is not None
            and held_object_lift_mm >= lift_threshold_mm
        )
        place_success = bool(official_success)
        if place_success:
            stage, stage_score = "place", 3
        elif lift_ever:
            stage, stage_score = "lift", 2
        elif grasp_ever:
            stage, stage_score = "grasp", 1
        else:
            stage, stage_score = "none", 0
        scene = row.get("scene") or row.get("background") or ""
        record = {
            "scene": scene,
            "run": row.get("run", ""),
            "exit_code": row.get("exit_code", ""),
            "new_final_success": bool(official_success),
            "success": bool(summary.get("success", official_success)),
            "success_ever": bool(summary.get("success_ever", False)),
            "strict_success_ever": bool(summary.get("strict_success_ever", False)),
            "partial_success_ever": bool(summary.get("partial_success_ever", False)),
            "score_final": summary.get("score_final"),
            "score_best": summary.get("score_best"),
            "score_total": summary.get("score_total"),
            "score_max_total": summary.get("score_max_total"),
            "score_mean": summary.get("score_mean"),
            "score_condition": summary.get("score_condition"),
            "max_pen_mm": summary.get("max_pen_mm"),
            "frames_with_pen": summary.get("frames_with_pen"),
            "lift_log_valid": lift_log_valid,
            "grasp_ever": grasp_ever,
            "lift_ever": lift_ever,
            "place_success": place_success,
            "stage": stage,
            "stage_score": stage_score,
            "loop_sec": summary.get("loop_sec"),
            "frames": summary.get("frames"),
            "rendered_frames": summary.get("rendered_frames"),
            "deterministic_seed": summary.get("deterministic_seed"),
            "policy_batch_client_id": summary.get("policy_batch_client_id"),
            "wall_stopped_early": bool(summary.get("wall_stopped_early", False)),
            "video": row.get("video", ""),
            "summary_path": str(summary_path),
            "cup_lift_mm": summary.get("cup_lift_mm"),
            "held_cup_lift_mm": held_object_lift_mm,
            "cup_displacement_mm": summary.get("cup_displacement_mm"),
        }
        out.append(record)
    def sort_key(record: dict[str, Any]) -> tuple[str, int, str]:
        run = str(record["run"])
        match = re.fullmatch(r"run_(\d+)", run)
        index = int(match.group(1)) if match is not None else 2**63 - 1
        return str(record["scene"]), index, run

    return sorted(out, key=sort_key)


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = SUMMARY_FIELDS
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def aggregate(
    records: list[dict[str, Any]],
    *,
    lift_threshold_mm: float = DEFAULT_LIFT_THRESHOLD_MM,
) -> dict[str, Any]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_scene[str(record["scene"])].append(record)

    scene_summaries = {}
    for scene, rows in sorted(by_scene.items()):
        successes = sum(1 for row in rows if _truthy(row["new_final_success"]))
        success_ever = sum(1 for row in rows if _truthy(row["success_ever"]))
        score_best = [_num(row.get("score_best")) for row in rows]
        score_final = [_num(row.get("score_final")) for row in rows]
        score_total = [_num(row.get("score_total")) for row in rows]
        grasp_count = sum(1 for row in rows if _truthy(row.get("grasp_ever")))
        lift_count = sum(1 for row in rows if _truthy(row.get("lift_ever")))
        place_count = sum(1 for row in rows if _truthy(row.get("place_success")))
        stage_scores = [_num(row.get("stage_score")) for row in rows]
        scene_summaries[scene] = {
            "runs": len(rows),
            "new_final_success_count": successes,
            "new_final_success_rate": successes / len(rows) if rows else 0.0,
            "success_ever_count": success_ever,
            "success_ever_rate": success_ever / len(rows) if rows else 0.0,
            "score_best_mean": _nanmean(score_best),
            "score_best_max": _nanmax(score_best),
            "score_final_mean": _nanmean(score_final),
            "score_total_sum": _nansum(score_total),
            "grasp_count": grasp_count,
            "grasp_rate": grasp_count / len(rows) if rows else 0.0,
            "lift_count": lift_count,
            "lift_rate": lift_count / len(rows) if rows else 0.0,
            "place_count": place_count,
            "place_rate": place_count / len(rows) if rows else 0.0,
            "stage_score_total": int(_nansum(stage_scores)),
            "stage_score_mean": _nanmean(stage_scores),
            "stage_score_max_total": 3 * len(rows),
        }
    total_runs = len(records)
    total_success = sum(1 for row in records if _truthy(row["new_final_success"]))
    success_comparison = _two_scene_success_comparison(records)
    return {
        "official_success_field": "new_final_success",
        "stage_score_definition": {"grasp": 1, "lift": 2, "place": 3},
        "lift_threshold_mm": lift_threshold_mm,
        "lift_requires_active_rule_grasp": True,
        "total_runs": total_runs,
        "new_final_success_count": total_success,
        "new_final_success_rate": total_success / total_runs if total_runs else 0.0,
        "two_scene_success_comparison": success_comparison,
        "scenes": scene_summaries,
    }


def _fisher_exact_two_sided(
    first_successes: int,
    first_total: int,
    second_successes: int,
    second_total: int,
) -> float:
    """Return the two-sided Fisher exact p-value for two success counts."""

    total_successes = first_successes + second_successes
    total = first_total + second_total
    denominator = math.comb(total, total_successes)

    def probability(first_count: int) -> float:
        return (
            math.comb(first_total, first_count)
            * math.comb(second_total, total_successes - first_count)
            / denominator
        )

    observed = probability(first_successes)
    lower = max(0, total_successes - second_total)
    upper = min(first_total, total_successes)
    # This matches the common probability-ordering definition used by
    # scipy.stats.fisher_exact for a two-sided alternative.
    probabilities = [probability(count) for count in range(lower, upper + 1)]
    # Use a relative comparison tolerance.  A fixed absolute epsilon would
    # include much more likely tables when the observed probability is tiny
    # (for example 0/100 versus 100/100).
    cutoff = observed * (1.0 + 1e-12)
    return min(1.0, sum(value for value in probabilities if value <= cutoff))


def _two_scene_success_comparison(
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Compare the success proportions for exactly two scenes."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["scene"])].append(record)
    if len(grouped) != 2:
        return None

    first_scene, second_scene = sorted(grouped)
    first_rows = grouped[first_scene]
    second_rows = grouped[second_scene]
    first_successes = sum(
        1 for row in first_rows if _truthy(row["new_final_success"])
    )
    second_successes = sum(
        1 for row in second_rows if _truthy(row["new_final_success"])
    )
    first_failures = len(first_rows) - first_successes
    second_failures = len(second_rows) - second_successes
    fisher_p = _fisher_exact_two_sided(
        first_successes,
        len(first_rows),
        second_successes,
        len(second_rows),
    )
    odds_denominator = first_failures * second_successes
    odds_numerator = first_successes * second_failures
    if odds_denominator:
        odds_ratio: float | None = odds_numerator / odds_denominator
    else:
        odds_ratio = None
    return {
        "scene_order": [first_scene, second_scene],
        "success_counts": [first_successes, second_successes],
        "run_counts": [len(first_rows), len(second_rows)],
        "success_rate_difference_second_minus_first": (
            second_successes / len(second_rows)
            - first_successes / len(first_rows)
        ),
        "fisher_exact_two_sided": {
            "p_value": fisher_p,
            "odds_ratio_first_over_second": odds_ratio,
            "odds_ratio_status": (
                "finite"
                if odds_denominator
                else "infinite" if odds_numerator else "undefined"
            ),
        },
        "primary_test": {
            "method": "fisher_exact_two_sided",
            "p_value": fisher_p,
            "alpha": 0.05,
            "statistically_significant": fisher_p < 0.05,
        },
    }


def _finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _nanmean(values: list[float]) -> float | None:
    vals = _finite(values)
    return sum(vals) / len(vals) if vals else None


def _nanmax(values: list[float]) -> float | None:
    vals = _finite(values)
    return max(vals) if vals else None


def _nansum(values: list[float]) -> float:
    return sum(_finite(values))


def plot_summary(summary: dict[str, Any], path: Path) -> None:
    scenes = list(summary["scenes"].keys())
    rates = [summary["scenes"][scene]["new_final_success_rate"] for scene in scenes]
    grasp_rates = [summary["scenes"][scene]["grasp_rate"] for scene in scenes]
    lift_rates = [summary["scenes"][scene]["lift_rate"] for scene in scenes]
    stage_scores = [summary["scenes"][scene]["stage_score_mean"] or 0.0 for scene in scenes]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    x = list(range(len(scenes)))
    width = 0.24
    axes[0].bar([i - width for i in x], grasp_rates, width=width, color="#4c78a8", label="grasp")
    axes[0].bar(x, lift_rates, width=width, color="#f2cf5b", label="lift")
    axes[0].bar([i + width for i in x], rates, width=width, color="#59a14f", label="place")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("success rate")
    axes[0].set_title("Episode stage achievement rates")
    axes[0].set_xticks(x, scenes, rotation=25, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()
    for i, scene in enumerate(scenes):
        stats = summary["scenes"][scene]
        axes[0].text(i + width, rates[i] + 0.02, f"{stats['place_count']}/{stats['runs']}", ha="center", fontsize=9)

    axes[1].bar(x, stage_scores, color="#b279a2")
    axes[1].set_ylim(0, 3.0)
    axes[1].set_ylabel("mean episode score")
    axes[1].set_title("Stage score: grasp=1, lift=2, place=3")
    axes[1].set_xticks(x, scenes, rotation=25, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("FR3/YUBI policy rollout evaluation")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--prefix", default="rollout_success_summary")
    ap.add_argument("--lift-threshold-mm", type=float, default=DEFAULT_LIFT_THRESHOLD_MM)
    args = ap.parse_args()

    run_root = args.run_root.resolve()
    out_dir = (args.out_dir or run_root).resolve()
    if args.lift_threshold_mm <= 0:
        ap.error("--lift-threshold-mm must be positive")
    records = load_rollouts(run_root, lift_threshold_mm=args.lift_threshold_mm)
    summary = aggregate(records, lift_threshold_mm=args.lift_threshold_mm)
    csv_path = out_dir / f"{args.prefix}.csv"
    json_path = out_dir / f"{args.prefix}.json"
    png_path = out_dir / f"{args.prefix}.png"
    write_csv(records, csv_path)
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    plot_summary(summary, png_path)
    print(json.dumps({"summary_csv": str(csv_path), "summary_json": str(json_path), "summary_png": str(png_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
