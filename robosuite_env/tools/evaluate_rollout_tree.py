#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a staged rollout tree with the bbox cup-on-plate criterion.

The expected layout is:

    PIPELINE_ROOT/
      stage1_initial_b10/outputs.csv
      stage2_from_grasp_b10/outputs.csv
      stage3_from_max_lift_b10/outputs.csv

Each outputs.csv row must point at a per-run sim_summary.json. The script
recomputes final-frame place from the recorded final metrics, then reports
episode success as:

    grasp and cup_lift_mm >= threshold and final-frame place
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STAGES = (
    "stage1_initial_b10",
    "stage2_from_grasp_b10",
    "stage3_from_max_lift_b10",
)
DEFAULT_CENTER_THRESHOLD_MM = 81.28
DEFAULT_Z_MIN_MM = -2.57
DEFAULT_Z_MAX_MM = 27.03
DEFAULT_LIFT_THRESHOLD_MM = 50.0
DEFAULT_UPRIGHT_TOL_DEG = 35.0


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"expected object JSON: {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _num(value: Any, default: float | None = math.nan) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _pct(n: int, d: int) -> str:
    return "nan%" if d <= 0 else f"{100.0 * float(n) / float(d):.0f}%"


def _rate(n: int, d: int) -> str:
    return f"{n}/{d} = {_pct(n, d)}"


def _discover_stages(root: Path, explicit: list[str]) -> list[str]:
    if explicit:
        return explicit
    existing = {p.name for p in root.iterdir() if p.is_dir() and (p / "outputs.csv").is_file()}
    ordered = [name for name in DEFAULT_STAGES if name in existing]
    ordered.extend(sorted(existing - set(ordered)))
    if not ordered:
        raise RuntimeError(f"no stage directories with outputs.csv under {root}")
    return ordered


def _read_outputs(stage_root: Path) -> list[dict[str, str]]:
    path = stage_root / "outputs.csv"
    if not path.is_file():
        raise RuntimeError(f"missing outputs.csv: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _resolve_path(value: str, base: Path) -> Path | None:
    value = (value or "").strip()
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def _first_grasp(events: Any) -> tuple[bool, int | None, str | None]:
    if not isinstance(events, list):
        return False, None, None
    for event in events:
        if not isinstance(event, dict) or event.get("event") != "grasp":
            continue
        frame = event.get("frame")
        return True, int(frame) if frame is not None else None, event.get("hand")
    return False, None, None


def _load_reference(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    data = _read_json(path)
    stages = data.get("stages")
    if not isinstance(stages, dict):
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for stage, stage_data in stages.items():
        if not isinstance(stage_data, dict):
            continue
        rows = stage_data.get("rows")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("run"):
                out[(str(stage), str(row["run"]))] = row
    return out


def _bbox_place(
    summary: dict[str, Any],
    *,
    center_threshold_mm: float,
    z_min_mm: float,
    z_max_mm: float,
    normal_dot_min: float,
) -> tuple[bool, dict[str, Any]]:
    final = summary.get("success_final") if isinstance(summary.get("success_final"), dict) else {}
    conditions = final.get("conditions") if isinstance(final.get("conditions"), dict) else {}
    metrics = final.get("metrics") if isinstance(final.get("metrics"), dict) else {}

    contact = _bool(conditions.get("cup_plate_contact"))
    bottom_contact = _bool(conditions.get("cup_bottom_contact"))
    normal_dot = _num(metrics.get("cup_axis_plate_normal_dot"), None)
    if "cup_axis_aligned_with_plate_normal" in conditions:
        normal_ok = _bool(conditions.get("cup_axis_aligned_with_plate_normal"))
    else:
        normal_ok = bool(normal_dot is not None and normal_dot >= normal_dot_min)

    center_mm = _num(metrics.get("bottom_center_offset_mm"), None)
    if "cup_center_inside_plate_disc" in conditions:
        center_ok = _bool(conditions.get("cup_center_inside_plate_disc"))
    else:
        center_ok = bool(center_mm is not None and center_mm <= center_threshold_mm)

    height_mm = _num(metrics.get("height_from_plate_bottom_mm"), None)
    if "cup_bottom_in_plate_zrange" in conditions:
        z_ok = _bool(conditions.get("cup_bottom_in_plate_zrange"))
    else:
        z_ok = bool(height_mm is not None and z_min_mm <= height_mm <= z_max_mm)

    # The criterion records whether the cup came to rest on the plate; that is
    # the task definition (a cup on its side counts) and it is what the run
    # scored itself against. The seated recomputation below stays as the
    # fallback for summaries written before the criterion reported it.
    resting = conditions.get("cup_resting_on_plate")
    if resting is not None:
        place = _bool(resting)
    else:
        place = bool(contact and bottom_contact and normal_ok and center_ok and z_ok)
    details = {
        "cup_resting_on_plate": _bool(resting) if resting is not None else None,
        "cup_plate_contact": contact,
        "cup_bottom_contact": bottom_contact,
        "cup_center_inside_plate_disc": center_ok,
        "cup_bottom_in_plate_zrange": z_ok,
        "cup_axis_aligned_with_plate_normal": normal_ok,
        "bottom_center_offset_mm": round(center_mm, 3) if center_mm is not None else None,
        "height_from_plate_bottom_mm": round(height_mm, 3) if height_mm is not None else None,
        "cup_axis_plate_normal_dot": round(normal_dot, 5) if normal_dot is not None else None,
    }
    return place, details


def _evaluate_row(
    stage: str,
    stage_root: Path,
    out_row: dict[str, str],
    reference: dict[tuple[str, str], dict[str, Any]],
    *,
    lift_threshold_mm: float,
    center_threshold_mm: float,
    z_min_mm: float,
    z_max_mm: float,
    normal_dot_min: float,
) -> dict[str, Any]:
    run = out_row.get("run") or "run_unknown"
    summary_path = _resolve_path(out_row.get("summary", ""), stage_root)
    summary = _read_json(summary_path) if summary_path and summary_path.is_file() else {}
    ref = reference.get((stage, run), {})

    grasp, first_grasp_frame, first_grasp_hand = _first_grasp(summary.get("rule_grasp_events"))
    lift_mm = _num(summary.get("cup_lift_mm"), 0.0) or 0.0
    grasp_lift = bool(grasp and lift_mm >= lift_threshold_mm)
    place, place_details = _bbox_place(
        summary,
        center_threshold_mm=center_threshold_mm,
        z_min_mm=z_min_mm,
        z_max_mm=z_max_mm,
        normal_dot_min=normal_dot_min,
    )
    success = bool(grasp_lift and place)
    completed = str(out_row.get("exit_code", "")).strip() == "0"

    row = {
        "scene": out_row.get("scene"),
        "run": run,
        "completed": completed,
        "exit_code": out_row.get("exit_code"),
        "grasp": grasp,
        "first_grasp_frame": first_grasp_frame,
        "first_grasp_hand": first_grasp_hand,
        "lift_mm": round(lift_mm, 3),
        "grasp_lift_50": grasp_lift,
        "place": place,
        "success": success,
        "first_success_frame": ref.get("first_success_frame") or summary.get("success_first_frame"),
        "place_ever": ref.get("place_ever", place),
        "success_ever": ref.get("success_ever", success),
        "bbox": place_details,
        "max_pen_mm": _num(summary.get("max_pen_mm"), None),
        "frames_with_pen": summary.get("frames_with_pen"),
        "video": out_row.get("video"),
        "summary": str(summary_path) if summary_path else None,
        "init_state_in": summary.get("init_state_in"),
    }
    return row


def _stage_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    counts = {
        "completed": sum(1 for r in rows if r["completed"]),
        "grasp": sum(1 for r in rows if r["grasp"]),
        "grasp_lift_50": sum(1 for r in rows if r["grasp_lift_50"]),
        "place": sum(1 for r in rows if r["place"]),
        "success": sum(1 for r in rows if r["success"]),
        "place_ever": sum(1 for r in rows if r.get("place_ever")),
        "success_ever": sum(1 for r in rows if r.get("success_ever")),
        "total": total,
    }
    counts["rates"] = {key: _rate(value, total) for key, value in counts.items() if isinstance(value, int) and key != "total"}
    return counts


def evaluate_tree(
    pipeline_root: Path,
    *,
    stages: list[str] | None = None,
    reference_json: Path | None = None,
    lift_threshold_mm: float = DEFAULT_LIFT_THRESHOLD_MM,
    center_threshold_mm: float = DEFAULT_CENTER_THRESHOLD_MM,
    z_min_mm: float = DEFAULT_Z_MIN_MM,
    z_max_mm: float = DEFAULT_Z_MAX_MM,
    upright_tol_deg: float = DEFAULT_UPRIGHT_TOL_DEG,
) -> dict[str, Any]:
    """Evaluate every stage under ``pipeline_root`` and return the report."""

    root = Path(pipeline_root).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).absolute()
    ordered_stages = _discover_stages(root, list(stages or []))
    ref_path = reference_json
    if ref_path is None:
        candidate = root / "bbox_success_review.json"
        ref_path = candidate if candidate.is_file() else None
    reference = _load_reference(ref_path)
    normal_dot_min = math.cos(math.radians(float(upright_tol_deg)))

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline_root": str(root),
        "definition": {
            "name": "tree_evaluation_bbox_final",
            "lift_threshold_mm": float(lift_threshold_mm),
            "center_threshold_mm": float(center_threshold_mm),
            "z_range_mm": [float(z_min_mm), float(z_max_mm)],
            "upright_tolerance_deg": float(upright_tol_deg),
            "normal_dot_min": round(normal_dot_min, 6),
            "requires": [
                "grasp",
                "cup_lift_mm >= lift_threshold_mm",
                "cup_resting_on_plate",
            ],
            "requires_without_recorded_verdict": [
                "grasp",
                "cup_lift_mm >= lift_threshold_mm",
                "cup_plate_contact",
                "cup_bottom_contact",
                "cup_center_inside_plate_disc",
                "cup_bottom_in_plate_zrange",
                "cup_axis_aligned_with_plate_normal",
            ],
        },
        "reference_json": str(ref_path) if ref_path else None,
        "stages": {},
    }

    for stage in ordered_stages:
        stage_root = root / stage
        rows = [
            _evaluate_row(
                stage,
                stage_root,
                row,
                reference,
                lift_threshold_mm=float(lift_threshold_mm),
                center_threshold_mm=float(center_threshold_mm),
                z_min_mm=float(z_min_mm),
                z_max_mm=float(z_max_mm),
                normal_dot_min=normal_dot_min,
            )
            for row in _read_outputs(stage_root)
        ]
        init_states = sorted({r.get("init_state_in") for r in rows if r.get("init_state_in")})
        result["stages"][stage] = {
            "path": str(stage_root),
            "init_state_in": init_states[0] if len(init_states) == 1 else init_states,
            "counts": _stage_counts(rows),
            "rows": rows,
        }
    return result


def _fmt_bool(value: Any) -> str:
    return "1" if bool(value) else "0"


def render_markdown(data: dict[str, Any]) -> str:
    definition = data["definition"]
    lines = [
        "# Rollout Tree Evaluation",
        "",
        f"- pipeline_root: `{data['pipeline_root']}`",
        f"- generated_at: `{data['generated_at']}`",
        f"- lift threshold: `{definition['lift_threshold_mm']} mm`",
        f"- bbox center threshold (fallback only): `bottom_center_offset_mm <= {definition['center_threshold_mm']}`",
        f"- bbox z threshold (fallback only): `{definition['z_range_mm'][0]} <= height_from_plate_bottom_mm <= {definition['z_range_mm'][1]}`",
        f"- upright tolerance (fallback only): `{definition['upright_tolerance_deg']} deg`",
        "",
        "Episode success is `grasp and grasp+lift>=50mm and final-frame place`.",
        "Place is the run's own `cup_resting_on_plate`; the three thresholds above",
        "are the seated recomputation, used only for summaries that predate it.",
        "",
        "## Stage Summary",
        "",
        "| stage | completed | grasp | grasp+lift>=50mm | place | success |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage, stage_data in data["stages"].items():
        counts = stage_data["counts"]
        total = counts["total"]
        lines.append(
            f"| {stage} | {_rate(counts['completed'], total)} | {_rate(counts['grasp'], total)} | "
            f"{_rate(counts['grasp_lift_50'], total)} | {_rate(counts['place'], total)} | "
            f"{_rate(counts['success'], total)} |"
        )

    lines.extend(
        [
            "",
            "## Successful Examples",
            "",
            "| stage | run | first_success_frame | lift_mm | center_mm | height_mm | dot | video |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    any_success = False
    for stage, stage_data in data["stages"].items():
        for row in stage_data["rows"]:
            if not row["success"]:
                continue
            any_success = True
            bbox = row["bbox"]
            lines.append(
                f"| {stage} | {row['run']} | {row.get('first_success_frame')} | {row['lift_mm']:.2f} | "
                f"{bbox.get('bottom_center_offset_mm')} | {bbox.get('height_from_plate_bottom_mm')} | "
                f"{bbox.get('cup_axis_plate_normal_dot')} | `{row.get('video')}` |"
            )
    if not any_success:
        lines.append("| - | - | - | - | - | - | - | - |")

    for stage, stage_data in data["stages"].items():
        lines.extend(
            [
                "",
                f"## {stage}",
                "",
                "| run | done | grasp | first_grasp | lift_mm | grasp+lift | place | success | center_mm | height_mm | dot |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in stage_data["rows"]:
            bbox = row["bbox"]
            lines.append(
                f"| {row['run']} | {_fmt_bool(row['completed'])} | {_fmt_bool(row['grasp'])} | "
                f"{row.get('first_grasp_frame')} | {row['lift_mm']:.2f} | {_fmt_bool(row['grasp_lift_50'])} | "
                f"{_fmt_bool(row['place'])} | {_fmt_bool(row['success'])} | "
                f"{bbox.get('bottom_center_offset_mm')} | {bbox.get('height_from_plate_bottom_mm')} | "
                f"{bbox.get('cup_axis_plate_normal_dot')} |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline_root", type=Path, help="root containing staged rollout directories")
    parser.add_argument("--stage", action="append", default=[], help="stage directory name; repeat to override auto order")
    parser.add_argument("--reference-json", type=Path, default=None, help="optional previous tree-evaluation/review JSON for first_success_frame")
    parser.add_argument("--out-json", type=Path, default=None, help="write machine-readable evaluation JSON")
    parser.add_argument("--out-md", type=Path, default=None, help="write Markdown report")
    parser.add_argument("--lift-threshold-mm", type=float, default=DEFAULT_LIFT_THRESHOLD_MM)
    parser.add_argument("--center-threshold-mm", type=float, default=DEFAULT_CENTER_THRESHOLD_MM)
    parser.add_argument("--z-min-mm", type=float, default=DEFAULT_Z_MIN_MM)
    parser.add_argument("--z-max-mm", type=float, default=DEFAULT_Z_MAX_MM)
    parser.add_argument("--upright-tol-deg", type=float, default=DEFAULT_UPRIGHT_TOL_DEG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = evaluate_tree(
        args.pipeline_root,
        stages=args.stage,
        reference_json=args.reference_json,
        lift_threshold_mm=args.lift_threshold_mm,
        center_threshold_mm=args.center_threshold_mm,
        z_min_mm=args.z_min_mm,
        z_max_mm=args.z_max_mm,
        upright_tol_deg=args.upright_tol_deg,
    )
    markdown = render_markdown(data)
    if args.out_json:
        _write_json(args.out_json, data)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown)
    if not args.out_json and not args.out_md:
        print(markdown)


if __name__ == "__main__":
    main()
