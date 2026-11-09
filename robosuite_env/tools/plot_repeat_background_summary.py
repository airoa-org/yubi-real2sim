#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Plot repeated rollout summaries for multiple GS backgrounds."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = tuple(plt.get_cmap("tab10").colors)


def _read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rec: dict[str, float] = {}
            for key, value in row.items():
                if value in ("", "None", "nan"):
                    rec[key] = math.nan
                elif value in ("True", "False"):
                    rec[key] = 1.0 if value == "True" else 0.0
                else:
                    try:
                        rec[key] = float(value)
                    except Exception:
                        rec[key] = math.nan
            rows.append(rec)
    if not rows:
        raise RuntimeError(f"empty trajectory CSV: {path}")
    return rows


def _arr(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([row.get(key, math.nan) for row in rows], dtype=float)


def _finite_xy(rows: list[dict[str, float]], prefix: str) -> tuple[np.ndarray, np.ndarray]:
    x = _arr(rows, f"{prefix}_x")
    y = _arr(rows, f"{prefix}_y")
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _load_runs(run_root: Path) -> list[dict]:
    runs: list[dict] = []
    background_dirs = sorted(
        path for path in run_root.iterdir()
        if path.is_dir() and any(path.glob("run_*/trajectory_analysis.csv"))
    )
    for index, background_dir in enumerate(background_dirs):
        bg = background_dir.name
        label = bg.replace("_", " ")
        color = COLORS[index % len(COLORS)]
        for csv_path in sorted(background_dir.glob("run_*/trajectory_analysis.csv")):
            run_dir = csv_path.parent
            summary_path = run_dir / "sim_summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            runs.append(
                {
                    "background": bg,
                    "label": label,
                    "color": color,
                    "run": run_dir.name,
                    "rows": _read_rows(csv_path),
                    "summary": summary,
                }
            )
    if not runs:
        raise RuntimeError(f"no trajectory_analysis.csv files under {run_root}")
    return runs


def _bounds(runs: list[dict], prefix: str) -> tuple[float, float, float, float] | None:
    vals = []
    for run in runs:
        x, y = _finite_xy(run["rows"], prefix)
        if len(x):
            vals.append((x, y))
    if not vals:
        return None
    all_x = np.concatenate([v[0] for v in vals])
    all_y = np.concatenate([v[1] for v in vals])
    span = max(float(all_x.max() - all_x.min()), float(all_y.max() - all_y.min()), 1e-6)
    pad = max(0.03, 0.08 * span)
    return float(all_x.min() - pad), float(all_x.max() + pad), float(all_y.min() - pad), float(all_y.max() + pad)


def _plot_xy(ax, runs: list[dict], prefix: str) -> None:
    bounds = _bounds(runs, prefix)
    used_labels = set()
    for run in runs:
        x, y = _finite_xy(run["rows"], prefix)
        if len(x) == 0:
            continue
        label = run["label"] if run["label"] not in used_labels else None
        used_labels.add(run["label"])
        ax.plot(x, y, color=run["color"], alpha=0.28, lw=1.2, label=label)
        ax.scatter(x[-1:], y[-1:], s=18, color=run["color"], alpha=0.65, marker="x")
    ax.set_title(f"{prefix} XY")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if bounds:
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
    ax.axis("equal")
    ax.grid(True, alpha=0.25)


def plot_xy_summary(runs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    for ax, prefix in zip(axes.ravel(), ("cup", "plate", "R_tcp", "L_tcp", "R_eef", "L_eef")):
        _plot_xy(ax, runs, prefix)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.suptitle("GS scene repeats: XY trajectories", fontsize=13, y=0.985)
    fig.legend(handles, labels, loc="lower center", ncol=min(5, max(1, len(labels))), fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_metric_summary(runs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    specs = (
        ("cup_z", "cup height [m]"),
        ("R_tcp_cup_dist_mm", "R TCP-cup distance [mm]"),
        ("L_tcp_cup_dist_mm", "L TCP-cup distance [mm]"),
        ("max_pen_mm", "max penetration [mm]"),
    )
    used_by_ax: list[set[str]] = [set() for _ in specs]
    for ax_i, (ax, (key, ylabel)) in enumerate(zip(axes.ravel(), specs)):
        for run in runs:
            x = _arr(run["rows"], "frame")
            y = _arr(run["rows"], key)
            mask = np.isfinite(x) & np.isfinite(y)
            if not mask.any():
                continue
            label = run["label"] if run["label"] not in used_by_ax[ax_i] else None
            used_by_ax[ax_i].add(run["label"])
            ax.plot(x[mask], y[mask], color=run["color"], alpha=0.28, lw=1.2, label=label)
        ax.set_title(key)
        ax.set_xlabel("frame")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.suptitle("GS scene repeats: rollout metrics", fontsize=13, y=0.985)
    fig.legend(handles, labels, loc="lower center", ncol=min(5, max(1, len(labels))), fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary_csv(runs: list[dict], path: Path) -> None:
    fields = [
        "background",
        "run",
        "new_final_success",
        "success",
        "success_ever",
        "score_total",
        "score_best",
        "max_pen_mm",
        "frames_with_pen",
        "min_R_tcp_cup_dist_mm",
        "min_L_tcp_cup_dist_mm",
        "cup_lift_mm",
        "cup_displacement_mm",
        "loop_sec",
        "wall_stopped_early",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            s = run["summary"]
            row = {"background": run["background"], "run": run["run"]}
            for key in fields[2:]:
                if key == "new_final_success":
                    row[key] = s.get("new_final_success", s.get("success"))
                else:
                    row[key] = s.get(key)
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--prefix", default="repeat_background_summary")
    args = ap.parse_args()

    run_root = args.run_root.resolve()
    out_dir = (args.out_dir or run_root).resolve()
    runs = _load_runs(run_root)
    xy_path = out_dir / f"{args.prefix}_xy.png"
    metrics_path = out_dir / f"{args.prefix}_metrics.png"
    summary_csv = out_dir / f"{args.prefix}.csv"
    plot_xy_summary(runs, xy_path)
    plot_metric_summary(runs, metrics_path)
    write_summary_csv(runs, summary_csv)
    print(json.dumps({"xy_png": str(xy_path), "metrics_png": str(metrics_path), "summary_csv": str(summary_csv)}, indent=2))


if __name__ == "__main__":
    main()
