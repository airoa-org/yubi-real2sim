#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Plot object and gripper trajectories from a sim.py log directory."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty sim log: {path}")
    return rows


def _vec(rec: dict, path: tuple[str, ...], n: int = 3) -> list[float]:
    obj = rec
    try:
        for key in path:
            obj = obj[key]
        vals = list(obj)
        return [float(vals[i]) if i < len(vals) else math.nan for i in range(n)]
    except Exception:
        return [math.nan] * n


def _num(rec: dict, path: tuple[str, ...], default: float = math.nan) -> float:
    obj = rec
    try:
        for key in path:
            obj = obj[key]
        return float(obj)
    except Exception:
        return default


def _collect(records: list[dict], summary: dict | None = None) -> list[dict]:
    rows: list[dict] = []
    for rec in records:
        frame = int(rec.get("frame", len(rows)))
        success = rec.get("success") if isinstance(rec.get("success"), dict) else {}
        score = success.get("score") if isinstance(success.get("score"), dict) else {}
        row = {
            "frame": frame,
            "R_tcp_x": _vec(rec, ("arms_metric", "R", "tcp_pos"))[0],
            "R_tcp_y": _vec(rec, ("arms_metric", "R", "tcp_pos"))[1],
            "R_tcp_z": _vec(rec, ("arms_metric", "R", "tcp_pos"))[2],
            "L_tcp_x": _vec(rec, ("arms_metric", "L", "tcp_pos"))[0],
            "L_tcp_y": _vec(rec, ("arms_metric", "L", "tcp_pos"))[1],
            "L_tcp_z": _vec(rec, ("arms_metric", "L", "tcp_pos"))[2],
            "R_eef_x": _vec(rec, ("arms_metric", "R", "eef_pos"))[0],
            "R_eef_y": _vec(rec, ("arms_metric", "R", "eef_pos"))[1],
            "R_eef_z": _vec(rec, ("arms_metric", "R", "eef_pos"))[2],
            "L_eef_x": _vec(rec, ("arms_metric", "L", "eef_pos"))[0],
            "L_eef_y": _vec(rec, ("arms_metric", "L", "eef_pos"))[1],
            "L_eef_z": _vec(rec, ("arms_metric", "L", "eef_pos"))[2],
            "R_grip_cmd": _num(rec, ("grip", "R")),
            "L_grip_cmd": _num(rec, ("grip", "L")),
            "R_grip_actual": _num(rec, ("grip_actual", "R")),
            "L_grip_actual": _num(rec, ("grip_actual", "L")),
            "max_pen_mm": _num(rec, ("max_pen_mm",), 0.0),
            "success": bool(success.get("success", False)),
            "score_points": int(score.get("points", 0)) if score else 0,
        }
        cup_position = _vec(rec, ("cup_pos",))
        plate_position = _vec(rec, ("plate_pos",))
        row.update(
            {
                "cup_x": cup_position[0],
                "cup_y": cup_position[1],
                "cup_z": cup_position[2],
                "plate_x": plate_position[0],
                "plate_y": plate_position[1],
                "plate_z": plate_position[2],
                "R_tcp_cup_dist_mm": _num(
                    rec,
                    ("arms_metric", "R", "tcp_cup_dist_mm"),
                ),
                "L_tcp_cup_dist_mm": _num(
                    rec,
                    ("arms_metric", "L", "tcp_cup_dist_mm"),
                ),
            }
        )
        rows.append(row)
    return rows


def _xy_track_specs(rows: list[dict]) -> tuple[tuple[str, str, str | None], ...]:
    object_specs = (
        ("cup", "tab:blue", "o"),
        ("plate", "tab:green", "s"),
    )
    return object_specs + (
        ("R_tcp", "tab:red", None),
        ("L_tcp", "tab:purple", None),
        ("R_eef", "tab:orange", None),
        ("L_eef", "tab:brown", None),
    )


def _arr(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([r[key] for r in rows], dtype=float)


def _finite_xy(rows: list[dict], px: str, py: str) -> tuple[np.ndarray, np.ndarray]:
    x = _arr(rows, px)
    y = _arr(rows, py)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_png(rows: list[dict], summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = _arr(rows, "frame")
    fig, ax = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax_xy, ax_z, ax_grip, ax_metric = ax.ravel()

    track_specs = _xy_track_specs(rows)
    for name, color, marker in track_specs:
        xkey, ykey = f"{name}_x", f"{name}_y"
        x, y = _finite_xy(rows, xkey, ykey)
        if len(x) == 0:
            continue
        if marker:
            ax_xy.scatter(x, y, s=14, c=color, label=name, marker=marker)
        else:
            ax_xy.plot(x, y, color=color, lw=1.8, label=name)
            ax_xy.scatter(x[:1], y[:1], s=28, c=color)
            ax_xy.scatter(x[-1:], y[-1:], s=44, c=color, marker="x")
    ax_xy.set_title("XY trajectories")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.legend(fontsize=8)

    for name, color, _marker in track_specs:
        z = _arr(rows, f"{name}_z")
        if np.isfinite(z).any():
            ax_z.plot(f, z, color=color, lw=1.5, label=name)
    ax_z.set_title("Height")
    ax_z.set_xlabel("frame")
    ax_z.set_ylabel("z [m]")
    ax_z.grid(True, alpha=0.25)
    ax_z.legend(fontsize=8)

    for key, color, style in (
        ("R_grip_cmd", "tab:red", "-"),
        ("R_grip_actual", "tab:red", "--"),
        ("L_grip_cmd", "tab:purple", "-"),
        ("L_grip_actual", "tab:purple", "--"),
    ):
        y = _arr(rows, key)
        if np.isfinite(y).any():
            ax_grip.plot(f, y, color=color, ls=style, lw=1.5, label=key)
    ax_grip.set_title("Gripper command vs actual")
    ax_grip.set_xlabel("frame")
    ax_grip.set_ylabel("finger angle [rad]")
    ax_grip.grid(True, alpha=0.25)
    ax_grip.legend(fontsize=8)

    for key, color in (
        ("R_tcp_cup_dist_mm", "tab:red"),
        ("L_tcp_cup_dist_mm", "tab:purple"),
        ("max_pen_mm", "tab:orange"),
        ("score_points", "tab:green"),
    ):
        y = _arr(rows, key)
        if np.isfinite(y).any():
            ax_metric.plot(f, y, color=color, lw=1.5, label=key)
    success = np.asarray([r["success"] for r in rows], dtype=bool)
    if success.any():
        ytop = ax_metric.get_ylim()[1]
        ax_metric.scatter(f[success], np.full(success.sum(), ytop), c="black", s=12, label="success")
    ax_metric.set_title("Distances, penetration, score")
    ax_metric.set_xlabel("frame")
    ax_metric.grid(True, alpha=0.25)
    ax_metric.legend(fontsize=8)

    title = Path(str(summary.get("logdir", ""))).name or "sim trajectory analysis"
    final_success = summary.get("new_final_success", summary.get("success"))
    minimum_distance = summary.get("min_R_tcp_cup_dist_mm")
    subtitle = (
        f"frames={summary.get('frames')} rendered={summary.get('rendered_frames')} "
        f"new_final_success={final_success} score_best={summary.get('score_best')} "
        f"min_R_tcp_cup={minimum_distance}mm"
    )
    fig.suptitle(f"{title}\n{subtitle}", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_10tracks_png(rows: list[dict], summary: dict, path: Path) -> None:
    """Write a clearer task-object / robot / gripper signal overview."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = _arr(rows, "frame")
    fig, axes = plt.subplots(5, 2, figsize=(13, 18), constrained_layout=True)
    axes_flat = axes.ravel()

    xy_specs = tuple(
        (name, color) for name, color, _marker in _xy_track_specs(rows)
    )
    xlim = ylim = None
    xy_vals = []
    for name, _color in xy_specs:
        x, y = _finite_xy(rows, f"{name}_x", f"{name}_y")
        if len(x):
            xy_vals.append((x, y))
    if xy_vals:
        all_x = np.concatenate([v[0] for v in xy_vals])
        all_y = np.concatenate([v[1] for v in xy_vals])
        pad = max(0.03, 0.08 * max(float(all_x.max() - all_x.min()), float(all_y.max() - all_y.min()), 1e-6))
        xlim = (float(all_x.min() - pad), float(all_x.max() + pad))
        ylim = (float(all_y.min() - pad), float(all_y.max() + pad))

    for ax, (name, color) in zip(axes_flat[:6], xy_specs):
        x, y = _finite_xy(rows, f"{name}_x", f"{name}_y")
        if len(x):
            ax.plot(x, y, color=color, lw=2.0)
            ax.scatter(x[:1], y[:1], s=32, c=color, label="start")
            ax.scatter(x[-1:], y[-1:], s=52, c=color, marker="x", label="end")
        ax.set_title(f"{name} XY track")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if xlim and ylim:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    for ax in axes_flat[len(xy_specs):6]:
        ax.axis("off")

    gripper_specs = (
        ("R_grip_cmd", "tab:red", "-"),
        ("R_grip_actual", "tab:red", "--"),
        ("L_grip_cmd", "tab:purple", "-"),
        ("L_grip_actual", "tab:purple", "--"),
    )
    for ax, (key, color, style) in zip(axes_flat[6:], gripper_specs):
        y = _arr(rows, key)
        if np.isfinite(y).any():
            ax.plot(f, y, color=color, ls=style, lw=1.7)
        ax.set_title(key)
        ax.set_xlabel("frame")
        ax.set_ylabel("finger angle [rad]")
        ax.grid(True, alpha=0.25)

    title = Path(str(summary.get("logdir", ""))).parent.name or "sim trajectory analysis"
    object_names = "cup, plate"
    signal_count = len(xy_specs) + len(gripper_specs)
    subtitle = (
        f"{signal_count} signals: {object_names}, R/L TCP, R/L EEF, "
        "R/L gripper command, R/L gripper actual"
    )
    fig.suptitle(f"{title}\n{subtitle}", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _norm_xy(rows: list[dict]) -> tuple[float, float, float, float]:
    vals = []
    for key, _color, _marker in _xy_track_specs(rows):
        x = _arr(rows, f"{key}_x")
        y = _arr(rows, f"{key}_y")
        mask = np.isfinite(x) & np.isfinite(y)
        vals.extend(zip(x[mask], y[mask]))
    if not vals:
        return -0.5, 0.5, -0.5, 0.5
    arr = np.asarray(vals, dtype=float)
    xmin, ymin = arr.min(axis=0)
    xmax, ymax = arr.max(axis=0)
    pad = max(0.03, 0.08 * max(xmax - xmin, ymax - ymin, 1e-6))
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def render_mp4(rows: list[dict], path: Path, fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    W, H = 900, 700
    xmin, xmax, ymin, ymax = _norm_xy(rows)
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    colors = {
        "cup": (220, 80, 40),
        "plate": (60, 160, 70),
    }
    colors.update(
        {
            "R_tcp": (40, 40, 220),
            "L_tcp": (160, 60, 160),
            "R_eef": (30, 140, 220),
            "L_eef": (90, 90, 120),
        }
    )
    right_distance_key = "R_tcp_cup_dist_mm"

    def to_px(x: float, y: float) -> tuple[int, int]:
        px = int(70 + (x - xmin) / max(xmax - xmin, 1e-9) * (W - 140))
        py = int(H - 70 - (y - ymin) / max(ymax - ymin, 1e-9) * (H - 140))
        return px, py

    tracks = {k: [] for k in colors}
    for i, row in enumerate(rows):
        img = np.full((H, W, 3), 245, np.uint8)
        cv.rectangle(img, (70, 70), (W - 70, H - 70), (210, 210, 210), 1)
        for key, color in colors.items():
            x = row[f"{key}_x"]
            y = row[f"{key}_y"]
            if np.isfinite(x) and np.isfinite(y):
                tracks[key].append(to_px(float(x), float(y)))
            pts = tracks[key]
            for p0, p1 in zip(pts[:-1], pts[1:]):
                cv.line(img, p0, p1, color, 2)
            if pts:
                marker = 6 if key in ("cup", "plate") else 4
                cv.circle(img, pts[-1], marker, color, -1)
        cv.putText(img, f"frame {int(row['frame'])}", (28, 34), cv.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        cv.putText(
            img,
            f"R dist {row[right_distance_key]:.1f} mm  pen {row['max_pen_mm']:.2f} mm  score {row['score_points']}",
            (28, 64),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
        )
        y0 = 95
        for j, (key, color) in enumerate(colors.items()):
            cv.circle(img, (W - 220, y0 + j * 24), 5, color, -1)
            cv.putText(img, key, (W - 205, y0 + j * 24 + 5), cv.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
        writer.write(img)
    writer.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logdir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--prefix", default="trajectory_analysis")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    logdir = args.logdir.resolve()
    out_dir = (args.out_dir or logdir).resolve()
    summary_path = logdir / "sim_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"logdir": str(logdir)}
    records = _read_jsonl(logdir / "sim_log.jsonl")
    rows = _collect(records, summary)

    csv_path = out_dir / f"{args.prefix}.csv"
    png_path = out_dir / f"{args.prefix}.png"
    ten_tracks_png_path = out_dir / f"{args.prefix}_10tracks.png"
    mp4_path = out_dir / f"{args.prefix}.mp4"
    write_csv(rows, csv_path)
    plot_png(rows, summary, png_path)
    plot_10tracks_png(rows, summary, ten_tracks_png_path)
    if not args.no_video:
        render_mp4(rows, mp4_path)

    result = {"csv": str(csv_path), "png": str(png_path), "ten_tracks_png": str(ten_tracks_png_path)}
    if not args.no_video:
        result["mp4"] = str(mp4_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
