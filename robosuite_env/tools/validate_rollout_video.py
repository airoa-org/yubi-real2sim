#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Decode every rollout video frame and compare it with sim metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import cv2
import numpy as np

from robosuite_env.rollout.outputs import open_video_capture

cv2.setNumThreads(max(1, int(os.environ.get("OPENCV_FOR_THREADS_NUM", "1"))))


# These checks intentionally target only catastrophic renderer failures.  Rollout
# videos can legitimately contain a mostly static scene or low-contrast regions,
# so the thresholds are deliberately much stricter than general-purpose video
# quality metrics.
_MIN_SPATIAL_CHECK_PIXELS = 64 * 64
_NEAR_BLACK_P99_LUMA = 10
_NEAR_UNIFORM_LUMA_RANGE = 6
_NEAR_UNIFORM_LUMA_STD = 2.5
_FROZEN_MEAN_ABS_DELTA = 0.05
_FROZEN_MAX_ABS_DELTA = 1
_LONG_FROZEN_SECONDS = 5.0
_LONG_FROZEN_MIN_FRAMES = 30
_ALTERNATING_PATTERN_THRESHOLD = 0.70
_PATTERN_BLOCK_SIZES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
_PATTERN_MAX_DIMENSION = 192
_MIN_ROI_SPATIAL_CHECK_PIXELS = 16 * 16
_ROI_SUSTAINED_ARTIFACT_FRACTION = 0.25
_ROI_SUSTAINED_ARTIFACT_MIN_FRAMES = 2
_SAVED_WRIST_MAX_FRAME_MAE = 10.0
_SAVED_WRIST_MAX_MEAN_MAE = 6.0
_SAVED_WRIST_MAX_FRAME_P95 = 18.0
_SAVED_WRIST_MAX_FRAME_P99 = 30.0
_SAVED_WRIST_BAD_PIXEL_ABS_ERROR = 20
_SAVED_WRIST_MAX_BAD_PIXEL_FRACTION = 0.025
_VIDEO_FPS_ABS_TOLERANCE = 0.05
_VIDEO_FPS_REL_TOLERANCE = 0.001
_VIDEO_DURATION_ABS_TOLERANCE_S = 0.001
_VIDEO_DURATION_REL_TOLERANCE = 0.002
_SAVED_WRIST_CANONICAL_LAYOUT_V2 = {
    "left": [4, 4, 180, 135],
    "right": [188, 4, 180, 135],
}
_SAVED_WRIST_CANONICAL_LAYOUT_V3 = {
    "left": [4, 4, 180, 135],
    "right": [188, 4, 180, 135],
}


class VideoValidationError(RuntimeError):
    """Quality failure with the complete decoded report attached."""

    def __init__(self, message: str, result: dict):
        super().__init__(message)
        self.result = result


def _luma_percentiles(gray: np.ndarray) -> tuple[int, int]:
    """Return exact uint8 p01/p99 values without sorting a full-size frame."""

    histogram = np.bincount(gray.reshape(-1), minlength=256)
    cumulative = np.cumsum(histogram)
    pixels = int(gray.size)
    p01 = int(np.searchsorted(cumulative, max(1, math.ceil(pixels * 0.01))))
    p99 = int(np.searchsorted(cumulative, max(1, math.ceil(pixels * 0.99))))
    return p01, p99


def _thumbnail(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    scale = min(1.0, _PATTERN_MAX_DIMENSION / max(height, width))
    if scale == 1.0:
        return gray
    return cv2.resize(
        gray,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _alternating_pattern(gray: np.ndarray) -> dict:
    """Measure full-frame alternating stripes / checkerboards.

    For a candidate block width, a genuine alternating pattern has a large
    difference at a one-block offset and repeats at a two-block offset.  The
    reported score is ``(one_block_delta - two_block_delta) / 255``.  Ordinary
    image edges and gradients generally have a non-zero two-block delta and
    therefore do not score highly.
    """

    sampled = _thumbnail(gray).astype(np.int16, copy=False)
    height, width = sampled.shape
    best = {
        "score": 0.0,
        "kind": None,
        "block_size_px": None,
        "horizontal_score": 0.0,
        "vertical_score": 0.0,
    }
    for block_size in _PATTERN_BLOCK_SIZES:
        horizontal = 0.0
        vertical = 0.0
        if width > 2 * block_size:
            one = float(
                np.mean(
                    np.abs(
                        sampled[:, block_size:] - sampled[:, :-block_size]
                    )
                )
            )
            two = float(
                np.mean(
                    np.abs(
                        sampled[:, 2 * block_size :]
                        - sampled[:, : -2 * block_size]
                    )
                )
            )
            horizontal = max(0.0, (one - two) / 255.0)
        if height > 2 * block_size:
            one = float(
                np.mean(
                    np.abs(
                        sampled[block_size:, :] - sampled[:-block_size, :]
                    )
                )
            )
            two = float(
                np.mean(
                    np.abs(
                        sampled[2 * block_size :, :]
                        - sampled[: -2 * block_size, :]
                    )
                )
            )
            vertical = max(0.0, (one - two) / 255.0)

        score = max(horizontal, vertical)
        if score <= best["score"]:
            continue
        if min(horizontal, vertical) >= _ALTERNATING_PATTERN_THRESHOLD:
            kind = "checkerboard"
        elif horizontal >= vertical:
            kind = "vertical_stripes"
        else:
            kind = "horizontal_stripes"
        best = {
            "score": score,
            "kind": kind,
            "block_size_px": block_size,
            "horizontal_score": horizontal,
            "vertical_score": vertical,
        }
    return best


def _short_indices(indices: list[int], *, limit: int = 20) -> dict:
    return {
        "count": len(indices),
        "indices": indices[:limit],
        "indices_truncated": len(indices) > limit,
    }


def _new_visual_tracker() -> dict:
    return {
        "near_black_indices": [],
        "near_uniform_indices": [],
        "alternating_indices": [],
        "previous_gray": None,
        "frozen_transitions": 0,
        "current_frozen_run_frames": 1,
        "max_frozen_run_frames": 1,
        "min_frame_delta_mean": None,
        "max_alternating": {
            "score": 0.0,
            "frame_index": None,
            "kind": None,
            "block_size_px": None,
            "horizontal_score": 0.0,
            "vertical_score": 0.0,
        },
        "min_luma_p99": None,
        "min_luma_std": None,
    }


def _update_visual_tracker(
    tracker: dict,
    gray: np.ndarray,
    frame_index: int,
    *,
    temporal_margin: int = 0,
) -> None:
    p01, p99 = _luma_percentiles(gray)
    luma_std = float(cv2.meanStdDev(gray)[1][0, 0])
    tracker["min_luma_p99"] = (
        p99
        if tracker["min_luma_p99"] is None
        else min(tracker["min_luma_p99"], p99)
    )
    tracker["min_luma_std"] = (
        luma_std
        if tracker["min_luma_std"] is None
        else min(tracker["min_luma_std"], luma_std)
    )
    if p99 <= _NEAR_BLACK_P99_LUMA:
        tracker["near_black_indices"].append(frame_index)
    elif (
        p99 - p01 <= _NEAR_UNIFORM_LUMA_RANGE
        and luma_std <= _NEAR_UNIFORM_LUMA_STD
    ):
        tracker["near_uniform_indices"].append(frame_index)

    pattern = _alternating_pattern(gray)
    if pattern["score"] > tracker["max_alternating"]["score"]:
        tracker["max_alternating"] = {**pattern, "frame_index": frame_index}
    if pattern["score"] >= _ALTERNATING_PATTERN_THRESHOLD:
        tracker["alternating_indices"].append(frame_index)

    temporal_source = gray
    if (
        temporal_margin > 0
        and gray.shape[0] > 2 * temporal_margin
        and gray.shape[1] > 2 * temporal_margin
    ):
        # Lossy codecs can leak motion from the surrounding overview into the
        # edge macroblocks of an otherwise frozen inset.
        temporal_source = gray[
            temporal_margin:-temporal_margin,
            temporal_margin:-temporal_margin,
        ]
    temporal_gray = _thumbnail(temporal_source)
    previous_gray = tracker["previous_gray"]
    if previous_gray is not None:
        delta = cv2.absdiff(temporal_gray, previous_gray)
        mean_delta = float(np.mean(delta))
        max_delta = int(np.max(delta))
        tracker["min_frame_delta_mean"] = (
            mean_delta
            if tracker["min_frame_delta_mean"] is None
            else min(tracker["min_frame_delta_mean"], mean_delta)
        )
        if (
            mean_delta <= _FROZEN_MEAN_ABS_DELTA
            and max_delta <= _FROZEN_MAX_ABS_DELTA
        ):
            tracker["frozen_transitions"] += 1
            tracker["current_frozen_run_frames"] += 1
            tracker["max_frozen_run_frames"] = max(
                tracker["max_frozen_run_frames"],
                tracker["current_frozen_run_frames"],
            )
        else:
            tracker["current_frozen_run_frames"] = 1
    tracker["previous_gray"] = temporal_gray


def _parse_wrist_roi_metadata(summary: dict) -> tuple[dict | None, str]:
    """Return validated SIM-panel ROI metadata, or a skip reason.

    Old summaries intentionally skip these checks. Guessing coordinates from a
    historical video's dimensions would risk validating the REAL panel or a
    clean-video layout as if it were a wrist inset.
    """

    if "video_wrist_insets_enabled" not in summary:
        return None, "metadata_not_declared"
    enabled = summary["video_wrist_insets_enabled"]
    if not isinstance(enabled, bool):
        raise RuntimeError("video_wrist_insets_enabled must be a boolean")
    if not enabled:
        return None, "disabled"

    offset = summary.get("video_sim_panel_x_offset")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise RuntimeError(
            "video_sim_panel_x_offset must be a non-negative integer when "
            "wrist insets are enabled"
        )
    layout = summary.get("video_wrist_inset_layout")
    if not isinstance(layout, dict) or "version" not in layout:
        raise RuntimeError(
            "video_wrist_inset_layout must contain version and ROI coordinates"
        )
    version = layout["version"]
    if version is None or isinstance(version, (dict, list, bool)):
        raise RuntimeError("video_wrist_inset_layout.version must be a scalar")

    parsed = {"version": version, "sim_panel_x_offset": offset, "regions": {}}
    for side in ("left", "right"):
        key = f"{side}_xywh"
        xywh = layout.get(key)
        if not isinstance(xywh, (list, tuple)) or len(xywh) != 4:
            raise RuntimeError(f"video_wrist_inset_layout.{key} must be [x,y,w,h]")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in xywh):
            raise RuntimeError(f"video_wrist_inset_layout.{key} must contain integers")
        x, y, width, height = xywh
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise RuntimeError(
                f"video_wrist_inset_layout.{key} has invalid bounds {list(xywh)}"
            )
        parsed["regions"][side] = {
            "panel_xywh": [x, y, width, height],
            "video_xywh": [offset + x, y, width, height],
        }
    return parsed, "enabled"


def _validate_roi_bounds(metadata: dict, shape: tuple[int, int, int]) -> None:
    frame_height, frame_width = shape[:2]
    for side, region in metadata["regions"].items():
        x, y, width, height = region["video_xywh"]
        if x + width > frame_width or y + height > frame_height:
            raise RuntimeError(
                f"wrist ROI {side} {region['video_xywh']} is outside decoded "
                f"frame {frame_width}x{frame_height}"
            )


def _temporal_findings(
    tracker: dict, decoded_frames: int, fps: float
) -> tuple[bool, bool, int]:
    temporal_checks_applied = decoded_frames >= 3
    long_frozen_threshold_frames = max(
        _LONG_FROZEN_MIN_FRAMES,
        math.ceil(_LONG_FROZEN_SECONDS * fps) if fps > 0.0 else 0,
    )
    entire_video_frozen = bool(
        temporal_checks_applied
        # Require one continuous near-whole-video run. Policy observations are
        # intentionally held between replans; aggregating many valid hold runs
        # would otherwise misclassify a sequence that changes every chunk.
        and tracker["max_frozen_run_frames"]
        >= math.ceil(0.90 * decoded_frames)
    )
    long_frozen_run = bool(
        not entire_video_frozen
        and tracker["max_frozen_run_frames"] >= long_frozen_threshold_frames
    )
    return entire_video_frozen, long_frozen_run, long_frozen_threshold_frames


def _tracker_report(
    tracker: dict,
    *,
    decoded_frames: int,
    fps: float,
    spatial_checks_applied: bool,
) -> dict:
    entire_frozen, long_frozen, threshold = _temporal_findings(
        tracker, decoded_frames, fps
    )
    return {
        "spatial_checks_applied": spatial_checks_applied,
        "temporal_checks_applied": decoded_frames >= 3,
        "near_black": _short_indices(tracker["near_black_indices"]),
        "near_uniform": _short_indices(tracker["near_uniform_indices"]),
        "frozen_transitions": tracker["frozen_transitions"],
        "max_frozen_run_frames": tracker["max_frozen_run_frames"],
        "entire_video_frozen": entire_frozen,
        "long_frozen_run": long_frozen,
        "long_frozen_threshold_frames": threshold,
        "minimum_frame_delta_mean": tracker["min_frame_delta_mean"],
        "alternating_artifacts": _short_indices(tracker["alternating_indices"]),
        "max_alternating_pattern": tracker["max_alternating"],
        "minimum_luma_p99": tracker["min_luma_p99"],
        "minimum_luma_std": tracker["min_luma_std"],
    }


def validate_video(video_path: Path, summary_path: Path) -> dict:
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise RuntimeError(f"video is missing or empty: {video_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    overview_pipeline = summary.get("overview_render_pipeline")
    if (
        summary.get("policy_batch_server")
        and overview_pipeline
        != "deferred_isolated_egl_after_wrist_teardown"
    ):
        raise RuntimeError(
            "batch-policy rollout used unsafe overview render pipeline "
            f"{overview_pipeline!r}; expected "
            "'deferred_isolated_egl_after_wrist_teardown'"
        )
    expected_frames = int(summary["rendered_frames"])
    if expected_frames <= 0:
        raise RuntimeError(f"invalid rendered_frames={expected_frames}")
    raw_expected_fps = summary.get("output_fps")
    if isinstance(raw_expected_fps, bool) or not isinstance(
        raw_expected_fps, (int, float)
    ):
        raise RuntimeError(
            "summary output_fps must be a positive finite number, got "
            f"{raw_expected_fps!r}"
        )
    expected_fps = float(raw_expected_fps)
    if not math.isfinite(expected_fps) or expected_fps <= 0.0:
        raise RuntimeError(
            "summary output_fps must be a positive finite number, got "
            f"{raw_expected_fps!r}"
        )
    source_frames = summary.get("frames")
    if (
        isinstance(source_frames, bool)
        or not isinstance(source_frames, int)
        or source_frames <= 0
    ):
        raise RuntimeError(
            "summary frames must be a positive integer, got "
            f"{source_frames!r}"
        )
    raw_playback_action_dt_s = summary.get("playback_action_dt_s")
    if isinstance(raw_playback_action_dt_s, bool) or not isinstance(
        raw_playback_action_dt_s, (int, float)
    ):
        raise RuntimeError(
            "summary playback_action_dt_s must be a positive finite number, "
            f"got {raw_playback_action_dt_s!r}"
        )
    playback_action_dt_s = float(raw_playback_action_dt_s)
    if not math.isfinite(playback_action_dt_s) or playback_action_dt_s <= 0.0:
        raise RuntimeError(
            "summary playback_action_dt_s must be a positive finite number, "
            f"got {raw_playback_action_dt_s!r}"
        )
    wrist_metadata, wrist_skip_reason = _parse_wrist_roi_metadata(summary)
    wrist_trackers = (
        {"left": _new_visual_tracker(), "right": _new_visual_tracker()}
        if wrist_metadata is not None
        else {}
    )

    capture = open_video_capture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        capture.release()
        raise RuntimeError(
            f"decoded video FPS must be a positive finite number, got {fps!r}"
        )
    fps_tolerance = max(
        _VIDEO_FPS_ABS_TOLERANCE,
        expected_fps * _VIDEO_FPS_REL_TOLERANCE,
    )
    fps_error = abs(fps - expected_fps)
    if fps_error > fps_tolerance:
        capture.release()
        raise RuntimeError(
            "decoded video FPS differs from summary output_fps: "
            f"decoded={fps:.9g}, expected={expected_fps:.9g}, "
            f"abs_error={fps_error:.9g}, tolerance={fps_tolerance:.9g}"
        )
    decoded_frames = 0
    shape = None
    digest = hashlib.sha256()
    full_tracker = _new_visual_tracker()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError(
                    f"invalid decoded frame {decoded_frames}: "
                    f"{getattr(frame, 'shape', None)}"
                )
            if shape is None:
                shape = tuple(int(value) for value in frame.shape)
                if wrist_metadata is not None:
                    _validate_roi_bounds(wrist_metadata, shape)
            elif tuple(frame.shape) != shape:
                raise RuntimeError(
                    f"frame shape changed at {decoded_frames}: "
                    f"{frame.shape} != {shape}"
                )
            digest.update(frame.tobytes())

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _update_visual_tracker(full_tracker, gray, decoded_frames)
            if wrist_metadata is not None:
                for side, region in wrist_metadata["regions"].items():
                    x, y, width, height = region["video_xywh"]
                    roi_gray = gray[y : y + height, x : x + width]
                    _update_visual_tracker(
                        wrist_trackers[side],
                        roi_gray,
                        decoded_frames,
                        temporal_margin=4,
                    )
            decoded_frames += 1
    finally:
        capture.release()
    if decoded_frames != expected_frames:
        raise RuntimeError(
            f"decoded {decoded_frames} frames, expected {expected_frames} "
            f"from {summary_path}"
        )
    expected_duration_s = source_frames * playback_action_dt_s
    summary_encoded_duration_s = expected_frames / expected_fps
    decoded_duration_s = decoded_frames / fps
    summary_duration_error_s = abs(
        summary_encoded_duration_s - expected_duration_s
    )
    duration_error_s = abs(decoded_duration_s - expected_duration_s)
    duration_tolerance_s = max(
        _VIDEO_DURATION_ABS_TOLERANCE_S,
        expected_duration_s * _VIDEO_DURATION_REL_TOLERANCE,
    )
    if summary_duration_error_s > duration_tolerance_s:
        raise RuntimeError(
            "summary video duration differs from its action cadence: "
            f"encoded={summary_encoded_duration_s:.9g}s, "
            f"expected={expected_duration_s:.9g}s, "
            f"abs_error={summary_duration_error_s:.9g}s, "
            f"tolerance={duration_tolerance_s:.9g}s"
        )
    if duration_error_s > duration_tolerance_s:
        raise RuntimeError(
            "decoded video duration differs from summary timing: "
            f"decoded={decoded_duration_s:.9g}s, "
            f"expected={expected_duration_s:.9g}s, "
            f"abs_error={duration_error_s:.9g}s, "
            f"tolerance={duration_tolerance_s:.9g}s"
        )

    spatial_checks_applied = bool(
        shape is not None and int(shape[0]) * int(shape[1]) >= _MIN_SPATIAL_CHECK_PIXELS
    )
    full_report = _tracker_report(
        full_tracker,
        decoded_frames=decoded_frames,
        fps=fps,
        spatial_checks_applied=spatial_checks_applied,
    )
    violations: list[str] = []
    warnings: list[str] = []
    if spatial_checks_applied and full_tracker["near_black_indices"]:
        violations.append(
            "near-black frames "
            f"count={len(full_tracker['near_black_indices'])}, "
            f"first={full_tracker['near_black_indices'][0]}, "
            f"p99_luma_threshold={_NEAR_BLACK_P99_LUMA}"
        )
    if spatial_checks_applied and full_tracker["near_uniform_indices"]:
        violations.append(
            "near-uniform frames "
            f"count={len(full_tracker['near_uniform_indices'])}, "
            f"first={full_tracker['near_uniform_indices'][0]}, "
            f"luma_range_threshold={_NEAR_UNIFORM_LUMA_RANGE}, "
            f"luma_std_threshold={_NEAR_UNIFORM_LUMA_STD}"
        )
    if spatial_checks_applied and full_tracker["alternating_indices"]:
        maximum = full_tracker["max_alternating"]
        violations.append(
            "severe alternating stripe/checkerboard artifact "
            f"count={len(full_tracker['alternating_indices'])}, "
            f"first={full_tracker['alternating_indices'][0]}, "
            f"max_score={maximum['score']:.3f}, "
            f"kind={maximum['kind']}, "
            f"block_size_px={maximum['block_size_px']}"
        )
    if full_report["entire_video_frozen"]:
        violations.append(
            f"entire video is frozen across {decoded_frames} frames"
        )
    elif full_report["long_frozen_run"]:
        # A settled robot or an intentional hold can produce a valid long
        # static interval. Without state-motion evidence this is diagnostic,
        # not a hard renderer failure.
        warnings.append(
            "long frozen run "
            f"frames={full_report['max_frozen_run_frames']}, "
            f"threshold={full_report['long_frozen_threshold_frames']}"
        )

    if wrist_metadata is None:
        wrist_report = {
            "checks_applied": False,
            "skip_reason": wrist_skip_reason,
            "violations": [],
            "warnings": [],
        }
    else:
        sustained_min_frames = min(
            decoded_frames,
            max(
                _ROI_SUSTAINED_ARTIFACT_MIN_FRAMES,
                math.ceil(decoded_frames * _ROI_SUSTAINED_ARTIFACT_FRACTION),
            ),
        )
        wrist_report = {
            "checks_applied": True,
            "layout_version": wrist_metadata["version"],
            "sim_panel_x_offset": wrist_metadata["sim_panel_x_offset"],
            "sustained_artifact_min_frames": sustained_min_frames,
            "violations": [],
            "warnings": [],
        }
        for side in ("left", "right"):
            region = wrist_metadata["regions"][side]
            tracker = wrist_trackers[side]
            _, _, width, height = region["video_xywh"]
            region_spatial_checks = (
                width * height >= _MIN_ROI_SPATIAL_CHECK_PIXELS
            )
            region_report = _tracker_report(
                tracker,
                decoded_frames=decoded_frames,
                fps=fps,
                spatial_checks_applied=region_spatial_checks,
            )
            region_violations: list[str] = []
            region_warnings: list[str] = []
            spatial_findings = (
                ("near-black", tracker["near_black_indices"]),
                ("near-uniform", tracker["near_uniform_indices"]),
                (
                    "severe alternating stripe/checkerboard artifact",
                    tracker["alternating_indices"],
                ),
            )
            for label, indices in spatial_findings:
                if region_spatial_checks and len(indices) >= sustained_min_frames:
                    detail = (
                        f"wrist ROI {side} {label} frames count={len(indices)}, "
                        f"first={indices[0]}"
                    )
                    if label.startswith("severe alternating"):
                        maximum = tracker["max_alternating"]
                        detail += (
                            f", max_score={maximum['score']:.3f}, "
                            f"kind={maximum['kind']}, "
                            f"block_size_px={maximum['block_size_px']}"
                        )
                    region_violations.append(detail)
            if region_report["entire_video_frozen"]:
                region_violations.append(
                    f"wrist ROI {side} entire video is frozen across "
                    f"{decoded_frames} frames"
                )
            elif region_report["long_frozen_run"]:
                region_warnings.append(
                    f"wrist ROI {side} long frozen run "
                    f"frames={region_report['max_frozen_run_frames']}, "
                    f"threshold={region_report['long_frozen_threshold_frames']}"
                )
            region_report.update(
                {
                    "ok": not region_violations,
                    "panel_xywh": region["panel_xywh"],
                    "video_xywh": region["video_xywh"],
                    "violations": region_violations,
                    "warnings": region_warnings,
                }
            )
            wrist_report[side] = region_report
            wrist_report["violations"].extend(region_violations)
            wrist_report["warnings"].extend(region_warnings)
        violations.extend(wrist_report["violations"])
        warnings.extend(wrist_report["warnings"])

    visual_quality = {
        "ok": not violations,
        **full_report,
        "fps": fps,
        "thresholds": {
            "minimum_spatial_check_pixels": _MIN_SPATIAL_CHECK_PIXELS,
            "minimum_roi_spatial_check_pixels": _MIN_ROI_SPATIAL_CHECK_PIXELS,
            "near_black_p99_luma": _NEAR_BLACK_P99_LUMA,
            "near_uniform_luma_range": _NEAR_UNIFORM_LUMA_RANGE,
            "near_uniform_luma_std": _NEAR_UNIFORM_LUMA_STD,
            "frozen_mean_abs_delta": _FROZEN_MEAN_ABS_DELTA,
            "frozen_max_abs_delta": _FROZEN_MAX_ABS_DELTA,
            "long_frozen_seconds": _LONG_FROZEN_SECONDS,
            "long_frozen_min_frames": _LONG_FROZEN_MIN_FRAMES,
            "long_frozen_threshold_frames": full_report[
                "long_frozen_threshold_frames"
            ],
            "alternating_pattern_score": _ALTERNATING_PATTERN_THRESHOLD,
            "roi_sustained_artifact_fraction": (
                _ROI_SUSTAINED_ARTIFACT_FRACTION
            ),
            "roi_sustained_artifact_min_frames": (
                _ROI_SUSTAINED_ARTIFACT_MIN_FRAMES
            ),
        },
        "wrist_rois": wrist_report,
        "violations": violations,
        "warnings": warnings,
    }
    result = {
        "ok": not violations,
        "video": str(video_path.resolve()),
        "summary": str(summary_path.resolve()),
        "decoded_frames": decoded_frames,
        "shape_hwc": list(shape) if shape is not None else None,
        "decoded_sha256": digest.hexdigest(),
        "file_size_bytes": video_path.stat().st_size,
        "overview_render_pipeline": overview_pipeline,
        "timing": {
            "summary_output_fps": expected_fps,
            "decoded_fps": fps,
            "fps_abs_error": fps_error,
            "fps_abs_tolerance": fps_tolerance,
            "source_frames": source_frames,
            "playback_action_dt_s": playback_action_dt_s,
            "expected_duration_s": expected_duration_s,
            "summary_encoded_duration_s": summary_encoded_duration_s,
            "summary_duration_abs_error_s": summary_duration_error_s,
            "decoded_duration_s": decoded_duration_s,
            "duration_abs_error_s": duration_error_s,
            "duration_abs_tolerance_s": duration_tolerance_s,
        },
        "visual_quality": visual_quality,
    }
    if violations:
        message = "visual quality validation failed: " + "; ".join(violations)
        result["error"] = message
        raise VideoValidationError(message, result)
    return result


def validate_saved_wrist_insets(
    video_path: Path,
    summary_path: Path,
    expected_rgb_frames,
    *,
    observation_frames=None,
    max_frame_mae: float = _SAVED_WRIST_MAX_FRAME_MAE,
    max_mean_mae: float = _SAVED_WRIST_MAX_MEAN_MAE,
    max_frame_p95: float = _SAVED_WRIST_MAX_FRAME_P95,
    max_frame_p99: float = _SAVED_WRIST_MAX_FRAME_P99,
    bad_pixel_abs_error: int = _SAVED_WRIST_BAD_PIXEL_ABS_ERROR,
    max_bad_pixel_fraction: float = _SAVED_WRIST_MAX_BAD_PIXEL_FRACTION,
) -> dict:
    """Verify every encoded wrist ROI against its exact scheduled source.

    ``expected_rgb_frames`` must contain one ``(left, right)`` pair per output
    frame, already transformed to the declared display geometry. This check is
    intentionally stricter than general visual-quality validation: a side
    swap, flip, rotation, stale chunk selection, or aspect-distorting resize
    all produce a hard failure.
    """

    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise RuntimeError(f"video is missing or empty: {video_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    obs_gs_only = summary.get("obs_gs_only", False)
    if not isinstance(obs_gs_only, bool):
        raise RuntimeError(
            f"exact saved-wrist validation requires boolean obs_gs_only, got "
            f"{obs_gs_only!r}"
        )
    if obs_gs_only:
        raise RuntimeError(
            "exact saved-wrist validation rejects GS-only diagnostic "
            "observations because they are not normal policy tensors"
        )
    if summary.get("video_wrist_inset_source") != "saved":
        raise RuntimeError(
            "exact saved-wrist validation requires "
            "video_wrist_inset_source='saved'"
        )
    metadata, reason = _parse_wrist_roi_metadata(summary)
    if metadata is None:
        raise RuntimeError(
            f"exact saved-wrist validation has no wrist ROI metadata: {reason}"
        )
    layout_version = metadata["version"]
    canonical_layouts = {
        2: _SAVED_WRIST_CANONICAL_LAYOUT_V2,
        3: _SAVED_WRIST_CANONICAL_LAYOUT_V3,
    }
    if layout_version not in canonical_layouts:
        raise RuntimeError(
            "exact saved-wrist validation requires layout version 2 or 3, got "
            f"{layout_version!r}"
        )
    canonical_layout = canonical_layouts[layout_version]
    for side in ("left", "right"):
        observed = metadata["regions"][side]["panel_xywh"]
        canonical = canonical_layout[side]
        if observed != canonical:
            raise RuntimeError(
                "exact saved-wrist validation requires canonical "
                f"v{layout_version} {side} "
                f"ROI {canonical}, got {observed}"
            )
    expected = tuple(expected_rgb_frames)
    recorded_observation_frames = (
        None if observation_frames is None else tuple(observation_frames)
    )
    rendered_frames = summary.get("rendered_frames")
    if (
        isinstance(rendered_frames, bool)
        or not isinstance(rendered_frames, int)
        or rendered_frames <= 0
    ):
        raise RuntimeError(f"invalid rendered_frames={rendered_frames!r}")
    if len(expected) != rendered_frames:
        raise RuntimeError(
            "saved wrist schedule length differs from rendered_frames: "
            f"{len(expected)} != {rendered_frames}"
        )
    if (
        recorded_observation_frames is not None
        and len(recorded_observation_frames) != rendered_frames
    ):
        raise RuntimeError(
            "saved wrist observation-frame schedule length differs from "
            f"rendered_frames: {len(recorded_observation_frames)} != "
            f"{rendered_frames}"
        )
    numeric_thresholds = (
        max_frame_mae,
        max_mean_mae,
        max_frame_p95,
        max_frame_p99,
        max_bad_pixel_fraction,
    )
    if any(not math.isfinite(float(value)) or value < 0.0 for value in numeric_thresholds):
        raise ValueError("saved wrist MAE thresholds must be finite and non-negative")
    if (
        isinstance(bad_pixel_abs_error, bool)
        or not isinstance(bad_pixel_abs_error, int)
        or bad_pixel_abs_error < 0
    ):
        raise ValueError("saved wrist bad-pixel threshold must be a non-negative integer")
    if max_bad_pixel_fraction > 1.0:
        raise ValueError("saved wrist bad-pixel fraction must be at most 1")

    side_errors: dict[str, dict[str, list[float]]] = {
        side: {"mae": [], "p95": [], "p99": [], "bad_fraction": []}
        for side in ("left", "right")
    }
    transition_frames: list[int] = []
    previous_observation_frame = None
    expected_schedule_digest = hashlib.sha256()
    capture = open_video_capture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    decoded = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if decoded >= len(expected):
                raise RuntimeError(
                    f"decoded more frames than saved wrist schedule ({len(expected)})"
                )
            if decoded == 0:
                _validate_roi_bounds(metadata, tuple(frame_bgr.shape))
            pair = expected[decoded]
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise RuntimeError(
                    f"saved wrist schedule frame {decoded} is not a left/right pair"
                )
            observation_frame = (
                recorded_observation_frames[decoded]
                if recorded_observation_frames is not None
                else (id(pair[0]), id(pair[1]))
            )
            expected_schedule_digest.update(decoded.to_bytes(8, "big"))
            expected_schedule_digest.update(
                str(
                    observation_frame
                    if recorded_observation_frames is not None
                    else decoded
                ).encode("utf-8")
            )
            if (
                previous_observation_frame is not None
                and observation_frame != previous_observation_frame
            ):
                transition_frames.append(decoded)
            previous_observation_frame = observation_frame
            for side_index, side in enumerate(("left", "right")):
                source_rgb = np.asarray(pair[side_index])
                x, y, width, height = metadata["regions"][side]["video_xywh"]
                expected_shape = (height, width, 3)
                if source_rgb.dtype != np.uint8 or source_rgb.shape != expected_shape:
                    raise RuntimeError(
                        f"saved wrist schedule frame {decoded} {side} must be "
                        f"uint8 {expected_shape}, got {source_rgb.dtype} "
                        f"{source_rgb.shape}"
                    )
                expected_schedule_digest.update(side.encode("ascii"))
                expected_schedule_digest.update(source_rgb.tobytes())
                encoded_roi = frame_bgr[y : y + height, x : x + width]
                source_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
                absolute_error = np.abs(
                    encoded_roi.astype(np.int16) - source_bgr.astype(np.int16)
                )
                side_errors[side]["mae"].append(float(np.mean(absolute_error)))
                side_errors[side]["p95"].append(
                    float(np.percentile(absolute_error, 95))
                )
                side_errors[side]["p99"].append(
                    float(np.percentile(absolute_error, 99))
                )
                side_errors[side]["bad_fraction"].append(
                    float(np.mean(absolute_error > bad_pixel_abs_error))
                )
            decoded += 1
    finally:
        capture.release()

    if decoded != rendered_frames:
        raise RuntimeError(
            f"decoded {decoded} frames, expected {rendered_frames} for saved wrist check"
        )

    report: dict[str, Any] = {
        "ok": True,
        "contract": "saved_policy_wrist_exact_frame_match_v1",
        "decoded_frames": decoded,
        "layout_version": metadata["version"],
        "transition_frames": transition_frames,
        "expected_schedule_sha256": expected_schedule_digest.hexdigest(),
        "thresholds": {
            "maximum_frame_mean_absolute_error": float(max_frame_mae),
            "mean_frame_mean_absolute_error": float(max_mean_mae),
            "maximum_frame_p95_absolute_error": float(max_frame_p95),
            "maximum_frame_p99_absolute_error": float(max_frame_p99),
            "bad_pixel_absolute_error": int(bad_pixel_abs_error),
            "maximum_frame_bad_pixel_fraction": float(max_bad_pixel_fraction),
        },
    }
    violations: list[str] = []
    for side in ("left", "right"):
        errors = side_errors[side]["mae"]
        worst_index = int(np.argmax(errors))
        maximum = float(errors[worst_index])
        mean_error = float(np.mean(errors))
        maximum_p95 = float(max(side_errors[side]["p95"]))
        maximum_p99 = float(max(side_errors[side]["p99"]))
        maximum_bad_fraction = float(max(side_errors[side]["bad_fraction"]))
        report[side] = {
            "mean_absolute_error": mean_error,
            "maximum_frame_mean_absolute_error": maximum,
            "maximum_frame_p95_absolute_error": maximum_p95,
            "maximum_frame_p99_absolute_error": maximum_p99,
            "maximum_frame_bad_pixel_fraction": maximum_bad_fraction,
            "worst_frame": worst_index,
        }
        if maximum > max_frame_mae:
            violations.append(
                f"saved wrist {side} frame {worst_index} MAE {maximum:.3f} "
                f"exceeds {max_frame_mae:.3f}"
            )
        if mean_error > max_mean_mae:
            violations.append(
                f"saved wrist {side} mean MAE {mean_error:.3f} "
                f"exceeds {max_mean_mae:.3f}"
            )
        if maximum_p95 > max_frame_p95:
            violations.append(
                f"saved wrist {side} frame p95 {maximum_p95:.3f} "
                f"exceeds {max_frame_p95:.3f}"
            )
        if maximum_p99 > max_frame_p99:
            violations.append(
                f"saved wrist {side} frame p99 {maximum_p99:.3f} "
                f"exceeds {max_frame_p99:.3f}"
            )
        if maximum_bad_fraction > max_bad_pixel_fraction:
            violations.append(
                f"saved wrist {side} bad-pixel fraction "
                f"{maximum_bad_fraction:.5f} exceeds "
                f"{max_bad_pixel_fraction:.5f}"
            )
    report["violations"] = violations
    if violations:
        report["ok"] = False
        raise VideoValidationError(
            "exact saved-wrist validation failed: " + "; ".join(violations),
            report,
        )
    return report


def write_json_atomic(path: Path, payload: dict) -> None:
    """Publish a JSON report atomically so interrupted jobs leave no partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validation_failure_result(
    video_path: Path,
    summary_path: Path,
    error: Exception,
) -> dict:
    if isinstance(error, VideoValidationError):
        result = dict(error.result)
    else:
        result = {
            "video": str(video_path.resolve()),
            "summary": str(summary_path.resolve()),
            "error": f"{type(error).__name__}: {error}",
        }
    result["ok"] = False
    result.setdefault("video", str(video_path.resolve()))
    result.setdefault("summary", str(summary_path.resolve()))
    result.setdefault("error", f"{type(error).__name__}: {error}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = validate_video(arguments.video, arguments.summary)
    except Exception as exc:
        result = _validation_failure_result(
            arguments.video,
            arguments.summary,
            exc,
        )
        write_json_atomic(arguments.output, result)
        print(json.dumps(result, sort_keys=True), file=sys.stderr)
        return 1
    write_json_atomic(arguments.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
