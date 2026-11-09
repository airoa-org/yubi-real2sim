# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from robosuite_env.tools import validate_rollout_video as validator
from robosuite_env.tools.validate_rollout_video import (
    VideoValidationError,
    validate_saved_wrist_insets,
    validate_video,
)


def _write_video(path: Path, frames: int, *, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (32, 24)
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((24, 32, 3), (index * 20) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_frames(path: Path, frames: list[np.ndarray], *, fps: float = 10.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    assert writer.isOpened()
    try:
        for frame in frames:
            assert frame.shape == (height, width, 3)
            writer.write(frame)
    finally:
        writer.release()


def _summary(frame_count: int, *, fps: float = 10.0, **extra) -> dict:
    return {
        "frames": frame_count,
        "rendered_frames": frame_count,
        "output_fps": fps,
        "playback_action_dt_s": 1.0 / fps,
        **extra,
    }


def _textured_frame(index: int, *, height: int = 72, width: int = 96) -> np.ndarray:
    yy, xx = np.indices((height, width))
    frame = np.stack(
        (
            (xx * 3 + index * 7) % 256,
            (yy * 5 + index * 11) % 256,
            ((xx + yy) * 2 + index * 13) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    cv2.rectangle(frame, (index % 20, 8), (index % 20 + 15, 24), (20, 220, 80), -1)
    return frame


@pytest.mark.parametrize(("frame_count", "fps"), ((4, 10.0), (300, 30.0)))
def test_validate_video_decodes_every_expected_frame(
    tmp_path: Path,
    frame_count: int,
    fps: float,
):
    video = tmp_path / f"rollout_{int(fps)}hz.avi"
    summary = tmp_path / "summary.json"
    _write_video(video, frame_count, fps=fps)
    summary.write_text(
        json.dumps(_summary(frame_count, fps=fps)),
        encoding="utf-8",
    )

    result = validate_video(video, summary)

    assert result["ok"] is True
    assert result["decoded_frames"] == frame_count
    assert result["shape_hwc"] == [24, 32, 3]
    assert result["visual_quality"]["ok"] is True
    assert result["visual_quality"]["spatial_checks_applied"] is False
    duration = frame_count / fps
    timing = result["timing"]
    assert timing["summary_output_fps"] == fps
    assert timing["decoded_fps"] == pytest.approx(fps)
    assert timing["expected_duration_s"] == pytest.approx(duration)
    assert timing["summary_encoded_duration_s"] == pytest.approx(duration)
    assert timing["decoded_duration_s"] == pytest.approx(duration)
    assert timing["duration_abs_error_s"] <= timing["duration_abs_tolerance_s"]


def test_validate_video_rejects_container_fps_mismatch(tmp_path: Path):
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    _write_video(video, 12, fps=10.0)
    summary.write_text(json.dumps(_summary(12, fps=30.0)), encoding="utf-8")

    with pytest.raises(RuntimeError, match="FPS differs from summary output_fps"):
        validate_video(video, summary)


@pytest.mark.parametrize("bad_fps", [None, True, 0, -1, float("nan"), float("inf"), "10"])
def test_validate_video_rejects_invalid_summary_output_fps(
    tmp_path: Path, bad_fps: object
):
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    _write_video(video, 4)
    payload = _summary(4)
    payload["output_fps"] = bad_fps
    summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="output_fps must be a positive finite"):
        validate_video(video, summary)


@pytest.mark.parametrize("bad_fps", [0.0, float("nan"), float("inf")])
def test_validate_video_rejects_invalid_decoded_fps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_fps: float
):
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    _write_video(video, 4)
    summary.write_text(json.dumps(_summary(4)), encoding="utf-8")
    real_capture = cv2.VideoCapture

    class InvalidFpsCapture:
        def __init__(self, *args):
            self._capture = real_capture(*args)

        def isOpened(self):
            return self._capture.isOpened()

        def get(self, property_id):
            if property_id == cv2.CAP_PROP_FPS:
                return bad_fps
            return self._capture.get(property_id)

        def read(self):
            return self._capture.read()

        def release(self):
            return self._capture.release()

    monkeypatch.setattr(validator.cv2, "VideoCapture", InvalidFpsCapture)

    with pytest.raises(RuntimeError, match="decoded video FPS must be"):
        validate_video(video, summary)


def test_validate_video_rejects_action_cadence_duration_mismatch(tmp_path: Path):
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    _write_video(video, 30, fps=30.0)
    payload = _summary(30, fps=30.0)
    payload["frames"] = 60
    summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        RuntimeError, match="duration differs from its action cadence"
    ):
        validate_video(video, summary)


def test_validate_video_rejects_truncated_frame_count(tmp_path: Path):
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    _write_video(video, 2)
    summary.write_text(json.dumps(_summary(3)), encoding="utf-8")

    with pytest.raises(RuntimeError, match="decoded 2 frames, expected 3"):
        validate_video(video, summary)


@pytest.mark.parametrize(
    ("pipeline", "accepted"),
    (
        ("inline_egl", False),
        ("deferred_isolated_egl_after_wrist_teardown", True),
    ),
)
def test_validate_video_rejects_inline_overview_for_batch_policy(
    tmp_path: Path,
    pipeline: str,
    accepted: bool,
):
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    _write_video(video, 2)
    summary.write_text(
        json.dumps(
            _summary(
                2,
                policy_batch_server="127.0.0.1:8765",
                overview_render_pipeline=pipeline,
            )
        ),
        encoding="utf-8",
    )

    if not accepted:
        with pytest.raises(RuntimeError, match="unsafe overview render pipeline"):
            validate_video(video, summary)
        return

    result = validate_video(video, summary)
    assert result["ok"] is True
    assert result["overview_render_pipeline"] == pipeline


def test_validate_video_reports_visual_quality_for_valid_motion(tmp_path: Path):
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    frames = [_textured_frame(index) for index in range(12)]
    _write_frames(video, frames)
    summary.write_text(json.dumps(_summary(len(frames))), encoding="utf-8")

    result = validate_video(video, summary)

    quality = result["visual_quality"]
    assert quality["ok"] is True
    assert quality["spatial_checks_applied"] is True
    assert quality["near_black"]["count"] == 0
    assert quality["near_uniform"]["count"] == 0
    assert quality["alternating_artifacts"]["count"] == 0
    assert quality["entire_video_frozen"] is False


@pytest.mark.parametrize(
    ("bad_frame", "message"),
    [
        (np.zeros((72, 96, 3), dtype=np.uint8), "near-black frames"),
        (np.full((72, 96, 3), 127, dtype=np.uint8), "near-uniform frames"),
    ],
)
def test_validate_video_rejects_catastrophic_blank_frame(
    tmp_path: Path, bad_frame: np.ndarray, message: str
):
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    frames = [_textured_frame(index) for index in range(12)]
    frames[6] = bad_frame
    _write_frames(video, frames)
    summary.write_text(json.dumps(_summary(len(frames))), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        validate_video(video, summary)


def test_validate_video_rejects_entire_frozen_video(tmp_path: Path):
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    frame = _textured_frame(3)
    frames = [frame.copy() for _ in range(12)]
    _write_frames(video, frames)
    summary.write_text(json.dumps(_summary(len(frames))), encoding="utf-8")

    with pytest.raises(RuntimeError, match="entire video is frozen"):
        validate_video(video, summary)


def test_validate_video_reports_long_frozen_run_as_warning(tmp_path: Path):
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    frozen = _textured_frame(4)
    frames = (
        [_textured_frame(index) for index in range(5)]
        + [frozen.copy() for _ in range(52)]
        + [_textured_frame(index) for index in range(5, 10)]
    )
    _write_frames(video, frames, fps=10.0)
    summary.write_text(json.dumps(_summary(len(frames))), encoding="utf-8")

    result = validate_video(video, summary)

    assert result["ok"] is True
    assert result["visual_quality"]["long_frozen_run"] is True
    assert any(
        "long frozen run" in warning
        for warning in result["visual_quality"]["warnings"]
    )


@pytest.mark.parametrize("artifact", ("stripes", "checkerboard"))
def test_validate_video_rejects_severe_alternating_stripes(
    tmp_path: Path,
    artifact: str,
):
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    frames = [_textured_frame(index) for index in range(12)]
    if artifact == "stripes":
        corrupted = np.zeros((72, 96, 3), dtype=np.uint8)
        for row in range(72):
            corrupted[row, :, :] = 255 if (row // 4) % 2 else 0
        message = "alternating stripe/checkerboard"
    else:
        yy, xx = np.indices((72, 96))
        checker = ((((xx // 4) + (yy // 4)) % 2) * 255).astype(np.uint8)
        corrupted = np.repeat(checker[:, :, None], 3, axis=2)
        message = r"alternating stripe/checkerboard.*kind=checkerboard"
    frames[7] = corrupted
    _write_frames(video, frames)
    summary.write_text(json.dumps(_summary(len(frames))), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        validate_video(video, summary)


_WRIST_LAYOUT = {
    "version": 1,
    "left_xywh": [16, 16, 48, 32],
    "right_xywh": [80, 16, 48, 32],
}


def _wrist_summary(frame_count: int, *, offset: int = 0) -> dict:
    return _summary(
        frame_count,
        video_wrist_insets_enabled=True,
        video_sim_panel_x_offset=offset,
        video_wrist_inset_layout=_WRIST_LAYOUT,
    )


def _wrist_test_frames(frame_count: int, *, width: int = 192) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    yy, xx = np.indices((128, width))
    base = np.stack(
        ((xx * 3) % 256, (yy * 5) % 256, ((xx + yy) * 2) % 256), axis=-1
    ).astype(np.uint8)
    for index in range(frame_count):
        frame = base.copy()
        cv2.rectangle(
            frame,
            (8 + index * 3, 88),
            (26 + index * 3, 110),
            (20, 230, 90),
            -1,
        )
        # Both insets normally have independent motion; tests replace one ROI
        # below while leaving the overview and the other inset healthy.
        cv2.circle(frame, (28 + index % 20, 31), 7, (240, 30, 180), -1)
        cv2.circle(frame, (92 + (index * 2) % 24, 31), 7, (30, 230, 220), -1)
        frames.append(frame)
    return frames


def test_validate_video_skips_wrist_rois_for_legacy_summary(tmp_path: Path):
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    frames = _wrist_test_frames(12)
    _write_frames(video, frames)
    summary.write_text(json.dumps(_summary(12)), encoding="utf-8")

    result = validate_video(video, summary)

    wrist = result["visual_quality"]["wrist_rois"]
    assert wrist["checks_applied"] is False
    assert wrist["skip_reason"] == "metadata_not_declared"


def test_validate_video_checks_wrist_rois_with_sim_panel_offset(tmp_path: Path):
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    frames = _wrist_test_frames(12, width=320)
    offset = 128
    for index, frame in enumerate(frames):
        cv2.circle(frame, (offset + 28 + index % 20, 31), 7, (220, 40, 190), -1)
        cv2.circle(
            frame,
            (offset + 92 + (index * 2) % 24, 31),
            7,
            (40, 220, 230),
            -1,
        )
    _write_frames(video, frames)
    summary.write_text(
        json.dumps(_wrist_summary(len(frames), offset=offset)), encoding="utf-8"
    )

    result = validate_video(video, summary)

    wrist = result["visual_quality"]["wrist_rois"]
    assert wrist["checks_applied"] is True
    assert wrist["left"]["video_xywh"] == [144, 16, 48, 32]
    assert wrist["right"]["video_xywh"] == [208, 16, 48, 32]
    assert wrist["violations"] == []


def test_validate_video_rejects_black_wrist_roi(tmp_path: Path):
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    frames = _wrist_test_frames(12)
    x, y, width, height = _WRIST_LAYOUT["left_xywh"]
    for frame in frames:
        frame[y : y + height, x : x + width] = 0
    _write_frames(video, frames)
    summary.write_text(
        json.dumps(_wrist_summary(len(frames))), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="wrist ROI left near-black"):
        validate_video(video, summary)


def test_validate_video_rejects_striped_wrist_roi(tmp_path: Path):
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    frames = _wrist_test_frames(12)
    x, y, width, height = _WRIST_LAYOUT["right_xywh"]
    stripes = np.zeros((height, width, 3), dtype=np.uint8)
    for column in range(width):
        stripes[:, column, :] = 255 if (column // 4) % 2 else 0
    for frame in frames:
        frame[y : y + height, x : x + width] = stripes
    _write_frames(video, frames)
    summary.write_text(
        json.dumps(_wrist_summary(len(frames))), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="wrist ROI right severe alternating"):
        validate_video(video, summary)


def test_validate_video_rejects_frozen_wrist_roi_with_moving_overview(
    tmp_path: Path,
):
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    frames = _wrist_test_frames(12)
    x, y, width, height = _WRIST_LAYOUT["left_xywh"]
    frozen = frames[0][y : y + height, x : x + width].copy()
    for frame in frames:
        frame[y : y + height, x : x + width] = frozen
    _write_frames(video, frames)
    summary.write_text(
        json.dumps(_wrist_summary(len(frames))), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="wrist ROI left entire video is frozen"):
        validate_video(video, summary)


def test_validate_video_accepts_repeated_policy_hold_runs(tmp_path: Path):
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    frames = _wrist_test_frames(40)
    x, y, width, height = _WRIST_LAYOUT["left_xywh"]
    for start in range(0, len(frames), 8):
        held = frames[start][y : y + height, x : x + width].copy()
        for frame in frames[start : start + 8]:
            frame[y : y + height, x : x + width] = held
    _write_frames(video, frames)
    summary.write_text(
        json.dumps(_wrist_summary(len(frames))), encoding="utf-8"
    )

    result = validate_video(video, summary)

    left = result["visual_quality"]["wrist_rois"]["left"]
    assert left["entire_video_frozen"] is False
    assert left["max_frozen_run_frames"] < len(frames)


def _asymmetric_saved_wrist(index: int, side: int) -> np.ndarray:
    yy, xx = np.indices((135, 180))
    image = np.stack(
        (
            20 + xx // 2 + index * 15 + side * 10,
            30 + yy + index * 12 + side * 15,
            40 + (xx + yy) // 2 + index * 10 + side * 20,
        ),
        axis=-1,
    ).astype(np.uint8)
    image[:25, :35] = (180, 50 + side * 20, 70 + index)
    image[-30:, -40:] = (60 + index, 180, 100 + side * 40)
    return image


def _write_exact_wrist_fixture(
    video: Path,
    summary: Path,
    schedule: list[tuple[np.ndarray, np.ndarray]],
    *,
    corrupt: str | None = None,
    layout_version: int = 2,
) -> None:
    yy, xx = np.indices((480, 640))
    base = np.stack(
        ((xx + yy) % 256, (2 * yy + xx) % 256, (xx + 2 * yy) % 256),
        axis=-1,
    ).astype(np.uint8)
    frames = [base.copy() for _ in schedule]
    left_x, left_y, width, height = (4, 4, 180, 135)
    right_x, right_y, _, _ = (188, 4, 180, 135)
    for index, (left_rgb, right_rgb) in enumerate(schedule):
        if corrupt == "swap":
            left_rgb, right_rgb = right_rgb, left_rgb
        elif corrupt == "flip_left":
            left_rgb = np.flip(left_rgb, axis=1)
        elif corrupt == "rgb_bgr":
            left_rgb = left_rgb[..., ::-1]
        elif corrupt == "old_square_stretch":
            square = np.zeros((224, 224, 3), dtype=np.uint8)
            square[28:195] = cv2.resize(left_rgb, (224, 167))
            left_rgb = cv2.resize(square, (180, 135))
        elif corrupt == "stale_boundary" and index == 2:
            left_rgb, right_rgb = schedule[index - 1]
        elif corrupt == "localized_left" and index == 2:
            left_rgb = left_rgb.copy()
            left_rgb[45:90, 60:120] = 255 - left_rgb[45:90, 60:120]
        frames[index][left_y : left_y + height, left_x : left_x + width] = cv2.cvtColor(
            left_rgb, cv2.COLOR_RGB2BGR
        )
        frames[index][right_y : right_y + height, right_x : right_x + width] = cv2.cvtColor(
            right_rgb, cv2.COLOR_RGB2BGR
        )
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 480)
    )
    assert writer.isOpened()
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    payload = _summary(
        len(frames),
        video_wrist_insets_enabled=True,
        video_sim_panel_x_offset=0,
    )
    payload["video_wrist_inset_source"] = "saved"
    payload["video_wrist_inset_layout"] = {
        "version": layout_version,
        "left_xywh": [4, 4, 180, 135],
        "right_xywh": [188, 4, 180, 135],
    }
    summary.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_saved_wrist_insets_checks_every_chunk_boundary(tmp_path: Path):
    video = tmp_path / "rollout.mp4"
    summary = tmp_path / "summary.json"
    pairs = [
        (_asymmetric_saved_wrist(index, 0), _asymmetric_saved_wrist(index, 1))
        for index in range(3)
    ]
    schedule = [pairs[0]] * 2 + [pairs[1]] * 2 + [pairs[2]] * 2
    _write_exact_wrist_fixture(video, summary, schedule)

    report = validate_saved_wrist_insets(
        video,
        summary,
        schedule,
        observation_frames=[0, 0, 2, 2, 4, 4],
    )

    assert report["ok"] is True
    assert report["transition_frames"] == [2, 4]
    assert report["left"]["worst_frame"] in range(6)


def test_validate_saved_wrist_insets_accepts_raw224_layout_v3(tmp_path: Path):
    video = tmp_path / "raw224.mp4"
    summary = tmp_path / "summary.json"
    pairs = [
        (_asymmetric_saved_wrist(index, 0), _asymmetric_saved_wrist(index, 1))
        for index in range(3)
    ]
    _write_exact_wrist_fixture(video, summary, pairs, layout_version=3)

    report = validate_saved_wrist_insets(video, summary, pairs)

    assert report["ok"] is True
    assert report["layout_version"] == 3


def test_validate_saved_wrist_insets_rejects_gs_only_diagnostic(tmp_path: Path):
    video = tmp_path / "gs-only.mp4"
    summary = tmp_path / "summary.json"
    pairs = [(_asymmetric_saved_wrist(0, 0), _asymmetric_saved_wrist(0, 1))]
    _write_exact_wrist_fixture(video, summary, pairs, layout_version=3)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["obs_gs_only"] = True
    summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="GS-only diagnostic"):
        validate_saved_wrist_insets(video, summary, pairs)


@pytest.mark.parametrize(
    "corrupt",
    [
        "swap",
        "flip_left",
        "rgb_bgr",
        "old_square_stretch",
        "stale_boundary",
        "localized_left",
    ],
)
def test_validate_saved_wrist_insets_rejects_side_or_orientation_bug(
    tmp_path: Path, corrupt: str
):
    video = tmp_path / f"{corrupt}.mp4"
    summary = tmp_path / "summary.json"
    pairs = [
        (_asymmetric_saved_wrist(index, 0), _asymmetric_saved_wrist(index, 1))
        for index in range(4)
    ]
    _write_exact_wrist_fixture(video, summary, pairs, corrupt=corrupt)

    with pytest.raises(VideoValidationError, match="exact saved-wrist"):
        validate_saved_wrist_insets(
            video,
            summary,
            pairs,
        )


def test_validate_saved_wrist_insets_rejects_noncanonical_v2_roi(tmp_path: Path):
    video = tmp_path / "wrong-roi.mp4"
    summary = tmp_path / "summary.json"
    pairs = [(_asymmetric_saved_wrist(0, 0), _asymmetric_saved_wrist(0, 1))]
    _write_exact_wrist_fixture(video, summary, pairs)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["video_wrist_inset_layout"]["left_xywh"] = [5, 4, 180, 135]
    summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="canonical v2 left ROI"):
        validate_saved_wrist_insets(video, summary, pairs)


def test_cli_writes_atomic_failure_report_for_quality_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    output = tmp_path / "reports" / "validation.json"
    frames = [_textured_frame(index) for index in range(12)]
    frames[6] = np.zeros((72, 96, 3), dtype=np.uint8)
    _write_frames(video, frames)
    summary.write_text(
        json.dumps(_summary(len(frames))),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        validator.sys,
        "argv",
        [
            "validate_rollout_video.py",
            str(video),
            "--summary",
            str(summary),
            "--output",
            str(output),
        ],
    )

    assert validator.main() == 1

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["visual_quality"]["ok"] is False
    assert "near-black frames" in report["error"]
    assert not list(output.parent.glob(".validation.json.*.tmp"))


def test_cli_writes_atomic_failure_report_for_corrupt_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "rollout.avi"
    summary = tmp_path / "summary.json"
    output = tmp_path / "validation.json"
    _write_video(video, 1)
    summary.write_text("{not-json", encoding="utf-8")
    output.write_text('{"old": true}\n', encoding="utf-8")
    monkeypatch.setattr(
        validator.sys,
        "argv",
        [
            "validate_rollout_video.py",
            str(video),
            "--summary",
            str(summary),
            "--output",
            str(output),
        ],
    )

    assert validator.main() == 1

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["video"] == str(video.resolve())
    assert report["summary"] == str(summary.resolve())
    assert report["error"].startswith("JSONDecodeError:")
    assert not list(tmp_path.glob(".validation.json.*.tmp"))


def test_cli_writes_atomic_failure_report_for_missing_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "missing.avi"
    summary = tmp_path / "summary.json"
    output = tmp_path / "validation.json"
    summary.write_text('{"rendered_frames": 1}\n', encoding="utf-8")
    monkeypatch.setattr(
        validator.sys,
        "argv",
        [
            "validate_rollout_video.py",
            str(video),
            "--summary",
            str(summary),
            "--output",
            str(output),
        ],
    )

    assert validator.main() == 1

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["video"] == str(video.resolve())
    assert "video is missing or empty" in report["error"]
    assert not list(tmp_path.glob(".validation.json.*.tmp"))
