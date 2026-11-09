# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from robosuite_env.tools import rewrite_saved_policy_obs_video as rewrite


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (640, 480),
    )
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def _wrist_pair() -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros((135, 180, 3), dtype=np.uint8)
    right = np.zeros((135, 180, 3), dtype=np.uint8)
    left[17:-17] = (10, 80, 180)
    right[17:-17] = (180, 80, 10)
    return left, right


def test_compose_raw224_video_replaces_only_declared_wrist_rois(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    base = np.full((480, 640, 3), 90, dtype=np.uint8)
    _write_video(source, [base, base])

    left, right = _wrist_pair()
    pair = (left, right)
    report = rewrite._compose_raw224_video(
        source,
        output,
        (pair, pair),
        sim_panel_x_offset=0,
    )

    assert report["frames"] == 2
    assert report["unique_display_pairs"] == 1
    capture = cv2.VideoCapture(str(output))
    ok, decoded = capture.read()
    capture.release()
    assert ok
    source_capture = cv2.VideoCapture(str(source))
    source_ok, decoded_source = source_capture.read()
    source_capture.release()
    assert source_ok
    for expected, x in ((left, 4), (right, 188)):
        roi = decoded[4:139, x : x + 180]
        expected_bgr = cv2.cvtColor(expected, cv2.COLOR_RGB2BGR)
        assert float(np.mean(np.abs(roi.astype(int) - expected_bgr.astype(int)))) < 5.0
        assert float(np.mean(roi[:15])) < 3.0
        assert float(np.mean(roi[-15:])) < 3.0
    assert (
        float(
            np.mean(
                np.abs(
                    decoded[250:, 400:].astype(int)
                    - decoded_source[250:, 400:].astype(int)
                )
            )
        )
        < 5.0
    )


def test_compose_raw224_video_rejects_a_schedule_length_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    base = np.full((480, 640, 3), 90, dtype=np.uint8)
    _write_video(source, [base, base])

    with pytest.raises(rewrite.RewriteError, match="more frames than saved OBS"):
        rewrite._compose_raw224_video(
            source,
            tmp_path / "short.mp4",
            (_wrist_pair(),),
            sim_panel_x_offset=0,
        )


def test_compose_raw224_video_rejects_a_frame_too_small_for_the_rois(
    tmp_path: Path,
) -> None:
    source = tmp_path / "small.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240)
    )
    assert writer.isOpened()
    writer.write(np.full((240, 320, 3), 90, dtype=np.uint8))
    writer.release()

    with pytest.raises(rewrite.RewriteError, match="too small for saved wrist ROIs"):
        rewrite._compose_raw224_video(
            source,
            tmp_path / "out.mp4",
            (_wrist_pair(),),
            sim_panel_x_offset=0,
        )


def test_raw_summary_declares_full_policy_tensor_contract() -> None:
    summary = {
        "video_wrist_inset_source": "saved",
        "video_wrist_inset_layout": {"version": 2},
    }

    updated = rewrite._raw_summary(summary)

    assert updated["video_wrist_inset_layout"] == rewrite.RAW_LAYOUT
    assert updated["video_wrist_display_contract"] == rewrite.RAW_DISPLAY_CONTRACT
    assert updated["state_replay_policy_obs_display"] == "raw224"
    for transform in updated["video_wrist_display_transforms"].values():
        assert transform["policy_content_xywh"] == [0, 0, 224, 224]
        assert transform["output_hw"] == [135, 180]
    # The caller's summary must not be mutated in place.
    assert summary["video_wrist_inset_layout"] == {"version": 2}


def test_raw_summary_rejects_a_non_saved_wrist_video() -> None:
    with pytest.raises(rewrite.RewriteError, match="not a saved-policy-OBS video"):
        rewrite._raw_summary({"video_wrist_inset_source": "live"})


def test_raw_summary_rejects_gs_only_diagnostic_observations() -> None:
    with pytest.raises(rewrite.RewriteError, match="GS-only diagnostic"):
        rewrite._raw_summary(
            {
                "video_wrist_inset_source": "saved",
                "obs_gs_only": True,
            }
        )


def test_selected_runs_requires_explicit_run_names(tmp_path: Path) -> None:
    scene_dir = tmp_path / "franka_flat"
    (scene_dir / "run_01").mkdir(parents=True)
    (scene_dir / "run_02").mkdir()

    assert rewrite._selected_runs(tmp_path, "franka_flat", "") == [
        scene_dir / "run_01",
        scene_dir / "run_02",
    ]
    assert rewrite._selected_runs(tmp_path, "franka_flat", "run_02") == [
        scene_dir / "run_02"
    ]
    with pytest.raises(rewrite.RewriteError, match="comma-separated run_"):
        rewrite._selected_runs(tmp_path, "franka_flat", "run_02,other")
    with pytest.raises(rewrite.RewriteError, match="duplicate entries"):
        rewrite._selected_runs(tmp_path, "franka_flat", "run_01,run_01")
    with pytest.raises(rewrite.RewriteError, match="do not exist"):
        rewrite._selected_runs(tmp_path, "franka_flat", "run_09")
    with pytest.raises(rewrite.RewriteError, match="scene directory does not exist"):
        rewrite._selected_runs(tmp_path, "scene9", "")


def test_rewrite_run_requires_the_full_rerender_artifact_set(tmp_path: Path) -> None:
    run_dir = tmp_path / "franka_flat" / "run_01"
    run_dir.mkdir(parents=True)

    with pytest.raises(rewrite.RewriteError, match="missing video"):
        rewrite.rewrite_run(run_dir)


def test_is_published_raw224_requires_an_exact_layout_and_validation(
    tmp_path: Path,
) -> None:
    video = tmp_path / "franka_flat_run_01.mp4"
    video.write_bytes(b"\x00" * 16)
    validation = tmp_path / "video_validation.json"
    summary = {
        "video_wrist_display_contract": rewrite.RAW_DISPLAY_CONTRACT,
        "video_wrist_inset_layout": dict(rewrite.RAW_LAYOUT),
    }

    assert rewrite._is_published_raw224(video, summary, validation) is False

    validation.write_text(
        '{"ok": true, "saved_wrist_exact_match": '
        '{"ok": true, "layout_version": 3}}',
        encoding="utf-8",
    )
    assert rewrite._is_published_raw224(video, summary, validation) is True

    stale = dict(summary)
    stale["video_wrist_inset_layout"] = {"version": 2}
    assert rewrite._is_published_raw224(video, stale, validation) is False


"""Contract tests for republishing an inline-rendered generation run."""

import json

from robosuite_env.tools import make_rollout_grid_video as grid
from tests._helpers.rollouts import write_rollout


def _generation_run(tmp_path: Path, *, wrist_display: str) -> Path:
    run_root = tmp_path / "run"
    write_rollout(
        run_root,
        1,
        frames=4,
        saved_policy_obs=True,
        exec_steps=2,
        wrist_display=wrist_display,
        textured=True,
    )
    return run_root / "test_scene" / "run_01"


def test_rewrite_publishes_a_generation_sidecar_without_editing_the_summary(
    tmp_path: Path,
) -> None:
    run_dir = _generation_run(tmp_path, wrist_display="presentation_crop")
    generation_summary = run_dir / "sim_summary.json"
    before = generation_summary.read_bytes()
    video = run_dir / "test_scene_run_01.mp4"

    report = rewrite.rewrite_run(run_dir)

    assert report["mode"] == "generation"
    assert report["skipped"] is False
    assert report["contract"] == rewrite.RAW_DISPLAY_CONTRACT
    # The generation record is provenance and must survive untouched.
    assert generation_summary.read_bytes() == before

    sidecar = run_dir / grid.SAVED_POLICY_OBS_DIRNAME
    published = json.loads((sidecar / "sim_summary.json").read_text())
    assert published["video_wrist_inset_source"] == "saved"
    assert published["video_wrist_inset_source_generated"] == "policy_observation"
    assert published["video_wrist_inset_layout"] == rewrite.RAW_LAYOUT
    assert published["video_wrist_display_contract"] == (
        rewrite.RAW_DISPLAY_CONTRACT
    )
    validation = json.loads((sidecar / "video_validation.json").read_text())
    assert validation["ok"] is True
    assert validation["saved_wrist_exact_match"]["ok"] is True
    assert validation["saved_wrist_exact_match"]["layout_version"] == 3
    assert validation["raw224_policy_obs_overlay"]["frames"] == 4
    provenance = json.loads((sidecar / "provenance.json").read_text())
    assert provenance["mode"] == "generated-video+raw224-policy-obs-overlay"
    assert provenance["postprocess"]["source_summary"]["path"] == str(
        generation_summary.resolve()
    )

    # The grid renderer must now accept the republished video.
    grid.validate_saved_policy_obs_inputs([video])


def test_rewrite_of_a_generation_run_is_idempotent(tmp_path: Path) -> None:
    run_dir = _generation_run(tmp_path, wrist_display="presentation_crop")

    first = rewrite.rewrite_run(run_dir)
    digest = rewrite._sha256(run_dir / "test_scene_run_01.mp4")
    second = rewrite.rewrite_run(run_dir)

    assert first["skipped"] is False
    assert second["skipped"] is True
    assert second["mode"] == "generation"
    assert rewrite._sha256(run_dir / "test_scene_run_01.mp4") == digest


def test_rewrite_skips_a_rollout_without_wrist_insets(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_rollout(run_root, 1, frames=2)
    run_dir = run_root / "test_scene" / "run_01"

    report = rewrite.rewrite_run(run_dir)

    assert report == {
        "ok": True,
        "skipped": True,
        "reason": "wrist_insets_disabled",
        "mode": "generation",
        "run": "run_01",
        "video": str(run_dir / "test_scene_run_01.mp4"),
        "summary": str(run_dir / "sim_summary.json"),
        "contract": None,
    }
    assert not (run_dir / grid.SAVED_POLICY_OBS_DIRNAME).exists()


def test_raw_summary_records_the_generated_wrist_source() -> None:
    updated = rewrite._raw_summary(
        {"video_wrist_inset_source": "policy_observation"}
    )

    assert updated["video_wrist_inset_source"] == "saved"
    assert updated["video_wrist_inset_source_generated"] == "policy_observation"
