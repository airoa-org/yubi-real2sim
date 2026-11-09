# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0


import json

from pathlib import Path



import numpy as np

import pytest



import robosuite_env.tools.rerender_third_person as rerender






from robosuite_env.rerender import validation as rerender_validation

from tests._helpers.rollouts import (
    _configure_fr3_policy_action_source,
    _configure_frame_hz_source,
    _source_registration_summary,
    run_dir,  # noqa: F401 -- pytest fixture
)

def test_validate_rerender_output_writes_atomic_success_report(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = run_dir / "scene_run_01.mp4"
    video.write_bytes(b"video")
    summary = run_dir / "rerender" / "sim_summary.json"
    summary.parent.mkdir(exist_ok=True)
    summary.write_text(
        json.dumps(_source_registration_summary(run_dir)), encoding="utf-8"
    )
    expected = {"ok": True, "decoded_frames": 1}
    calls = []

    def fake_validate(video_path: Path, summary_path: Path) -> dict:
        calls.append((video_path, summary_path))
        return expected

    monkeypatch.setattr(rerender_validation, "validate_video", fake_validate)

    result = rerender.validate_rerender_output(run_dir, video)

    assert result == expected
    assert calls == [(video, summary)]
    report = run_dir / "rerender" / "video_validation.json"
    assert json.loads(report.read_text(encoding="utf-8")) == expected
    assert not list(report.parent.glob(".video_validation.json.*.tmp"))

def test_validate_rerender_output_rejects_registration_path_drift(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = run_dir / "scene_run_01.mp4"
    video.write_bytes(b"video")
    alternate = run_dir / "alternate_w2gs.npy"
    np.save(alternate, np.eye(4, dtype=np.float64))
    replay_summary = _source_registration_summary(run_dir)
    replay_summary["gs_w2gs"] = str(alternate.resolve())
    summary = run_dir / "rerender" / "sim_summary.json"
    summary.parent.mkdir(exist_ok=True)
    summary.write_text(json.dumps(replay_summary), encoding="utf-8")
    monkeypatch.setattr(
        rerender_validation,
        "validate_video",
        lambda *_: {"ok": True, "decoded_frames": 1},
    )

    with pytest.raises(
        rerender.RerenderError,
        match="registration path or SHA-256 differs from generation",
    ):
        rerender.validate_rerender_output(run_dir, video)

def _exact_policy_action_replay_summary(run_dir: Path, frames: int = 300) -> dict:
    return {
        **_source_registration_summary(run_dir),
        "robot": "fr3",
        "physics": "dynamic",
        "frames": frames,
        "rendered_frames": frames,
        "state_log_replay": str(
            (run_dir / "logs" / "sim_log.jsonl").resolve()
        ),
        "replay_arm_source": "generation_state_log",
        "replay_object_source": "generation_state_log",
        "policy_action_hz_requested": 0.0,
        "policy_action_hz_effective": None,
        "policy_action_dt_s": None,
        "policy_action_dt_nominal_s": None,
        "frame_hz_effective": 30.0,
        "frame_dt_s": round(1.0 / 30.0, 6),
        "output_fps": 30,
        "playback_action_dt_s": round(1.0 / 30.0, 6),
        "dynamic_subframes": 1,
        "dynamic_action_sim_dt_s": None,
        "dynamic_control_repeats": None,
        "video_wrist_inset_source": "off",
    }

def _exact_video_timing(frames: int = 300) -> dict:
    duration = frames / 30.0
    return {
        "ok": True,
        "decoded_frames": frames,
        "timing": {
            "summary_output_fps": 30,
            "decoded_fps": 30.0,
            "fps_abs_tolerance": 0.001,
            "expected_duration_s": duration,
            "decoded_duration_s": duration,
            "duration_abs_tolerance_s": 1.0 / 30.0,
        },
    }

def test_validate_rerender_output_attests_exact_policy_action_playback(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_fr3_policy_action_source(run_dir)
    source = rerender.prepare_source_run(
        run_dir,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )
    video = run_dir / "scene_run_01.mp4"
    video.write_bytes(b"video")
    replay_summary_path = run_dir / "rerender" / "sim_summary.json"
    replay_summary_path.parent.mkdir(exist_ok=True)
    replay_summary_path.write_text(
        json.dumps(_exact_policy_action_replay_summary(run_dir)),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        rerender_validation,
        "validate_video",
        lambda *_: _exact_video_timing(),
    )

    result = rerender.validate_rerender_output(
        source,
        video,
        expected_wrist_insets="off",
    )

    cadence = result["policy_action_playback_contract"]
    assert cadence["source_policy_action_hz"] == 30.0
    assert cadence["replay_summary"]["output_fps"] == 30
    assert cadence["decoded_video"] == {
        "fps": 30.0,
        "frames": 300,
        "duration_s": 10.0,
    }
    assert cadence["no_time_stretch"] is True

def test_validate_rerender_output_rejects_policy_action_time_stretch(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_fr3_policy_action_source(run_dir)
    source = rerender.prepare_source_run(
        run_dir,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )
    video = run_dir / "scene_run_01.mp4"
    video.write_bytes(b"video")
    replay_summary = _exact_policy_action_replay_summary(run_dir)
    replay_summary["output_fps"] = 29
    replay_summary_path = run_dir / "rerender" / "sim_summary.json"
    replay_summary_path.parent.mkdir(exist_ok=True)
    replay_summary_path.write_text(json.dumps(replay_summary), encoding="utf-8")
    monkeypatch.setattr(
        rerender_validation,
        "validate_video",
        lambda *_: _exact_video_timing(),
    )

    with pytest.raises(rerender.RerenderError, match="output_fps=29"):
        rerender.validate_rerender_output(source, video)

    failure = json.loads(
        (run_dir / "rerender" / "video_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["ok"] is False
    assert "output_fps=29" in failure["error"]

def test_validate_rerender_output_rejects_summary_changed_during_validation(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    video = run_dir / "scene_run_01.mp4"
    video.write_bytes(b"video")
    summary = run_dir / "rerender" / "sim_summary.json"
    summary.parent.mkdir(exist_ok=True)
    summary.write_text(
        json.dumps(_source_registration_summary(run_dir)), encoding="utf-8"
    )

    def mutate_summary(*_args) -> dict:
        summary.write_text('{"changed": true}', encoding="utf-8")
        return {"ok": True, "decoded_frames": 1}

    monkeypatch.setattr(rerender_validation, "validate_video", mutate_summary)

    with pytest.raises(rerender.RerenderError, match="video validation failed"):
        rerender.validate_rerender_output(run_dir, video)

    failure = json.loads(
        (run_dir / "rerender" / "video_validation.json").read_text(
            encoding="utf-8"
        )
    )
    # The change itself is only reported; what still fails the
    # video is the summary it was changed into, which is no longer valid.
    assert "rerender summary" in failure["error"]
    assert "rerender summary changed" in capsys.readouterr().err

def test_validate_rerender_output_enforces_exact_saved_wrist_match(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = run_dir / "scene_run_01.mp4"
    video.write_bytes(b"video")
    summary = run_dir / "rerender" / "sim_summary.json"
    summary.parent.mkdir(exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                **_source_registration_summary(run_dir),
                "video_wrist_inset_source": "saved",
                "video_wrist_inset_layout": {"version": 3},
            }
        ),
        encoding="utf-8",
    )
    source = rerender.prepare_source_run(
        run_dir,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )
    display_pair = (
        np.zeros((135, 180, 3), dtype=np.uint8),
        np.ones((135, 180, 3), dtype=np.uint8),
    )
    raw_display_pair = (
        np.full((135, 180, 3), 2, dtype=np.uint8),
        np.full((135, 180, 3), 3, dtype=np.uint8),
    )
    schedule = rerender.ReplayPolicyInsetSchedule(
        frames=((np.zeros((224, 224, 3), dtype=np.uint8),) * 2,),
        raw_display_frames=(raw_display_pair,),
        display_frames=(display_pair,),
        display_transforms=(),
        observation_frames=(0,),
        unique_pairs=1,
    )
    calls = []
    monkeypatch.setattr(
        rerender_validation, "validate_video", lambda *_: {"ok": True, "decoded_frames": 1}
    )
    monkeypatch.setattr(rerender_validation, "load_replay_policy_insets", lambda *_: schedule)

    def fake_exact(video_path, summary_path, frames, *, observation_frames):
        calls.append((video_path, summary_path, frames, observation_frames))
        return {"ok": True, "contract": "exact"}

    monkeypatch.setattr(rerender_validation, "validate_saved_wrist_insets", fake_exact)

    result = rerender.validate_rerender_output(source, video)

    assert result["saved_wrist_exact_match"] == {"ok": True, "contract": "exact"}
    assert calls == [(video, summary, schedule.raw_display_frames, (0,))]

def test_validate_rerender_output_rejects_requested_wrist_mode_mismatch(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = run_dir / "scene_run_01.mp4"
    video.write_bytes(b"video")
    summary = run_dir / "rerender" / "sim_summary.json"
    summary.parent.mkdir(exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                **_source_registration_summary(run_dir),
                "video_wrist_inset_source": "off",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        rerender_validation, "validate_video", lambda *_: {"ok": True, "decoded_frames": 1}
    )

    with pytest.raises(rerender.RerenderError, match="differs from requested"):
        rerender.validate_rerender_output(
            run_dir, video, expected_wrist_insets="saved"
        )

def test_validate_rerender_output_rejects_incomplete_replay(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = run_dir / "scene_run_01.mp4"
    video.write_bytes(b"video")
    summary = run_dir / "rerender" / "sim_summary.json"
    summary.parent.mkdir(exist_ok=True)
    summary.write_text(
        json.dumps(_source_registration_summary(run_dir)), encoding="utf-8"
    )
    monkeypatch.setattr(
        rerender_validation,
        "validate_video",
        lambda *_: {"ok": True, "decoded_frames": 0},
    )

    with pytest.raises(rerender.RerenderError, match="incomplete replay"):
        rerender.validate_rerender_output(run_dir, video)

    report = json.loads(
        (run_dir / "rerender" / "video_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["ok"] is False
    assert "0 of 1 logged frames" in report["error"]

@pytest.mark.parametrize("complete_quality_failure", (False, True))
def test_validate_rerender_output_preserves_complete_quality_failure(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete_quality_failure: bool,
) -> None:
    video = run_dir / "scene_run_01.mp4"
    video.write_bytes(b"broken")
    summary = run_dir / "rerender" / "sim_summary.json"
    summary.parent.mkdir(exist_ok=True)
    summary.write_text("{}", encoding="utf-8")

    complete_report = {
        "ok": False,
        "error": "visual quality validation failed: wrist ROI left near-black",
        "visual_quality": {
            "wrist_rois": {
                "checks_applied": True,
                "left": {"ok": False, "near_black": {"count": 300}},
            }
        },
    }

    def fake_validate(video_path: Path, summary_path: Path) -> dict:
        if complete_quality_failure:
            raise rerender.VideoValidationError(
                complete_report["error"],
                complete_report,
            )
        raise RuntimeError("wrist ROI left entire video is frozen")

    monkeypatch.setattr(rerender_validation, "validate_video", fake_validate)

    with pytest.raises(rerender.RerenderError, match="video validation failed"):
        rerender.validate_rerender_output(run_dir, video)

    report = json.loads(
        (run_dir / "rerender" / "video_validation.json").read_text(
            encoding="utf-8"
        )
    )
    if complete_quality_failure:
        assert report == complete_report
        return

    assert report["ok"] is False
    assert "wrist ROI left entire video is frozen" in report["error"]
    assert report["video"] == str(video.resolve())
    assert report["summary"] == str(summary.resolve())

def _prepare_provenance_inputs(run_dir: Path, token: str) -> dict:
    """Build one complete, valid write_rerender_provenance input set."""

    source = rerender.prepare_source_run(
        run_dir,
        renderer_code=rerender._git_code_identity(),
    )
    video = run_dir / "scene_run_01.mp4"
    video.write_bytes(b"video")
    validation = {
        "ok": True,
        "video": str(video.resolve()),
        "summary": str((run_dir / "rerender" / "sim_summary.json").resolve()),
        "decoded_frames": 1,
        "decoded_sha256": "b" * 64,
        "file_size_bytes": video.stat().st_size,
    }
    validation_path = run_dir / "rerender" / "video_validation.json"
    validation_path.parent.mkdir(exist_ok=True)
    rerender_summary = run_dir / "rerender" / "sim_summary.json"
    rerender_summary.write_text("{}\n", encoding="utf-8")
    validation["rerender_summary_sha256"] = rerender._sha256_file(
        rerender_summary
    )
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    scene_manifest = run_dir / "scene_background.json"
    scene_manifest.write_text("{}", encoding="utf-8")
    scene_ply = run_dir / "scene.ply"
    scene_ply.write_bytes(b"ply")
    source_registration = source.source_records["gs_registration"]
    scene = {
        "tag": "scene",
        "id": "scene-id",
        "contract_digest": "c" * 64,
        "manifest": rerender._file_record(scene_manifest, include_state=True),
        "ply": rerender._file_record(scene_ply, include_state=True),
        "w2gs": {
            "kind": source_registration["kind"],
            "path": source_registration["path"],
            "file": source_registration["file"],
            "raw_matrix": source_registration["raw_matrix"],
        },
    }
    return {
        "source": source,
        "gs_token": token,
        "scene": scene,
        "video_path": video,
        "validation": validation,
    }

@pytest.mark.parametrize("record_view_overrides", (False, True))
def test_write_rerender_provenance_uses_strict_schema_and_redacts_token(
    run_dir: Path,
    record_view_overrides: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    token = "top-secret-token"
    inputs = _prepare_provenance_inputs(run_dir, token)
    source = inputs["source"]
    scene = inputs["scene"]
    video = inputs["video_path"]
    validation = inputs["validation"]
    source_registration = source.source_records["gs_registration"]
    view_overrides = (
        {
            "--view-azimuth": "45",
            "--view-lookat": "0.4 0.5 0.6",
        }
        if record_view_overrides
        else None
    )
    replay = rerender.build_replay_argv(
        source.argv,
        run_dir=run_dir,
        gs_port=9999,
        gs_token=token,
        render_lock="/tmp/rerender.lock",
        wrist_insets="off",
        source_summary=source.summary,
        view_overrides=view_overrides,
    )
    payload = rerender.write_rerender_provenance(
        source=source,
        replay=replay,
        gs_token=token,
        scene=scene,
        video_path=video,
        validation=validation,
        view_overrides=view_overrides,
    )

    assert set(payload) == {
        "schema_version",
        "kind",
        "mode",
        "created_at",
        "source",
        "renderer",
        "scene",
        "output",
    }
    assert payload["schema_version"] == 5
    assert payload["kind"] == "real2sim-rerender-provenance"
    assert payload["mode"] == "diagnostic"
    assert set(payload["renderer"]) == {"sanitized_argv", "code", "view_overrides"}
    assert payload["renderer"]["view_overrides"] == view_overrides
    assert "--gs-server-token" not in payload["renderer"]["sanitized_argv"]
    assert token not in json.dumps(payload)
    assert set(payload["scene"]) == {
        "tag",
        "id",
        "contract_digest",
        "manifest",
        "ply",
        "w2gs",
    }
    assert set(payload["scene"]["w2gs"]) == {
        "kind",
        "path",
        "file",
        "raw_matrix",
    }
    assert set(payload["output"]) == {
        "video",
        "validation",
        "rerender_summary",
        "decoded_frames",
        "decoded_sha256",
    }
    assert set(payload["output"]["video"]) == {"path", "sha256", "state"}
    assert len(payload["output"]["video"]["sha256"]) == 64
    assert set(payload["output"]["validation"]) == {"path", "sha256", "state"}
    assert set(payload["output"]["rerender_summary"]) == {
        "path",
        "sha256",
        "state",
    }
    assert (
        payload["output"]["rerender_summary"]["sha256"]
        == validation["rerender_summary_sha256"]
    )
    assert set(payload["output"]["video"]["state"]) == {
        "device",
        "inode",
        "mode",
        "size",
        "mtime_ns",
        "ctime_ns",
    }
    provenance_path = run_dir / "rerender" / "provenance.json"
    assert json.loads(provenance_path.read_text(encoding="utf-8")) == payload
    assert not list(provenance_path.parent.glob(".provenance.json.*.tmp"))
    if view_overrides is not None:
        sanitized = payload["renderer"]["sanitized_argv"]
        assert sanitized[sanitized.index("--view-azimuth") + 1] == "45"
        assert (
            sanitized[sanitized.index("--view-lookat") + 1]
            == "0.4 0.5 0.6"
        )

    changed = np.eye(4, dtype=np.float64)
    changed[0, 3] = 0.25
    np.save(Path(source_registration["path"]), changed)
    capsys.readouterr()
    # A registration moved after validation is reported and the provenance is
    # still written, rather than refused.
    rerender.write_rerender_provenance(
        source=source,
        replay=replay,
        gs_token=token,
        scene=scene,
        video_path=video,
        validation=validation,
    )
    assert "source GS registration changed" in capsys.readouterr().err

def test_write_rerender_provenance_warns_on_changed_source(
    run_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = rerender.prepare_source_run(
        run_dir,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )
    with (run_dir / "logs" / "sim_log.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"frame": 1}\n')

    # The appended state log is reported, not refused. This source
    # then fails on its renderer identity, which carries no core source set --
    # a check that cannot be made at all, and so still an error.
    with pytest.raises(rerender.RerenderError, match="no exact core source file set"):
        rerender.write_rerender_provenance(
            source=source,
            replay=["sim.py", "--gs-server-token", "token"],
            gs_token="token",
            scene={},
            video_path=run_dir / "missing.mp4",
            validation={},
        )
    assert "state log changed" in capsys.readouterr().err


def _frame_hz_replay_summary(run_dir: Path, frames: int, hz: int) -> dict:
    """A replay summary as the worker writes it for ``--frame-hz hz``."""

    return {
        **_source_registration_summary(run_dir),
        "frames": frames,
        "rendered_frames": frames,
        "output_fps": hz,
        "playback_action_dt_s": round(1.0 / hz, 6),
        "frame_dt_s": round(1.0 / hz, 6),
        "frame_hz_effective": float(hz),
        "dynamic_subframes": 1,
        "video_wrist_inset_source": "off",
    }


def _decoded_timing(frames: int, hz: int, summary_hz: int) -> dict:
    """``validate_video``'s report for an MP4 decoded at ``hz``."""

    duration = frames / hz
    return {
        "ok": True,
        "decoded_frames": frames,
        "timing": {
            "summary_output_fps": summary_hz,
            "decoded_fps": float(hz),
            "expected_duration_s": duration,
            "decoded_duration_s": duration,
        },
    }


def _validate_frame_hz_replay(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    replay_hz: int,
    decoded_hz: int | None = None,
) -> dict:
    _configure_frame_hz_source(run_dir, frames=300, frame_hz=10)
    source = rerender.prepare_source_run(
        run_dir,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )
    video = run_dir / "scene_run_01.mp4"
    video.write_bytes(b"video")
    replay_summary_path = run_dir / "rerender" / "sim_summary.json"
    replay_summary_path.parent.mkdir(exist_ok=True)
    replay_summary_path.write_text(
        json.dumps(_frame_hz_replay_summary(run_dir, 300, replay_hz)),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        rerender_validation,
        "validate_video",
        lambda *_: _decoded_timing(300, decoded_hz or replay_hz, replay_hz),
    )
    return rerender.validate_rerender_output(
        source, video, expected_wrist_insets="off"
    )


def test_validate_rerender_output_attests_frame_hz_replay_against_source(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _validate_frame_hz_replay(run_dir, monkeypatch, replay_hz=10)

    contract = result["frame_hz_playback_contract"]
    assert contract["kind"] == "frame-hz-to-exact-state-replay"
    assert contract["source_playback_dt_s"] == 0.1
    assert contract["replay_output_fps"] == 10
    assert contract["replay_expected_duration_s"] == pytest.approx(30.0)
    assert contract["decoded_video"] == {
        "fps": 10.0,
        "frames": 300,
        "duration_s": 30.0,
    }
    assert contract["no_time_stretch"] is True


def test_validate_rerender_output_rejects_self_consistent_wrong_speed_replay(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 20 Hz replay of a 10 Hz run agrees with itself but not with its source.

    The rerender summary and the decoded MP4 both say 20 fps over 15 s, so the
    summary-only video check passes; the source recorded 0.1 s per row.
    """

    with pytest.raises(
        rerender.RerenderError, match="output_fps=20, but the source run"
    ):
        _validate_frame_hz_replay(run_dir, monkeypatch, replay_hz=20)

    failure = json.loads(
        (run_dir / "rerender" / "video_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["ok"] is False
    assert "but the source run" in failure["error"]


def test_validate_rerender_output_rejects_frame_hz_mp4_at_the_wrong_rate(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(rerender.RerenderError, match="decoded_fps=20.0"):
        _validate_frame_hz_replay(
            run_dir, monkeypatch, replay_hz=10, decoded_hz=20
        )


def test_prepare_source_run_rejects_frame_hz_source_with_another_clock(
    run_dir: Path,
) -> None:
    _configure_frame_hz_source(run_dir, frames=300, frame_hz=10)
    summary_path = run_dir / "sim_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["playback_action_dt_s"] = 0.05
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(
        rerender.RerenderError, match="playback_action_dt_s=0.05s"
    ):
        rerender.prepare_source_run(
            run_dir,
            renderer_code={"commit": "a" * 40, "dirty": True},
        )


#: The retained FR3 ten-rollout profiles (fr3/flat cup, both public rooms).
RETAINED_FR3_PROFILES = tuple(
    f"config/release/fr3/flat/{room}_cup_cup1trim.yaml"
    for room in ("reading_room", "a_room")
)


@pytest.mark.parametrize("relative", RETAINED_FR3_PROFILES)
def test_retained_fr3_profiles_get_a_source_bound_replay_clock(
    relative: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each retained profile's worker arguments select the --frame-hz contract.

    The summary values are the worker's output for a 10 Hz clock with one
    dynamic subframe (see ``_configure_frame_hz_source``).
    """

    import os

    from real2sim import launcher
    from robosuite_env.rerender import cadence
    from tests._helpers.repo import REPO_ROOT

    monkeypatch.setitem(os.environ, "REAL2SIM_INPUTS_ROOT", str(REPO_ROOT))
    plan = launcher.load_rollout_config(
        REPO_ROOT / relative, timestamp="20260730_120000"
    )
    argv = ["python", "robosuite_env/sim.py", *plan["resolved"]["simulation"]["cli_args"]]
    frames = int(plan["resolved"]["runner"]["frames"])
    source_summary = {"frames": frames, "playback_action_dt_s": 0.1}

    assert cadence._fr3_policy_action_playback_contract(argv, source_summary) is None
    contract = cadence._frame_hz_playback_contract(argv, source_summary)

    assert contract is not None
    assert contract["source_frame_hz"] == 10.0
    assert contract["replay_output_fps"] == 10
    assert contract["replay_expected_duration_s"] == pytest.approx(frames * 0.1)
    with pytest.raises(rerender.RerenderError, match="playback_action_dt_s"):
        cadence._frame_hz_playback_contract(
            argv, {"frames": frames, "playback_action_dt_s": 0.05}
        )


def test_provenance_records_whether_the_checked_checkout_held_still(tmp_path: Path) -> None:
    """The official generation check is stored with the renderer's recheck.

    Completion refuses a render whose checkout changed while it ran, so the
    recheck must notice a compared file that changed and a renderer code
    change, and must leave a diagnostic source without a check alone.
    """

    import hashlib
    import types

    from robosuite_env.rerender.validation import _source_with_generation_recheck

    asset = tmp_path / "robosuite_env" / "assets" / "box.xml"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"<mujoco/>\n")
    check = {
        "rule": "sealed_sources",
        "checkout_root": str(tmp_path),
        "commit": "c" * 40,
        "dirty": True,
        "run_seal": {"path": str(tmp_path / "seal.json"), "sha256": "e" * 64},
        "coverage": {"files": [], "trees": []},
        "files": {
            "robosuite_env/assets/box.xml": hashlib.sha256(b"<mujoco/>\n").hexdigest()
        },
    }
    official = types.SimpleNamespace(
        source_records={"code_commit": "b" * 40, "official_generation_check": check}
    )

    held = _source_with_generation_recheck(official, True)
    assert held["official_generation_check"] == {**check, "unchanged_at_write": True}
    assert "unchanged_at_write" not in check
    assert not _source_with_generation_recheck(official, False)[
        "official_generation_check"
    ]["unchanged_at_write"]

    asset.write_bytes(b"<mujoco><worldbody/></mujoco>\n")
    assert not _source_with_generation_recheck(official, True)[
        "official_generation_check"
    ]["unchanged_at_write"]

    diagnostic = types.SimpleNamespace(source_records={"expected_frames": 3})
    assert _source_with_generation_recheck(diagnostic, True) == {"expected_frames": 3}
