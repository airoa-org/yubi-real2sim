# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from robosuite_env.rollout import outputs as outputs_module
from robosuite_env.rollout import metrics as metrics_module
from robosuite_env.rollout.contacts import collect_arm_penetrations
from robosuite_env.rollout.metrics import FrameMetricsCollector
from robosuite_env.rollout.outputs import (
    ArraySpool,
    EmptyArraySpool,
    VIDEO_DECODER_THREADS,
    open_video_capture,
    resolve_summary_paths,
    write_policy_chunk_records,
    write_rgb_video,
    write_summary_artifacts,
)
from robosuite_env.rollout.policy_chunks import (
    PolicyPlanConfig,
    build_policy_chunk_result,
    build_policy_plan_record,
)
def _contact(geom1, geom2, dist, *, pos=(0.0, 0.0, 0.0)):
    return SimpleNamespace(
        geom1=geom1,
        geom2=geom2,
        dist=dist,
        pos=np.asarray(pos, dtype=float),
        frame=np.eye(3),
    )


def test_arm_penetrations_preserve_worker_schema_and_contact_order():
    data = SimpleNamespace(
        ncon=4,
        contact=[
            _contact(10, 90, -0.0012345),
            _contact(90, 20, -0.002),
            _contact(10, 20, -0.0005),
            _contact(10, 91, 0.001),
        ],
    )

    result = collect_arm_penetrations(
        data=data,
        arm_geom_ids={10, 20},
        right_arm_geom_ids={10},
        geom_name=lambda geom_id: f"geom-{geom_id}",
        classify_other=lambda geom_id, side: f"{side}:{geom_id}",
    )

    assert result == [
        {
            "armgeom": "geom-10",
            "other": "geom-90",
            "cat": "R:90",
            "depth_mm": 1.234,
        },
        {
            "armgeom": "geom-20",
            "other": "geom-90",
            "cat": "L:90",
            "depth_mm": 2.0,
        },
        {
            "armgeom": "geom-10",
            "other": "geom-20",
            "cat": "R:20",
            "depth_mm": 0.5,
        },
    ]


def test_frame_metrics_preserve_geometry_keys_and_camera_adjustment(monkeypatch):
    monkeypatch.setattr(
        metrics_module,
        "get_wrist_render_profile",
        lambda name: SimpleNamespace(camera_backward_m=0.2),
    )

    class FakeMujoco:
        class mjtObj:
            mjOBJ_CAMERA = 7

        @staticmethod
        def mj_name2id(model, object_type, name):
            assert model == "model"
            assert object_type == 7
            return 0 if name.startswith("right") else -1

    data = SimpleNamespace(
        qpos=np.asarray(
            [
                0.1,
                0.2,
                0.3,
                1.0,
                0.0,
                0.0,
                0.0,
                0.4,
                0.5,
                0.6,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        cam_xmat=np.asarray([np.eye(3).ravel()]),
        cam_xpos=np.asarray([[1.0, 2.0, 3.0]]),
    )
    poses = {
        "R": (np.asarray([0.1, 0.2, 0.4]), np.asarray([1.0, 0, 0, 0])),
        "L": (np.asarray([0.1, 0.2, 0.5]), np.asarray([1.0, 0, 0, 0])),
    }
    tcp = {
        "R": (np.asarray([0.1, 0.2, 0.3]), np.asarray([1.0, 0, 0, 0])),
        "L": (np.asarray([0.1, 0.2, 0.6]), np.asarray([1.0, 0, 0, 0])),
    }
    collector = FrameMetricsCollector(
        mujoco=FakeMujoco,
        model="model",
        data=data,
        arm={
            "R": {"gripper_prefix": "right"},
            "L": {"gripper_prefix": "left"},
        },
        cup_qpos_address=0,
        plate_qpos_address=7,
        render_profile_name="profile",
        eef_pose=lambda side: poses[side],
        policy_tcp_pose=lambda side: tcp[side],
    )

    result = collector.collect(
        {
            "R": (np.asarray([0.1, 0.2, 0.3]), np.zeros(4)),
            "L": (np.asarray([0.1, 0.2, 0.7]), np.zeros(4)),
        }
    )

    assert result["cup_pos"] == [0.1, 0.2, 0.3]
    assert result["cup_quat"] == [1.0, 0.0, 0.0, 0.0]
    assert result["plate_pos"] == [0.4, 0.5, 0.6]
    assert result["arms_metric"]["R"]["tcp_cup_dist_mm"] == 0.0
    assert result["arms_metric"]["L"]["target_tcp_res_mm"] == 100.0
    assert result["arms_metric"]["R"]["wrist_front_camera"] == {
        "name": "right_wrist_fisheye_front",
        "position": [1.0, 2.0, 3.2],
        "right": [1.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0],
        "forward": [-0.0, -0.0, -1.0],
    }
    assert "wrist_front_camera" not in result["arms_metric"]["L"]


def _policy_plan_config() -> PolicyPlanConfig:
    return PolicyPlanConfig(
        execution_steps=16,
        overlap_steps=8,
        sampler_configured="paired_weighted",
        collapse_reason=None,
        prev_weight_configured=0.75,
        step_shift_configured=2,
        sample_num_steps=10,
        policy_delta_anchor="observation",
        policy_pose_repr="delta",
    )


def test_policy_chunk_builders_preserve_common_and_extension_keys():
    prefix = np.arange(16, dtype=float).reshape(2, 8)
    chunk = np.arange(24, dtype=float).reshape(3, 8)
    record = build_policy_plan_record(
        _policy_plan_config(),
        chunk_index=3,
        observation_frame=32,
        inference_frame=40,
        execution_start=8,
        sampler="single",
        prev_weight=None,
        dynamic_prev_weight=False,
        prev_weight_schedule=None,
        step_shift=0,
        conditioned=True,
        execution_anchor={"R": {"position": [1.0, 2.0, 3.0]}},
        prefix_error=1.0e-6,
        conditioning_prefix=prefix,
        conditioning_rebase_meta={"basis": "left_hand_root"},
        chunk=chunk,
        extra={"previous_observation_frame": 24},
    )
    assert record["execution_stop"] == 24
    assert record["conditioning_prefix"] == prefix.tolist()
    assert record["actions"] == chunk.tolist()
    assert record["policy_pose_repr"] == "delta"
    assert record["previous_observation_frame"] == 24

    result = build_policy_chunk_result(
        chunk=chunk,
        chunk_index=3,
        obs_meta={"observation_frame": 32},
        obs_log={"path": "obs.png"},
        policy_result_meta={"timing": 1.5},
        prefix_error=1.0e-6,
        bridge_error=None,
        bridge_rebased_prev_distance=0.25,
        prev_weight=0.75,
        execution_start=0,
        sampler="paired_weighted",
        conditioned=True,
        inference_frame=40,
        conditioning_rebase_meta={"basis": "left_hand_root"},
        target_trajectory={"R": {}},
        extra={"dynamic_prev_weight": True},
    )
    assert result["chunk"] is chunk
    assert result["obs_meta"] == {"observation_frame": 32}
    assert result["dynamic_prev_weight"] is True

    with pytest.raises(ValueError, match="duplicate common fields"):
        build_policy_plan_record(
            _policy_plan_config(),
            chunk_index=0,
            observation_frame=0,
            inference_frame=0,
            execution_start=0,
            sampler="single",
            prev_weight=None,
            dynamic_prev_weight=False,
            prev_weight_schedule=None,
            step_shift=0,
            conditioned=False,
            execution_anchor={},
            prefix_error=None,
            conditioning_prefix=None,
            conditioning_rebase_meta=None,
            chunk=np.zeros((1, 8)),
            extra={"chunk_index": 99},
        )


def test_output_writers_preserve_json_compatibility_contract(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    chunks = [{"chunk_index": 0}, {"chunk_index": 1}]
    chunk_path = write_policy_chunk_records(str(log_dir), chunks)
    assert [
        json.loads(line)
        for line in Path(chunk_path).read_text(encoding="utf-8").splitlines()
    ] == chunks

    explicit = tmp_path / "reports" / "summary.json"
    default_path, summary_path = resolve_summary_paths(
        str(log_dir), str(explicit)
    )
    summary = {
        "frames": 1,
        "frames_with_pen": 1,
        "max_pen_mm": 2.0,
        "summary_path": summary_path,
    }
    records = [
        {
            "frame": 0,
            "arms": {"R": {"ok": True}, "L": {"ok": True}},
            "max_pen_mm": 2.0,
            "penetration": [{"depth_mm": 2.0}],
        }
    ]
    artifacts = write_summary_artifacts(
        str(log_dir), summary_path, summary, records
    )

    assert artifacts.default_summary_path == default_path
    assert json.loads(Path(default_path).read_text()) == summary
    assert json.loads(explicit.read_text()) == summary
    assert json.loads(Path(artifacts.ik_results_path).read_text())["frames"] == [
        {"frame": 0, "R": {"ok": True}, "L": {"ok": True}}
    ]
    penetration = json.loads(
        Path(artifacts.penetration_results_path).read_text()
    )
    assert penetration["meta"] == {
        "max_pen_mm": 2.0,
        "frames_with_pen": 1,
        "frames": 1,
    }


def test_array_spool_replays_arrays_in_order_and_releases(tmp_path, monkeypatch):
    # pytest's tmp_path may be tmpfs. The test opts in explicitly; rollout workers default to a disk-backed LOGDIR spool.
    monkeypatch.setenv("REAL2SIM_ALLOW_TMPFS_SPOOL", "1")
    spool = ArraySpool(directory=str(tmp_path))
    first = np.arange(6, dtype=np.uint8).reshape(1, 2, 3)
    second = first + 10
    spool.append(first)
    spool.append(second)

    assert len(spool) == 2
    replayed = list(spool)
    assert np.array_equal(replayed[0], first)
    assert np.array_equal(replayed[1], second)

    # A caller may inspect an early frame before capture has finished. A later
    # append must restore the tracked data end instead of overwriting an array.
    partial_reader = iter(spool)
    assert np.array_equal(next(partial_reader), first)
    third = first + 20
    spool.append(third)
    partial_reader.close()
    replayed = list(spool)
    assert len(spool) == 3
    assert np.array_equal(replayed[2], third)

    spool.close()
    with pytest.raises(ValueError, match="closed array spool"):
        list(spool)


def test_empty_array_spool_never_touches_storage_and_rejects_frames():
    spool = EmptyArraySpool()

    assert len(spool) == 0
    assert list(spool) == []
    spool.close()
    with pytest.raises(RuntimeError, match="video capture is disabled"):
        spool.append(np.zeros((1,), dtype=np.uint8))


def test_array_spool_defaults_to_working_filesystem_instead_of_tempfile_dir(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("REAL2SIM_ALLOW_TMPFS_SPOOL", "1")
    working_directory = tmp_path / "checkout"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    monkeypatch.delenv("REAL2SIM_FRAME_SPOOL_DIR", raising=False)

    with ArraySpool(min_free_bytes=0) as spool:
        assert Path(spool.directory) == working_directory

    configured_directory = tmp_path / "configured"
    explicit_directory = tmp_path / "explicit"
    monkeypatch.setenv(
        "REAL2SIM_FRAME_SPOOL_DIR", str(configured_directory)
    )

    with ArraySpool(
        directory=explicit_directory,
        min_free_bytes=0,
    ) as spool:
        assert Path(spool.directory) == explicit_directory
        assert explicit_directory.is_dir()
        assert not configured_directory.exists()


def test_array_spool_rejects_memory_backed_filesystem_by_default(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(outputs_module, "_filesystem_type", lambda _path: "tmpfs")
    monkeypatch.delenv("REAL2SIM_ALLOW_TMPFS_SPOOL", raising=False)

    with pytest.raises(OSError, match="memory-backed filesystem") as exc:
        ArraySpool(directory=tmp_path, min_free_bytes=0)

    assert exc.value.errno == outputs_module.errno.ENOTSUP
    monkeypatch.setenv("REAL2SIM_ALLOW_TMPFS_SPOOL", "1")

    with ArraySpool(directory=tmp_path, min_free_bytes=0) as spool:
        spool.append(np.zeros((1,), dtype=np.uint8))
        assert len(spool) == 1


def test_array_spool_fails_before_capture_when_capacity_is_insufficient(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("REAL2SIM_ALLOW_TMPFS_SPOOL", "1")
    monkeypatch.setattr(
        outputs_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1024),
    )

    with pytest.raises(
        OSError, match="insufficient frame spool capacity"
    ) as exc:
        ArraySpool(
            directory=tmp_path,
            reserve_bytes=800,
            min_free_bytes=300,
        )

    assert exc.value.errno == outputs_module.errno.ENOSPC


def test_array_spool_preallocates_requested_capacity_when_supported(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("REAL2SIM_ALLOW_TMPFS_SPOOL", "1")
    reservations = []
    monkeypatch.setattr(
        outputs_module.os,
        "posix_fallocate",
        lambda descriptor, offset, length: reservations.append(
            (descriptor, offset, length)
        ),
    )
    frame = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

    with ArraySpool(
        directory=tmp_path,
        reserve_bytes=4096,
        min_free_bytes=0,
    ) as spool:
        spool.append(frame)
        assert np.array_equal(list(spool)[0], frame)

    assert len(reservations) == 1
    assert reservations[0][1:] == (0, 4096)


def test_rgb_video_writer_streams_atomically_and_releases(monkeypatch, tmp_path):
    written = []

    class FakeWriter:
        def __init__(self, path, *_args):
            self.path = Path(path)

        def isOpened(self):
            return True

        def write(self, frame):
            written.append(frame)

        def release(self):
            self.path.write_bytes(b"encoded-mp4")
            written.append("released")

    monkeypatch.setattr(
        "robosuite_env.rollout.outputs.cv.VideoWriter",
        FakeWriter,
    )
    # The fake writer emits placeholder bytes, not a real MP4; skip the
    # H.264 publish transcode so the atomic-streaming contract stays testable.
    monkeypatch.setattr(
        "robosuite_env.rollout.outputs._resolve_ffmpeg", lambda: None
    )
    monkeypatch.setattr(
        "robosuite_env.rollout.outputs.cv.VideoWriter_fourcc",
        lambda *args: 123,
    )
    monkeypatch.setattr(
        "robosuite_env.rollout.outputs.cv.cvtColor",
        lambda frame, code: frame + 1,
    )
    frames = (
        np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(2)
    )
    output = tmp_path / "out.mp4"

    assert write_rgb_video(frames, str(output), 10.0) == 2
    assert np.array_equal(written[0], np.ones((2, 3, 3), dtype=np.uint8))
    assert np.array_equal(written[1], np.ones((2, 3, 3), dtype=np.uint8))
    assert written[2] == "released"
    assert output.read_bytes() == b"encoded-mp4"
    assert not list(tmp_path.glob(".*.partial.*"))


def test_rgb_video_writer_keeps_previous_output_on_stream_failure(
    monkeypatch, tmp_path
):
    class FakeWriter:
        def __init__(self, path, *_args):
            self.path = Path(path)

        def isOpened(self):
            return True

        def write(self, _frame):
            pass

        def release(self):
            self.path.write_bytes(b"partial")

    monkeypatch.setattr(
        "robosuite_env.rollout.outputs.cv.VideoWriter", FakeWriter
    )
    output = tmp_path / "out.mp4"
    output.write_bytes(b"previous-good-video")
    frames = iter(
        (
            np.zeros((2, 3, 3), dtype=np.uint8),
            np.zeros((3, 3, 3), dtype=np.uint8),
        )
    )

    with pytest.raises(ValueError, match="dimensions changed"):
        write_rgb_video(frames, str(output), 10.0)

    assert output.read_bytes() == b"previous-good-video"
    assert not list(tmp_path.glob(".*.partial.*"))


def test_rgb_video_writer_rejects_unopened_codec_without_replacing_output(
    monkeypatch, tmp_path
):
    released = []

    class ClosedWriter:
        def isOpened(self):
            return False

        def release(self):
            released.append(True)

    monkeypatch.setattr(
        "robosuite_env.rollout.outputs.cv.VideoWriter",
        lambda *_args: ClosedWriter(),
    )
    output = tmp_path / "out.mp4"
    output.write_bytes(b"previous-good-video")

    with pytest.raises(RuntimeError, match="failed to open"):
        write_rgb_video(
            iter((np.zeros((2, 3, 3), dtype=np.uint8),)),
            str(output),
            10.0,
        )

    assert released == [True]
    assert output.read_bytes() == b"previous-good-video"
    assert not list(tmp_path.glob(".*.partial.*"))


def test_rgb_video_writer_preserves_requested_container_suffix(
    monkeypatch, tmp_path
):
    opened_paths = []

    class FakeWriter:
        def __init__(self, path, *_args):
            self.path = Path(path)
            opened_paths.append(self.path)

        def isOpened(self):
            return True

        def write(self, _frame):
            pass

        def release(self):
            self.path.write_bytes(b"encoded-video")

    monkeypatch.setattr(
        "robosuite_env.rollout.outputs._resolve_ffmpeg", lambda: None
    )
    monkeypatch.setattr(
        "robosuite_env.rollout.outputs.cv.VideoWriter", FakeWriter
    )
    output = tmp_path / "out.avi"

    assert write_rgb_video(
        iter((np.zeros((2, 3, 3), dtype=np.uint8),)),
        str(output),
        10.0,
    ) == 1

    assert opened_paths[0].suffix == ".avi"
    assert output.read_bytes() == b"encoded-video"


def test_h264_transcode_runs_ffmpeg_with_one_thread_on_any_host(
    monkeypatch, tmp_path
):
    """An H.264 transcode must not start threads per host core.

    With FFmpeg's defaults the decoder and the filter graph start about one
    thread per visible core, and libx264 1.5 per core plus lookahead threads.
    A grid encode can then exhaust a container's task limit, and libx264 fails
    to open. libx264 writes the thread count it used into the stream, so the
    encoded bytes show what the real encoder did.

    The rollout writer and the grid tool find FFmpeg with ``_resolve_ffmpeg``:
    next to the interpreter, where ``scripts/bootstrap.sh`` links the locked
    ``imageio-ffmpeg`` binary, then on PATH. A plain ``uv sync`` provides
    neither, so fall back to that locked binary instead of skipping.
    """

    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("needs two visible CPUs, where libx264 defaults to several threads")
    ffmpeg = outputs_module._resolve_ffmpeg()
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # Record the arguments, then run the real FFmpeg with them.
    arguments = tmp_path / "arguments.txt"
    recorder = tmp_path / "ffmpeg"
    recorder.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" > "{arguments}"\n'
        f'exec "{ffmpeg}" "$@"\n',
        encoding="utf-8",
    )
    recorder.chmod(0o755)
    monkeypatch.setattr(outputs_module, "_resolve_ffmpeg", lambda: str(recorder))
    target = tmp_path / "rollout.mp4"

    # The rollout writer's real path: the OpenCV mp4v intermediate, the
    # transcode with the FFmpeg that _resolve_ffmpeg returns, the publish.
    frames = (np.full((128, 160, 3), 20 * index, dtype=np.uint8) for index in range(10))
    assert write_rgb_video(frames, str(target), 10.0) == 10

    argv = arguments.read_text(encoding="utf-8").splitlines()
    input_index = argv.index("-i")
    # The decoder's thread count is an input option, so it precedes -i.
    assert argv[input_index - 2 : input_index] == ["-threads", "1"]
    assert argv[argv.index("-filter_threads") + 1] == "1"
    output_options = argv[input_index:]
    assert output_options[output_options.index("-threads") + 1] == "1"
    # At 128 rows libx264 would choose min(1.5 per core, 4) threads by default.
    encoder = re.search(
        rb"x264 - core \d+.* threads=(\d+) lookahead_threads=(\d+) ",
        target.read_bytes(),
    )
    assert encoder is not None
    assert encoder.groups() == (b"1", b"1")
    assert not list(tmp_path.glob(".rollout.*"))


def _process_threads() -> int:
    return len(os.listdir("/proc/self/task"))


def _decode_all(capture) -> list[np.ndarray]:
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            return frames
        frames.append(frame)


def test_video_reader_starts_only_the_helper_decoder_threads(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """An OpenCV reader must start only the decoder threads the helper sets.

    By default OpenCV's FFmpeg backend starts cv2.getNumberOfCPUs() decoder
    threads for each reader of a frame-threaded codec such as mp4v, so the grid
    tool's 31 open validation readers need about 496 threads on 16 CPUs and
    exhaust a 512-task container limit. Readers from the helper decode in the
    calling thread: opening and reading several of them adds no thread, and they
    decode the same pixels as a default reader.

    On a 1-CPU host a default reader starts no thread either. So the test ends
    with a positive control that does not depend on the CPU count: with the
    helper's setting raised to 4, a helper reader must report 4 and start
    threads. This shows that the thread count is measured live and that the
    helper passes its setting to the decoder.
    """

    video = tmp_path / "rollout.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 128))
    assert writer.isOpened()
    yy, xx = np.indices((128, 160))
    for index in range(12):
        writer.write(
            np.stack(
                ((xx * 3 + index * 7) % 256, (yy * 5 + index * 11) % 256, (xx + yy) % 256),
                axis=-1,
            ).astype(np.uint8)
        )
    writer.release()
    default_reader = cv2.VideoCapture(str(video))
    expected = _decode_all(default_reader)
    default_reader.release()
    assert len(expected) == 12

    before = _process_threads()
    readers = [open_video_capture(video) for _ in range(8)]
    try:
        assert _process_threads() <= before
        for reader in readers:
            assert reader.isOpened()
            assert reader.getBackendName() == "FFMPEG"
            assert reader.get(cv2.CAP_PROP_N_THREADS) == VIDEO_DECODER_THREADS == 1
        for reader in readers:
            decoded = _decode_all(reader)
            assert len(decoded) == len(expected)
            assert all(np.array_equal(a, b) for a, b in zip(decoded, expected))
        assert _process_threads() <= before
    finally:
        for reader in readers:
            reader.release()

    with monkeypatch.context() as patch:
        patch.setattr(outputs_module, "VIDEO_DECODER_THREADS", 4)
        before = _process_threads()
        control = open_video_capture(video)
        try:
            assert control.isOpened()
            assert control.get(cv2.CAP_PROP_N_THREADS) == 4
            assert _process_threads() > before
        finally:
            control.release()


def test_every_repository_video_reader_opens_through_the_one_thread_helper():
    """Readers opened with cv2.VideoCapture directly get per-CPU decoder threads."""

    repository = Path(__file__).resolve().parents[3]
    helper_file = repository / "robosuite_env" / "rollout" / "outputs.py"
    helper = [
        node
        for node in ast.parse(helper_file.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == open_video_capture.__name__
    ]
    assert len(helper) == 1, helper_file
    helper_range = range(helper[0].lineno, helper[0].end_lineno + 1)
    direct, in_helper = [], []
    for package in ("robosuite_env", "src", "scene", "scripts"):
        directory = repository / package
        assert directory.is_dir(), f"scan directory is missing: {directory}"
        sources = sorted(directory.rglob("*.py"))
        assert sources, f"scan directory has no Python file: {directory}"
        for path in sources:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                name = getattr(function, "attr", getattr(function, "id", None))
                if name != "VideoCapture":
                    continue
                if path == helper_file and node.lineno in helper_range:
                    in_helper.append(node.lineno)
                else:
                    direct.append(f"{path.relative_to(repository)}:{node.lineno}")
    assert direct == []
    assert len(in_helper) == 1


import threading
import subprocess


from contextlib import contextmanager

from robosuite_env.rollout.resource_lifecycle import (
    GlobalResourceCleanup,
    IdempotentCleanup,
)


def test_cleanup_runs_every_step_once_and_keeps_failure_order():
    calls = []

    def fail():
        calls.append("fail")
        raise RuntimeError("close failed")

    cleanup = IdempotentCleanup(
        (
            ("first", lambda: calls.append("first")),
            ("broken", fail),
            ("last", lambda: calls.append("last")),
        )
    )

    failures = cleanup.run()
    assert calls == ["first", "fail", "last"]
    assert [failure.resource for failure in failures] == ["broken"]
    assert str(failures[0]) == "broken: RuntimeError: close failed"
    assert cleanup.run() is failures
    assert calls == ["first", "fail", "last"]


def test_cleanup_is_exactly_once_under_concurrent_callers():
    calls = []
    barrier = threading.Barrier(5)
    cleanup = IdempotentCleanup((("resource", lambda: calls.append("close")),))

    def run_cleanup():
        barrier.wait()
        cleanup.run()

    threads = [threading.Thread(target=run_cleanup) for _ in range(4)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2.0)

    assert calls == ["close"]
    assert cleanup.finished is True


def test_global_resource_cleanup_closes_renderers_before_environment():
    calls = []

    @contextmanager
    def render_lock():
        calls.append("lock")
        yield
        calls.append("unlock")

    class Resource:
        def __init__(self, name):
            self.name = name

        def close(self):
            calls.append(f"close:{self.name}")

        def release(self):
            calls.append(f"release:{self.name}")

    shared_wrist = Resource("shared-wrist")
    resources = {
        "logf": Resource("log"),
        "_OVERVIEW_VIDEO_FRAMES": Resource("overview"),
        "_WRIST_VIDEO_FRAMES": Resource("wrist-spool"),
        "realcap": Resource("video"),
        "wfR_obs": shared_wrist,
        "wfL_obs": shared_wrist,
        "wfR": None,
        "wfL": None,
        "ren": Resource("overview-renderer"),
        "env": Resource("env"),
    }
    cleanup = GlobalResourceCleanup(resources, render_lock=render_lock).cleanup

    assert cleanup.run() == ()
    assert calls == [
        "close:log",
        "close:overview",
        "close:wrist-spool",
        "release:video",
        "lock",
        "close:shared-wrist",
        "close:overview-renderer",
        "unlock",
        "lock",
        "close:env",
        "unlock",
    ]
    assert all(value is None for value in resources.values())


@pytest.mark.parametrize(
    "steps, error",
    [
        ((), "at least one step"),
        ((("", lambda: None),), "must not be empty"),
        (
            (("same", lambda: None), ("same", lambda: None)),
            "must be unique",
        ),
    ],
)
def test_cleanup_rejects_invalid_step_contract(steps, error):
    with pytest.raises(ValueError, match=error):
        IdempotentCleanup(steps)


def test_global_cleanup_detaches_borrower_then_retries_shared_renderer_owner():
    calls = []

    @contextmanager
    def render_lock():
        yield

    class Borrower:
        def close(self):
            calls.append("borrower")

    class FailOnceOwner:
        def __init__(self):
            self.attempts = 0

        def close(self):
            self.attempts += 1
            calls.append(f"owner-{self.attempts}")
            if self.attempts == 1:
                raise RuntimeError("transient EGL close failure")

    owner = FailOnceOwner()
    resources = {
        "wfL_obs": Borrower(),
        "wfR_obs": owner,
    }

    cleanup = GlobalResourceCleanup(resources, render_lock=render_lock).cleanup

    assert cleanup.run() == ()
    assert calls == ["borrower", "owner-1", "owner-2"]
    assert resources["wfL_obs"] is None
    assert resources["wfR_obs"] is None


def test_global_cleanup_terminates_owned_fallback_server_and_closes_log():
    calls = []

    @contextmanager
    def render_lock():
        yield

    class Process:
        def poll(self):
            calls.append("poll")
            return None

        def terminate(self):
            calls.append("terminate")

        def wait(self, timeout):
            calls.append(f"wait:{timeout}")
            if timeout == 10:
                raise subprocess.TimeoutExpired("gs", timeout)

        def kill(self):
            calls.append("kill")

    class Log:
        def close(self):
            calls.append("close-log")

    resources = {
        "GS_SERVER_PROCESS": Process(),
        "GS_SERVER_LOG": Log(),
    }

    cleanup = GlobalResourceCleanup(resources, render_lock=render_lock).cleanup

    assert cleanup.run() == ()
    assert calls == [
        "poll",
        "terminate",
        "wait:10",
        "kill",
        "wait:5",
        "close-log",
    ]
    assert resources["GS_SERVER_PROCESS"] is None
    assert resources["GS_SERVER_LOG"] is None

from tests._helpers.worker_ast import *  # noqa: F403

def test_renderer_close_order_tuples_are_verbatim() -> None:
    paths = {
        "capture": PACKAGE_ROOT / "rollout" / "render" / "capture.py",
        "policy": PACKAGE_ROOT / "rollout" / "observe" / "wrist.py",
        "overview": PACKAGE_ROOT / "rollout" / "render" / "overview.py",
        "composite": PACKAGE_ROOT / "rollout" / "render" / "composite.py",
    }

    def tuples_for(path: Path, call_name: str) -> list[tuple[str, ...]]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        result = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) != call_name:
                continue
            tuple_nodes = [arg for arg in node.args if isinstance(arg, ast.Tuple)]
            assert len(tuple_nodes) == 1
            result.append(ast.literal_eval(tuple_nodes[0]))
        return result

    assert tuples_for(paths["capture"], "live_names") == [
        ("wfR", "wfL", "wfR_obs", "wfL_obs", "ren")
    ]
    assert tuples_for(paths["capture"], "close_renderers") == [
        ("wfL", "wfR")
    ]
    assert tuples_for(paths["policy"], "live_names") == [
        ("wfR_obs", "wfL_obs", "wfR", "wfL", "ren")
    ]
    assert tuples_for(paths["policy"], "close_renderers") == [
        ("wfL_obs", "wfR_obs")
    ]
    assert tuples_for(paths["overview"], "close_renderers") == [
        ("ren",),
        ("ren",),
    ]
    assert tuples_for(paths["composite"], "close_renderers") == [
        ("wfL_obs", "wfR_obs", "wfL", "wfR", "ren"),
        ("ren",),
    ]
