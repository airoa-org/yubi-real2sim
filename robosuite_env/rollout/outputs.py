# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Writers for rollout logs, summaries, and rendered videos."""

from __future__ import annotations

from dataclasses import dataclass
import errno
from itertools import chain
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

import cv2 as cv
import numpy as np


DEFAULT_ARRAY_SPOOL_MIN_FREE_BYTES = 256 * 1024 * 1024
_ARRAY_SPOOL_APPEND_OVERHEAD_BYTES = 1024 * 1024
_MEMORY_BACKED_FILESYSTEMS = frozenset({"tmpfs", "ramfs", "devtmpfs"})


@dataclass(frozen=True)
class SummaryArtifacts:
    default_summary_path: str
    summary_path: str
    ik_results_path: str
    penetration_results_path: str


def resolve_array_spool_directory(
    directory: str | os.PathLike[str] | None = None,
) -> str:
    """Return a writable filesystem directory for frame spools.

    ``tempfile.TemporaryFile(dir=None)`` commonly selects ``/tmp``, which is a
    memory-backed tmpfs on many rollout hosts.  Prefer an explicit caller
    choice, then ``REAL2SIM_FRAME_SPOOL_DIR``, and finally the current working
    directory.  The final fallback keeps the potentially large frame data on
    the same filesystem as the running checkout instead of silently charging
    it to process memory.
    """

    selected = directory
    if selected is None:
        selected = os.environ.get("REAL2SIM_FRAME_SPOOL_DIR") or os.getcwd()
    path = os.path.abspath(os.fspath(selected))
    os.makedirs(path, exist_ok=True)
    if not os.path.isdir(path):
        raise NotADirectoryError(path)
    if not os.access(path, os.W_OK | os.X_OK):
        raise PermissionError(f"array spool directory is not writable: {path}")
    filesystem = _filesystem_type(path)
    if (
        filesystem in _MEMORY_BACKED_FILESYSTEMS
        and os.environ.get("REAL2SIM_ALLOW_TMPFS_SPOOL") != "1"
    ):
        raise OSError(
            errno.ENOTSUP,
            "frame spool directory uses a memory-backed filesystem; choose "
            "disk-backed REAL2SIM_FRAME_SPOOL_DIR or explicitly set "
            f"REAL2SIM_ALLOW_TMPFS_SPOOL=1: directory={path} fstype={filesystem}",
        )
    return path


def _filesystem_type(path: str) -> str | None:
    """Resolve the longest Linux mount point containing ``path``."""

    try:
        resolved = os.path.realpath(path)
        best: tuple[int, str] | None = None
        with open("/proc/self/mountinfo", encoding="utf-8") as handle:
            for line in handle:
                left, separator, right = line.partition(" - ")
                if not separator:
                    continue
                fields = left.split()
                filesystem_fields = right.split()
                if len(fields) < 5 or not filesystem_fields:
                    continue
                mountpoint = fields[4].replace("\\040", " ")
                try:
                    common = os.path.commonpath((resolved, mountpoint))
                except ValueError:
                    continue
                if common != mountpoint:
                    continue
                candidate = (len(mountpoint), filesystem_fields[0])
                if best is None or candidate[0] > best[0]:
                    best = candidate
        return None if best is None else best[1]
    except OSError:
        return None


def _require_array_spool_capacity(
    directory: str,
    *,
    required_bytes: int,
    min_free_bytes: int,
) -> None:
    free_bytes = int(shutil.disk_usage(directory).free)
    target_bytes = required_bytes + min_free_bytes
    if free_bytes < target_bytes:
        raise OSError(
            errno.ENOSPC,
            "insufficient frame spool capacity: "
            f"directory={directory} free={free_bytes} "
            f"required={required_bytes} safety_margin={min_free_bytes}",
        )


class ArraySpool:
    """Temporarily store arrays on disk and replay them in append order.

    Rollout videos can contain hundreds of overview and wrist images.  Keeping
    those arrays in Python lists makes every parallel worker retain hundreds of
    megabytes until the final MP4 is encoded.  This small spool preserves the
    arrays exactly while letting the operating system reclaim their process
    memory between capture and encoding.  Callers that know the upper bound
    should pass ``reserve_bytes`` so insufficient storage fails before capture.
    """

    def __init__(
        self,
        *,
        directory: str | os.PathLike[str] | None = None,
        reserve_bytes: int = 0,
        min_free_bytes: int = DEFAULT_ARRAY_SPOOL_MIN_FREE_BYTES,
    ) -> None:
        if (
            isinstance(reserve_bytes, bool)
            or not isinstance(reserve_bytes, int)
            or reserve_bytes < 0
        ):
            raise ValueError("reserve_bytes must be a non-negative integer")
        if (
            isinstance(min_free_bytes, bool)
            or not isinstance(min_free_bytes, int)
            or min_free_bytes < 0
        ):
            raise ValueError("min_free_bytes must be a non-negative integer")
        self.directory = resolve_array_spool_directory(directory)
        _require_array_spool_capacity(
            self.directory,
            required_bytes=reserve_bytes,
            min_free_bytes=min_free_bytes,
        )
        self._file = tempfile.TemporaryFile(
            mode="w+b",
            prefix=".real2sim-frame-spool-",
            dir=self.directory,
        )
        self._count = 0
        self._write_offset = 0
        self._min_free_bytes = min_free_bytes
        self._reserved_bytes = 0
        if reserve_bytes:
            try:
                os.posix_fallocate(self._file.fileno(), 0, reserve_bytes)
            except AttributeError:
                # Capacity was checked above and every append is checked again.
                pass
            except OSError as exc:
                if exc.errno in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
                    # Some network filesystems cannot physically preallocate.
                    # Retain fail-fast checks as the portable fallback.
                    pass
                else:
                    self._file.close()
                    raise
            else:
                self._reserved_bytes = reserve_bytes

    def append(self, array: Any) -> None:
        if self._file.closed:
            raise ValueError("cannot append to a closed array spool")
        value = np.asarray(array)
        if value.dtype.hasobject:
            raise ValueError("array spool does not support object arrays")
        required_end = (
            self._write_offset
            + int(value.nbytes)
            + _ARRAY_SPOOL_APPEND_OVERHEAD_BYTES
        )
        allocated_end = max(self._reserved_bytes, self._write_offset)
        unreserved_growth = max(0, required_end - allocated_end)
        if unreserved_growth:
            _require_array_spool_capacity(
                self.directory,
                required_bytes=unreserved_growth,
                min_free_bytes=self._min_free_bytes,
            )
        # Iteration seeks back to the beginning, and a physical reservation
        # extends the file to its full capacity. Track the actual data end
        # explicitly instead of using SEEK_END in either case.
        self._file.seek(self._write_offset)
        np.save(self._file, value, allow_pickle=False)
        self._write_offset = self._file.tell()
        self._count += 1

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._file.closed:
            raise ValueError("cannot read a closed array spool")
        self._file.flush()
        self._file.seek(0)
        for _ in range(self._count):
            yield np.load(self._file, allow_pickle=False)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> ArraySpool:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class EmptyArraySpool:
    """Zero-I/O spool used when video capture is explicitly disabled."""

    def append(self, _array: Any) -> None:
        raise RuntimeError("cannot append frames when video capture is disabled")

    def __len__(self) -> int:
        return 0

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(())

    def close(self) -> None:
        return None


def _write_json(path: str, payload: Any, *, indent: int) -> None:
    with open(path, "w") as output_file:
        json.dump(payload, output_file, indent=indent)


def _write_jsonl(path: str, records: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "w") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")


def write_policy_chunk_records(
    log_dir: str,
    records: Sequence[Mapping[str, Any]],
) -> str:
    """Write the policy planner/execution audit stream and return its path."""

    path = os.path.join(log_dir, "policy_chunks.jsonl")
    _write_jsonl(path, records)
    return path


def resolve_summary_paths(
    log_dir: str,
    explicit_summary_path: str | None,
) -> tuple[str, str]:
    """Return the mandatory log summary and requested summary destination."""

    default_path = os.path.join(log_dir, "sim_summary.json")
    summary_path = (
        os.path.abspath(explicit_summary_path)
        if explicit_summary_path
        else default_path
    )
    return default_path, summary_path


def write_summary_artifacts(
    log_dir: str,
    summary_path: str,
    summary: Mapping[str, Any],
    frame_records: Sequence[Mapping[str, Any]],
) -> SummaryArtifacts:
    """Write the rollout summary and both compatibility result files."""

    default_summary_path = os.path.join(log_dir, "sim_summary.json")
    _write_json(default_summary_path, summary, indent=2)
    if os.path.abspath(summary_path) != os.path.abspath(default_summary_path):
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        _write_json(summary_path, summary, indent=2)

    ik_results_path = os.path.join(log_dir, "ik_results.json")
    ik_frames = [
        {
            "frame": record["frame"],
            "R": record["arms"]["R"],
            "L": record["arms"]["L"],
        }
        for record in frame_records
    ]
    _write_json(
        ik_results_path,
        {"meta": summary, "frames": ik_frames},
        indent=1,
    )

    penetration_results_path = os.path.join(
        log_dir, "penetration_results.json"
    )
    penetration_frames = [
        {
            "frame": record["frame"],
            "max_pen_mm": record["max_pen_mm"],
            "penetration": record["penetration"],
        }
        for record in frame_records
    ]
    penetration_meta = {
        key: summary[key]
        for key in ("max_pen_mm", "frames_with_pen", "frames")
    }
    _write_json(
        penetration_results_path,
        {"meta": penetration_meta, "frames": penetration_frames},
        indent=1,
    )
    return SummaryArtifacts(
        default_summary_path=default_summary_path,
        summary_path=summary_path,
        ik_results_path=ik_results_path,
        penetration_results_path=penetration_results_path,
    )


def _resolve_ffmpeg() -> str | None:
    """Locate the repository-pinned ffmpeg used to publish H.264 videos.

    ``scripts/bootstrap.sh`` installs ffmpeg next to the venv interpreter; a
    PATH lookup covers non-bootstrap environments.  Returns ``None`` when no
    ffmpeg is available (the OpenCV mp4v intermediate is then published
    unchanged, keeping contract tests runnable without ffmpeg).
    """

    candidate = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return shutil.which("ffmpeg")


# Threads for each H.264 transcode's decoder, filter graph and libx264 encoder.
# Left unset, FFmpeg starts threads per visible host core (libx264 alone about
# 1.5 per core plus lookahead threads), regardless of a container's CPU quota:
# about 66 threads for the 1920x1080 grid on 16 cores. Under a container task
# limit shared with the policy and 3DGS servers, libx264 then cannot start its
# threads and fails with "Generic error in an external library". One thread
# each keeps a transcode at six threads on any host. libx264's output depends
# on its thread count, so this also makes the H.264 bytes independent of the
# host's core count.
FFMPEG_THREADS = "1"

# Decoder threads for each OpenCV video reader. Left unset, OpenCV's FFmpeg
# backend starts one decoder thread per cv2.getNumberOfCPUs() for every open
# reader: 8 per reader under an 8-CPU container quota, 16 on a 16-core host.
# A 30-rollout grid validates with 31 readers open, which then needs about 496
# threads at 16 CPUs and exhausts a 512-task container limit (avcodec_open2
# fails with EAGAIN). One thread decodes in the calling thread, so a reader
# adds no threads on any host; decoded frames do not depend on this count.
VIDEO_DECODER_THREADS = 1


def open_video_capture(path: str | os.PathLike[str]) -> cv.VideoCapture:
    """Open an OpenCV FFmpeg reader that decodes with one thread.

    Every reader in the repository opens through this helper, so the number of
    threads held by open readers no longer grows with the host's CPU count.
    The FFmpeg backend is requested explicitly, with no fallback to another
    backend. If FFmpeg cannot open the file or rejects the parameters, OpenCV
    logs the reason and ``isOpened()`` is False, so callers must check it.
    """

    return cv.VideoCapture(
        os.fspath(path),
        cv.CAP_FFMPEG,
        [cv.CAP_PROP_N_THREADS, VIDEO_DECODER_THREADS],
    )


def _transcode_to_h264(source_path: str, target_path: str, ffmpeg: str) -> None:
    """Re-encode the OpenCV mp4v intermediate into browser-safe H.264."""

    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel", "error",
            "-filter_threads", FFMPEG_THREADS,
            "-threads", FFMPEG_THREADS,
            "-i", source_path,
            "-c:v", "libx264",
            "-threads", FFMPEG_THREADS,
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "medium",
            "-movflags", "+faststart",
            target_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg H.264 transcode failed for {target_path}: "
            f"{result.stderr.strip()[-500:]}"
        )
    if not os.path.isfile(target_path) or os.path.getsize(target_path) == 0:
        raise RuntimeError(f"ffmpeg produced no H.264 output: {target_path}")


def write_rgb_video(frames: Iterable[Any], path: str, fps: float) -> int:
    """Stream RGB frames into an atomically published H.264 rollout MP4.

    OpenCV writes an mp4v intermediate (pip OpenCV ships no H.264 encoder);
    the pinned ffmpeg then transcodes it to H.264/yuv420p so the published
    video plays inline in browsers.  Without ffmpeg the mp4v intermediate is
    published as-is.
    """

    frame_iterator = iter(frames)
    try:
        first_frame = next(frame_iterator)
    except StopIteration:
        return 0
    height, width = first_frame.shape[:2]
    expected_shape = (height, width)
    target_path = os.path.abspath(path)
    target_directory = os.path.dirname(target_path)
    os.makedirs(target_directory, exist_ok=True)
    target_name = os.path.basename(target_path)
    target_stem, target_suffix = os.path.splitext(target_name)
    if not target_suffix:
        target_stem, target_suffix = target_name, ".mp4"
    temporary_fd, temporary_path = tempfile.mkstemp(
        prefix=f".{target_stem}.partial.",
        # OpenCV selects the container from the filename extension, so retain
        # the caller's requested container while still using a hidden temp.
        suffix=target_suffix,
        dir=target_directory,
    )
    os.close(temporary_fd)
    # OpenCV expects to create the container itself.  Keeping only the unique
    # name also lets a failed encode be removed without touching an older,
    # valid output at ``path``.
    os.unlink(temporary_path)
    writer = cv.VideoWriter(
        temporary_path,
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if hasattr(writer, "isOpened") and not writer.isOpened():
        writer.release()
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise RuntimeError(f"failed to open rollout video writer: {path}")
    count = 0
    try:
        try:
            for frame in chain((first_frame,), frame_iterator):
                if frame.shape[:2] != expected_shape:
                    raise ValueError(
                        "rollout video frame dimensions changed: "
                        f"expected {expected_shape}, got {frame.shape[:2]}"
                    )
                writer.write(cv.cvtColor(frame, cv.COLOR_RGB2BGR))
                count += 1
        finally:
            writer.release()
        if not os.path.isfile(temporary_path) or os.path.getsize(temporary_path) == 0:
            raise RuntimeError(f"rollout video writer produced no output: {path}")
        ffmpeg = _resolve_ffmpeg()
        if ffmpeg is None:
            os.replace(temporary_path, target_path)
        else:
            encoded_fd, encoded_path = tempfile.mkstemp(
                prefix=f".{target_stem}.h264.",
                suffix=target_suffix,
                dir=target_directory,
            )
            os.close(encoded_fd)
            try:
                _transcode_to_h264(temporary_path, encoded_path, ffmpeg)
                os.replace(encoded_path, target_path)
            except BaseException:
                if os.path.exists(encoded_path):
                    os.unlink(encoded_path)
                raise
            finally:
                if os.path.exists(temporary_path):
                    os.unlink(temporary_path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return count


__all__ = [
    "ArraySpool",
    "DEFAULT_ARRAY_SPOOL_MIN_FREE_BYTES",
    "EmptyArraySpool",
    "SummaryArtifacts",
    "VIDEO_DECODER_THREADS",
    "open_video_capture",
    "resolve_array_spool_directory",
    "resolve_summary_paths",
    "write_policy_chunk_records",
    "write_rgb_video",
    "write_summary_artifacts",
]
