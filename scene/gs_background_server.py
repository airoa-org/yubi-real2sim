#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""HTTP 3DGS renderer with one thread owning every CUDA operation.

HTTP handlers only parse requests, wait on a bounded render queue, and encode
CPU-resident images.  Model loading, camera tensor construction, Gaussian
rasterization, and the final GPU-to-CPU transfer all happen on the dedicated
owner thread.  This prevents concurrent request threads from using the same
CUDA renderer state while keeping the legacy ``POST /render`` API intact.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import hmac
import io
import inspect
import json
import math
import os
import queue
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import cv2 as cv
import numpy as np
from PIL import Image

if __package__:
    from .gs_background_protocol import (
        BATCH_CAPABILITY,
        BATCH_CONTENT_TYPE,
        BATCH_PROTOCOL_VERSION,
        BatchProtocolError,
        CAPABILITIES_HEADER,
        IDEMPOTENT_BATCH_CAPABILITY,
        IDEMPOTENT_BATCH_PROTOCOL_VERSION,
        KB4_CAPABILITY,
        KB4_FIELDS,
        KB4_NEAR_CAPABILITY,
        RESULT_TTL_HEADER,
        RGBD_BATCH_CAPABILITY,
        RGBD_BATCH_CONTENT_TYPE,
        pack_jpeg_batch,
        pack_rgbd_batch,
        unpack_jpeg_batch,
        validate_request_id,
        validate_camera_settings,
        validate_kb4_near,
        kb4_valid_mask,
    )
else:
    # The production runner executes this file directly. Importing through
    # ``scene.gs_background_protocol`` here would cache this repository's
    # ``scene`` namespace before GaussianWrapping imports its own
    # ``scene.gaussian_model`` package, shadowing the renderer dependency.
    from gs_background_protocol import (  # type: ignore[no-redef]
        BATCH_CAPABILITY,
        BATCH_CONTENT_TYPE,
        BATCH_PROTOCOL_VERSION,
        BatchProtocolError,
        CAPABILITIES_HEADER,
        IDEMPOTENT_BATCH_CAPABILITY,
        IDEMPOTENT_BATCH_PROTOCOL_VERSION,
        KB4_CAPABILITY,
        KB4_FIELDS,
        KB4_NEAR_CAPABILITY,
        RESULT_TTL_HEADER,
        RGBD_BATCH_CAPABILITY,
        RGBD_BATCH_CONTENT_TYPE,
        pack_jpeg_batch,
        pack_rgbd_batch,
        unpack_jpeg_batch,
        validate_request_id,
        validate_camera_settings,
        validate_kb4_near,
        kb4_valid_mask,
    )


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(float(n), 1e-12)


class RenderQueueFull(RuntimeError):
    """Raised when the bounded GPU owner queue cannot accept more work."""


@contextlib.contextmanager
def _exclusive_render_lock(path: str):
    """Join the node-local MuJoCo/3DGS lock for one physical GPU."""

    if not path:
        yield
        return
    lock_path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _locked_gpu_renderer(initializer, render_lock: str):
    """Initialize and invoke one GPU renderer inside the same lock domain."""

    lock_path = str(render_lock).strip()
    if not lock_path:
        return initializer()
    with _exclusive_render_lock(lock_path):
        render_many = initializer()

    def render_many_locked(payloads):
        with _exclusive_render_lock(lock_path):
            return render_many(payloads)

    return render_many_locked


class RequestIdConflict(RuntimeError):
    """Raised when one request id is reused for a different render payload."""


class EncodedResultTooLarge(RuntimeError):
    """Raised when encoded recovery bytes exceed their admission reservation."""


@dataclass
class _RegistryEntry:
    payload_digest: str
    future: Future[Any]
    created_at: float
    finished_at: float | None = None
    result_bytes_reservation: int = 1


class IdempotentRenderRegistry:
    """Count- and byte-bounded mapping from one id/payload to one Future.

    Pending entries are never evicted. Completed entries remain available for
    ``result_ttl_sec`` so a client that timed out can prove terminal state and
    retrieve the original result without submitting another GPU render.
    """

    def __init__(
        self,
        submitter: Callable[[Sequence[dict[str, Any]]], Future[Any]],
        *,
        capacity: int = 128,
        result_byte_capacity: int = 512 * 1024 * 1024,
        result_ttl_sec: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if int(capacity) < 1:
            raise ValueError("registry capacity must be >= 1")
        if int(result_byte_capacity) < 1:
            raise ValueError("registry result_byte_capacity must be >= 1")
        if not math.isfinite(float(result_ttl_sec)) or float(result_ttl_sec) <= 0:
            raise ValueError("registry result_ttl_sec must be finite and > 0")
        self._submitter = submitter
        self._capacity = int(capacity)
        self._result_byte_capacity = int(result_byte_capacity)
        self._reserved_result_bytes = 0
        self._result_ttl_sec = float(result_ttl_sec)
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, _RegistryEntry] = OrderedDict()

    @staticmethod
    def _payload_digest(payloads: Sequence[dict[str, Any]]) -> str:
        canonical = json.dumps(
            list(payloads),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _expire_locked(self, now: float) -> None:
        expired = [
            request_id
            for request_id, entry in self._entries.items()
            if entry.finished_at is not None
            and now - entry.finished_at >= self._result_ttl_sec
        ]
        for request_id in expired:
            entry = self._entries.pop(request_id, None)
            if entry is not None:
                self._reserved_result_bytes -= entry.result_bytes_reservation

    def _mark_finished(self, request_id: str, future: Future[Any]) -> None:
        with self._lock:
            current = self._entries.get(request_id)
            if current is not None and current.future is future:
                if current.finished_at is None:
                    current.finished_at = self._clock()

    def submit_or_join(
        self,
        request_id: str,
        payloads: Sequence[dict[str, Any]],
        *,
        result_bytes_reservation: int = 1,
    ) -> tuple[Future[Any], bool]:
        """Return the sole Future for this id, creating it only when absent."""

        request_id = validate_request_id(request_id)
        result_bytes_reservation = int(result_bytes_reservation)
        if result_bytes_reservation < 1:
            raise ValueError("result_bytes_reservation must be >= 1")
        snapshots = tuple(dict(payload) for payload in payloads)
        digest = self._payload_digest(snapshots)
        with self._lock:
            self._expire_locked(self._clock())
            existing = self._entries.get(request_id)
            if existing is not None:
                if existing.payload_digest != digest:
                    raise RequestIdConflict(
                        f"request_id {request_id!r} was already used for a "
                        "different render payload"
                    )
                self._entries.move_to_end(request_id)
                return existing.future, False
            if len(self._entries) >= self._capacity:
                raise RenderQueueFull(
                    "idempotent render registry is full; pending or retained "
                    "requests cannot be evicted safely"
                )
            if (
                self._reserved_result_bytes + result_bytes_reservation
                > self._result_byte_capacity
            ):
                raise RenderQueueFull(
                    "idempotent render registry byte budget is full; retained "
                    "results cannot be evicted before their advertised TTL"
                )
            self._reserved_result_bytes += result_bytes_reservation
            try:
                future = self._submitter(snapshots)
            except BaseException:
                self._reserved_result_bytes -= result_bytes_reservation
                raise
            entry = _RegistryEntry(
                digest,
                future,
                self._clock(),
                result_bytes_reservation=result_bytes_reservation,
            )
            self._entries[request_id] = entry
            future.add_done_callback(
                lambda completed: self._mark_finished(request_id, completed)
            )
            return future, True

    def acknowledge(self, request_id: str) -> bool:
        """Drop a terminal cached result after the client received it."""

        request_id = validate_request_id(request_id)
        with self._lock:
            self._expire_locked(self._clock())
            entry = self._entries.get(request_id)
            if entry is None or not entry.future.done():
                return False
            removed = self._entries.pop(request_id, None)
            if removed is not None:
                self._reserved_result_bytes -= removed.result_bytes_reservation
            return True

    def status(self, request_id: str) -> tuple[str, str | None]:
        """Return ``pending``, ``done``, ``error``, or ``unknown``."""

        request_id = validate_request_id(request_id)
        with self._lock:
            self._expire_locked(self._clock())
            entry = self._entries.get(request_id)
            if entry is None:
                return "unknown", None
            self._entries.move_to_end(request_id)
            future = entry.future
            if not future.done():
                return "pending", None
            if future.cancelled():
                return "error", "CancelledError: render future was cancelled"
            try:
                error = future.exception()
            except BaseException as exc:
                error = exc
            if error is not None:
                return "error", f"{type(error).__name__}: {error}"
            # Prevent expiry in the narrow interval between a successful
            # status probe and the client's same-id result retrieval.
            entry.finished_at = self._clock()
            return "done", None


@dataclass(frozen=True)
class _RenderJob:
    payloads: tuple[dict[str, Any], ...]
    future: Future[list[tuple[np.ndarray, np.ndarray | None]]]


_STOP = object()


class GpuRenderOwner:
    """Run initialization and all render work on one dedicated thread."""

    def __init__(
        self,
        initializer: Callable[
            [], Callable[[Sequence[dict[str, Any]]], list[tuple[np.ndarray, np.ndarray | None]]]
        ],
        *,
        queue_capacity: int = 64,
        thread_name: str = "gs-cuda-owner",
    ) -> None:
        if int(queue_capacity) < 1:
            raise ValueError("queue_capacity must be >= 1")
        self._initializer = initializer
        self._queue_capacity = int(queue_capacity)
        # Reserve one queue slot for the stop sentinel. Job admission is
        # limited separately, so close() can always enqueue shutdown even when
        # every public job slot is occupied.
        self._queue: queue.Queue[_RenderJob | object] = queue.Queue(
            maxsize=self._queue_capacity + 1
        )
        self._job_slots = threading.BoundedSemaphore(self._queue_capacity)
        self._state_lock = threading.Lock()
        self._ready: Future[None] = Future()
        self._stopped = threading.Event()
        self._started = False
        self._closing = False
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=True,
        )

    @property
    def thread_ident(self) -> int | None:
        return self._thread.ident

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def start(self, *, timeout: float | None = None) -> None:
        with self._state_lock:
            if self._started:
                raise RuntimeError("GPU render owner has already been started")
            self._started = True
        self._thread.start()
        self._ready.result(timeout=timeout)

    def submit(self, payloads: Sequence[dict[str, Any]]) -> Future[list[tuple[np.ndarray, np.ndarray | None]]]:
        with self._state_lock:
            if not self._started:
                raise RuntimeError("GPU render owner has not been started")
            if self._closing or self._stopped.is_set():
                raise RuntimeError("GPU render owner is not running")
            if not self._job_slots.acquire(blocking=False):
                raise RenderQueueFull("GPU render queue is full")
            future: Future[list[tuple[np.ndarray, np.ndarray | None]]] = Future()
            job = _RenderJob(tuple(dict(payload) for payload in payloads), future)
            try:
                self._queue.put_nowait(job)
            except queue.Full as exc:  # Reserved sentinel slot makes this defensive.
                self._job_slots.release()
                raise RenderQueueFull("GPU render queue is full") from exc
        return future

    def close(self, *, timeout: float | None = 30.0) -> None:
        with self._state_lock:
            if not self._started:
                return
            if not self._closing and not self._stopped.is_set():
                self._closing = True
                # A dedicated slot is reserved for this sentinel; unlike a
                # best-effort queue put, shutdown cannot be lost at capacity.
                self._queue.put_nowait(_STOP)
        self._thread.join(
            timeout=None if timeout is None else max(float(timeout), 0.0)
        )
        if self._thread.is_alive():
            raise RuntimeError("GPU render owner did not stop before timeout")

    def _fail_queued(self, exc: BaseException) -> None:
        while True:
            try:
                pending = self._queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(pending, _RenderJob):
                self._job_slots.release()
                if not pending.future.done():
                    pending.future.set_exception(exc)

    def _run(self) -> None:
        try:
            render_many = self._initializer()
        except BaseException as exc:
            self._ready.set_exception(exc)
            with self._state_lock:
                self._closing = True
            self._fail_queued(exc)
            self._stopped.set()
            return
        self._ready.set_result(None)
        active_job: _RenderJob | None = None
        try:
            while True:
                item = self._queue.get()
                if item is _STOP:
                    return
                if not isinstance(item, _RenderJob):
                    continue
                self._job_slots.release()
                active_job = item
                try:
                    images = render_many(item.payloads)
                except BaseException as exc:
                    if not item.future.done():
                        item.future.set_exception(exc)
                    # Ordinary request errors are isolated to one job. Fatal
                    # exceptions must stop the owner after resolving the
                    # active Future, otherwise an HTTP handler can wait forever.
                    if not isinstance(exc, Exception):
                        raise
                else:
                    if not item.future.done():
                        item.future.set_result(images)
                finally:
                    active_job = None
        except BaseException as exc:
            with self._state_lock:
                self._closing = True
            if active_job is not None and not active_job.future.done():
                active_job.future.set_exception(exc)
            self._fail_queued(exc)
            return
        finally:
            self._stopped.set()


def _cache_static_sh(gaussians):
    import torch

    device = gaussians._features_rest.device
    staged = torch.cat((
        gaussians._features_dc.detach().cpu(), gaussians._features_rest.detach().cpu()
    ), dim=1)
    gaussians._features_dc = gaussians._features_rest = None  # Freed before the GPU copy, so loading peaks no higher than main.
    torch.cuda.empty_cache()
    features = staged.to(device)
    gaussians._render_features = features
    gaussians._features_dc = features[:, :1, :]
    gaussians._features_rest = features[:, 1:, :]


def _static_sh_model(base):
    class StaticGaussianModel(base):
        @property
        def get_features(self):
            return self._render_features

    return StaticGaussianModel


def _native_kb4_supported(render_ours, rasterization) -> bool:
    """Check the adapter, Python settings and loaded native forward ABI together."""

    try:  # an importable but older checkout lacks the wrist-model API that validation uses
        probe = dict(camera_model="kb4", fx=1.0, fy=1.0, cx=1.0, cy=1.0,
                     k1=0.0, k2=0.0, k3=0.0, k4=0.0, theta_max=1.0)
        kb4_valid_mask(3, 3, validate_camera_settings(probe, 3, 3))
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        print(f"[gsbg] KB4 disabled: install this checkout in the server interpreter ({exc})", flush=True)
        return False
    if "camera_settings" not in inspect.signature(render_ours).parameters:
        return False
    fields = getattr(rasterization.GaussianRasterizationSettings, "_fields", ())
    if not {"camera_model", "kb4_near", *KB4_FIELDS}.issubset(fields):
        print("[gsbg] KB4 disabled: the installed rasterizer lacks kb4_near; rebuild it from the "
              "pinned GaussianWrapping (scripts/bootstrap.sh)", flush=True)
        return False
    signature = getattr(rasterization._C.rasterize_gaussians, "__doc__", "") or ""
    # fx, fy, cx, cy, k1..k4, theta_max and kb4_near (the pinned GaussianWrapping, after cd8cb18)
    return "arg25: Optional[Annotated[list[float], FixedSize(10)]]" in signature


def _make_gpu_renderer(args, ply_path: Path):
    """Return an initializer that imports and initializes CUDA on its caller."""

    gw = args.gw_repo.resolve()

    def initialize():
        # These imports and every operation that can initialize or synchronize
        # CUDA deliberately live inside the owner thread.
        #
        # The only sys.path edit in this repository (the allowlist in
        # tests/distribution/test_import_roots.py names it). GaussianWrapping
        # is not an installable package: its renderer modules are importable
        # only from the checkout given by --gw-repo. Its top-level ``scene``
        # package also clashes with this repository's ``scene`` package, so
        # the runners start this file by path (sys.path[0] is then scene/,
        # not the repository root) and the protocol import above avoids
        # ``scene.*``. Keep this here until GaussianWrapping is installable
        # or its package no longer shadows ours.
        sys.path.insert(0, str(gw / "gaussian_wrapping"))
        sys.path.insert(0, str(gw))
        sys.path.insert(0, str(gw / "submodules"))
        import torch
        from gaussian_renderer import GaussianModel
        from gaussian_renderer.ours import render_ours
        import diff_gaussian_rasterization_ours as rasterization
        from utils.graphics_utils import getProjectionMatrix, getWorld2View2

        args.kb4_supported = _native_kb4_supported(render_ours, rasterization)
        if args.kb4_supported:
            for label, path in (
                ("adapter", inspect.getfile(render_ours)),
                ("rasterization", rasterization.__file__),
                ("native", rasterization._C.__file__),
            ):
                path = Path(path).resolve()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                print(f"[gsbg] KB4 development dependency {label}={path} sha256={digest}", flush=True)

        torch.set_num_threads(int(args.torch_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # A dependency may have initialized the inter-op pool while being
            # imported.  CUDA ownership is unaffected, so retain that pool.
            pass

        class _Pipe:
            convert_SHs_python = False
            compute_cov3D_python = False
            debug = False  # noqa: V107  # GaussianWrapping render_ours() reads pipe.debug (gaussian_wrapping/gaussian_renderer/ours.py:54 at cd8cb18f)

        class _Cam:
            pass

        print(f"[gsbg] loading {ply_path}", flush=True)
        gaussians = _static_sh_model(GaussianModel)(3)
        gaussians.load_ply(str(ply_path))
        _cache_static_sh(gaussians)  # The scene is immutable across render requests.
        print(f"[gsbg] loaded {gaussians.get_xyz.shape[0]} gaussians", flush=True)
        pipe = _Pipe()
        try:
            bg_values = [float(token) for token in args.bg_color.split()]
        except ValueError as exc:
            raise SystemExit(f"--bg-color must be three floats: {args.bg_color!r}") from exc
        if len(bg_values) != 3 or not all(0.0 <= value <= 1.0 for value in bg_values):
            raise SystemExit(f"--bg-color must be three 0..1 floats: {args.bg_color!r}")
        bg = torch.tensor(bg_values, dtype=torch.float32, device="cuda")
        print(f"[gsbg] far-field colour {bg_values}", flush=True)

        def render_one(
            payload: dict[str, Any]
        ) -> tuple[np.ndarray, np.ndarray | None]:
            width = payload.get("width", args.width)
            height = payload.get("height", args.height)
            settings = None
            valid = None
            if "camera_settings" in payload:
                if not args.kb4_supported:
                    raise ValueError("installed GaussianWrapping lacks kb4-radial-depth-v1 native support")
                settings = validate_camera_settings(payload["camera_settings"], width, height)
            width, height = int(width), int(height)
            if width < 1 or height < 1:
                raise ValueError("render width and height must be positive")
            if width * height > int(args.max_render_pixels):
                raise ValueError(
                    f"render size {width}x{height} exceeds "
                    f"{int(args.max_render_pixels)} pixels"
                )
            if settings is not None:
                valid = kb4_valid_mask(width, height, settings)
                if "kb4_near" not in payload:
                    raise ValueError("a KB4 request must carry kb4_near (GS units)")
                settings = {**settings, "kb4_near": validate_kb4_near(payload["kb4_near"])}
            pos = np.asarray(payload["pos"], dtype=np.float64)
            forward = np.asarray(payload["forward"], dtype=np.float64)
            up = np.asarray(payload["up"], dtype=np.float64)
            if pos.shape != (3,) or forward.shape != (3,) or up.shape != (3,):
                raise ValueError("pos, forward, and up must each contain 3 values")
            if not all(np.all(np.isfinite(value)) for value in (pos, forward, up)):
                raise ValueError("camera vectors must contain only finite values")
            if np.linalg.norm(forward) <= 1e-9 or np.linalg.norm(up) <= 1e-9:
                raise ValueError("forward and up vectors must be non-zero")
            forward = _norm(forward)
            up = _norm(up)
            right_raw = np.cross(forward, up)
            if np.linalg.norm(right_raw) <= 1e-6:
                raise ValueError("forward and up vectors must not be parallel")
            right = _norm(right_raw)
            up = _norm(np.cross(right, forward))
            fovy_deg = float(payload.get("fovy_deg", 45.0))
            if not math.isfinite(fovy_deg) or not 0.1 < fovy_deg < 179.0:
                raise ValueError("fovy_deg must be finite and between 0.1 and 179")
            fovy = math.radians(fovy_deg)
            fovx = 2.0 * math.atan(
                math.tan(fovy / 2.0) * (width / max(height, 1))
            )
            znear = float(payload.get("znear", 0.01))
            zfar = float(payload.get("zfar", 100.0))
            if (
                not math.isfinite(znear)
                or not math.isfinite(zfar)
                or znear <= 0.0
                or zfar <= znear
            ):
                raise ValueError("znear and zfar must be finite with 0 < znear < zfar")

            # MuJoCo/OpenGL camera: +x right, +y up, -z forward.
            # 3DGS camera used here: +x right, +y down, +z forward.
            r_w2c = np.stack([right, -up, forward], axis=0)
            t_w2c = -r_w2c @ pos
            proj = getProjectionMatrix(
                znear=znear,
                zfar=zfar,
                fovX=fovx,
                fovY=fovy,
            ).transpose(0, 1).cuda()
            wv = torch.tensor(
                getWorld2View2(r_w2c.T, t_w2c, np.zeros(3), 1.0)
            ).transpose(0, 1).cuda()
            full = (wv.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

            cam = _Cam()
            cam.FoVx = fovx  # noqa: V101  # render_ours() reads it
            cam.FoVy = fovy  # noqa: V101  # render_ours() reads it
            cam.image_width = width  # noqa: V101  # render_ours() reads it
            cam.image_height = height  # noqa: V101  # render_ours() reads it
            cam.world_view_transform = wv  # noqa: V101  # render_ours() reads it
            cam.projection_matrix = proj
            cam.full_proj_transform = full  # noqa: V101  # render_ours() reads it
            cam.camera_center = wv.inverse()[3, :3]  # noqa: V101  # render_ours() reads it
            cam.zfar = zfar
            cam.znear = znear
            cam.scale = 1.0

            want_depth = bool(payload.get("want_depth", False))
            if settings is None:
                out = render_ours(cam, gaussians, pipe, bg, require_depth=want_depth)
            else:
                out = render_ours(cam, gaussians, pipe, bg, require_depth=want_depth, camera_settings=settings)
            # .cpu() is the synchronization boundary.  Return only owned CPU
            # ndarrays so no CUDA tensor escapes the owner thread.
            rgb = (
                out["render"]
                .clamp(0.0, 1.0)
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
                .__mul__(255.0)
                .astype(np.uint8)
            )
            if valid is not None:
                rgb[~valid] = 0
            if not want_depth:
                return rgb, None
            # ``median_depth`` is radial for KB4 and optical-axis for pinhole,
            # depth in GS-native distance units. Pixels the median-threshold
            # crossing never reaches (no/insufficient gaussian coverage, i.e.
            # background) are written as exactly 0.0 by the CUDA kernel
            # (cuda_rasterizer/sample_forward.cu: ``inside_range[p] ? ... :
            # 0.f``) -- that is the authoritative "no reconstructed geometry
            # here" sentinel, not an approximation.
            depth = (
                out["median_depth"]
                .detach()
                .cpu()
                .numpy()
                .reshape(height, width)
                .astype(np.float32)
            )
            if valid is not None:
                depth[~valid] = np.nan
            return rgb, depth

        def render_many(
            payloads: Sequence[dict[str, Any]]
        ) -> list[tuple[np.ndarray, np.ndarray | None]]:
            with torch.inference_mode():
                return [render_one(payload) for payload in payloads]

        # Model loading and tensor preparation above enqueue asynchronous CUDA
        # work. Complete it before the initializer returns so a surrounding
        # physical-GPU lock is a real completion boundary.
        torch.cuda.synchronize()
        return render_many

    return initialize


def _encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    """Encode one CPU-resident RGB array without touching CUDA."""

    output = io.BytesIO()
    Image.fromarray(image).save(output, format="JPEG", quality=int(quality))
    return output.getvalue()


_DEPTH_QUANTIZATION_LEVELS = 65534.0


def _encode_depth_png(depth: np.ndarray, znear: float, zfar: float) -> bytes:
    """Quantize GS-native depth to a 16-bit PNG matching the client decoder.

    Code 0 means "no valid reconstructed geometry": non-finite, non-positive,
    or outside [znear, zfar]. Codes 1..65535 linearly cover [znear, zfar].
    See ``gs_background_client._decode_depth_png`` for the matching decode.
    """

    if not math.isfinite(znear) or not math.isfinite(zfar) or zfar <= znear:
        raise ValueError("znear and zfar must be finite with znear < zfar")
    depth = np.asarray(depth, dtype=np.float64)
    quantized = np.zeros(depth.shape, dtype=np.uint16)
    # Do not turn an out-of-contract depth into a plausible occluder at the
    # near/far endpoint. RGB remains usable, while code 0 makes the client
    # fall back conservatively for the z-test at that pixel.
    valid = (
        np.isfinite(depth)
        & (depth >= float(znear))
        & (depth <= float(zfar))
    )
    if np.any(valid):
        unit = (depth[valid] - znear) / (zfar - znear)
        codes = np.rint(1.0 + unit * _DEPTH_QUANTIZATION_LEVELS)
        quantized[valid] = np.clip(codes, 1.0, 65535.0).astype(np.uint16)
    ok, encoded = cv.imencode(".png", quantized)
    if not ok:
        raise RuntimeError("failed to encode depth PNG")
    return encoded.tobytes()


def _encoded_result_reservation_bytes(
    payloads: Sequence[dict[str, Any]],
    *,
    default_width: int,
    default_height: int,
) -> int:
    """Return a conservative admission budget for a packed JPEG/RGBD batch."""

    total_pixels = sum(
        int(payload.get("width", default_width))
        * int(payload.get("height", default_height))
        for payload in payloads
    )
    wants_depth = any(bool(payload.get("want_depth", False)) for payload in payloads)
    # Eight bytes per RGB pixel plus a generous per-image framing allowance
    # exceeds ordinary PIL JPEG output while keeping admission deterministic.
    # A 16-bit depth PNG rarely compresses below 2 raw bytes/pixel; reserve
    # 3 bytes/pixel of margin for its DEFLATE framing when depth is requested.
    reservation = 12 + total_pixels * 8 + len(payloads) * 4096
    if wants_depth:
        reservation += total_pixels * 3 + len(payloads) * 4096
    return reservation


def _chain_encoded_batch(
    owner: GpuRenderOwner,
    jpeg_executor: ThreadPoolExecutor,
    payloads: Sequence[dict[str, Any]],
    quality: int,
    *,
    admission: threading.BoundedSemaphore | None = None,
    max_encoded_bytes: int | None = None,
) -> Future[bytes]:
    """Submit GPU work, then retain only compressed CPU bytes for recovery."""

    if admission is not None and not admission.acquire(blocking=False):
        raise RenderQueueFull("JPEG encode backlog is full")
    admission_lock = threading.Lock()
    admission_released = False

    def release_admission() -> None:
        nonlocal admission_released
        if admission is None:
            return
        with admission_lock:
            if not admission_released:
                admission_released = True
                admission.release()

    encoded: Future[bytes] = Future()
    try:
        raw = owner.submit(payloads)
    except BaseException:
        release_admission()
        raise

    def transfer_encode_result(encoding: Future[bytes]) -> None:
        if encoded.done():
            release_admission()
            return
        try:
            body = encoding.result()
            if max_encoded_bytes is not None and len(body) > int(max_encoded_bytes):
                raise EncodedResultTooLarge(
                    f"encoded JPEG batch is {len(body)} bytes; admission "
                    f"reserved {int(max_encoded_bytes)} bytes"
                )
            encoded.set_result(body)
        except BaseException as exc:
            encoded.set_exception(exc)
        finally:
            release_admission()

    def submit_encoding(
        rendering: Future[list[tuple[np.ndarray, np.ndarray | None]]]
    ) -> None:
        if encoded.done():
            release_admission()
            return
        try:
            images = rendering.result()
        except BaseException as exc:
            encoded.set_exception(exc)
            release_admission()
            return

        def encode() -> bytes:
            wants_depth = any(
                bool(payload.get("want_depth", False)) for payload in payloads
            )
            if not wants_depth:
                return pack_jpeg_batch(
                    _encode_jpeg(rgb, quality) for rgb, _depth in images
                )
            items = []
            for payload, (rgb, depth) in zip(payloads, images):
                jpeg = _encode_jpeg(rgb, quality)
                depth_png = (
                    _encode_depth_png(
                        depth, float(payload["znear"]), float(payload["zfar"])
                    )
                    if depth is not None and payload.get("want_depth", False)
                    else None
                )
                items.append((jpeg, depth_png))
            return pack_rgbd_batch(items)

        try:
            encoding = jpeg_executor.submit(encode)
        except BaseException as exc:
            encoded.set_exception(exc)
            release_admission()
            return
        encoding.add_done_callback(transfer_encode_result)

    raw.add_done_callback(submit_encoding)
    return encoded


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, type=Path)
    default_gw_repo = Path(
        os.environ.get(
            "GAUSSIAN_WRAPPING_REPO",
            Path(__file__).resolve().parents[1] / "third_party/GaussianWrapping",
        )
    )
    ap.add_argument("--gw-repo", default=default_gw_repo, type=Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--jpeg-quality", type=int, default=82)
    ap.add_argument("--ready-token", default="")
    ap.add_argument(
        "--render-lock",
        default="",
        help=(
            "node-local lock shared with MuJoCo workers on this physical GPU"
        ),
    )
    ap.add_argument("--render-queue-capacity", type=int, default=64)
    ap.add_argument("--render-registry-capacity", type=int, default=128)
    ap.add_argument(
        "--render-registry-max-bytes",
        type=int,
        default=512 * 1024 * 1024,
    )
    ap.add_argument("--render-result-ttl-sec", type=float, default=300.0)
    ap.add_argument("--jpeg-threads", type=int, default=2)
    ap.add_argument("--jpeg-backlog-capacity", type=int, default=8)
    ap.add_argument("--max-batch-size", type=int, default=64)
    ap.add_argument(
        "--max-batch-render-pixels",
        type=int,
        default=4096 * 4096,
    )
    ap.add_argument("--max-request-bytes", type=int, default=2 * 1024 * 1024)
    ap.add_argument("--max-render-pixels", type=int, default=4096 * 4096)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument(
        "--bg-color",
        default="0 0 0",
        help=(
            "far-field colour as three 0..1 floats, used where no gaussian "
            "covers a pixel. The depth sentinel is the rasterizer's own "
            "median_depth==0 and is unaffected by this."
        ),
    )
    args = ap.parse_args()
    if args.render_queue_capacity < 1:
        ap.error("--render-queue-capacity must be >= 1")
    if args.render_registry_capacity < 1:
        ap.error("--render-registry-capacity must be >= 1")
    if args.render_registry_max_bytes < 1:
        ap.error("--render-registry-max-bytes must be >= 1")
    if (
        not math.isfinite(args.render_result_ttl_sec)
        or args.render_result_ttl_sec <= 0
    ):
        ap.error("--render-result-ttl-sec must be finite and > 0")
    if args.jpeg_threads < 1:
        ap.error("--jpeg-threads must be >= 1")
    if args.jpeg_backlog_capacity < 1:
        ap.error("--jpeg-backlog-capacity must be >= 1")
    if not 1 <= args.jpeg_quality <= 95:
        ap.error("--jpeg-quality must be in 1..95")
    if args.max_batch_size < 1:
        ap.error("--max-batch-size must be >= 1")
    if args.max_batch_render_pixels < 1:
        ap.error("--max-batch-render-pixels must be >= 1")
    if args.max_request_bytes < 1:
        ap.error("--max-request-bytes must be >= 1")
    if args.max_render_pixels < 1:
        ap.error("--max-render-pixels must be >= 1")
    if args.torch_threads < 1:
        ap.error("--torch-threads must be >= 1")

    ply_path = args.ply.expanduser().resolve()
    if not ply_path.is_file():
        raise SystemExit(f"3DGS PLY is not a file: {ply_path}")
    ply_digest = hashlib.sha256()
    with ply_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            ply_digest.update(chunk)
    # Keep the authenticated body exactly backward compatible: the launcher
    # compares it by equality. Unauthenticated health omits the secret token;
    # new capabilities are advertised in response headers.
    identity = {
        "status": "ok",
        "ready_token": args.ready_token,
        "ply_path": str(ply_path),
        "ply_sha256": ply_digest.hexdigest(),
    }

    args.kb4_supported = False
    initialize_renderer = _make_gpu_renderer(args, ply_path)

    def initialize_locked_renderer():
        # Loading the CUDA model creates a context and allocates device state,
        # so it obeys the same physical-GPU exclusion boundary as requests.
        return _locked_gpu_renderer(initialize_renderer, args.render_lock)

    owner = GpuRenderOwner(
        initialize_locked_renderer,
        queue_capacity=args.render_queue_capacity,
    )
    owner.start()
    jpeg_executor = ThreadPoolExecutor(
        max_workers=args.jpeg_threads,
        thread_name_prefix="gs-jpeg",
    )
    jpeg_admission = threading.BoundedSemaphore(args.jpeg_backlog_capacity)

    def submit_encoded_batch(
        payloads: Sequence[dict[str, Any]],
    ) -> Future[bytes]:
        reservation = _encoded_result_reservation_bytes(
            payloads,
            default_width=args.width,
            default_height=args.height,
        )
        return _chain_encoded_batch(
            owner,
            jpeg_executor,
            payloads,
            args.jpeg_quality,
            admission=jpeg_admission,
            max_encoded_bytes=reservation,
        )

    registry = IdempotentRenderRegistry(
        submit_encoded_batch,
        capacity=args.render_registry_capacity,
        result_byte_capacity=args.render_registry_max_bytes,
        result_ttl_sec=args.render_result_ttl_sec,
    )

    class Handler(BaseHTTPRequestHandler):
        def _send_bytes(
            self,
            status: int,
            body: bytes,
            content_type: str,
            *,
            capabilities: bool = False,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if capabilities:
                advertised_capabilities = [
                    BATCH_CAPABILITY,
                    RGBD_BATCH_CAPABILITY,
                ]
                if args.kb4_supported:
                    advertised_capabilities.append(KB4_CAPABILITY)
                    advertised_capabilities.append(KB4_NEAR_CAPABILITY)
                result_ttl = float(args.render_result_ttl_sec)
                if math.isfinite(result_ttl) and result_ttl > 0.0:
                    advertised_capabilities.append(
                        IDEMPOTENT_BATCH_CAPABILITY
                    )
                    self.send_header(RESULT_TTL_HEADER, f"{result_ttl:g}")
                self.send_header(
                    CAPABILITIES_HEADER,
                    ",".join(advertised_capabilities),
                )
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # The client may time out while its already-queued render is
                # completing.  The owner remains healthy for later requests.
                pass

        def _send_exception(self, status: int, exc: BaseException) -> None:
            message = f"{type(exc).__name__}: {exc}".encode(
                "utf-8", errors="replace"
            )
            self._send_bytes(status, message, "text/plain; charset=utf-8")

        def _send_json(self, status: int, value: Any) -> None:
            body = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
            self._send_bytes(status, body, "application/json")

        def _authorized(self) -> bool:
            if not args.ready_token:
                return True
            return hmac.compare_digest(
                self.headers.get("X-Real2Sim-GS-Token", "").encode("utf-8"),
                args.ready_token.encode("utf-8"),
            )

        def _read_json(self) -> Any:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > int(args.max_request_bytes):
                raise ValueError(
                    f"request body length {length} is outside 1..{args.max_request_bytes}"
                )
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self):  # noqa: V105  # BaseHTTPRequestHandler hook
            parsed = urlsplit(self.path)
            if parsed.path == "/health":
                health_identity = (
                    identity
                    if self._authorized()
                    else {
                        key: value
                        for key, value in identity.items()
                        if key != "ready_token"
                    }
                )
                body = (
                    json.dumps(health_identity, sort_keys=True) + "\n"
                ).encode("utf-8")
                self._send_bytes(
                    200,
                    body,
                    "application/json",
                    capabilities=True,
                )
                return
            if parsed.path == "/render_status":
                if not self._authorized():
                    self._send_bytes(
                        409,
                        b"server identity token mismatch",
                        "text/plain",
                    )
                    return
                try:
                    values = parse_qs(
                        parsed.query,
                        keep_blank_values=True,
                        strict_parsing=True,
                    ).get("request_id", [])
                    if len(values) != 1:
                        raise ValueError(
                            "/render_status requires exactly one request_id"
                        )
                    request_id = validate_request_id(values[0])
                    status, error = registry.status(request_id)
                    response = {
                        "version": IDEMPOTENT_BATCH_PROTOCOL_VERSION,
                        "request_id": request_id,
                        "status": status,
                    }
                    if error is not None:
                        response["error"] = error
                    self._send_json(200, response)
                except (BatchProtocolError, ValueError) as exc:
                    self._send_exception(400, exc)
                return
            self._send_bytes(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: V105  # BaseHTTPRequestHandler hook
            if self.path not in ("/render", "/render_batch", "/render_ack"):
                self._send_bytes(404, b"not found", "text/plain")
                return
            if not self._authorized():
                self._send_bytes(409, b"server identity token mismatch", "text/plain")
                return
            try:
                request = self._read_json()
                if self.path == "/render_ack":
                    if not isinstance(request, dict):
                        raise ValueError(
                            "/render_ack body must be a JSON object"
                        )
                    if (
                        request.get("version")
                        != IDEMPOTENT_BATCH_PROTOCOL_VERSION
                    ):
                        raise ValueError(
                            "/render_ack requires version "
                            f"{IDEMPOTENT_BATCH_PROTOCOL_VERSION}"
                        )
                    request_id = validate_request_id(
                        request.get("request_id")
                    )
                    acknowledged = registry.acknowledge(request_id)
                    self._send_json(
                        200,
                        {
                            "version": IDEMPOTENT_BATCH_PROTOCOL_VERSION,
                            "request_id": request_id,
                            "acknowledged": acknowledged,
                        },
                    )
                    return
                if self.path == "/render":
                    if not isinstance(request, dict):
                        raise ValueError("/render body must be a JSON object")
                    packed = submit_encoded_batch([request]).result()
                    if request.get("want_depth", False):
                        # Keep the opt-in RGBD framing intact; only legacy
                        # (no want_depth) callers get the raw unwrapped JPEG.
                        self._send_bytes(200, packed, RGBD_BATCH_CONTENT_TYPE, capabilities="camera_settings" in request)
                        return
                    body = unpack_jpeg_batch(packed, expected_count=1)[0]
                    self._send_bytes(200, body, "image/jpeg")
                    return

                if not isinstance(request, dict):
                    raise ValueError("/render_batch body must be a JSON object")
                version = request.get("version")
                if version not in (
                    BATCH_PROTOCOL_VERSION,
                    IDEMPOTENT_BATCH_PROTOCOL_VERSION,
                ):
                    raise ValueError(
                        "/render_batch requires version "
                        f"{BATCH_PROTOCOL_VERSION} or "
                        f"{IDEMPOTENT_BATCH_PROTOCOL_VERSION}"
                    )
                renders = request.get("renders")
                if not isinstance(renders, list) or not renders:
                    raise ValueError("/render_batch renders must be a non-empty list")
                if len(renders) > int(args.max_batch_size):
                    raise ValueError(
                        f"batch size {len(renders)} exceeds {args.max_batch_size}"
                    )
                if not all(isinstance(render, dict) for render in renders):
                    raise ValueError("each /render_batch render must be an object")
                total_pixels = 0
                for render in renders:
                    width = int(render.get("width", args.width))
                    height = int(render.get("height", args.height))
                    if width < 1 or height < 1:
                        raise ValueError(
                            "render width and height must be positive"
                        )
                    total_pixels += width * height
                if total_pixels > int(args.max_batch_render_pixels):
                    raise ValueError(
                        f"batch total {total_pixels} pixels exceeds "
                        f"{args.max_batch_render_pixels}"
                    )
                if version == IDEMPOTENT_BATCH_PROTOCOL_VERSION:
                    request_id = validate_request_id(request.get("request_id"))
                    render_future, _created = registry.submit_or_join(
                        request_id,
                        renders,
                        result_bytes_reservation=(
                            _encoded_result_reservation_bytes(
                                renders,
                                default_width=args.width,
                                default_height=args.height,
                            )
                        ),
                    )
                else:
                    render_future = submit_encoded_batch(renders)
                result = render_future.result()
                body = result
                wants_depth = any(
                    bool(render.get("want_depth", False)) for render in renders
                )
                self._send_bytes(
                    200,
                    body,
                    RGBD_BATCH_CONTENT_TYPE if wants_depth else BATCH_CONTENT_TYPE,
                    capabilities=any("camera_settings" in render for render in renders),
                )
            except RenderQueueFull as exc:
                self._send_exception(503, exc)
            except RequestIdConflict as exc:
                self._send_exception(409, exc)
            except (
                BatchProtocolError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                self._send_exception(400, exc)
            except Exception as exc:
                self._send_exception(500, exc)

        def log_message(self, fmt, *args):  # noqa: V105  # BaseHTTPRequestHandler hook
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True  # noqa: V101  # ThreadingMixIn option
    print(
        f"[gsbg] serving http://{args.host}:{args.port} "
        f"owner_thread={owner.thread_ident} capabilities="
        f"{BATCH_CAPABILITY},{RGBD_BATCH_CAPABILITY},"
        f"{IDEMPOTENT_BATCH_CAPABILITY}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        owner.close()
        jpeg_executor.shutdown(wait=True, cancel_futures=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
