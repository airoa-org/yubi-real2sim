#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Barrier-batched policy server for multi-simulation rollout.

Run one server per GPU.  Each simulator process connects as a lightweight
client and sends one unbatched observation at every policy chunk boundary.  The
server waits until ``--batch-size`` observations are ready, runs one batched
policy inference, then returns the corresponding action chunk to each client.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import socketserver
import struct
import threading
import time
import traceback
from typing import Any
import uuid

import msgpack
import numpy as np

from . import policy_batch_infer as pbi


_HEADER = struct.Struct(">Q")


def array_sha256(value: Any) -> str:
    """Hash an array including its dtype and shape, not only its raw bytes."""

    arr = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(arr.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


class KeyedNoiseBank:
    """Persist RF initial noise keyed by stable client and inference sequence."""

    def __init__(
        self,
        *,
        seed: int,
        action_horizon: int,
        action_dim: int,
        directory: str | os.PathLike[str] | None = None,
    ) -> None:
        if int(action_horizon) < 1 or int(action_dim) < 1:
            raise ValueError(
                f"positive action shape required, got horizon={action_horizon} dim={action_dim}"
            )
        self.seed = int(seed)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.directory = Path(directory).expanduser().resolve() if directory else None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _identity_digest(self, client_id: str, sequence: int) -> bytes:
        payload = f"rf-noise-v1\0{self.seed}\0{client_id}\0{int(sequence)}".encode("utf-8")
        return hashlib.sha256(payload).digest()

    def _path(self, client_id: str, sequence: int) -> Path | None:
        if self.directory is None:
            return None
        key = self._identity_digest(client_id, sequence).hex()[:20]
        return self.directory / f"seq_{int(sequence):06d}_{key}.npy"

    def _generate(self, client_id: str, sequence: int) -> np.ndarray:
        identity = self._identity_digest(client_id, sequence)
        words = np.frombuffer(identity, dtype=">u4").astype(np.uint32).tolist()
        seed_words = [self.seed & 0xFFFFFFFF, (self.seed >> 32) & 0xFFFFFFFF, *words]
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed_words)))
        return generator.standard_normal(
            (self.action_horizon, self.action_dim), dtype=np.float32
        )

    def get(self, client_id: str, sequence: int) -> tuple[np.ndarray, dict[str, Any]]:
        path = self._path(client_id, sequence)
        with self._lock:
            reused = bool(path is not None and path.is_file())
            if reused:
                noise = np.load(path, allow_pickle=False)
            else:
                noise = self._generate(client_id, sequence)
                if path is not None:
                    temp = path.with_name(
                        f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
                    )
                    with temp.open("wb") as stream:
                        np.save(stream, noise, allow_pickle=False)
                    os.replace(temp, path)
        noise = np.asarray(noise)
        expected = (self.action_horizon, self.action_dim)
        if noise.shape != expected or noise.dtype != np.float32:
            raise ValueError(
                f"invalid noise bank entry {path}: shape={noise.shape} dtype={noise.dtype}; "
                f"expected {expected} float32"
            )
        return noise, {
            "mode": "keyed_noise_bank",
            "format_version": 1,
            "generator": "numpy.PCG64.standard_normal.float32",
            "seed": self.seed,
            "client_id": str(client_id),
            "sequence": int(sequence),
            "shape": [self.action_horizon, self.action_dim],
            "dtype": noise.dtype.str,
            "sha256": array_sha256(noise),
            "path": str(path) if path is not None else None,
            "reused": reused,
        }


def policy_action_shape(
    policy: Any,
    *,
    fallback_action_horizon: int = 0,
) -> tuple[int, int]:
    """Read the loaded model action shape across OpenPI model variants."""

    model = getattr(policy, "_model", None)
    model_config = getattr(model, "config", None)
    action_horizon = getattr(model, "action_horizon", None)
    if action_horizon is None:
        action_horizon = getattr(model_config, "action_horizon", None)
    if action_horizon is None:
        action_horizon = fallback_action_horizon
    action_dim = getattr(model, "action_dim", None)
    if action_dim is None:
        action_dim = getattr(model_config, "action_dim", None)

    horizon = int(action_horizon or 0)
    dim = int(action_dim or 0)
    if horizon < 1 or dim < 1:
        raise ValueError(
            "could not determine a positive policy action shape: "
            f"horizon={horizon} dim={dim}"
        )
    return horizon, dim


def runtime_determinism_metadata() -> dict[str, Any]:
    """Capture the process settings needed to audit deterministic inference."""

    metadata: dict[str, Any] = {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "numpy_version": np.__version__,
    }
    try:
        import jax
        import jaxlib

        metadata.update(
            {
                "jax_version": jax.__version__,
                "jaxlib_version": jaxlib.__version__,
                "jax_default_backend": jax.default_backend(),
                "jax_local_devices": [
                    {
                        "id": int(getattr(device, "id", -1)),
                        "process_index": int(getattr(device, "process_index", -1)),
                        "platform": str(getattr(device, "platform", "unknown")),
                        "device_kind": str(getattr(device, "device_kind", "unknown")),
                    }
                    for device in jax.local_devices()
                ],
            }
        )
    except Exception as exc:  # pragma: no cover - only used by the real policy server.
        metadata["jax_probe_error"] = f"{type(exc).__name__}: {exc}"
    return metadata


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and np.dtype(obj.dtype).kind in ("V", "O", "c"):
        raise ValueError(f"unsupported dtype for msgpack numpy transport: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "data": obj.tobytes(),
            "dtype": obj.dtype.str,
            "shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {"__npgeneric__": True, "data": obj.item(), "dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: dict) -> Any:
    if "__ndarray__" in obj:
        return np.ndarray(buffer=obj["data"], dtype=np.dtype(obj["dtype"]), shape=tuple(obj["shape"]))
    if "__npgeneric__" in obj:
        return np.dtype(obj["dtype"]).type(obj["data"])
    return obj


def pack_message(message: dict) -> bytes:
    return msgpack.packb(message, default=_pack_array, use_bin_type=True)


def unpack_message(payload: bytes) -> dict:
    return msgpack.unpackb(payload, object_hook=_unpack_array, raw=False)


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks = []
    remaining = int(nbytes)
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed while reading packet")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_packet(sock: socket.socket, message: dict) -> None:
    payload = pack_message(message)
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def recv_packet(sock: socket.socket) -> dict:
    header = _recv_exact(sock, _HEADER.size)
    (size,) = _HEADER.unpack(header)
    if size <= 0:
        raise ValueError(f"invalid packet size {size}")
    return unpack_message(_recv_exact(sock, size))


def parse_address(address: str, default_port: int = 8765) -> tuple[str, int]:
    text = str(address).strip()
    for prefix in ("tcp://", "policy://"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if ":" not in text:
        return text or "127.0.0.1", int(default_port)
    host, port = text.rsplit(":", 1)
    return host or "127.0.0.1", int(port)


class BatchPolicyClient:
    """Synchronous client with the same ``infer(obs)`` shape as OpenPI policies."""

    def __init__(
        self,
        address: str,
        *,
        client_id: str | None = None,
        timeout_sec: float | None = None,
    ) -> None:
        self.address = address
        self.client_id = client_id or f"sim-{uuid.uuid4().hex[:8]}"
        self.timeout_sec = timeout_sec
        self._seq = 0
        self._lock = threading.Lock()
        host, port = parse_address(address)
        self._sock = socket.create_connection((host, port), timeout=timeout_sec)
        self._sock.settimeout(timeout_sec)
        self._closed = False
        try:
            hello = recv_packet(self._sock)
            if hello.get("type") != "hello":
                raise RuntimeError(f"unexpected policy batch server hello: {hello}")
            self.metadata = hello.get("metadata", {})
            send_packet(
                self._sock,
                {"type": "register", "client_id": self.client_id},
            )
            registered = recv_packet(self._sock)
            if registered.get("type") == "error":
                raise RuntimeError(
                    registered.get("message", "policy batch server registration error")
                )
            if registered.get("type") != "registered":
                raise RuntimeError(
                    "unexpected policy batch server registration response: "
                    f"{registered}"
                )
        except BaseException:
            self._sock.close()
            self._closed = True
            raise

    def infer(self, obs: dict) -> dict:
        with self._lock:
            self._seq += 1
            seq = self._seq
            send_packet(
                self._sock,
                {
                    "type": "infer",
                    "client_id": self.client_id,
                    "sequence": seq,
                    "obs": obs,
                },
            )
            response = recv_packet(self._sock)
        if response.get("type") == "error":
            raise RuntimeError(response.get("message", "policy batch server error"))
        if response.get("type") != "infer_result":
            raise RuntimeError(f"unexpected policy batch server response: {response}")
        if int(response.get("sequence", -1)) != seq:
            raise RuntimeError(f"policy batch server sequence mismatch: got {response.get('sequence')} expected {seq}")
        result = response["result"]
        result["batch_policy"] = response.get("batch", {})
        return result

    def close(self) -> None:
        with self._lock:
            if getattr(self, "_closed", False):
                return
            try:
                send_packet(
                    self._sock,
                    {"type": "close", "client_id": self.client_id},
                )
                response = recv_packet(self._sock)
                if response.get("type") == "error":
                    raise RuntimeError(
                        response.get("message", "policy batch server close error")
                    )
                if (
                    response.get("type") != "closed"
                    or response.get("client_id") != self.client_id
                ):
                    raise RuntimeError(
                        "unexpected policy batch server close response: "
                        f"{response}"
                    )
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._closed = True


@dataclass
class _Request:
    client_id: str
    sequence: int
    obs: dict | None
    submitted_at: float = field(default_factory=time.monotonic)
    event: threading.Event = field(default_factory=threading.Event)
    result: Any | None = None
    error: BaseException | None = None


class BatchCoordinator:
    def __init__(
        self,
        policy: Any,
        *,
        batch_size: int,
        max_wait_sec: float | None,
        exec_steps: int | None = None,
        client_order: list[str] | None = None,
        noise_bank: KeyedNoiseBank | None = None,
        inference_manifest: str | os.PathLike[str] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.policy = policy
        self.batch_size = int(batch_size)
        self.max_wait_sec = max_wait_sec
        self.exec_steps = exec_steps
        self.client_order = [str(item) for item in (client_order or []) if str(item)]
        self.noise_bank = noise_bank
        self.inference_manifest = (
            Path(inference_manifest).expanduser().resolve() if inference_manifest else None
        )
        if self.inference_manifest is not None:
            self.inference_manifest.parent.mkdir(parents=True, exist_ok=True)
        self._pending: list[_Request] = []
        self._cv = threading.Condition()
        self._closed = False
        self._fatal_error: RuntimeError | None = None
        self._wave_clients: list[str] = []
        self._active_clients: set[str] = set()
        self._retired_clients: set[str] = set()
        self._wave_full = False
        self._batch_index = 0
        self._worker = threading.Thread(target=self._run, name="policy-batch-worker", daemon=True)
        self._worker.start()

    @staticmethod
    def _client_base_id(client_id: str) -> str:
        return re.sub(r"_run_\d+$", "", str(client_id))

    def _sort_batch(self, batch: list[_Request]) -> list[_Request]:
        return sorted(batch, key=lambda req: self._client_sort_key(req.client_id))

    def _client_sort_key(self, client_id: str) -> tuple[int, int, str]:
        if not self.client_order:
            try:
                rank = self._wave_clients.index(str(client_id))
            except ValueError:
                rank = len(self._wave_clients)
            return (0, rank, str(client_id))
        exact_rank = {client_id: idx for idx, client_id in enumerate(self.client_order)}
        base_rank = {self._client_base_id(client_id): idx for idx, client_id in enumerate(self.client_order)}
        client_id = str(client_id)
        if client_id in exact_rank:
            return (0, exact_rank[client_id], client_id)
        base_id = self._client_base_id(client_id)
        if base_id in base_rank:
            return (1, base_rank[base_id], client_id)
        return (2, len(self.client_order), client_id)

    def register_client(self, client_id: str) -> None:
        client_id = str(client_id)
        if not client_id:
            raise ValueError("policy batch client_id must not be empty")
        with self._cv:
            self._register_client_locked(client_id)
            self._cv.notify_all()

    def _register_client_locked(self, client_id: str) -> None:
        if self._closed:
            raise RuntimeError("policy batch coordinator is closed")
        if self._fatal_error is not None:
            raise RuntimeError(str(self._fatal_error)) from self._fatal_error
        if client_id in self._active_clients:
            return
        if client_id in self._retired_clients:
            raise RuntimeError(
                f"policy batch client {client_id!r} already retired from this wave"
            )
        if len(self._wave_clients) >= self.batch_size:
            raise RuntimeError(
                "policy batch wave already has its full client roster "
                f"{self._wave_clients}; cannot register {client_id!r}"
            )
        self._wave_clients.append(client_id)
        self._active_clients.add(client_id)
        if len(self._wave_clients) == self.batch_size:
            if self.client_order:
                self._wave_clients.sort(key=self._client_sort_key)
            self._wave_full = True

    def retire_client(self, client_id: str) -> None:
        client_id = str(client_id)
        with self._cv:
            if self._closed or self._fatal_error is not None:
                return
            if client_id in self._retired_clients:
                return
            if client_id not in self._active_clients:
                raise RuntimeError(
                    f"cannot retire unknown policy batch client {client_id!r}"
                )
            if any(req.client_id == client_id for req in self._pending):
                self._fail_client_locked(
                    client_id,
                    "closed while an inference request was pending",
                )
                return
            self._active_clients.remove(client_id)
            self._retired_clients.add(client_id)
            if self._wave_full and not self._active_clients:
                self._reset_wave_locked()
            self._cv.notify_all()

    def fail_client(self, client_id: str, cause: BaseException | str) -> None:
        with self._cv:
            self._fail_client_locked(str(client_id), cause)

    def _fail_client_locked(
        self, client_id: str, cause: BaseException | str
    ) -> None:
        if self._closed or self._fatal_error is not None:
            return
        detail = str(cause) or type(cause).__name__
        error = RuntimeError(
            f"policy batch client {client_id!r} disconnected without a clean "
            f"close: {detail}"
        )
        self._fatal_error = error
        for req in self._pending:
            req.error = error
            req.event.set()
        self._pending.clear()
        self._cv.notify_all()

    def _reset_wave_locked(self) -> None:
        if self._pending:
            raise RuntimeError("cannot reset a policy batch wave with pending requests")
        self._wave_clients.clear()
        self._active_clients.clear()
        self._retired_clients.clear()
        self._wave_full = False

    def submit(self, client_id: str, sequence: int, obs: dict) -> dict:
        req = _Request(client_id=client_id, sequence=int(sequence), obs=obs)
        result = self._submit_request(req)
        assert isinstance(result, dict)
        return result

    def _submit_request(self, req: _Request) -> Any:
        with self._cv:
            if self._closed:
                raise RuntimeError("policy batch coordinator is closed")
            if self._fatal_error is not None:
                raise RuntimeError(str(self._fatal_error)) from self._fatal_error
            self._register_client_locked(req.client_id)
            if any(pending.client_id == req.client_id for pending in self._pending):
                raise RuntimeError(
                    f"policy batch client {req.client_id!r} already has a pending request"
                )
            self._pending.append(req)
            self._cv.notify_all()
        req.event.wait()
        if req.error is not None:
            raise RuntimeError(str(req.error)) from req.error
        assert req.result is not None
        return req.result

    def close(self) -> None:
        with self._cv:
            self._closed = True
            for req in self._pending:
                req.error = RuntimeError("policy batch coordinator closed")
                req.event.set()
            self._pending.clear()
            self._cv.notify_all()
        self._worker.join(timeout=5.0)

    def _take_batch(self) -> list[_Request] | None:
        with self._cv:
            if self.max_wait_sec is None:
                while (
                    not self._closed
                    and self._fatal_error is None
                    and not self._strict_batch_ready_locked()
                ):
                    self._cv.wait()
                if self._closed or self._fatal_error is not None:
                    return None
                batch = list(self._pending)
                self._pending.clear()
            else:
                while (
                    not self._pending
                    and not self._closed
                    and self._fatal_error is None
                ):
                    self._cv.wait()
                if self._closed or self._fatal_error is not None:
                    return None
                deadline = self._pending[0].submitted_at + self.max_wait_sec
                while (
                    len(self._pending) < self.batch_size
                    and not self._closed
                    and self._fatal_error is None
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cv.wait(timeout=remaining)
                if self._closed or self._fatal_error is not None:
                    return None

                count = min(len(self._pending), self.batch_size)
                batch = self._pending[:count]
                del self._pending[:count]
            self._batch_index += 1
            return batch

    def _strict_batch_ready_locked(self) -> bool:
        if not self._wave_full or not self._active_clients:
            return False
        pending_ids = {req.client_id for req in self._pending}
        return pending_ids == self._active_clients

    def _execution_slots(
        self, batch: list[_Request]
    ) -> list[tuple[_Request, bool]]:
        batch = self._sort_batch(batch)
        if self.max_wait_sec is not None or len(batch) == self.batch_size:
            return [(req, False) for req in batch]
        by_client = {req.client_id: req for req in batch}
        fallback = batch[0]
        slots = []
        for client_id in self._wave_clients:
            req = by_client.get(client_id)
            if req is None:
                if client_id not in self._retired_clients:
                    raise RuntimeError(
                        "strict policy batch is missing a live client request: "
                        f"{client_id!r}"
                    )
                slots.append((fallback, True))
            else:
                slots.append((req, False))
        if len(slots) != self.batch_size:
            raise RuntimeError(
                f"strict policy batch built {len(slots)} slots, expected "
                f"{self.batch_size}"
            )
        return slots

    def _run(self) -> None:
        while True:
            batch = self._take_batch()
            if batch is None:
                return
            batch_index = self._batch_index
            batch = self._sort_batch(batch)
            ids = [req.client_id for req in batch]
            seqs = [req.sequence for req in batch]
            start = time.monotonic()
            try:
                slots = self._execution_slots(batch)
                noise_records: list[dict[str, Any] | None] = [None] * len(slots)
                noise = None
                if self.noise_bank is not None:
                    noise_items = []
                    noise_cache = {}
                    for index, (req, _padding) in enumerate(slots):
                        key = (req.client_id, req.sequence)
                        if key not in noise_cache:
                            noise_cache[key] = self.noise_bank.get(*key)
                        item, noise_records[index] = noise_cache[key]
                        noise_items.append(item)
                    noise = np.stack(noise_items, axis=0)
                results: Any = pbi.infer_many(
                    self.policy,
                    [req.obs for req, _padding in slots],
                    noise=noise,
                )
                if len(results) != len(slots):
                    raise RuntimeError(
                        f"policy returned {len(results)} results for batch size "
                        f"{len(slots)}"
                    )
                infer_ms = (time.monotonic() - start) * 1000.0
                manifest_rows = []
                for index, ((req, padding), result) in enumerate(zip(slots, results)):
                    if padding:
                        continue
                    common_meta = {
                        "batch_index": int(batch_index),
                        "batch_size": int(len(slots)),
                        "requested_batch_size": int(self.batch_size),
                        "client_ids": ids,
                        "sequences": seqs,
                        "server_infer_ms": infer_ms,
                        "noise": noise_records[index],
                    }
                    manifest_row = {
                        "batch_index": int(batch_index),
                        "client_id": str(req.client_id),
                        "sequence": int(req.sequence),
                        "batch_size": int(len(slots)),
                        "client_ids": ids,
                        "sequences": seqs,
                        "noise": noise_records[index],
                    }
                    result = dict(result)
                    action_hash = array_sha256(result["actions"])
                    result["batch_policy"] = {
                        **common_meta,
                        "action_sha256": action_hash,
                    }
                    req.result = result
                    manifest_row["action_sha256"] = action_hash
                    manifest_rows.append(manifest_row)
                if self.inference_manifest is not None:
                    with self.inference_manifest.open("a", encoding="utf-8") as stream:
                        for row in manifest_rows:
                            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            except BaseException as exc:  # noqa: BLE001 - propagate to all waiting clients.
                for req in batch:
                    req.error = exc
            finally:
                for req in batch:
                    req.event.set()


class _ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True  # noqa: V107  # ThreadingMixIn option


class BatchPolicyTCPServer:
    def __init__(
        self,
        policy: Any,
        *,
        host: str,
        port: int,
        batch_size: int,
        max_wait_sec: float | None,
        client_order: list[str] | None = None,
        noise_bank: KeyedNoiseBank | None = None,
        inference_manifest: str | os.PathLike[str] | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.coordinator = BatchCoordinator(
            policy,
            batch_size=batch_size,
            max_wait_sec=max_wait_sec,
            client_order=client_order,
            noise_bank=noise_bank,
            inference_manifest=inference_manifest,
        )
        self.metadata = metadata or {}

        coordinator = self.coordinator
        metadata = self.metadata

        class Handler(socketserver.BaseRequestHandler):
            def setup(self) -> None:  # noqa: V105  # BaseRequestHandler hook
                send_packet(self.request, {"type": "hello", "metadata": metadata})

            def handle(self) -> None:
                client_id = None
                while True:
                    try:
                        message = recv_packet(self.request)
                    except BaseException as exc:  # noqa: BLE001 - transport failure.
                        if client_id is not None:
                            coordinator.fail_client(client_id, exc)
                        return
                    request_type = message.get("type")
                    if request_type == "register":
                        claimed_id = str(message.get("client_id") or "")
                        registered = False
                        try:
                            if not claimed_id:
                                raise RuntimeError(
                                    "policy batch registration requires client_id"
                                )
                            if client_id is not None and claimed_id != client_id:
                                raise RuntimeError(
                                    "policy batch connection cannot change client_id "
                                    f"from {client_id!r} to {claimed_id!r}"
                                )
                            coordinator.register_client(claimed_id)
                            client_id = claimed_id
                            registered = True
                            send_packet(
                                self.request,
                                {"type": "registered", "client_id": client_id},
                            )
                        except BaseException as exc:  # noqa: BLE001
                            if registered:
                                coordinator.fail_client(client_id, exc)
                            try:
                                send_packet(
                                    self.request,
                                    {"type": "error", "message": traceback.format_exc()},
                                )
                            except BaseException:
                                pass
                            return
                        continue
                    if request_type == "close":
                        claimed_id = str(message.get("client_id") or "")
                        if client_id is None or claimed_id != client_id:
                            if client_id is not None:
                                coordinator.fail_client(
                                    client_id,
                                    "connection sent close with a different client_id",
                                )
                            return
                        coordinator.retire_client(client_id)
                        send_packet(
                            self.request,
                            {"type": "closed", "client_id": client_id},
                        )
                        return
                    if request_type != "infer":
                        send_packet(
                            self.request,
                            {"type": "error", "message": f"unknown request {request_type!r}"},
                        )
                        continue
                    try:
                        claimed_id = str(message.get("client_id") or "")
                        if client_id is None:
                            raise RuntimeError(
                                "policy batch client must register before inference"
                            )
                        if claimed_id != client_id:
                            raise RuntimeError(
                                "policy batch inference client_id mismatch: "
                                f"registered {client_id!r}, got {claimed_id!r}"
                            )
                        sequence = int(message.get("sequence", 0))
                        result = coordinator.submit(client_id, sequence, message["obs"])
                        batch_meta = result.pop("batch_policy", {})
                        send_packet(
                            self.request,
                            {
                                "type": "infer_result",
                                "sequence": sequence,
                                "result": result,
                                "batch": batch_meta,
                            },
                        )
                    except BaseException:  # noqa: BLE001
                        try:
                            send_packet(
                                self.request,
                                {"type": "error", "message": traceback.format_exc()},
                            )
                        except BaseException as exc:  # noqa: BLE001
                            if client_id is not None:
                                coordinator.fail_client(client_id, exc)
                            return

        self._server = _ThreadedTCPServer((host, int(port)), Handler)

    @property
    def server_address(self) -> tuple[str, int]:
        host, port = self._server.server_address
        return str(host), int(port)

    def serve_forever(self) -> None:
        try:
            self._server.serve_forever()
        finally:
            self.coordinator.close()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self.coordinator.close()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Start one batched OpenPI policy server for many sim.py workers.")
    ap.add_argument("--policy", required=True, help="OpenPI checkpoint directory.")
    ap.add_argument(
        "--config",
        default=None,
        help="OpenPI config name matching the checkpoint (required with --policy).",
    )
    ap.add_argument("--config-src", choices=["deploy", "training"], default="deploy")
    ap.add_argument("--ah", type=int, default=0, help="override OpenPI model action horizon.")
    ap.add_argument("--instruction", default="Pick up the cup with your right hand and set it on the plate")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--batch-size", type=int, required=True, help="number of sim workers to synchronize per chunk.")
    ap.add_argument(
        "--max-wait-ms",
        type=float,
        default=0.0,
        help="0 = strict barrier until batch-size clients arrive; >0 allows partial batches after this wait.",
    )
    ap.add_argument("--ready-file", default="", help="write JSON metadata here after the server socket is bound.")
    ap.add_argument("--metadata-json", default="", help="extra JSON object exposed to clients in the hello packet.")
    ap.add_argument(
        "--client-order",
        default="",
        help="comma-separated client_id or scene-tag order used to sort each completed batch before inference.",
    )
    ap.add_argument(
        "--noise-seed",
        type=int,
        default=None,
        help="opt-in deterministic RF noise seed keyed by (client_id, inference sequence).",
    )
    ap.add_argument(
        "--noise-bank-dir",
        default="",
        help="persist/reuse deterministic RF noise .npy entries; requires --noise-seed.",
    )
    ap.add_argument(
        "--inference-manifest",
        default="",
        help="append per-client noise and action SHA-256 records as JSONL.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    if not args.config:
        raise SystemExit(2)
    policy = pbi.load_openpi_policy(
        args.policy,
        config=args.config,
        config_src=args.config_src,
        action_horizon=args.ah,
        instruction=args.instruction,
        repo_root=repo_root,
    )
    policy_identifier = os.path.abspath(os.path.expanduser(args.policy))
    extra_meta = json.loads(args.metadata_json) if args.metadata_json else {}
    max_wait_sec = None if float(args.max_wait_ms) <= 0.0 else float(args.max_wait_ms) / 1000.0
    client_order = [item.strip() for item in str(args.client_order or "").split(",") if item.strip()]
    if args.noise_bank_dir and args.noise_seed is None:
        raise SystemExit("--noise-bank-dir requires --noise-seed")
    noise_bank = None
    action_shape = None
    if args.noise_seed is not None:
        action_shape = policy_action_shape(
            policy,
            fallback_action_horizon=args.ah,
        )
        noise_bank = KeyedNoiseBank(
            seed=args.noise_seed,
            action_horizon=action_shape[0],
            action_dim=action_shape[1],
            directory=args.noise_bank_dir or None,
        )
    metadata = {
        "policy": policy_identifier,
        "config": args.config,
        "config_src": args.config_src,
        "ah": int(args.ah),
        "batch_size": int(args.batch_size),
        "max_wait_ms": float(args.max_wait_ms),
        "client_order": client_order,
        "noise_seed": args.noise_seed,
        "noise_action_shape": list(action_shape) if action_shape is not None else None,
        "noise_bank_dir": (
            str(Path(args.noise_bank_dir).expanduser().resolve()) if args.noise_bank_dir else None
        ),
        "inference_manifest": (
            str(Path(args.inference_manifest).expanduser().resolve())
            if args.inference_manifest else None
        ),
        "runtime_determinism": runtime_determinism_metadata(),
        **extra_meta,
        **pbi.policy_load_metadata(policy),
        "sample_num_steps": pbi.policy_sample_num_steps(policy),
    }
    server = BatchPolicyTCPServer(
        policy,
        host=args.host,
        port=args.port,
        batch_size=args.batch_size,
        max_wait_sec=max_wait_sec,
        client_order=client_order,
        noise_bank=noise_bank,
        inference_manifest=args.inference_manifest or None,
        metadata=metadata,
    )
    host, port = server.server_address
    ready = {"host": host, "port": port, "address": f"{host}:{port}", "metadata": metadata, "pid": os.getpid()}
    if args.ready_file:
        ready_path = Path(args.ready_file)
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        ready_path.write_text(json.dumps(ready, indent=2) + "\n")
    print(f"[policy-batch-server] listening on {host}:{port} batch_size={args.batch_size}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
