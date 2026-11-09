# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import json
import queue
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from unittest import mock

import numpy as np
import pytest

from scene import gs_background_server
from scene.gs_background_client import GSBackgroundClient
from scene.gs_background_server import (
    GpuRenderOwner,
    IdempotentRenderRegistry,
    RenderQueueFull,
    RequestIdConflict,
    _chain_encoded_batch,
    _locked_gpu_renderer,
)
from scene.gs_background_protocol import (
    BATCH_CAPABILITY,
    BATCH_CONTENT_TYPE,
    BATCH_PROTOCOL_VERSION,
    CAPABILITIES_HEADER,
    IDEMPOTENT_BATCH_CAPABILITY,
    IDEMPOTENT_BATCH_PROTOCOL_VERSION,
    RESULT_TTL_HEADER,
    RGBD_BATCH_CAPABILITY,
    RGBD_BATCH_CONTENT_TYPE,
    unpack_jpeg_batch,
    unpack_rgbd_batch,
)


def test_locked_gpu_renderer_owns_lock_during_initialization_and_render():
    events = []
    lock_depth = 0

    @contextmanager
    def render_lock(path):
        nonlocal lock_depth
        assert path == "/tmp/gpu.lock"
        assert lock_depth == 0
        lock_depth = 1
        events.append("lock-enter")
        try:
            yield
        finally:
            events.append("lock-exit")
            lock_depth = 0

    def initialize():
        assert lock_depth == 1
        events.append("initialize")

        def render_many(payloads):
            assert lock_depth == 1
            events.append(("render", payloads))
            return ["done"]

        return render_many

    with mock.patch(
        "scene.gs_background_server._exclusive_render_lock",
        new=render_lock,
    ):
        render_many = _locked_gpu_renderer(initialize, "/tmp/gpu.lock")
        assert lock_depth == 0
        assert render_many([{"camera": 1}]) == ["done"]

    assert events == [
        "lock-enter",
        "initialize",
        "lock-exit",
        "lock-enter",
        ("render", [{"camera": 1}]),
        "lock-exit",
    ]


def test_initializer_and_all_render_jobs_run_on_one_owner_thread():
    call_threads = []

    def initialize():
        call_threads.append(("initialize", threading.get_ident()))

        def render_many(payloads):
            call_threads.append(("render", threading.get_ident()))
            return [
                (np.full((1, 1, 3), payload["value"], dtype=np.uint8), None)
                for payload in payloads
            ]

        return render_many

    owner = GpuRenderOwner(initialize, queue_capacity=16)
    owner.start(timeout=2)
    try:
        with ThreadPoolExecutor(max_workers=8) as callers:
            futures = [
                callers.submit(
                    lambda value=value: owner.submit(
                        [{"value": value}]
                    ).result(timeout=2)[0][0]
                )
                for value in range(8)
            ]
            images = [future.result(timeout=2) for future in futures]
    finally:
        owner.close(timeout=2)

    assert [int(image[0, 0, 0]) for image in images] == list(range(8))
    assert owner.thread_ident is not None
    assert {thread_id for _, thread_id in call_threads} == {owner.thread_ident}
    assert [kind for kind, _ in call_threads].count("initialize") == 1
    assert [kind for kind, _ in call_threads].count("render") == 8


def test_request_exception_does_not_kill_owner_thread():
    def initialize():
        def render_many(payloads):
            if payloads[0].get("fail"):
                raise ValueError("bad camera")
            return [(np.zeros((1, 1, 3), dtype=np.uint8), None)]

        return render_many

    owner = GpuRenderOwner(initialize)
    owner.start(timeout=2)
    try:
        with pytest.raises(ValueError, match="bad camera"):
            owner.submit([{"fail": True}]).result(timeout=2)
        image, depth = owner.submit([{}]).result(timeout=2)[0]
    finally:
        owner.close(timeout=2)

    assert image.shape == (1, 1, 3)
    assert depth is None


def test_fatal_render_exception_resolves_active_and_queued_futures():
    first_started = threading.Event()
    release_first = threading.Event()

    def initialize():
        def render_many(_payloads):
            first_started.set()
            assert release_first.wait(timeout=2)
            raise SystemExit("fatal CUDA owner failure")

        return render_many

    owner = GpuRenderOwner(initialize, queue_capacity=2)
    owner.start(timeout=2)
    active = owner.submit([{}])
    assert first_started.wait(timeout=2)
    queued = owner.submit([{}])
    release_first.set()

    with pytest.raises(SystemExit, match="fatal CUDA owner failure"):
        active.result(timeout=2)
    with pytest.raises(SystemExit, match="fatal CUDA owner failure"):
        queued.result(timeout=2)
    owner.close(timeout=2)
    assert not owner.is_alive


def test_cancelled_future_does_not_kill_owner_thread():
    first_started = threading.Event()
    release_first = threading.Event()

    def initialize():
        def render_many(payloads):
            if payloads[0].get("block"):
                first_started.set()
                assert release_first.wait(timeout=2)
            return [(np.zeros((1, 1, 3), dtype=np.uint8), None)]

        return render_many

    owner = GpuRenderOwner(initialize, queue_capacity=2)
    owner.start(timeout=2)
    active = owner.submit([{"block": True}])
    assert first_started.wait(timeout=2)
    cancelled = owner.submit([{}])
    assert cancelled.cancel()
    release_first.set()

    assert active.result(timeout=2)[0][0].shape == (1, 1, 3)
    with pytest.raises(CancelledError):
        cancelled.result(timeout=2)
    assert owner.submit([{}]).result(timeout=2)[0][0].shape == (1, 1, 3)
    owner.close(timeout=2)
    assert not owner.is_alive


def test_initializer_failure_fails_job_queued_while_start_is_waiting():
    initializer_entered = threading.Event()
    initializer_may_fail = threading.Event()
    start_error = []

    def initialize():
        initializer_entered.set()
        assert initializer_may_fail.wait(timeout=2)
        raise RuntimeError("model load failed")

    owner = GpuRenderOwner(initialize)

    def start_owner():
        try:
            owner.start(timeout=2)
        except BaseException as exc:
            start_error.append(exc)

    starter = threading.Thread(target=start_owner)
    starter.start()
    assert initializer_entered.wait(timeout=2)
    pending = owner.submit([{}])
    initializer_may_fail.set()
    starter.join(timeout=2)

    assert not starter.is_alive()
    assert len(start_error) == 1
    assert isinstance(start_error[0], RuntimeError)
    with pytest.raises(RuntimeError, match="model load failed"):
        pending.result(timeout=2)
    owner.close(timeout=2)
    assert not owner.is_alive


def test_close_uses_reserved_stop_slot_when_public_queue_is_full():
    first_started = threading.Event()
    allow_first_to_finish = threading.Event()

    def initialize():
        def render_many(payloads):
            if payloads[0]["value"] == 1:
                first_started.set()
                assert allow_first_to_finish.wait(timeout=2)
            return [(np.zeros((1, 1, 3), dtype=np.uint8), None)]

        return render_many

    owner = GpuRenderOwner(initialize, queue_capacity=1)
    owner.start(timeout=2)
    first = owner.submit([{"value": 1}])
    assert first_started.wait(timeout=2)
    second = owner.submit([{"value": 2}])
    with pytest.raises(RuntimeError, match="queue is full"):
        owner.submit([{"value": 3}])

    closer = threading.Thread(target=lambda: owner.close(timeout=2))
    closer.start()
    allow_first_to_finish.set()
    closer.join(timeout=2)

    assert not closer.is_alive()
    assert first.result(timeout=2).pop()[0].shape == (1, 1, 3)
    assert second.result(timeout=2).pop()[0].shape == (1, 1, 3)
    assert not owner.is_alive


def test_registry_same_id_and_payload_join_exactly_one_future():
    submitted = []

    def submit(payloads):
        future = Future()
        submitted.append((payloads, future))
        return future

    registry = IdempotentRenderRegistry(submit, capacity=2)
    first, first_created = registry.submit_or_join(
        "request-1", [{"width": 2, "camera": {"x": 1}}]
    )
    second, second_created = registry.submit_or_join(
        "request-1", [{"camera": {"x": 1}, "width": 2}]
    )

    assert first_created is True
    assert second_created is False
    assert first is second
    assert len(submitted) == 1
    assert registry.status("request-1") == ("pending", None)

    first.set_result(b"encoded")
    assert registry.status("request-1") == ("done", None)
    assert registry.acknowledge("request-1") is True
    assert registry.status("request-1") == ("unknown", None)


def test_registry_rejects_request_id_payload_mismatch_without_submission():
    submitted = []

    def submit(payloads):
        submitted.append(payloads)
        return Future()

    registry = IdempotentRenderRegistry(submit)
    registry.submit_or_join("same-id", [{"camera": 1}])

    with pytest.raises(RequestIdConflict, match="different render payload"):
        registry.submit_or_join("same-id", [{"camera": 2}])

    assert len(submitted) == 1


def test_registry_never_evicts_pending_and_expires_terminal_results():
    now = [10.0]
    futures = []

    def submit(_payloads):
        future = Future()
        futures.append(future)
        return future

    registry = IdempotentRenderRegistry(
        submit,
        capacity=1,
        result_ttl_sec=5.0,
        clock=lambda: now[0],
    )
    registry.submit_or_join("pending", [{"camera": 1}])
    now[0] = 100.0
    with pytest.raises(RenderQueueFull, match="cannot be evicted safely"):
        registry.submit_or_join("other", [{"camera": 2}])

    futures[0].set_exception(ValueError("bad camera"))
    assert registry.status("pending") == ("error", "ValueError: bad camera")
    now[0] += 5.0
    assert registry.status("pending") == ("unknown", None)
    _, created = registry.submit_or_join("other", [{"camera": 2}])
    assert created is True


def test_registry_byte_budget_rejects_before_submission_and_releases_on_ack():
    submitted = []

    def submit(payloads):
        future = Future()
        submitted.append((payloads, future))
        return future

    registry = IdempotentRenderRegistry(
        submit,
        capacity=4,
        result_byte_capacity=10,
    )
    first, _ = registry.submit_or_join(
        "first",
        [{"camera": 1}],
        result_bytes_reservation=6,
    )
    with pytest.raises(RenderQueueFull, match="byte budget"):
        registry.submit_or_join(
            "rejected",
            [{"camera": 2}],
            result_bytes_reservation=5,
        )
    assert len(submitted) == 1

    first.set_result(b"jpeg")
    assert registry.acknowledge("first") is True
    _, created = registry.submit_or_join(
        "accepted",
        [{"camera": 2}],
        result_bytes_reservation=5,
    )
    assert created is True
    assert len(submitted) == 2


def test_jpeg_backlog_rejects_before_second_gpu_submission():
    render_started = threading.Event()
    allow_render = threading.Event()
    render_count = []

    def initialize():
        def render_many(payloads):
            render_count.append(len(payloads))
            render_started.set()
            assert allow_render.wait(timeout=2)
            return [(np.zeros((2, 2, 3), dtype=np.uint8), None)]

        return render_many

    owner = GpuRenderOwner(initialize, queue_capacity=4)
    owner.start(timeout=2)
    admission = threading.BoundedSemaphore(1)
    try:
        with ThreadPoolExecutor(max_workers=1) as encoder:
            first = _chain_encoded_batch(
                owner,
                encoder,
                [{"camera": 1}],
                82,
                admission=admission,
            )
            assert render_started.wait(timeout=2)
            with pytest.raises(RenderQueueFull, match="JPEG encode backlog"):
                _chain_encoded_batch(
                    owner,
                    encoder,
                    [{"camera": 2}],
                    82,
                    admission=admission,
                )
            assert render_count == [1]
            allow_render.set()
            assert unpack_jpeg_batch(first.result(timeout=2), expected_count=1)
    finally:
        allow_render.set()
        owner.close(timeout=2)


def test_idempotent_future_retains_encoded_bytes_not_raw_images():
    def initialize():
        return lambda payloads: [
            (np.full((2, 3, 3), payload["value"], dtype=np.uint8), None)
            for payload in payloads
        ]

    owner = GpuRenderOwner(initialize)
    owner.start(timeout=2)
    with ThreadPoolExecutor(max_workers=1) as encoder:
        encoded = _chain_encoded_batch(
            owner,
            encoder,
            [{"value": 10}, {"value": 20}],
            82,
        ).result(timeout=2)
    owner.close(timeout=2)

    assert isinstance(encoded, bytes)
    assert len(unpack_jpeg_batch(encoded, expected_count=2)) == 2


def test_depth_encoder_marks_out_of_range_values_as_missing():
    encoded = gs_background_server._encode_depth_png(
        np.array(
            [[0.0, 0.49, 0.5, 2.0, 5.0, 5.01, np.inf, np.nan]],
            dtype=np.float32,
        ),
        znear=0.5,
        zfar=5.0,
    )
    client = GSBackgroundClient(url="http://127.0.0.1:1")
    decoded = client._decode_depth_png(
        encoded, {"znear": 0.5, "zfar": 5.0}
    )

    assert decoded is not None
    assert np.all(~np.isfinite(decoded[0, [0, 1, 5, 6, 7]]))
    assert np.allclose(decoded[0, [2, 3, 4]], [0.5, 2.0, 5.0], atol=0.01)


def _initialize_http_renderer():
    def render_many(payloads):
        results = []
        for payload in payloads:
            height = int(payload.get("height", 2))
            width = int(payload.get("width", 3))
            rgb = np.full((height, width, 3), 80, dtype=np.uint8)
            depth = None
            if payload.get("want_depth", False):
                depth = np.full((height, width), 1.5, dtype=np.float32)
                depth[0, 0] = 0.0
            results.append((rgb, depth))
        return results

    return render_many


@contextmanager
def _running_server(argv):
    servers = queue.Queue()

    class CapturingServer(ThreadingHTTPServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            servers.put(self)

    result = []
    errors = []

    def run_server():
        try:
            result.append(gs_background_server.main())
        except BaseException as exc:
            errors.append(exc)

    with mock.patch.object(sys, "argv", argv), mock.patch.object(
        gs_background_server,
        "ThreadingHTTPServer",
        CapturingServer,
    ), mock.patch.object(
        gs_background_server,
        "_make_gpu_renderer",
        return_value=_initialize_http_renderer,
    ):
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        server = servers.get(timeout=2)
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            yield base_url
        finally:
            server.shutdown()
            thread.join(timeout=3)

    assert not thread.is_alive()
    assert errors == []
    assert result == [0]


def test_actual_http_handler_redacts_health_and_renders_without_cuda(tmp_path):
    ply = tmp_path / "scene.ply"
    ply.write_bytes(b"fake-ply-for-http-test")
    argv = [
        "gs_background_server.py",
        "--ply",
        str(ply),
        "--port",
        "0",
        "--ready-token",
        "secret-token",
        "--render-result-ttl-sec",
        "12.5",
    ]
    with _running_server(argv) as base_url:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                public_identity = json.load(response)
                capabilities = response.headers[CAPABILITIES_HEADER]
                ttl = response.headers[RESULT_TTL_HEADER]
            assert "ready_token" not in public_identity
            assert BATCH_CAPABILITY in capabilities
            assert RGBD_BATCH_CAPABILITY in capabilities
            assert IDEMPOTENT_BATCH_CAPABILITY in capabilities
            assert ttl == "12.5"

            auth_headers = {"X-Real2Sim-GS-Token": "secret-token"}
            request = urllib.request.Request(
                base_url + "/health",
                headers=auth_headers,
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                private_identity = json.load(response)
            assert private_identity["ready_token"] == "secret-token"
            assert private_identity["ply_path"] == str(ply.resolve())

            envelope = json.dumps(
                {
                    "version": IDEMPOTENT_BATCH_PROTOCOL_VERSION,
                    "request_id": "http-test-1",
                    "renders": [{"width": 3, "height": 2}],
                }
            ).encode()
            render_request = urllib.request.Request(
                base_url + "/render_batch",
                data=envelope,
                headers={
                    **auth_headers,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(render_request, timeout=2) as response:
                assert BATCH_CONTENT_TYPE in response.headers["Content-Type"]
                encoded = response.read()
            assert len(unpack_jpeg_batch(encoded, expected_count=1)) == 1

            wrong_token = urllib.request.Request(
                base_url + "/render_status?request_id=http-test-1",
                headers={"X-Real2Sim-GS-Token": "wrong-token"},
            )
            try:
                urllib.request.urlopen(wrong_token, timeout=2)
            except urllib.error.HTTPError as exc:
                assert exc.code == 409
            else:
                raise AssertionError("wrong token unexpectedly reached render status")


def test_render_batch_want_depth_returns_rgbd_framing_with_decodable_depth(tmp_path):
    ply = tmp_path / "scene.ply"
    ply.write_bytes(b"fake-ply-for-http-depth-test")
    argv = [
        "gs_background_server.py",
        "--ply",
        str(ply),
        "--port",
        "0",
        "--ready-token",
        "secret-token",
        "--render-lock",
        str(tmp_path / "gpu-render.lock"),
    ]
    with _running_server(argv) as base_url:
            auth_headers = {"X-Real2Sim-GS-Token": "secret-token"}
            camera_payload = {
                "width": 3,
                "height": 2,
                "znear": 0.5,
                "zfar": 5.0,
                "want_depth": True,
            }
            envelope = json.dumps(
                {
                    "version": BATCH_PROTOCOL_VERSION,
                    "renders": [camera_payload],
                }
            ).encode()
            render_request = urllib.request.Request(
                base_url + "/render_batch",
                data=envelope,
                headers={
                    **auth_headers,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(render_request, timeout=2) as response:
                assert RGBD_BATCH_CONTENT_TYPE in response.headers["Content-Type"]
                encoded = response.read()
            [(jpeg, depth_png)] = unpack_rgbd_batch(encoded, expected_count=1)
            assert jpeg
            assert depth_png

            client = GSBackgroundClient(url=base_url)
            depth = client._decode_depth_png(depth_png, camera_payload)
            assert depth.shape == (2, 3)
            assert not np.isfinite(depth[0, 0])
            assert np.allclose(depth[0, 1:], 1.5, atol=0.01)
            assert np.allclose(depth[1, :], 1.5, atol=0.01)

            # A legacy request without want_depth still gets the plain JPEG
            # batch framing, matching pre-depth behavior exactly.
            legacy_envelope = json.dumps(
                {
                    "version": BATCH_PROTOCOL_VERSION,
                    "renders": [{"width": 3, "height": 2}],
                }
            ).encode()
            legacy_request = urllib.request.Request(
                base_url + "/render_batch",
                data=legacy_envelope,
                headers={
                    **auth_headers,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(legacy_request, timeout=2) as response:
                assert BATCH_CONTENT_TYPE in response.headers["Content-Type"]
                assert RGBD_BATCH_CONTENT_TYPE not in response.headers["Content-Type"]
                legacy_encoded = response.read()
            assert len(unpack_jpeg_batch(legacy_encoded, expected_count=1)) == 1
    assert (tmp_path / "gpu-render.lock").is_file()


def test_the_client_reports_which_admission_limit_refused_the_render():
    """A 503 must name its reason, not just say Service Unavailable.

    The server has four RenderQueueFull sites, and the 503 body names which
    queue limit fired. The client must keep that body rather than report only
    "HTTPError: HTTP Error 503: Service Unavailable".
    """

    import io
    import urllib.error
    from unittest import mock

    from scene.gs_background_client import GSBackgroundClient

    client = GSBackgroundClient.__new__(GSBackgroundClient)
    client.timeout_sec = 5.0
    client.last_error = None
    client._batch_supported = True
    client._idempotent_batch_supported = True
    client.url = "http://127.0.0.1:11700"

    error = urllib.error.HTTPError(
        "http://127.0.0.1:11700", 503, "Service Unavailable", {},
        io.BytesIO(b"JPEG encode backlog is full"),
    )
    with mock.patch.object(
        client, "_wire_camera_payload", return_value={}
    ), mock.patch("urllib.request.urlopen", side_effect=error):
        result, status = client._render_batch_once([{}], request_id="r1")

    assert status == "retryable"
    assert result is None
    assert "JPEG encode backlog is full" in client.last_error
