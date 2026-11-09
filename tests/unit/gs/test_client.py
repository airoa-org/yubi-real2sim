# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import json
import urllib.error
import unittest
from unittest import mock

import cv2 as cv
import numpy as np

from scene.gs_background_client import (
    GSBackgroundClient,
    apply_srgb_color_transform,
    apply_srgb_gain,
    compose_world_to_gs,
)
from scene.gs_background_protocol import (
    BATCH_CAPABILITY,
    BATCH_CONTENT_TYPE,
    CAPABILITIES_HEADER,
    IDEMPOTENT_BATCH_CAPABILITY,
    IDEMPOTENT_BATCH_PROTOCOL_VERSION,
    RESULT_TTL_HEADER,
    pack_jpeg_batch,
)


class _Response:
    def __init__(self, payload, headers=None):
        self.payload = payload
        self.headers = {} if headers is None else headers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


def _idempotent_health_headers(capability):
    return {
        CAPABILITIES_HEADER: capability,
        RESULT_TTL_HEADER: "300",
    }


class GSBackgroundClientTest(unittest.TestCase):
    def make_client(self, **kwargs):
        # Most legacy unit tests exercise /render directly. Batch-specific
        # tests opt in explicitly; production keeps prefer_batch=True.
        kwargs.setdefault("prefer_batch", False)
        client = GSBackgroundClient("http://127.0.0.1:1", **kwargs)
        client.camera_payload = mock.Mock(return_value={"camera": "fixed"})
        return client

    def test_retries_transient_failure_and_returns_image(self):
        expected_bgr = np.full((3, 4, 3), (10, 20, 30), dtype=np.uint8)
        ok, encoded = cv.imencode(".png", expected_bgr)
        self.assertTrue(ok)
        client = self.make_client(timeout_sec=17.0, max_attempts=3, strict=True)

        with mock.patch(
            "scene.gs_background_client.urllib.request.urlopen",
            side_effect=[urllib.error.URLError("queued"), _Response(encoded.tobytes())],
        ) as urlopen:
            image = client.render(object(), 4, 3)

        np.testing.assert_array_equal(image, cv.cvtColor(expected_bgr, cv.COLOR_BGR2RGB))
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 17.0)
        self.assertEqual(client.fail_count, 1)
        self.assertEqual(client.attempt_count, 2)
        self.assertEqual(client.last_render_attempts, 2)
        self.assertEqual(client.request_count, 2)
        self.assertIsNone(client.last_error)

    def test_render_request_carries_server_identity_token(self):
        expected_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        ok, encoded = cv.imencode(".png", expected_bgr)
        self.assertTrue(ok)
        client = self.make_client(ready_token="launch-token")

        with mock.patch(
            "scene.gs_background_client.urllib.request.urlopen",
            return_value=_Response(encoded.tobytes()),
        ) as urlopen:
            client.render(object(), 2, 2)

        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.get_header("X-real2sim-gs-token"),
            "launch-token",
        )
        health_client = self.make_client(
            ready_token="launch-token",
            prefer_batch=True,
        )

        with mock.patch(
            "scene.gs_background_client.urllib.request.urlopen",
            return_value=_Response(b"{}"),
        ) as urlopen:
            self.assertFalse(health_client.supports_render_batch())

        request = urlopen.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/health"))
        self.assertEqual(
            request.get_header("X-real2sim-gs-token"),
            "launch-token",
        )

    def test_strict_mode_raises_after_attempts_are_exhausted(self):
        client = self.make_client(max_attempts=2, strict=True)

        with mock.patch(
            "scene.gs_background_client.urllib.request.urlopen",
            side_effect=TimeoutError("busy"),
        ) as urlopen:
            with self.assertRaisesRegex(RuntimeError, "failed after 2 attempts"):
                client.render(object(), 4, 3)

        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(client.fail_count, 2)
        self.assertEqual(client.attempt_count, 2)
        self.assertEqual(client.request_count, 2)
        self.assertIn("TimeoutError", client.last_error)

        fallback_client = self.make_client(max_attempts=1, strict=False)

        with mock.patch(
            "scene.gs_background_client.urllib.request.urlopen",
            side_effect=OSError("offline"),
        ):
            self.assertIsNone(fallback_client.render(object(), 4, 3))

    def test_srgb_gain_changes_only_requested_channels(self):
        image = np.array([[[100, 120, 200], [250, 10, 1]]], dtype=np.uint8)

        corrected = apply_srgb_gain(image, [1.10, 1.0, 0.80])

        np.testing.assert_array_equal(
            corrected,
            np.array([[[110, 120, 160], [255, 10, 1]]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            image,
            np.array([[[100, 120, 200], [250, 10, 1]]], dtype=np.uint8),
        )
        with self.assertRaisesRegex(ValueError, "three finite non-negative"):
            apply_srgb_gain(image, [1.0, -0.1, 1.0])

    def test_client_applies_srgb_gain_after_decode(self):
        expected_bgr = np.full((2, 2, 3), (100, 120, 200), dtype=np.uint8)
        ok, encoded = cv.imencode(".png", expected_bgr)
        self.assertTrue(ok)
        client = self.make_client(color_gain_rgb=[1.05, 1.0, 0.75])

        with mock.patch(
            "scene.gs_background_client.urllib.request.urlopen",
            return_value=_Response(encoded.tobytes()),
        ):
            image = client.render(object(), 2, 2)

        np.testing.assert_array_equal(
            image,
            np.full((2, 2, 3), (210, 120, 75), dtype=np.uint8),
        )

    def test_srgb_gamma_then_gain_is_black_preserving(self):
        image = np.array([[[128, 64, 255], [0, 0, 0]]], dtype=np.uint8)

        corrected = apply_srgb_color_transform(
            image,
            gain_rgb=[1.0, 2.0, 0.5],
            gamma_rgb=[2.0, 1.0, 0.5],
        )

        np.testing.assert_array_equal(
            corrected,
            np.array([[[64, 128, 128], [0, 0, 0]]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            image,
            np.array([[[128, 64, 255], [0, 0, 0]]], dtype=np.uint8),
        )

    def test_calibration_world_offset_is_composed_before_world_to_gs(self):
        matrix = np.array(
            [
                [1.5, -0.2, 0.3, 4.0],
                [0.1, 2.0, -0.4, -3.0],
                [-0.5, 0.6, 1.7, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        offset = np.array([0.0, 0.0, 0.8], dtype=np.float64)
        active_world_position = np.array([0.2, -0.7, 0.0, 1.0])

        effective = compose_world_to_gs(matrix, offset)

        expected_calibration_position = active_world_position.copy()
        expected_calibration_position[:3] += offset
        np.testing.assert_allclose(
            effective @ active_world_position,
            matrix @ expected_calibration_position,
        )
        np.testing.assert_allclose(effective[:3, :3], matrix[:3, :3])
        np.testing.assert_allclose(
            effective[:3, 3],
            matrix[:3, 3] + matrix[:3, :3] @ offset,
        )
        with self.assertRaisesRegex(ValueError, "mat_w2gs must be 4x4"):
            compose_world_to_gs(np.eye(3), np.zeros(3))
        with self.assertRaisesRegex(ValueError, "must contain three values"):
            compose_world_to_gs(np.eye(4), np.zeros(2))
        with self.assertRaisesRegex(ValueError, "only finite values"):
            compose_world_to_gs(np.eye(4), np.array([0.0, 0.0, np.nan]))

    def test_batch_endpoint_reduces_requests_but_keeps_logical_counts(self):
        first_bgr = np.full((2, 2, 3), (10, 20, 30), dtype=np.uint8)
        second_bgr = np.full((2, 2, 3), (40, 50, 60), dtype=np.uint8)
        encoded = []
        for image in (first_bgr, second_bgr):
            ok, item = cv.imencode(".jpg", image)
            self.assertTrue(ok)
            encoded.append(item.tobytes())
        health = _Response(
            b"{}",
            {CAPABILITIES_HEADER: BATCH_CAPABILITY},
        )
        batch = _Response(
            pack_jpeg_batch(encoded),
            {"Content-Type": BATCH_CONTENT_TYPE},
        )
        client = self.make_client(
            max_attempts=2, strict=True, prefer_batch=True
        )

        with mock.patch(
            "scene.gs_background_client.urllib.request.urlopen",
            side_effect=[health, batch],
        ) as urlopen:
            images = client.render_payloads([{"camera": 1}, {"camera": 2}])

        self.assertEqual(len(images), 2)
        self.assertEqual(client.attempt_count, 2)
        self.assertEqual(client.request_count, 1)
        self.assertEqual(client.fail_count, 0)
        self.assertTrue(urlopen.call_args_list[0].args[0].full_url.endswith("/health"))
        self.assertTrue(
            urlopen.call_args_list[1].args[0].full_url.endswith("/render_batch")
        )

    def test_old_server_health_falls_back_to_individual_render_requests(self):
        bgr = np.full((2, 2, 3), (10, 20, 30), dtype=np.uint8)
        ok, encoded = cv.imencode(".jpg", bgr)
        self.assertTrue(ok)
        client = self.make_client(
            max_attempts=1, strict=True, prefer_batch=True
        )

        with mock.patch(
            "scene.gs_background_client.urllib.request.urlopen",
            side_effect=[
                _Response(b"{}"),
                _Response(encoded.tobytes()),
                _Response(encoded.tobytes()),
            ],
        ) as urlopen:
            images = client.render_payloads([{"camera": 1}, {"camera": 2}])

        self.assertEqual(len(images), 2)
        self.assertEqual(client.attempt_count, 2)
        self.assertEqual(client.request_count, 2)
        self.assertEqual(client.fail_count, 0)
        self.assertTrue(urlopen.call_args_list[1].args[0].full_url.endswith("/render"))
        self.assertTrue(urlopen.call_args_list[2].args[0].full_url.endswith("/render"))

    def test_health_capability_result_is_cached_across_batches(self):
        bgr = np.full((2, 2, 3), (10, 20, 30), dtype=np.uint8)
        ok, encoded = cv.imencode(".jpg", bgr)
        self.assertTrue(ok)
        batch = pack_jpeg_batch([encoded.tobytes()])
        client = self.make_client(
            max_attempts=1, strict=True, prefer_batch=True
        )

        with mock.patch(
            "scene.gs_background_client.urllib.request.urlopen",
            side_effect=[
                _Response(
                    b"{}", {CAPABILITIES_HEADER: BATCH_CAPABILITY}
                ),
                _Response(batch, {"Content-Type": BATCH_CONTENT_TYPE}),
                _Response(batch, {"Content-Type": BATCH_CONTENT_TYPE}),
            ],
        ) as urlopen:
            client.render_payloads([{"camera": 1}])
            client.render_payloads([{"camera": 2}])

        health_requests = [
            call.args[0]
            for call in urlopen.call_args_list
            if call.args[0].full_url.endswith("/health")
        ]
        self.assertEqual(len(health_requests), 1)
        self.assertEqual(client.attempt_count, 2)
        self.assertEqual(client.request_count, 2)

        fallback_client = self.make_client(
            max_attempts=1, strict=True, prefer_batch=True
        )
        with mock.patch(
            "scene.gs_background_client.urllib.request.urlopen",
            side_effect=[
                _Response(b"{}"),
                _Response(encoded.tobytes()),
                _Response(encoded.tobytes()),
            ],
        ) as urlopen:
            fallback_client.render_payloads([{"camera": 1}])
            fallback_client.render_payloads([{"camera": 2}])

        health_requests = [
            call.args[0]
            for call in urlopen.call_args_list
            if call.args[0].full_url.endswith("/health")
        ]
        self.assertEqual(len(health_requests), 1)
        self.assertEqual(fallback_client.attempt_count, 2)
        self.assertEqual(fallback_client.request_count, 2)

    def test_idempotent_capability_requires_positive_finite_ttl(self):
        client = self.make_client(prefer_batch=True)

        for invalid_ttl in (None, "0", "-1", "nan", "inf"):
            headers = {
                CAPABILITIES_HEADER: IDEMPOTENT_BATCH_CAPABILITY,
            }
            if invalid_ttl is not None:
                headers[RESULT_TTL_HEADER] = invalid_ttl
            client._batch_supported = None
            with mock.patch(
                "scene.gs_background_client.urllib.request.urlopen",
                return_value=_Response(b"{}", headers),
            ):
                self.assertFalse(client.supports_render_batch())
            self.assertFalse(client._idempotent_batch_supported)
            self.assertIsNone(client._idempotent_result_ttl_sec)

    def test_disappearing_batch_endpoint_falls_back_without_double_counting_faces(self):
        bgr = np.full((2, 2, 3), (10, 20, 30), dtype=np.uint8)
        ok, encoded = cv.imencode(".jpg", bgr)
        self.assertTrue(ok)
        client = self.make_client(
            max_attempts=1, strict=True, prefer_batch=True
        )
        unsupported = urllib.error.HTTPError(
            "http://127.0.0.1:1/render_batch",
            404,
            "not found",
            {},
            None,
        )

        with mock.patch(
            "scene.gs_background_client.urllib.request.urlopen",
            side_effect=[
                _Response(
                    b"{}", {CAPABILITIES_HEADER: BATCH_CAPABILITY}
                ),
                unsupported,
                _Response(encoded.tobytes()),
                _Response(encoded.tobytes()),
            ],
        ):
            images = client.render_payloads([{"camera": 1}, {"camera": 2}])

        self.assertEqual(len(images), 2)
        # The 404 proves that no batch raster occurred. Only the two legacy
        # face renders count as logical attempts; all three POSTs remain
        # visible in the transport-level request metric.
        self.assertEqual(client.attempt_count, 2)
        self.assertEqual(client.request_count, 3)
        self.assertEqual(client.fail_count, 0)

    def test_batch_retry_counts_faces_and_transport_requests_separately(self):
        bgr = np.full((2, 2, 3), (10, 20, 30), dtype=np.uint8)
        ok, encoded = cv.imencode(".jpg", bgr)
        self.assertTrue(ok)
        client = self.make_client(
            max_attempts=2, strict=True, prefer_batch=True
        )

        with mock.patch(
            "scene.gs_background_client.urllib.request.urlopen",
            side_effect=[
                _Response(
                    b"{}", {CAPABILITIES_HEADER: BATCH_CAPABILITY}
                ),
                TimeoutError("busy"),
                _Response(
                    pack_jpeg_batch([encoded.tobytes(), encoded.tobytes()]),
                    {"Content-Type": BATCH_CONTENT_TYPE},
                ),
            ],
        ):
            images = client.render_payloads([{"camera": 1}, {"camera": 2}])

        self.assertEqual(len(images), 2)
        self.assertEqual(client.attempt_count, 4)
        self.assertEqual(client.request_count, 2)
        self.assertEqual(client.fail_count, 2)
        self.assertEqual(client.last_render_attempts, 2)

    def test_unaccepted_503_retry_reuses_request_id_and_acks_success(self):
        bgr = np.full((2, 2, 3), (10, 20, 30), dtype=np.uint8)
        ok, encoded = cv.imencode(".jpg", bgr)
        self.assertTrue(ok)
        client = self.make_client(
            max_attempts=2, strict=True, prefer_batch=True
        )
        capability = f"{BATCH_CAPABILITY},{IDEMPOTENT_BATCH_CAPABILITY}"
        unavailable = urllib.error.HTTPError(
            "http://127.0.0.1:1/render_batch",
            503,
            "queue full before admission",
            {},
            None,
        )

        with mock.patch(
            "scene.gs_background_client.urllib.request.urlopen",
            side_effect=[
                _Response(b"{}", _idempotent_health_headers(capability)),
                unavailable,
                _Response(
                    pack_jpeg_batch([encoded.tobytes(), encoded.tobytes()]),
                    {"Content-Type": BATCH_CONTENT_TYPE},
                ),
                _Response(b'{"acknowledged":true}'),
            ],
        ) as urlopen:
            images = client.render_payloads([{"camera": 1}, {"camera": 2}])

        render_requests = [
            call.args[0]
            for call in urlopen.call_args_list
            if call.args[0].full_url.endswith("/render_batch")
        ]
        envelopes = [json.loads(request.data) for request in render_requests]
        self.assertEqual(len(images), 2)
        self.assertEqual(len(render_requests), 2)
        self.assertEqual(
            envelopes[0]["request_id"], envelopes[1]["request_id"]
        )
        self.assertEqual(
            envelopes[0]["version"], IDEMPOTENT_BATCH_PROTOCOL_VERSION
        )
        self.assertEqual(client.attempt_count, 4)
        self.assertEqual(client.fail_count, 2)
        self.assertEqual(client.request_count, 2)
        self.assertEqual(client.control_request_count, 1)

    def test_uncertain_final_timeout_waits_for_done_then_recovers_same_result(self):
        bgr = np.full((2, 2, 3), (10, 20, 30), dtype=np.uint8)
        ok, encoded = cv.imencode(".jpg", bgr)
        self.assertTrue(ok)
        client = self.make_client(
            max_attempts=3, strict=True, prefer_batch=True
        )
        capability = f"{BATCH_CAPABILITY},{IDEMPOTENT_BATCH_CAPABILITY}"

        def status(state):
            return _Response(
                json.dumps(
                    {
                        "version": IDEMPOTENT_BATCH_PROTOCOL_VERSION,
                        "request_id": request_id[0],
                        "status": state,
                    }
                ).encode()
            )

        request_id = [None]

        def urlopen(request, **_kwargs):
            if request.full_url.endswith("/health"):
                return _Response(b"{}", _idempotent_health_headers(capability))
            if request.full_url.endswith("/render_batch"):
                envelope = json.loads(request.data)
                if request_id[0] is None:
                    request_id[0] = envelope["request_id"]
                    raise TimeoutError("handler timed out after submission")
                self.assertEqual(envelope["request_id"], request_id[0])
                return _Response(
                    pack_jpeg_batch([encoded.tobytes()]),
                    {"Content-Type": BATCH_CONTENT_TYPE},
                )
            if "/render_status?" in request.full_url:
                status_calls[0] += 1
                return status("pending" if status_calls[0] == 1 else "done")
            if request.full_url.endswith("/render_ack"):
                return _Response(b'{"acknowledged":true}')
            self.fail(request.full_url)

        status_calls = [0]
        with mock.patch(
            "scene.gs_background_client.urllib.request.urlopen",
            side_effect=urlopen,
        ), mock.patch("scene.gs_background_client.time.sleep"):
            images = client.render_payloads([{"camera": 1}])

        self.assertEqual(len(images), 1)
        self.assertEqual(client.attempt_count, 2)
        self.assertEqual(client.fail_count, 1)
        self.assertEqual(client.request_count, 2)
        self.assertEqual(client.control_request_count, 3)
        self.assertIsNone(client.last_error)

    def test_uncertain_render_status_error_is_a_strict_failure(self):
        client = self.make_client(
            max_attempts=1, strict=True, prefer_batch=True
        )
        capability = f"{BATCH_CAPABILITY},{IDEMPOTENT_BATCH_CAPABILITY}"
        request_id = [None]

        def urlopen(request, **_kwargs):
            if request.full_url.endswith("/health"):
                return _Response(b"{}", _idempotent_health_headers(capability))
            if request.full_url.endswith("/render_batch"):
                request_id[0] = json.loads(request.data)["request_id"]
                raise TimeoutError("response lost")
            if "/render_status?" in request.full_url:
                return _Response(
                    json.dumps(
                        {
                            "version": IDEMPOTENT_BATCH_PROTOCOL_VERSION,
                            "request_id": request_id[0],
                            "status": "error",
                            "error": "RuntimeError: CUDA render failed",
                        }
                    ).encode()
                )
            self.fail(request.full_url)

        with mock.patch(
            "scene.gs_background_client.urllib.request.urlopen",
            side_effect=urlopen,
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA render failed"):
                client.render_payloads([{"camera": 1}])

        self.assertEqual(client.request_count, 1)
        self.assertEqual(client.control_request_count, 1)

    def test_explicit_connection_refusal_is_safe_terminal_failure(self):
        client = self.make_client(
            max_attempts=1, strict=True, prefer_batch=True
        )
        capability = f"{BATCH_CAPABILITY},{IDEMPOTENT_BATCH_CAPABILITY}"

        with mock.patch(
            "scene.gs_background_client.urllib.request.urlopen",
            side_effect=[
                _Response(b"{}", _idempotent_health_headers(capability)),
                TimeoutError("response lost"),
                urllib.error.URLError(ConnectionRefusedError("server dead")),
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "server dead"):
                client.render_payloads([{"camera": 1}])

        self.assertEqual(client.request_count, 1)
        self.assertEqual(client.control_request_count, 1)

    def test_pending_recovery_has_deadline_and_never_resubmits(self):
        capability = f"{BATCH_CAPABILITY},{IDEMPOTENT_BATCH_CAPABILITY}"
        cases = ((True, 3, 0.11), (False, 1, 0.01))
        for strict, max_attempts, timeout in cases:
            with self.subTest(strict=strict):
                client = self.make_client(
                    max_attempts=max_attempts,
                    strict=strict,
                    prefer_batch=True,
                    recovery_timeout_sec=timeout,
                )
                request_id = [None]
                render_calls = [0]

                def urlopen(request, **_kwargs):
                    if request.full_url.endswith("/health"):
                        return _Response(
                            b"{}", _idempotent_health_headers(capability)
                        )
                    if request.full_url.endswith("/render_batch"):
                        render_calls[0] += 1
                        request_id[0] = json.loads(request.data)["request_id"]
                        raise TimeoutError("response lost")
                    if "/render_status?" in request.full_url:
                        return _Response(
                            json.dumps(
                                {
                                    "version": IDEMPOTENT_BATCH_PROTOCOL_VERSION,
                                    "request_id": request_id[0],
                                    "status": "pending",
                                }
                            ).encode()
                        )
                    self.fail(request.full_url)

                clock = [0.0]
                patches = (
                    mock.patch(
                        "scene.gs_background_client.urllib.request.urlopen",
                        side_effect=urlopen,
                    ),
                    mock.patch(
                        "scene.gs_background_client.time.monotonic",
                        side_effect=lambda: clock[0],
                    ),
                    mock.patch(
                        "scene.gs_background_client.time.sleep",
                        side_effect=lambda delay: clock.__setitem__(
                            0, clock[0] + delay
                        ),
                    ),
                )
                with patches[0], patches[1], patches[2]:
                    if strict:
                        with self.assertRaisesRegex(
                            RuntimeError, "recovery timed out"
                        ):
                            client.render_payloads([{"camera": 1}])
                    else:
                        self.assertEqual(
                            client.render_payloads([{"camera": 1}]), [None]
                        )
                        self.assertIn("recovery timed out", client.last_error)

                self.assertEqual(render_calls[0], 1)
                self.assertEqual(client.request_count, 1)
                self.assertGreaterEqual(client.control_request_count, 1)

    def test_recovery_deadline_is_measured_from_request_start_within_ttl(self):
        client = self.make_client(
            max_attempts=3,
            strict=True,
            prefer_batch=True,
            recovery_timeout_sec=20.0,
        )
        capability = f"{BATCH_CAPABILITY},{IDEMPOTENT_BATCH_CAPABILITY}"
        request_id = [None]
        clock = [10.0]

        def urlopen(request, **_kwargs):
            if request.full_url.endswith("/health"):
                return _Response(
                    b"{}",
                    {
                        CAPABILITIES_HEADER: capability,
                        RESULT_TTL_HEADER: "0.1",
                    },
                )
            if request.full_url.endswith("/render_batch"):
                request_id[0] = json.loads(request.data)["request_id"]
                clock[0] = 10.08
                raise TimeoutError("response became uncertain")
            if "/render_status?" in request.full_url:
                return _Response(
                    json.dumps(
                        {
                            "version": IDEMPOTENT_BATCH_PROTOCOL_VERSION,
                            "request_id": request_id[0],
                            "status": "pending",
                        }
                    ).encode()
                )
            self.fail(request.full_url)

        with mock.patch(
            "scene.gs_background_client.urllib.request.urlopen",
            side_effect=urlopen,
        ), mock.patch(
            "scene.gs_background_client.time.monotonic",
            side_effect=lambda: clock[0],
        ), mock.patch(
            "scene.gs_background_client.time.sleep",
            side_effect=lambda delay: clock.__setitem__(0, clock[0] + delay),
        ):
            with self.assertRaisesRegex(RuntimeError, "recovery timed out"):
                client.render_payloads([{"camera": 1}])

        self.assertAlmostEqual(clock[0], 10.1)
        self.assertEqual(client.request_count, 1)

    def test_new_server_routes_single_render_through_idempotent_batch(self):
        bgr = np.full((2, 2, 3), (10, 20, 30), dtype=np.uint8)
        ok, encoded = cv.imencode(".jpg", bgr)
        self.assertTrue(ok)
        client = self.make_client(strict=True, prefer_batch=True)

        with mock.patch(
            "scene.gs_background_client.urllib.request.urlopen",
            side_effect=[
                _Response(
                    b"{}",
                    _idempotent_health_headers(
                        IDEMPOTENT_BATCH_CAPABILITY
                    ),
                ),
                _Response(
                    pack_jpeg_batch([encoded.tobytes()]),
                    {"Content-Type": BATCH_CONTENT_TYPE},
                ),
                _Response(b'{"acknowledged":true}'),
            ],
        ) as urlopen:
            image = client.render(object(), 2, 2)

        self.assertEqual(image.shape, (2, 2, 3))
        render_request = urlopen.call_args_list[1].args[0]
        envelope = json.loads(render_request.data)
        self.assertTrue(render_request.full_url.endswith("/render_batch"))
        self.assertEqual(len(envelope["renders"]), 1)
        self.assertIn("request_id", envelope)
        self.assertEqual(client.request_count, 1)
        self.assertEqual(client.control_request_count, 1)


if __name__ == "__main__":
    unittest.main()
