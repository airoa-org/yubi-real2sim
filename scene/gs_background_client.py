#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Client-side 3DGS background compositing for MuJoCo renders.

The 3DGS itself is rendered by scene/gs_background_server.py in the
GaussianWrapping environment. This module keeps MuJoCo in charge of physics and
foreground rendering, then replaces only the no-geometry pixels with the 3DGS
image rendered from the same camera.
"""
from __future__ import annotations

import json
import errno
import math
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

from scene.gs_background_protocol import (
    BATCH_CAPABILITY,
    BATCH_CONTENT_TYPE,
    BATCH_PROTOCOL_VERSION,
    CAPABILITIES_HEADER,
    BatchProtocolError,
    IDEMPOTENT_BATCH_CAPABILITY,
    IDEMPOTENT_BATCH_PROTOCOL_VERSION,
    KB4_CAPABILITY,
    DEFAULT_KB4_NEAR_CULL,
    KB4_NEAR_CAPABILITY,
    KB4_NEAR_CULL_MODES,
    LEGACY_KB4_NEAR_GS_UNITS,
    RGBD_BATCH_CONTENT_TYPE,
    RESULT_TTL_HEADER,
    unpack_jpeg_batch,
    unpack_rgbd_batch,
    kb4_valid_mask,
    validate_camera_settings,
    validate_kb4_near,
)


_DEPTH_SCALE_TO_METERS_KEY = "_real2sim_depth_scale_to_meters"
_DEPTH_QUANTIZATION_LEVELS = 65534.0
_DEFAULT_DEPTH_EPSILON_M = 0.002


def _uniform_linear_scale(matrix: np.ndarray) -> float:
    """Return the uniform world-to-GS scale of a similarity transform."""

    linear = np.asarray(matrix, dtype=np.float64)[:3, :3]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    scale = float(np.mean(singular_values))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("mat_w2gs must have a finite positive scale")
    if not np.allclose(singular_values, scale, rtol=1e-5, atol=1e-8):
        raise ValueError(
            "mat_w2gs must use a uniform scale for metric depth conversion"
        )
    return scale


def apply_srgb_gain(
    image: np.ndarray,
    gain_rgb: np.ndarray | tuple[float, float, float] | list[float] | None,
) -> np.ndarray:
    """Apply a black-preserving per-channel gain to an sRGB uint8 image."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError(
            "GS color input must be a uint8 RGB image with shape [H,W,3]"
        )
    if gain_rgb is None:
        return array.copy()
    gain = np.asarray(gain_rgb, dtype=np.float64)
    if (
        gain.shape != (3,)
        or not np.all(np.isfinite(gain))
        or np.any(gain < 0.0)
    ):
        raise ValueError(
            "GS color gain must contain three finite non-negative RGB values"
        )
    if np.array_equal(gain, np.ones(3, dtype=np.float64)):
        return array.copy()
    corrected = array.astype(np.float32) * gain.astype(np.float32)[None, None, :]
    return np.rint(np.clip(corrected, 0.0, 255.0)).astype(np.uint8)


def apply_srgb_color_transform(
    image: np.ndarray,
    gain_rgb: np.ndarray | tuple[float, float, float] | list[float] | None,
    gamma_rgb: np.ndarray | tuple[float, float, float] | list[float] | None,
) -> np.ndarray:
    """Apply a black-preserving GS-only sRGB gamma curve followed by gain."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError(
            "GS color input must be a uint8 RGB image with shape [H,W,3]"
        )
    gamma = (
        np.ones(3, dtype=np.float64)
        if gamma_rgb is None
        else np.asarray(gamma_rgb, dtype=np.float64)
    )
    if (
        gamma.shape != (3,)
        or not np.all(np.isfinite(gamma))
        or np.any(gamma <= 0.0)
    ):
        raise ValueError(
            "GS color gamma must contain three finite positive RGB values"
        )
    if np.array_equal(gamma, np.ones(3, dtype=np.float64)):
        return apply_srgb_gain(array, gain_rgb)
    gain = (
        np.ones(3, dtype=np.float64)
        if gain_rgb is None
        else np.asarray(gain_rgb, dtype=np.float64)
    )
    if (
        gain.shape != (3,)
        or not np.all(np.isfinite(gain))
        or np.any(gain < 0.0)
    ):
        raise ValueError(
            "GS color gain must contain three finite non-negative RGB values"
        )
    normalized = array.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma.astype(np.float32)[None, None, :])
    corrected *= gain.astype(np.float32)[None, None, :]
    return np.rint(np.clip(corrected, 0.0, 1.0) * 255.0).astype(np.uint8)


def compose_world_to_gs(
    mat_w2gs: np.ndarray, calibration_world_offset: np.ndarray
) -> np.ndarray:
    """Convert a calibrated world->GS matrix to the active simulator world.

    ``calibration_world_offset`` is added to positions in the active MuJoCo
    world before applying ``mat_w2gs``.  Equivalently, this returns
    ``mat_w2gs @ translation(calibration_world_offset)``.  Keeping the
    composition here makes the sign and multiplication order testable; it is
    especially important when the alignment matrix contains rotation/scale.
    """

    matrix = np.asarray(mat_w2gs, dtype=np.float64)
    offset = np.asarray(calibration_world_offset, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"mat_w2gs must be 4x4, got {matrix.shape}")
    if offset.shape != (3,):
        raise ValueError(
            "calibration_world_offset must contain three values, "
            f"got {offset.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("mat_w2gs must contain only finite values")
    if not np.all(np.isfinite(offset)):
        raise ValueError("calibration_world_offset must contain only finite values")
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = offset
    return matrix @ translation


@dataclass(frozen=True)
class GSRenderResult:
    """One GS render, with optional depth in MuJoCo meters."""

    rgb: np.ndarray | None
    depth: np.ndarray | None
    depth_convention: str = "optical_axis"


@dataclass
class GSBackgroundClient:
    url: str
    ready_token: str = ""
    fovy_deg: float = 45.0
    timeout_sec: float = 2.0
    znear: float = 0.01
    zfar: float = 100.0
    # KB4 near cull (gs_background_protocol.KB4_NEAR_CULL_MODES): "znear" drops only Gaussians closer
    # than znear metres; "legacy" reproduces runs rendered before the cull was configurable.
    kb4_near_cull: str = DEFAULT_KB4_NEAR_CULL
    fail_count: int = 0
    rot_R: np.ndarray | None = None     # 3x3: rotate the GS frame relative to the MuJoCo scene (axis-alignment fix)
    rot_piv: np.ndarray | None = None   # pivot (world m) the rotation is applied about (default origin)
    mat_w2gs: np.ndarray | None = None  # 4x4: world(MuJoCo) -> GS-native transform = inv(GS object's world matrix
                                        # from the hand-alignment blend). Takes precedence over rot_R when set.
    world_offset: np.ndarray | None = None  # rigid world translation T of the whole scene. Subtracted from the camera
                                            # world pos so the GS backdrop shifts by +T and stays glued to the
                                            # offset desk/robots (covers mat_w2gs, rot_R, and identity paths).
    max_attempts: int = 1
    strict: bool = False
    retry_delay_sec: float = 0.0
    recovery_timeout_sec: float = 300.0
    attempt_count: int = 0
    request_count: int = 0
    control_request_count: int = 0
    last_render_attempts: int = 0
    last_error: str | None = None
    prefer_batch: bool = True
    color_gain_rgb: (
        np.ndarray | tuple[float, float, float] | list[float] | None
    ) = None
    color_gamma_rgb: (
        np.ndarray | tuple[float, float, float] | list[float] | None
    ) = None
    _batch_supported: bool | None = field(default=None, init=False, repr=False)
    _idempotent_batch_supported: bool = field(
        default=False, init=False, repr=False
    )
    _idempotent_result_ttl_sec: float | None = field(
        default=None, init=False, repr=False
    )
    _capabilities: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)

    def camera_payload(self, scene, width: int, height: int, fovy_deg: float | None = None, *, camera_settings: dict | None = None) -> dict:
        # Keep the HTTP/retry client importable in the small release CI
        # environment. MuJoCo is required only when converting a live scene
        # camera, not for protocol negotiation, recovery, or JPEG decoding.
        import mujoco

        pos = np.zeros(3, dtype=np.float64)
        forward = np.zeros(3, dtype=np.float64)
        up = np.zeros(3, dtype=np.float64)
        mujoco.mjv_cameraInModel(pos, forward, up, scene)
        if self.world_offset is not None:   # shift GS backdrop by +T: render as if the camera were at pos-T
            pos = pos - np.asarray(self.world_offset, float)
        depth_scale_to_meters = 1.0
        if self.mat_w2gs is not None:   # full SE3(+scale) world->GS-native (from hand alignment in Blender)
            M = np.asarray(self.mat_w2gs, float)
            if M.shape != (4, 4) or not np.all(np.isfinite(M)):
                raise ValueError("mat_w2gs must be a finite 4x4 matrix")
            world_to_gs_scale = _uniform_linear_scale(M)
            depth_scale_to_meters = 1.0 / world_to_gs_scale
            pos = M[:3, :3] @ pos + M[:3, 3]
            R = M[:3, :3]
            forward = R @ forward; forward /= np.linalg.norm(forward) + 1e-12
            up = R @ up; up /= np.linalg.norm(up) + 1e-12
        elif self.rot_R is not None:   # render GS as if its frame were rotated by rot_R about rot_piv
            piv = np.zeros(3) if self.rot_piv is None else np.asarray(self.rot_piv, float)
            pos = self.rot_R @ (pos - piv) + piv
            forward = self.rot_R @ forward
            up = self.rot_R @ up
        fovy = self.fovy_deg if fovy_deg is None else fovy_deg
        payload = {
            "pos": [float(x) for x in pos],
            "forward": [float(x) for x in forward],
            "up": [float(x) for x in up],
            "width": int(width),
            "height": int(height),
            "fovy_deg": float(fovy),
            # The GS rasterizer and its depth encoding operate in GS-native
            # distance units. Keep the inverse scale as client-only metadata
            # so decoded depth is returned in MuJoCo meters.
            "znear": float(self.znear / depth_scale_to_meters),
            "zfar": float(self.zfar / depth_scale_to_meters),
            _DEPTH_SCALE_TO_METERS_KEY: depth_scale_to_meters,
        }
        if camera_settings is not None:
            payload["camera_settings"] = validate_camera_settings(camera_settings, width, height)
            payload["kb4_near"] = self.kb4_near_gs_units(depth_scale_to_meters)
        return payload

    def kb4_near_gs_units(self, depth_scale_to_meters: float) -> float:
        """The KB4 near cull distance this client requests, in GS units."""

        if self.kb4_near_cull == "legacy":
            return LEGACY_KB4_NEAR_GS_UNITS
        if self.kb4_near_cull != "znear":
            raise ValueError(f"kb4_near_cull must be one of {KB4_NEAR_CULL_MODES}, got {self.kb4_near_cull!r}")
        return validate_kb4_near(float(self.znear) / float(depth_scale_to_meters))

    def kb4_near_meters(self) -> float:
        """The KB4 near cull distance in MuJoCo metres for the current world-to-GS transform."""

        scale = 1.0 if self.mat_w2gs is None else _uniform_linear_scale(np.asarray(self.mat_w2gs, float))
        return self.kb4_near_gs_units(1.0 / scale) / scale

    def _request_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            **(
                {"X-Real2Sim-GS-Token": self.ready_token}
                if self.ready_token
                else {}
            ),
        }

    @staticmethod
    def _response_header(response, name: str) -> str:
        headers = getattr(response, "headers", None)
        if headers is not None and hasattr(headers, "get"):
            value = headers.get(name, "")
            return "" if value is None else str(value)
        getheader = getattr(response, "getheader", None)
        if callable(getheader):
            value = getheader(name, "")
            return "" if value is None else str(value)
        return ""

    def _refresh_capabilities(self) -> bool:
        request = urllib.request.Request(
            self.url.rstrip("/") + "/health",
            headers=self._request_headers(),
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=float(self.timeout_sec)
            ) as response:
                # Consume the response so the connection can be reused.  The
                # identity JSON remains unchanged for old launchers.
                response.read()
                advertised = self._response_header(
                    response, CAPABILITIES_HEADER
                )
                advertised_ttl = self._response_header(
                    response, RESULT_TTL_HEADER
                )
        except (urllib.error.URLError, TimeoutError, OSError):
            # Do not cache a transient health failure.  The legacy /render
            # fallback below still reports the actual rendering failure.
            self._capabilities = frozenset()
            return False
        capabilities = {
            capability.strip()
            for capability in advertised.split(",")
            if capability.strip()
        }
        self._capabilities = frozenset(capabilities)
        try:
            result_ttl_sec = float(advertised_ttl)
        except (TypeError, ValueError):
            result_ttl_sec = math.nan
        valid_result_ttl = (
            math.isfinite(result_ttl_sec) and result_ttl_sec > 0.0
        )
        self._idempotent_batch_supported = (
            IDEMPOTENT_BATCH_CAPABILITY in capabilities
            and valid_result_ttl
        )
        self._idempotent_result_ttl_sec = (
            result_ttl_sec if self._idempotent_batch_supported else None
        )
        self._batch_supported = (
            BATCH_CAPABILITY in capabilities
            or self._idempotent_batch_supported
        )
        return True

    def supports_render_batch(self) -> bool:
        """Detect the v1 endpoint once from the backward-compatible health API."""

        if not self.prefer_batch:
            return False
        if self._batch_supported is None and not self._refresh_capabilities():
            return False
        return bool(self._batch_supported)

    def require_camera_model(self, model: str) -> None:
        """Fail before capture if the server cannot supply radial KB4 RGBD."""

        if model != "kb4":
            raise ValueError(f"unsupported required GS camera model: {model!r}")
        refreshed = self._refresh_capabilities()
        for capability in (KB4_CAPABILITY, KB4_NEAR_CAPABILITY):
            if not refreshed or capability not in self._capabilities:
                raise RuntimeError(f"3DGS server at {self.url} requires capability {capability}")

    def _require_response_capability(self, response, payloads) -> None:
        if any("camera_settings" in payload for payload in payloads):
            capabilities = self._response_header(response, CAPABILITIES_HEADER).split(",")
            if KB4_CAPABILITY not in {value.strip() for value in capabilities}:
                raise RuntimeError("3DGS response does not confirm KB4 radial depth support")
            if KB4_NEAR_CAPABILITY not in {value.strip() for value in capabilities}:
                raise RuntimeError("3DGS response does not confirm the requested KB4 near cull")

    def _decode_jpeg(self, encoded: bytes) -> np.ndarray | None:
        data = np.frombuffer(encoded, dtype=np.uint8)
        bgr = cv.imdecode(data, cv.IMREAD_COLOR)
        if bgr is None:
            self.last_error = "OpenCV could not decode the 3DGS response"
            return None
        self.last_error = None
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        return apply_srgb_color_transform(
            rgb, self.color_gain_rgb, self.color_gamma_rgb
        )

    @staticmethod
    def _wire_camera_payload(camera_payload: dict, *, want_depth: bool) -> dict:
        """Copy a camera payload and remove client-only depth metadata."""

        wire_payload = dict(camera_payload)
        if "camera_settings" in camera_payload:
            wire_payload["camera_settings"] = validate_camera_settings(
                camera_payload["camera_settings"], camera_payload["width"], camera_payload["height"]
            )
            if "kb4_near" not in camera_payload:
                raise ValueError("a KB4 camera payload must carry kb4_near (GS units); build it with camera_payload()")
            wire_payload["kb4_near"] = validate_kb4_near(camera_payload["kb4_near"])
        wire_payload.pop(_DEPTH_SCALE_TO_METERS_KEY, None)
        if want_depth:
            wire_payload["want_depth"] = True
        else:
            wire_payload.pop("want_depth", None)
        return wire_payload

    def _decode_depth_png(
        self, encoded: bytes, camera_payload: dict
    ) -> np.ndarray | None:
        data = np.frombuffer(encoded, dtype=np.uint8)
        quantized = cv.imdecode(data, cv.IMREAD_UNCHANGED)
        if (
            quantized is None
            or quantized.ndim != 2
            or quantized.dtype != np.uint16
        ):
            shape = None if quantized is None else quantized.shape
            dtype = None if quantized is None else quantized.dtype
            self.last_error = (
                "3DGS depth response must be a 16-bit single-channel PNG; "
                f"got shape={shape}, dtype={dtype}"
            )
            return None
        try:
            znear = float(camera_payload["znear"])
            zfar = float(camera_payload["zfar"])
            scale_to_meters = float(
                camera_payload.get(_DEPTH_SCALE_TO_METERS_KEY, 1.0)
            )
        except (KeyError, TypeError, ValueError) as exc:
            self.last_error = f"invalid camera depth metadata: {exc}"
            return None
        if (
            not math.isfinite(znear)
            or not math.isfinite(zfar)
            or not math.isfinite(scale_to_meters)
            or znear <= 0.0
            or zfar <= znear
            or scale_to_meters <= 0.0
        ):
            self.last_error = (
                "camera depth metadata must satisfy 0 < znear < zfar and "
                "depth scale must be finite and positive"
            )
            return None

        depth_native = np.full(quantized.shape, np.inf, dtype=np.float32)
        valid = quantized != 0
        if np.any(valid):
            unit = (quantized[valid].astype(np.float64) - 1.0) / (
                _DEPTH_QUANTIZATION_LEVELS
            )
            decoded = znear + unit * (zfar - znear)
            depth_native[valid] = decoded.astype(np.float32)
        depth_meters = depth_native * np.float32(scale_to_meters)
        self.last_error = None
        return depth_meters

    def _decode_render_result(
        self,
        jpeg: bytes,
        depth_png: bytes | None,
        camera_payload: dict,
    ) -> GSRenderResult | None:
        rgb = self._decode_jpeg(jpeg)
        if rgb is None:
            return None
        expected_shape = None
        if "height" in camera_payload or "width" in camera_payload:
            try:
                expected_shape = (
                    int(camera_payload["height"]),
                    int(camera_payload["width"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                self.last_error = f"invalid camera raster dimensions: {exc}"
                return None
            if expected_shape[0] <= 0 or expected_shape[1] <= 0:
                self.last_error = (
                    "camera raster dimensions must be positive, got "
                    f"{expected_shape}"
                )
                return None
            if rgb.shape[:2] != expected_shape:
                self.last_error = (
                    "3DGS RGB response shape does not match its camera request: "
                    f"expected={expected_shape}, got={rgb.shape[:2]}"
                )
                return None
        settings = camera_payload.get("camera_settings")
        if settings is not None:
            valid = kb4_valid_mask(camera_payload["width"], camera_payload["height"], settings)
            rgb[~valid] = 0
            if depth_png is None:
                raise RuntimeError("KB4 response is missing radial depth")
        if depth_png is None:
            return GSRenderResult(rgb=rgb, depth=None)
        depth = self._decode_depth_png(depth_png, camera_payload)
        if depth is None:
            return None
        if depth.shape != rgb.shape[:2]:
            self.last_error = (
                "3DGS depth response shape does not match RGB: "
                f"rgb={rgb.shape[:2]}, depth={depth.shape}"
            )
            return None
        if settings is not None:
            depth[~valid] = np.nan
        return GSRenderResult(rgb=rgb, depth=depth, depth_convention="radial" if settings is not None else "optical_axis")

    def _render_payload_once(
        self, camera_payload: dict, *, want_depth: bool = False
    ) -> GSRenderResult | None:
        payload = json.dumps(
            self._wire_camera_payload(camera_payload, want_depth=want_depth)
        ).encode("utf-8")
        req = urllib.request.Request(
            self.url.rstrip("/") + "/render",
            data=payload,
            headers=self._request_headers(),
            method="POST",
        )
        self.request_count += 1
        try:
            with urllib.request.urlopen(req, timeout=float(self.timeout_sec)) as resp:
                self._require_response_capability(resp, [camera_payload])
                content_type = self._response_header(resp, "Content-Type")
                encoded = resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.fail_count += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None
        if want_depth and RGBD_BATCH_CONTENT_TYPE in content_type:
            try:
                [(jpeg, depth_png)] = unpack_rgbd_batch(
                    encoded, expected_count=1
                )
            except (BatchProtocolError, ValueError) as exc:
                self.last_error = f"BatchProtocolError: {exc}"
                self.fail_count += 1
                return None
            result = self._decode_render_result(
                jpeg, depth_png, camera_payload
            )
        else:
            if "camera_settings" in camera_payload:
                raise RuntimeError("KB4 response requires RGBD framing")
            # Old servers ignore want_depth and return their legacy JPEG.
            image = self._decode_jpeg(encoded)
            result = (
                None
                if image is None
                else GSRenderResult(rgb=image, depth=None)
            )
        if result is None:
            self.fail_count += 1
        return result

    def _render_payload_with_retries(
        self, payload: dict, *, want_depth: bool = False
    ) -> GSRenderResult | None:
        attempts = max(1, int(self.max_attempts))
        self.last_render_attempts = 0
        for attempt in range(1, attempts + 1):
            self.attempt_count += 1
            self.last_render_attempts = attempt
            result = self._render_payload_once(payload, want_depth=want_depth)
            if result is not None:
                return result
            if attempt < attempts and float(self.retry_delay_sec) > 0.0:
                time.sleep(float(self.retry_delay_sec))
        if self.strict or "camera_settings" in payload:
            raise RuntimeError(
                f"3DGS render failed after {attempts} attempts at {self.url}: {self.last_error}"
            )
        return None

    def _render_batch_once(
        self,
        payloads: Sequence[dict],
        *,
        request_id: str | None,
        want_depth: bool = False,
        timeout_sec: float | None = None,
    ) -> tuple[list[GSRenderResult] | None, str | None]:
        version = (
            IDEMPOTENT_BATCH_PROTOCOL_VERSION
            if request_id is not None
            else BATCH_PROTOCOL_VERSION
        )
        envelope = {
            "version": version,
            "renders": [
                self._wire_camera_payload(payload, want_depth=want_depth)
                for payload in payloads
            ],
        }
        if request_id is not None:
            envelope["request_id"] = request_id
        body = json.dumps(
            envelope,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url.rstrip("/") + "/render_batch",
            data=body,
            headers=self._request_headers(),
            method="POST",
        )
        self.request_count += 1
        try:
            request_timeout = (
                float(self.timeout_sec)
                if timeout_sec is None
                else min(float(self.timeout_sec), float(timeout_sec))
            )
            with urllib.request.urlopen(
                request, timeout=max(request_timeout, 0.001)
            ) as response:
                self._require_response_capability(response, payloads)
                content_type = self._response_header(response, "Content-Type")
                encoded = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 405):
                self._batch_supported = False
                self._idempotent_batch_supported = False
                self.last_error = None
                return None, "unsupported"
            # The server puts the reason in the body -- which of the four
            # admission limits refused the request, and its numbers. Keep it
            # in the error instead of a bare "HTTP Error 503".
            detail = ""
            try:
                body = exc.read()
            except Exception:                       # noqa: BLE001 - diagnostics only
                body = b""
            if body:
                detail = body.decode("utf-8", "replace").strip()
                if len(detail) > 300:
                    detail = detail[:300] + "..."
            self.last_error = (
                f"{type(exc).__name__}: {exc}" + (f" -- {detail}" if detail else "")
            )
            if exc.code == 503:
                # The server contract emits 503 only when admission failed
                # before any GPU or JPEG work was accepted. Retrying the same
                # id is therefore safe and cannot duplicate a render.
                return None, "retryable"
            return None, "terminal"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            if self._is_connection_refused(exc):
                return None, "dead"
            return None, "uncertain"
        is_rgbd = RGBD_BATCH_CONTENT_TYPE in content_type
        is_legacy = not content_type or BATCH_CONTENT_TYPE in content_type
        if not is_rgbd and not is_legacy:
            self.last_error = (
                "3DGS batch response has unexpected Content-Type "
                f"{content_type!r}"
            )
            return None, "invalid"
        try:
            if is_rgbd:
                encoded_items = unpack_rgbd_batch(
                    encoded, expected_count=len(payloads)
                )
            else:
                # A stale batch server ignores want_depth and keeps returning
                # the v1 JPEG framing. Preserve a clean depth=None fallback.
                encoded_items = [
                    (jpeg, None)
                    for jpeg in unpack_jpeg_batch(
                        encoded, expected_count=len(payloads)
                    )
                ]
        except BatchProtocolError as exc:
            self.last_error = f"BatchProtocolError: {exc}"
            return None, "invalid"
        results: list[GSRenderResult] = []
        for camera_payload, (jpeg, depth_png) in zip(payloads, encoded_items):
            result = self._decode_render_result(
                jpeg, depth_png, camera_payload
            )
            if result is None:
                return None, "invalid"
            results.append(result)
        return results, None

    @staticmethod
    def _is_connection_refused(exc: BaseException) -> bool:
        """Recognize only an explicit refusal; other transport errors are unsafe."""

        seen: set[int] = set()
        current: object | None = exc
        while isinstance(current, BaseException) and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, ConnectionRefusedError):
                return True
            if getattr(current, "errno", None) == errno.ECONNREFUSED:
                return True
            reason = getattr(current, "reason", None)
            if isinstance(reason, BaseException):
                current = reason
                continue
            current = current.__cause__ or current.__context__
        return False

    def _render_status_once(
        self,
        request_id: str,
        *,
        timeout_sec: float | None = None,
    ) -> tuple[str | None, str | None]:
        query = urllib.parse.urlencode({"request_id": request_id})
        request = urllib.request.Request(
            self.url.rstrip("/") + "/render_status?" + query,
            headers=self._request_headers(),
            method="GET",
        )
        self.control_request_count += 1
        request_timeout = (
            float(self.timeout_sec)
            if timeout_sec is None
            else min(float(self.timeout_sec), float(timeout_sec))
        )
        try:
            with urllib.request.urlopen(
                request, timeout=max(request_timeout, 0.001)
            ) as response:
                response_body = response.read()
        except urllib.error.HTTPError as exc:
            # A live server that does not recognize the status API, or whose
            # identity token differs, cannot be the still-running v2 owner.
            if exc.code in (404, 405, 409):
                return "unknown", f"HTTPError: {exc}"
            return None, f"HTTPError: {exc}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if self._is_connection_refused(exc):
                return "dead", f"{type(exc).__name__}: {exc}"
            return None, f"{type(exc).__name__}: {exc}"
        try:
            value = json.loads(response_body.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("status body must be an object")
            if value.get("version") != IDEMPOTENT_BATCH_PROTOCOL_VERSION:
                raise ValueError("status protocol version mismatch")
            if value.get("request_id") != request_id:
                raise ValueError("status request_id mismatch")
            status = value.get("status")
            if status not in ("pending", "done", "error", "unknown"):
                raise ValueError(f"invalid render status {status!r}")
            error = value.get("error")
            return status, None if error is None else str(error)
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _acknowledge_render(self, request_id: str) -> None:
        body = json.dumps(
            {
                "version": IDEMPOTENT_BATCH_PROTOCOL_VERSION,
                "request_id": request_id,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url.rstrip("/") + "/render_ack",
            data=body,
            headers=self._request_headers(),
            method="POST",
        )
        self.control_request_count += 1
        try:
            with urllib.request.urlopen(
                request, timeout=float(self.timeout_sec)
            ) as response:
                response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
            # ACK is a memory optimization. The TTL is the recovery fallback,
            # and failure to ACK must never turn a received render into a loss.
            pass

    def _recover_idempotent_batch(
        self,
        payloads: Sequence[dict],
        request_id: str,
        *,
        request_started_at: float,
        want_depth: bool = False,
    ) -> list[GSRenderResult] | None:
        """Wait until an uncertain render is terminal, then fetch it by id."""

        recovery_timeout = float(self.recovery_timeout_sec)
        if not np.isfinite(recovery_timeout) or recovery_timeout <= 0.0:
            self.last_error = (
                "3DGS recovery_timeout_sec must be finite and greater than zero"
            )
            return None
        result_ttl = self._idempotent_result_ttl_sec
        if (
            result_ttl is None
            or not math.isfinite(result_ttl)
            or result_ttl <= 0.0
        ):
            self.last_error = (
                "3DGS idempotent recovery requires a positive finite "
                f"{RESULT_TTL_HEADER} health header"
            )
            return None
        now = time.monotonic()
        # A result may have completed immediately after submission, before a
        # transport timeout was reported. Measure the TTL conservatively from
        # request start, not from the later beginning of recovery.
        deadline = min(
            now + recovery_timeout,
            float(request_started_at) + result_ttl,
        )
        effective_timeout = max(deadline - now, 0.0)
        poll_delay = max(float(self.retry_delay_sec), 0.05)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self.last_error = (
                    "3DGS idempotent render recovery timed out after "
                    f"{effective_timeout:g}s without resubmitting request_id "
                    f"{request_id}"
                )
                return None
            status, detail = self._render_status_once(
                request_id,
                timeout_sec=remaining,
            )
            if status == "pending" or status is None:
                if detail is not None:
                    self.last_error = detail
                remaining = deadline - time.monotonic()
                if remaining > 0.0:
                    time.sleep(min(poll_delay, remaining))
                continue
            if status == "done":
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    continue
                self.attempt_count += len(payloads)
                results, outcome = self._render_batch_once(
                    payloads,
                    request_id=request_id,
                    want_depth=want_depth,
                    timeout_sec=remaining,
                )
                if results is not None:
                    self._acknowledge_render(request_id)
                    self.last_error = None
                    return results
                self.fail_count += len(payloads)
                if outcome in ("uncertain", "retryable"):
                    continue
                self.last_error = self.last_error or (
                    "terminal failure while retrieving completed 3DGS render"
                )
                return None
            if status == "error":
                self.last_error = detail or "3DGS render owner reported an error"
            elif status == "dead":
                self.last_error = detail or "3DGS render server refused connection"
            else:
                self.last_error = detail or (
                    "3DGS render request is unknown to the live server"
                )
            return None

    def _render_payload_results(
        self, payloads: Sequence[dict], *, want_depth: bool
    ) -> list[GSRenderResult | None]:
        """Render immutable camera snapshots, batching when the server supports it.

        ``attempt_count`` and ``fail_count`` remain logical per-camera counts;
        ``request_count`` records render POSTs, while
        ``control_request_count`` records status and ACK requests.
        """

        snapshots = [dict(payload) for payload in payloads]
        direct = any("camera_settings" in payload for payload in snapshots)
        if direct:
            if not want_depth:
                raise ValueError("KB4 rendering requires RGBD results")
            self.require_camera_model("kb4")
            for payload in snapshots:
                if "camera_settings" in payload:
                    payload["camera_settings"] = validate_camera_settings(
                        payload["camera_settings"], payload["width"], payload["height"]
                    )
        if not snapshots:
            self.last_render_attempts = 0
            return []
        if not self.supports_render_batch():
            return [
                self._render_payload_with_retries(
                    payload, want_depth=want_depth
                )
                for payload in snapshots
            ]

        attempts = max(1, int(self.max_attempts))
        request_id = (
            uuid.uuid4().hex if self._idempotent_batch_supported else None
        )
        self.last_render_attempts = 0
        final_outcome: str | None = None
        request_started_at: float | None = None
        for attempt in range(1, attempts + 1):
            self.last_render_attempts = attempt
            self.attempt_count += len(snapshots)
            current_request_started_at = time.monotonic()
            results, outcome = self._render_batch_once(
                snapshots,
                request_id=request_id,
                want_depth=want_depth,
            )
            if request_id is not None:
                request_started_at = current_request_started_at
            if outcome == "unsupported":
                # The health response and endpoint can change during a server
                # restart.  A 404/405 proves no render was performed, so safe
                # legacy fallback does not double-count logical attempts.
                self.attempt_count -= len(snapshots)
                return [
                    self._render_payload_with_retries(
                        payload, want_depth=want_depth
                    )
                    for payload in snapshots
                ]
            if results is not None:
                if request_id is not None:
                    self._acknowledge_render(request_id)
                self.last_error = None
                return results
            self.fail_count += len(snapshots)
            final_outcome = outcome
            if request_id is not None and outcome in ("uncertain", "invalid"):
                # Once acceptance is uncertain, another POST could duplicate
                # work if a restarted or non-conforming server lost the id.
                # Status recovery is the only safe next operation.
                break
            if outcome in ("dead", "terminal"):
                break
            if attempt < attempts and float(self.retry_delay_sec) > 0.0:
                time.sleep(float(self.retry_delay_sec))
        if request_id is not None and final_outcome in ("uncertain", "invalid"):
            if request_started_at is None:  # Defensive: a v2 POST always sets it.
                request_started_at = time.monotonic()
            results = self._recover_idempotent_batch(
                snapshots,
                request_id,
                request_started_at=request_started_at,
                want_depth=want_depth,
            )
            if results is not None:
                return results
        if self.strict or direct:
            raise RuntimeError(
                f"3DGS batch render failed after {attempts} attempts at "
                f"{self.url}: {self.last_error}"
            )
        return [None] * len(snapshots)

    def render_payloads(
        self, payloads: Sequence[dict]
    ) -> list[np.ndarray | None]:
        """Render camera payloads and return RGB-only legacy results."""

        return [
            None if result is None else result.rgb
            for result in self._render_payload_results(
                payloads, want_depth=False
            )
        ]

    def render_payloads_with_depth(
        self, payloads: Sequence[dict]
    ) -> list[GSRenderResult]:
        """Render camera payloads with optional optical-axis depth in meters."""

        return [
            GSRenderResult(rgb=None, depth=None) if result is None else result
            for result in self._render_payload_results(
                payloads, want_depth=True
            )
        ]

    def render_batch(
        self,
        scenes: Sequence,
        width: int,
        height: int,
        fovy_deg: float | None = None,
    ) -> list[np.ndarray | None]:
        """Snapshot several MuJoCo scenes and render them through one request."""

        return self.render_payloads(
            [
                self.camera_payload(
                    scene, width, height, fovy_deg=fovy_deg
                )
                for scene in scenes
            ]
        )

    def render(self, scene, width: int, height: int, fovy_deg: float | None = None) -> np.ndarray | None:
        payload = self.camera_payload(
            scene, width, height, fovy_deg=fovy_deg
        )
        if self.supports_render_batch():
            return self.render_payloads([payload])[0]
        result = self._render_payload_with_retries(payload, want_depth=False)
        return None if result is None else result.rgb


def _snapshot_renderer_state(renderer) -> tuple[np.ndarray | None, bool, bool]:
    """Snapshot MuJoCo render modes and scene flags before a target switch."""

    scene = getattr(renderer, "scene", None)
    flags = getattr(scene, "flags", None)
    return (
        None if flags is None else np.asarray(flags).copy(),
        bool(getattr(renderer, "_depth_rendering", False)),
        bool(getattr(renderer, "_segmentation_rendering", False)),
    )


def _restore_renderer_state(renderer, state) -> None:
    """Restore a renderer even when MuJoCo failed before its own cleanup."""

    flags, depth_rendering, segmentation_rendering = state
    # MuJoCo 3.8.1 restores mjRND_SEGMENT / mjRND_IDCOLOR only at the end of
    # a successful render().  In particular, an invalid segmentation id can
    # raise before that line and poison all later RGB renders.  Restore both
    # public modes and the underlying flags in our own finally block.
    if hasattr(renderer, "disable_depth_rendering"):
        renderer.disable_depth_rendering()
    if hasattr(renderer, "disable_segmentation_rendering"):
        renderer.disable_segmentation_rendering()
    if depth_rendering:
        renderer.enable_depth_rendering()
    elif segmentation_rendering:
        renderer.enable_segmentation_rendering()
    if flags is not None:
        current = getattr(getattr(renderer, "scene", None), "flags", None)
        if current is not None:
            np.copyto(current, flags)


def render_depth(renderer) -> np.ndarray:
    """Return MuJoCo's metric optical-axis depth without leaking render state."""

    state = _snapshot_renderer_state(renderer)
    try:
        renderer.enable_depth_rendering()
        return renderer.render().copy()
    finally:
        _restore_renderer_state(renderer, state)


def _mujoco_far_clip(renderer) -> float | None:
    """Return the metric MuJoCo far clip used by Renderer.render(depth)."""

    model = getattr(renderer, "_model", None)
    try:
        far = float(model.stat.extent) * float(model.vis.map.zfar)
    except (AttributeError, TypeError, ValueError):
        return None
    return far if math.isfinite(far) and far > 0.0 else None


def _mujoco_clear_depth(renderer) -> float | None:
    """Reproduce MuJoCo's float32 reverse-Z depth for a clear pixel."""

    model = getattr(renderer, "_model", None)
    try:
        extent = float(model.stat.extent)
        near = float(model.vis.map.znear) * extent
        far = float(model.vis.map.zfar) * extent
    except (AttributeError, TypeError, ValueError):
        return None
    if not (
        math.isfinite(near)
        and math.isfinite(far)
        and 0.0 < near < far
    ):
        return None

    # Keep the operation order and precision aligned with
    # mujoco.Renderer.render(). Algebraically this is ``far``, but float32
    # cancellation can move it by much more than a few ulps for large
    # far/near ratios (for example 10,000:1).
    zfar = np.float32(far)
    znear = np.float32(near)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        c_coef = -(zfar + znear) / (zfar - znear)
        d_coef = -(np.float32(2.0) * zfar * znear) / (zfar - znear)
        c_coef = np.float32(-0.5) * c_coef - np.float32(0.5)
        d_coef = np.float32(-0.5) * d_coef
        clear = np.float32(np.float64(d_coef) / np.float64(c_coef))
    result = float(clear)
    return result if math.isfinite(result) and result > 0.0 else None


def foreground_mask_from_depth(
    depth: np.ndarray,
    *,
    far_clip: float | None,
    clear_depth: float | None = None,
) -> np.ndarray:
    """Classify visible MuJoCo geometry from one metric depth readback.

    The clear buffer is the known model far clip, so using the observed
    maximum is incorrect when geometry covers the entire image.  Invalid or
    non-positive samples are kept as foreground: an unreliable MuJoCo depth
    must never punch a 3DGS-colored hole through a robot or manipulated part.
    """

    values = np.asarray(depth)
    if values.ndim != 2:
        raise ValueError(f"MuJoCo depth must have shape [H,W], got {values.shape}")
    valid = np.isfinite(values) & (values > 0.0)
    invalid = ~valid
    if far_clip is None:
        if not valid.any():
            return np.ones(values.shape, dtype=bool)
        # This compatibility branch is for renderer-like test doubles which
        # do not expose a model.  Treat a constant valid image as foreground.
        valid_values = values[valid]
        observed_far = float(np.max(valid_values))
        observed_near = float(np.min(valid_values))
        if np.isclose(observed_near, observed_far, rtol=1e-6, atol=1e-8):
            return np.ones(values.shape, dtype=bool)
        cutoff = observed_far * 0.995
    else:
        far = float(far_clip)
        if not math.isfinite(far) or far <= 0.0:
            raise ValueError("far_clip must be finite and positive")
        if clear_depth is not None:
            clear = float(clear_depth)
            if not math.isfinite(clear) or clear <= 0.0:
                raise ValueError("clear_depth must be finite and positive")
            # A clear pixel is converted to one deterministic float32 value.
            # Allow several ulps for platform differences without discarding
            # the last 0.5% of the usable depth range.
            ulp = abs(float(np.spacing(np.float32(clear))))
            tolerance = max(8.0 * ulp, clear * 8.0 * np.finfo(np.float32).eps)
            cutoff = clear - tolerance
        else:
            # Compatibility for renderer-like objects that expose only zfar.
            # The wider tolerance is robust to reverse-Z float32 cancellation.
            cutoff = far * 0.995
    return invalid | (valid & (values < cutoff))


def render_depth_and_foreground_mask(renderer) -> tuple[np.ndarray, np.ndarray]:
    """Capture depth plus a geometry-accurate foreground segmentation mask.

    Depth alone is not a reliable occupancy signal under shared EGL load: a
    clear face can contain finite conversion artifacts and hide a near-camera
    gripper when thresholded. Segmentation supplies occupancy; depth remains
    available only for the MuJoCo-vs-3DGS z-test. Both renders restore all
    renderer state even when MuJoCo's segmentation decoder raises.
    """

    depth = render_depth(renderer)
    mask = render_foreground_mask(renderer)
    return depth, mask


def render_foreground_mask(renderer) -> np.ndarray:
    """Return pixels occupied by a visible MuJoCo geom.

    Depth-buffer thresholding is fragile for very dark, near-camera YUBI
    meshes: pixels on the gripper can be mistaken for the far/background
    plane and the 3DGS backdrop then leaks through as colored speckles.
    MuJoCo's segmentation target keeps the foreground decision independent of
    the material color and depth range.
    """

    state = _snapshot_renderer_state(renderer)
    try:
        renderer.enable_segmentation_rendering()
        private_fields = (
            "_rect",
            "_scene",
            "_mjr_context",
            "_height",
            "_width",
        )
        if all(hasattr(renderer, name) for name in private_fields):
            # MuJoCo 3.8's public segmentation conversion indexes a table with
            # the encoded segid and raises IndexError when a shared EGL readback
            # contains a stale/out-of-range ID. Occupancy needs no object-ID
            # remap: encoded color 0 is background and every other code is a
            # visible geom (or a conservative stale-pixel foreground).
            import mujoco

            renderer._scene.flags[int(mujoco.mjtRndFlag.mjRND_SEGMENT)] = True
            renderer._scene.flags[int(mujoco.mjtRndFlag.mjRND_IDCOLOR)] = True
            if renderer._mjr_context is None:
                raise RuntimeError("render cannot be called after close")
            gl_context = getattr(renderer, "_gl_context", None)
            if gl_context:
                gl_context.make_current()
            raw = np.empty(
                (int(renderer._height), int(renderer._width), 3),
                dtype=np.uint8,
            )
            mujoco.mjr_render(
                renderer._rect,
                renderer._scene,
                renderer._mjr_context,
            )
            mujoco.mjr_readPixels(
                raw,
                None,
                renderer._rect,
                renderer._mjr_context,
            )
            if gl_context:
                raw[:] = np.flipud(raw)
            encoded = raw.astype(np.uint32)
            return (
                encoded[..., 0]
                + encoded[..., 1] * np.uint32(2**8)
                + encoded[..., 2] * np.uint32(2**16)
            ) != 0
        segmentation = renderer.render().copy()
    except (IndexError, ValueError, RuntimeError):
        # Some MuJoCo/EGL combinations can return an invalid encoded
        # segmentation ID after a large offscreen face render.  Keep the
        # foreground compositing path alive with the older depth criterion;
        # this is only a fallback when segmentation itself is unusable.
        _restore_renderer_state(renderer, state)
        depth = render_depth(renderer)
        return foreground_mask_from_depth(
            depth,
            far_clip=_mujoco_far_clip(renderer),
            clear_depth=_mujoco_clear_depth(renderer),
        )
    finally:
        _restore_renderer_state(renderer, state)
    if segmentation.ndim != 3 or segmentation.shape[-1] < 2:
        raise RuntimeError(
            "MuJoCo segmentation render must have shape (H, W, 2); "
            f"got {segmentation.shape}"
        )
    return (segmentation[..., 0] >= 0) & (segmentation[..., 1] >= 0)


def composite_background_mask(
    rgb: np.ndarray, foreground: np.ndarray, background: np.ndarray
) -> np.ndarray:
    """Composite a GS background using an explicit MuJoCo foreground mask."""

    if background.shape[:2] != rgb.shape[:2]:
        background = cv.resize(
            background, (rgb.shape[1], rgb.shape[0]), interpolation=cv.INTER_AREA
        )
    mask = np.asarray(foreground, dtype=bool)
    if mask.shape != rgb.shape[:2]:
        raise ValueError(
            f"foreground mask shape {mask.shape} does not match RGB {rgb.shape[:2]}"
        )
    out = background.copy()
    out[mask] = rgb[mask]
    return out


def _resize_composite_map(
    value: np.ndarray,
    shape: tuple[int, int],
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [H,W], got {array.shape}")
    if array.shape != shape:
        array = cv.resize(
            array,
            (shape[1], shape[0]),
            interpolation=cv.INTER_NEAREST,
        )
    return array


def _mujoco_wins_mask(
    shape: tuple[int, int],
    mujoco_depth: np.ndarray | None,
    mujoco_foreground_mask: np.ndarray,
    gs_depth: np.ndarray | None,
    *,
    depth_epsilon_m: float = _DEFAULT_DEPTH_EPSILON_M,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Return ``(mujoco_wins, mj_depth, gs_depth)`` resized to ``shape``.

    This is the z-test of :func:`composite_background_depth`.
    """

    epsilon = float(depth_epsilon_m)
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("depth_epsilon_m must be finite and non-negative")
    foreground = _resize_composite_map(
        mujoco_foreground_mask, shape, name="mujoco_foreground_mask"
    ).astype(bool, copy=False)
    if gs_depth is None:
        gs_values = np.full(shape, np.inf, dtype=np.float64)
    else:
        gs_values = _resize_composite_map(
            gs_depth, shape, name="gs_depth"
        ).astype(np.float64, copy=False)

    if mujoco_depth is None:
        return foreground, None, gs_values
    mj_values = _resize_composite_map(
        mujoco_depth, shape, name="mujoco_depth"
    ).astype(np.float64, copy=False)
    gs_finite = np.isfinite(gs_values) & (gs_values > 0.0)
    mj_finite = np.isfinite(mj_values) & (mj_values > 0.0)
    mujoco_wins = foreground & (
        (~mj_finite)
        | (~gs_finite)
        | (mj_finite & (mj_values <= gs_values + epsilon))
    )
    return mujoco_wins, mj_values, gs_values


def composite_background_depth(
    mujoco_rgb: np.ndarray,
    mujoco_depth: np.ndarray | None,
    mujoco_foreground_mask: np.ndarray,
    gs_rgb: np.ndarray,
    gs_depth: np.ndarray | None,
    *,
    depth_epsilon_m: float = _DEFAULT_DEPTH_EPSILON_M,
) -> np.ndarray:
    """Z-test MuJoCo foreground against GS optical-axis depth, in meters.

    A GS code-0 pixel decodes to non-finite depth and therefore represents no
    reconstructed geometry. MuJoCo wins such a pixel exactly when its explicit
    foreground mask is true. Missing MuJoCo depth falls back to that same mask
    behavior for compatibility with renderers that cannot expose depth.
    """

    foreground_rgb = np.asarray(mujoco_rgb)
    background_rgb = np.asarray(gs_rgb)
    if foreground_rgb.ndim != 3 or foreground_rgb.shape[-1] != 3:
        raise ValueError(
            "mujoco_rgb must have shape [H,W,3], "
            f"got {foreground_rgb.shape}"
        )
    if background_rgb.ndim != 3 or background_rgb.shape[-1] != 3:
        raise ValueError(
            f"gs_rgb must have shape [H,W,3], got {background_rgb.shape}"
        )
    shape = foreground_rgb.shape[:2]
    if background_rgb.shape[:2] != shape:
        background_rgb = cv.resize(
            background_rgb,
            (shape[1], shape[0]),
            interpolation=cv.INTER_AREA,
        )

    mujoco_wins, _, _ = _mujoco_wins_mask(
        shape,
        mujoco_depth,
        mujoco_foreground_mask,
        gs_depth,
        depth_epsilon_m=depth_epsilon_m,
    )

    output = background_rgb.copy()
    output[mujoco_wins] = foreground_rgb[mujoco_wins]
    return output
