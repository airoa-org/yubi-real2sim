# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Pure wire-format helpers for batched 3DGS JPEG responses.

The module intentionally depends only on the Python standard library so the
simulator and the GaussianWrapping runtime can share the exact same framing
contract even when their binary Python dependencies differ.
"""
from __future__ import annotations

import struct
import re
import math
from functools import lru_cache
from collections.abc import Iterable


BATCH_PROTOCOL_VERSION = 1
BATCH_CAPABILITY = "render_batch_v1"
IDEMPOTENT_BATCH_PROTOCOL_VERSION = 2
IDEMPOTENT_BATCH_CAPABILITY = "render_batch_idempotent_v2"
BATCH_CONTENT_TYPE = "application/x-real2sim-jpeg-batch-v1"
RGBD_BATCH_PROTOCOL_VERSION = 1
RGBD_BATCH_CAPABILITY = "render_batch_rgbd_v1"
RGBD_BATCH_CONTENT_TYPE = "application/x-real2sim-rgbd-batch-v1"
CAPABILITIES_HEADER = "X-Real2Sim-GS-Capabilities"
KB4_CAPABILITY = "kb4-radial-depth-v1"
KB4_FIELDS = ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4", "theta_max")
KB4_THETA_MARGIN = 0.5  # [rad] the native rasterizer needs theta_max + KB4_THETA_MARGIN < pi (config.h)
RESULT_TTL_HEADER = "X-Real2Sim-GS-Result-TTL-Sec"
# The KB4 rasterizer drops every Gaussian whose mean lies within kb4_near GS units of the camera
# centre. A KB4 request carries that distance; the patched GaussianWrapping build takes it per call.
KB4_NEAR_CAPABILITY = "kb4-near-v1"
KB4_NEAR_CULL_MODES = ("znear", "legacy")
# The mode of a new run: the worker, the launcher (which always passes it) and GSBackgroundClient share it.
DEFAULT_KB4_NEAR_CULL = "znear"
# znear: the client's metric znear (the same near limit its depth encoding uses), in GS units.
# legacy: 0.2 GS units, the distance GaussianWrapping cd8cb18 fixed in config.h NEAR_PLANE; every
# run recorded before kb4_near existed rendered with it, so their replays select it.
LEGACY_KB4_NEAR_GS_UNITS = 0.2

_REQUEST_ID_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

_MAGIC = b"R2SGSB1\x00"
_RGBD_MAGIC = b"R2SGRGBD1\x00"
_U32 = struct.Struct("!I")


class BatchProtocolError(ValueError):
    """Raised when a batched JPEG payload violates the v1 framing contract."""


def validate_camera_settings(value: object, width: int, height: int) -> dict:
    """Validate the opt-in KB4 camera without accepting native renderer overrides."""

    if not isinstance(value, dict) or value.get("camera_model") != "kb4":
        raise ValueError("camera_settings must be an object with camera_model='kb4'")
    if set(value) != {"camera_model", *KB4_FIELDS}:
        raise ValueError("camera_settings must contain exactly the KB4 intrinsics and theta_max")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (width, height)):
        raise ValueError("KB4 width and height must be positive integers")
    settings = {"camera_model": "kb4"}
    for key in KB4_FIELDS:
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f"KB4 {key} must be a finite number")
        settings[key] = float(number)
    if settings["fx"] <= 0.0 or settings["fy"] <= 0.0:
        raise ValueError("KB4 fx and fy must be positive")
    if not 0.0 < settings["theta_max"] < math.pi - KB4_THETA_MARGIN:
        raise ValueError(f"KB4 theta_max must be between 0 and pi - {KB4_THETA_MARGIN} radians")
    camera = _kb4_camera(width, height, settings)
    if camera.min_distort_slope(settings["theta_max"]) <= 0.0:
        raise ValueError("KB4 distortion must be invertible up to theta_max")
    return settings


def validate_kb4_near(value: object) -> float:
    """A KB4 request's near cull distance in GS units: finite and positive."""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
        raise ValueError("kb4_near must be a finite positive distance in GS units")
    return float(value)


def _kb4_camera(width, height, settings):
    # Legacy framing stays stdlib-only; the opt-in camera shares the wrist model.
    from robosuite_env.rendering.wrist_render_profiles import FisheyeIntrinsics

    return FisheyeIntrinsics(width, height, **{key: settings[key] for key in KB4_FIELDS[:-1]})


@lru_cache(maxsize=16)
def _kb4_valid_mask(width, height, values):
    from robosuite_env.rendering.wrist_render_profiles import camera_rays

    settings = dict(zip(KB4_FIELDS, values))
    _rays, valid = camera_rays(_kb4_camera(width, height, settings), settings["theta_max"])
    valid.setflags(write=False)
    return valid


def kb4_valid_mask(width: int, height: int, settings: dict):
    """Return the shared source-circle mask, retaining rays beyond 90 degrees."""

    settings = validate_camera_settings(settings, width, height)
    return _kb4_valid_mask(width, height, tuple(settings[key] for key in KB4_FIELDS))


def validate_request_id(value: object) -> str:
    """Return a bounded URL-safe request id or raise ``BatchProtocolError``."""

    if not isinstance(value, str) or _REQUEST_ID_RE.fullmatch(value) is None:
        raise BatchProtocolError(
            "request_id must be 1..128 URL-safe ASCII characters "
            "([A-Za-z0-9._-]) and start with an alphanumeric character"
        )
    return value


def pack_jpeg_batch(jpegs: Iterable[bytes]) -> bytes:
    """Pack JPEG byte strings as magic, count, then u32-length-prefixed items."""

    items = tuple(bytes(jpeg) for jpeg in jpegs)
    if len(items) > 0xFFFFFFFF:
        raise BatchProtocolError("JPEG batch contains too many images")
    parts = [_MAGIC, _U32.pack(len(items))]
    for index, jpeg in enumerate(items):
        if not jpeg:
            raise BatchProtocolError(f"JPEG batch item {index} is empty")
        if len(jpeg) > 0xFFFFFFFF:
            raise BatchProtocolError(f"JPEG batch item {index} is too large")
        parts.extend((_U32.pack(len(jpeg)), jpeg))
    return b"".join(parts)


def unpack_jpeg_batch(
    payload: bytes | bytearray | memoryview,
    *,
    expected_count: int | None = None,
    max_count: int = 4096,
) -> list[bytes]:
    """Decode and strictly validate a v1 length-prefixed JPEG batch."""

    view = memoryview(payload)
    header_size = len(_MAGIC) + _U32.size
    if len(view) < header_size:
        raise BatchProtocolError("JPEG batch is truncated before its header")
    if bytes(view[: len(_MAGIC)]) != _MAGIC:
        raise BatchProtocolError("JPEG batch has an invalid magic value")
    count = _U32.unpack_from(view, len(_MAGIC))[0]
    if count > int(max_count):
        raise BatchProtocolError(
            f"JPEG batch count {count} exceeds the limit {int(max_count)}"
        )
    if expected_count is not None and count != int(expected_count):
        raise BatchProtocolError(
            f"JPEG batch count {count} does not match expected {int(expected_count)}"
        )

    offset = header_size
    items: list[bytes] = []
    for index in range(count):
        if len(view) - offset < _U32.size:
            raise BatchProtocolError(
                f"JPEG batch is truncated before item {index} length"
            )
        item_size = _U32.unpack_from(view, offset)[0]
        offset += _U32.size
        if item_size == 0:
            raise BatchProtocolError(f"JPEG batch item {index} is empty")
        if len(view) - offset < item_size:
            raise BatchProtocolError(
                f"JPEG batch item {index} is truncated: "
                f"need {item_size}, have {len(view) - offset}"
            )
        items.append(bytes(view[offset : offset + item_size]))
        offset += item_size
    if offset != len(view):
        raise BatchProtocolError(
            f"JPEG batch has {len(view) - offset} trailing bytes"
        )
    return items


def _rgbd_item_bytes(value: object, *, index: int, field: str) -> bytes:
    """Coerce one RGBD item field from a bytes-compatible buffer."""

    try:
        view = memoryview(value)
    except TypeError as exc:
        raise BatchProtocolError(
            f"RGBD batch item {index} {field} is not bytes-like"
        ) from exc
    return bytes(view)


def pack_rgbd_batch(
    items: Iterable[tuple[bytes, bytes | None]],
) -> bytes:
    """Pack JPEG/depth-PNG pairs into the opt-in RGBD batch framing.

    Every item contains a nonempty JPEG. A missing depth image is encoded as
    a zero-length depth payload; callers may pass either ``None`` or ``b""``
    for that case.
    """

    encoded_items: list[tuple[bytes, bytes]] = []
    for index, item in enumerate(items):
        try:
            item_values = tuple(item)
        except TypeError as exc:
            raise BatchProtocolError(
                f"RGBD batch item {index} must be a JPEG/depth pair"
            ) from exc
        if len(item_values) != 2:
            raise BatchProtocolError(
                f"RGBD batch item {index} must be a JPEG/depth pair"
            )

        jpeg = _rgbd_item_bytes(item_values[0], index=index, field="JPEG")
        if not jpeg:
            raise BatchProtocolError(f"RGBD batch item {index} JPEG is empty")
        if len(jpeg) > 0xFFFFFFFF:
            raise BatchProtocolError(
                f"RGBD batch item {index} JPEG is too large"
            )

        depth_value = item_values[1]
        depth_png = (
            b""
            if depth_value is None
            else _rgbd_item_bytes(depth_value, index=index, field="depth PNG")
        )
        if len(depth_png) > 0xFFFFFFFF:
            raise BatchProtocolError(
                f"RGBD batch item {index} depth PNG is too large"
            )
        encoded_items.append((jpeg, depth_png))

    if len(encoded_items) > 0xFFFFFFFF:
        raise BatchProtocolError("RGBD batch contains too many images")

    parts = [_RGBD_MAGIC, _U32.pack(len(encoded_items))]
    for jpeg, depth_png in encoded_items:
        parts.extend(
            (
                _U32.pack(len(jpeg)),
                _U32.pack(len(depth_png)),
                jpeg,
                depth_png,
            )
        )
    return b"".join(parts)


def unpack_rgbd_batch(
    payload: bytes | bytearray | memoryview,
    *,
    expected_count: int | None = None,
    max_count: int = 4096,
) -> list[tuple[bytes, bytes | None]]:
    """Decode and strictly validate an RGBD JPEG/depth-PNG batch."""

    try:
        view = memoryview(payload)
    except TypeError as exc:
        raise BatchProtocolError("RGBD batch payload is not bytes-like") from exc
    header_size = len(_RGBD_MAGIC) + _U32.size
    if len(view) < header_size:
        raise BatchProtocolError("RGBD batch is truncated before its header")
    if bytes(view[: len(_RGBD_MAGIC)]) != _RGBD_MAGIC:
        raise BatchProtocolError("RGBD batch has an invalid magic value")

    if not isinstance(max_count, int) or isinstance(max_count, bool) or max_count < 0:
        raise BatchProtocolError("RGBD batch max_count must be a nonnegative integer")
    count = _U32.unpack_from(view, len(_RGBD_MAGIC))[0]
    if count > max_count:
        raise BatchProtocolError(
            f"RGBD batch count {count} exceeds the limit {max_count}"
        )
    if expected_count is not None:
        if (
            not isinstance(expected_count, int)
            or isinstance(expected_count, bool)
            or expected_count < 0
        ):
            raise BatchProtocolError(
                "RGBD batch expected_count must be a nonnegative integer"
            )
        if count != expected_count:
            raise BatchProtocolError(
                f"RGBD batch count {count} does not match expected {expected_count}"
            )

    offset = header_size
    decoded_items: list[tuple[bytes, bytes | None]] = []
    lengths_size = 2 * _U32.size
    for index in range(count):
        if len(view) - offset < lengths_size:
            raise BatchProtocolError(
                f"RGBD batch is truncated before item {index} lengths"
            )
        jpeg_size = _U32.unpack_from(view, offset)[0]
        depth_size = _U32.unpack_from(view, offset + _U32.size)[0]
        offset += lengths_size

        if jpeg_size == 0:
            raise BatchProtocolError(f"RGBD batch item {index} JPEG is empty")
        item_size = jpeg_size + depth_size
        remaining = len(view) - offset
        if remaining < item_size:
            raise BatchProtocolError(
                f"RGBD batch item {index} is truncated: "
                f"need {item_size}, have {remaining}"
            )

        jpeg_end = offset + jpeg_size
        depth_end = jpeg_end + depth_size
        jpeg = bytes(view[offset:jpeg_end])
        depth_png = bytes(view[jpeg_end:depth_end]) if depth_size else None
        decoded_items.append((jpeg, depth_png))
        offset = depth_end

    if offset != len(view):
        raise BatchProtocolError(
            f"RGBD batch has {len(view) - offset} trailing bytes"
        )
    return decoded_items


__all__ = [
    "BATCH_CAPABILITY",
    "BATCH_CONTENT_TYPE",
    "BATCH_PROTOCOL_VERSION",
    "BatchProtocolError",
    "CAPABILITIES_HEADER",
    "IDEMPOTENT_BATCH_CAPABILITY",
    "IDEMPOTENT_BATCH_PROTOCOL_VERSION",
    "RESULT_TTL_HEADER",
    "RGBD_BATCH_CAPABILITY",
    "RGBD_BATCH_CONTENT_TYPE",
    "RGBD_BATCH_PROTOCOL_VERSION",
    "pack_jpeg_batch",
    "pack_rgbd_batch",
    "unpack_jpeg_batch",
    "unpack_rgbd_batch",
    "validate_request_id",
]
