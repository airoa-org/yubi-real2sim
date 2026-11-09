# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import struct

import pytest

from scene.gs_background_protocol import (
    BATCH_CONTENT_TYPE,
    BatchProtocolError,
    RGBD_BATCH_CAPABILITY,
    RGBD_BATCH_CONTENT_TYPE,
    RGBD_BATCH_PROTOCOL_VERSION,
    pack_jpeg_batch,
    pack_rgbd_batch,
    unpack_jpeg_batch,
    unpack_rgbd_batch,
    validate_request_id,
)


def test_jpeg_batch_round_trip_preserves_item_boundaries():
    items = [b"jpeg-one", b"\xff\xd8jpeg-two\xff\xd9"]

    encoded = pack_jpeg_batch(items)

    assert unpack_jpeg_batch(encoded, expected_count=2) == items


def test_rgbd_batch_round_trip_preserves_depth_and_missing_depth():
    items = [
        (b"jpeg-one", b"\x89PNG\r\n\x1a\ndepth-one"),
        (b"jpeg-two", None),
        (b"jpeg-three", b""),
    ]

    encoded = pack_rgbd_batch(items)

    assert RGBD_BATCH_PROTOCOL_VERSION == 1
    assert RGBD_BATCH_CAPABILITY == "render_batch_rgbd_v1"
    assert RGBD_BATCH_CONTENT_TYPE != BATCH_CONTENT_TYPE
    assert unpack_rgbd_batch(encoded, expected_count=3) == [
        items[0],
        (b"jpeg-two", None),
        (b"jpeg-three", None),
    ]
def _rgbd_truncated_lengths_frame():
    encoded = bytearray(pack_rgbd_batch([]))
    encoded[-4:] = struct.pack("!I", 1)
    return encoded


def _rgbd_zero_jpeg_frame():
    jpeg = b"jpeg"
    encoded = bytearray(pack_rgbd_batch([(jpeg, None)]))
    offset = len(encoded) - 2 * 4 - len(jpeg)
    encoded[offset : offset + 4] = struct.pack("!I", 0)
    return encoded


@pytest.mark.parametrize(
    "unpack, payload, message",
    [
        (unpack_jpeg_batch, b"short", "truncated before its header"),
        (unpack_jpeg_batch, b"wrong!!!" + struct.pack("!I", 0), "invalid magic"),
        (unpack_jpeg_batch, pack_jpeg_batch([b"one"])[:-1], "item 0 is truncated"),
        (unpack_jpeg_batch, pack_jpeg_batch([b"one"]) + b"extra", "trailing bytes"),
        (unpack_jpeg_batch, pack_rgbd_batch([(b"jpeg", None)]), "invalid magic"),
        (unpack_rgbd_batch, b"short", "truncated before its header"),
        (unpack_rgbd_batch, b"wrong-rgbd" + struct.pack("!I", 0), "invalid magic"),
        (unpack_rgbd_batch, pack_rgbd_batch([(b"jpeg", b"depth")])[:-1], "item 0 is truncated"),
        (unpack_rgbd_batch, pack_rgbd_batch([(b"jpeg", None)]) + b"extra", "trailing bytes"),
        (unpack_rgbd_batch, pack_jpeg_batch([b"jpeg"]), "invalid magic"),
        (unpack_rgbd_batch, _rgbd_truncated_lengths_frame(), "truncated before item 0 lengths"),
        (unpack_rgbd_batch, _rgbd_zero_jpeg_frame(), "item 0 JPEG is empty"),
    ],
)
def test_batch_codecs_reject_malformed_frames(unpack, payload, message):
    with pytest.raises(BatchProtocolError, match=message):
        unpack(payload)


@pytest.mark.parametrize(
    "operation, message",
    [
        (lambda: unpack_jpeg_batch(pack_jpeg_batch([b"one"]), expected_count=2), "does not match expected"),
        (lambda: pack_jpeg_batch([b""]), "item 0 is empty"),
        (lambda: pack_rgbd_batch([(b"", b"depth")]), "item 0 JPEG is empty"),
        (lambda: pack_rgbd_batch([(b"jpeg",)]), "must be a JPEG/depth pair"),
        (lambda: pack_rgbd_batch([(b"jpeg", "depth")]), "depth PNG is not bytes-like"),
        (lambda: unpack_rgbd_batch(pack_rgbd_batch([(b"jpeg", None)]), expected_count=2), "does not match expected"),
        (lambda: unpack_rgbd_batch(pack_rgbd_batch([(b"jpeg", None)]), max_count=0), "exceeds the limit"),
    ],
)
def test_batch_codecs_reject_invalid_items_and_counts(operation, message):
    with pytest.raises(BatchProtocolError, match=message):
        operation()


@pytest.mark.parametrize("value", ["a", "abc-DEF_012.9", "0" * 128])
def test_request_id_accepts_bounded_url_safe_ascii(value):
    assert validate_request_id(value) == value


@pytest.mark.parametrize(
    "value",
    [None, "", "-leading", "contains/slash", "contains space", "é", "x" * 129],
)
def test_request_id_rejects_unsafe_or_unbounded_values(value):
    with pytest.raises(BatchProtocolError, match="request_id"):
        validate_request_id(value)
