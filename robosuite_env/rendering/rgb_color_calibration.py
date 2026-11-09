#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Shared sRGB calibration applied to simulated wrist RGB observations."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CALIBRATION_SCHEMA_VERSION = 1
DEFAULT_RGB_COLOR_CALIBRATION = (
    Path(__file__).resolve().parent.parent / "assets" / "wrist_rgb_calibration.json"
)


class RGBColorCalibrationError(ValueError):
    """Raised when an RGB calibration asset is invalid."""


@dataclass(frozen=True)
class RGBColorCalibration:
    path: Path
    sha256: str
    space: str
    matrix: np.ndarray
    bias: np.ndarray
    metadata: dict[str, Any]

    def report(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "path": str(self.path),
            "sha256": self.sha256,
            "space": self.space,
            "matrix_rgb": self.matrix.tolist(),
            "bias_rgb": self.bias.tolist(),
            "metadata": self.metadata,
        }


def _as_vector(payload: Any, *, name: str) -> np.ndarray:
    array = np.asarray(payload, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise RGBColorCalibrationError(f"{name} must be a finite RGB vector")
    return array


def load_rgb_color_calibration(
    path: str | Path = DEFAULT_RGB_COLOR_CALIBRATION,
) -> RGBColorCalibration:
    """Load (and memoise) the calibration asset for one path.

    The pair factory builds two renderers per policy observation and per
    rendered video frame, so re-reading and re-hashing this file on every
    construction would put the read on the hot path.  The cache is
    keyed on the file's identity (path plus device/inode/mtime/size), so a
    replaced or edited asset is still re-read and re-hashed rather than
    served stale.  The returned record is shared between callers and must be
    treated as immutable; ``matrix`` and ``bias`` are handed out read-only.
    """

    expanded = Path(path).expanduser()
    try:
        observed = expanded.stat()
    except OSError:
        # Let the uncached read raise the canonical unreadable-file error.
        return _read_rgb_color_calibration(expanded)
    return _cached_rgb_color_calibration(
        str(expanded),
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_mtime_ns),
        int(observed.st_size),
    )


@functools.lru_cache(maxsize=8)
def _cached_rgb_color_calibration(
    path: str,
    device: int,
    inode: int,
    mtime_ns: int,
    size: int,
) -> RGBColorCalibration:
    del device, inode, mtime_ns, size  # keyed for freshness only
    return _read_rgb_color_calibration(Path(path))


def _read_rgb_color_calibration(expanded: Path) -> RGBColorCalibration:
    source = expanded.resolve()
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise RGBColorCalibrationError(
            f"RGB color calibration file is not readable: {source}"
        ) from exc
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RGBColorCalibrationError(
            f"RGB color calibration is not valid JSON: {source}"
        ) from exc
    if not isinstance(payload, dict):
        raise RGBColorCalibrationError("RGB color calibration root must be an object")
    if payload.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise RGBColorCalibrationError(
            f"only RGB color calibration schema_version {CALIBRATION_SCHEMA_VERSION} is supported"
        )
    if payload.get("space") != "srgb_uint8":
        raise RGBColorCalibrationError("RGB color calibration space must be 'srgb_uint8'")
    matrix = np.asarray(payload.get("matrix_rgb"), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise RGBColorCalibrationError("matrix_rgb must be a finite 3x3 matrix")
    bias = _as_vector(payload.get("bias_rgb"), name="bias_rgb")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise RGBColorCalibrationError("metadata must be an object")
    # The record is memoised and therefore shared; freeze the arrays so a
    # caller cannot mutate the transform out from under another renderer.
    matrix.flags.writeable = False  # noqa: V101  # numpy array flag
    bias.flags.writeable = False  # noqa: V101  # numpy array flag
    return RGBColorCalibration(
        path=source,
        sha256=hashlib.sha256(raw).hexdigest(),
        space="srgb_uint8",
        matrix=matrix,
        bias=bias,
        metadata=metadata,
    )


def apply_rgb_color_calibration(
    image: np.ndarray,
    calibration: RGBColorCalibration,
) -> np.ndarray:
    """Apply the checked sRGB RGB->RGB transform without changing shape."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise RGBColorCalibrationError(
            f"RGB image must have shape [H,W,3], got {array.shape}"
        )
    values = np.asarray(array, dtype=np.float32)
    transformed = values @ calibration.matrix.astype(np.float32).T
    transformed += calibration.bias.astype(np.float32)
    return np.rint(np.clip(transformed, 0.0, 255.0)).astype(np.uint8)
