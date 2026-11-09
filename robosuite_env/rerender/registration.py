# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""GS registration capture and generation-identity assertions."""


from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from robosuite_env.rerender.common import (
    RerenderError,
    _GS_OFF_VALUES,
    _last_option_value,
    _resolved_recorded_path,
)
from robosuite_env.rerender.provenance import _assert_record_current, _file_record


def _registration_matrix(
    summary: dict[str, Any], key: str, *, label: str
) -> np.ndarray | None:
    """Load one runtime-reported registration matrix without coercive fallback."""

    raw = summary.get(key)
    if raw is None:
        return None
    if (
        not isinstance(raw, list)
        or len(raw) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in raw)
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for row in raw
            for item in row
        )
    ):
        raise RerenderError(f"{label} {key} must be a 4x4 numeric array or null")
    matrix = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise RerenderError(f"{label} {key} must contain only finite numbers")
    return matrix


def _load_registration_file(
    path: Path, *, label: str
) -> tuple[dict[str, Any], np.ndarray]:
    """Fingerprint and load one stable world-to-GS file as one identity."""

    path = path.resolve()
    record = _file_record(path, include_state=True)
    try:
        value = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise RerenderError(f"cannot load {label} {path}: {exc}") from exc
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (4, 4)
        or np.issubdtype(value.dtype, np.bool_)
        or not np.issubdtype(value.dtype, np.number)
        or np.issubdtype(value.dtype, np.complexfloating)
    ):
        raise RerenderError(f"{label} must contain one real 4x4 NumPy matrix: {path}")
    matrix = np.asarray(value, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise RerenderError(f"{label} contains non-finite values: {path}")
    _assert_record_current(record, label=label, verify_hash=True)
    return record, matrix


def _capture_summary_gs_registration(
    argv: list[str],
    summary: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Bind the path and matrices that one worker says it actually used.

    The worker currently reports the loaded path and the raw/effective matrices,
    but not a file digest.  Hashing the still-current file and requiring its
    loaded matrix to equal the runtime raw matrix closes that gap for post-hoc
    rendering.  Missing or internally inconsistent reports fail closed.
    """

    reported = summary.get("gs_w2gs")
    raw_matrix = _registration_matrix(summary, "gs_w2gs_raw", label=label)
    effective_matrix = _registration_matrix(
        summary, "gs_w2gs_effective", label=label
    )
    raw_offset = summary.get("gs_calibration_world_offset_m")
    if (
        not isinstance(raw_offset, list)
        or len(raw_offset) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in raw_offset
        )
    ):
        raise RerenderError(
            f"{label} gs_calibration_world_offset_m must be a 3-value numeric array"
        )
    calibration_offset = np.asarray(raw_offset, dtype=np.float64)
    if not np.isfinite(calibration_offset).all():
        raise RerenderError(
            f"{label} gs_calibration_world_offset_m must contain only finite numbers"
        )
    if (raw_matrix is None) != (effective_matrix is None):
        raise RerenderError(
            f"{label} gs_w2gs_raw and gs_w2gs_effective must both be matrices or null"
        )

    path: Path | None = None
    file_record: dict[str, Any] | None = None
    if reported == "inline-matrix":
        kind = "inline"
        if raw_matrix is None:
            raise RerenderError(f"{label} inline GS registration has no raw matrix")
    elif reported in (None, ""):
        kind = "identity"
        if raw_matrix is not None:
            raise RerenderError(
                f"{label} reports registration matrices without a gs_w2gs source"
            )
        if np.any(calibration_offset != 0.0):
            raise RerenderError(
                f"{label} identity GS registration requires zero calibration offset"
            )
    elif isinstance(reported, str):
        kind = "file"
        if raw_matrix is None:
            raise RerenderError(f"{label} file GS registration has no raw matrix")
        path = _resolved_recorded_path(reported, label=f"{label} gs_w2gs")
        file_record, loaded_matrix = _load_registration_file(
            path, label=f"{label} GS registration"
        )
        if not np.array_equal(raw_matrix, loaded_matrix):
            raise RerenderError(
                f"{label} runtime raw GS matrix differs from its registered file: {path}"
            )
    else:
        raise RerenderError(f"{label} gs_w2gs has invalid type {type(reported).__name__}")

    if raw_matrix is not None:
        translation = np.eye(4, dtype=np.float64)
        translation[:3, 3] = calibration_offset
        expected_effective = raw_matrix @ translation
        if not np.array_equal(effective_matrix, expected_effective):
            raise RerenderError(
                f"{label} effective GS matrix does not equal raw @ Translation(offset)"
            )

    explicit = _last_option_value(argv, "--gs-transform")
    if explicit is not None:
        if explicit.strip().lower() in _GS_OFF_VALUES:
            if kind != "identity":
                raise RerenderError(
                    f"{label} disagrees with recorded --gs-transform={explicit!r}"
                )
        else:
            explicit_path = _resolved_recorded_path(
                explicit, label="recorded --gs-transform"
            )
            if kind != "file" or path != explicit_path:
                raise RerenderError(
                    f"{label} GS registration path differs from recorded "
                    f"--gs-transform: {path} != {explicit_path}"
                )

    return {
        "kind": kind,
        "path": str(path) if path is not None else None,
        "file": file_record,
        "raw_matrix": raw_matrix.tolist() if raw_matrix is not None else None,
        "effective_matrix": (
            effective_matrix.tolist() if effective_matrix is not None else None
        ),
        "calibration_world_offset_m": calibration_offset.tolist(),
    }


def _registration_raw_matrix(registration: dict[str, Any]) -> np.ndarray | None:
    raw = registration.get("raw_matrix")
    return None if raw is None else np.asarray(raw, dtype=np.float64)


def _assert_registration_matches(
    source: dict[str, Any],
    observed: dict[str, Any],
    *,
    label: str,
    compare_effective: bool,
) -> None:
    """Require an observed registration to preserve the generated identity."""

    source_raw = _registration_raw_matrix(source)
    observed_raw = _registration_raw_matrix(observed)
    if (source_raw is None) != (observed_raw is None) or (
        source_raw is not None and not np.array_equal(source_raw, observed_raw)
    ):
        raise RerenderError(f"{label} raw GS matrix differs from generation")

    source_kind = source.get("kind")
    observed_kind = observed.get("kind")
    if source_kind == "identity":
        if observed_kind != "identity":
            raise RerenderError(f"{label} GS registration source differs from generation")
    elif source_kind == "file":
        source_file = source.get("file")
        observed_file = observed.get("file")
        if (
            observed_kind != "file"
            or not isinstance(source_file, dict)
            or not isinstance(observed_file, dict)
            or source.get("path") != observed.get("path")
            or source_file.get("sha256") != observed_file.get("sha256")
        ):
            raise RerenderError(
                f"{label} GS registration path or SHA-256 differs from generation"
            )
    elif source_kind == "inline":
        if observed_kind != "inline":
            raise RerenderError(f"{label} GS registration source differs from generation")
    else:
        raise RerenderError("generation GS registration has an invalid source kind")

    if compare_effective:
        if source.get("calibration_world_offset_m") != observed.get(
            "calibration_world_offset_m"
        ):
            raise RerenderError(
                f"{label} GS calibration world offset differs from generation"
            )
        source_effective = source.get("effective_matrix")
        observed_effective = observed.get("effective_matrix")
        if (source_effective is None) != (observed_effective is None) or (
            source_effective is not None
            and not np.array_equal(
                np.asarray(source_effective, dtype=np.float64),
                np.asarray(observed_effective, dtype=np.float64),
            )
        ):
            raise RerenderError(f"{label} effective GS matrix differs from generation")


def _required_finite_summary_number(
    summary: dict[str, Any], key: str, *, positive: bool = False
) -> float:
    raw = summary.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise RerenderError(
            f"source summary {key} must be a finite number, got {raw!r}"
        )
    value = float(raw)
    if not math.isfinite(value) or (positive and value <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise RerenderError(
            f"source summary {key} must be {qualifier}, got {raw!r}"
        )
    return value


def _require_summary_close(
    summary: dict[str, Any],
    key: str,
    expected: float,
    *,
    tolerance: float = 1e-6,
) -> float:
    observed = _required_finite_summary_number(summary, key)
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
        raise RerenderError(
            f"source summary {key}={observed:g} disagrees with expected "
            f"{expected:g} (absolute tolerance {tolerance:g})"
        )
    return observed
