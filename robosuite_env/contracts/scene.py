# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Validate one 3DGS scene and its matching MuJoCo table collision bundle.

The v2 ``background.json`` is the single source of truth for a scene.  It
binds the rendered 3DGS background, its world transform, and the invisible
table collision used by MuJoCo.  Callers must select one exact scene path;
this module intentionally does not search recursively for manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from robosuite_env.contracts.provenance import sha256_file


SCENE_SCHEMA_VERSION = 2
# Table-height consistency check (schema v2). ``calibration_table_z_m`` is the
# z coordinate, in MuJoCo world metres, of the 3DGS table-top plane after the
# point cloud is mapped through inv(mat_w2gs). The bound table collision is
# normalized to a z=0 table-top in the same ``mujoco_world`` frame, so a
# correctly baked scene declares ~0; a w2gs calibrated for another table_z
# convention (for example the +0.8 m source calibration) is rejected here
# instead of silently producing a floating or buried table at runtime.
TABLE_Z_METHOD = "ply-plane-fit-v1"
TABLE_Z_TOLERANCE_M = 0.05
TABLE_Z_RUNTIME_M = 0.0
# Fixture point clouds (a handful of vertices) carry no table surface; the
# measurement is skipped below this vertex count and the declared value is
# accepted as-is. Every real 3DGS scene has hundreds of thousands of points.
TABLE_Z_MIN_VERTICES = 100
_TABLE_Z_SAMPLE_STRIDE = 50
_TABLE_Z_XY_RADIUS_M = 0.15
_TABLE_Z_XY_CENTER = (0.0, -0.10)
_TABLE_Z_HISTOGRAM_RANGE_M = (-1.5, 1.5)
_TABLE_Z_HISTOGRAM_BIN_M = 0.005
_TABLE_Z_REFINE_WINDOW_M = 0.01
_PLY_HEADER_LIMIT_BYTES = 1 << 20
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PLY_KEYS = (
    "ply",
    "gs",
    "gs_ply",
    "point_cloud_ply",
    "ply_path",
    "path",
)
_W2GS_PATH_KEYS = (
    "w2gs",
    "gs_w2gs",
    "mat_w2gs_path",
    "world_to_gs",
    "world_to_gs_path",
)
_UNSAFE_LEGACY_ALIGNMENT_FIELDS = frozenset(
    {
        "alignment",
        "transform",
        "w2gs_matrix",
        "world_to_gs_matrix",
        "rot_R",
        "rotation_matrix",
        "rot",
        "gs_rot",
        "rotation",
        "pivot",
        "rot_piv",
        "rotation_pivot",
        "gs_pivot",
        "pivz",
        "gs_pivz",
        "pivot_z",
        "rot_pivz",
        "use_legacy_alignment",
        "legacy_alignment",
    }
)


class SceneContractError(RuntimeError):
    """Raised when a scene is incomplete, mismatched, or not auditable."""


def scene_tag(name: Any) -> str:
    """Normalize a scene name exactly as the batch output directory does."""

    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_").lower()
    return normalized or "gs_scene"


# The canonical single-source implementation; the private alias keeps this
# module's call sites (and the tests that patch them) unchanged.
_sha256 = sha256_file


def _expected_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise SceneContractError(
            f"scene field {field} must be a lowercase SHA-256 digest"
        )
    return value


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SceneContractError(f"scene field {field} must be an object")
    return value


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SceneContractError(f"scene field {field} must be a non-empty string")
    return value.strip()


def _manifest_path(scene_path: str | os.PathLike[str]) -> Path:
    supplied = Path(scene_path).expanduser()
    if supplied.is_dir():
        manifest = supplied / "background.json"
    elif supplied.name == "background.json":
        manifest = supplied
    else:
        raise SceneContractError(
            "scene path must be a scene directory or its exact background.json"
        )
    manifest = manifest.resolve()
    if not manifest.is_file():
        raise SceneContractError(f"scene background.json not found: {manifest}")
    return manifest


def _resolve_path(value: Any, field: str, base: Path) -> Path:
    raw = _nonempty_string(value, field)
    expanded = Path(os.path.expandvars(raw)).expanduser()
    path = expanded if expanded.is_absolute() else base / expanded
    return path.resolve()


def _require_file(value: Any, field: str, base: Path) -> Path:
    path = _resolve_path(value, field, base)
    if not path.is_file():
        raise SceneContractError(f"scene field {field} does not name a file: {path}")
    return path


def _require_scene_file(value: Any, field: str, base: Path) -> Path:
    """Resolve one file that must be content-addressed by the scene artifact."""

    path = _require_file(value, field, base)
    try:
        path.relative_to(base.resolve())
    except ValueError as exc:
        raise SceneContractError(
            f"scene field {field} must stay inside the scene directory: {path}"
        ) from exc
    return path


def _verify_file_hash(path: Path, expected: Any, field: str) -> str:
    wanted = _expected_sha256(expected, field)
    actual = _sha256(path)
    if actual != wanted:
        raise SceneContractError(
            f"scene field {field} does not match {path}: expected {wanted}, got {actual}"
        )
    return actual


def _inline_matrix(value: Any, field: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 4:
        raise SceneContractError(f"scene field {field} must be a 4x4 numeric array")
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise SceneContractError(
                f"scene field {field} must be a 4x4 numeric array"
            )
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in row
        ):
            raise SceneContractError(
                f"scene field {field} must contain only finite numbers"
            )
    return np.asarray(value, dtype=np.float64)


def _loaded_matrix(path: Path, field: str) -> np.ndarray:
    try:
        value = (
            np.loadtxt(path)
            if path.suffix.lower() in {".txt", ".csv"}
            else np.load(path, allow_pickle=False)
        )
    except Exception as exc:
        raise SceneContractError(
            f"scene field {field} is not a readable NumPy matrix: {path}"
        ) from exc
    if not isinstance(value, np.ndarray):
        raise SceneContractError(
            f"scene field {field} must contain one NumPy array: {path}"
        )
    if (
        np.issubdtype(value.dtype, np.bool_)
        or not np.issubdtype(value.dtype, np.number)
        or np.issubdtype(value.dtype, np.complexfloating)
    ):
        raise SceneContractError(
            f"scene field {field} must contain a real numeric matrix: {path}"
        )
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SceneContractError(
            f"scene field {field} must contain a real numeric matrix: {path}"
        ) from exc


def _validate_transform(matrix: np.ndarray, field: str) -> np.ndarray:
    if matrix.shape != (4, 4):
        raise SceneContractError(f"scene field {field} must be a 4x4 matrix")
    if not np.isfinite(matrix).all():
        raise SceneContractError(f"scene field {field} must contain only finite numbers")
    if not np.allclose(
        matrix[3],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=1e-9,
    ):
        raise SceneContractError(
            f"scene field {field} must have homogeneous bottom row [0, 0, 0, 1]"
        )

    linear = matrix[:3, :3]
    determinant = float(np.linalg.det(linear))
    if not math.isfinite(determinant) or determinant <= 1e-12:
        raise SceneContractError(
            f"scene field {field} must have a positive non-degenerate 3x3 determinant"
        )
    gram = linear.T @ linear
    scale_squared = float(np.trace(gram) / 3.0)
    if not math.isfinite(scale_squared) or scale_squared <= 1e-12:
        raise SceneContractError(
            f"scene field {field} must have a positive non-degenerate scale"
        )
    if not np.allclose(
        gram,
        np.eye(3, dtype=np.float64) * scale_squared,
        # Registration matrices are stored as float32 and accumulate small
        # solver round-off. This accepts sub-0.001% scale noise while still
        # rejecting meaningful anisotropic scale or shear.
        rtol=1e-5,
        atol=max(1e-10, scale_squared * 1e-5),
    ):
        raise SceneContractError(
            f"scene field {field} 3x3 block must be orthogonal with uniform scale"
        )
    return matrix


def ply_vertex_count(ply_path: str | os.PathLike[str]) -> int:
    """Return the ``element vertex`` count from a PLY header without reading data.

    A header with no vertex element counts as zero vertices. Anything that is
    not a PLY header (missing magic or ``end_header`` within the first MiB)
    is a contract error: a schema-v2 scene must ship a real point cloud.
    """

    path = Path(ply_path)
    try:
        with path.open("rb") as stream:
            head = stream.read(_PLY_HEADER_LIMIT_BYTES)
    except OSError as exc:
        raise SceneContractError(f"scene point cloud is unreadable: {path}") from exc
    lines = head.replace(b"\r\n", b"\n").split(b"\n")
    if not lines or lines[0].strip() != b"ply":
        raise SceneContractError(f"scene point cloud is not a PLY file: {path}")
    count = 0
    for raw in lines[1:]:
        fields = raw.strip().split()
        if fields[:1] == [b"end_header"]:
            return count
        if len(fields) == 3 and fields[0] == b"element" and fields[1] == b"vertex":
            try:
                count = int(fields[2])
            except ValueError as exc:
                raise SceneContractError(
                    f"scene point cloud has an invalid vertex count: {path}"
                ) from exc
            if count < 0:
                raise SceneContractError(
                    f"scene point cloud has an invalid vertex count: {path}"
                )
    raise SceneContractError(f"scene point cloud PLY header is incomplete: {path}")


def measure_gs_table_plane_z(
    ply_path: str | os.PathLike[str],
    w2gs_matrix: Any,
    *,
    sample_stride: int = _TABLE_Z_SAMPLE_STRIDE,
    xy_radius_m: float = _TABLE_Z_XY_RADIUS_M,
    xy_center: tuple[float, float] = _TABLE_Z_XY_CENTER,
) -> float:
    """Measure the dominant 3DGS surface height above the table probe region.

    Every ``sample_stride``-th PLY vertex is mapped into MuJoCo world through
    ``inv(w2gs_matrix)`` as a homogeneous 4x4 (uniform scale is fine; the
    result is divided by w). Points whose |x - cx| and |y - cy| both stay
    within ``xy_radius_m`` of ``xy_center`` are histogrammed along z in 5 mm
    bins over [-1.5, 1.5] m; the result is the median of the points within
    1 cm of the modal bin. The procedure is deterministic and walks the PLY
    through a memory map in bounded chunks (20k sampled points at a time), so
    heap usage stays flat regardless of point-cloud size; the strided read does
    touch file-backed pages (reclaimable, ~3 GB resident on a 10 GB PLY). This
    is ``calibration_table_z_method`` ``ply-plane-fit-v1``.
    """

    try:
        from plyfile import PlyData
    except ImportError as exc:
        # plyfile==1.1 is pinned in both uv.lock and requirements-lock.ci.txt,
        # so this only fires in an ad-hoc environment; fail with a clear message.
        raise SceneContractError(
            "plyfile is required to measure the GS table plane"
        ) from exc

    if isinstance(sample_stride, bool) or not isinstance(sample_stride, int) or sample_stride <= 0:
        raise SceneContractError("sample_stride must be a positive integer")
    radius = float(xy_radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise SceneContractError("xy_radius_m must be a positive finite number")
    center = np.asarray(tuple(xy_center), dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise SceneContractError("xy_center must contain two finite numbers")
    matrix = np.asarray(w2gs_matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise SceneContractError("w2gs_matrix must be a finite 4x4 matrix")
    try:
        gs_to_world = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise SceneContractError("w2gs_matrix is singular") from exc

    path = Path(ply_path)
    try:
        ply = PlyData.read(os.fspath(path), mmap="r")
    except Exception as exc:
        raise SceneContractError(f"scene point cloud is not a readable PLY: {path}") from exc
    try:
        vertex = ply["vertex"]
    except KeyError as exc:
        raise SceneContractError(f"scene point cloud has no vertex element: {path}") from exc
    rows = vertex.data
    names = set(rows.dtype.names or ())
    if not {"x", "y", "z"} <= names:
        raise SceneContractError(f"scene point cloud vertices lack x/y/z: {path}")

    count = int(vertex.count)
    chunk = sample_stride * 20_000
    selected: list[np.ndarray] = []
    for start in range(0, count, chunk):
        block = rows[start : start + chunk : sample_stride]
        if len(block) == 0:
            continue
        points = np.empty((len(block), 4), dtype=np.float64)
        points[:, 0] = block["x"]
        points[:, 1] = block["y"]
        points[:, 2] = block["z"]
        points[:, 3] = 1.0
        world = points @ gs_to_world.T
        w = world[:, 3]
        finite = np.isfinite(world).all(axis=1) & (np.abs(w) > 1e-12)
        world = world[finite]
        xyz = world[:, :3] / world[:, 3:4]
        probe = (
            (np.abs(xyz[:, 0] - center[0]) <= radius)
            & (np.abs(xyz[:, 1] - center[1]) <= radius)
        )
        if probe.any():
            selected.append(xyz[probe, 2])
    if not selected:
        raise SceneContractError(
            "no sampled GS points fall inside the table probe region "
            f"(center {tuple(center.tolist())}, radius {radius} m): {path}"
        )
    z = np.concatenate(selected)
    low, high = _TABLE_Z_HISTOGRAM_RANGE_M
    bins = int(round((high - low) / _TABLE_Z_HISTOGRAM_BIN_M))
    counts, edges = np.histogram(z, bins=bins, range=(low, high))
    if counts.max() <= 0:
        raise SceneContractError(
            "no sampled GS points in the table probe region lie within "
            f"[{low}, {high}] m of the MuJoCo world origin: {path}"
        )
    mode_index = int(np.argmax(counts))
    mode_z = 0.5 * (edges[mode_index] + edges[mode_index + 1])
    near_mode = z[np.abs(z - mode_z) <= _TABLE_Z_REFINE_WINDOW_M]
    if near_mode.size == 0:
        return float(mode_z)
    return float(np.median(near_mode))


def _declared_table_z(spec: dict[str, Any]) -> float:
    value = spec.get("calibration_table_z_m")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SceneContractError(
            "schema v2 scene must declare calibration_table_z_m as a finite "
            "number (the 3DGS table-top z in MuJoCo world after inv(mat_w2gs); "
            "re-run `python -m robosuite_env.contracts.scene upgrade` to measure it)"
        )
    method = spec.get("calibration_table_z_method", TABLE_Z_METHOD)
    if method != TABLE_Z_METHOD:
        raise SceneContractError(
            "scene field calibration_table_z_method must equal "
            f"{TABLE_Z_METHOD!r}; got {method!r}"
        )
    declared = float(value)
    if abs(declared - TABLE_Z_RUNTIME_M) > TABLE_Z_TOLERANCE_M:
        raise SceneContractError(
            f"scene field calibration_table_z_m={declared:+.4f} m places the 3DGS "
            f"table-top outside the mujoco_world table_z={TABLE_Z_RUNTIME_M:g} "
            f"convention (tolerance {TABLE_Z_TOLERANCE_M:g} m); bake the "
            "correction into mat_w2gs with `upgrade --calibration-world-offset "
            f"0 0 {declared:+.4f}` from the legacy manifest"
        )
    return declared


def _verify_table_z(
    declared: float,
    *,
    ply_path: Path,
    matrix: np.ndarray,
) -> float | None:
    """Measure the GS table plane and compare it with the declared value.

    Returns the measured z, or ``None`` when the PLY is a fixture with fewer
    than ``TABLE_Z_MIN_VERTICES`` vertices (documented skip).
    """

    if ply_vertex_count(ply_path) < TABLE_Z_MIN_VERTICES:
        return None
    measured = measure_gs_table_plane_z(ply_path, matrix)
    if abs(declared - measured) > TABLE_Z_TOLERANCE_M:
        raise SceneContractError(
            f"scene field calibration_table_z_m={declared:+.4f} m disagrees with "
            f"the 3DGS table plane measured through inv(mat_w2gs): "
            f"{measured:+.4f} m (tolerance {TABLE_Z_TOLERANCE_M:g} m); the "
            "point cloud or transform was calibrated for a different table_z "
            "convention than this contract declares"
        )
    return measured


def compute_tree_sha256(directory: Path, files: Iterable[Path]) -> str:
    """Hash selected files by relative path and content digest.

    The format is stable and independent of filesystem enumeration order:
    ``relative/path``, a NUL byte, the file SHA-256, and a newline per file.
    """

    root = directory.resolve()
    records: list[tuple[str, Path]] = []
    for supplied in files:
        path = Path(supplied).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise SceneContractError(
                f"collision file escapes its declared directory: {path}"
            ) from exc
        if not path.is_file():
            raise SceneContractError(f"collision path is not a file: {path}")
        records.append((relative, path))

    digest = hashlib.sha256()
    for relative, path in sorted(records):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _collision_contract(
    spec: Any,
    *,
    base: Path,
) -> tuple[Path, tuple[Path, ...], str]:
    collision = _mapping(spec, "table.collision")
    allowed = {"directory", "glob", "count", "tree_sha256", "frame"}
    unknown = sorted(set(collision) - allowed)
    if unknown:
        raise SceneContractError(
            "scene field table.collision contains unsupported fields: "
            + ", ".join(unknown)
        )
    if collision.get("frame") != "mujoco_world":
        raise SceneContractError(
            "scene field table.collision.frame must equal mujoco_world"
        )

    directory = _resolve_path(
        collision.get("directory"),
        "table.collision.directory",
        base,
    )
    if not directory.is_dir():
        raise SceneContractError(
            "scene field table.collision.directory does not name a directory: "
            f"{directory}"
        )
    pattern = _nonempty_string(collision.get("glob"), "table.collision.glob")
    pattern_path = Path(pattern)
    if pattern_path.is_absolute() or ".." in pattern_path.parts:
        raise SceneContractError(
            "scene field table.collision.glob must stay inside its directory"
        )
    expected_count = collision.get("count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
    ):
        raise SceneContractError(
            "scene field table.collision.count must be a positive integer"
        )

    matched: dict[str, Path] = {}
    root = directory.resolve()
    for candidate in directory.glob(pattern):
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise SceneContractError(
                f"collision file escapes its declared directory: {candidate}"
            ) from exc
        matched[relative] = resolved
    files = tuple(path for _, path in sorted(matched.items()))
    if not files:
        raise SceneContractError(
            f"scene table collision glob matched no files: {directory / pattern}"
        )
    if len(files) != expected_count:
        raise SceneContractError(
            "scene table collision count mismatch: "
            f"expected {expected_count}, got {len(files)}"
        )

    actual_tree_hash = compute_tree_sha256(directory, files)
    wanted_tree_hash = _expected_sha256(
        collision.get("tree_sha256"),
        "table.collision.tree_sha256",
    )
    if actual_tree_hash != wanted_tree_hash:
        raise SceneContractError(
            "scene field table.collision.tree_sha256 does not match selected "
            f"collision files: expected {wanted_tree_hash}, got {actual_tree_hash}"
        )
    return directory, files, actual_tree_hash


def _pick(spec: dict[str, Any], keys: Iterable[str]) -> tuple[str | None, Any]:
    for key in keys:
        value = spec.get(key)
        if value not in (None, ""):
            return key, value
    return None, None


def _legacy_contract(
    spec: dict[str, Any],
    *,
    manifest: Path,
) -> dict[str, Any]:
    base = manifest.parent
    ply_key, ply_value = _pick(spec, _PLY_KEYS)
    if ply_key is None:
        raise SceneContractError(f"legacy scene has no PLY path: {manifest}")
    ply_path = _require_file(ply_value, ply_key, base)
    ply_hash = _sha256(ply_path)

    inline_value = spec.get("mat_w2gs")
    w2gs_key, w2gs_value = _pick(spec, _W2GS_PATH_KEYS)
    if inline_value not in (None, "") and w2gs_key is not None:
        raise SceneContractError(
            "legacy scene must not define both mat_w2gs and a w2gs file"
        )
    if inline_value not in (None, ""):
        if "w2gs_sha256" in spec:
            raise SceneContractError(
                "legacy inline mat_w2gs must not define file-only w2gs_sha256"
            )
        matrix = _validate_transform(
            _inline_matrix(inline_value, "mat_w2gs"),
            "mat_w2gs",
        )
        w2gs_path = None
        w2gs_hash = None
    elif w2gs_key is not None:
        w2gs_path = _require_file(w2gs_value, w2gs_key, base)
        matrix = _validate_transform(
            _loaded_matrix(w2gs_path, w2gs_key),
            w2gs_key,
        )
        w2gs_hash = (
            _verify_file_hash(
                w2gs_path,
                spec.get("w2gs_sha256"),
                "w2gs_sha256",
            )
            if "w2gs_sha256" in spec
            else _sha256(w2gs_path)
        )
    else:
        w2gs_path = None
        matrix = np.eye(4, dtype=np.float64)
        w2gs_hash = None

    name = str(spec.get("name") or base.name)
    scene_id = str(spec.get("id") or name)
    digest_payload = {
        "manifest": spec,
        "ply_sha256": ply_hash,
        "w2gs_sha256": w2gs_hash,
        "w2gs_matrix": matrix.tolist(),
    }
    contract_digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": int(spec.get("schema_version", 1)),
        "id": scene_id,
        "name": name,
        "manifest_path": manifest,
        "background_dir": base,
        "ply_path": ply_path,
        "ply_sha256": ply_hash,
        "w2gs_path": w2gs_path,
        "w2gs_matrix": matrix,
        "collision_directory": None,
        "collision_files": (),
        "collision_tree_sha256": None,
        "contract_digest": contract_digest,
    }


def validate_scene_contract(
    path: str | os.PathLike[str],
    *,
    strict: bool = True,
    preverified_ply_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate exactly one scene directory or ``background.json``.

    Strict mode accepts only schema v2, including the bound table collision.
    ``strict=False`` retains read-only compatibility with legacy v1 scenes;
    those reports contain no table collision and are not suitable for official
    rollouts. ``preverified_ply_sha256`` may contain a lowercase SHA-256
    digest computed while loading a v2 PLY. The digest remains matched against
    the manifest, but the PLY bytes are not read a second time.
    """

    manifest = _manifest_path(path)
    try:
        spec = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SceneContractError(
            f"scene background.json is unreadable: {manifest}"
        ) from exc
    if not isinstance(spec, dict):
        raise SceneContractError(f"scene background.json must contain an object: {manifest}")

    schema_version = spec.get("schema_version", 1)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise SceneContractError("scene field schema_version must be an integer")
    if schema_version != SCENE_SCHEMA_VERSION:
        if strict:
            raise SceneContractError(
                f"official scene requires schema_version {SCENE_SCHEMA_VERSION}; "
                f"got {schema_version} in {manifest}"
            )
        if schema_version != 1:
            raise SceneContractError(
                f"unsupported scene schema_version {schema_version}: {manifest}"
            )
        if preverified_ply_sha256 is not None:
            raise SceneContractError(
                "preverified_ply_sha256 requires a schema v2 scene"
            )
        return _legacy_contract(spec, manifest=manifest)

    return _validate_v2_spec(
        spec,
        manifest=manifest,
        preverified_ply_sha256=preverified_ply_sha256,
    )


def _v2_transform(
    spec: dict[str, Any], base: Path
) -> tuple[np.ndarray, Path | None, str | None]:
    """The verified world -> GS transform of one v2 spec: matrix, file, file hash."""

    inline_present = spec.get("mat_w2gs") not in (None, "")
    path_present = spec.get("w2gs") not in (None, "")
    if inline_present == path_present:
        raise SceneContractError(
            "schema v2 scene must define exactly one of mat_w2gs or w2gs"
        )
    if inline_present:
        if "w2gs_sha256" in spec:
            raise SceneContractError(
                "inline mat_w2gs must not define the file-only field w2gs_sha256"
            )
        matrix = _validate_transform(
            _inline_matrix(spec["mat_w2gs"], "mat_w2gs"),
            "mat_w2gs",
        )
        return matrix, None, None
    w2gs_path = _require_scene_file(spec["w2gs"], "w2gs", base)
    w2gs_hash = _verify_file_hash(
        w2gs_path,
        spec.get("w2gs_sha256"),
        "w2gs_sha256",
    )
    matrix = _validate_transform(
        _loaded_matrix(w2gs_path, "w2gs"),
        "w2gs",
    )
    return matrix, w2gs_path, w2gs_hash

_CALIBRATION_SOURCE_FIELDS = frozenset({"manifest", "ply_sha256"})

def _calibration_source(
    spec: dict[str, Any],
    *,
    base: Path,
    matrix: np.ndarray,
    declared_table_z: float,
) -> dict[str, Any] | None:
    """Resolve the clean scene a counterfactual variant is calibrated by.

    A variant built from a validated scene keeps that scene's transform byte for
    byte, so its registration is the source's registration.  Re-measuring the
    table plane on the variant's own cloud asks a deliberately damaged tabletop
    where the table is: displacing it spreads the modal bin, and on a desk with
    open floor beneath it the floor then wins and the check reports a
    calibration error that does not exist.  Restricting where the measurement
    looks cannot fix that without also blinding the check to what it exists to
    catch, a cloud registered for the +0.8 m convention, whose floor sits
    exactly at z = 0.

    So a variant names its source.  The source must be a directly calibrated v2
    scene (no chain) that pins the point cloud the variant names; its transform
    must equal the variant's and the two must declare the same table height;
    and the source must then validate as a complete v2 scene whose table plane
    was actually measured.  That measurement, unchanged in method, is the one
    the variant reports.
    """

    if "calibration_source" not in spec:
        return None
    source = _mapping(spec.get("calibration_source"), "calibration_source")
    unknown = sorted(set(source) - _CALIBRATION_SOURCE_FIELDS)
    if unknown:
        raise SceneContractError(
            "scene field calibration_source permits only manifest and ply_sha256; "
            "found: " + ", ".join(unknown)
        )
    manifest = _require_file(
        source.get("manifest"), "calibration_source.manifest", base
    )
    wanted_hash = _expected_sha256(
        source.get("ply_sha256"), "calibration_source.ply_sha256"
    )
    try:
        source_spec = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SceneContractError(
            f"calibration_source manifest is unreadable: {manifest}"
        ) from exc
    if not isinstance(source_spec, dict):
        raise SceneContractError(
            f"calibration_source manifest must contain an object: {manifest}"
        )
    if source_spec.get("schema_version") != SCENE_SCHEMA_VERSION:
        raise SceneContractError(
            f"calibration_source {manifest} is not a schema v{SCENE_SCHEMA_VERSION} scene"
        )
    if "calibration_source" in source_spec:
        raise SceneContractError(
            f"calibration_source {manifest} is itself a variant; name the scene "
            "that was calibrated directly"
        )
    if source_spec.get("ply_sha256") != wanted_hash:
        raise SceneContractError(
            "scene field calibration_source.ply_sha256 does not match the "
            f"ply_sha256 declared by {manifest}"
        )
    # The cheap comparisons first, so a mismatched variant fails before the
    # source cloud is hashed.
    source_matrix, _, _ = _v2_transform(source_spec, manifest.parent)
    if not np.array_equal(source_matrix, matrix):
        raise SceneContractError(
            "scene transform differs from its calibration_source; a variant "
            "must keep its source's mat_w2gs exactly"
        )
    source_table_z = _declared_table_z(source_spec)
    if source_table_z != declared_table_z:
        raise SceneContractError(
            f"scene field calibration_table_z_m={declared_table_z:+.4f} m differs "
            f"from its calibration_source's {source_table_z:+.4f} m"
        )
    # The source vouches for the variant's registration only as a complete,
    # valid scene -- hashed cloud and transform, bound collision, and a table
    # plane that was actually measured.  A fixture-sized source skips that
    # measurement, and accepting it would let any cloud borrow its approval.
    source_report = _validate_v2_spec(source_spec, manifest=manifest)
    if source_report["gs_table_plane_z_m"] is None:
        raise SceneContractError(
            f"calibration_source {manifest} has too few vertices for its table "
            "plane to be measured, so it cannot vouch for another scene's "
            "registration"
        )
    return {
        "manifest": manifest,
        "ply_path": source_report["ply_path"],
        "ply_sha256": source_report["ply_sha256"],
        "gs_table_plane_z_m": source_report["gs_table_plane_z_m"],
    }

def _validate_v2_spec(
    spec: dict[str, Any],
    *,
    manifest: Path,
    preverified_ply_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate an in-memory v2 payload relative to its final scene path."""

    if spec.get("schema_version") != SCENE_SCHEMA_VERSION:
        raise SceneContractError(
            f"scene field schema_version must equal {SCENE_SCHEMA_VERSION}"
        )

    base = manifest.parent
    scene_id = _nonempty_string(spec.get("id"), "id")
    if not _ID_RE.fullmatch(scene_id):
        raise SceneContractError(
            "scene field id must use only letters, numbers, period, underscore, or dash"
        )
    name = _nonempty_string(spec.get("name"), "name")

    # These legacy aliases can change the point cloud or camera transform in
    # sim.py after this validator has approved ply_path / w2gs. A v2 contract
    # must express the complete alignment through its canonical fields only.
    conflicting_legacy_fields = sorted(
        set(spec)
        & (
            (set(_PLY_KEYS) - {"ply_path"})
            | (set(_W2GS_PATH_KEYS) - {"w2gs"})
            | _UNSAFE_LEGACY_ALIGNMENT_FIELDS
        )
    )
    if conflicting_legacy_fields:
        raise SceneContractError(
            "schema v2 scene contains transform / point-cloud aliases outside "
            "the canonical contract: "
            + ", ".join(conflicting_legacy_fields)
        )

    if "ply_path" not in spec:
        raise SceneContractError("schema v2 scene must define ply_path")
    ply_path = _require_scene_file(spec.get("ply_path"), "ply_path", base)
    expected_ply_hash = _expected_sha256(spec.get("ply_sha256"), "ply_sha256")
    if preverified_ply_sha256 is None:
        ply_hash = _sha256(ply_path)
    else:
        ply_hash = _expected_sha256(
            preverified_ply_sha256,
            "preverified_ply_sha256",
        )
    if ply_hash != expected_ply_hash:
        raise SceneContractError(
            "scene field ply_sha256 does not match "
            f"{ply_path}: expected {expected_ply_hash}, got {ply_hash}"
        )

    matrix, w2gs_path, w2gs_hash = _v2_transform(spec, base)

    # The hashes above prove the bytes; this proves the geometry: the verified
    # point cloud, mapped through the verified transform, must put the table
    # top where the declared value (and the z=0 table collision) says it is.
    # A counterfactual variant proves it on the clean scene it was derived
    # from instead; see _calibration_source.
    declared_table_z = _declared_table_z(spec)
    calibration = _calibration_source(
        spec,
        base=base,
        matrix=matrix,
        declared_table_z=declared_table_z,
    )
    measured_table_z = (
        _verify_table_z(declared_table_z, ply_path=ply_path, matrix=matrix)
        if calibration is None
        else calibration["gs_table_plane_z_m"]
    )

    table = _mapping(spec.get("table"), "table")
    unsupported_table_fields = sorted(set(table) - {"appearance", "collision"})
    if unsupported_table_fields:
        raise SceneContractError(
            "scene field table permits only appearance and collision; found: "
            + ", ".join(unsupported_table_fields)
        )
    if table.get("appearance") != "embedded_in_3dgs":
        raise SceneContractError(
            "scene field table.appearance must equal embedded_in_3dgs"
        )
    collision_directory, collision_files, collision_tree_hash = _collision_contract(
        table.get("collision"),
        base=base,
    )

    digest_payload = {
        "manifest": spec,
        "resolved_content": {
            "ply_sha256": ply_hash,
            "w2gs_sha256": w2gs_hash,
            "w2gs_matrix": matrix.tolist(),
            "collision_tree_sha256": collision_tree_hash,
        },
    }
    if calibration is not None:
        digest_payload["resolved_content"]["calibration_source_ply_sha256"] = (
            calibration["ply_sha256"]
        )
    contract_digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SCENE_SCHEMA_VERSION,
        "id": scene_id,
        "name": name,
        "manifest_path": manifest,
        "background_dir": base,
        "ply_path": ply_path,
        "ply_sha256": ply_hash,
        "w2gs_path": w2gs_path,
        "w2gs_matrix": matrix,
        "calibration_table_z_m": declared_table_z,
        # None only for fixture point clouds below TABLE_Z_MIN_VERTICES.
        "gs_table_plane_z_m": measured_table_z,
        "calibration_source_manifest": (
            None if calibration is None else calibration["manifest"]
        ),
        "collision_directory": collision_directory,
        "collision_files": collision_files,
        "collision_tree_sha256": collision_tree_hash,
        "contract_digest": contract_digest,
    }


def resolve_table_collision(
    scene_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Resolve and verify only the table collision bound by one v2 scene.

    ``validate_scene_contract`` proves the whole scene, which means hashing a
    multi-gigabyte point cloud.  Callers that need nothing but the MuJoCo table
    collision -- the geometry that defines where the tabletop physically is --
    pay for that verification here instead: the declared directory is resolved
    (``${REAL2SIM_INPUTS_ROOT}`` and other environment templates expand exactly
    as they do in the full contract, and a relative path stays relative to the
    scene directory), the glob is matched, the file count is checked, and the
    tree digest must match.  Every failure raises; nothing falls back.

    Returns ``directory`` / ``files`` / ``tree_sha256`` for the selected files.
    """

    manifest = _manifest_path(scene_path)
    try:
        spec = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SceneContractError(
            f"scene background.json is unreadable: {manifest}"
        ) from exc
    if not isinstance(spec, dict):
        raise SceneContractError(
            f"scene background.json must contain an object: {manifest}"
        )
    if spec.get("schema_version") != SCENE_SCHEMA_VERSION:
        raise SceneContractError(
            f"scene {manifest} is not schema v{SCENE_SCHEMA_VERSION}; the table "
            "collision binding only exists in schema v2"
        )
    table = _mapping(spec.get("table"), "table")
    directory, files, tree_hash = _collision_contract(
        table.get("collision"),
        base=manifest.parent,
    )
    return {
        "manifest_path": manifest,
        "directory": directory,
        "files": files,
        "tree_sha256": tree_hash,
    }


def build_scene_contract(
    scene_path: str | os.PathLike[str],
    *,
    table_collision_directory: str | os.PathLike[str],
    scene_id: str | None = None,
    collision_glob: str = "*.obj",
    calibration_world_offset: Iterable[float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Upgrade one legacy background definition to a complete v2 payload.

    This function is side-effect free. Use the module CLI with ``--write`` to
    replace ``background.json`` after reviewing the printed payload.
    """

    manifest = _manifest_path(scene_path)
    try:
        source = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SceneContractError(f"scene background.json is unreadable: {manifest}") from exc
    if not isinstance(source, dict):
        raise SceneContractError("scene background.json must contain an object")
    source_schema = source.get("schema_version", 1)
    if source_schema == SCENE_SCHEMA_VERSION:
        raise SceneContractError(
            "scene already uses schema v2; validate it instead of upgrading it"
        )
    if source_schema != 1:
        raise SceneContractError(
            f"cannot upgrade unsupported scene schema_version {source_schema!r}"
        )
    base = manifest.parent

    unsafe_alignment_fields = sorted(
        set(source) & _UNSAFE_LEGACY_ALIGNMENT_FIELDS
    )
    if unsafe_alignment_fields:
        raise SceneContractError(
            "legacy scene uses runtime alignment fields that cannot be safely "
            "migrated; bake them into one mat_w2gs first: "
            + ", ".join(unsafe_alignment_fields)
        )

    try:
        calibration_offset = np.asarray(
            tuple(calibration_world_offset),
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise SceneContractError(
            "calibration_world_offset must contain three finite numbers"
        ) from exc
    if calibration_offset.shape != (3,) or not np.isfinite(calibration_offset).all():
        raise SceneContractError(
            "calibration_world_offset must contain three finite numbers"
        )

    ply_key, ply_value = _pick(source, _PLY_KEYS)
    if ply_key is None:
        raise SceneContractError("scene has no point-cloud path to upgrade")
    ply_path = _require_file(ply_value, ply_key, base)
    ply_hash = _sha256(ply_path)

    inline_value = source.get("mat_w2gs")
    w2gs_key, w2gs_value = _pick(source, _W2GS_PATH_KEYS)
    if inline_value not in (None, "") and w2gs_key is not None:
        raise SceneContractError("scene defines both mat_w2gs and a w2gs file")
    if inline_value not in (None, ""):
        raw_w2gs = _validate_transform(
            _inline_matrix(inline_value, "mat_w2gs"),
            "mat_w2gs",
        )
        w2gs_path = None
    elif w2gs_key is not None:
        w2gs_path = _require_file(w2gs_value, w2gs_key, base)
        raw_w2gs = _validate_transform(
            _loaded_matrix(w2gs_path, w2gs_key),
            w2gs_key,
        )
    else:
        raise SceneContractError("scene has no world-to-GS transform to upgrade")
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = calibration_offset
    effective_w2gs = _validate_transform(
        raw_w2gs @ translation,
        "effective mat_w2gs",
    )
    # Measure with the EFFECTIVE matrix: this is the transform the contract
    # will carry, so the recorded value is exactly what validation re-measures.
    # Fixture point clouds (below TABLE_Z_MIN_VERTICES) cannot be measured and
    # record the convention value 0.0, which validation accepts unmeasured.
    if ply_vertex_count(ply_path) >= TABLE_Z_MIN_VERTICES:
        calibration_table_z = measure_gs_table_plane_z(ply_path, effective_w2gs)
    else:
        calibration_table_z = TABLE_Z_RUNTIME_M

    raw_collision_directory = os.fspath(table_collision_directory)
    collision_directory = Path(
        os.path.expandvars(raw_collision_directory)
    ).expanduser().resolve()
    if not collision_directory.is_dir():
        raise SceneContractError(
            f"table collision directory does not exist: {collision_directory}"
        )
    pattern_path = Path(collision_glob)
    if pattern_path.is_absolute() or ".." in pattern_path.parts:
        raise SceneContractError("table collision glob must stay inside its directory")
    collision_files = tuple(
        sorted(
            path.resolve()
            for path in collision_directory.glob(collision_glob)
            if path.is_file()
        )
    )
    if not collision_files:
        raise SceneContractError(
            f"table collision glob matched no files: {collision_directory / collision_glob}"
        )
    collision_tree_hash = compute_tree_sha256(
        collision_directory,
        collision_files,
    )

    name = _nonempty_string(source.get("name") or base.name, "name")
    if scene_id is None:
        id_prefix = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._") or "scene"
        content_identity = hashlib.sha256(
            json.dumps(
                {
                    "ply_sha256": ply_hash,
                    "w2gs_matrix": effective_w2gs.tolist(),
                    "collision_tree_sha256": collision_tree_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        scene_id = f"{id_prefix}-{content_identity[:12]}"
    scene_id = _nonempty_string(scene_id, "id")
    if not _ID_RE.fullmatch(scene_id):
        raise SceneContractError(
            "scene id must use only letters, numbers, period, underscore, or dash"
        )

    payload = dict(source)
    for key in set(_PLY_KEYS) | set(_W2GS_PATH_KEYS) | {
        "schema_version",
        "id",
        "ply_sha256",
        "w2gs_sha256",
        "calibration_table_z_m",
        "calibration_table_z_method",
        "table",
    }:
        payload.pop(key, None)
    payload.update(
        {
            "schema_version": SCENE_SCHEMA_VERSION,
            "id": scene_id,
            "name": name,
            "ply_path": str(ply_value),
            "ply_sha256": ply_hash,
        }
    )
    if np.any(calibration_offset != 0.0):
        # sim.py historically applied M_effective = M_raw @ Translation(offset)
        # at runtime. Official v2 runs forbid that independent adjustment, so
        # migration bakes the identical effective transform into the contract.
        payload["mat_w2gs"] = effective_w2gs.tolist()
    elif inline_value not in (None, ""):
        payload["mat_w2gs"] = raw_w2gs.tolist()
    else:
        assert w2gs_path is not None
        payload["w2gs"] = str(w2gs_value)
        payload["w2gs_sha256"] = _sha256(w2gs_path)
    payload["calibration_table_z_m"] = float(calibration_table_z)
    payload["calibration_table_z_method"] = TABLE_Z_METHOD
    try:
        collision_value = collision_directory.relative_to(base).as_posix()
    except ValueError:
        # External collision is artifact-bound by the launcher. Preserve an
        # explicit environment reference so the scene release remains
        # portable; otherwise record an unambiguous absolute path rather than
        # a fragile ../../ traversal tied to the current directory depth.
        collision_value = (
            raw_collision_directory
            if "$" in raw_collision_directory
            else str(collision_directory)
        )
    payload["table"] = {
        "appearance": "embedded_in_3dgs",
        "collision": {
            "directory": collision_value,
            "glob": collision_glob,
            "count": len(collision_files),
            "tree_sha256": collision_tree_hash,
            "frame": "mujoco_world",
        },
    }
    # Validate the generated bytes against the original manifest directory
    # before callers can replace background.json. This catches content races,
    # bad relative paths, and incomplete conversion without risking the only
    # legacy manifest.
    _validate_v2_spec(payload, manifest=manifest)
    return payload


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create or validate one exact 3DGS + table scene contract."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("scene")
    validate_parser.add_argument("--legacy", action="store_true")
    upgrade_parser = subparsers.add_parser("upgrade")
    upgrade_parser.add_argument("scene")
    upgrade_parser.add_argument("--table-collision-dir", required=True)
    upgrade_parser.add_argument("--id")
    upgrade_parser.add_argument("--collision-glob", default="*.obj")
    upgrade_parser.add_argument(
        "--calibration-world-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help=(
            "legacy translation applied before world-to-GS; it is baked into "
            "the schema-v2 mat_w2gs"
        ),
    )
    upgrade_parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "validate":
        report = validate_scene_contract(args.scene, strict=not args.legacy)
        print(
            json.dumps(
                {
                    "id": report["id"],
                    "name": report["name"],
                    "manifest": str(report["manifest_path"]),
                    "contract_digest": report["contract_digest"],
                    "table_collision_parts": len(report["collision_files"]),
                    "calibration_table_z_m": report["calibration_table_z_m"],
                    "gs_table_plane_z_m": report["gs_table_plane_z_m"],
                },
                indent=2,
            )
        )
        return 0

    payload = build_scene_contract(
        args.scene,
        table_collision_directory=args.table_collision_dir,
        scene_id=args.id,
        collision_glob=args.collision_glob,
        calibration_world_offset=args.calibration_world_offset,
    )
    manifest = _manifest_path(args.scene)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.write:
        temporary = manifest.with_name(f".{manifest.name}.tmp-{os.getpid()}")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, manifest)
        validate_scene_contract(manifest, strict=True)
        print(f"wrote {manifest}")
    else:
        print(rendered, end="")
    return 0


__all__ = [
    "SCENE_SCHEMA_VERSION",
    "TABLE_Z_METHOD",
    "TABLE_Z_MIN_VERTICES",
    "TABLE_Z_RUNTIME_M",
    "TABLE_Z_TOLERANCE_M",
    "SceneContractError",
    "build_scene_contract",
    "compute_tree_sha256",
    "measure_gs_table_plane_z",
    "ply_vertex_count",
    "resolve_table_collision",
    "scene_tag",
    "validate_scene_contract",
]


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except SceneContractError as exc:
        raise SystemExit(f"scene contract error: {exc}") from exc
