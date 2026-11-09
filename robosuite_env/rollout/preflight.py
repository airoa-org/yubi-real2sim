# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Validate rollout assets and official-run provenance before MuJoCo starts.

This module owns the fail-closed startup checks that must finish before the
worker constructs an environment.  Keeping them together makes their order
explicit and leaves :mod:`robosuite_env.rollout.worker` focused on execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import sys
from typing import Any, Sequence

import numpy as np

from robosuite_env.contracts.assets import (
    runtime_bundle_ids,
    validate_repository_assets,
)
from robosuite_env.contracts.scene import validate_scene_contract
from robosuite_env.contracts.worker import validate_worker_contract
from robosuite_env.paths import PACKAGE_ROOT
from robosuite_env.rollout.cli import DEFAULT_GS_PLY, LEGACY_GS_ALIGN


__all__ = [
    "argv_has",
    "resolve_gs_background",
    "RuntimePreflight",
    "bind_validated_gs_registration",
    "reported_gs_w2gs",
    "run_runtime_preflight",
]


_DIR = str(PACKAGE_ROOT)

_GS_OFF_VALUES = {"", "none", "off", "false", "0", "no"}
_GS_COMPOSITE_MODES = {"depth", "foreground"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

def argv_has(*names):
    for arg in sys.argv[1:]:
        for name in names:
            if arg == name or arg.startswith(name + "="):
                return True
    return False

def _gs_is_off(value):
    return value is not None and str(value).strip().lower() in _GS_OFF_VALUES

def _gs_path(value, base_dir):
    s = os.path.expandvars(os.path.expanduser(str(value)))
    if not os.path.isabs(s) and base_dir:
        s = os.path.normpath(os.path.join(base_dir, s))
    return s

def _gs_pick(d, *keys):
    if not isinstance(d, dict):
        return None
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_gs_alignment_sha256(spec, path, source):
    align = _gs_pick(spec, "alignment", "transform")
    align_dict = align if isinstance(align, dict) else {}
    expected = _gs_pick(align_dict, "w2gs_sha256", "sha256")
    if expected is None:
        expected = _gs_pick(spec, "w2gs_sha256")
    if expected is None:
        return None
    if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
        raise SystemExit(
            f"[sim] GS background {source} w2gs_sha256 must be lowercase SHA-256"
        )
    try:
        actual = _sha256_file(path)
    except OSError as exc:
        raise SystemExit(
            f"[sim] GS background {source} cannot hash world->GS matrix {path}: {exc}"
        ) from exc
    if actual != expected:
        raise SystemExit(
            f"[sim] GS background {source} world->GS SHA-256 mismatch: "
            f"expected {expected}, got {actual} ({path})"
        )
    return actual


def _gs_composite_mode(value, source):
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "z_test": "depth",
        "ztest": "depth",
        "foreground_mask": "foreground",
        "mask": "foreground",
    }
    mode = aliases.get(mode, mode)
    if mode not in _GS_COMPOSITE_MODES:
        raise SystemExit(
            f"[sim] GS background {source} has invalid composite_mode "
            f"{value!r}; expected depth or foreground"
        )
    return mode

def _load_gs_index(index_path):
    if not os.path.isfile(index_path):
        return {}
    try:
        with open(index_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[sim] warning: could not read GS background index {index_path}: {e}", flush=True)
        return {}
    raw = data.get("backgrounds", data) if isinstance(data, dict) else data
    entries = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("id")
            if not name:
                continue
            entries[str(name)] = dict(item)
            for alias in item.get("aliases", []):
                entries[str(alias)] = dict(item)
    elif isinstance(raw, dict):
        for name, item in raw.items():
            if isinstance(item, str):
                spec = {"name": name, "ply": item}
            elif isinstance(item, dict):
                spec = dict(item)
                spec.setdefault("name", name)
            else:
                continue
            entries[str(name)] = spec
            for alias in spec.get("aliases", []):
                entries[str(alias)] = spec
    return entries

def _builtin_gs_preset(name):
    if str(name).strip().lower() in {"default", "standard", "img1239", "1400"}:
        return {
            "name": "standard",
            "ply": DEFAULT_GS_PLY,
            "w2gs": LEGACY_GS_ALIGN,
            "port": 8097,
        }
    return None

def _find_gs_preset(name, index_path):
    entries = _load_gs_index(index_path)
    if name in entries:
        return entries[name], os.path.dirname(index_path), f"preset:{name}"
    lname = str(name).lower()
    for key, spec in entries.items():
        if str(key).lower() == lname:
            return spec, os.path.dirname(index_path), f"preset:{key}"
    spec = _builtin_gs_preset(name)
    if spec is not None:
        return spec, _DIR, f"builtin:{spec['name']}"
    return None, None, None

def _gs_matrix_or_path(value):
    if value is None:
        return None, None
    if isinstance(value, str):
        return None, value
    arr = np.asarray(value, dtype=float)
    if arr.shape != (4, 4):
        raise SystemExit(f"[sim] GS world->GS matrix must be 4x4, got {arr.shape}")
    return arr, None

def _extract_gs_alignment(spec, base_dir):
    align = _gs_pick(spec, "alignment", "transform")
    align_dict = align if isinstance(align, dict) else {}
    candidates = []
    if isinstance(align, str):
        candidates.append(align)
    candidates.extend([
        _gs_pick(align_dict, "mat_w2gs", "w2gs_matrix", "world_to_gs_matrix", "matrix"),
        _gs_pick(align_dict, "w2gs", "gs_w2gs", "mat_w2gs_path", "world_to_gs", "world_to_gs_path", "path"),
        _gs_pick(spec, "mat_w2gs", "w2gs_matrix", "world_to_gs_matrix"),
        _gs_pick(spec, "w2gs", "gs_w2gs", "mat_w2gs_path", "world_to_gs", "world_to_gs_path"),
    ])
    for value in candidates:
        mat, path = _gs_matrix_or_path(value)
        if mat is not None:
            return mat, ""
        if path:
            return None, _gs_path(path, base_dir)
    if bool(_gs_pick(spec, "use_legacy_alignment", "legacy_alignment")) and os.path.isfile(LEGACY_GS_ALIGN):
        return None, LEGACY_GS_ALIGN
    return None, ""

def _extract_gs_rotation(spec):
    align = _gs_pick(spec, "alignment", "transform")
    align_dict = align if isinstance(align, dict) else {}
    rot_R = _gs_pick(align_dict, "rot_R", "rotation_matrix")
    if rot_R is None:
        rot_R = _gs_pick(spec, "rot_R", "rotation_matrix")
    if rot_R is not None:
        arr = np.asarray(rot_R, dtype=float)
        if arr.shape != (3, 3):
            raise SystemExit(f"[sim] GS rotation matrix must be 3x3, got {arr.shape}")
        return "", arr
    rot = _gs_pick(align_dict, "rot", "gs_rot", "rotation")
    if rot is None:
        rot = _gs_pick(spec, "rot", "gs_rot", "rotation")
    return ("" if rot is None else str(rot)), None

def _extract_gs_pivot(spec):
    align = _gs_pick(spec, "alignment", "transform")
    align_dict = align if isinstance(align, dict) else {}
    piv = _gs_pick(align_dict, "pivot", "rot_piv", "rotation_pivot", "gs_pivot")
    if piv is None:
        piv = _gs_pick(spec, "pivot", "rot_piv", "rotation_pivot", "gs_pivot")
    if piv is not None:
        arr = np.asarray(piv, dtype=float)
        if arr.shape != (3,):
            raise SystemExit(f"[sim] GS rotation pivot must have 3 values, got {arr.shape}")
        return arr, float(arr[2])
    pivz = _gs_pick(align_dict, "pivz", "gs_pivz", "pivot_z", "rot_pivz")
    if pivz is None:
        pivz = _gs_pick(spec, "pivz", "gs_pivz", "pivot_z", "rot_pivz")
    if pivz is not None:
        z = float(pivz)
        return np.array([0.0, 0.0, z]), z
    return None, None

def _apply_gs_definition(a, spec, base_dir, source):
    if isinstance(spec, str):
        spec = {"ply": spec}
    if not isinstance(spec, dict):
        raise SystemExit(f"[sim] GS background {source} must be a JSON object or .ply path")
    if spec.get("disabled") is True or spec.get("enabled") is False:
        a.gs = ""; a.gs_background_source = source; return
    point_cloud = _gs_pick(spec, "point_cloud")
    point_cloud = point_cloud if isinstance(point_cloud, dict) else {}
    ply = (_gs_pick(spec, "ply", "gs", "gs_ply", "point_cloud_ply", "ply_path", "path")
           or _gs_pick(point_cloud, "ply", "path"))
    if _gs_is_off(ply):
        a.gs = ""; a.gs_background_source = source; return
    if not ply:
        raise SystemExit(f"[sim] GS background {source} does not define a ply path")
    a.gs = _gs_path(ply, base_dir)
    a.gs_background_source = source
    a.gs_background_name = str(_gs_pick(spec, "name", "id") or "")
    if not argv_has("--gs-composite-mode"):
        composite_mode = _gs_pick(
            spec,
            "composite_mode",
            "gs_composite_mode",
        )
        if composite_mode is not None:
            a.gs_composite_mode = _gs_composite_mode(
                composite_mode,
                source,
            )

    server = _gs_pick(spec, "server")
    server = server if isinstance(server, dict) else {}
    if not argv_has("--gs-port"):
        port = _gs_pick(spec, "port", "gs_port") or _gs_pick(server, "port")
        if port is not None:
            a.gs_port = int(port)
    if not argv_has("--gs-rot"):
        rot, rot_R = _extract_gs_rotation(spec)
        if rot_R is not None:
            a.gs_rot_R = rot_R
            a.gs_rot = ""
        elif rot:
            a.gs_rot = rot
    if not argv_has("--gs-pivz", "--gs-pivot"):
        pivot, pivz = _extract_gs_pivot(spec)
        if pivot is not None:
            a.gs_pivot = pivot
        if pivz is not None:
            a.gs_pivz = pivz
    elif argv_has("--gs-pivot"):
        a.gs_pivot = np.asarray(a.gs_pivot, dtype=float)
    else:
        a.gs_pivot = np.array([0.0, 0.0, float(a.gs_pivz)])

    if not argv_has("--gs-transform"):
        mat, path = _extract_gs_alignment(spec, base_dir)
        expected_alignment_sha256 = _gs_pick(spec, "w2gs_sha256")
        if mat is not None and expected_alignment_sha256 is not None:
            raise SystemExit(
                f"[sim] GS background {source} inline world->GS matrix must not "
                "define file-only w2gs_sha256"
            )
        if path:
            _verify_gs_alignment_sha256(spec, path, source)
        a.gs_mat_w2gs = mat
        a.gs_w2gs = path
    if argv_has("--gs"):
        cli_gs = getattr(a, "_gs_cli_value", a.gs)
        a.gs = "" if _gs_is_off(cli_gs) else _gs_path(cli_gs, os.getcwd())

def resolve_gs_background(a):
    a._gs_cli_value = a.gs
    a.gs_background_source = "--gs"
    a.gs_background_name = ""
    a.gs_mat_w2gs = None
    a.gs_w2gs = LEGACY_GS_ALIGN if os.path.isfile(LEGACY_GS_ALIGN) else ""
    a.gs_rot_R = None
    if argv_has("--gs-transform"):
        a.gs_w2gs = "" if _gs_is_off(a.gs_transform) else _gs_path(a.gs_transform, os.getcwd())
    if a.gs_pivot is not None:
        a.gs_pivot = np.asarray(a.gs_pivot, dtype=float)

    if a.gs_background is None:
        if _gs_is_off(a.gs):
            a.gs = ""
            a.gs_w2gs = ""
            a.gs_background_source = "disabled (--gs)"
        return

    value = str(a.gs_background).strip()
    if _gs_is_off(value):
        a.gs = ""
        a.gs_w2gs = ""
        a.gs_background_source = "disabled (--gs-background)"
        return
    if value.startswith("{"):
        try:
            spec = json.loads(value)
        except json.JSONDecodeError as e:
            raise SystemExit(f"[sim] invalid --gs-background JSON: {e}") from e
        _apply_gs_definition(a, spec, os.getcwd(), "inline-json")
    elif os.path.isdir(_gs_path(value, os.getcwd())):
        path = os.path.join(_gs_path(value, os.getcwd()), "background.json")
        if not os.path.isfile(path):
            raise SystemExit(f"[sim] GS background directory has no background.json: {path}")
        with open(path, "r") as f:
            spec = json.load(f)
        _apply_gs_definition(a, spec, os.path.dirname(path), path)
    elif value.lower().endswith(".json"):
        path = _gs_path(value, os.getcwd())
        if not os.path.isfile(path):
            raise SystemExit(f"[sim] GS background JSON not found: {path}")
        with open(path, "r") as f:
            spec = json.load(f)
        _apply_gs_definition(a, spec, os.path.dirname(path), path)
    elif value.lower().endswith(".ply") or os.path.sep in value or (os.path.altsep and os.path.altsep in value):
        _apply_gs_definition(a, {"ply": value}, os.getcwd(), "ply-path")
    else:
        index_path = _gs_path(a.gs_background_index, os.getcwd())
        spec, base_dir, source = _find_gs_preset(value, index_path)
        if spec is None:
            idx = index_path if os.path.isfile(index_path) else "(index missing)"
            raise SystemExit(f"[sim] unknown GS background preset {value!r}; checked {idx}")
        _apply_gs_definition(a, spec, base_dir, source)


@dataclass(frozen=True)
class RuntimePreflight:
    """Validated runtime inputs consumed by rollout execution and reporting."""

    asset_contract_report: dict[str, Any]
    scene_contract_report: dict[str, Any] | None
    official_run_manifest_path: str | None
    official_worker_contract_path: str | None
    execution_class: str


def _argv_has(argv: Sequence[str], *names: str) -> bool:
    return any(
        argument == name or argument.startswith(name + "=")
        for argument in argv
        for name in names
    )


def bind_validated_gs_registration(
    arguments: Any, scene_contract_report: dict[str, Any]
) -> None:
    """Hand the worker the validated world-to-GS matrix and its file identity.

    The worker consumes only the matrix that passed contract validation, so
    storage details stay inside the contract loader.  A file-form contract
    must still keep its registered path in ``gs_w2gs``: the summary reports
    that path, and post-hoc rerendering compares the generation record with
    the current scene by registration kind, path and SHA-256, not only by
    matrix value, so collapsing a file-form scene into an inline record would
    make such a rerender fail with "GS registration source differs".
    """

    arguments.gs_mat_w2gs = np.asarray(
        scene_contract_report["w2gs_matrix"],
        dtype=float,
    )
    w2gs_path = scene_contract_report.get("w2gs_path")
    arguments.gs_w2gs = (
        "" if w2gs_path is None else os.path.realpath(os.fspath(w2gs_path))
    )


def reported_gs_w2gs(arguments: Any) -> str | None:
    """Name the GS registration source the worker used, as the summary reports it.

    A bound file path wins over the inline sentinel because the validated
    matrix may have been loaded from that file by the scene contract.
    """

    if arguments.gs_w2gs:
        return arguments.gs_w2gs
    return "inline-matrix" if arguments.gs_mat_w2gs is not None else None


def run_runtime_preflight(
    arguments: Any,
    *,
    worker_argv: Sequence[str],
    package_root: str | os.PathLike[str],
) -> RuntimePreflight:
    """Resolve the GS scene and validate every asset contract in startup order."""

    resolve_gs_background(arguments)

    try:
        runtime_asset_bundle_ids = runtime_bundle_ids(
            arguments.robot,
            arguments.objects,
        )
        asset_contract_report = validate_repository_assets(
            os.path.join(os.fspath(package_root), "assets"),
            bundle_ids=runtime_asset_bundle_ids,
        )
    except Exception as exc:
        raise SystemExit(f"[sim] repository asset preflight failed: {exc}") from exc

    scene_contract_report = None
    official_run_manifest_path = None
    official_worker_contract_path = None
    execution_class = "diagnostic"

    if arguments.strict_assets:
        try:
            forbidden_gs_overrides = tuple(
                option
                for option in (
                    "--gs",
                    "--gs-transform",
                    "--gs-rot",
                    "--gs-pivot",
                    "--gs-pivz",
                )
                if _argv_has(worker_argv, option)
            )
            if forbidden_gs_overrides:
                raise ValueError(
                    "official simulation forbids GS transform / point-cloud "
                    "overrides outside --gs-background: "
                    + ", ".join(forbidden_gs_overrides)
                )
            if np.any(
                np.asarray(arguments.gs_calibration_world_offset, dtype=float) != 0.0
            ):
                raise ValueError(
                    "official simulation requires --gs-calibration-world-offset "
                    "0 0 0; bake registration into the scene contract's w2gs"
                )
            if arguments.gs_rot or arguments.gs_rot_R is not None:
                raise ValueError(
                    "official scene applies an extra GS rotation outside w2gs; "
                    "bake it into the validated scene transform"
                )

            official_manifest_path = os.path.realpath(
                os.path.expanduser(arguments.official_run_manifest)
            )
            if (
                not arguments.official_run_manifest
                or not os.path.isfile(official_manifest_path)
            ):
                raise ValueError(
                    "--strict-assets requires the launcher's --official-run-manifest"
                )

            raw_worker_argv = list(worker_argv)
            worker_contract_occurrences = [
                index
                for index, value in enumerate(raw_worker_argv)
                if value == "--official-worker-contract"
                or value.startswith("--official-worker-contract=")
            ]
            if len(worker_contract_occurrences) != 1:
                raise ValueError(
                    "official simulator requires exactly one --official-worker-contract"
                )
            contract_index = worker_contract_occurrences[0]
            if raw_worker_argv[contract_index] != "--official-worker-contract":
                raise ValueError(
                    "official worker contract must use a separate path argument"
                )
            if contract_index + 1 >= len(raw_worker_argv):
                raise ValueError("--official-worker-contract requires a path")
            declared_contract_path = raw_worker_argv[contract_index + 1]
            if os.path.realpath(declared_contract_path) != os.path.realpath(
                arguments.official_worker_contract
            ):
                raise ValueError(
                    "parsed official worker contract path differs from argv"
                )
            contract_argv = (
                raw_worker_argv[:contract_index]
                + raw_worker_argv[contract_index + 2 :]
            )
            validate_worker_contract(
                arguments.official_worker_contract,
                official_manifest_path,
                contract_argv,
            )

            try:
                with open(official_manifest_path, encoding="utf-8") as stream:
                    official_manifest = json.load(stream)
                official_contracts = official_manifest["preflight"]["contracts"]
                expected_asset_digest = official_contracts["assets"][
                    "contract_digest"
                ]
                expected_asset_bundles = tuple(
                    official_contracts["assets"]["bundles"]
                )
                if not arguments.official_scene_tag:
                    raise KeyError("--official-scene-tag")
                expected_scene = official_contracts["scenes"][
                    arguments.official_scene_tag
                ]
                expected_scene_id = expected_scene["id"]
                expected_scene_manifest = os.path.realpath(
                    os.path.expanduser(expected_scene["manifest_path"])
                )
                expected_scene_digest = expected_scene["contract_digest"]
                expected_table_digest = expected_scene["collision_tree_sha256"]
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
            ) as exc:
                raise ValueError(
                    "official run manifest lacks validated contracts: "
                    f"{official_manifest_path}"
                ) from exc

            if runtime_asset_bundle_ids != expected_asset_bundles:
                raise ValueError(
                    "runtime asset bundle selection differs from launcher preflight: "
                    f"runtime={runtime_asset_bundle_ids}, "
                    f"launcher={expected_asset_bundles}"
                )

            if arguments.gs_background is None:
                raise ValueError(
                    "official simulation requires an exact --gs-background directory "
                    "or background.json"
                )
            if not arguments.gs_server_token:
                raise ValueError(
                    "official simulation requires the batch runner's "
                    "--gs-server-token"
                )
            scene_contract_report = validate_scene_contract(
                arguments.gs_background,
                strict=True,
            )
            if os.path.realpath(arguments.gs) != os.path.realpath(
                os.fspath(scene_contract_report["ply_path"])
            ):
                raise ValueError(
                    "resolved GS point cloud differs from the validated scene contract"
                )

            bind_validated_gs_registration(arguments, scene_contract_report)
            if arguments.desk != "scene":
                raise ValueError(
                    "official simulation requires --desk scene; table appearance and "
                    "collision are selected together by --gs-background"
                )
            if asset_contract_report["contract_digest"] != expected_asset_digest:
                raise ValueError(
                    "repository assets changed after launcher preflight"
                )
            actual_scene_manifest = os.path.realpath(
                os.fspath(scene_contract_report["manifest_path"])
            )
            if actual_scene_manifest != expected_scene_manifest:
                raise ValueError(
                    f"worker scene {arguments.official_scene_tag!r} must use "
                    f"{expected_scene_manifest}, got {actual_scene_manifest}"
                )
            if scene_contract_report["id"] != expected_scene_id:
                raise ValueError("3DGS scene id changed after launcher preflight")
            if scene_contract_report["contract_digest"] != expected_scene_digest:
                raise ValueError("3DGS scene changed after launcher preflight")
            if (
                scene_contract_report["collision_tree_sha256"]
                != expected_table_digest
            ):
                raise ValueError("table collision changed after launcher preflight")

            official_run_manifest_path = official_manifest_path
            official_worker_contract_path = os.path.realpath(
                arguments.official_worker_contract
            )
            execution_class = "official"
            arguments.desk_world_aligned = True
        except Exception as exc:
            raise SystemExit(f"[sim] strict asset preflight failed: {exc}") from exc

    return RuntimePreflight(
        asset_contract_report=asset_contract_report,
        scene_contract_report=scene_contract_report,
        official_run_manifest_path=official_run_manifest_path,
        official_worker_contract_path=official_worker_contract_path,
        execution_class=execution_class,
    )
