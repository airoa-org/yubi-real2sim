# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""GS background server lifecycle, environments, and scene identity."""


from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from robosuite_env.contracts.scene import (
    SceneContractError,
    validate_scene_contract,
)
# ``stop_process`` is re-exported under its historical private name: this
# module is where ``rerender.run`` and ``rerender.__init__`` import
# ``_stop_process`` from.
from robosuite_env.gs_launch import (  # noqa: F401
    GsServerError,
    launch_gs_server,
    stop_process as _stop_process,
    with_loopback_no_proxy,
)

from robosuite_env.rerender.common import (
    GpuBinding,
    REPO_ROOT,
    RerenderError,
    SourceRun,
    _GS_OFF_VALUES,
    _last_option_value,
    _load_json_object,
    _resolved_recorded_path,
)
from robosuite_env.rerender.gpu import _coerce_gpu_binding, resolve_cuda_device
from robosuite_env.rerender.provenance import (
    _assert_record_current,
    _file_record,
    _file_state,
)
from robosuite_env.rerender.registration import (
    _assert_registration_matches,
    _load_registration_file,
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _scene_ply(scene_dir: Path) -> Path:
    background = json.loads(
        (scene_dir / "background.json").read_text(encoding="utf-8")
    )
    ply = Path(background["ply_path"])
    if not ply.is_absolute():
        ply = scene_dir / ply
    ply = ply.resolve()
    if not ply.is_file():
        raise RerenderError(f"scene ply does not exist: {ply}")
    return ply


def gs_server_environment(gpu: int | GpuBinding) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = (
        gpu.cuda_token if isinstance(gpu, GpuBinding) else resolve_cuda_device(gpu)
    )
    # The server owns one serialized CUDA renderer. Large inherited CPU pools
    # only oversubscribe the host when several GPU services coexist.
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        environment[name] = "1"
    return environment


def replay_worker_environment(
    gpu: int | GpuBinding, *, recorded: dict[str, str] | None = None
) -> dict[str, str]:
    """Build the split CUDA/EGL environment for a state-only replay worker.

    The GS server owns CUDA.  The replay child needs only MuJoCo EGL, so hide
    CUDA from that process and pass the exact host physical EGL ordinal.  An
    empty CUDA namespace also bypasses robosuite's invalid UUID/numeric import
    assertion while preventing an accidental policy/JAX CUDA initialization.
    Recorded rollout environments are merged first and then the device and GL
    contract is overwritten fail-closed.
    """

    binding = _coerce_gpu_binding(gpu)
    environment = dict(os.environ)
    if recorded is not None:
        environment.update(recorded)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["MUJOCO_EGL_DEVICE_ID"] = str(binding.egl_physical_index)
    environment["MUJOCO_GL"] = "egl"
    environment["PYOPENGL_PLATFORM"] = "egl"
    environment.pop("REAL2SIM_DEFERRED_MUJOCO_EGL_DEVICE_ID", None)
    environment["OPENCV_FOR_THREADS_NUM"] = "1"
    environment["INIT_STATE_JSON"] = ""
    environment["MIMICGEN_INIT_EPISODE"] = ""
    environment["MIMICGEN_INIT_STATE_LOG_ROOT"] = ""
    environment["REAL2SIM_OFFICIAL_RUN_MANIFEST"] = ""
    # The worker's GS client reaches this server over loopback with urllib;
    # an inherited proxy would otherwise receive every render request.
    return with_loopback_no_proxy(environment)


def _start_gs_server(
    python: str,
    ply: Path,
    port: int,
    token: str,
    log_path: Path,
    gpu: int | GpuBinding,
    *,
    render_lock: str = "",
) -> tuple[subprocess.Popen, dict[str, str]]:
    """Start the scene's GS server and return it once it proves its identity.

    The argv, the ready deadline, and the four-part ``/health`` check live in
    :mod:`robosuite_env.gs_launch` so the rollout worker's fallback launcher
    cannot drift away from them again.  Only the failure type is translated
    here: every caller in this package already handles
    :class:`~robosuite_env.rerender.common.RerenderError`.
    """

    try:
        return launch_gs_server(
            python=python,
            ply=ply,
            port=port,
            ready_token=token,
            log_path=log_path,
            env=gs_server_environment(gpu),
            cwd=REPO_ROOT,
            render_lock=render_lock,
        )
    except GsServerError as exc:
        raise RerenderError(str(exc)) from exc


def _capture_scene_inputs(
    scene_dir: Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, int]]:
    manifest_path = (scene_dir / "background.json").resolve()
    manifest_record = _file_record(manifest_path, include_state=True)
    try:
        ply_path = _scene_ply(scene_dir)
        ply_state = _file_state(ply_path)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RerenderError(f"cannot resolve scene inputs from {scene_dir}: {exc}") from exc
    _assert_record_current(manifest_record, label="scene manifest")
    return manifest_path, manifest_record, ply_path, ply_state


def _current_gs_registration(
    report: dict[str, Any], source: SourceRun, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Resolve the registration the replay worker will use for one source."""

    explicit = _last_option_value(source.argv, "--gs-transform")
    if explicit is not None:
        if explicit.strip().lower() in _GS_OFF_VALUES:
            return {
                "kind": "identity",
                "path": None,
                "file": None,
                "raw_matrix": None,
                "effective_matrix": None,
            }
        path = _resolved_recorded_path(explicit, label="recorded --gs-transform")
        file_record, matrix = _load_registration_file(
            path, label="current explicit GS registration"
        )
        return {
            "kind": "file",
            "path": str(path),
            "file": file_record,
            "raw_matrix": matrix.tolist(),
            "effective_matrix": None,
        }

    raw_path = report.get("w2gs_path")
    report_matrix = np.asarray(report.get("w2gs_matrix"), dtype=np.float64)
    if report_matrix.shape != (4, 4) or not np.isfinite(report_matrix).all():
        raise RerenderError("scene contract has no valid world-to-GS matrix")
    if raw_path is not None:
        path = Path(raw_path).resolve()
        file_record, loaded_matrix = _load_registration_file(
            path, label="current scene GS registration"
        )
        if not np.array_equal(report_matrix, loaded_matrix):
            raise RerenderError(
                "scene contract matrix differs from its current w2gs file"
            )
        return {
            "kind": "file",
            "path": str(path),
            "file": file_record,
            "raw_matrix": loaded_matrix.tolist(),
            "effective_matrix": None,
        }

    inline_present = manifest.get("mat_w2gs") not in (None, "")
    if inline_present:
        kind = "inline"
    else:
        if not np.array_equal(report_matrix, np.eye(4, dtype=np.float64)):
            raise RerenderError(
                "scene contract reports a non-identity matrix without a "
                "world-to-GS source"
            )
        kind = "identity"
    return {
        "kind": kind,
        "path": None,
        "file": None,
        "raw_matrix": None if kind == "identity" else report_matrix.tolist(),
        "effective_matrix": None,
    }


def _registration_scene_signature(registration: dict[str, Any]) -> tuple[Any, ...]:
    file_record = registration.get("file")
    return (
        registration.get("kind"),
        registration.get("path"),
        file_record.get("sha256") if isinstance(file_record, dict) else None,
        json.dumps(registration.get("raw_matrix"), separators=(",", ":")),
    )


def _validated_scene_identity(
    *,
    scene_tag: str,
    scene_dir: Path,
    sources: list[SourceRun],
    health: dict[str, str],
    manifest_path: Path,
    manifest_record: dict[str, Any],
    ply_path: Path,
    ply_state: dict[str, int],
) -> dict[str, Any]:
    """Validate scene bytes and official authority using the server's PLY digest."""

    _assert_record_current(manifest_record, label="scene manifest")
    try:
        if _file_state(ply_path) != ply_state:
            raise RerenderError(
                f"scene PLY changed while the GS server loaded it: {ply_path}"
            )
    except OSError as exc:
        raise RerenderError(f"cannot recheck scene PLY {ply_path}: {exc}") from exc
    if health["ply_path"] != str(ply_path.resolve()):
        raise RerenderError(f"{scene_tag}: GS server loaded a different scene PLY")

    manifest = _load_json_object(manifest_path, label="scene manifest")
    schema_version = manifest.get("schema_version", 1)
    official_modes = {source.official for source in sources}
    if len(official_modes) != 1:
        raise RerenderError(f"{scene_tag}: official and diagnostic sources are mixed")
    official = official_modes == {True}
    preverified_digest = health["ply_sha256"] if schema_version == 2 else None
    try:
        report = validate_scene_contract(
            scene_dir,
            strict=official,
            preverified_ply_sha256=preverified_digest,
        )
    except (SceneContractError, OSError, ValueError, TypeError) as exc:
        raise RerenderError(f"{scene_tag}: scene contract is invalid: {exc}") from exc
    if Path(report["manifest_path"]).resolve() != manifest_path:
        raise RerenderError(f"{scene_tag}: scene contract resolved another manifest")
    if Path(report["ply_path"]).resolve() != ply_path.resolve():
        raise RerenderError(f"{scene_tag}: scene contract resolved another PLY")
    if report.get("ply_sha256") != health["ply_sha256"]:
        raise RerenderError(f"{scene_tag}: scene contract and GS PLY digest differ")

    current_registrations: list[dict[str, Any]] = []
    for source in sources:
        source_registration = source.source_records.get("gs_registration")
        if not isinstance(source_registration, dict):
            raise RerenderError(
                f"{source.run_dir}: source has no fail-closed GS registration record"
            )
        current_registration = _current_gs_registration(
            report, source, manifest
        )
        _assert_registration_matches(
            source_registration,
            current_registration,
            label=f"{source.run_dir}: current scene",
            compare_effective=False,
        )
        current_registrations.append(current_registration)

        if not source.official:
            continue
        assert source.official_authority is not None
        authority = source.official_authority["scene"]
        expected_manifest = _resolved_recorded_path(
            authority.get("manifest_path"),
            label="official scene manifest_path",
        )
        expected_digest = authority.get("contract_digest")
        expected_ply_digest = authority.get("ply_sha256")
        if (
            authority.get("id") != report.get("id")
            or expected_manifest != manifest_path
            or expected_digest != report.get("contract_digest")
            or expected_ply_digest != health["ply_sha256"]
        ):
            raise RerenderError(
                f"{source.run_dir}: scene differs from official launch preflight"
            )
        summary_scene = source.summary.get("scene_contract")
        if not isinstance(summary_scene, dict):
            raise RerenderError(f"{source.run_dir}: summary has no scene contract")
        summary_manifest = _resolved_recorded_path(
            summary_scene.get("manifest"),
            label="summary scene manifest",
        )
        if (
            summary_scene.get("tag") != scene_tag
            or summary_scene.get("id") != report.get("id")
            or summary_scene.get("sha256") != report.get("contract_digest")
            or summary_manifest != manifest_path
        ):
            raise RerenderError(
                f"{source.run_dir}: summary scene identity differs from preflight"
            )

    signatures = {
        _registration_scene_signature(registration)
        for registration in current_registrations
    }
    if len(signatures) != 1:
        raise RerenderError(
            f"{scene_tag}: selected sources resolve different GS registrations"
        )
    current_registration = current_registrations[0]

    _assert_record_current(manifest_record, label="scene manifest")
    try:
        current_ply_state = _file_state(ply_path)
    except OSError as exc:
        raise RerenderError(f"cannot snapshot scene PLY {ply_path}: {exc}") from exc
    if current_ply_state != ply_state:
        raise RerenderError(f"scene PLY changed during contract validation: {ply_path}")
    return {
        "tag": scene_tag,
        "id": report.get("id"),
        "contract_digest": report.get("contract_digest"),
        "manifest": manifest_record,
        "ply": {
            "path": str(ply_path.resolve()),
            "sha256": health["ply_sha256"],
            "state": current_ply_state,
        },
        "w2gs": {
            "kind": current_registration["kind"],
            "path": current_registration["path"],
            "file": current_registration["file"],
            "raw_matrix": current_registration["raw_matrix"],
        },
    }
