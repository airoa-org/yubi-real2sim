# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Post-run provenance and rerender validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Mapping

from real2sim._bootstrap import REPO_ROOT
from real2sim.runner.argv import _argv_value, worker_command_head_length
from robosuite_env.contracts import provenance as provenance_contract

_FILE_STATE_KEYS = frozenset(
    {"device", "inode", "mode", "size", "mtime_ns", "ctime_ns"}
)

_RERENDER_PROVENANCE_SCHEMA_VERSION = 5

_BATCH_POLICY_LINEAGE_FILENAMES = provenance_contract.BATCH_POLICY_LINEAGE_FILENAMES

# Re-exported through real2sim.launcher. Completion checks the renderer's
# recorded core list, not this one.
_CORE_RENDERER_SOURCE_PATHS = provenance_contract.CORE_RENDERER_SOURCE_PATHS

_file_sha256 = provenance_contract.sha256_file

_file_state = provenance_contract.file_state

_tracked_diff_identity = provenance_contract.git_tracked_diff_identity


def _record_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return Path(value).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None

def _validate_provenance_file_record(
    record: Any,
    *,
    expected_path: Path,
    label: str,
    errors: list[str],
    require_hash: bool,
    require_state: bool,
    verify_hash: bool = True,
) -> str | None:
    required = {"path"}
    if require_hash:
        required.add("sha256")
    if require_state:
        required.add("state")
    if not isinstance(record, dict) or set(record) != required:
        errors.append(f"{label} provenance record has an invalid structure")
        return None
    expected_path = expected_path.resolve()
    if _record_path(record.get("path")) != expected_path:
        errors.append(f"{label} provenance path is not canonical")
        return None
    recorded_hash: str | None = None
    if require_hash:
        raw_hash = record.get("sha256")
        if not isinstance(raw_hash, str) or re.fullmatch(
            r"[0-9a-f]{64}", raw_hash
        ) is None:
            errors.append(f"{label} provenance SHA-256 is invalid")
        else:
            recorded_hash = raw_hash
    try:
        before = _file_state(expected_path)
        digest = (
            _file_sha256(expected_path)
            if require_hash and verify_hash
            else recorded_hash
        )
        after = _file_state(expected_path)
    except OSError as exc:
        errors.append(
            f"{label} provenance file is missing or unreadable: "
            f"{type(exc).__name__}"
        )
        return None
    if before != after:
        errors.append(f"{label} provenance file changed while it was verified")
        return None
    if require_hash and verify_hash and recorded_hash is not None:
        if recorded_hash != digest:
            errors.append(f"{label} provenance SHA-256 does not match the current file")
    if require_state:
        state = record.get("state")
        if not isinstance(state, dict) or set(state) != _FILE_STATE_KEYS:
            errors.append(f"{label} provenance stat snapshot has an invalid structure")
        elif state != after:
            errors.append(
                f"{label} provenance stat snapshot does not match the current file"
            )
    return digest

_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _is_canonical_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts and path.as_posix() == value


def _validate_recorded_core_source(
    record: Any,
    *,
    checkout_root: Path,
    relative_path: str,
    commit: str | None,
    errors: list[str],
) -> None:
    """Check one recorded core renderer file without reading the checkout.

    The record must name the file in the checkout the renderer checked. For a
    render at the run's clean commit, its hash must match that commit's blob,
    which a later pull or edit of the checkout does not change. A moved-on
    render's hashes are compared with the seal by the generation check.
    """

    label = f"renderer source {relative_path}"
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "state"}:
        errors.append(f"{label} provenance record has an invalid structure")
        return
    if record.get("path") != str(checkout_root / relative_path):
        errors.append(f"{label} provenance path is not in the checked checkout")
        return
    digest = record.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        errors.append(f"{label} provenance SHA-256 is invalid")
        return
    state = record.get("state")
    if not isinstance(state, dict) or set(state) != _FILE_STATE_KEYS:
        errors.append(f"{label} provenance stat snapshot has an invalid structure")
    if commit is None:
        return
    try:
        blob = subprocess.run(
            ["git", "cat-file", "blob", f"{commit}:{relative_path}"],
            cwd=checkout_root,
            check=True,
            # Never fetch: the commit the renderer ran is in its own checkout.
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(
            f"{label} could not be read from commit {commit}: {type(exc).__name__}"
        )
        return
    if hashlib.sha256(blob).hexdigest() != digest:
        errors.append(
            f"{label} provenance SHA-256 does not match commit {commit}"
        )


def _recorded_run_argv(run_command_path: Path) -> list[str]:
    text = run_command_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = next(
        index for index, line in enumerate(lines) if line.strip() == "```bash"
    )
    stop = next(
        index
        for index in range(start + 1, len(lines))
        if lines[index].strip() == "```"
    )
    tokens = shlex.split(" ".join(lines[start + 1 : stop]))
    index = 0
    while (
        index < len(tokens)
        and "=" in tokens[index]
        and not tokens[index].startswith("-")
    ):
        name, _, _value = tokens[index].partition("=")
        if not name.isidentifier():
            break
        index += 1
    argv = tokens[index:]
    if worker_command_head_length(argv) is None:
        raise ValueError("unexpected recorded command head")
    return argv

def _scene_ply_from_manifest(manifest_path: Path) -> Path:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scene manifest must contain a JSON object")
    raw_ply = next(
        (
            payload[key]
            for key in ("ply_path", "ply", "point_cloud", "point_cloud_path")
            if key in payload and payload[key] not in (None, "")
        ),
        None,
    )
    if not isinstance(raw_ply, str):
        raise ValueError("scene manifest has no PLY path")
    ply_path = Path(raw_ply).expanduser()
    if not ply_path.is_absolute():
        ply_path = manifest_path.parent / ply_path
    return ply_path.resolve()

def _validate_posthoc_provenance(
    *,
    provenance_path: Path,
    validation_path: Path,
    validation: Mapping[str, Any],
    expected_video: Path,
    expected_summary: Path,
    expected_frames: int,
    configured_frames: int | None = None,
    run_dir: Path,
    scene_tag_value: str,
    official_manifest_path: Path,
    run_manifest: Mapping[str, Any] | None,
    expected_scene_id: str | None,
    expected_scene_digest: str | None,
    expected_scene_manifest: Path,
    expected_scene_ply_sha256: str | None,
) -> list[str]:
    errors: list[str] = []
    recorded_frames_budget = (
        expected_frames if configured_frames is None else configured_frames
    )
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [
            "rerender provenance is missing or unreadable: "
            f"{type(exc).__name__}"
        ]
    top_level_keys = {
        "schema_version",
        "kind",
        "mode",
        "created_at",
        "source",
        "renderer",
        "scene",
        "output",
    }
    optional_cadence_key = "policy_action_playback_contract"
    if isinstance(provenance, dict) and optional_cadence_key in provenance:
        top_level_keys.add(optional_cadence_key)
    if not isinstance(provenance, dict) or set(provenance) != top_level_keys:
        return ["rerender provenance has an invalid top-level structure"]
    if provenance.get("schema_version") != _RERENDER_PROVENANCE_SCHEMA_VERSION:
        errors.append(
            "rerender provenance schema_version must be "
            f"{_RERENDER_PROVENANCE_SCHEMA_VERSION}"
        )
    if provenance.get("kind") != "real2sim-rerender-provenance":
        errors.append("rerender provenance kind is invalid")
    expected_mode = "official"
    if provenance.get("mode") != expected_mode:
        errors.append(
            f"rerender provenance mode={provenance.get('mode')!r}, "
            f"expected {expected_mode!r}"
        )
    created_at = provenance.get("created_at")
    if not isinstance(created_at, str) or re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[^\s]+Z", created_at
    ) is None:
        errors.append("rerender provenance created_at is invalid")

    source = provenance.get("source")
    official_source_keys = {
        "run_manifest",
        "worker_contract",
        "code_commit",
        "official_generation_check",
    }
    source_keys = {
        "generation_summary",
        "state_log",
        "run_command",
        "batch_policy_lineage",
        "expected_frames",
        "gs_registration",
    } | official_source_keys
    if isinstance(source, dict) and optional_cadence_key in source:
        source_keys.add(optional_cadence_key)
    optional_render_state_key = "render_state_log_validation"
    if isinstance(source, dict) and optional_render_state_key in source:
        source_keys.add(optional_render_state_key)
    if not isinstance(source, dict) or set(source) != source_keys:
        errors.append("rerender provenance source has an invalid structure")
        source = {}
    state_log_path = run_dir / "logs" / "sim_log.jsonl"
    run_command_path = run_dir / "RUN_COMMAND.md"
    _validate_provenance_file_record(
        source.get("generation_summary"),
        expected_path=expected_summary,
        label="generation summary",
        errors=errors,
        require_hash=True,
        require_state=True,
    )
    state_log = source.get("state_log")
    if not isinstance(state_log, dict) or set(state_log) != {
        "path",
        "sha256",
        "rows",
        "state",
    }:
        errors.append("state log provenance record has an invalid structure")
    else:
        state_log_file = {key: state_log[key] for key in ("path", "sha256", "state")}
        _validate_provenance_file_record(
            state_log_file,
            expected_path=state_log_path,
            label="state log",
            errors=errors,
            require_hash=True,
            require_state=True,
        )
        try:
            with state_log_path.open("r", encoding="utf-8") as stream:
                current_rows = sum(1 for line in stream if line.strip())
        except (OSError, UnicodeError) as exc:
            errors.append(
                "state log row count could not be verified: "
                f"{type(exc).__name__}"
            )
        else:
            if state_log.get("rows") != current_rows:
                errors.append(
                    "state log provenance row count does not match the current file"
                )
            if current_rows != expected_frames:
                errors.append(
                    f"state log has {current_rows} rows, expected {expected_frames}"
                )
    _validate_provenance_file_record(
        source.get("run_command"),
        expected_path=run_command_path,
        label="run command",
        errors=errors,
        require_hash=True,
        require_state=True,
    )
    if source.get("expected_frames") != expected_frames:
        errors.append("rerender provenance expected_frames is invalid")

    def valid_matrix(value: Any, *, nullable: bool) -> bool:
        if value is None:
            return nullable
        return (
            isinstance(value, list)
            and len(value) == 4
            and all(isinstance(row, list) and len(row) == 4 for row in value)
            and all(
                not isinstance(item, bool)
                and isinstance(item, (int, float))
                and math.isfinite(float(item))
                for row in value
                for item in row
            )
        )

    generation_registration = source.get("gs_registration")
    registration_keys = {
        "kind",
        "path",
        "file",
        "raw_matrix",
        "effective_matrix",
        "calibration_world_offset_m",
    }
    if (
        not isinstance(generation_registration, dict)
        or set(generation_registration) != registration_keys
        or generation_registration.get("kind")
        not in {"file", "inline", "identity"}
        or not valid_matrix(
            generation_registration.get("raw_matrix"), nullable=True
        )
        or not valid_matrix(
            generation_registration.get("effective_matrix"), nullable=True
        )
        or not (
            isinstance(
                generation_registration.get("calibration_world_offset_m"),
                list,
            )
            and len(generation_registration["calibration_world_offset_m"]) == 3
            and all(
                not isinstance(item, bool)
                and isinstance(item, (int, float))
                and math.isfinite(float(item))
                for item in generation_registration[
                    "calibration_world_offset_m"
                ]
            )
        )
    ):
        errors.append("generation GS registration provenance is invalid")
        generation_registration = {}

    registration_kind = generation_registration.get("kind")
    registration_path = _record_path(generation_registration.get("path"))
    registration_file = generation_registration.get("file")
    if registration_kind == "file":
        if registration_path is None:
            errors.append("generation GS registration path is not canonical")
        else:
            _validate_provenance_file_record(
                registration_file,
                expected_path=registration_path,
                label="generation GS registration",
                errors=errors,
                require_hash=True,
                require_state=True,
            )
        if generation_registration.get("raw_matrix") is None:
            errors.append("generation file GS registration has no raw matrix")
    else:
        if generation_registration and (
            generation_registration.get("path") is not None
            or registration_file is not None
        ):
            errors.append(
                "generation non-file GS registration must not bind a file"
            )
        if registration_kind == "inline" and generation_registration.get(
            "raw_matrix"
        ) is None:
            errors.append("generation inline GS registration has no raw matrix")
        if registration_kind == "identity" and any(
            generation_registration.get(key) is not None
            for key in ("raw_matrix", "effective_matrix")
        ):
            errors.append("generation identity GS registration has matrices")

    try:
        generation_summary_payload = json.loads(
            expected_summary.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(
            "generation summary GS registration could not be read: "
            f"{type(exc).__name__}"
        )
        generation_summary_payload = {}
    if isinstance(generation_summary_payload, dict) and generation_registration:
        if generation_summary_payload.get("gs_w2gs_raw") != (
            generation_registration.get("raw_matrix")
        ):
            errors.append(
                "generation GS registration raw matrix differs from summary"
            )
        if generation_summary_payload.get("gs_w2gs_effective") != (
            generation_registration.get("effective_matrix")
        ):
            errors.append(
                "generation GS registration effective matrix differs from summary"
            )
        if generation_summary_payload.get("gs_calibration_world_offset_m") != (
            generation_registration.get("calibration_world_offset_m")
        ):
            errors.append(
                "generation GS registration offset differs from summary"
            )
        reported_registration = generation_summary_payload.get("gs_w2gs")
        if registration_kind == "file":
            if _record_path(reported_registration) != registration_path:
                errors.append(
                    "generation GS registration path differs from summary"
                )
        elif registration_kind == "inline":
            if reported_registration != "inline-matrix":
                errors.append(
                    "generation inline GS registration differs from summary"
                )
        elif registration_kind == "identity" and reported_registration not in (
            None,
            "",
        ):
            errors.append(
                "generation identity GS registration differs from summary"
            )

    render_state_validation = source.get(optional_render_state_key)
    if render_state_validation is not None:
        expected_render_state_keys = {
            "complete",
            "inventory_sha256",
            "model_signature",
            "rows_written",
            "target",
            "task_object_names",
        }
        if (
            not isinstance(render_state_validation, dict)
            or set(render_state_validation) != expected_render_state_keys
            or render_state_validation.get("complete") is not True
            or render_state_validation.get("rows_written") != expected_frames
            or not isinstance(render_state_validation.get("target"), str)
            or not isinstance(
                render_state_validation.get("task_object_names"), list
            )
        ):
            errors.append("render-state log validation provenance is invalid")

    lineage = source.get("batch_policy_lineage")
    expected_batch_root = run_dir.parents[1].resolve()
    if not isinstance(lineage, dict) or set(lineage) != {
        "batch_root",
        "optional_filenames",
        "files",
    }:
        errors.append("batch policy lineage provenance has an invalid structure")
    else:
        if _record_path(lineage.get("batch_root")) != expected_batch_root:
            errors.append("batch policy lineage root is not canonical")
        if lineage.get("optional_filenames") != list(
            _BATCH_POLICY_LINEAGE_FILENAMES
        ):
            errors.append("batch policy lineage candidate file set is invalid")
        lineage_files = lineage.get("files")
        if not isinstance(lineage_files, dict) or any(
            filename not in _BATCH_POLICY_LINEAGE_FILENAMES
            for filename in lineage_files if isinstance(filename, str)
        ):
            errors.append("batch policy lineage file records are invalid")
            lineage_files = {}
        observed_lineage_names: set[str] = set()
        for filename in _BATCH_POLICY_LINEAGE_FILENAMES:
            candidate = expected_batch_root / filename
            if candidate.is_symlink():
                errors.append(
                    f"batch policy lineage file must not be a symlink: {filename}"
                )
            elif candidate.exists():
                if candidate.is_file():
                    observed_lineage_names.add(filename)
                else:
                    errors.append(
                        f"batch policy lineage path is not a file: {filename}"
                    )
        if set(lineage_files) != observed_lineage_names:
            errors.append("batch policy lineage file presence changed")
        for filename, record in lineage_files.items():
            if filename not in _BATCH_POLICY_LINEAGE_FILENAMES:
                continue
            _validate_provenance_file_record(
                record,
                expected_path=expected_batch_root / filename,
                label=f"batch policy lineage {filename}",
                errors=errors,
                require_hash=True,
                require_state=True,
            )

    expected_code_commit: str | None = None
    worker_contract_path = run_dir / "official_worker_contract.json"
    _validate_provenance_file_record(
        source.get("run_manifest"),
        expected_path=official_manifest_path,
        label="official run manifest",
        errors=errors,
        require_hash=True,
        require_state=False,
    )
    _validate_provenance_file_record(
        source.get("worker_contract"),
        expected_path=worker_contract_path,
        label="official worker contract",
        errors=errors,
        require_hash=True,
        require_state=False,
    )
    if isinstance(run_manifest, Mapping):
        code = run_manifest.get("code")
        if isinstance(code, Mapping):
            raw_commit = code.get("commit")
            if isinstance(raw_commit, str):
                expected_code_commit = raw_commit
    if (
        expected_code_commit is None
        or re.fullmatch(r"[0-9a-f]{40}", expected_code_commit) is None
    ):
        errors.append("official run manifest has no valid code commit")
    elif source.get("code_commit") != expected_code_commit:
        errors.append("rerender source code commit differs from the official run")

    renderer = provenance.get("renderer")
    # robosuite_env/rerender/validation.py also records the optional
    # third-person view overrides (None for a canonical replay).
    if (
        not isinstance(renderer, dict)
        or set(renderer) - {"view_overrides"} != {"sanitized_argv", "code"}
        or not (
            renderer.get("view_overrides") is None
            or isinstance(renderer.get("view_overrides"), dict)
        )
    ):
        errors.append("rerender provenance renderer has an invalid structure")
        renderer = {}
    elif renderer.get("view_overrides"):
        errors.append(
            "rerender used third-person view overrides; canonical videos must "
            "replay the recorded viewpoint"
        )
    renderer_argv = renderer.get("sanitized_argv")
    if (
        not isinstance(renderer_argv, list)
        or not renderer_argv
        or any(not isinstance(item, str) for item in renderer_argv)
    ):
        errors.append("rerender sanitized argv must be a non-empty string array")
    else:
        expected_rerender_summary = run_dir / "rerender" / "sim_summary.json"
        expected_bindings = {
            "--state-log-replay": state_log_path,
            "--out": expected_video,
            "--summary-path": expected_rerender_summary,
        }
        for option, expected_path in expected_bindings.items():
            if (
                _record_path(_argv_value(renderer_argv, option))
                != expected_path.resolve()
            ):
                errors.append(
                    f"rerender sanitized argv has an invalid {option} binding"
                )
        forbidden = {
            "--no-render",
            "--official-run-manifest",
            "--official-worker-contract",
            "--official-scene-tag",
        }
        if any(token.partition("=")[0] in forbidden for token in renderer_argv):
            errors.append(
                "rerender sanitized argv retains forbidden generation options"
            )
    renderer_code = renderer.get("code")
    if not isinstance(renderer_code, dict) or set(renderer_code) != {
        "commit",
        "dirty",
        "core_source_files",
        "tracked_diff",
        "gaussian_wrapping",
    }:
        errors.append("rerender renderer code provenance has an invalid structure")
    else:
        renderer_commit = renderer_code.get("commit")
        if (
            not isinstance(renderer_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", renderer_commit) is None
            or not isinstance(renderer_code.get("dirty"), bool)
        ):
            errors.append("rerender renderer code provenance is invalid")
        else:
            # The renderer recorded the check it made before rendering, and
            # its recheck when it wrote this provenance. Verify that record
            # against the run's seal; never read the checkout as it is now,
            # which a pull or an edit since the render may have changed.
            generation_check = (
                source.get("official_generation_check")
                if isinstance(source, dict)
                else None
            )
            for problem in provenance_contract.official_generation_check_errors(
                official_manifest_path.parent,
                run_manifest,
                renderer_code,
                generation_check,
            ):
                errors.append(
                    f"official rerender cannot stand in for the run's code: {problem}"
                )
            checked = (
                generation_check
                if isinstance(generation_check, dict)
                and isinstance(generation_check.get("checkout_root"), str)
                else None
            )
            core_source_files = renderer_code.get("core_source_files")
            if checked is None:
                pass
            elif (
                not isinstance(core_source_files, dict)
                or not core_source_files
                or not all(
                    _is_canonical_relative(relative_path)
                    for relative_path in core_source_files
                )
            ):
                errors.append(
                    "rerender renderer core source provenance is invalid"
                )
            else:
                checkout_root = Path(checked["checkout_root"])
                # The renderer's own core list, as recorded; not the list of
                # the code that reports, which a later pull may have changed.
                for relative_path in sorted(core_source_files):
                    _validate_recorded_core_source(
                        core_source_files.get(relative_path),
                        checkout_root=checkout_root,
                        relative_path=relative_path,
                        commit=renderer_commit
                        if checked.get("rule")
                        == provenance_contract.RULE_GENERATION_COMMIT
                        else None,
                        errors=errors,
                    )
            tracked_diff = renderer_code.get("tracked_diff")
            if not isinstance(tracked_diff, dict) or set(tracked_diff) != {
                "base_commit",
                "sha256",
                "size_bytes",
            }:
                errors.append("rerender tracked diff provenance is invalid")
            elif (
                tracked_diff.get("base_commit") != renderer_commit
                or not isinstance(tracked_diff.get("sha256"), str)
                or re.fullmatch(
                    r"[0-9a-f]{64}", tracked_diff.get("sha256", "")
                )
                is None
                or isinstance(tracked_diff.get("size_bytes"), bool)
                or not isinstance(tracked_diff.get("size_bytes"), int)
                or tracked_diff.get("size_bytes") < 0
            ):
                errors.append("rerender tracked diff provenance is invalid")
            elif renderer_code.get("dirty") is False and tracked_diff != {
                "base_commit": renderer_commit,
                "sha256": _EMPTY_SHA256,
                "size_bytes": 0,
            }:
                errors.append(
                    "rerender tracked diff provenance is not empty for a "
                    "clean renderer checkout"
                )

    scene = provenance.get("scene")
    if not isinstance(scene, dict) or set(scene) != {
        "tag",
        "id",
        "contract_digest",
        "manifest",
        "ply",
        "w2gs",
    }:
        errors.append("rerender provenance scene has an invalid structure")
        scene = {}
    if scene.get("tag") != scene_tag_value:
        errors.append("rerender provenance scene tag is invalid")
    if scene.get("id") != expected_scene_id:
        errors.append("rerender provenance scene id differs from launch preflight")
    if scene.get("contract_digest") != expected_scene_digest:
        errors.append("rerender provenance scene contract differs from launch preflight")
    _validate_provenance_file_record(
        scene.get("manifest"),
        expected_path=expected_scene_manifest,
        label="scene manifest",
        errors=errors,
        require_hash=True,
        require_state=True,
    )
    try:
        expected_ply = _scene_ply_from_manifest(expected_scene_manifest)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        errors.append(
            "scene PLY could not be resolved from the current manifest: "
            f"{type(exc).__name__}"
        )
        expected_ply = expected_scene_manifest.parent / "__invalid_scene_ply__"
    current_ply_hash = _validate_provenance_file_record(
        scene.get("ply"),
        expected_path=expected_ply,
        label="scene PLY",
        errors=errors,
        require_hash=True,
        require_state=True,
        # The authenticated GS service already read and hashed the PLY while
        # loading it. The matching stat seal proves that the same inode bytes
        # remain current, avoiding a redundant full-file read here.
        verify_hash=False,
    )
    if (
        expected_scene_ply_sha256 is not None
        and current_ply_hash != expected_scene_ply_sha256
    ):
        errors.append("rerender scene PLY differs from launch preflight")

    scene_registration = scene.get("w2gs")
    scene_registration_keys = {"kind", "path", "file", "raw_matrix"}
    if (
        not isinstance(scene_registration, dict)
        or set(scene_registration) != scene_registration_keys
        or scene_registration.get("kind")
        not in {"file", "inline", "identity"}
        or not valid_matrix(scene_registration.get("raw_matrix"), nullable=True)
    ):
        errors.append("rerender scene GS registration provenance is invalid")
        scene_registration = {}
    scene_registration_kind = scene_registration.get("kind")
    scene_registration_path = _record_path(scene_registration.get("path"))
    scene_registration_file = scene_registration.get("file")
    if scene_registration_kind == "file":
        if scene_registration_path is None:
            errors.append("rerender scene GS registration path is not canonical")
        else:
            scene_registration_sha256 = _validate_provenance_file_record(
                scene_registration_file,
                expected_path=scene_registration_path,
                label="rerender scene GS registration",
                errors=errors,
                require_hash=True,
                require_state=True,
            )
            if isinstance(registration_file, dict) and (
                scene_registration_sha256 != registration_file.get("sha256")
            ):
                errors.append(
                    "rerender scene GS registration SHA-256 differs from generation"
                )
    elif scene_registration:
        if (
            scene_registration.get("path") is not None
            or scene_registration_file is not None
        ):
            errors.append("rerender scene non-file registration binds a file")
        if scene_registration_kind == "inline" and scene_registration.get(
            "raw_matrix"
        ) is None:
            errors.append("rerender scene inline registration has no raw matrix")
        if scene_registration_kind == "identity" and scene_registration.get(
            "raw_matrix"
        ) is not None:
            errors.append("rerender scene identity registration has a raw matrix")

    if generation_registration and scene_registration:
        if (
            scene_registration_kind != registration_kind
            or scene_registration.get("path")
            != generation_registration.get("path")
            or scene_registration.get("raw_matrix")
            != generation_registration.get("raw_matrix")
        ):
            errors.append(
                "rerender scene GS registration differs from generation"
            )

    explicit_transform = (
        _argv_value(renderer_argv, "--gs-transform")
        if isinstance(renderer_argv, list)
        else None
    )
    if explicit_transform is not None:
        if explicit_transform.strip().lower() in {
            "",
            "none",
            "off",
            "false",
            "0",
            "no",
        }:
            if scene_registration_kind != "identity":
                errors.append(
                    "rerender scene GS registration differs from disabled override"
                )
        else:
            explicit_registration_path = Path(explicit_transform).expanduser()
            if not explicit_registration_path.is_absolute():
                explicit_registration_path = REPO_ROOT / explicit_registration_path
            explicit_registration_path = explicit_registration_path.resolve()
            if (
                scene_registration_kind != "file"
                or scene_registration_path != explicit_registration_path
            ):
                errors.append(
                    "rerender scene GS registration differs from explicit override"
                )
    try:
        scene_manifest_payload = json.loads(
            expected_scene_manifest.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(
            "scene GS registration could not be resolved from the manifest: "
            f"{type(exc).__name__}"
        )
        scene_manifest_payload = {}
    if explicit_transform is None and isinstance(scene_manifest_payload, dict):
        manifest_inline = scene_manifest_payload.get("mat_w2gs")
        manifest_path_value = next(
            (
                scene_manifest_payload[key]
                for key in (
                    "w2gs",
                    "gs_w2gs",
                    "mat_w2gs_path",
                    "world_to_gs",
                    "world_to_gs_path",
                )
                if scene_manifest_payload.get(key) not in (None, "")
            ),
            None,
        )
        if manifest_inline is not None:
            if (
                scene_registration_kind != "inline"
                or scene_registration.get("raw_matrix") != manifest_inline
            ):
                errors.append(
                    "rerender scene GS registration differs from manifest inline matrix"
                )
        elif isinstance(manifest_path_value, str):
            manifest_registration_path = Path(manifest_path_value).expanduser()
            if not manifest_registration_path.is_absolute():
                manifest_registration_path = (
                    expected_scene_manifest.parent / manifest_registration_path
                )
            manifest_registration_path = manifest_registration_path.resolve()
            if (
                scene_registration_kind != "file"
                or scene_registration_path != manifest_registration_path
            ):
                errors.append(
                    "rerender scene GS registration path differs from manifest"
                )
            manifest_registration_sha256 = scene_manifest_payload.get(
                "w2gs_sha256"
            )
            if manifest_registration_sha256 is not None and (
                not isinstance(scene_registration_file, dict)
                or scene_registration_file.get("sha256")
                != manifest_registration_sha256
            ):
                errors.append(
                    "rerender scene GS registration SHA-256 differs from manifest"
                )
        elif scene_registration_kind != "identity":
            errors.append(
                "rerender scene GS registration differs from manifest identity"
            )

    rerender_summary_path = run_dir / "rerender" / "sim_summary.json"
    try:
        rerender_summary_payload = json.loads(
            rerender_summary_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(
            "rerender summary GS registration could not be read: "
            f"{type(exc).__name__}"
        )
        rerender_summary_payload = {}
    if isinstance(rerender_summary_payload, dict) and generation_registration:
        if rerender_summary_payload.get("gs_w2gs_raw") != (
            generation_registration.get("raw_matrix")
        ):
            errors.append("rerender summary raw GS registration differs from generation")
        if rerender_summary_payload.get("gs_w2gs_effective") != (
            generation_registration.get("effective_matrix")
        ):
            errors.append(
                "rerender summary effective GS registration differs from generation"
            )
        if rerender_summary_payload.get("gs_calibration_world_offset_m") != (
            generation_registration.get("calibration_world_offset_m")
        ):
            errors.append("rerender summary GS registration offset differs from generation")
        rerender_reported_registration = rerender_summary_payload.get("gs_w2gs")
        if registration_kind == "file" and (
            _record_path(rerender_reported_registration) != registration_path
        ):
            errors.append("rerender summary GS registration path differs from generation")
        elif registration_kind == "inline" and (
            rerender_reported_registration != "inline-matrix"
        ):
            errors.append("rerender summary inline GS registration differs from generation")
        elif registration_kind == "identity" and rerender_reported_registration not in (
            None,
            "",
        ):
            errors.append("rerender summary identity GS registration differs from generation")
    if validation.get("gs_registration_exact_match") is not True:
        errors.append("video validation has no exact GS registration attestation")

    output = provenance.get("output")
    if not isinstance(output, dict) or set(output) != {
        "video",
        "validation",
        "rerender_summary",
        "decoded_frames",
        "decoded_sha256",
    }:
        errors.append("rerender provenance output has an invalid structure")
        output = {}
    _validate_provenance_file_record(
        output.get("video"),
        expected_path=expected_video,
        label="rerender video",
        errors=errors,
        require_hash=True,
        require_state=True,
    )
    _validate_provenance_file_record(
        output.get("validation"),
        expected_path=validation_path,
        label="rerender video validation",
        errors=errors,
        require_hash=True,
        require_state=True,
    )
    rerender_summary_path = run_dir / "rerender" / "sim_summary.json"
    rerender_summary_sha256 = _validate_provenance_file_record(
        output.get("rerender_summary"),
        expected_path=rerender_summary_path,
        label="rerender summary",
        errors=errors,
        require_hash=True,
        require_state=True,
    )
    if (
        not isinstance(validation.get("rerender_summary_sha256"), str)
        or validation.get("rerender_summary_sha256") != rerender_summary_sha256
    ):
        errors.append(
            "rerender summary provenance differs from video validation"
        )
    if output.get("decoded_frames") != expected_frames:
        errors.append("rerender provenance decoded frame count is invalid")
    decoded_sha256 = validation.get("decoded_sha256")
    if (
        not isinstance(decoded_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", decoded_sha256) is None
        or output.get("decoded_sha256") != decoded_sha256
    ):
        errors.append("rerender provenance decoded SHA-256 is invalid")

    source_cadence = source.get(optional_cadence_key)
    provenance_cadence = provenance.get(optional_cadence_key)
    validation_cadence = validation.get(optional_cadence_key)
    cadence_presence = (
        optional_cadence_key in source,
        optional_cadence_key in provenance,
        optional_cadence_key in validation,
    )
    if any(cadence_presence) and not all(cadence_presence):
        errors.append(
            "policy-action playback provenance must be present in source, "
            "output validation, and top level"
        )
    elif all(cadence_presence):
        source_cadence_keys = {
            "schema_version",
            "kind",
            "source_policy_action_hz",
            "source_policy_action_dt_nominal_s",
            "source_frames",
            "source_nominal_duration_s",
            "source_control_hz",
            "source_control_tick_count",
            "source_simulated_duration_s",
            "source_boundary_phase_error_max_abs_s",
            "replay_frame_hz",
            "replay_output_fps",
            "replay_expected_frames",
            "replay_expected_duration_s",
            "mapping",
        }
        replay_cadence_keys = source_cadence_keys | {
            "replay_summary",
            "decoded_video",
            "no_time_stretch",
        }
        if not all(
            isinstance(value, dict)
            for value in (source_cadence, provenance_cadence, validation_cadence)
        ):
            errors.append("policy-action playback provenance is invalid")
        elif (
            set(source_cadence) != source_cadence_keys
            or set(provenance_cadence) != replay_cadence_keys
            or set(validation_cadence) != replay_cadence_keys
        ):
            errors.append("policy-action playback provenance structure is invalid")
        else:
            if provenance_cadence != validation_cadence:
                errors.append(
                    "policy-action playback provenance differs from video validation"
                )
            if any(
                provenance_cadence.get(key) != value
                for key, value in source_cadence.items()
            ):
                errors.append(
                    "policy-action playback provenance differs from generation source"
                )
            if (
                source_cadence.get("schema_version") != 1
                or source_cadence.get("kind")
                != "fr3-policy-action-to-exact-state-replay"
                or source_cadence.get("mapping")
                != "one saved state row to one encoded frame"
                or provenance_cadence.get("no_time_stretch") is not True
            ):
                errors.append("policy-action playback contract identity is invalid")

            def _cadence_number(key: str, *, positive: bool = False) -> float | None:
                raw = source_cadence.get(key)
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    errors.append(f"policy-action playback {key} is not numeric")
                    return None
                value = float(raw)
                if not math.isfinite(value) or (positive and value <= 0.0):
                    errors.append(f"policy-action playback {key} is invalid")
                    return None
                return value

            action_hz = _cadence_number(
                "source_policy_action_hz", positive=True
            )
            action_dt = _cadence_number(
                "source_policy_action_dt_nominal_s", positive=True
            )
            nominal_duration = _cadence_number(
                "source_nominal_duration_s", positive=True
            )
            control_hz = _cadence_number("source_control_hz", positive=True)
            simulated_duration = _cadence_number(
                "source_simulated_duration_s", positive=True
            )
            boundary_error = _cadence_number(
                "source_boundary_phase_error_max_abs_s"
            )
            replay_hz = _cadence_number("replay_frame_hz", positive=True)
            replay_duration = _cadence_number(
                "replay_expected_duration_s", positive=True
            )
            source_frames = source_cadence.get("source_frames")
            replay_frames = source_cadence.get("replay_expected_frames")
            replay_fps = source_cadence.get("replay_output_fps")
            tick_count = source_cadence.get("source_control_tick_count")
            integer_fields = {
                "source_frames": source_frames,
                "replay_expected_frames": replay_frames,
                "replay_output_fps": replay_fps,
                "source_control_tick_count": tick_count,
            }
            for key, value in integer_fields.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                ):
                    errors.append(f"policy-action playback {key} is invalid")
            if source_frames != expected_frames or replay_frames != expected_frames:
                errors.append("policy-action playback frame count is invalid")
            if (
                action_hz is not None
                and action_dt is not None
                and not math.isclose(
                    action_dt, 1.0 / action_hz, rel_tol=0.0, abs_tol=1e-9
                )
            ):
                errors.append("policy-action playback action period is inconsistent")
            if (
                action_hz is not None
                and nominal_duration is not None
                and isinstance(source_frames, int)
                and not isinstance(source_frames, bool)
                and not math.isclose(
                    nominal_duration,
                    source_frames / action_hz,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
            ):
                errors.append("policy-action playback source duration is inconsistent")
            if (
                action_hz is not None
                and (
                    not math.isclose(
                        action_hz,
                        float(round(action_hz)),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or control_hz is None
                    or action_hz > control_hz
                    or replay_hz is None
                    or not math.isclose(
                        replay_hz, action_hz, rel_tol=0.0, abs_tol=1e-12
                    )
                    or replay_fps != int(round(action_hz))
                )
            ):
                errors.append("policy-action playback replay rate is inconsistent")
            if (
                nominal_duration is not None
                and (
                    replay_duration is None
                    or not math.isclose(
                        replay_duration,
                        nominal_duration,
                        rel_tol=0.0,
                        abs_tol=1e-8,
                    )
                )
            ):
                errors.append("policy-action playback replay duration is inconsistent")
            if (
                control_hz is not None
                and simulated_duration is not None
                and isinstance(tick_count, int)
                and not isinstance(tick_count, bool)
                and not math.isclose(
                    simulated_duration,
                    tick_count / control_hz,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
            ):
                errors.append(
                    "policy-action playback simulated duration is inconsistent"
                )
            if (
                action_hz is not None
                and control_hz is not None
                and isinstance(source_frames, int)
                and not isinstance(source_frames, bool)
                and isinstance(tick_count, int)
                and not isinstance(tick_count, bool)
                and tick_count
                != int(
                    math.floor(
                        source_frames * control_hz / action_hz + 0.5 + 1e-12
                    )
                )
            ):
                errors.append("policy-action playback control tick count is invalid")
            if (
                control_hz is not None
                and simulated_duration is not None
                and nominal_duration is not None
                and abs(simulated_duration - nominal_duration)
                > 0.5 / control_hz + 1e-8
            ):
                errors.append("policy-action playback timing error exceeds its bound")
            if (
                control_hz is not None
                and boundary_error is not None
                and (
                    boundary_error < 0.0
                    or boundary_error > 0.5 / control_hz + 1e-8
                )
            ):
                errors.append(
                    "policy-action playback boundary phase error is invalid"
                )

            try:
                generation_summary = json.loads(
                    expected_summary.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append(
                    "generation summary could not be checked for action cadence: "
                    f"{type(exc).__name__}"
                )
            else:
                generation_bindings = {
                    "source_policy_action_hz": (
                        "policy_action_hz_effective",
                        action_hz,
                    ),
                    "source_policy_action_dt_nominal_s": (
                        "policy_action_dt_nominal_s",
                        action_dt,
                    ),
                    "source_control_hz": ("dynamic_control_hz", control_hz),
                    "source_simulated_duration_s": (
                        "fr3_dynamic_simulated_duration_s",
                        simulated_duration,
                    ),
                    "source_boundary_phase_error_max_abs_s": (
                        "fr3_dynamic_boundary_phase_error_s_max_abs",
                        boundary_error,
                    ),
                }
                for contract_key, (summary_key, contract_value) in (
                    generation_bindings.items()
                ):
                    summary_value = generation_summary.get(summary_key)
                    if (
                        contract_value is None
                        or isinstance(summary_value, bool)
                        or not isinstance(summary_value, (int, float))
                        or not math.isfinite(float(summary_value))
                        or not math.isclose(
                            float(summary_value),
                            contract_value,
                            rel_tol=0.0,
                            abs_tol=(
                                1e-6
                                if contract_key
                                in {
                                    "source_policy_action_hz",
                                    "source_policy_action_dt_nominal_s",
                                }
                                else 1e-8
                            ),
                        )
                    ):
                        errors.append(
                            "policy-action playback generation binding differs "
                            f"for {contract_key}"
                        )
                if generation_summary.get("frames") != source_frames:
                    errors.append(
                        "policy-action playback generation frame count differs"
                    )
                if generation_summary.get(
                    "fr3_dynamic_control_tick_count_total"
                ) != tick_count or generation_summary.get(
                    "fr3_dynamic_runtime_control_tick_count_total"
                ) != tick_count:
                    errors.append(
                        "policy-action playback generation tick count differs"
                    )
                if generation_summary.get("fr3_dynamic_tick_schedule") != (
                    "source_to_controller_cumulative_nearest_half_up"
                ) or generation_summary.get(
                    "fr3_dynamic_runtime_tick_count_matches_schedule"
                ) is not True:
                    errors.append(
                        "policy-action playback generation schedule is invalid"
                    )
                for summary_key in (
                    "policy_action_hz_requested",
                    "frame_hz_effective",
                ):
                    raw = generation_summary.get(summary_key)
                    if (
                        action_hz is None
                        or isinstance(raw, bool)
                        or not isinstance(raw, (int, float))
                        or not math.isclose(
                            float(raw), action_hz, rel_tol=0.0, abs_tol=1e-6
                        )
                    ):
                        errors.append(
                            "policy-action playback generation rate differs at "
                            + summary_key
                        )
                for summary_key in (
                    "policy_action_dt_s",
                    "frame_dt_s",
                    "playback_action_dt_s",
                ):
                    raw = generation_summary.get(summary_key)
                    if (
                        action_dt is None
                        or isinstance(raw, bool)
                        or not isinstance(raw, (int, float))
                        or not math.isclose(
                            float(raw), action_dt, rel_tol=0.0, abs_tol=1e-6
                        )
                    ):
                        errors.append(
                            "policy-action playback generation period differs at "
                            + summary_key
                        )
                if generation_summary.get("output_fps") != replay_fps:
                    errors.append(
                        "policy-action playback generation output FPS differs"
                    )

            try:
                generation_argv = _recorded_run_argv(run_command_path)
                command_action_hz = float(
                    _argv_value(generation_argv, "--policy-action-hz") or "nan"
                )
                command_control_hz = float(
                    _argv_value(generation_argv, "--dynamic-control-hz") or 20.0
                )
                command_frames = int(
                    _argv_value(generation_argv, "--frames") or 0
                )
                command_frame_hz = float(
                    _argv_value(generation_argv, "--frame-hz") or 0.0
                )
                command_repeats = int(
                    _argv_value(generation_argv, "--dynamic-control-repeats")
                    or 0
                )
            except (
                OSError,
                UnicodeError,
                StopIteration,
                ValueError,
            ) as exc:
                errors.append(
                    "recorded run command could not be checked for action cadence: "
                    f"{type(exc).__name__}"
                )
            else:
                if (
                    action_hz is None
                    or not math.isclose(
                        command_action_hz,
                        action_hz,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or control_hz is None
                    or not math.isclose(
                        command_control_hz,
                        control_hz,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or command_frames != recorded_frames_budget
                    or command_frame_hz != 0.0
                    or command_repeats != 0
                ):
                    errors.append(
                        "policy-action playback contract differs from the "
                        "recorded generation command"
                    )

            replay_record = provenance_cadence.get("replay_summary")
            decoded_record = provenance_cadence.get("decoded_video")
            if not isinstance(replay_record, dict) or set(replay_record) != {
                "frame_hz_effective",
                "frame_dt_s",
                "output_fps",
                "frames",
                "rendered_frames",
                "playback_action_dt_s",
                "state_log_replay",
            }:
                errors.append("policy-action replay summary contract is invalid")
            else:
                try:
                    replay_summary = json.loads(
                        rerender_summary_path.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    errors.append(
                        "rerender summary could not be checked for action cadence: "
                        f"{type(exc).__name__}"
                    )
                else:
                    for key in (
                        "frame_hz_effective",
                        "frame_dt_s",
                        "output_fps",
                        "frames",
                        "rendered_frames",
                        "playback_action_dt_s",
                    ):
                        if replay_record.get(key) != replay_summary.get(key):
                            errors.append(
                                "policy-action replay summary differs at " + key
                            )
                    if (
                        _record_path(replay_record.get("state_log_replay"))
                        != state_log_path.resolve()
                        or _record_path(replay_summary.get("state_log_replay"))
                        != state_log_path.resolve()
                    ):
                        errors.append(
                            "policy-action replay summary state log is invalid"
                        )
                    replay_expected_values = {
                        "output_fps": replay_fps,
                        "frames": expected_frames,
                        "rendered_frames": expected_frames,
                    }
                    for key, expected_value in replay_expected_values.items():
                        if replay_record.get(key) != expected_value:
                            errors.append(
                                "policy-action replay summary contract differs at "
                                + key
                            )
                    replay_float_values = {
                        "frame_hz_effective": replay_hz,
                        "frame_dt_s": action_dt,
                        "playback_action_dt_s": action_dt,
                    }
                    for key, expected_value in replay_float_values.items():
                        raw = replay_record.get(key)
                        if (
                            expected_value is None
                            or isinstance(raw, bool)
                            or not isinstance(raw, (int, float))
                            or not math.isfinite(float(raw))
                            or not math.isclose(
                                float(raw),
                                expected_value,
                                rel_tol=0.0,
                                abs_tol=1e-6,
                            )
                        ):
                            errors.append(
                                "policy-action replay summary contract differs at "
                                + key
                            )
            if not isinstance(decoded_record, dict) or set(decoded_record) != {
                "fps",
                "frames",
                "duration_s",
            }:
                errors.append("policy-action decoded-video contract is invalid")
            else:
                timing = validation.get("timing")
                if not isinstance(timing, Mapping):
                    errors.append(
                        "video validation timing is missing for action cadence"
                    )
                else:
                    expected_decoded = {
                        "fps": timing.get("decoded_fps"),
                        "frames": validation.get("decoded_frames"),
                        "duration_s": timing.get("decoded_duration_s"),
                    }
                    if decoded_record != expected_decoded:
                        errors.append(
                            "policy-action decoded-video contract differs from "
                            "video validation"
                        )
                    decoded_fps = decoded_record.get("fps")
                    decoded_duration = decoded_record.get("duration_s")
                    if (
                        replay_hz is None
                        or isinstance(decoded_fps, bool)
                        or not isinstance(decoded_fps, (int, float))
                        or not math.isclose(
                            float(decoded_fps),
                            replay_hz,
                            rel_tol=0.0,
                            abs_tol=0.05,
                        )
                        or decoded_record.get("frames") != expected_frames
                        or replay_duration is None
                        or isinstance(decoded_duration, bool)
                        or not isinstance(decoded_duration, (int, float))
                        or not math.isclose(
                            float(decoded_duration),
                            replay_duration,
                            rel_tol=0.0,
                            abs_tol=max(0.001, replay_duration * 0.002),
                        )
                    ):
                        errors.append(
                            "policy-action decoded-video timing is inconsistent"
                        )
    return errors
