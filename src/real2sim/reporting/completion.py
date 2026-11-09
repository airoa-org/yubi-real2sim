# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Rollout completion and output contract checks."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from real2sim._bootstrap import REPO_ROOT
from real2sim.config.paths import GRID_TEMPLATES, RUN_MANIFEST_FILE
from real2sim.reporting.posthoc_provenance import (
    _record_path,
    _validate_posthoc_provenance,
)

def rollout_completion_errors(plan: Mapping[str, Any]) -> list[str]:
    """Return reasons why the runner output is not one complete evaluation.

    The aggregate summary intentionally tolerates missing per-run files so it
    can describe partial experiments. A configured evaluation, however,
    must not be reported as completed unless every expected worker exited
    cleanly and wrote a readable full-length summary. Final video intent is
    independent of whether frames were rendered live or in the required
    post-hoc pass; explicitly disabled video is the only video-free state.
    """

    run_root = Path(plan["run_root"])
    # The criterion a run must have been scored by. "auto"/none leaves it to
    # the worker and is not checked.
    _goal = str(((plan["resolved"].get("task") or {}).get("goal_condition")) or "").strip()
    expected_criterion = _goal if _goal and _goal.lower() not in {"auto", "none"} else None
    render = plan["resolved"]["render"]
    live_render = bool(render["enabled"])
    posthoc_required = bool(
        render.get("posthoc_required", render.get("forced_posthoc", False))
    )
    video_required = bool(live_render or posthoc_required)
    outputs_path = run_root / "outputs.csv"
    if not outputs_path.is_file():
        return [f"missing runner output table: {outputs_path}"]
    try:
        with outputs_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            required_columns = {"scene", "run", "exit_code", "video", "summary"}
            if posthoc_required:
                required_columns.add("video_validation")
            if reader.fieldnames is None or not required_columns.issubset(
                reader.fieldnames
            ):
                return ["outputs.csv is missing required completion columns"]
            rows = list(reader)
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        return [f"could not read outputs.csv: {type(exc).__name__}: {exc}"]

    repeats = int(plan["resolved"]["runner"]["repeats"])
    expected_frames = int(plan["resolved"]["runner"]["frames"])
    expected_keys = {
        (scene["tag"], f"run_{repeat:02d}")
        for scene in plan["resolved"]["scenes"]
        for repeat in range(1, repeats + 1)
    }
    configured_scene_manifests: dict[str, Path] = {}
    for scene_config in plan["resolved"]["scenes"]:
        scene_path = Path(scene_config["path"]).expanduser()
        if not scene_path.is_absolute():
            scene_path = Path(plan["repo_root"]) / scene_path
        scene_path = scene_path.resolve()
        configured_scene_manifests[scene_config["tag"]] = (
            scene_path / "background.json" if scene_path.is_dir() else scene_path
        )
    observed: dict[tuple[str, str], dict[str, str]] = {}
    errors: list[str] = []
    expected_asset_digest: str | None = None
    expected_scene_digests: dict[str, str] = {}
    expected_scene_ids: dict[str, str] = {}
    expected_scene_manifests: dict[str, Path] = {}
    expected_scene_ply_hashes: dict[str, str] = {}
    run_manifest: Mapping[str, Any] | None = None
    official_manifest_path = run_root / RUN_MANIFEST_FILE
    try:
        run_manifest = json.loads(
            official_manifest_path.read_text(encoding="utf-8")
        )
        contracts = run_manifest["preflight"]["contracts"]
        raw_asset_digest = contracts["assets"]["contract_digest"]
        raw_scenes = contracts["scenes"]
        if (
            not isinstance(raw_asset_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", raw_asset_digest) is None
            or not isinstance(raw_scenes, dict)
        ):
            raise ValueError("invalid contract digest structure")
        expected_asset_digest = raw_asset_digest
        for tag, report in raw_scenes.items():
            if not isinstance(tag, str) or not isinstance(report, dict):
                raise ValueError("invalid scene contract structure")
            digest = report.get("contract_digest")
            scene_id = report.get("id")
            manifest_path = report.get("manifest_path")
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or not isinstance(scene_id, str)
                or not scene_id
                or not isinstance(manifest_path, str)
                or not manifest_path
            ):
                raise ValueError(f"invalid scene contract record for {tag!r}")
            expected_scene_digests[tag] = digest
            expected_scene_ids[tag] = scene_id
            expected_scene_manifests[tag] = Path(manifest_path).resolve()
            ply_hash = report.get("ply_sha256")
            if (
                not isinstance(ply_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", ply_hash) is None
            ):
                raise ValueError(f"invalid scene PLY digest for {tag!r}")
            expected_scene_ply_hashes[tag] = ply_hash
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(
            "official run manifest contracts are missing or unreadable: "
            f"{type(exc).__name__}: {exc}"
        )
    for row_index, row in enumerate(rows, start=2):
        key = (str(row.get("scene") or ""), str(row.get("run") or ""))
        if key not in expected_keys:
            errors.append(f"outputs.csv row {row_index} has unexpected rollout {key!r}")
            continue
        if key in observed:
            errors.append(f"outputs.csv contains duplicate rollout {key!r}")
            continue
        observed[key] = row

    for scene, run in sorted(expected_keys - set(observed)):
        errors.append(f"outputs.csv is missing rollout {scene}/{run}")

    for key in sorted(expected_keys & set(observed)):
        scene, run = key
        row = observed[key]
        label = f"{scene}/{run}"
        run_dir = run_root / scene / run
        rollout_expected_frames = expected_frames
        expected_summary = run_dir / "sim_summary.json"
        expected_video = run_dir / f"{scene}_{run}.mp4"
        expected_rerender_validation = run_dir / "rerender" / "video_validation.json"
        expected_rerender_provenance = run_dir / "rerender" / "provenance.json"
        if row.get("exit_code") != "0":
            errors.append(f"{label} has worker exit_code={row.get('exit_code')!r}")
        if row.get("summary") != str(expected_summary):
            errors.append(f"{label} has an unexpected summary path")
        if row.get("video") != str(expected_video):
            errors.append(f"{label} has an unexpected video path")

        summary: Any = None
        summary_read = False
        try:
            summary = json.loads(expected_summary.read_text(encoding="utf-8"))
            summary_read = True
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(
                f"{label} summary is missing or unreadable: {type(exc).__name__}"
            )
        if isinstance(summary, dict):
            # A run is only scored if it says so: a summary without a boolean
            # verdict is an error, not an ordinary failure.
            for field in ("success", "new_final_success"):
                if field in summary and not isinstance(summary[field], bool):
                    errors.append(f"{label} {field} is not a boolean")
            if not isinstance(summary.get("success"), bool):
                errors.append(f"{label} has no boolean success verdict")
            if expected_criterion:
                criterion = summary.get("success_criterion")
                criterion_name = (
                    criterion.get("name") if isinstance(criterion, dict) else None
                )
                if criterion_name != expected_criterion:
                    errors.append(
                        f"{label} was scored by {criterion_name!r}, "
                        f"expected {expected_criterion!r}"
                    )
            # The start-state check records what it found. A start the guard
            # reports as penetrating is not the scene the profile asked for.
            start_state = summary.get("start_state")
            if isinstance(start_state, dict) and start_state.get("overlaps"):
                errors.append(
                    f"{label} started with {len(start_state['overlaps'])} "
                    "penetrating contact(s) beyond tolerance"
                )
            if summary.get("execution_class") != "official":
                errors.append(
                    f"{label} execution_class={summary.get('execution_class')!r}, "
                    "expected official"
                )
            asset_digest = summary.get("asset_contract_sha256")
            if (
                not isinstance(asset_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", asset_digest) is None
            ):
                errors.append(f"{label} has no valid asset contract SHA-256")
            elif (
                expected_asset_digest is not None
                and asset_digest != expected_asset_digest
            ):
                errors.append(
                    f"{label} asset contract differs from launch preflight"
                )
            summary_manifest = summary.get("official_run_manifest")
            valid_summary_manifest = False
            if isinstance(summary_manifest, str):
                try:
                    valid_summary_manifest = (
                        Path(summary_manifest).resolve()
                        == official_manifest_path.resolve()
                    )
                except (OSError, RuntimeError, ValueError):
                    pass
            if not valid_summary_manifest:
                errors.append(
                    f"{label} does not reference the launcher run manifest"
                )
            worker_contract = summary.get("official_worker_contract")
            expected_worker_contract = (
                run_root / scene / run / "official_worker_contract.json"
            ).resolve()
            valid_worker_contract = False
            worker_contract_error: str | None = None
            if isinstance(worker_contract, str):
                try:
                    worker_contract_path = Path(worker_contract).resolve()
                except (
                    OSError,
                    UnicodeError,
                ):
                    pass
                else:
                    if worker_contract_path == expected_worker_contract:
                        try:
                            from robosuite_env.contracts.worker import (
                                WorkerContractError,
                                validate_worker_contract,
                            )
                        except ImportError as exc:
                            worker_contract_error = (
                                f"worker contract validator import failed: {exc}"
                            )
                        else:
                            try:
                                worker_payload = json.loads(
                                    worker_contract_path.read_text(encoding="utf-8")
                                )
                                worker_argv = worker_payload.get("argv", [])
                                validate_worker_contract(
                                    worker_contract_path,
                                    official_manifest_path,
                                    worker_argv,
                                )
                                valid_worker_contract = True
                            except WorkerContractError as exc:
                                worker_contract_error = f"worker contract invalid: {exc}"
                            except (
                                OSError,
                                UnicodeError,
                                json.JSONDecodeError,
                                TypeError,
                                ValueError,
                                RuntimeError,
                            ):
                                pass
            if not valid_worker_contract:
                if worker_contract_error is None:
                    worker_contract_error = "has no valid launcher-owned worker contract"
                errors.append(f"{label} {worker_contract_error}")
            scene_contract = summary.get("scene_contract")
            if not isinstance(scene_contract, dict):
                errors.append(f"{label} has no scene contract provenance")
            else:
                scene_digest = scene_contract.get("sha256")
                if (
                    not isinstance(scene_digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", scene_digest) is None
                ):
                    errors.append(f"{label} has no valid scene contract SHA-256")
                elif (
                    scene in expected_scene_digests
                    and scene_digest != expected_scene_digests[scene]
                ):
                    errors.append(
                        f"{label} scene contract differs from launch preflight"
                    )
                if expected_scene_digests and scene not in expected_scene_digests:
                    errors.append(
                        f"{label} has no matching scene in launch preflight"
                    )
                elif scene in expected_scene_digests:
                    if scene_contract.get("id") != expected_scene_ids[scene]:
                        errors.append(
                            f"{label} scene id differs from launch preflight"
                        )
                    summary_scene_manifest = scene_contract.get("manifest")
                    valid_scene_manifest = False
                    if isinstance(summary_scene_manifest, str):
                        try:
                            valid_scene_manifest = (
                                Path(summary_scene_manifest).resolve()
                                == expected_scene_manifests[scene]
                            )
                        except (OSError, RuntimeError, ValueError):
                            pass
                    if not valid_scene_manifest:
                        errors.append(
                            f"{label} scene manifest differs from launch preflight"
                        )
                if scene_contract.get("table_appearance") != "embedded_in_3dgs":
                    errors.append(
                        f"{label} does not declare table appearance embedded in 3DGS"
                    )
                collision_parts = scene_contract.get("table_collision_parts")
                if (
                    isinstance(collision_parts, bool)
                    or not isinstance(collision_parts, int)
                    or collision_parts <= 0
                ):
                    errors.append(
                        f"{label} has invalid table collision part count"
                    )
            frames = summary.get("frames")
            rendered_frames = summary.get("rendered_frames")
            wall_stopped_early = summary.get("wall_stopped_early")
            if isinstance(frames, bool) or frames != rollout_expected_frames:
                errors.append(
                    f"{label} frames={frames!r}, expected {rollout_expected_frames}"
                )
            if live_render and (
                isinstance(rendered_frames, bool)
                or not isinstance(rendered_frames, int)
                or rendered_frames < rollout_expected_frames
            ):
                errors.append(
                    f"{label} rendered_frames={rendered_frames!r}, "
                    f"expected at least {rollout_expected_frames}"
                )
            if wall_stopped_early is not False:
                errors.append(
                    f"{label} wall_stopped_early={wall_stopped_early!r}, expected false"
                )
        elif summary_read:
            # Read but not an object -- JSON null included.
            errors.append(f"{label} summary must contain a JSON object")

        if posthoc_required:
            if row.get("video_validation") != str(expected_rerender_validation):
                errors.append(
                    f"{label} outputs.csv does not reference rerender/video_validation.json"
                )
            validation: Any = None
            try:
                validation = json.loads(
                    expected_rerender_validation.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append(
                    f"{label} rerender video validation is missing or unreadable: "
                    f"{type(exc).__name__}"
                )
            if isinstance(validation, dict):
                if validation.get("ok") is not True:
                    errors.append(
                        f"{label} rerender video validation is not successful"
                    )
                if _record_path(validation.get("video")) != expected_video.resolve():
                    errors.append(
                        f"{label} rerender validation does not reference the canonical video"
                    )
                expected_rerender_summary = run_dir / "rerender" / "sim_summary.json"
                if (
                    _record_path(validation.get("summary"))
                    != expected_rerender_summary.resolve()
                ):
                    errors.append(
                        f"{label} rerender validation uses an unexpected summary"
                    )
                if validation.get("decoded_frames") != rollout_expected_frames:
                    errors.append(
                        f"{label} rerender decoded_frames="
                        f"{validation.get('decoded_frames')!r}, expected "
                        f"{rollout_expected_frames}"
                    )
                decoded_hash = validation.get("decoded_sha256")
                if (
                    not isinstance(decoded_hash, str)
                    or re.fullmatch(r"[0-9a-f]{64}", decoded_hash) is None
                ):
                    errors.append(f"{label} rerender decoded SHA-256 is invalid")
                try:
                    video_size = expected_video.stat().st_size
                except OSError:
                    video_size = None
                if (
                    video_size is not None
                    and validation.get("file_size_bytes") != video_size
                ):
                    errors.append(
                        f"{label} rerender validation file size differs from "
                        "the canonical video"
                    )
            elif validation is not None:
                errors.append(
                    f"{label} rerender video validation must be a JSON object"
                )

            expected_scene_manifest = expected_scene_manifests.get(
                scene, configured_scene_manifests[scene]
            )
            errors.extend(
                f"{label} {error}"
                for error in _validate_posthoc_provenance(
                    provenance_path=expected_rerender_provenance,
                    validation_path=expected_rerender_validation,
                    validation=(validation if isinstance(validation, dict) else {}),
                    expected_video=expected_video,
                    expected_summary=expected_summary,
                    expected_frames=rollout_expected_frames,
                    configured_frames=expected_frames,
                    run_dir=run_dir,
                    scene_tag_value=scene,
                    official_manifest_path=official_manifest_path,
                    run_manifest=run_manifest,
                    expected_scene_id=expected_scene_ids.get(scene),
                    expected_scene_digest=expected_scene_digests.get(scene),
                    expected_scene_manifest=expected_scene_manifest,
                    expected_scene_ply_sha256=expected_scene_ply_hashes.get(scene),
                )
            )

        if video_required:
            try:
                if not expected_video.is_file() or expected_video.stat().st_size <= 0:
                    errors.append(f"{label} video is missing or empty")
            except OSError as exc:
                errors.append(
                    f"{label} video could not be inspected: {type(exc).__name__}"
                )
    rollout_count = int(plan["resolved"]["runner"]["repeats"])
    if video_required and rollout_count in GRID_TEMPLATES:
        expected_template = Path(
            plan["resolved"]["render"]["grid_template"]
        ).resolve()
        scenes = plan["resolved"]["scenes"]
        for scene in scenes:
            grid_root = run_root if len(scenes) == 1 else run_root / scene["tag"]
            grid_video = grid_root / f"rollout_grid_{rollout_count}.mp4"
            grid_poster = grid_root / f"rollout_grid_{rollout_count}_poster.jpg"
            grid_manifest = grid_root / f"rollout_grid_{rollout_count}.json"
            for label, path in (
                ("video", grid_video),
                ("poster", grid_poster),
                ("manifest", grid_manifest),
            ):
                try:
                    if not path.is_file() or path.stat().st_size <= 0:
                        errors.append(
                            f"required {scene['tag']} grid {label} is missing or empty: {path}"
                        )
                except OSError as exc:
                    errors.append(
                        f"required {scene['tag']} grid {label} could not be inspected: "
                        f"{type(exc).__name__}"
                    )
            if grid_manifest.is_file():
                try:
                    grid = json.loads(grid_manifest.read_text(encoding="utf-8"))
                    template_bytes = expected_template.read_bytes()
                    wanted_hash = hashlib.sha256(template_bytes).hexdigest()
                    raw_template_path = Path(str(grid.get("template_path", "")))
                    manifest_template_path = (
                        raw_template_path.resolve()
                        if raw_template_path.is_absolute()
                        else (REPO_ROOT / raw_template_path).resolve()
                    )
                    if manifest_template_path != expected_template:
                        errors.append("grid manifest uses an unexpected template path")
                    if grid.get("template_sha256") != wanted_hash:
                        errors.append("grid manifest template SHA-256 is invalid")
                    if grid.get("geometry", {}).get("rollout_cells") != rollout_count:
                        errors.append("grid manifest rollout count is invalid")
                    if grid.get("render_contract", {}).get("template_only") is not True:
                        errors.append("grid manifest is not template-only")
                    # The canonical renderer is always invoked with an exact
                    # scene selector, including single-scene runs.
                    expected_scene = scene["tag"]
                    if grid.get("scene") != expected_scene:
                        errors.append("grid manifest scene selection is invalid")
                    # The clear-headers templates title the grid with the
                    # scene tag; the policy-eval template used for post-hoc
                    # finalization of 10-rollout batches titles it with the run name.
                    if grid.get("title") not in {expected_scene, run_root.name}:
                        errors.append("grid manifest scene title is invalid")
                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ):
                    errors.append("required grid manifest is unreadable or invalid")
    return errors
