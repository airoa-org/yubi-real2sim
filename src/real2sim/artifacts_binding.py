# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Bind schema-v2 artifact roles to verified asset contracts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from real2sim.artifacts import (
    ArtifactRecord,
    ArtifactVerification,
    ManifestError,
    load_manifest,
    reuse_artifact_verification,
    verify_artifacts,
)
from real2sim.config._types import ConfigError
from real2sim.runner.argv import _extra_argument_value
from robosuite_env.contracts.assets import (
    AssetContractError,
    runtime_bundle_ids,
    validate_repository_assets,
)
from robosuite_env.contracts.scene import (
    SceneContractError,
    scene_tag,
    validate_scene_contract,
)


def _schema_v2_asset_bundle_ids(plan: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the exact repository bundles used by one schema-v2 run."""

    return runtime_bundle_ids(
        plan["resolved"]["robot"],
        plan["resolved"]["task"]["objects"],
    )

def _validate_schema_v2_contracts(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the sole asset and scene inputs used by an official run."""

    asset_root = Path(plan["repo_root"]) / "robosuite_env" / "assets"
    try:
        asset_report = validate_repository_assets(
            asset_root,
            bundle_ids=_schema_v2_asset_bundle_ids(plan),
        )
    except AssetContractError as exc:
        raise ConfigError(f"repository asset contract failed: {exc}") from exc

    scene_reports: dict[str, dict[str, Any]] = {}
    seen_scene_ids: set[str] = set()
    seen_visual_identities: set[tuple[str, str]] = set()
    for scene in plan["resolved"]["scenes"]:
        # The configured path is authoritative. The contract loader accepts
        # exactly that directory or background.json and never searches parents.
        try:
            report = validate_scene_contract(scene["path"], strict=True)
        except SceneContractError as exc:
            raise ConfigError(
                f"scene contract failed for {scene['tag']!r}: {exc}"
            ) from exc
        discovered_tag = scene_tag(report["name"])
        if discovered_tag != scene["tag"]:
            raise ConfigError(
                f"scenes tag {scene['tag']!r} does not match runner-discovered tag "
                f"{discovered_tag!r} from {report['manifest_path']}"
            )
        scene_id = str(report["id"])
        if scene_id in seen_scene_ids:
            raise ConfigError(
                f"official comparison contains duplicate scene id {scene_id!r}"
            )
        seen_scene_ids.add(scene_id)
        visual_identity = (
            str(report["ply_sha256"]),
            hashlib.sha256(
                json.dumps(
                    report["w2gs_matrix"].tolist(),
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        )
        if visual_identity in seen_visual_identities:
            raise ConfigError(
                "official comparison contains duplicate 3DGS visual content; "
                "scene id/name differences do not make a distinct background"
            )
        seen_visual_identities.add(visual_identity)
        scene_reports[scene["tag"]] = {
            "id": scene_id,
            "manifest_path": str(report["manifest_path"]),
            "contract_digest": str(report["contract_digest"]),
            "collision_tree_sha256": str(report["collision_tree_sha256"]),
            "collision_directory": str(report["collision_directory"]),
            "collision_files": [str(path) for path in report["collision_files"]],
            "ply_sha256": visual_identity[0],
            "w2gs_matrix_sha256": visual_identity[1],
        }

    table_hashes = {
        report["collision_tree_sha256"] for report in scene_reports.values()
    }
    if len(table_hashes) != 1:
        raise ConfigError(
            "all scenes in one official comparison must use the same table "
            "collision contract"
        )

    result = {
        "assets": {
            "manifest_path": str(asset_report["manifest_path"]),
            "contract_digest": str(asset_report["contract_digest"]),
            "bundles": list(_schema_v2_asset_bundle_ids(plan)),
        },
        "scenes": scene_reports,
    }
    return result

def _artifact_role_paths(
    plan: Mapping[str, Any],
    contract_report: Mapping[str, Any] | None = None,
) -> dict[str, tuple[Path, bool]]:
    """Return every artifact-backed runtime path and whether roots must match."""

    resolved = plan["resolved"]
    repo_root = Path(plan["repo_root"])
    roles: dict[str, tuple[Path, bool]] = {
        "checkpoint": (Path(resolved["policy"]["checkpoint"]), True),
        "runtime-assets": (repo_root, True),
        "robot-assets": (repo_root, True),
    }
    cached_scenes = (
        contract_report.get("scenes", {})
        if contract_report is not None
        else {}
    )
    for scene in resolved["scenes"]:
        cached = cached_scenes.get(scene["tag"])
        if isinstance(cached, Mapping) and isinstance(
            cached.get("manifest_path"), str
        ):
            scene_root = Path(cached["manifest_path"]).parent
        else:
            configured_scene = Path(scene["path"])
            scene_root = (
                configured_scene.parent
                if configured_scene.name == "background.json"
                else configured_scene
            )
        roles[f"scene.{scene['tag']}"] = (scene_root, True)
    episode = resolved["task"]["policy_image_episode"]
    if episode is not None:
        roles["policy-episode"] = (Path(episode), True)
    init_state = resolved["simulation"]["init_state"]
    if init_state is not None:
        roles["initial-state"] = (Path(init_state), True)
    desk_value = _extra_argument_value(resolved["simulation"]["cli_args"], "--desk")
    artifact_bindings = (
        resolved.get("reproducibility", {}).get("artifacts", {})
        if resolved.get("reproducibility") is not None
        else {}
    )
    if desk_value == "scene":
        external_collision_roots: set[Path] = set()
        for scene in resolved["scenes"]:
            cached = cached_scenes.get(scene["tag"])
            if isinstance(cached, Mapping):
                collision_directory = cached.get("collision_directory")
                manifest_path = cached.get("manifest_path")
                if not isinstance(collision_directory, str) or not isinstance(
                    manifest_path, str
                ):
                    raise ConfigError(
                        "scene preflight report lacks manifest/collision path: "
                        f"{scene['tag']}"
                    )
            else:
                try:
                    report = validate_scene_contract(scene["path"], strict=True)
                except SceneContractError as exc:
                    raise ConfigError(
                        f"scene contract failed for {scene['tag']!r}: {exc}"
                    ) from exc
                collision_directory = str(report["collision_directory"])
                manifest_path = str(report["manifest_path"])
            collision_path = Path(collision_directory).resolve()
            scene_root = Path(manifest_path).resolve().parent
            try:
                collision_path.relative_to(scene_root)
            except ValueError:
                external_collision_roots.add(collision_path.parent)
        if external_collision_roots:
            if "registered-desk" not in artifact_bindings:
                raise ConfigError(
                    "scene table collision outside the scene artifact requires "
                    "a reproducibility.artifacts registered-desk binding"
                )
            if len(external_collision_roots) != 1:
                raise ConfigError(
                    "registered-desk artifact requires all external table "
                    "collisions to use one bundle"
                )
            roles["registered-desk"] = (
                next(iter(external_collision_roots)),
                True,
            )
    elif desk_value not in {None, "robosuite", "scene"}:
        desk_path = Path(desk_value)
        if not desk_path.is_absolute():
            desk_path = repo_root / desk_path
        roles["registered-desk"] = (Path(os.path.abspath(desk_path)), False)
    return roles

def verify_plan_artifacts(
    plan: Mapping[str, Any],
    contract_report: Mapping[str, Any] | None = None,
    *,
    artifact_verification: ArtifactVerification | None = None,
) -> tuple[dict[str, Any] | None, list[ArtifactRecord]]:
    """Verify schema-v2 bindings and return manifest plus immutable records."""

    resolved = plan["resolved"]
    reproducibility = resolved.get("reproducibility")
    if reproducibility is None:
        return None, []
    try:
        manifest = load_manifest(reproducibility["artifact_manifest"])
        bindings = reproducibility["artifacts"]
        role_paths = _artifact_role_paths(plan, contract_report)
        provenance_roles = {
            role for role in bindings if role.startswith("provenance.")
        }
        runtime_bindings = {
            role: artifact_id
            for role, artifact_id in bindings.items()
            if role not in provenance_roles
        }
        unknown_roles = sorted(set(runtime_bindings) - set(role_paths))
        missing_roles = sorted(set(role_paths) - set(runtime_bindings))
        if unknown_roles:
            raise ConfigError(
                "reproducibility.artifacts contains roles unused by this config: "
                + ", ".join(unknown_roles)
            )
        if missing_roles:
            raise ConfigError(
                "reproducibility.artifacts is missing runtime input roles: "
                + ", ".join(missing_roles)
            )
        selected_ids = list(dict.fromkeys(bindings.values()))
        if artifact_verification is None:
            records = verify_artifacts(
                manifest,
                selected_ids,
                repo_root=Path(plan["repo_root"]),
                environment=reproducibility["environment"],
            )
        else:
            records = reuse_artifact_verification(
                artifact_verification,
                manifest,
                selected_ids,
                repo_root=Path(plan["repo_root"]),
                environment=reproducibility["environment"],
            )
    except ManifestError as exc:
        raise ConfigError(f"artifact manifest validation failed: {exc}") from exc

    by_id = {record.artifact_id: record for record in records}
    exact_root_roles = {"checkpoint", "policy-episode", "initial-state"} | {
        role for role in role_paths if role.startswith("scene.")
    }
    for role, (expected_path, require_exact) in role_paths.items():
        record = by_id[runtime_bindings[role]]
        installed = Path(record.path)
        expected = Path(os.path.abspath(expected_path))
        if require_exact or role in exact_root_roles or record.kind == "file":
            matches = installed == expected
        else:
            try:
                expected.relative_to(installed)
                matches = True
            except ValueError:
                matches = False
        if not matches:
            raise ConfigError(
                f"artifact role {role} resolves to {installed}, but the config uses {expected}"
            )
    if reproducibility["production"]:
        fixture_ids = sorted(
            record.artifact_id for record in records if record.access == "fixture"
        )
        if fixture_ids:
            raise ConfigError(
                "production run cannot use tiny fixture artifacts: " + ", ".join(fixture_ids)
            )
    return manifest, records
