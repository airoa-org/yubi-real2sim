# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed validation for repository-owned simulation asset bundles.

The manifest describes logical bundles while the MJCF remains the source of
truth for the exact files that MuJoCo loads.  This keeps asset updates to one
manifest edit and prevents visual and collision meshes from being swapped.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping
import xml.etree.ElementTree as ET

from robosuite_env.contracts.provenance import sha256_file


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROLES = frozenset({"visual", "collision"})

#: Top-level manifest section pinning each object bundle's runtime mesh
#: set (bundle id -> digest); see :func:`runtime_mesh_set_sha256`. It sits
#: outside the bundle definitions so adding a pin leaves every contract
#: digest unchanged: the digest already covers each selected file's bytes.
RUNTIME_MESH_SET_SECTION = "runtime_mesh_set_sha256"


_CUP_RUNTIME_GEOMETRY: Mapping[str, Any] = MappingProxyType(
    {
        "source_collision_geometry": MappingProxyType(
            {
                "bounds_min_m": (
                    -0.0365698154612,
                    -0.0366110705429,
                    -0.0003977204099,
                ),
                "bounds_max_m": (
                    0.0365501498264,
                    0.0365593006575,
                    0.0846023112278,
                ),
                "z_min_m": -0.0003977204099,
                "z_max_m": 0.0846023112278,
                "xy_radius_m": 0.0368424624415595,
            }
        ),
        "compiled_collision_observed": MappingProxyType(
            {
                "bounds_min_m": (
                    -0.03656981502,
                    -0.03661107137,
                    -0.00039772038,
                ),
                "bounds_max_m": (
                    0.03655015037,
                    0.03655929975,
                    0.08460230953,
                ),
                "z_min_m": -0.00039772038,
                "z_max_m": 0.08460230953,
                "xy_radius_m": 0.03684246303,
            }
        ),
        "semantic_sites_m": MappingProxyType(
            {
                "bottom_site": (0.0, 0.0, 0.0),
                "top_site": (0.0, 0.0, 0.08274),
                "horizontal_radius_site": (0.0355, 0.0, 0.0),
            }
        ),
    }
)
_PLATE_RUNTIME_GEOMETRY: Mapping[str, Any] = MappingProxyType(
    {
        "source_collision_geometry": MappingProxyType(
            {
                "bounds_min_m": (-0.0900072699960, -0.0900020114997, -0.0006518344298),
                "bounds_max_m": (0.0899927975995, 0.0899949425125, 0.0196416020400),
                "z_min_m": -0.0006518344298,
                "z_max_m": 0.0196416020400,
                "xy_radius_m": 0.0902683802941864,
            }
        ),
        "compiled_collision_observed": MappingProxyType(
            {
                "bounds_min_m": (
                    -0.09000727343,
                    -0.09000201499,
                    -0.00065183445,
                ),
                "bounds_max_m": (
                    0.08999279623,
                    0.08999494445,
                    0.01964160098,
                ),
                "z_min_m": -0.00065183445,
                "z_max_m": 0.01964160098,
                "xy_radius_m": 0.09026837957,
            }
        ),
        "semantic_sites_m": MappingProxyType(
            {
                "bottom_site": (0.0, 0.0, 0.0),
                "top_site": (0.0, 0.0, 0.018912),
                "horizontal_radius_site": (0.089371, 0.0, 0.0),
            }
        ),
    }
)

# Source collision extrema are computed from the canonical OBJ vertices after
# The four office objects the box-packing task manipulates. Their geometry is
# generated from the TRELLIS.2 reconstructions by
# scripts/build_office_object_assets.py and mirrors
# robosuite_env/assets/manifest.json.
#
# The box mesh is scaled to 11/14 of its original height, at
# 0.315980 x 0.315980 x 0.248269 over 23 collision hulls. Each hull has its own
# density, solved so the 0.100 kg box's centre of mass lands 33 mm up instead
# of the 87 mm a uniform density gave, which is what kept the box from tipping
# off the desk. The hulls leave about 7% of the cavity floor uncovered. A 24th
# collision geom, the zero-mass 3 mm base slab box_023 from the Flat lineage,
# closes that floor: an object over such a gap rests on the slab instead of on
# the table under the box, 2.9 mm higher. The slab adds no mass and lies inside
# the extents below, which include the flared dust flaps (x), the lid leaning
# back (y) and the lid top (z).
_BOX_RUNTIME_GEOMETRY: Mapping[str, Any] = MappingProxyType(
    {
        "source_collision_geometry": MappingProxyType(
            {
                "bounds_min_m": (
                    -0.1442892431,
                    -0.1590984104,
                    0.0,
                ),
                "bounds_max_m": (
                    0.1442892431,
                    0.1590984104,
                    0.2319938796,
                ),
                "z_min_m": 0.0,
                "z_max_m": 0.2319938796,
                "xy_radius_m": 0.2092299346154,
            }
        ),
        "compiled_collision_observed": MappingProxyType(
            {
                "bounds_min_m": (
                    -0.14428924061,
                    -0.15909841343,
                    -1.79e-09,
                ),
                "bounds_max_m": (
                    0.14428924085,
                    0.15909841062,
                    0.23199387932,
                ),
                "z_min_m": -1.79e-09,
                "z_max_m": 0.23199387932,
                "xy_radius_m": 0.20922993309,
            }
        ),
        "semantic_sites_m": MappingProxyType(
            {
                "bottom_site": (0.0, 0.0, 0.0),
                "top_site": (0.0, 0.0, 0.240357),
                "horizontal_radius_site": (0.1591, 0.0, 0.0),
            }
        ),
    }
)


_TAPE_RUNTIME_GEOMETRY: Mapping[str, Any] = MappingProxyType(
    {
        "source_collision_geometry": MappingProxyType(
            {
                "bounds_min_m": (
                    -0.0402294178,
                    -0.0403202454,
                    0.0,
                ),
                "bounds_max_m": (
                    0.0402294178,
                    0.0403202454,
                    0.0517162724,
                ),
                "z_min_m": 0.0,
                "z_max_m": 0.0517162724,
                "xy_radius_m": 0.0410464225071,
            }
        ),
        "compiled_collision_observed": MappingProxyType(
            {
                "bounds_min_m": (
                    -0.04022941577,
                    -0.04032024545,
                    -4.2e-10,
                ),
                "bounds_max_m": (
                    0.04022941558,
                    0.04032024545,
                    0.0517162705,
                ),
                "z_min_m": -4.2e-10,
                "z_max_m": 0.0517162705,
                "xy_radius_m": 0.041046422,
            }
        ),
        "semantic_sites_m": MappingProxyType(
            {
                "bottom_site": (0.0, 0.0, 0.0),
                "top_site": (0.0, 0.0, 0.0517),
                "horizontal_radius_site": (0.0403, 0.0, 0.0),
            }
        ),
    }
)


_PEN_RUNTIME_GEOMETRY: Mapping[str, Any] = MappingProxyType(
    {
        "source_collision_geometry": MappingProxyType(
            {
                "bounds_min_m": (
                    -0.0089154115,
                    -0.0732479427,
                    0.0,
                ),
                "bounds_max_m": (
                    0.0089154115,
                    0.0732479427,
                    0.0176485748,
                ),
                "z_min_m": 0.0,
                "z_max_m": 0.0176485748,
                "xy_radius_m": 0.0732783284129,
            }
        ),
        "compiled_collision_observed": MappingProxyType(
            {
                "bounds_min_m": (
                    -0.00891541152,
                    -0.0732479382,
                    6e-11,
                ),
                "bounds_max_m": (
                    0.00891541169,
                    0.0732479389,
                    0.01764857448,
                ),
                "z_min_m": 6e-11,
                "z_max_m": 0.01764857448,
                "xy_radius_m": 0.07327832391,
            }
        ),
        "semantic_sites_m": MappingProxyType(
            {
                "bottom_site": (0.0, 0.0, 0.0),
                "top_site": (0.0, 0.0, 0.0176),
                "horizontal_radius_site": (0.0732, 0.0, 0.0),
            }
        ),
    }
)


_LAN_RUNTIME_GEOMETRY: Mapping[str, Any] = MappingProxyType(
    {
        "source_collision_geometry": MappingProxyType(
            {
                "bounds_min_m": (
                    -0.0497167053,
                    -0.0535337693,
                    0.0,
                ),
                "bounds_max_m": (
                    0.0497167053,
                    0.0535337693,
                    0.0117789879,
                ),
                "z_min_m": 0.0,
                "z_max_m": 0.0117789879,
                "xy_radius_m": 0.068011231936,
            }
        ),
        "compiled_collision_observed": MappingProxyType(
            {
                "bounds_min_m": (
                    -0.04971670606,
                    -0.05353376862,
                    7e-11,
                ),
                "bounds_max_m": (
                    0.04971670613,
                    0.05353376823,
                    0.01177898759,
                ),
                "z_min_m": 7e-11,
                "z_max_m": 0.01177898759,
                "xy_radius_m": 0.06801123509,
            }
        ),
        "semantic_sites_m": MappingProxyType(
            {
                "bottom_site": (0.0, 0.0, 0.0),
                "top_site": (0.0, 0.0, 0.0118),
                "horizontal_radius_site": (0.0535, 0.0, 0.0),
            }
        ),
    }
)




# MJCF mesh scale, in object-local metres.  The compiled observation records
# MuJoCo's float32 representation separately.  Aliases deliberately share the
# same immutable mapping because colour-only variants cannot acquire
# independent placement or success geometry.
OBJECT_RUNTIME_GEOMETRY: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "cup": _CUP_RUNTIME_GEOMETRY,
        "plate": _PLATE_RUNTIME_GEOMETRY,
        "box": _BOX_RUNTIME_GEOMETRY,
        "tape": _TAPE_RUNTIME_GEOMETRY,
        "pen": _PEN_RUNTIME_GEOMETRY,
        "lan": _LAN_RUNTIME_GEOMETRY,
    }
)


class AssetContractError(ValueError):
    """Raised when repository assets do not satisfy their manifest."""


def runtime_geometry_for_object(object_key: str) -> Mapping[str, Any]:
    """Return immutable object-local placement and success geometry."""

    normalized = str(object_key).strip().lower()
    try:
        return OBJECT_RUNTIME_GEOMETRY[normalized]
    except KeyError as exc:
        raise AssetContractError(
            f"no runtime geometry is registered for object: {object_key!r}"
        ) from exc


def runtime_bundle_ids(robot: str, objects: str) -> tuple[str, ...]:
    """Return the central asset bundles required by one simulation."""

    normalized_robot = str(robot).strip().lower()
    from robosuite_env.robots.profiles import SUPPORTED_ROBOT_NAMES

    if normalized_robot not in set(SUPPORTED_ROBOT_NAMES):
        raise AssetContractError(f"unsupported simulation robot: {robot!r}")
    bundles = ["yubi_right", "yubi_left"]
    from robosuite_env.robots.profiles import robot_embodiment

    if robot_embodiment(normalized_robot) == "fr3":
        # Both FR3 rigs draw on the same tracked arm-geometry bundle.
        bundles.append("fr3")
    normalized_objects = str(objects).strip().lower()
    if normalized_objects == "box_pack":
        # Box packing manipulates its own four objects and never the cup/plate
        # pair, so its bundle set replaces them rather than extending them.
        bundles.extend(("box", "tape", "pen", "lan"))
    else:
        bundles.extend(("cup", "plate"))
    return tuple(bundles)


def _sha256(path: Path) -> str:
    """Hash one asset file, reporting unreadable files as a contract failure."""

    try:
        return sha256_file(path)
    except OSError as exc:
        raise AssetContractError(f"asset file is unreadable: {path}: {exc}") from exc


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AssetContractError(f"asset manifest is unreadable: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise AssetContractError(f"asset manifest root must be an object: {path}")
    if raw.get("schema_version") != 1:
        raise AssetContractError(
            f"asset manifest schema_version must be 1: {path}"
        )
    if not isinstance(raw.get("bundles"), dict) or not raw["bundles"]:
        raise AssetContractError(f"asset manifest bundles must be a non-empty object: {path}")
    if not isinstance(raw.get("forbidden"), dict):
        raise AssetContractError(f"asset manifest forbidden must be an object: {path}")
    return raw


def _relative_file(
    raw_path: str,
    *,
    base: Path,
    asset_root: Path,
    context: str,
) -> tuple[Path, str]:
    raw_path = raw_path.strip()
    candidate = Path(raw_path)
    if not raw_path or candidate.is_absolute() or raw_path.startswith("~"):
        raise AssetContractError(
            f"{context} must use a non-empty path relative to the asset root: {raw_path!r}"
        )
    if ".." in candidate.parts:
        raise AssetContractError(
            f"{context} escapes the asset root with '..': {raw_path}"
        )
    resolved = (base / candidate).resolve()
    try:
        relative = resolved.relative_to(asset_root)
    except ValueError as exc:
        raise AssetContractError(
            f"{context} escapes the asset root {asset_root}: {raw_path}"
        ) from exc
    if not resolved.is_file():
        raise AssetContractError(f"{context} is missing: {resolved}")
    return resolved, relative.as_posix()


def _normalized_prefixes(value: Any, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise AssetContractError(f"{context} must be a non-empty array")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise AssetContractError(f"{context} entries must be non-empty strings")
        path = Path(item)
        if path.is_absolute() or ".." in path.parts or item.startswith("~"):
            raise AssetContractError(f"{context} entry escapes the asset root: {item}")
        prefix = path.as_posix().rstrip("/") + "/"
        normalized.append(prefix)
    return tuple(normalized)


def _contact_value(geom: ET.Element, key: str, *, context: str) -> int:
    raw = geom.get(key, "1")
    try:
        return int(raw)
    except ValueError as exc:
        raise AssetContractError(
            f"{context} geom {geom.get('name', '<unnamed>')} has invalid {key}={raw!r}"
        ) from exc


def _classify_meshes(
    root: ET.Element,
    *,
    context: str,
) -> tuple[dict[str, ET.Element], dict[str, str]]:
    meshes: dict[str, ET.Element] = {}
    for mesh in root.findall("./asset/mesh"):
        name = (mesh.get("name") or "").strip()
        if not name:
            raise AssetContractError(f"{context} contains an unnamed mesh asset")
        if name in meshes:
            raise AssetContractError(f"{context} declares mesh {name!r} more than once")
        meshes[name] = mesh
    if not meshes:
        raise AssetContractError(f"{context} declares no mesh assets")

    observed: dict[str, set[str]] = {name: set() for name in meshes}
    for geom in root.iter("geom"):
        mesh_name = geom.get("mesh")
        if mesh_name is None:
            continue
        if mesh_name not in meshes:
            raise AssetContractError(
                f"{context} geom {geom.get('name', '<unnamed>')} references "
                f"undeclared mesh {mesh_name!r}"
            )
        visual = (
            _contact_value(geom, "contype", context=context) == 0
            and _contact_value(geom, "conaffinity", context=context) == 0
        )
        observed[mesh_name].add("visual" if visual else "collision")

    roles: dict[str, str] = {}
    for mesh_name, mesh_roles in observed.items():
        if not mesh_roles:
            raise AssetContractError(
                f"{context} mesh {mesh_name!r} is unused, so its role is undefined"
            )
        if len(mesh_roles) != 1:
            raise AssetContractError(
                f"{context} mesh {mesh_name!r} is used as both visual and collision geometry"
            )
        roles[mesh_name] = next(iter(mesh_roles))
    return meshes, roles


def _finite_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssetContractError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AssetContractError(f"{context} must be a finite number")
    return result


def _runtime_vector(value: Any, *, context: str) -> list[float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
    ):
        raise AssetContractError(f"{context} must be a three-number array")
    return [
        _finite_number(component, context=f"{context}[{index}]")
        for index, component in enumerate(value)
    ]


def _normalize_collision_geometry(value: Any, *, context: str) -> dict[str, Any]:
    collision = value
    if not isinstance(collision, Mapping):
        raise AssetContractError(f"{context} must be an object")
    collision_keys = {
        "bounds_min_m",
        "bounds_max_m",
        "z_min_m",
        "z_max_m",
        "xy_radius_m",
    }
    if set(collision) != collision_keys:
        raise AssetContractError(
            f"{context} must contain exactly {sorted(collision_keys)}"
        )
    bounds_min = _runtime_vector(
        collision["bounds_min_m"],
        context=f"{context}.bounds_min_m",
    )
    bounds_max = _runtime_vector(
        collision["bounds_max_m"],
        context=f"{context}.bounds_max_m",
    )
    if any(lower > upper for lower, upper in zip(bounds_min, bounds_max)):
        raise AssetContractError(f"{context} bounds_min_m exceeds bounds_max_m")
    z_min = _finite_number(
        collision["z_min_m"], context=f"{context}.z_min_m"
    )
    z_max = _finite_number(
        collision["z_max_m"], context=f"{context}.z_max_m"
    )
    radius = _finite_number(
        collision["xy_radius_m"], context=f"{context}.xy_radius_m"
    )
    if z_min != bounds_min[2] or z_max != bounds_max[2]:
        raise AssetContractError(
            f"{context} z extrema must match the bounds vectors"
        )
    if radius <= 0.0:
        raise AssetContractError(f"{context}.xy_radius_m must be positive")
    return {
        "bounds_min_m": bounds_min,
        "bounds_max_m": bounds_max,
        "z_min_m": z_min,
        "z_max_m": z_max,
        "xy_radius_m": radius,
    }


def _normalize_runtime_geometry(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AssetContractError(f"{context} must be an object")
    geometry_keys = {
        "source_collision_geometry",
        "compiled_collision_observed",
        "semantic_sites_m",
    }
    if set(value) != geometry_keys:
        raise AssetContractError(
            f"{context} must contain exactly {sorted(geometry_keys)}"
        )

    source_collision = _normalize_collision_geometry(
        value["source_collision_geometry"],
        context=f"{context}.source_collision_geometry",
    )
    compiled_collision = _normalize_collision_geometry(
        value["compiled_collision_observed"],
        context=f"{context}.compiled_collision_observed",
    )

    sites = value["semantic_sites_m"]
    required_sites = {"bottom_site", "top_site", "horizontal_radius_site"}
    if not isinstance(sites, Mapping) or set(sites) != required_sites:
        raise AssetContractError(
            f"{context}.semantic_sites_m must contain exactly {sorted(required_sites)}"
        )
    normalized_sites = {
        name: _runtime_vector(
            sites[name], context=f"{context}.semantic_sites_m.{name}"
        )
        for name in sorted(required_sites)
    }
    return {
        "source_collision_geometry": source_collision,
        "compiled_collision_observed": compiled_collision,
        "semantic_sites_m": normalized_sites,
    }


def _validate_runtime_geometry(
    bundle_id: str,
    definition: Mapping[str, Any],
    root: ET.Element,
) -> dict[str, Any] | None:
    context = f"asset bundle {bundle_id!r}.runtime_geometry"
    raw_geometry = definition.get("runtime_geometry")
    expected_geometry = OBJECT_RUNTIME_GEOMETRY.get(bundle_id)
    if raw_geometry is None:
        if expected_geometry is not None:
            raise AssetContractError(f"{context} is required")
        return None

    normalized = _normalize_runtime_geometry(raw_geometry, context=context)
    if expected_geometry is not None:
        expected = _normalize_runtime_geometry(
            expected_geometry,
            context=f"registered runtime geometry for {bundle_id!r}",
        )
        if normalized != expected:
            raise AssetContractError(
                f"{context} differs from OBJECT_RUNTIME_GEOMETRY"
            )

    sites_by_name: dict[str, list[ET.Element]] = {}
    for site in root.iter("site"):
        name = (site.get("name") or "").strip()
        if name:
            sites_by_name.setdefault(name, []).append(site)
    for site_name, expected_position in normalized["semantic_sites_m"].items():
        matches = sites_by_name.get(site_name, [])
        if len(matches) != 1:
            raise AssetContractError(
                f"{context} requires exactly one MJCF site {site_name!r}, "
                f"found {len(matches)}"
            )
        raw_position = matches[0].get("pos", "0 0 0").split()
        try:
            observed_position = [float(component) for component in raw_position]
        except ValueError as exc:
            raise AssetContractError(
                f"{context} MJCF site {site_name!r} has an invalid pos"
            ) from exc
        if len(observed_position) != 3 or observed_position != expected_position:
            raise AssetContractError(
                f"{context} MJCF site {site_name!r} position differs: "
                f"expected {expected_position}, got {observed_position}"
            )
    return normalized


def _validate_bundle(
    bundle_id: str,
    definition: Any,
    *,
    asset_root: Path,
) -> tuple[dict[str, Any], set[Path]]:
    context = f"asset bundle {bundle_id!r}"
    if not isinstance(definition, Mapping):
        raise AssetContractError(f"{context} definition must be an object")
    mjcf_raw = definition.get("mjcf")
    if not isinstance(mjcf_raw, str):
        raise AssetContractError(f"{context}.mjcf must be a string")
    xml_path, xml_relative = _relative_file(
        mjcf_raw,
        base=asset_root,
        asset_root=asset_root,
        context=f"{context}.mjcf",
    )
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise AssetContractError(f"{context} MJCF is unreadable: {xml_path}: {exc}") from exc
    if root.find(".//include") is not None:
        raise AssetContractError(
            f"{context} uses <include>, which could load geometry outside this "
            "single-file contract; flatten the MJCF before registering it"
        )

    runtime_geometry = _validate_runtime_geometry(bundle_id, definition, root)

    meshes, mesh_roles = _classify_meshes(root, context=context)
    role_definitions = definition.get("roles")
    required_roles = definition.get("required_roles")
    if not isinstance(role_definitions, Mapping):
        raise AssetContractError(f"{context}.roles must be an object")
    if (
        not isinstance(required_roles, list)
        or not required_roles
        or any(
            not isinstance(role, str) or role not in _ROLES
            for role in required_roles
        )
        or len(set(required_roles)) != len(required_roles)
    ):
        raise AssetContractError(
            f"{context}.required_roles must be a unique non-empty subset of {sorted(_ROLES)}"
        )

    compiler = root.find("./compiler")
    mesh_base = xml_path.parent
    texture_base = xml_path.parent
    if compiler is not None:
        unsupported_resolution = sorted(
            attribute
            for attribute in ("assetdir", "strippath")
            if attribute in compiler.attrib
        )
        if unsupported_resolution:
            raise AssetContractError(
                f"{context} compiler uses unsupported path resolution: "
                + ", ".join(unsupported_resolution)
                + "; flatten paths before registering the bundle"
            )
        for attribute in ("meshdir", "texturedir"):
            directory = compiler.get(attribute)
            if directory:
                directory_path = Path(directory)
                if (
                    directory_path.is_absolute()
                    or directory.startswith("~")
                    or ".." in directory_path.parts
                ):
                    raise AssetContractError(
                        f"{context} compiler {attribute} must be relative and stay "
                        f"under the MJCF directory: {directory}"
                    )
                resolved_dir = (xml_path.parent / directory_path).resolve()
                try:
                    resolved_dir.relative_to(asset_root)
                except ValueError as exc:
                    raise AssetContractError(
                        f"{context} compiler {attribute} escapes the asset root: {directory}"
                    ) from exc
                if attribute == "meshdir":
                    mesh_base = resolved_dir
                else:
                    texture_base = resolved_dir

    selected: set[Path] = {xml_path}
    role_files: dict[str, list[str]] = {role: [] for role in _ROLES}
    for mesh_name, mesh in meshes.items():
        raw_file = mesh.get("file")
        if raw_file is None:
            raise AssetContractError(f"{context} mesh {mesh_name!r} has no file")
        path, relative = _relative_file(
            raw_file,
            base=mesh_base,
            asset_root=asset_root,
            context=f"{context} mesh {mesh_name!r}",
        )
        selected.add(path)
        role_files[mesh_roles[mesh_name]].append(relative)

    # Check every other file-backed MJCF asset as well (textures and includes).
    for element in root.iter():
        if element.tag == "mesh" or element.get("file") is None:
            continue
        base = texture_base if element.tag == "texture" else xml_path.parent
        path, _ = _relative_file(
            element.get("file", ""),
            base=base,
            asset_root=asset_root,
            context=f"{context} <{element.tag}> file",
        )
        selected.add(path)

    role_report: dict[str, Any] = {}
    for role in sorted(_ROLES):
        files = sorted(role_files[role])
        role_definition = role_definitions.get(role)
        if not isinstance(role_definition, Mapping):
            raise AssetContractError(f"{context}.roles.{role} must be an object")
        expected_count = role_definition.get("expected_count")
        if isinstance(expected_count, bool) or not isinstance(expected_count, int):
            raise AssetContractError(
                f"{context}.roles.{role}.expected_count must be an integer"
            )
        if len(files) != expected_count:
            raise AssetContractError(
                f"{context} expected {expected_count} {role} meshes, found {len(files)}"
            )
        if role in required_roles and not files:
            raise AssetContractError(f"{context} is missing required role {role!r}")
        prefixes = _normalized_prefixes(
            role_definition.get("allowed_prefixes"),
            context=f"{context}.roles.{role}.allowed_prefixes",
        )
        for relative in files:
            if not any(relative.startswith(prefix) for prefix in prefixes):
                raise AssetContractError(
                    f"{context} {role} mesh is outside its allowed directories: "
                    f"{relative}; allowed={list(prefixes)}"
                )
        if len(files) != len(set(files)):
            duplicates = sorted(
                {relative for relative in files if files.count(relative) > 1}
            )
            raise AssetContractError(
                f"{context} {role} role references the same mesh file more than once: "
                f"{duplicates}"
            )
        expected_files = role_definition.get("expected_files")
        if expected_files is not None:
            if (
                not isinstance(expected_files, list)
                or any(not isinstance(item, str) for item in expected_files)
                or len(set(expected_files)) != len(expected_files)
            ):
                raise AssetContractError(
                    f"{context}.roles.{role}.expected_files must be a unique string array"
                )
            if files != sorted(expected_files):
                missing = sorted(set(expected_files) - set(files))
                unexpected = sorted(set(files) - set(expected_files))
                raise AssetContractError(
                    f"{context} {role} mesh set differs from expected_files: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        role_report[role] = {"count": len(files), "files": files}

    reused = sorted(set(role_files["visual"]) & set(role_files["collision"]))
    if reused:
        raise AssetContractError(
            f"{context} uses the same file as visual and collision geometry: {reused}"
        )

    report = {
        "mjcf": xml_relative,
        "roles": role_report,
    }
    if runtime_geometry is not None:
        report["runtime_geometry"] = runtime_geometry
    return report, selected


def _resolve_bundle_definition(
    bundle_id: str,
    bundles: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Resolve a visual-only MJCF derivative against its geometry contract."""

    definition = bundles[bundle_id]
    if not isinstance(definition, Mapping):
        raise AssetContractError(
            f"asset bundle {bundle_id!r} definition must be an object"
        )
    parent_id = definition.get("derived_from")
    if parent_id is None:
        return definition
    if (
        not isinstance(parent_id, str)
        or not parent_id
        or parent_id == bundle_id
        or parent_id not in bundles
    ):
        raise AssetContractError(
            f"asset bundle {bundle_id!r}.derived_from is invalid: {parent_id!r}"
        )
    unknown = sorted(set(definition) - {"derived_from", "mjcf", "expected_sha256"})
    if unknown or not isinstance(definition.get("mjcf"), str):
        raise AssetContractError(
            f"derived asset bundle {bundle_id!r} may define only derived_from, "
            f"mjcf, and expected_sha256; unexpected={unknown}"
        )
    parent = bundles[parent_id]
    if not isinstance(parent, Mapping) or "derived_from" in parent:
        raise AssetContractError(
            f"asset bundle {bundle_id!r} must derive directly from a full bundle"
        )
    resolved = dict(parent)
    resolved["mjcf"] = definition["mjcf"]
    if "expected_sha256" in definition:
        resolved["expected_sha256"] = definition["expected_sha256"]
    return resolved


def _expected_bundle_hashes(
    bundle_id: str,
    definition: Mapping[str, Any],
) -> dict[str, str]:
    raw = definition.get("expected_sha256", {})
    if not isinstance(raw, Mapping):
        raise AssetContractError(
            f"asset bundle {bundle_id!r}.expected_sha256 must be an object"
        )
    expected: dict[str, str] = {}
    for relative, digest in raw.items():
        path = Path(relative) if isinstance(relative, str) else None
        if (
            path is None
            or not relative
            or path.is_absolute()
            or relative.startswith("~")
            or ".." in path.parts
        ):
            raise AssetContractError(
                f"asset bundle {bundle_id!r}.expected_sha256 has an invalid path: "
                f"{relative!r}"
            )
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise AssetContractError(
                f"asset bundle {bundle_id!r}.expected_sha256 has an invalid digest "
                f"for {relative!r}"
            )
        expected[path.as_posix()] = digest
    return expected


def _validate_forbidden_config(
    value: Any,
) -> tuple[tuple[str, ...], frozenset[str], frozenset[int]]:
    if not isinstance(value, Mapping):
        raise AssetContractError("asset manifest forbidden must be an object")
    globs = value.get("path_globs")
    hashes = value.get("sha256")
    candidate_sizes = value.get("candidate_sizes")
    if (
        not isinstance(globs, list)
        or not globs
        or any(not isinstance(item, str) or not item for item in globs)
    ):
        raise AssetContractError("forbidden.path_globs must be a non-empty string array")
    if (
        not isinstance(hashes, list)
        or not hashes
        or any(not isinstance(item, str) or not _SHA256_RE.fullmatch(item) for item in hashes)
    ):
        raise AssetContractError(
            "forbidden.sha256 must be a non-empty array of lowercase SHA-256 strings"
        )
    if (
        not isinstance(candidate_sizes, list)
        or not candidate_sizes
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in candidate_sizes
        )
    ):
        raise AssetContractError(
            "forbidden.candidate_sizes must be a non-empty array of non-negative integers"
        )
    return tuple(globs), frozenset(hashes), frozenset(candidate_sizes)


def _matches_glob(relative: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(relative, pattern) or (
        pattern.startswith("**/") and fnmatch.fnmatchcase(relative, pattern[3:])
    )


def _scan_forbidden(
    asset_root: Path,
    *,
    path_globs: tuple[str, ...],
    forbidden_hashes: frozenset[str],
    candidate_sizes: frozenset[int],
    hash_cache: dict[Path, str],
) -> dict[str, int]:
    files_seen = 0
    hashes_computed = 0
    for path in sorted(asset_root.rglob("*")):
        if not path.is_file():
            continue
        files_seen += 1
        relative = path.relative_to(asset_root).as_posix()
        if any(_matches_glob(relative, pattern) for pattern in path_globs):
            raise AssetContractError(
                f"forbidden asset path detected under {asset_root}: {relative}"
            )
        resolved = path.resolve()
        try:
            resolved.relative_to(asset_root)
        except ValueError:
            # An unrelated external symlink is not a repository-owned asset.
            # Selected MJCF references to it are rejected separately.
            continue
        if path.stat().st_size not in candidate_sizes:
            continue
        digest = hash_cache.get(resolved)
        if digest is None:
            digest = _sha256(resolved)
            hash_cache[resolved] = digest
            hashes_computed += 1
        if digest in forbidden_hashes:
            raise AssetContractError(
                f"forbidden asset content detected (renaming is not allowed): "
                f"{relative} sha256={digest}"
            )
    return {"files_scanned": files_seen, "hashes_computed": hashes_computed}


def validate_forbidden_asset_tree(
    asset_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, int]:
    """Apply the repository-wide forbidden-asset rules to another asset tree.

    Generated robot bundles live outside ``robosuite_env/assets`` but must obey
    the same filename and content bans.  This helper deliberately exposes only
    that shared deny-list; bundle ownership remains with the caller's contract.
    """

    root = Path(asset_root).resolve()
    if not root.is_dir():
        raise AssetContractError(f"asset root is not a directory: {root}")
    manifest_file = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else Path(__file__).resolve().parent.parent / "assets" / "manifest.json"
    )
    manifest = _load_manifest(manifest_file)
    path_globs, forbidden_hashes, candidate_sizes = _validate_forbidden_config(
        manifest["forbidden"]
    )
    return _scan_forbidden(
        root,
        path_globs=path_globs,
        forbidden_hashes=forbidden_hashes,
        candidate_sizes=candidate_sizes,
        hash_cache={},
    )


def runtime_mesh_set_sha256(mesh_sha256: Mapping[str, str]) -> str:
    """Digest one bundle's runtime mesh set, independent of checkout location.

    ``mesh_sha256`` maps every mesh file the bundle's MJCF selects (visual and
    collision, as asset-root-relative POSIX paths) to its SHA-256. The digest
    is the SHA-256 of the ``sha256sum`` listing of those files sorted by path,
    so a copied tree can be checked with coreutils alone::

        cd robosuite_env/assets
        sha256sum <mesh files> | LC_ALL=C sort -k2 | sha256sum
    """

    listing = "".join(
        f"{digest}  {relative}\n"
        for relative, digest in sorted(mesh_sha256.items())
    )
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()


def _runtime_mesh_set_pins(manifest: Mapping[str, Any]) -> dict[str, str]:
    raw = manifest.get(RUNTIME_MESH_SET_SECTION)
    if raw is None:
        return {}
    context = f"asset manifest {RUNTIME_MESH_SET_SECTION}"
    if not isinstance(raw, Mapping):
        raise AssetContractError(f"{context} must be an object")
    bundles = manifest["bundles"]
    pins: dict[str, str] = {}
    for bundle_id, digest in raw.items():
        definition = bundles.get(bundle_id)
        if not isinstance(definition, Mapping):
            raise AssetContractError(f"{context} pins unknown bundle {bundle_id!r}")
        if "derived_from" in definition:
            raise AssetContractError(
                f"{context} pins derived bundle {bundle_id!r}; a derived bundle "
                "shares its parent's meshes and is checked against the parent's pin"
            )
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise AssetContractError(
                f"{context} has an invalid digest for {bundle_id!r}"
            )
        pins[bundle_id] = digest
    return pins


def _verify_runtime_mesh_set(
    bundle_id: str,
    bundles: Mapping[str, Any],
    report: Mapping[str, Any],
    pins: Mapping[str, str],
    *,
    selected_by_relative: Mapping[str, Path],
    hash_cache: dict[Path, str],
    asset_root: Path,
) -> str | None:
    """Bind the meshes MuJoCo will load to the set the manifest was measured on.

    The object meshes are gitignored build outputs. Every other check in this
    contract -- roles, counts, file names, pinned runtime geometry, MJCF sites
    -- reads tracked text, so a stale or partial mesh copy under a current
    MJCF passes them all and loads with the wrong shape and origin. A bundle
    that pins runtime geometry must therefore pin its mesh set too; a derived
    bundle is checked against its parent's pin.
    """

    definition = bundles[bundle_id]
    owner = definition.get("derived_from") or bundle_id
    expected = pins.get(owner)
    if expected is None:
        if "runtime_geometry" in report or bundle_id in OBJECT_RUNTIME_GEOMETRY:
            raise AssetContractError(
                f"asset bundle {bundle_id!r} pins runtime geometry but "
                f"{RUNTIME_MESH_SET_SECTION} pins no mesh set for {owner!r}; a "
                "stale mesh copy would load without error"
            )
        return None
    mesh_sha256: dict[str, str] = {}
    for role in sorted(_ROLES):
        for relative in report["roles"][role]["files"]:
            path = selected_by_relative[relative]
            digest = hash_cache.get(path)
            if digest is None:
                digest = _sha256(path)
                hash_cache[path] = digest
            mesh_sha256[relative] = digest
    observed = runtime_mesh_set_sha256(mesh_sha256)
    if observed != expected:
        raise AssetContractError(
            f"asset bundle {bundle_id!r} runtime mesh set under {asset_root} is "
            f"not the pinned one: {RUNTIME_MESH_SET_SECTION}[{owner!r}] is "
            f"{expected}, but the {len(mesh_sha256)} meshes its MJCF selects "
            f"hash to {observed}. The meshes are gitignored build outputs, so an "
            "out-of-date or partial copy still loads, and under the current MJCF "
            "it gives the object a different shape and origin. Rebuild or "
            "re-copy them until `python -m robosuite_env.contracts.assets "
            f"--bundle {bundle_id}` passes."
        )
    return observed


def validate_repository_assets(
    asset_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    bundle_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate selected asset bundles and return their content contract.

    ``bundle_ids=None`` validates every bundle.  The returned digest is stable
    across checkout locations and changes if the manifest or any selected
    MJCF-referenced file changes.
    """

    root = Path(asset_root).resolve()
    if not root.is_dir():
        raise AssetContractError(f"asset root is not a directory: {root}")
    manifest_file = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else root / "manifest.json"
    )
    manifest = _load_manifest(manifest_file)
    bundles = manifest["bundles"]

    if bundle_ids is None:
        selected_ids = tuple(sorted(bundles))
    else:
        if isinstance(bundle_ids, (str, bytes)):
            raise AssetContractError("bundle_ids must be an iterable of bundle names")
        requested_ids = tuple(bundle_ids)
        if not requested_ids or any(
            not isinstance(item, str) for item in requested_ids
        ):
            raise AssetContractError("bundle_ids must contain at least one bundle name")
        selected_ids = tuple(dict.fromkeys(requested_ids))
        unknown = sorted(set(selected_ids) - set(bundles))
        if unknown:
            raise AssetContractError(
                f"asset manifest has no bundle(s): {', '.join(unknown)}"
            )
        selected_ids = tuple(sorted(selected_ids))

    path_globs, forbidden_hashes, candidate_sizes = _validate_forbidden_config(
        manifest["forbidden"]
    )
    mesh_set_pins = _runtime_mesh_set_pins(manifest)
    hash_cache: dict[Path, str] = {}
    forbidden_report = _scan_forbidden(
        root,
        path_globs=path_globs,
        forbidden_hashes=forbidden_hashes,
        candidate_sizes=candidate_sizes,
        hash_cache=hash_cache,
    )

    bundle_report: dict[str, Any] = {}
    resolved_definitions: dict[str, Mapping[str, Any]] = {}
    selected_paths: set[Path] = set()
    for bundle_id in selected_ids:
        resolved_definition = _resolve_bundle_definition(bundle_id, bundles)
        report, paths = _validate_bundle(
            bundle_id,
            resolved_definition,
            asset_root=root,
        )
        expected_hashes = _expected_bundle_hashes(bundle_id, resolved_definition)
        selected_by_relative = {
            path.relative_to(root).as_posix(): path for path in paths
        }
        for relative, expected_digest in expected_hashes.items():
            expected_path = selected_by_relative.get(relative)
            if expected_path is None:
                raise AssetContractError(
                    f"asset bundle {bundle_id!r}.expected_sha256 path is not selected "
                    f"by its MJCF: {relative}"
                )
            observed_digest = hash_cache.get(expected_path)
            if observed_digest is None:
                observed_digest = _sha256(expected_path)
                hash_cache[expected_path] = observed_digest
            if observed_digest != expected_digest:
                raise AssetContractError(
                    f"asset bundle {bundle_id!r} expected SHA-256 {expected_digest} "
                    f"for {relative}, got {observed_digest}"
                )
        if expected_hashes:
            report["expected_sha256"] = expected_hashes
        mesh_set = _verify_runtime_mesh_set(
            bundle_id,
            bundles,
            report,
            mesh_set_pins,
            selected_by_relative=selected_by_relative,
            hash_cache=hash_cache,
            asset_root=root,
        )
        if mesh_set is not None:
            report[RUNTIME_MESH_SET_SECTION] = mesh_set
        bundle_report[bundle_id] = report
        resolved_definitions[bundle_id] = resolved_definition
        selected_paths.update(paths)

    selected_files: list[dict[str, Any]] = []
    for path in sorted(selected_paths):
        digest = hash_cache.get(path)
        if digest is None:
            digest = _sha256(path)
            hash_cache[path] = digest
        selected_files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": digest,
            }
        )

    digest_payload = {
        "schema_version": manifest["schema_version"],
        "bundle_definitions": {
            bundle_id: {
                "manifest": bundles[bundle_id],
                "resolved": resolved_definitions[bundle_id],
            }
            for bundle_id in selected_ids
        },
        "forbidden": manifest["forbidden"],
        "selected_files": selected_files,
    }
    contract_digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    return {
        "schema_version": 1,
        "asset_root": str(root),
        "manifest_path": str(manifest_file),
        "contract_digest": contract_digest,
        "bundles": bundle_report,
        "selected_files": selected_files,
        "forbidden_scan": forbidden_report,
    }


__all__ = [
    "AssetContractError",
    "OBJECT_RUNTIME_GEOMETRY",
    "RUNTIME_MESH_SET_SECTION",
    "runtime_bundle_ids",
    "runtime_geometry_for_object",
    "runtime_mesh_set_sha256",
    "validate_forbidden_asset_tree",
    "validate_repository_assets",
]


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate repository simulation visual/collision assets."
    )
    parser.add_argument(
        "--asset-root",
        default=str(Path(__file__).resolve().parent.parent / "assets"),
    )
    parser.add_argument("--bundle", action="append", dest="bundles")
    args = parser.parse_args()
    report = validate_repository_assets(
        args.asset_root,
        bundle_ids=args.bundles,
    )
    print(
        json.dumps(
            {
                "asset_root": report["asset_root"],
                "manifest": report["manifest_path"],
                "contract_digest": report["contract_digest"],
                "bundles": {
                    bundle: {
                        role: details["count"]
                        for role, details in value["roles"].items()
                    }
                    for bundle, value in report["bundles"].items()
                },
                "forbidden_scan": report["forbidden_scan"],
                RUNTIME_MESH_SET_SECTION: {
                    bundle: value[RUNTIME_MESH_SET_SECTION]
                    for bundle, value in report["bundles"].items()
                    if RUNTIME_MESH_SET_SECTION in value
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except AssetContractError as exc:
        raise SystemExit(f"asset contract error: {exc}") from exc
