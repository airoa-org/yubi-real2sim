# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Loader for hand-authored object layouts.

The Duo box layouts are not measured from a capture:
they were arranged on a placement board over the trimmed table extent. This
module is the contract for them, kept deliberately small.

A file here must say who placed the layout, with what, when and why, and it
must not carry capture evidence that would make a chosen arrangement read as
measured.

A layout's z is not trusted at runtime. It was written against a flat plane at
table z = 0 and means "resting on the table", so the registered environment
seats every placed object on the compiled table collision instead (see
:func:`seat_lift` and ``CupPlateLiftGW._seat_layout_placements``).
"""

from __future__ import annotations

from datetime import datetime
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

MANUAL_PLACEMENT_KIND = "manual_placement"
REQUIRED_SOURCE_KEYS = frozenset(
    {"kind", "scene", "placed_by", "placed_at", "tool", "rationale"}
)
#: Evidence only a capture can produce.
FORBIDDEN_SOURCE_KEYS = frozenset(
    {
        "dataset",
        "episode_index",
        "episode_id",
        "frame_index",
        "calibration",
        "annotations",
        "source_hashes_sha256",
    }
)


class LayoutError(ValueError):
    """Raised when a hand-authored layout violates its contract."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LayoutError(f"{label} must be a mapping")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LayoutError(f"{label} must be a non-empty string")
    return value


def _vector(value: Any, size: int, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise LayoutError(f"{label} must be a sequence of {size} numbers")
    if len(value) != size:
        raise LayoutError(f"{label} must have exactly {size} entries")
    out = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise LayoutError(f"{label}[{index}] must be a number")
        item = float(item)
        if not math.isfinite(item):
            raise LayoutError(f"{label}[{index}] must be finite")
        out.append(item)
    return out


def validate_manual_layout(
    payload: Any, *, expected_objects: Sequence[str]
) -> dict[str, Any]:
    """Validate one hand-authored layout and return it canonicalized."""

    root = _mapping(payload, "layout root")
    for key in ("schema_version", "frame", "source", "objects"):
        if key not in root:
            raise LayoutError(f"layout root is missing required key: {key}")
    if root["schema_version"] != 1 or isinstance(root["schema_version"], bool):
        raise LayoutError("schema_version must be integer 1")
    if root["frame"] != "table":
        raise LayoutError("frame must be 'table'")

    source = _mapping(root["source"], "source")
    if source.get("kind") != MANUAL_PLACEMENT_KIND:
        raise LayoutError(f"source.kind must be {MANUAL_PLACEMENT_KIND!r}")
    missing = sorted(REQUIRED_SOURCE_KEYS - set(source))
    if missing:
        raise LayoutError(f"source is missing required keys: {missing}")
    borrowed = sorted(FORBIDDEN_SOURCE_KEYS & set(source))
    if borrowed:
        raise LayoutError(
            "a hand-authored layout must not carry capture evidence, which "
            f"would read as measured; remove {borrowed}"
        )
    for key in ("scene", "placed_by", "tool", "rationale"):
        _text(source[key], f"source.{key}")
    stamp = _text(source["placed_at"], "source.placed_at")
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LayoutError(f"source.placed_at must be ISO 8601: {exc}") from exc
    if parsed.tzinfo is None:
        raise LayoutError("source.placed_at must carry a UTC offset")
    # Optional: the rigs the layout was placed for. Absent means any rig that
    # stands on source.scene; tasks/layout_selection.py require_layout_rig
    # enforces it.
    if "robots" in source:
        robots = source["robots"]
        if (
            not isinstance(robots, Sequence)
            or isinstance(robots, (str, bytes))
            or not robots
        ):
            raise LayoutError("source.robots must be a non-empty list of robot names")
        names = [_text(name, f"source.robots[{index}]") for index, name in enumerate(robots)]
        if len(set(names)) != len(names):
            raise LayoutError(f"source.robots names a rig more than once: {names}")

    objects = _mapping(root["objects"], "objects")
    expected = list(expected_objects)
    if sorted(objects) != sorted(expected):
        raise LayoutError(
            f"objects must be exactly {sorted(expected)}; got {sorted(objects)}"
        )

    canonical: dict[str, Any] = {
        "schema_version": 1,
        "frame": "table",
        "source": dict(source),
        "objects": {},
    }
    for name in expected:
        label = f"objects.{name}"
        record = _mapping(objects[name], label)
        for key in ("position_m", "quaternion_wxyz"):
            if key not in record:
                raise LayoutError(f"{label} is missing required key: {key}")
        position = _vector(record["position_m"], 3, f"{label}.position_m")
        quaternion = _vector(record["quaternion_wxyz"], 4, f"{label}.quaternion_wxyz")
        norm = math.sqrt(sum(v * v for v in quaternion))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise LayoutError(
                f"{label}.quaternion_wxyz must be unit length; norm={norm:.9g}"
            )
        entry = {
            "position_m": position,
            "quaternion_wxyz": quaternion,
            "asset": _text(record.get("asset", name), f"{label}.asset"),
        }
        # The measured weight is carried so the compiled mass can be checked
        # against it.
        if record.get("mass_kg") is not None:
            mass = record["mass_kg"]
            if not isinstance(mass, Real) or isinstance(mass, bool):
                raise LayoutError(f"{label}.mass_kg must be a number")
            if not (float(mass) > 0.0) or not math.isfinite(float(mass)):
                raise LayoutError(
                    f"{label}.mass_kg must be finite and positive, got {mass!r}"
                )
            entry["mass_kg"] = float(mass)
        canonical["objects"][name] = entry
    return canonical


#: How far a compiled body may sit from its layout's measured weight.
LAYOUT_MASS_TOLERANCE = 0.01

#: The measured weights, for the tasks whose layouts do not carry them.
MEASURED_MASSES_PATH = (
    Path(__file__).resolve().parents[1]
    / "obj_calib"
    / "object_masses_measured.json"
)


def load_measured_masses(path=None) -> dict[str, float]:
    """The measured weight of each asset, keyed by asset name.

    One file, read by every check. cup_plate records no weights in its own
    contract, so this is what it is checked against; and a layout that does
    carry a weight is checked against this too, so the same object cannot
    claim two different masses in two places.
    """

    payload = json.loads(Path(path or MEASURED_MASSES_PATH).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("unit") != "kg":
        raise LayoutError(
            f"{MEASURED_MASSES_PATH.name} must be schema_version 1 in kg"
        )
    masses = {}
    for name, value in (payload.get("objects") or {}).items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise LayoutError(f"{name} mass must be a number")
        if not (float(value) > 0.0) or not math.isfinite(float(value)):
            raise LayoutError(f"{name} mass must be finite and positive")
        masses[str(name)] = float(value)
    if not masses:
        raise LayoutError(f"{MEASURED_MASSES_PATH.name} records no objects")
    return masses


def assert_layout_masses(
    model,
    layout: Mapping[str, Any],
    *,
    mujoco,
    tolerance: float = LAYOUT_MASS_TOLERANCE,
    label: str = "layout",
    measured_masses=None,
    skip_bodies=(),
    renamed_bodies=None,
):
    """Fail if a compiled body does not weigh what its layout says it does.

    robosuite's ``base.xml`` sets ``inertiagrouprange="0 0"``, so only group-0
    geoms carry mass and a bare compile of the asset XML reports something
    else entirely -- which is why this reads ``model.body_mass`` rather than
    recomputing anything. Objects whose layout records no weight are reported
    but do not fail: not every layout has been measured.

    Returns the per-object report so a run can seal what it checked.
    """

    report = {"tolerance": float(tolerance), "checked": [], "unmeasured": []}
    problems = []
    skip = set(skip_bodies or ())
    # A file that is neither shape is not a layout at all -- the cup/plate
    # position files are cup_xy/plate_xy and nothing else. Saying so is not the
    # same as saying a layout was checked and came out clean, and the caller
    # has to be able to tell those apart.
    if not isinstance(layout, Mapping):
        report["not_a_layout"] = "root is not a mapping"
        return report
    try:
        table = load_measured_masses(measured_masses)
    except FileNotFoundError:
        table = {}
    # Two spellings are in the wild. The box and container layouts put mass_kg
    # inside each object; Duo's 46 cup_plate layouts put one top-level dict
    # keyed by object name. Read both, so a weight is not skipped for being
    # written the other way.
    top_level = layout.get("mass_kg")
    if top_level is not None and not isinstance(top_level, Mapping):
        raise LayoutError("layout mass_kg must be a mapping of object -> kg")
    top_level = dict(top_level or {})
    # The runtime renames the object the task selected: whichever of pen/tape/lan
    # box_pack picked compiles as ``cube_main``. Looking only for the layout's
    # own name reported exactly that object -- the one the policy actually
    # manipulates -- as unmeasured, so the check passed by skipping it.
    def resolve(name, asset):
        for candidate in (f"{name}_main", f"{asset}_main", name, asset):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
            if body_id >= 0:
                return body_id, candidate
        return -1, None

    # A rollout compiles one task's objects, so three of the four layouts this
    # is asked about describe a scene that is not in this model. Recognise that
    # by its shape -- nothing in the file resolves -- and say so, rather than
    # failing a cup run because the box layout's tape has no body.
    objects = layout.get("objects") or {}
    if not isinstance(objects, Mapping):
        report["not_a_layout"] = "objects is not a mapping"
        return report
    entries = [(name, entry) for name, entry in objects.items()
               if isinstance(entry, Mapping)]
    if not entries and not top_level:
        report["not_a_layout"] = "no objects and no mass_kg"
        return report
    if not entries and top_level:
        # Duo's cup_plate files are cup_xy / plate_xy / mass_kg with no objects
        # mapping at all. Reading only "objects" made this return absent and
        # check neither weight -- the exact failure the top-level support was
        # added to prevent.
        entries = [(name, {"asset": name}) for name in sorted(top_level)]
    present = sum(1 for name, entry in entries
                  if resolve(name, entry.get("asset", name))[0] >= 0)
    if not present:
        report["absent"] = True
        return report

    # Which asset the renamed body holds, told by the caller that built it.
    # Guessing "the first entry that did not resolve" is wrong the moment a
    # layout lists a distractor before the selected object: in box_pack, lan
    # comes before pen, so lan would claim cube_main and pen -- the object the
    # policy is actually holding -- would then fail for having no body.
    renamed = {str(body): str(asset) for body, asset in (renamed_bodies or {}).items()}
    selected = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_main")
    selected_asset = renamed.get("cube_main")
    claimed_selected = False
    for name, entry in entries:
        asset = entry.get("asset", name)
        recorded = entry.get("mass_kg")
        if recorded is None:
            recorded = top_level.get(name)
        measured = table.get(asset)
        # The measured table is the source. A layout may repeat a weight, and
        # then it has to agree: two records of one object that disagree is a
        # defect wherever it sits, and picking one silently would hide it.
        if recorded is not None and measured is not None:
            if abs(float(recorded) / float(measured) - 1.0) > tolerance:
                problems.append(
                    f"{name} ({asset}): the layout records "
                    f"{float(recorded)*1000:.3f} g but {MEASURED_MASSES_PATH.name} "
                    f"records {float(measured)*1000:.3f} g"
                )
                continue
        want = measured if measured is not None else recorded
        body_id, _matched = resolve(name, asset)
        # Fail closed: an unmapped cube_main is not claimed by anyone, so a
        # distractor cannot validate itself against the selected object's mass.
        claimable = (selected_asset is not None and asset == selected_asset)
        if body_id < 0 and selected >= 0 and not claimed_selected and claimable:
            # One layout entry per scene is the selected object and carries no
            # body of its own name. Claim it once, and only for the asset the
            # caller says that body holds, so a distractor listed earlier
            # cannot take it.
            body_id = selected
            _matched = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, selected)
            claimed_selected = True
        # Exempt the body whose mass was deliberately replaced, and only that
        # body. The test comes after the claim above so one skipped name cannot
        # excuse every entry that failed to resolve.
        if _matched is not None and _matched in skip:
            report["unmeasured"].append(
                {"object": name, "reason": "mass deliberately overridden"}
            )
            continue
        if body_id < 0:
            if want is not None:
                problems.append(
                    f"{name} ({asset}): layout records "
                    f"{float(want)*1000:.3f} g but no compiled body could be "
                    "resolved"
                )
            else:
                report["unmeasured"].append(
                    {"object": name, "reason": "no compiled body"}
                )
            continue
        got = float(model.body_mass[body_id])
        if want is None:
            report["unmeasured"].append(
                {"object": name, "compiled_kg": round(got, 9), "reason": "no mass_kg"}
            )
            continue
        ratio = got / float(want) if float(want) else float("inf")
        row = {
            "object": name,
            "asset": asset,
            "source": ("measured_table" if measured is not None else "layout"),
            "layout_kg": round(float(want), 9),
            "compiled_kg": round(got, 9),
            "ratio": round(ratio, 6),
        }
        report["checked"].append(row)
        if abs(ratio - 1.0) > tolerance:
            problems.append(
                f"{name} ({asset}): layout says {float(want)*1000:.3f} g, "
                f"compiled {got*1000:.3f} g, ratio {ratio:.3f}"
            )
    # "Present, and nothing in it was checked" is the state this whole function
    # exists to make impossible to reach quietly. A layout that records no
    # weights at all lands here: the objects resolve, every
    # one is reported unmeasured, and a reader would see a check that passed.
    # Say it instead, and let the caller decide whether that is acceptable.
    report["measured"] = bool(report["checked"])
    if problems:
        raise LayoutError(
            f"{label}: compiled object masses disagree with the layout by more "
            f"than {tolerance*100:.0f}%:\n  " + "\n  ".join(problems)
        )
    return report


def load_manual_layout(
    path: str | Path, *, expected_objects: Sequence[str]
) -> dict[str, Any]:
    """Read and validate a hand-authored layout JSON."""

    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LayoutError(f"layout JSON is unreadable: {resolved}: {exc}") from exc
    return validate_manual_layout(payload, expected_objects=expected_objects)


# ---------------------------------------------------------------------------
# Seating: what an authored "on the table" pose means once a table exists.
#
# The hand-authored Duo boards all write z
# against a flat reference plane at table z = 0 while meaning "resting on the
# table". The compiled table is not that plane: the Duo collision top sits at
# +34 mm, and a CoACD-decomposed table is many hulls whose
# top varies by millimetres and whose legs and apron sit below it. A typed z
# cannot survive that, so the registered environment translates each placed
# object vertically until its collision hulls just touch the upper surface of
# whatever supports it. XY and orientation stay exactly as authored. The
# geometry here has no MuJoCo dependency so it can be tested on its own.

#: Slack for "does this XY lie inside a hull's footprint", in metres.
SEAT_FOOTPRINT_TOLERANCE_M = 1e-7
_SEAT_CHUNK = 4096


def convex_hull_halfspaces(points: Any) -> tuple[Any, Any]:
    """Return ``(equations, vertices)`` of the convex hull of N x 3 ``points``.

    Rows of ``equations`` are ``[nx, ny, nz, offset]`` with unit normals and
    ``n . p + offset <= 0`` inside: the hull MuJoCo collides a mesh geom as.
    """

    import numpy as np
    from scipy.spatial import ConvexHull, QhullError

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 4:
        raise LayoutError("a collision hull needs at least four 3-D points")
    if not np.all(np.isfinite(pts)):
        raise LayoutError("collision hull points must be finite")
    try:
        hull = ConvexHull(pts)
    except QhullError as exc:
        raise LayoutError(f"collision hull is degenerate: {exc}") from exc
    return hull.equations.copy(), pts[hull.vertices].copy()


def _vertical_extent(points_xy: Any, equations: Any) -> tuple[Any, Any, Any]:
    """Top z, bottom z and footprint membership of one hull at each XY."""

    import numpy as np

    normal = equations[:, :3]
    offset = equations[:, 3]
    up = normal[:, 2] > 1e-12
    down = normal[:, 2] < -1e-12
    side = ~(up | down)
    linear = points_xy @ normal[:, :2].T + offset
    count = len(points_xy)
    top = (
        np.min(-linear[:, up] / normal[up, 2], axis=1)
        if up.any()
        else np.full(count, np.inf)
    )
    bottom = (
        np.max(-linear[:, down] / normal[down, 2], axis=1)
        if down.any()
        else np.full(count, -np.inf)
    )
    inside = top >= bottom - SEAT_FOOTPRINT_TOLERANCE_M
    if side.any():
        inside &= np.all(linear[:, side] <= SEAT_FOOTPRINT_TOLERANCE_M, axis=1)
    return top, bottom, inside


def hull_surface_z(
    points_xy: Any, hulls: Sequence[tuple[Any, Any]], *, upper: bool
) -> Any:
    """Upper (or lower) surface z of a hull set over each XY.

    ``upper=True`` gives the highest hull top over each point and ``-inf``
    where no hull covers it; ``upper=False`` the lowest hull bottom and
    ``+inf``. A leg or apron hull under a table top therefore never wins the
    upper surface, and a point off every hull reports no surface at all
    rather than borrowing a neighbour's height.
    """

    import numpy as np

    xy = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    out = np.full(len(xy), -np.inf if upper else np.inf)
    for equations, vertices in hulls:
        lo = vertices[:, :2].min(axis=0) - SEAT_FOOTPRINT_TOLERANCE_M
        hi = vertices[:, :2].max(axis=0) + SEAT_FOOTPRINT_TOLERANCE_M
        candidates = np.nonzero(np.all((xy >= lo) & (xy <= hi), axis=1))[0]
        for start in range(0, len(candidates), _SEAT_CHUNK):
            index = candidates[start : start + _SEAT_CHUNK]
            top, bottom, inside = _vertical_extent(xy[index], equations)
            hit = index[inside]
            if upper:
                out[hit] = np.maximum(out[hit], top[inside])
            else:
                out[hit] = np.minimum(out[hit], bottom[inside])
    return out


#: Footprints overlapping by no more than this merely touch: no rigid body can
#: be carried across a sliver that thin, so the pair shares no support.
SEAT_FOOTPRINT_OVERLAP_M = 1e-6
#: Tried in order on a program that is feasible by construction.
_OVERLAP_METHODS = ("highs", "highs-ipm", "highs-ds")


def _footprint_polygon(vertices: Any) -> Any:
    """Counter-clockwise convex polygon of the XY projection of ``vertices``.

    A convex hull's shadow on the table is exactly the convex hull of its
    vertices' XY (Andrew's monotone chain; collinear points are dropped).
    """

    import numpy as np

    points = sorted(set(map(tuple, np.asarray(vertices, dtype=float)[:, :2].tolist())))
    if len(points) < 3:
        return np.asarray(points, dtype=float).reshape(-1, 2)

    def turn(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    upper: list = []
    for point in points:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _footprint_overlap_width(first: Any, second: Any) -> float:
    """How far two convex XY polygons overlap along their least-overlapping edge normal.

    Positive: they share area. Zero or negative: they touch or lie apart
    (separating axis theorem), and the magnitude is the gap along that axis.
    """

    import numpy as np

    if len(first) < 3 or len(second) < 3:
        return -math.inf
    width = math.inf
    for polygon in (first, second):
        edges = np.roll(polygon, -1, axis=0) - polygon
        normals = np.column_stack([-edges[:, 1], edges[:, 0]])
        lengths = np.linalg.norm(normals, axis=1)
        normals = normals[lengths > 0.0] / lengths[lengths > 0.0, None]
        a = first @ normals.T
        b = second @ normals.T
        overlap = np.minimum(a.max(axis=0), b.max(axis=0)) - np.maximum(
            a.min(axis=0), b.min(axis=0)
        )
        width = min(width, float(overlap.min()))
    return width


def _max_vertical_overlap(
    object_hull: tuple[Any, Any], support_hull: tuple[Any, Any]
) -> float | None:
    """Exact max of (support top - object bottom) over one pair's footprint.

    A linear program in ``(x, y, z_support, z_object)``: ``(x, y, z_support)``
    lies in the support hull, ``(x, y, z_object)`` in the object hull, and
    ``z_support - z_object`` is maximised. ``None`` when the two footprints do
    not share area.

    Whether they share area is decided geometrically first, not by the
    solver. For disjoint footprints the program is infeasible, and HiGHS's
    simplex does not always prove it: for some disjoint pairs it stops at
    model status kUnknown (15, "primal_status is Infeasible") with no
    certificate, which scipy reports as status 4. Once
    disjoint pairs never reach the solver, every program it sees is feasible
    and bounded, so an answer other than optimal from every method is a real
    failure and raises rather than being read as "no overlap".
    """

    import numpy as np
    from scipy.optimize import linprog

    support_eq, support_vertices = support_hull
    object_eq, object_vertices = object_hull
    width = _footprint_overlap_width(
        _footprint_polygon(object_vertices), _footprint_polygon(support_vertices)
    )
    if width <= SEAT_FOOTPRINT_OVERLAP_M:
        return None
    a_ub = np.vstack(
        [
            np.hstack([support_eq[:, :3], np.zeros((len(support_eq), 1))]),
            np.hstack(
                [object_eq[:, :2], np.zeros((len(object_eq), 1)), object_eq[:, 2:3]]
            ),
        ]
    )
    b_ub = -np.concatenate([support_eq[:, 3], object_eq[:, 3]])
    failures = []
    for method in _OVERLAP_METHODS:
        result = linprog(
            c=[0.0, 0.0, -1.0, 1.0],
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=[(None, None)] * 4,
            method=method,
        )
        if result.status == 0:
            return float(-result.fun)
        failures.append(f"{method}: status {result.status}: {result.message}")
    raise LayoutError(
        "seating overlap program did not solve for hulls whose footprints share "
        f"{width * 1000.0:.3f} mm of width: " + "; ".join(failures)
    )


def seat_lift(
    object_hulls: Sequence[tuple[Any, Any]],
    support_hulls: Sequence[tuple[Any, Any]],
) -> float | None:
    """Vertical translation that rests ``object_hulls`` on ``support_hulls``.

    Positive raises the object, negative lowers it. The value is the largest
    vertical overlap between the support's upper surface and the object's
    lower surface over their shared footprint, exact for convex hull sets.
    Returns ``None`` when no part of the object lies over the support.

    Hull vertices of both sets give a cheap lower bound, which is already
    exact on a flat support such as the Duo table. An overlap can still peak
    where an edge of one surface crosses an edge of the other -- a narrow
    CoACD edge hull running under a wide flat base -- so every hull pair whose
    height bound could beat it is solved exactly, most promising first.
    """

    import numpy as np

    if not object_hulls or not support_hulls:
        raise LayoutError("seating needs at least one object hull and one support hull")
    object_points = np.vstack([vertices for _, vertices in object_hulls])
    support_points = np.vstack([vertices for _, vertices in support_hulls])
    support_top = hull_surface_z(object_points[:, :2], support_hulls, upper=True)
    object_bottom = hull_surface_z(support_points[:, :2], object_hulls, upper=False)
    sampled = np.concatenate(
        [
            (support_top - object_points[:, 2])[np.isfinite(support_top)],
            (support_points[:, 2] - object_bottom)[np.isfinite(object_bottom)],
        ]
    )
    best = float(sampled.max()) if sampled.size else None

    candidates = []
    for object_hull in object_hulls:
        object_vertices = object_hull[1]
        lo = object_vertices[:, :2].min(axis=0) - SEAT_FOOTPRINT_TOLERANCE_M
        hi = object_vertices[:, :2].max(axis=0) + SEAT_FOOTPRINT_TOLERANCE_M
        floor = float(object_vertices[:, 2].min())
        for support_hull in support_hulls:
            support_vertices = support_hull[1]
            if np.any(support_vertices[:, :2].min(axis=0) > hi) or np.any(
                support_vertices[:, :2].max(axis=0) < lo
            ):
                continue
            bound = float(support_vertices[:, 2].max()) - floor
            candidates.append((bound, object_hull, support_hull))
    candidates.sort(key=lambda item: item[0], reverse=True)
    for bound, object_hull, support_hull in candidates:
        if best is not None and bound <= best + SEAT_FOOTPRINT_TOLERANCE_M:
            break
        overlap = _max_vertical_overlap(object_hull, support_hull)
        if overlap is not None and (best is None or overlap > best):
            best = overlap
    return best


__all__ = [
    "LayoutError",
    "MANUAL_PLACEMENT_KIND",
    "SEAT_FOOTPRINT_TOLERANCE_M",
    "convex_hull_halfspaces",
    "hull_surface_z",
    "load_manual_layout",
    "seat_lift",
    "validate_manual_layout",
]


def assert_measured_masses(
    model,
    *,
    mujoco,
    tolerance: float = LAYOUT_MASS_TOLERANCE,
    measured_masses=None,
    extra_bodies=None,
    skip_bodies=(),
):
    """Every object of the measured table that this model compiled weighs right.

    This does not need a layout, which is the point: cup_plate places its two
    objects from RS_CUP_XY/RS_PLATE_XY and has no layout file at all, so a
    layout-driven check could never reach it. Sweeping the table against the
    compiled bodies covers every task the same way, including the ones whose
    contracts record no weights.

    Bodies are matched by the asset's compiled name and by ``cube_main``, which
    is what the runtime renames the task's selected object to.
    """

    table = load_measured_masses(measured_masses)
    report = {"tolerance": float(tolerance), "checked": [], "absent": []}
    problems = []
    # The task's selected object is compiled under a name that says nothing
    # about which asset it is -- cup_plate's cup is cube_main -- so the caller,
    # which built it and knows, names it here, so the one object the policy
    # actually manipulates is checked too.
    skip = set(skip_bodies or ())
    for body_name, asset in sorted(dict(extra_bodies or {}).items()):
        if body_name in skip:
            continue
        want = table.get(asset)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if want is None or body_id < 0:
            continue
        got = float(model.body_mass[body_id])
        ratio = got / float(want) if float(want) else float("inf")
        report["checked"].append({
            "asset": asset, "body": body_name,
            "measured_kg": round(float(want), 9),
            "compiled_kg": round(got, 9), "ratio": round(ratio, 6),
        })
        if abs(ratio - 1.0) > tolerance:
            problems.append(
                f"{body_name} ({asset}): {MEASURED_MASSES_PATH.name} records "
                f"{float(want)*1000:.3f} g, compiled {got*1000:.3f} g, "
                f"ratio {ratio:.3f}"
            )
    for asset, want in sorted(table.items()):
        found = []
        for candidate in (f"{asset}_main", asset):
            if candidate in skip:
                continue
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
            if body_id >= 0:
                found.append((candidate, body_id))
        if not found:
            report["absent"].append(asset)
            continue
        for candidate, body_id in found:
            got = float(model.body_mass[body_id])
            ratio = got / float(want) if float(want) else float("inf")
            report["checked"].append({
                "asset": asset,
                "body": candidate,
                "measured_kg": round(float(want), 9),
                "compiled_kg": round(got, 9),
                "ratio": round(ratio, 6),
            })
            if abs(ratio - 1.0) > tolerance:
                problems.append(
                    f"{candidate}: {MEASURED_MASSES_PATH.name} records "
                    f"{float(want)*1000:.3f} g, compiled {got*1000:.3f} g, "
                    f"ratio {ratio:.3f}"
                )
    if problems:
        raise LayoutError(
            "compiled object masses disagree with the measured masses by "
            f"more than {tolerance*100:.0f}%:\n  " + "\n  ".join(problems)
        )
    return report

