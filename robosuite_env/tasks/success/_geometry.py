# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Model and option helpers shared by the success criteria.

The criteria stay independent classes on purpose -- the registry instantiates
exactly one per rollout, and a shared base class would tie unrelated tasks
together through ``__init__`` -- but each of them has to do the same small
things: read a threshold out of the options dict, round a vector for the JSON
log, find a body by a fragment of its name, measure a body's collision
footprint, and know where the table is. Each of those lived in two or three
copies. This is the one copy.

``collision_footprint`` places each hull's vertices by the geom's own pose
*before* measuring; taking the max of the raw per-hull extents would measure
only the largest single hull in its own frame.

``TableSurface`` is the other load-bearing one. Heights are measured from the
table collision geometry that is compiled into the model -- the thing objects
actually rest on -- rather than from the schema-v2 reference plane
(``table_z_m``, always 0.0), so a change to the table is picked up by these
criteria automatically.

The quaternion expansion here is deliberately not ``cup_on_plate._quat_to_mat``:
that one normalises first, which is right for a free body's ``xquat`` but would
change, in the last bits, extents that scoring thresholds are compared against.
MuJoCo's compiler already hands out unit ``geom_quat`` and ``body_quat``.
"""

from __future__ import annotations

import math

import re

import numpy as np


# mujoco.mjtGeom values, spelled out so the unit tests' fake models need no
# MuJoCo import.
_GEOM_BOX = 6
_GEOM_MESH = 7

# ConvexHull equations and MuJoCo's compiled mesh vertices can differ at the
# sub-micrometre level. 0.1 um absorbs that numerical roundoff while remaining
# four orders below the millimetre-scale physical scoring tolerances.
_COLLISION_HULL_NUMERICAL_TOLERANCE_M = 1e-7
# ConvexHull normals are unit length analytically; components this small are
# numerical zero when deciding whether a face supplies a vertical bound.
_COLLISION_HULL_NORMAL_EPSILON = 1e-9
# At roughly 1,000 faces on the largest current CoACD hull, this caps the
# temporary point-by-face height matrix near 16 MiB while retaining NumPy's
# array-level evaluation instead of one Python call per collision vertex.
_HEIGHT_QUERY_BATCH_POINTS = 2048

TABLE_TOP_OVERRIDE_OPTION = "table_top_z_m"
REFERENCE_PLANE_OPTION = "table_z_m"


def as_float(options, key, default):
    """``options[key]`` as a float, falling back to ``default`` if unreadable.

    Options arrive from CLI/JSON/env plumbing, so a key can be absent, ``None``
    or a string that does not parse; none of those should crash a rollout.
    """

    try:
        return float(options.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def optional_float(options, key, *, owner):
    """``options[key]`` as a finite float, or ``None`` when the key is unset.

    Unlike ``as_float`` there is no default to fall back on: an override that
    is present but unreadable is a configuration error, and ignoring it would
    silently score against something the config did not ask for.
    """

    value = options.get(key)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = math.nan
    if isinstance(value, bool) or not math.isfinite(result):
        raise ValueError(
            f"{owner}: {key} must be a finite height in metres; got {value!r}"
        )
    return result


def round_list(value, ndigits=6):
    """Flatten ``value`` to a plain list of rounded floats, for JSON logs."""

    return [round(float(v), ndigits) for v in np.asarray(value, dtype=float).ravel()]


def body_id_by_token(model, token, *, owner):
    """Index of the first body whose name *contains* ``token``.

    Criteria name their objects by fragment ("box", "tape") because
    the layouts prefix and suffix the full body names per scene. ``owner`` is
    the criterion's name, so a model missing the object says which check went
    looking for it.
    """

    for index in range(model.nbody):
        if token in (model.body(index).name or ""):
            return index
    raise ValueError(f"{owner}: no body matching {token!r} in the model")


def _quat_to_mat(quat):
    w, x, y, z = np.asarray(quat, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def collision_footprint(model, body_id):
    """Extent of the body's collision geoms, in the body frame, as ``(lo, hi)``.

    Each hull's vertices are mesh-local, so they have to be placed by the
    geom's own pose before the extents mean anything; taking the max of the
    raw per-hull extents measures the biggest single hull instead.
    """

    lo = None
    hi = None
    for index in range(model.ngeom):
        if model.geom_bodyid[index] != body_id or model.geom_group[index] != 0:
            continue
        mesh = model.geom_dataid[index]
        if mesh < 0:
            continue
        start = model.mesh_vertadr[mesh]
        count = model.mesh_vertnum[mesh]
        verts = np.asarray(model.mesh_vert[start : start + count], dtype=float)
        mat = _quat_to_mat(model.geom_quat[index])
        placed = verts @ mat.T + np.asarray(model.geom_pos[index], dtype=float)
        lo = placed.min(axis=0) if lo is None else np.minimum(lo, placed.min(axis=0))
        hi = placed.max(axis=0) if hi is None else np.maximum(hi, placed.max(axis=0))
    if lo is None:
        raise ValueError("no collision geometry to measure")
    return lo, hi


# -- the table ----------------------------------------------------------------


def is_table_collision_geom(geom_name, body_name):
    """Whether a geom is part of the registered table's collision.

    The registered GW envs compile the scene contract's CoACD parts as
    ``gw_desk_coll_<j>`` directly under the worldbody, for every rig (FR3 flat,
    FR3 Duo) and every scene; the other spellings are the arena and
    fixture names.
    """

    name = (geom_name or "").lower()
    body = (body_name or "").lower()
    return bool(
        name == "table_collision"
        or name.startswith("gw_desk_coll_")
        or name.startswith("desk_coll_")
        or "table_collision" in name
        or "desk_collision" in name
        or "table_collision" in body
        or body.startswith("gw_desk_coll_")
    )


def _model_geom_name(model, geom_id):
    try:
        return model.geom(int(geom_id)).name or f"geom{geom_id}"
    except (AttributeError, TypeError, IndexError, KeyError):
        return f"geom{geom_id}"


def _model_body_name(model, body_id):
    try:
        return model.body(int(body_id)).name or f"body{body_id}"
    except (AttributeError, TypeError, IndexError, KeyError):
        return f"body{body_id}"


def table_collision_geom_ids(model, *, geom_name=None, body_name=None):
    """Ids of every geom ``is_table_collision_geom`` accepts.

    ``geom_name`` / ``body_name`` map an id to its name; by default they read
    ``model.geom(i).name`` and ``model.body(i).name``.
    """

    geom_name = geom_name or (lambda index: _model_geom_name(model, index))
    body_name = body_name or (lambda index: _model_body_name(model, index))
    return frozenset(
        index
        for index in range(int(model.ngeom))
        if is_table_collision_geom(
            geom_name(index), body_name(int(model.geom_bodyid[index]))
        )
    )


def _geom_pose_in_body(model, geom_id, ancestor):
    """``(rotation, position)`` of a geom in ``ancestor``'s frame, else ``None``.

    Composed from the compiled model -- the geom's own pose and every body pose
    between it and ``ancestor`` -- rather than read from ``MjData``, so the
    answer does not depend on whether ``mj_forward`` has run yet (the trap that
    made the container's construction-time rest height read 0.0). ``None`` when the
    geom is not rigidly attached to ``ancestor``: it hangs off another branch,
    or a joint or mocap body sits in between. ``ancestor`` 0 is the world, so
    there a pose means a geom that cannot move.
    """

    rotation = _quat_to_mat(model.geom_quat[geom_id])
    position = np.asarray(model.geom_pos[geom_id], dtype=float)
    body_pos = getattr(model, "body_pos", None)
    body_quat = getattr(model, "body_quat", None)
    parents = getattr(model, "body_parentid", None)
    joint_counts = getattr(model, "body_jntnum", None)
    mocap_ids = getattr(model, "body_mocapid", None)
    ancestor = int(ancestor)
    body = int(model.geom_bodyid[geom_id])
    for _ in range(int(getattr(model, "nbody", 0)) + 1):
        if body == ancestor:
            return rotation, position
        if body <= 0:
            return None
        if joint_counts is not None and int(joint_counts[body]) > 0:
            return None
        if mocap_ids is not None and int(mocap_ids[body]) >= 0:
            return None
        if body_pos is not None and body_quat is not None:
            body_rotation = _quat_to_mat(body_quat[body])
            rotation = body_rotation @ rotation
            position = body_rotation @ position + np.asarray(
                body_pos[body], dtype=float
            )
        if parents is None:
            # A model with no body tree: every body hangs off the world.
            return (rotation, position) if ancestor == 0 else None
        body = int(parents[body])
    return None


def _static_geom_pose(model, geom_id):
    """World ``(rotation, position)`` of a geom that cannot move, else ``None``."""

    return _geom_pose_in_body(model, geom_id, 0)


def _geom_local_vertices(model, geom_id):
    """Vertices of a mesh or box geom in its own frame; ``None`` otherwise."""

    types = getattr(model, "geom_type", None)
    kind = None if types is None else int(types[geom_id])
    data_ids = getattr(model, "geom_dataid", None)
    if (
        kind in (None, _GEOM_MESH)
        and data_ids is not None
        and int(data_ids[geom_id]) >= 0
    ):
        mesh = int(data_ids[geom_id])
        start = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        return np.asarray(model.mesh_vert[start : start + count], dtype=float)
    if kind == _GEOM_BOX:
        hx, hy, hz = np.asarray(model.geom_size[geom_id], dtype=float)[:3]
        return np.array(
            [
                [sx * hx, sy * hy, sz * hz]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ]
        )
    return None


def _hull_halfspaces(points):
    """``(k, 4)`` rows ``[n, d]`` with ``n . p + d <= 0`` inside the hull.

    MuJoCo collides against a mesh geom's convex hull, so the hull -- not the
    raw triangles -- is the surface an object rests on. A flat point set has
    no hull; its axis-aligned box stands in.
    """

    from scipy.spatial import ConvexHull, QhullError

    try:
        return np.asarray(ConvexHull(points).equations, dtype=float)
    except (QhullError, ValueError):
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        return np.array(
            [
                [1.0, 0.0, 0.0, -hi[0]],
                [-1.0, 0.0, 0.0, lo[0]],
                [0.0, 1.0, 0.0, -hi[1]],
                [0.0, -1.0, 0.0, lo[1]],
                [0.0, 0.0, 1.0, -hi[2]],
                [0.0, 0.0, -1.0, lo[2]],
            ]
        )


def _hull_vertices(points):
    """The extreme points of a vertex set: all a footprint or extent can use."""

    from scipy.spatial import ConvexHull, QhullError

    try:
        return points[ConvexHull(points).vertices]
    except (QhullError, ValueError):
        return points


def subtree_collision_geom_ids(model, root_body):
    """Collision-enabled geoms rigidly attached below ``root_body``."""

    contype = getattr(model, "geom_contype", None)
    conaffinity = getattr(model, "geom_conaffinity", None)
    result = []
    for geom_id in range(int(model.ngeom)):
        if (
            contype is not None
            and conaffinity is not None
            and not (int(contype[geom_id]) or int(conaffinity[geom_id]))
        ):
            continue
        if _geom_pose_in_body(model, geom_id, root_body) is not None:
            result.append(geom_id)
    return frozenset(result)


def collision_geom_vertices_in_body(model, geom_id, root_body):
    """One collision geom's hull vertices in ``root_body``'s frame."""

    pose = _geom_pose_in_body(model, geom_id, root_body)
    local = _geom_local_vertices(model, geom_id)
    if pose is None or local is None or len(local) == 0:
        return None
    rotation, position = pose
    return _hull_vertices(local @ rotation.T + position)


def subtree_collision_vertex_parts(model, root_body):
    """Measurable ``(geom_id, hull vertices)`` below ``root_body``."""

    pieces = []
    for geom_id in sorted(subtree_collision_geom_ids(model, root_body)):
        vertices = collision_geom_vertices_in_body(model, geom_id, root_body)
        if vertices is not None:
            pieces.append((geom_id, vertices))
    return tuple(pieces)


def subtree_collision_vertices(model, root_body):
    """Collision hull vertices of a body and its rigid children, in its frame.

    robosuite nests an object's geoms one body below the free-joint root, so
    the root's own geoms are not the object; every collision geom rigidly
    attached to ``root_body`` is. Only each hull's extreme vertices are kept.
    ``None`` if nothing measurable is attached.
    """

    pieces = [
        vertices
        for _, vertices in subtree_collision_vertex_parts(model, root_body)
    ]
    return np.vstack(pieces) if pieces else None


class TableSurface:
    """The table's supporting surface, measured from its collision geometry.

    ``height_at(x, y)`` is the upper envelope of the table's collision hulls
    over one plan point: the height an object standing there rests at. That is
    deliberately not the highest collision vertex (``top_z_m``). A single slab
    has one top, but a CoACD-decomposed table can be 25 parts
    whose top faces are not coplanar -- its highest vertex is a corner of the
    main slab 5.5 mm above other supporting parts -- and a 3 mm penetration
    tolerance cannot absorb a datum that is off by that much.
    """

    def __init__(self, hulls, geoms, unmeasured=()):
        self._hulls = tuple(hulls)
        self.geoms = tuple(geoms)
        self.unmeasured = tuple(unmeasured)

    @classmethod
    def from_model(cls, model, geom_ids, *, geom_name=None):
        geom_name = geom_name or (lambda index: _model_geom_name(model, index))
        contype = getattr(model, "geom_contype", None)
        conaffinity = getattr(model, "geom_conaffinity", None)
        hulls = []
        names = []
        unmeasured = []
        for geom_id in sorted(int(value) for value in geom_ids):
            name = geom_name(geom_id)
            if (
                contype is not None
                and conaffinity is not None
                and not (int(contype[geom_id]) or int(conaffinity[geom_id]))
            ):
                unmeasured.append(f"{name} (collision disabled)")
                continue
            local = _geom_local_vertices(model, geom_id)
            pose = _static_geom_pose(model, geom_id)
            if local is None or len(local) == 0 or pose is None:
                unmeasured.append(name)
                continue
            rotation, position = pose
            world = local @ rotation.T + position
            hulls.append(
                (_hull_halfspaces(world), world.min(axis=0), world.max(axis=0))
            )
            names.append(name)
        return cls(hulls, names, unmeasured)

    def __bool__(self):
        return bool(self._hulls)

    @property
    def top_z_m(self):
        """Highest table collision vertex, or ``None`` with no table."""

        if not self._hulls:
            return None
        return max(float(hi[2]) for _, _, hi in self._hulls)

    def height_at(self, x, y):
        """Upper envelope of the table hulls above ``(x, y)``; ``None`` if off it."""

        tolerance = _COLLISION_HULL_NUMERICAL_TOLERANCE_M
        best = None
        for equations, lo, hi in self._hulls:
            if not (
                lo[0] - tolerance <= x <= hi[0] + tolerance
                and lo[1] - tolerance <= y <= hi[1] + tolerance
            ):
                continue
            normal_z = equations[:, 2]
            # n . p + d <= 0  ->  n_z * z <= -(d + n_x x + n_y y)
            bound = -(equations[:, 3] + equations[:, 0] * x + equations[:, 1] * y)
            up = normal_z > _COLLISION_HULL_NORMAL_EPSILON
            down = normal_z < -_COLLISION_HULL_NORMAL_EPSILON
            side = ~(up | down)
            if np.any(bound[side] < -tolerance) or not np.any(up):
                continue
            top = float(np.min(bound[up] / normal_z[up]))
            bottom = (
                float(np.max(bound[down] / normal_z[down]))
                if np.any(down)
                else -math.inf
            )
            if top + tolerance < bottom:
                continue
            best = top if best is None else max(best, top)
        return best

    def support_height(self, points_xy):
        """Highest table surface under any of ``points_xy``; ``None`` if none."""

        heights = [
            height
            for height in (
                self.height_at(float(x), float(y)) for x, y in points_xy
            )
            if height is not None
        ]
        return max(heights) if heights else None

    def summary(self):
        top = self.top_z_m
        return {
            "geoms": list(self.geoms),
            "geom_count": len(self.geoms),
            "top_vertex_z_m": None if top is None else round(top, 6),
            "unmeasured_geoms": list(self.unmeasured),
        }


def footprint_grid_xy(position, rotation, lo, hi, *, steps=5):
    """World plan points of a ``steps`` x ``steps`` grid over a body footprint.

    ``lo`` / ``hi`` are the body-frame collision extents from
    ``collision_footprint``; a rigid body rests on the highest table surface
    anywhere under it, so the datum samples the whole footprint, not its centre.
    """

    xs = np.linspace(float(lo[0]), float(hi[0]), steps)
    ys = np.linspace(float(lo[1]), float(hi[1]), steps)
    local = np.array([[x, y, 0.0] for x in xs for y in ys])
    world = local @ np.asarray(rotation, dtype=float).reshape(3, 3).T + np.asarray(
        position, dtype=float
    )
    return world[:, :2]


def _datum_record(*, z_m, source, reference_plane_z_m, surface, **extra):
    record = {
        "z_m": None if z_m is None else round(float(z_m), 6),
        "source": source,
        "reference_plane_z_m": round(float(reference_plane_z_m), 6),
        "above_reference_plane_mm": (
            None
            if z_m is None
            else round((float(z_m) - float(reference_plane_z_m)) * 1000.0, 3)
        ),
        "table_collision": surface.summary(),
    }
    record.update(extra)
    return record


class TableDatum:
    """One table height per episode, measured under a body's footprint.

    Resolution order: an explicit ``table_top_z_m`` option, else the table
    collision surface under the named body's footprint, else (footprint not
    over the table at all) the table's highest collision vertex -- each labelled
    in ``describe()``. The ``table_z_m`` option is only ever *reported*, as the
    reference plane: the runtime always passes it (the schema-v2 plane, 0.0),
    so it cannot tell "a config chose this" from "nobody did", and preferring
    it is exactly the defect this replaces.

    The footprint is read on the first evaluation, not at construction:
    construction happens before the objects are released onto the table, when
    body poses are still the spawn placeholders.
    """

    def __init__(self, *, model, options, owner, geom_ids=None):
        self.owner = owner
        self.reference_plane_z_m = as_float(options, REFERENCE_PLANE_OPTION, 0.0)
        self.override_m = optional_float(
            options, TABLE_TOP_OVERRIDE_OPTION, owner=owner
        )
        geom_ids = table_collision_geom_ids(model) if geom_ids is None else geom_ids
        self.surface = TableSurface.from_model(model, geom_ids)
        if self.override_m is None and not self.surface:
            raise ValueError(
                f"{owner}: heights are measured from the table, but the model has "
                "no measurable table collision geometry (gw_desk_coll_* / "
                "table_collision) and no table_top_z_m override was given"
            )
        self.z_m = None
        self._record = None
        if self.override_m is not None:
            self.z_m = self.override_m
            self._record = _datum_record(
                z_m=self.override_m,
                source=f"option:{TABLE_TOP_OVERRIDE_OPTION}",
                reference_plane_z_m=self.reference_plane_z_m,
                surface=self.surface,
            )

    def resolve(self, position, rotation, lo, hi, *, subject, time_s=None):
        """Latch the datum under ``subject``'s footprint; return it in metres."""

        if self.z_m is not None:
            return self.z_m
        support = self.surface.support_height(
            footprint_grid_xy(position, rotation, lo, hi)
        )
        if support is not None:
            z_m = support
            source = f"model:table_collision_surface_under_{subject}_footprint"
        else:
            z_m = self.surface.top_z_m
            source = (
                "model:table_collision_top_vertex "
                f"({subject} footprint was not over the table)"
            )
        self.z_m = float(z_m)
        self._record = _datum_record(
            z_m=self.z_m,
            source=source,
            reference_plane_z_m=self.reference_plane_z_m,
            surface=self.surface,
            measured_under=subject,
            measured_at_xy_m=round_list(np.asarray(position, dtype=float)[:2]),
            measured_at_sim_time_s=(
                None if time_s is None else round(float(time_s), 6)
            ),
        )
        return self.z_m

    def describe(self):
        if self._record is not None:
            return dict(self._record)
        return _datum_record(
            z_m=None,
            source="pending: measured under the object's footprint at the first evaluation",
            reference_plane_z_m=self.reference_plane_z_m,
            surface=self.surface,
        )


def data_time(data):
    """``data.time`` as a float, or ``None`` if the data has no usable clock."""

    try:
        value = float(data.time)
    except (AttributeError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


__all__ = [
    "REFERENCE_PLANE_OPTION",
    "TABLE_TOP_OVERRIDE_OPTION",
    "TableDatum",
    "TableSurface",
    "as_float",
    "body_id_by_token",
    "collision_geom_vertices_in_body",
    "collision_footprint",
    "collision_geom_vertices_in_body",
    "data_time",
    "footprint_grid_xy",
    "geom_local_extent",
    "geom_local_vertices",
    "gripper_geoms_by_side",
    "gripper_prefix_by_side",
    "is_table_collision_geom",
    "optional_float",
    "round_list",
    "subtree_collision_geom_ids",
    "subtree_collision_vertex_parts",
    "subtree_collision_vertices",
    "table_collision_geom_ids",
]


# --- box_pack helpers ------------------------------------------------------
# The box criterion (box_pack.py) resolves hands by gripper-subtree prefix and
# reads raw mesh vertices.

# Which subtree is which hand is NOT readable from the side word in its name:
#
#   Duo   right gripper0_right   left gripper1_right   <- the ARM INDEX decides
#
# Duo's left gripper is called "gripper1_right", so a rig that read the word
# would put both of Duo's hands on the right. So the mapping is derived from
# the compiled model: two roots differing in index are the Duo shape, and
# anything else raises.
_GRIPPER_ROOT_RE = re.compile(r"^(gripper(\d+)_(right|left))_")


def gripper_prefix_by_side(model, *, owner):
    """``{"right": prefix, "left": prefix}`` for whichever rig this model is."""

    roots = {}
    for body in range(int(model.nbody)):
        match = _GRIPPER_ROOT_RE.match(model.body(body).name or "")
        if match:
            roots[match.group(1)] = (int(match.group(2)), match.group(3))
    if len(roots) != 2:
        raise RuntimeError(
            f"{owner}: expected exactly two gripper subtrees, found "
            f"{sorted(roots) or 'none'}"
        )
    (_, (index_a, word_a)), (_, (index_b, word_b)) = sorted(roots.items())
    # Return the SHORTEST prefix that still separates the two hands. Not every
    # body under a gripper repeats the subtree root: the finger bodies are
    # gripper<N>_finger_left and gripper<N>_finger_right, so a prefix that
    # includes the root's side word matches the root and nothing below it, and
    # the walk up from a finger geom only reaches it if the compiled tree
    # actually nests them. On Duo the arm index alone is unambiguous, so use it.
    if index_a != index_b and word_a == word_b:
        # Duo: one robosuite robot per arm, robot0 is the right one.
        return {"right": f"gripper{min(index_a, index_b)}_",
                "left": f"gripper{max(index_a, index_b)}_"}
    raise RuntimeError(
        f"{owner}: cannot tell the two grippers apart from {sorted(roots)}; "
        "they do not differ in the arm index alone"
    )


def gripper_geoms_by_side(model, *, owner, fingers_only=True):
    """Split the gripper geoms into ``{"left": ..., "right": ...}``.

    Membership is decided by walking each geom's body up to its gripper
    subtree root, whose name carries the arm index. Do not match on the words
    "left" and "right" anywhere in a body or geom name: every YUBI gripper
    contains a body named ``finger_left`` *and* one named ``finger_right``, so
    a substring test for "left" collects the left finger of *both* arms and
    one for "right" the right finger of both (plus every ``*_right_gripper``
    subtree root and ``robot<N>_right_hand``). The two sets come out disjoint,
    but each spans both arms -- so one gripper closing its two fingers on the
    container would set "left holds it" and "right holds it" at once. Tokens
    such as ``yubi_left``/``yubi_right`` match nothing on this rig either: the
    compiled body is ``gripper<N>_yubi_geom``.

    With ``fingers_only`` (the default) only geoms under a ``finger`` body
    count, so a palm or guard brushing the object is not "holding" it.

    Raises if the two sides do not come out non-empty and disjoint, so a
    renamed model fails loudly instead of quietly scoring one hand as two.
    """

    prefix_by_side = gripper_prefix_by_side(model, owner=owner)

    def classify(body_id):
        body = int(body_id)
        on_finger = False
        while body > 0:
            name = model.body(body).name or ""
            if "finger" in name.lower():
                on_finger = True
            for side, prefix in prefix_by_side.items():
                if name.startswith(prefix):
                    if fingers_only and not on_finger:
                        return None
                    return side
            body = int(model.body_parentid[body])
        return None

    sides = {side: set() for side in prefix_by_side}
    for index in range(int(model.ngeom)):
        side = classify(model.geom_bodyid[index])
        if side is not None:
            sides[side].add(index)

    empty = sorted(side for side, geoms in sides.items() if not geoms)
    if empty:
        raise RuntimeError(
            f"{owner}: found no gripper geoms for {', '.join(empty)} -- expected "
            f"subtrees named {sorted(prefix_by_side.values())}"
        )
    shared = sides["left"] & sides["right"]
    if shared:
        raise RuntimeError(
            f"{owner}: {len(shared)} geoms were assigned to both grippers"
        )
    return sides


def geom_local_extent(model, geom_id):
    """One collision geom's z extent in its body frame, or ``None``.

    Same placement as :func:`collision_footprint`, for a single hull: the
    vertices are mesh-local, so the geom's own pose has to be applied before
    the extent means anything in the body frame.
    """

    placed = geom_local_vertices(model, geom_id)
    if placed is None:
        return None
    return float(placed[:, 2].min()), float(placed[:, 2].max())


def geom_local_vertices(model, geom_id):
    """A mesh geom's vertices placed in its owning body frame.

    MuJoCo stores mesh vertices in the mesh frame and a separate pose on each
    geom.  Collision classification and cavity construction both need all
    hulls in one coordinate system, so keep that transform in one place.
    Non-mesh geoms return ``None``.
    """

    mesh = model.geom_dataid[geom_id]
    if mesh < 0:
        return None
    start = model.mesh_vertadr[mesh]
    count = model.mesh_vertnum[mesh]
    verts = np.asarray(model.mesh_vert[start : start + count], dtype=float)
    w, x, y, z = np.asarray(model.geom_quat[geom_id], dtype=float)
    mat = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    placed = verts @ mat.T + np.asarray(model.geom_pos[geom_id], dtype=float)
    return placed
