# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Success criterion for the twelve box-packing prompts.

Each prompt names one object and one hand, in one of two directions, so the
criterion is parameterised rather than fixed: ``box_pack_target_object`` and
``box_pack_hand`` for the identity, ``box_pack_direction`` for which way the
object is meant to travel.

Three things are checked.

* The object ends where the prompt asked -- inside the box for ``into_box``,
  on the table outside it for ``out_of_box``.
* The named hand is the one that moved it. A rollout that nudges the tape in
  with the wrong gripper has not followed the instruction, so the hand is
  latched the first time it grips the object clear of the surface.
* The box itself stayed on the table. Knocking the box off and calling the
  contents "packed" is a failure, and it is a failure for the whole episode:
  once the box has left the table the run cannot recover.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ..equalities import RULE_GRASP_WELD_NAMES
from ._geometry import (
    as_float,
    body_id_by_token,
    geom_local_vertices,
    gripper_geoms_by_side,
)


_HANDS = {"l": "left", "left": "left", "r": "right", "right": "right"}
_DIRECTIONS = {
    "in": "into_box",
    "into_box": "into_box",
    "out": "out_of_box",
    "out_of_box": "out_of_box",
}

# The cross-section must clear the floor/wall seam, where a hull can collapse
# to a point, but still represent the cavity immediately above its floor.  A
# fraction of the derived cavity height scales to other box sizes and avoids a
# model-specific distance.
_CAVITY_SLICE_HEIGHT_FRACTION = 0.01


def _hull_line_interval(equations, *, axis, fixed_value, z_value):
    """Interval occupied by a convex hull along one horizontal line.

    ``scipy.spatial.ConvexHull.equations`` describes the hull as half-spaces
    whose interior is non-positive.  Fixing z and the other horizontal axis
    reduces those half-spaces to an exact interval, without rasterising the
    collision mesh.
    """

    other_axis = 1 - axis
    coefficients = equations[:, axis]
    constants = (
        equations[:, other_axis] * fixed_value
        + equations[:, 2] * z_value
        + equations[:, 3]
    )
    lower = -np.inf
    upper = np.inf
    tolerance = 1e-9
    for coefficient, constant in zip(coefficients, constants):
        if abs(float(coefficient)) <= 1e-12:
            if constant > tolerance:
                return None
            continue
        boundary = -float(constant) / float(coefficient)
        if coefficient > 0.0:
            upper = min(upper, boundary)
        else:
            lower = max(lower, boundary)
    if lower > upper + tolerance:
        return None
    return float(lower), float(upper)


def _enclosed_axis_bounds(walls, *, axis, seed_xy, slice_z):
    """Nearest wall material on the negative and positive axis rays."""

    seed = float(seed_xy[axis])
    intervals = []
    for wall in walls:
        interval = _hull_line_interval(
            wall["equations"],
            axis=axis,
            fixed_value=float(seed_xy[1 - axis]),
            z_value=slice_z,
        )
        if interval is not None:
            intervals.append((wall["geom"], *interval))

    tolerance = 1e-9
    crossing = [
        gid
        for gid, lo, hi in intervals
        if lo <= seed + tolerance and hi >= seed - tolerance
    ]
    if crossing:
        raise ValueError(
            "box_pack: floor-derived cavity centre intersects wall collision "
            f"geometry {crossing}"
        )
    negative = [hi for _, _, hi in intervals if hi <= seed + tolerance]
    positive = [lo for _, lo, _ in intervals if lo >= seed - tolerance]
    if not negative or not positive:
        axis_name = "x" if axis == 0 else "y"
        raise ValueError(
            f"box_pack: collision walls do not enclose the cavity along {axis_name}"
        )
    return max(negative), min(positive)


def _derive_box_cavity(model, body_id, *, floor_cut_m):
    """Derive local ``(lo, hi)`` cavity bounds and floor geoms from hulls.

    Hulls ending below ``floor_cut_m`` form the floor.  Of the remaining
    hulls, those beginning at the floor top form the walls; higher hulls are
    lid flaps and do not contribute.  The floor AABB centre supplies a point
    known to lie near the cavity centre.  At a slice just above the floor, the
    nearest wall intersection in each of the four horizontal directions gives
    the enclosed, axis-aligned free region.  All calculations stay in the box
    body frame.
    """

    hulls = []
    for geom_id in range(int(model.ngeom)):
        if (
            int(model.geom_bodyid[geom_id]) != int(body_id)
            or int(model.geom_group[geom_id]) != 0
        ):
            continue
        vertices = geom_local_vertices(model, geom_id)
        if vertices is None:
            continue
        vertices = np.asarray(vertices, dtype=float)
        hulls.append(
            {
                "geom": int(geom_id),
                "vertices": vertices,
                "lo": vertices.min(axis=0),
                "hi": vertices.max(axis=0),
            }
        )
    if not hulls:
        raise ValueError("box_pack: box has no collision mesh hulls")

    floor = [hull for hull in hulls if hull["hi"][2] <= floor_cut_m]
    if not floor:
        raise ValueError(
            "box_pack: no floor collision hulls end below "
            f"box_pack_floor_cut_m={floor_cut_m:g}"
        )
    floor_geom_ids = {hull["geom"] for hull in floor}
    floor_top = max(float(hull["hi"][2]) for hull in floor)
    all_z_lo = min(float(hull["lo"][2]) for hull in hulls)
    all_z_hi = max(float(hull["hi"][2]) for hull in hulls)
    seam_tolerance = max(1e-9, (all_z_hi - all_z_lo) * 1e-6)
    walls = [
        hull
        for hull in hulls
        if hull["geom"] not in floor_geom_ids
        and hull["lo"][2] <= floor_top + seam_tolerance
    ]
    if not walls:
        raise ValueError("box_pack: no wall collision hulls meet the box floor")

    wall_top = max(float(hull["hi"][2]) for hull in walls)
    if wall_top <= floor_top:
        raise ValueError("box_pack: collision walls do not rise above the floor")
    slice_z = floor_top + _CAVITY_SLICE_HEIGHT_FRACTION * (wall_top - floor_top)

    for wall in walls:
        try:
            wall["equations"] = ConvexHull(wall["vertices"]).equations
        except QhullError as error:
            raise ValueError(
                "box_pack: collision geom "
                f"{wall['geom']} is not a three-dimensional convex hull"
            ) from error

    floor_xy_lo = np.min([hull["lo"][:2] for hull in floor], axis=0)
    floor_xy_hi = np.max([hull["hi"][:2] for hull in floor], axis=0)
    seed_xy = (floor_xy_lo + floor_xy_hi) * 0.5
    x_lo, x_hi = _enclosed_axis_bounds(
        walls, axis=0, seed_xy=seed_xy, slice_z=slice_z
    )
    y_lo, y_hi = _enclosed_axis_bounds(
        walls, axis=1, seed_xy=seed_xy, slice_z=slice_z
    )
    lo = np.array([x_lo, y_lo, floor_top], dtype=float)
    hi = np.array([x_hi, y_hi, wall_top], dtype=float)
    if np.any(lo >= hi):
        raise ValueError(
            f"box_pack: invalid collision-derived cavity bounds {lo}, {hi}"
        )
    return np.stack([lo, hi]), floor_geom_ids


class BoxPackCriterion:
    """MuJoCo-backed "right object, right hand, right side of the box" check."""

    name = "object_in_box"

    def __init__(self, *, mujoco, model, data, options=None):
        self.mujoco = mujoco
        self.m = model
        self.d = data
        self.options = dict(options or {})

        target = str(self.options.get("box_pack_object", "tape")).strip().lower()
        # robosuite_env.tasks.box_pack is the contract and speaks "R"/"L" and
        # "in"/"out"; spell both out here so the option can be written either
        # way and describe() still reads as English.
        hand = _HANDS.get(
            str(self.options.get("box_pack_hand", "right")).strip().lower()
        )
        direction = _DIRECTIONS.get(
            str(self.options.get("box_pack_direction", "in")).strip().lower()
        )
        if hand is None:
            raise ValueError(
                "box_pack hand must be one of "
                f"{sorted(_HANDS)}; got {self.options.get('box_pack_hand')!r}"
            )
        if direction is None:
            raise ValueError(
                "box_pack direction must be one of "
                f"{sorted(_DIRECTIONS)}; got {self.options.get('box_pack_direction')!r}"
            )
        self.target = target
        self.hand = hand
        self.direction = direction

        self.table_z_m = as_float(self.options, "table_z_m", 0.0)
        # Clear of the surface: below this the object may simply be sliding.
        self.grip_lift_m = as_float(self.options, "box_pack_grip_lift_m", 0.02)
        self.margin_m = as_float(self.options, "box_pack_margin_m", 0.005)
        # The box has left the table once its base drops this far below it.
        self.box_fall_m = as_float(self.options, "box_pack_fall_m", 0.05)

        self.object_body = self._body_id(target)
        self.box_body = self._body_id("box")
        floor_cut_m = as_float(self.options, "box_pack_floor_cut_m", 0.05)
        (
            self.box_cavity_local_bounds_m,
            self.box_floor_geoms,
        ) = _derive_box_cavity(
            self.m,
            self.box_body,
            floor_cut_m=floor_cut_m,
        )
        self.hand_geoms = gripper_geoms_by_side(self.m, owner="box_pack")
        self.object_geoms = self._body_geoms(self.object_body)
        self._gripped_by = {"left": False, "right": False}
        self._box_fell = False
        # The packing criterion: a packing run succeeds at the first output
        # frame where the object is inside the cavity and no hand holds it,
        # and later frames are not looked at.
        self._rule_grasp_welds = self._rule_grasp_weld_ids()
        self._output_frames = 0
        self._released_in_box_frame = None
        # Read where the box starts here, before the first step, rather than on
        # the first evaluate(): that is the first output frame, a tenth of the
        # control rate in, and a box already settling or already nudged would
        # have become its own reference.
        self._box_start_z_m = float(self.d.xpos[self.box_body][2])

    # -- model lookups ------------------------------------------------------

    def _body_id(self, token):
        return body_id_by_token(self.m, token, owner="box_pack")

    def _touching_box_floor(self):
        for index in range(int(self.d.ncon)):
            contact = self.d.contact[index]
            a, b = int(contact.geom1), int(contact.geom2)
            if (a in self.object_geoms and b in self.box_floor_geoms) or (
                b in self.object_geoms and a in self.box_floor_geoms
            ):
                return True
        return False

    def _rule_grasp_weld_ids(self):
        ids = []
        for name in RULE_GRASP_WELD_NAMES.values():
            eq_id = int(
                self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_EQUALITY, name)
            ) if self.mujoco is not None else -1
            if eq_id >= 0:
                ids.append(eq_id)
        return tuple(ids)

    def _held(self):
        """Is a hand holding the object at this instant?

        With the rule grasp, holding is a weld: either rule-grasp weld active
        with the object as one of its bodies -- the same signal Flat recorded
        as object_rule_grasp_inactive. A model without those welds (physics
        grasp) has nothing to latch, so a finger touching the object counts.
        """

        if self._rule_grasp_welds:
            for eq_id in self._rule_grasp_welds:
                if not bool(self.d.eq_active[eq_id]):
                    continue
                if self.object_body in (
                    int(self.m.eq_obj1id[eq_id]),
                    int(self.m.eq_obj2id[eq_id]),
                ):
                    return True
            return False
        return self._touching_hand("left") or self._touching_hand("right")

    def _body_geoms(self, body_id):
        return {
            index
            for index in range(self.m.ngeom)
            if self.m.geom_bodyid[index] == body_id
        }

    def tick(self, *, sample_id=None):
        """Sample the latched box-departure test once per control step.

        The box must not leave the table at any point,
        and evaluate() only runs on output frames -- a tenth of the control
        rate -- so a box knocked off and bouncing back would go unseen.
        """

        self._note_box_height(float(self.d.xpos[self.box_body][2]))
        return None

    def _note_box_height(self, height_m):
        if abs(height_m - self._box_start_z_m) > self.box_fall_m:
            self._box_fell = True

    # -- evaluation ---------------------------------------------------------
    def describe(self):
        """The prompt binding and thresholds this run was scored against."""

        cavity_lo, cavity_hi = self.box_cavity_local_bounds_m
        return {
            "name": self.name,
            "object": self.target,
            "hand": self.hand,
            "direction": self.direction,
            "table_z_m": self.table_z_m,
            "box_cavity_local_bounds_m": {
                "min": [round(float(v), 6) for v in cavity_lo],
                "max": [round(float(v), 6) for v in cavity_hi],
            },
            "grip_lift_m": self.grip_lift_m,
            "margin_m": self.margin_m,
            "box_fall_m": self.box_fall_m,
            "success_rule": (
                "first_release_in_box_e"
                if self.direction == "into_box"
                else "placed_and_box_on_table"
            ),
            "release_signal": (
                "rule_grasp_weld" if self._rule_grasp_welds else "finger_contact"
            ),
        }

    def _touching_hand(self, side):
        geoms = self.hand_geoms[side]
        for index in range(self.d.ncon):
            contact = self.d.contact[index]
            a, b = int(contact.geom1), int(contact.geom2)
            if (a in self.object_geoms and b in geoms) or (
                b in self.object_geoms and a in geoms
            ):
                return True
        return False

    def evaluate(self):
        object_pos = np.asarray(self.d.xpos[self.object_body], dtype=float)
        box_pos = np.asarray(self.d.xpos[self.box_body], dtype=float)
        box_mat = np.asarray(self.d.xmat[self.box_body], dtype=float).reshape(3, 3)

        # The box leaving the table ends the episode's chances, so latch it.
        # Measured as displacement from where the box started, which catches it
        # being lifted away as well as knocked off.
        self._note_box_height(float(box_pos[2]))
        box_on_table = not self._box_fell

        lifted = float(object_pos[2] - self.table_z_m) >= self.grip_lift_m
        for side in ("left", "right"):
            if lifted and self._touching_hand(side):
                self._gripped_by[side] = True

        local = box_mat.T @ (object_pos - box_pos)
        cavity_lo, cavity_hi = self.box_cavity_local_bounds_m
        inside_plan = bool(
            np.all(local[:2] >= cavity_lo[:2] + self.margin_m)
            and np.all(local[:2] <= cavity_hi[:2] - self.margin_m)
        )
        inside_height = bool(cavity_lo[2] <= local[2] <= cavity_hi[2])
        in_cavity = inside_plan and inside_height
        on_table = (
            not inside_plan
            and abs(float(object_pos[2] - self.table_z_m)) <= self.grip_lift_m
        )

        # Packed means the object is in the cavity
        # and the box never left the table. Which hand did it and whether it is
        # touching the floor are recorded, and both gate strict_success, but
        # neither decides whether the object got packed.
        # Touching a floor hull is not enough on its own: the floor slab has an
        # underside and an outer rim, so an object shoved beneath or alongside
        # the box registers the same contact. Require it to be over the box's
        # cavity as well, which is what "in the box, on its floor" means.
        on_box_floor = bool(self._touching_box_floor() and inside_plan)
        placed = in_cavity if self.direction == "into_box" else on_table
        hand_ok = self._gripped_by[self.hand]
        wrong_hand = self._gripped_by["left" if self.hand == "right" else "right"]
        contained_and_box_on_table = bool(placed and box_on_table)
        held = self._held()
        self._output_frames += 1
        if (
            self.direction == "into_box"
            and self._released_in_box_frame is None
            and in_cavity
            and not held
        ):
            self._released_in_box_frame = self._output_frames - 1
        if self.direction == "into_box":
            success = self._released_in_box_frame is not None
        else:
            success = contained_and_box_on_table
        strict_placement = on_box_floor if self.direction == "into_box" else placed

        return {
            "task": self.name,
            "success": success,
            # Strict additionally requires floor contact for packing, and asks
            # that the named hand did the work without the other taking over.
            "strict_success": bool(
                success and strict_placement and hand_ok and not wrong_hand
            ),
            "conditions": {
                "object_placed": placed,
                "object_in_box": in_cavity,
                "object_on_box_floor": on_box_floor,
                "object_on_table": on_table,
                "moved_by_named_hand": hand_ok,
                "moved_by_other_hand": wrong_hand,
                "box_on_table": box_on_table,
                "object_held": held,
                "first_release_in_box": self._released_in_box_frame is not None,
                # The verdict before criterion (e): in the cavity now, and the
                # box never left the table.
                "contained_and_box_on_table": contained_and_box_on_table,
            },
            "metrics": {
                "object": self.target,
                "hand": self.hand,
                "direction": self.direction,
                "first_release_in_box_output_frame": self._released_in_box_frame,
                "in_box_frame_mm": [round(float(v) * 1000.0, 2) for v in local],
                "box_cavity_local_bounds_mm": {
                    "min": [round(float(v) * 1000.0, 2) for v in cavity_lo],
                    "max": [round(float(v) * 1000.0, 2) for v in cavity_hi],
                },
                "object_height_mm": round(
                    float(object_pos[2] - self.table_z_m) * 1000.0, 2
                ),
                "box_height_mm": round(float(box_pos[2] - self.table_z_m) * 1000.0, 2),
                # Where the box is and which way it faces, in the world. The
                # object's own coordinates are reported in the box's frame
                # above, so without these a pushed or turned box could only be
                # reconstructed from the object -- and that inversion fails the
                # moment the box tips, which is exactly the moment worth seeing.
                "box_world_mm": [round(float(v) * 1000.0, 2) for v in box_pos],
                "box_yaw_deg": round(
                    math.degrees(math.atan2(box_mat[1, 0], box_mat[0, 0])), 3
                ),
                # Angle between the box's up axis and the world's: 0 while it
                # stands, rising as it tips over an edge.
                "box_tilt_deg": round(
                    math.degrees(math.acos(max(-1.0, min(1.0, float(box_mat[2, 2]))))),
                    3,
                ),
            },
            "score": {
                "condition": "object_in_box_containment",
                "points": int(placed) + int(hand_ok) + int(box_on_table),
                "max_points": 3,
                "enabled": True,
            },
        }
