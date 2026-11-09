# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Geometric success criterion for the cup-on-plate task."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._geometry import as_float, round_list


def _quat_to_mat(q_wxyz):
    w, x, y, z = [float(v) for v in q_wxyz]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _circle_margin_mm(center_offset, moving_radius_m, target_radius_m, slack_m=0.0):
    """Positive margin means the moving footprint is fully inside the target footprint."""
    radial = float(np.linalg.norm(np.asarray(center_offset, dtype=float)))
    return (float(target_radius_m) + float(slack_m) - radial - float(moving_radius_m)) * 1000.0


@dataclass(frozen=True)
class CupOnPlateSpec:
    cup_joint_names: tuple[str, ...] = ("cube_joint0", "cup_joint", "cup_joint0")
    plate_joint_names: tuple[str, ...] = ("plate_joint0", "plate_joint")
    cup_body_names: tuple[str, ...] = ("cube_main", "cup", "cup_main")
    plate_body_names: tuple[str, ...] = ("plate_main", "plate")
    cup_name_tokens: tuple[str, ...] = ("cube", "cup")
    plate_name_tokens: tuple[str, ...] = ("plate",)
    cup_bottom_local: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cup_bottom_normal_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    cup_bottom_radius_m: float = 0.0355
    plate_bottom_local: tuple[float, float, float] = (0.0, 0.0, 0.0)
    plate_support_local: tuple[float, float, float] = (0.0, 0.0, 0.018912)
    plate_normal_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    plate_bottom_radius_m: float = 0.089371


class CupOnPlateCriterion:
    """Fast MuJoCo-backed cup-on-plate success check.

    Per-frame work is a contact scan plus constant-size pose math. Object ids,
    contact geom sets, radii, and local support points are cached at init.
    """

    name = "cup_on_plate"

    def __init__(self, *, mujoco, model, data, options=None, spec=None):
        self.mujoco = mujoco
        self.m = model
        self.d = data
        self.options = dict(options or {})
        self.spec = spec or CupOnPlateSpec()
        self.contact_tolerance_m = as_float(self.options, "contact_tolerance_m", 0.003)
        self.contact_penetration_tolerance_m = as_float(self.options, "contact_penetration_tolerance_m", 0.005)
        self.place_contact_penetration_tolerance_m = as_float(
            self.options,
            "place_contact_penetration_tolerance_m",
            max(self.contact_penetration_tolerance_m, 0.015),
        )
        self.plane_gap_tolerance_m = as_float(self.options, "plane_gap_tolerance_m", 0.010)
        self.plane_penetration_tolerance_m = as_float(
            self.options, "plane_penetration_tolerance_m", min(self.contact_tolerance_m, 0.002)
        )
        self.bottom_contact_tolerance_m = as_float(self.options, "bottom_contact_tolerance_m", 0.012)
        self.footprint_slack_m = as_float(self.options, "footprint_slack_m", 0.003)
        self.score_condition = str(self.options.get("score_condition", "cup_on_plate_contact_points")).strip().lower()
        if self.score_condition == "auto":
            self.score_condition = "cup_on_plate_contact_points"
        if self.score_condition not in ("", "none", "off", "false", "0", "no", "cup_on_plate_contact_points"):
            raise ValueError(f"unknown cup_on_plate score condition: {self.score_condition}")
        upright_deg = as_float(self.options, "upright_tolerance_deg", 35.0)
        self.min_normal_dot = math.cos(math.radians(max(0.0, min(180.0, upright_deg))))
        self.geometry_mode = str(self.options.get("geometry_mode", "legacy")).strip().lower() or "legacy"
        # Which point decides "on top of the plate, not under it":
        #   "middle" (default) -- the cup's middle along its own axis must be
        #     above the plate floor. Every Duo cupplate published number was scored
        #     this way.
        #   "lowest" -- the cup's lowest collision point must be, measured from
        #     the centre of its collision box. Stricter: it also catches a cup
        #     half sunk through the plate. The Flat lineage scored this way.
        self.above_plate_test = (
            str(self.options.get("cup_above_plate_test", "middle")).strip().lower()
            or "middle"
        )
        if self.above_plate_test not in {"middle", "lowest"}:
            raise ValueError(
                "cup_on_plate cup_above_plate_test must be 'middle' or 'lowest', "
                f"got {self.above_plate_test!r}"
            )
        if self.geometry_mode not in ("legacy", "bbox"):
            raise ValueError(f"unknown cup_on_plate geometry mode: {self.geometry_mode}")

        self.cup_body = self._resolve_body(self.spec.cup_joint_names, self.spec.cup_body_names, self.spec.cup_name_tokens)
        self.plate_body = self._resolve_body(self.spec.plate_joint_names, self.spec.plate_body_names, self.spec.plate_name_tokens)
        self.cup_geoms = self._collect_geoms(self.cup_body, self.spec.cup_name_tokens)
        self.plate_geoms = self._collect_geoms(self.plate_body, self.spec.plate_name_tokens)
        if self.cup_body < 0 or self.plate_body < 0:
            raise ValueError("could not resolve cup/plate bodies for cup_on_plate success criterion")
        if not self.cup_geoms or not self.plate_geoms:
            raise ValueError("could not resolve cup/plate collision geoms for cup_on_plate success criterion")

        self._cup_bottom_local = np.asarray(self.spec.cup_bottom_local, dtype=float)
        self._plate_bottom_local = np.asarray(self.spec.plate_bottom_local, dtype=float)
        self._plate_support_local = np.asarray(self.spec.plate_support_local, dtype=float)
        self._cup_normal_local = self._unit(self.spec.cup_bottom_normal_local)
        self._plate_normal_local = self._unit(self.spec.plate_normal_local)
        self._plate_bbox = self._compute_plate_bbox() if self.geometry_mode == "bbox" else None
        (
            self._cup_center_local,
            self._cup_half_extents_local,
            self._cup_points_local,
        ) = self._cup_extents_local()
        self._cup_axis_mid_m = self._cup_axis_midpoint_local()

    def _compute_plate_bbox(self):
        """Plate extents in the plate body frame, from collision geometry at init.

        Mesh geoms use their (convex-hull) vertices so a round plate gets its true
        disc radius; non-mesh geoms fall back to the 8 corners of geom_aabb, which
        overestimates the radial reach of a wide box-like geom by up to sqrt(2).
        """
        gxpos = getattr(self.d, "geom_xpos", None)
        gxmat = getattr(self.d, "geom_xmat", None)
        if gxpos is None or gxmat is None:
            raise ValueError("cup_on_plate geometry mode 'bbox' needs data.geom_xpos/geom_xmat")
        plate_pos, plate_R = self._body_pose(self.plate_body)
        radial = 0.0
        z_min, z_max = float("inf"), float("-inf")
        for gid in self.plate_geoms:
            pts = self._geom_points(gid)
            if pts is None or not len(pts):
                continue
            R = np.asarray(gxmat[gid], dtype=float).reshape(3, 3)
            world = np.asarray(gxpos[gid], dtype=float) + pts @ R.T
            local = (world - plate_pos) @ plate_R
            radial = max(radial, float(np.hypot(local[:, 0], local[:, 1]).max()))
            z_min = min(z_min, float(local[:, 2].min()))
            z_max = max(z_max, float(local[:, 2].max()))
        if not np.isfinite(z_min) or radial <= 0.0:
            raise ValueError("cup_on_plate geometry mode 'bbox' could not measure plate collision geometry")
        return {"radial_m": radial, "z_min_m": z_min, "z_max_m": z_max}

    def _cup_axis_midpoint_local(self):
        """Half-height of the cup along its own axis, from collision geometry.

        The cup body origin sits at its base, so the origin is a poor stand-in
        for the cup's centre once the cup is on its side. Offsetting by this
        much along the cup axis gives a point that stays near the middle of
        the cup in any orientation.
        """
        gxpos = getattr(self.d, "geom_xpos", None)
        gxmat = getattr(self.d, "geom_xmat", None)
        if gxpos is None or gxmat is None:
            return float(self.spec.cup_bottom_radius_m)
        cup_pos, cup_R = self._body_pose(self.cup_body)
        z_min, z_max = float("inf"), float("-inf")
        for gid in self.cup_geoms:
            pts = self._geom_points(gid)
            if pts is None or not len(pts):
                continue
            R = np.asarray(gxmat[gid], dtype=float).reshape(3, 3)
            world = np.asarray(gxpos[gid], dtype=float) + pts @ R.T
            local = (world - cup_pos) @ cup_R
            z_min = min(z_min, float(local[:, 2].min()))
            z_max = max(z_max, float(local[:, 2].max()))
        if not np.isfinite(z_min):
            return float(self.spec.cup_bottom_radius_m)
        return 0.5 * (z_min + z_max)

    def _cup_extents_local(self):
        """The cup's collision shape in its own frame.

        Returns the centre of its bounding box, the half-extents of that box,
        and the collision points themselves. The points are what the floor test
        measures: the box's diagonal reaches 15 mm below the real cup, enough
        to call a cup lying on the plate sunk through it, and which way the cup
        happens to have rolled should not decide that. The box is kept for the
        centre and as the fallback when there is no geometry to measure.

        The cup body origin sits at its base, so the origin is a poor stand-in
        for the cup once it is on its side. The centre of this box tracks the
        middle of the cup in any orientation, and the half-extents give the
        cup's reach along any direction, which is what says how far below the
        plate the cup actually descends.

        Measured once: the cup's collision geoms are rigid children of the cup
        body, so the box is the same in the body frame at every pose. An
        articulated cup would need this per frame.
        """
        radius = float(self.spec.cup_bottom_radius_m)
        fallback = (
            np.asarray([0.0, 0.0, radius], dtype=float),
            np.asarray([radius, radius, radius], dtype=float),
            None,
        )
        gxpos = getattr(self.d, "geom_xpos", None)
        gxmat = getattr(self.d, "geom_xmat", None)
        if gxpos is None or gxmat is None:
            return fallback
        cup_pos, cup_R = self._body_pose(self.cup_body)
        collected = []
        for gid in self.cup_geoms:
            pts = self._geom_points(gid)
            if pts is None or not len(pts):
                continue
            R = np.asarray(gxmat[gid], dtype=float).reshape(3, 3)
            world = np.asarray(gxpos[gid], dtype=float) + pts @ R.T
            collected.append((world - cup_pos) @ cup_R)
        if not collected:
            return fallback
        local = np.unique(np.round(np.concatenate(collected), 9), axis=0)
        lo = local.min(axis=0)
        hi = local.max(axis=0)
        return 0.5 * (lo + hi), 0.5 * (hi - lo), local

    def _geom_points(self, gid):
        """Geom shape sample points in the geom frame: mesh vertices, else AABB corners."""
        dataid = getattr(self.m, "geom_dataid", None)
        mesh_vert = getattr(self.m, "mesh_vert", None)
        if dataid is not None and mesh_vert is not None and int(dataid[gid]) >= 0:
            mid = int(dataid[gid])
            adr = int(self.m.mesh_vertadr[mid])
            num = int(self.m.mesh_vertnum[mid])
            if num > 0:
                return np.asarray(mesh_vert[adr:adr + num], dtype=float).reshape(-1, 3)
        aabb = getattr(self.m, "geom_aabb", None)
        if aabb is None:
            return None
        center = np.asarray(aabb[gid][:3], dtype=float)
        half = np.asarray(aabb[gid][3:6], dtype=float)
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
        return center + signs * half

    def describe(self):
        out = {
            "name": self.name,
            "contact_tolerance_mm": round(self.contact_tolerance_m * 1000.0, 3),
            "contact_penetration_tolerance_mm": round(self.contact_penetration_tolerance_m * 1000.0, 3),
            "place_contact_penetration_tolerance_mm": round(self.place_contact_penetration_tolerance_m * 1000.0, 3),
            "plane_gap_tolerance_mm": round(self.plane_gap_tolerance_m * 1000.0, 3),
            "plane_penetration_tolerance_mm": round(self.plane_penetration_tolerance_m * 1000.0, 3),
            "bottom_contact_tolerance_mm": round(self.bottom_contact_tolerance_m * 1000.0, 3),
            "footprint_slack_mm": round(self.footprint_slack_m * 1000.0, 3),
            "upright_tolerance_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, self.min_normal_dot)))), 3),
            "score_condition": self.score_condition,
            "cup_body": self._body_name(self.cup_body),
            "plate_body": self._body_name(self.plate_body),
            "cup_collision_geoms": len(self.cup_geoms),
            "plate_collision_geoms": len(self.plate_geoms),
            "cup_bottom_radius_m": float(self.spec.cup_bottom_radius_m),
            "plate_bottom_radius_m": float(self.spec.plate_bottom_radius_m),
            "plate_support_local": round_list(self._plate_support_local),
        }
        if self.geometry_mode != "legacy":
            out["geometry_mode"] = self.geometry_mode
            out["plate_bbox"] = {
                "radial_mm": round(self._plate_bbox["radial_m"] * 1000.0, 3),
                "z_min_mm": round(self._plate_bbox["z_min_m"] * 1000.0, 3),
                "z_max_mm": round(self._plate_bbox["z_max_m"] * 1000.0, 3),
            }
        return out

    def evaluate(self):
        contacts = self._cup_plate_contacts()
        geom_contact = bool(contacts)
        contact_dist_m = min((c["dist"] for c in contacts), default=float("inf"))

        cup_pos, cup_R = self._body_pose(self.cup_body)
        plate_pos, plate_R = self._body_pose(self.plate_body)

        cup_bottom = cup_pos + cup_R @ self._cup_bottom_local
        plate_bottom = plate_pos + plate_R @ self._plate_bottom_local
        plate_support = plate_pos + plate_R @ self._plate_support_local
        plate_n = self._unit(plate_R @ self._plate_normal_local)
        cup_n = self._unit(cup_R @ self._cup_normal_local)

        to_cup = cup_bottom - plate_bottom
        height_from_plate_bottom_m = float(np.dot(to_cup, plate_n))
        projected = to_cup - height_from_plate_bottom_m * plate_n
        center_offset_m = float(np.linalg.norm(projected))
        inside_margin_mm = _circle_margin_mm(
            projected,
            self.spec.cup_bottom_radius_m,
            self.spec.plate_bottom_radius_m,
            self.footprint_slack_m,
        )

        support_gap_m = float(np.dot(cup_bottom - plate_support, plate_n))
        normal_dot = float(np.dot(cup_n, plate_n))
        strict_bottom_touch, strict_side_touch, strict_touch_metrics = (
            self._classify_touches(
                contacts,
                cup_bottom,
                cup_n,
                penetration_tolerance_m=self.contact_penetration_tolerance_m,
            )
        )
        bottom_touch, side_touch, touch_metrics = self._classify_touches(
            contacts,
            cup_bottom,
            cup_n,
            penetration_tolerance_m=self.place_contact_penetration_tolerance_m,
        )
        contact_ok = bool(
            geom_contact
            and contact_dist_m >= -self.contact_penetration_tolerance_m
            and contact_dist_m <= self.contact_tolerance_m
        )
        place_contact_ok = bool(
            geom_contact
            and contact_dist_m >= -self.place_contact_penetration_tolerance_m
            and contact_dist_m <= self.contact_tolerance_m
        )
        support_gap_ok = bool(
            support_gap_m >= -self.plane_penetration_tolerance_m
            and support_gap_m <= self.plane_gap_tolerance_m
        )
        vertical_span_ok = bool(
            height_from_plate_bottom_m >= -self.plane_penetration_tolerance_m
            and height_from_plate_bottom_m <= (float(self._plate_support_local[2]) + self.plane_gap_tolerance_m)
        )
        inside_ok = bool(inside_margin_mm >= -1e-9)
        normal_ok = bool(normal_dot >= self.min_normal_dot)

        # A cup on its side is still on the plate. The seated tests above all
        # measure the cup's bottom
        # disc, which stops describing the cup once it tips, so success rests
        # on two orientation-independent facts instead: the cup touches the
        # plate, and its middle is over the plate and not under it.
        if self.above_plate_test == "middle":
            cup_middle = cup_pos + cup_R @ np.asarray([0.0, 0.0, self._cup_axis_mid_m])
        else:
            cup_middle = cup_pos + cup_R @ self._cup_center_local
        middle_offset = cup_middle - plate_bottom
        middle_planar = middle_offset - float(np.dot(middle_offset, plate_n)) * plate_n
        middle_offset_m = float(np.linalg.norm(middle_planar))
        plate_reach_m = (
            float(self._plate_bbox["radial_m"]) if self._plate_bbox
            else float(self.spec.plate_bottom_radius_m)
        )
        over_plate = bool(middle_offset_m <= plate_reach_m + self.footprint_slack_m)
        middle_height_m = float(np.dot(middle_offset, plate_n))
        plate_floor_m = (
            float(self._plate_bbox["z_min_m"]) if self._plate_bbox
            else float(self._plate_bottom_local[2])
        )
        # On top of the plate, not under it: a cup that has sunk through the
        # plate also touches it and also sits over it. The test is the cup's
        # lowest point, not its middle -- half a cup is 38 mm, so a middle
        # above the plate says nothing about where the base is.
        plate_n_in_cup = cup_R.T @ plate_n
        cup_base_height_m = float(np.dot(cup_pos - plate_bottom, plate_n))
        if self._cup_points_local is not None:
            # The collision points themselves, so a cup rolled onto its side is
            # measured where it actually touches down rather than at the corner
            # of a box drawn around it.
            cup_lowest_m = cup_base_height_m + float(
                (self._cup_points_local @ plate_n_in_cup).min()
            )
        else:
            cup_lowest_m = middle_height_m - float(
                np.dot(np.abs(plate_n_in_cup), self._cup_half_extents_local)
            )
        if self.above_plate_test == "middle":
            above_plate = bool(
                middle_height_m >= plate_floor_m - self.plane_penetration_tolerance_m
            )
        else:
            above_plate = bool(
                cup_lowest_m >= plate_floor_m - self.plane_penetration_tolerance_m
            )
        resting_on_plate = bool(place_contact_ok and over_plate and above_plate)

        # The seated criterion is kept as strict_success, with the two-tier
        # contact tolerances this rig added: it is the stricter seated
        # criterion, and the only place the bottom-disc geometry decides anything.
        strict_success = bool(
            contact_ok
            and strict_bottom_touch
            and support_gap_ok
            and inside_ok
            and normal_ok
        )
        success = resting_on_plate
        score = self._score(
            resting_on_plate=resting_on_plate,
            inside_ok=inside_ok,
            bottom_touch=bottom_touch,
            side_touch=side_touch,
        )

        bbox_extra = None
        if self.geometry_mode == "bbox":
            bb = self._plate_bbox
            disc_ok = bool(center_offset_m <= bb["radial_m"] + self.footprint_slack_m)
            z_lo = bb["z_min_m"] - self.plane_penetration_tolerance_m
            z_full_ok = bool(z_lo <= height_from_plate_bottom_m <= bb["z_max_m"] + self.plane_gap_tolerance_m)
            z_partial_hi = bb["z_max_m"] + float(self.spec.cup_bottom_radius_m) + self.plane_gap_tolerance_m
            z_partial_ok = bool(z_lo <= height_from_plate_bottom_m <= z_partial_hi)
            strict_success = bool(
                contact_ok
                and strict_bottom_touch
                and normal_ok
                and disc_ok
                and z_full_ok
            )
            success = resting_on_plate
            partial = bool(place_contact_ok and disc_ok and z_partial_ok and not success)
            bbox_extra = {
                "partial": partial,
                "conditions": {
                    "cup_center_inside_plate_disc": disc_ok,
                    "cup_bottom_in_plate_zrange": z_full_ok,
                },
                "metrics": {
                    "plate_disc_radius_mm": round(bb["radial_m"] * 1000.0, 3),
                    "plate_z_range_mm": [round(bb["z_min_m"] * 1000.0, 3), round(bb["z_max_m"] * 1000.0, 3)],
                },
            }

        result = {
            "task": self.name,
            "success": success,
            "strict_success": strict_success,
            "conditions": {
                "cup_plate_contact": contact_ok,
                "cup_plate_place_contact": place_contact_ok,
                "cup_bottom_contact": bottom_touch,
                "cup_side_contact": side_touch,
                "cup_bottom_contact_strict": strict_bottom_touch,
                "cup_side_contact_strict": strict_side_touch,
                "cup_bottom_on_plate_support": support_gap_ok,
                "cup_bottom_within_plate_vertical_span": vertical_span_ok,
                "cup_bottom_inside_plate_bottom_prism": inside_ok,
                "cup_resting_on_plate": resting_on_plate,
                "cup_middle_over_plate": over_plate,
                "cup_above_plate": above_plate,
                "cup_axis_aligned_with_plate_normal": normal_ok,
            },
            "metrics": {
                "contact_dist_mm": round(float(contact_dist_m) * 1000.0, 3) if geom_contact else None,
                "bottom_contact_count": int(touch_metrics["bottom_contact_count"]),
                "side_contact_count": int(touch_metrics["side_contact_count"]),
                "min_bottom_contact_height_mm": touch_metrics["min_bottom_contact_height_mm"],
                "min_side_contact_height_mm": touch_metrics["min_side_contact_height_mm"],
                "strict_bottom_contact_count": int(
                    strict_touch_metrics["bottom_contact_count"]
                ),
                "strict_side_contact_count": int(
                    strict_touch_metrics["side_contact_count"]
                ),
                "support_gap_mm": round(support_gap_m * 1000.0, 3),
                "height_from_plate_bottom_mm": round(height_from_plate_bottom_m * 1000.0, 3),
                "bottom_center_offset_mm": round(center_offset_m * 1000.0, 3),
                "middle_center_offset_mm": round(middle_offset_m * 1000.0, 3),
                "middle_height_above_plate_mm": round(middle_height_m * 1000.0, 3),
                # Heights are measured from the plate body origin, as the
                # other heights here are; the plate's floor is plate_floor_m
                # below that, not zero.
                "cup_lowest_above_plate_origin_mm": round(cup_lowest_m * 1000.0, 3),
                "inside_margin_mm": round(float(inside_margin_mm), 3),
                "cup_axis_plate_normal_dot": round(normal_dot, 5),
            },
            "poses": {
                "cup_bottom": round_list(cup_bottom, 5),
                "plate_bottom": round_list(plate_bottom, 5),
                "plate_support": round_list(plate_support, 5),
            },
            "score": score,
        }
        if bbox_extra is not None:
            result["partial_success"] = bbox_extra["partial"]
            result["conditions"].update(bbox_extra["conditions"])
            result["metrics"].update(bbox_extra["metrics"])
        return result

    def _resolve_body(self, joint_names, body_names, tokens):
        for name in joint_names:
            jid = self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                return int(self.m.jnt_bodyid[jid])
        for name in body_names:
            bid = self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                return int(bid)
        for bid in range(self.m.nbody):
            name = self._body_name(bid).lower()
            if any(tok in name for tok in tokens):
                return int(bid)
        return -1

    def _collect_geoms(self, body_id, tokens):
        body_ids = self._body_subtree(body_id)
        out = set()
        for gid in range(self.m.ngeom):
            if not (int(self.m.geom_contype[gid]) or int(self.m.geom_conaffinity[gid])):
                continue
            if int(self.m.geom_bodyid[gid]) in body_ids:
                out.add(int(gid))
        if out:
            return frozenset(out)
        for gid in range(self.m.ngeom):
            if not (int(self.m.geom_contype[gid]) or int(self.m.geom_conaffinity[gid])):
                continue
            gname = self._geom_name(gid).lower()
            bname = self._body_name(int(self.m.geom_bodyid[gid])).lower()
            if any(tok in gname or tok in bname for tok in tokens):
                out.add(int(gid))
        return frozenset(out)

    def _body_subtree(self, body_id):
        body_id = int(body_id)
        if body_id < 0:
            return frozenset()
        out = {body_id}
        parent = getattr(self.m, "body_parentid", None)
        if parent is None:
            return frozenset(out)
        changed = True
        while changed:
            changed = False
            for bid in range(self.m.nbody):
                if bid not in out and int(parent[bid]) in out:
                    out.add(int(bid))
                    changed = True
        return frozenset(out)

    def _cup_plate_contacts(self):
        out = []
        for ci in range(self.d.ncon):
            c = self.d.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self.cup_geoms and g2 in self.plate_geoms) or (g2 in self.cup_geoms and g1 in self.plate_geoms):
                cup_geom = g1 if g1 in self.cup_geoms else g2
                plate_geom = g2 if cup_geom == g1 else g1
                pos = np.asarray(getattr(c, "pos", np.zeros(3)), dtype=float).copy()
                out.append({"dist": float(c.dist), "pos": pos, "cup_geom": cup_geom, "plate_geom": plate_geom})
        return out

    def _classify_touches(
        self, contacts, cup_bottom, cup_axis, *, penetration_tolerance_m
    ):
        bottom_heights = []
        side_heights = []
        radius = float(self.spec.cup_bottom_radius_m) + float(self.footprint_slack_m)
        for c in contacts:
            if (
                c["dist"] < -float(penetration_tolerance_m)
                or c["dist"] > self.contact_tolerance_m
            ):
                continue
            rel = np.asarray(c["pos"], dtype=float) - cup_bottom
            h = float(np.dot(rel, cup_axis))
            radial = float(np.linalg.norm(rel - h * cup_axis))
            if abs(h) <= self.bottom_contact_tolerance_m and radial <= radius:
                bottom_heights.append(abs(h) * 1000.0)
            else:
                side_heights.append(h * 1000.0)
        bottom_touch = bool(bottom_heights)
        side_touch = bool(side_heights)
        metrics = {
            "bottom_contact_count": len(bottom_heights),
            "side_contact_count": len(side_heights),
            "min_bottom_contact_height_mm": round(min(bottom_heights), 3) if bottom_heights else None,
            "min_side_contact_height_mm": round(min(side_heights, key=abs), 3) if side_heights else None,
        }
        return bottom_touch, side_touch, metrics

    def _score(self, *, resting_on_plate, inside_ok, bottom_touch, side_touch):
        """Points for the same geometry success uses.

        Scoring and success both start from resting_on_plate, so a cup resting
        on the plate rim scores as well as succeeds; the bottom/side split only
        says how the cup rests: seated, or on its side.
        """
        if self.score_condition in ("", "none", "off", "false", "0", "no"):
            return {"condition": self.score_condition, "points": 0, "max_points": 0, "enabled": False}
        points = 0
        reason = "no_score"
        if resting_on_plate and bottom_touch:
            points = 2
            reason = "resting_on_plate_and_bottom_touch"
        elif resting_on_plate and side_touch:
            points = 1
            reason = "resting_on_plate_and_side_touch"
        elif not resting_on_plate:
            reason = "not_resting_on_plate"
        elif not (bottom_touch or side_touch):
            reason = "no_cup_plate_touch"
        return {
            "condition": self.score_condition,
            "enabled": True,
            "points": int(points),
            "max_points": 2,
            "reason": reason,
            "resting_on_plate": bool(resting_on_plate),
            "bottom_inside_prism": bool(inside_ok),
            "bottom_touch": bool(bottom_touch),
            "side_touch": bool(side_touch),
        }

    def _body_pose(self, body_id):
        pos = np.asarray(self.d.xpos[body_id], dtype=float).copy()
        R = np.asarray(self.d.xmat[body_id], dtype=float).reshape(3, 3).copy()
        return pos, R

    def _body_name(self, bid):
        return self.mujoco.mj_id2name(self.m, self.mujoco.mjtObj.mjOBJ_BODY, int(bid)) or f"body{bid}"

    def _geom_name(self, gid):
        return self.mujoco.mj_id2name(self.m, self.mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"geom{gid}"

    @staticmethod
    def _unit(v):
        arr = np.asarray(v, dtype=float)
        n = float(np.linalg.norm(arr))
        if n <= 1e-12:
            return arr
        return arr / n


__all__ = ["CupOnPlateCriterion", "CupOnPlateSpec", "_circle_margin_mm", "_quat_to_mat"]
