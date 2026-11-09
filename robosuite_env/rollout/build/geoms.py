# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Compiled geometry classification and contact inventory helpers."""

from __future__ import annotations

import functools

import mujoco
import numpy as np

from robosuite_env.rollout.contacts import collect_arm_penetrations
from robosuite_env.rollout.control.rule_grasp import _has_free_joint_ancestor, gname


def initialize_geometry_sets(state) -> None:
    """Populate the compiled arm, gripper, finger, and cup geom registries."""

    state.ARM_GEOMS = {
        key: {
            geom_id
            for geom_id in range(state.m.ngeom)
            if (
                mujoco.mj_id2name(
                    state.m,
                    mujoco.mjtObj.mjOBJ_BODY,
                    state.m.geom_bodyid[geom_id],
                )
                or ""
            ).startswith(state.ARM[key]["body_prefixes"])
        }
        for key in ("R", "L")
    }
    state.ALLARM = state.ARM_GEOMS["R"] | state.ARM_GEOMS["L"]
    state.CUP_GEOMS = {
        geom_id
        for geom_id in range(state.m.ngeom)
        if classify(state, geom_id, "R") == "cup"
    }
    # Graspability follows the weld's free-joint requirement, excluding arms.
    # CUP_GEOMS uses body/geom names and therefore misses movable objects such
    # as boxes, tape and pens. Free-joint ancestry works for every task target.
    state.GRASPABLE_GEOMS = {
        geom_id
        for geom_id in range(state.m.ngeom)
        if geom_id not in state.ALLARM
        and _has_free_joint_ancestor(state, int(state.m.geom_bodyid[geom_id]))
    }
    state.GRIP_GEOMS = {key: set() for key in ("R", "L")}
    state.FINGER_SIDE_GEOMS = {
        key: {"left": set(), "right": set()} for key in ("R", "L")
    }
    state.INNER_PAD_SIDE_GEOMS = {
        key: {"left": set(), "right": set()} for key in ("R", "L")
    }
    for key in ("R", "L"):
        for geom_id in state.ARM_GEOMS[key]:
            name = (gname(state, geom_id) + " " + _geom_body_name(state, geom_id)).lower()
            if "finger" in name or "gripper" in name:
                state.GRIP_GEOMS[key].add(geom_id)
            if "finger_left" in name:
                state.FINGER_SIDE_GEOMS[key]["left"].add(geom_id)
            if "finger_right" in name:
                state.FINGER_SIDE_GEOMS[key]["right"].add(geom_id)
            if "finger_left_pad_collision" in name:
                state.INNER_PAD_SIDE_GEOMS[key]["left"].add(geom_id)
            if "finger_right_pad_collision" in name:
                state.INNER_PAD_SIDE_GEOMS[key]["right"].add(geom_id)

def _geom_body_name(state,g):
    return mujoco.mj_id2name(state.m,mujoco.mjtObj.mjOBJ_BODY,state.m.geom_bodyid[g]) or ""

def classify(state,o,k):
    bn=_geom_body_name(state,o)
    if "cube" in (bn+gname(state,o)) or "cup" in (bn+gname(state,o)): return "cup"
    if "plate" in (bn+gname(state,o)): return "plate"
    if o in state.ARM_GEOMS[k]: return "self"
    if o in state.ALLARM: return "other_arm"
    return "table/world"

def gripper_cup_contacts(state,contact_tol_mm=None):
    tol_m = (float(contact_tol_mm) / 1000.0) if contact_tol_mm is not None else float("inf")
    out={}
    for key in ("R","L"):
        contacts=[]
        for ci in range(state.d.ncon):
            c=state.d.contact[ci]
            g1,g2=int(c.geom1),int(c.geom2)
            if g1 in state.GRIP_GEOMS[key] and g2 in state.CUP_GEOMS:
                gg,cg=g1,g2
            elif g2 in state.GRIP_GEOMS[key] and g1 in state.CUP_GEOMS:
                gg,cg=g2,g1
            else:
                continue
            if float(c.dist) > tol_m:
                continue
            force6=np.zeros(6,dtype=np.float64)
            try:
                mujoco.mj_contactForce(state.m,state.d,ci,force6)
            except Exception:
                force6[:]=0.0
            contacts.append({
                "gripper_geom":gname(state,gg),
                "cup_geom":gname(state,cg),
                "dist_mm":round(float(c.dist)*1000.0,3),
                "normal_force_n":round(float(force6[0]),4),
            })
        dists=[c["dist_mm"] for c in contacts]
        forces=[c["normal_force_n"] for c in contacts]
        out[key]={
            "count":len(contacts),
            "min_dist_mm":round(float(min(dists)),3) if dists else None,
            "max_pen_mm":round(float(max(0.0, -min(dists))),3) if dists else 0.0,
            "normal_force_sum_n":round(float(sum(forces)),4),
            "contacts":contacts[:8],
        }
    return out

def penetrations(state):
    return collect_arm_penetrations(
        data=state.d,
        arm_geom_ids=state.ALLARM,
        right_arm_geom_ids=state.ARM_GEOMS["R"],
        geom_name=functools.partial(gname,state),
        classify_other=functools.partial(classify,state),
    )

__all__ = [
    "_geom_body_name",
    "classify",
    "gripper_cup_contacts",
    "initialize_geometry_sets",
    "penetrations",
]
