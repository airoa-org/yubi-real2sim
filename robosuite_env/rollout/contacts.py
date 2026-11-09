# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Pure contact and penetration metrics used by rollout workers."""

from __future__ import annotations

from collections.abc import Callable, Set
from typing import Any


def collect_arm_penetrations(
    *,
    data: Any,
    arm_geom_ids: Set[int],
    right_arm_geom_ids: Set[int],
    geom_name: Callable[[int], str],
    classify_other: Callable[[int, str], str],
) -> list[dict[str, Any]]:
    """Return penetrating contacts involving either robot arm.

    The record order and field schema intentionally match the historical
    worker-local ``penetrations`` helper.
    """

    arm_geom_ids = {int(value) for value in arm_geom_ids}
    right_arm_geom_ids = {int(value) for value in right_arm_geom_ids}
    output: list[dict[str, Any]] = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        if float(contact.dist) >= 0.0:
            continue
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        arm1 = geom1 in arm_geom_ids
        arm2 = geom2 in arm_geom_ids
        if not (arm1 or arm2):
            continue
        arm_geom = geom1 if arm1 else geom2
        other_geom = geom2 if arm1 else geom1
        side = "R" if arm_geom in right_arm_geom_ids else "L"
        output.append(
            {
                "armgeom": geom_name(arm_geom),
                "other": geom_name(other_geom),
                "cat": classify_other(other_geom, side),
                "depth_mm": round(-float(contact.dist) * 1000.0, 3),
            }
        )
    return output


__all__ = ["collect_arm_penetrations"]
