# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Small, import-safe predicates for rollout contact overrides."""

from __future__ import annotations


def is_legacy_finger_contact_override_target(
    *,
    body_name: str,
    geom_name: str,
    geom_group: int,
) -> bool:
    """Return whether a geom may receive the legacy fingertip override.

    Shared YUBI exterior guards own a separate low-friction, hard-contact
    contract in MJCF.  Their compiled names are namespace-prefixed by
    robosuite, so both the standalone ``guard_*`` and compiled
    ``gripper*_guard_*`` forms are excluded explicitly.
    """

    name = str(geom_name)
    is_guard = name.startswith("guard_") or "_guard_" in name
    return int(geom_group) == 0 and "finger" in str(body_name) and not is_guard


def legacy_finger_contact_override_ids(model, mujoco_module):
    """Yield exactly the compiled geom IDs eligible for legacy overrides."""

    for geom_id in range(model.ngeom):
        body_name = (
            mujoco_module.mj_id2name(
                model,
                mujoco_module.mjtObj.mjOBJ_BODY,
                model.geom_bodyid[geom_id],
            )
            or ""
        )
        geom_name = (
            mujoco_module.mj_id2name(
                model,
                mujoco_module.mjtObj.mjOBJ_GEOM,
                geom_id,
            )
            or ""
        )
        if is_legacy_finger_contact_override_target(
            body_name=body_name,
            geom_name=geom_name,
            geom_group=int(model.geom_group[geom_id]),
        ):
            yield geom_id


__all__ = [
    "is_legacy_finger_contact_override_target",
    "legacy_finger_contact_override_ids",
]
