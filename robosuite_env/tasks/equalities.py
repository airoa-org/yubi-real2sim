# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo equality declarations shared by manipulation-task environments."""

from __future__ import annotations

import xml.etree.ElementTree as ET


RULE_GRASP_WELD_NAMES = {
    "R": "rule_grasp_weld_R",
    "L": "rule_grasp_weld_L",
}


def add_rule_grasp_weld_equalities(
    root: ET.Element,
    *,
    placeholder_bodies: tuple[str, str],
) -> tuple[ET.Element, ...]:
    """Declare one inactive runtime-retargetable weld for each hand.

    MuJoCo fixes the equality count when it compiles the XML, so each weld has
    to name a body pair up front; rule-grasp replaces their compiled IDs and
    relative pose only when a grasp latches, and the declared pair is never
    welded as written (``active="false"``).

    ``placeholder_bodies`` is that pair. The caller names it -- normally its
    own two task objects -- so the compiled XML does not depend on which body
    ``worldbody`` happens to yield first, and one model always fingerprints the
    same way.
    """

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("manipulation task XML has no worldbody")
    body_names = [str(name) for name in placeholder_bodies]
    if len(body_names) != 2 or body_names[0] == body_names[1]:
        raise ValueError(
            "rule-grasp weld placeholders must be two distinct body names, got "
            f"{body_names}"
        )
    declared = {body.get("name") for body in worldbody.iter("body")}
    missing = [name for name in body_names if name not in declared]
    if missing:
        raise ValueError(
            "rule-grasp weld placeholder body is not in the task XML: "
            + ", ".join(missing)
        )

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    existing_names = {
        element.get("name") for element in equality if element.get("name")
    }
    conflicts = sorted(set(RULE_GRASP_WELD_NAMES.values()) & existing_names)
    if conflicts:
        raise ValueError(
            "rule-grasp weld equality name already exists: " + ", ".join(conflicts)
        )

    welds = []
    for hand in ("R", "L"):
        welds.append(
            ET.SubElement(
                equality,
                "weld",
                name=RULE_GRASP_WELD_NAMES[hand],
                body1=body_names[0],
                body2=body_names[1],
                active="false",
                solref="0.005 1",
                solimp="0.99 0.999 0.001",
                torquescale="1",
            )
        )
    return tuple(welds)


__all__ = ["RULE_GRASP_WELD_NAMES", "add_rule_grasp_weld_equalities"]
