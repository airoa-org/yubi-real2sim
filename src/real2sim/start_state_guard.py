# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Check the first frame before spending an episode on it.

A rollout that starts with two objects inside each other is not a rollout of
the task: the first tenth of a second is the solver shoving them apart, and
whatever the policy does afterwards was measured on a scene nobody placed. The
failures this catches are quiet ones -- an object seated on a layout that moved
under it, a part spawned at the world origin because a sampler was dropped, a
container straddling the table edge -- and each costs a full batch before anyone
looks at a video.

The check is MuJoCo's own narrow phase, not a second geometry implementation:
step nothing, call ``mj_forward`` and read ``data.contact``. A contact whose
``dist`` is negative is a penetration of exactly that depth, measured on the
collision geometry the episode will actually use, with the same margins.

    report = check_start_state(model, data)
    if report.failed:
        raise SystemExit(report.describe())

Contacts are not defects by themselves -- a cup resting on a table is a contact
at ``dist`` about zero, and compliant contacts settle a little way in by
design. Only depth past ``penetration_tol_m`` counts, which is why the default
is millimetres rather than zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

__all__ = [
    "Overlap",
    "StartStateReport",
    "check_start_state",
]

# Compliant contacts (solimp/solref) sit a fraction of a millimetre in when
# they are doing their job. 2 mm is past that and well under the smallest
# object dimension in any current task, so it separates "resting" from "spawned
# inside".
DEFAULT_PENETRATION_TOL_M = 0.002


@dataclass(frozen=True)
class Overlap:
    """Two collision geoms that start the episode inside one another."""

    geom_a: str
    geom_b: str
    body_a: str
    body_b: str
    depth_m: float

    @property
    def depth_mm(self) -> float:
        return self.depth_m * 1000.0

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return (
            f"{self.body_a}/{self.geom_a} into {self.body_b}/{self.geom_b} "
            f"by {self.depth_mm:.2f} mm"
        )


@dataclass
class StartStateReport:
    """What the first frame looks like, and whether it is usable."""

    overlaps: list[Overlap] = field(default_factory=list)
    contacts: int = 0
    penetration_tol_m: float = DEFAULT_PENETRATION_TOL_M
    ignored: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return bool(self.overlaps)

    @property
    def worst_mm(self) -> float:
        return max((o.depth_mm for o in self.overlaps), default=0.0)

    def describe(self) -> str:
        head = (
            f"start state: {self.contacts} contact(s), "
            f"{len(self.overlaps)} past the {self.penetration_tol_m*1000:.1f} mm "
            "penetration tolerance"
        )
        if not self.overlaps:
            return head + " -- clear"
        lines = [head + ":"]
        lines += [f"  {overlap}" for overlap in sorted(
            self.overlaps, key=lambda o: -o.depth_m)]
        return "\n".join(lines)

    def as_dict(self) -> dict:
        """The shape that goes into ``sim_summary.json``."""
        return {
            "contacts": self.contacts,
            "penetration_tol_mm": round(self.penetration_tol_m * 1000.0, 3),
            "worst_penetration_mm": round(self.worst_mm, 3),
            "overlaps": [
                {
                    "geom_a": o.geom_a,
                    "geom_b": o.geom_b,
                    "body_a": o.body_a,
                    "body_b": o.body_b,
                    "depth_mm": round(o.depth_mm, 3),
                }
                for o in sorted(self.overlaps, key=lambda o: -o.depth_m)
            ],
            "ignored_pairs": sorted(self.ignored),
        }


def _name(model, obj_enum, index: int, fallback: str) -> str:
    import mujoco

    got = mujoco.mj_id2name(model, obj_enum, int(index))
    return got if got else f"{fallback}{int(index)}"


def check_start_state(
    model,
    data,
    *,
    penetration_tol_m: float = DEFAULT_PENETRATION_TOL_M,
    ignore_body_prefixes: Sequence[str] = (),
    ignore_pairs: Sequence[tuple[str, str]] = (),
    forward: bool = True,
) -> StartStateReport:
    """Report collision pairs that begin the episode interpenetrating.

    ``ignore_body_prefixes`` drops pairs where *both* bodies match a prefix --
    the robot's own links touch each other by construction, and a gripper
    closed on its own pad is not a placement defect. A pair is only excused
    when both sides match, so a link driven into an object is still reported.
    """
    import mujoco

    if forward:
        mujoco.mj_forward(model, data)

    tol = abs(float(penetration_tol_m))
    excused = {frozenset(pair) for pair in ignore_pairs}
    prefixes = tuple(ignore_body_prefixes)

    report = StartStateReport(penetration_tol_m=tol)
    report.contacts = int(data.ncon)

    # One geom pair produces up to four contact points -- a box resting on a
    # box gives one per corner. They are the same defect seen four times, so
    # keep the deepest per pair; otherwise "two objects overlap" is reported as
    # four overlaps and the count stops meaning anything.
    deepest: dict[tuple[int, int], Overlap] = {}
    seen_ignored: set[str] = set()

    for index in range(int(data.ncon)):
        contact = data.contact[index]
        depth = -float(contact.dist)
        if depth <= tol:
            continue
        g1, g2 = int(contact.geom1), int(contact.geom2)
        geom_a = _name(model, mujoco.mjtObj.mjOBJ_GEOM, g1, "geom")
        geom_b = _name(model, mujoco.mjtObj.mjOBJ_GEOM, g2, "geom")
        body_a = _name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g1], "body")
        body_b = _name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g2], "body")

        pair_names = "/".join(sorted((body_a, body_b)))
        if frozenset((body_a, body_b)) in excused or frozenset((geom_a, geom_b)) in excused:
            seen_ignored.add(pair_names)
            continue
        if prefixes and body_a.startswith(prefixes) and body_b.startswith(prefixes):
            seen_ignored.add(pair_names)
            continue

        key = (g1, g2) if g1 <= g2 else (g2, g1)
        previous = deepest.get(key)
        if previous is None or depth > previous.depth_m:
            deepest[key] = Overlap(
                geom_a=geom_a,
                geom_b=geom_b,
                body_a=body_a,
                body_b=body_b,
                depth_m=depth,
            )

    report.overlaps = list(deepest.values())
    report.ignored = sorted(seen_ignored)
    return report
