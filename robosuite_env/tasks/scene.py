#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Stand a rig's mounted roots on the common table-origin scene.

Two ways to place a robot, and they are not the same operation wearing
different clothes. The calibrated path puts every base slot where a measured
or tuned layout says it stands, table-top relative. The stock path spreads
robots on robosuite's nominal table offsets: it never reads the table top,
never sets an orientation, and exists for the diagnostic environments that
have no registered scene to be calibrated against.

Which one a task uses is the task's decision, made once. A rig that asks for a
calibrated layout and quietly receives the nominal one is the failure this
module is shaped to prevent -- it looks plausible from most camera angles.
"""

from __future__ import annotations

import numpy as np

from ..robots.base_placement import RobotBasePlacement
from ..robots.profiles import get_robot_profile


def robosuite_base_euler(rpy_rad) -> np.ndarray:
    """Re-express a URDF ``(r, p, y)`` triple in the order robosuite composes.

    ``fr3_base_from_blend.json`` and the Duo mount file both quote rpy the way
    URDF does, ``Rz(y) @ Ry(p) @ Rx(r)``. ``set_base_ori`` hands its argument to
    robosuite's ``euler2mat``, which composes the other way,
    ``Rx(r) @ Ry(p) @ Rz(y)``. Passing URDF angles straight through silently
    orients the base wrongly whenever roll or pitch is non-zero -- a yaw-only
    pose is identical under both, which is why the two-pedestal layouts never
    exposed it.
    """

    roll, pitch, yaw = (float(v) for v in np.asarray(rpy_rad, dtype=float))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    urdf = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    # Solve Rx(a) @ Ry(b) @ Rz(c) == urdf.
    b = np.arcsin(np.clip(urdf[0, 2], -1.0, 1.0))
    c = np.arctan2(-urdf[0, 1], urdf[0, 0])
    a = np.arctan2(-urdf[1, 2], urdf[2, 2])
    return np.array([a, b, c])


def place_robots_at_calibrated_bases(
    robots,
    *,
    table_top,
    placement: RobotBasePlacement,
) -> None:
    """Put every base slot the rig declares where its calibration says.

    The loop runs once per FR3 pedestal; there is no per-rig branch, because
    the slots are data the profile already declares.
    """

    profile = get_robot_profile(placement.robot)
    expected = len(profile.robosuite_robots)
    if len(robots) != expected:
        raise ValueError(
            f"{placement.robot} places {expected} robosuite robot(s), "
            f"got {len(robots)}"
        )

    table_top = np.asarray(table_top, dtype=float)
    for slot in profile.base_slots:
        pose = placement.pose(slot.name)
        model = robots[slot.robot_index].robot_model
        position = table_top + pose.pos_m
        model.set_base_xpos(position)
        model.set_base_ori(robosuite_base_euler(pose.rpy_rad))
        print(
            f"[scene] {placement.robot} base {slot.name} -> "
            f"pos={np.round(position, 4).tolist()} "
            f"rpy={np.round(np.degrees(pose.rpy_rad), 3).tolist()}deg "
            f"({placement.source_sha256[:12]})",
            flush=True,
        )


def place_robots_at_stock_offsets(robots, *, table_length: float) -> None:
    """Spread robots on robosuite's nominal table offsets.

    Deliberately ignores the table top and leaves every orientation alone:
    this is the un-registered diagnostic layout, and pretending otherwise
    would make it look like a calibration.
    """

    base = np.asarray(
        robots[0].robot_model.base_xpos_offset["table"](table_length),
        dtype=float,
    )
    count = len(robots)
    for index, robot in enumerate(robots):
        offset = np.array(
            [0.0, (index - (count - 1) / 2.0) * 0.5, 0.0], dtype=float
        )
        robot.robot_model.set_base_xpos(base + offset)
