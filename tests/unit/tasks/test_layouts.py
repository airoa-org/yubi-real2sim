# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The hand-authored layout contract.

These mirror the recorded pose contract's rejections: a layout that was placed
by hand must say so, and must not borrow evidence from a capture that never
saw it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from robosuite_env.tasks.layouts import LayoutError, validate_manual_layout


REPO_ROOT = Path(__file__).resolve().parents[3]
BOX_LAYOUT = REPO_ROOT / "robosuite_env/obj_calib/layout_box_pack_duo_manual.json"
BOX_OBJECTS = ("box", "tape", "pen", "lan")


def _payload():
    return json.loads(BOX_LAYOUT.read_text(encoding="utf-8"))


def test_the_committed_layouts_validate():
    box = validate_manual_layout(_payload(), expected_objects=BOX_OBJECTS)
    assert set(box["objects"]) == set(BOX_OBJECTS)
    for key in ("scene", "placed_by", "placed_at", "tool", "rationale"):
        assert box["source"][key]


def test_the_box_faces_the_operator():
    """The board set it to 180 deg; a yaw of zero would face it away."""

    layout = validate_manual_layout(_payload(), expected_objects=BOX_OBJECTS)

    w, x, y, z = layout["objects"]["box"]["quaternion_wxyz"]
    assert (round(w, 6), round(x, 6), round(y, 6), round(z, 6)) == (0.0, 0.0, 0.0, 1.0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda v: v["source"].update(kind="lerobot_frame0_observation"),
         "source.kind must be"),
        (lambda v: v["source"].update(dataset="unrelated_capture"),
         "must not carry capture evidence"),
        (lambda v: v["source"].update(episode_index=0),
         "must not carry capture evidence"),
        (lambda v: v["source"].update(
            calibration={"path": "x", "file_sha256": "0" * 64}),
         "must not carry capture evidence"),
        (lambda v: v["source"].pop("rationale"), "missing required keys"),
        (lambda v: v["source"].pop("placed_by"), "missing required keys"),
        (lambda v: v["source"].update(placed_at="yesterday"),
         "must be ISO 8601"),
        (lambda v: v["source"].update(placed_at="2026-09-06T04:19:14"),
         "must carry a UTC offset"),
        (lambda v: v.update(frame="world"), "frame must be 'table'"),
        (lambda v: v.update(schema_version=2), "schema_version must be integer 1"),
        (lambda v: v["objects"].pop("pen"), "objects must be exactly"),
        (lambda v: v["objects"].update(stapler=v["objects"]["pen"]),
         "objects must be exactly"),
        (lambda v: v["objects"]["tape"].update(quaternion_wxyz=[2.0, 0.0, 0.0, 0.0]),
         "must be unit length"),
        (lambda v: v["objects"]["tape"].update(position_m=[0.0, 0.0]),
         "must have exactly 3 entries"),
    ],
)
def test_a_layout_that_misstates_its_origin_is_rejected(mutation, message):
    payload = copy.deepcopy(_payload())
    mutation(payload)
    with pytest.raises(LayoutError, match=message):
        validate_manual_layout(payload, expected_objects=BOX_OBJECTS)
