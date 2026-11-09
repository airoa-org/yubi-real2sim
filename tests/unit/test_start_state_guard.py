# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""An episode that starts with objects inside each other is not the task.

Objects that start interpenetrating -- for example an object left at the world
origin -- are reported before the first step, so such runs cannot pass for a
policy result.

These build real MuJoCo models, because the point of the check is that it reads
MuJoCo's own narrow phase rather than a second geometry implementation.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from real2sim.start_state_guard import DEFAULT_PENETRATION_TOL_M, check_start_state


def _model(xml: str):
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model)


TWO_BOXES = """
<mujoco>
  <worldbody>
    <body name="left" pos="0 0 0.05">
      <freejoint/>
      <geom name="left_g" type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="right" pos="{gap} 0 0.05">
      <freejoint/>
      <geom name="right_g" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_a_clear_start_state_passes():
    model, data = _model(TWO_BOXES.format(gap=0.2))
    report = check_start_state(model, data)
    assert not report.failed
    assert report.overlaps == []
    assert "clear" in report.describe()


def test_objects_inside_each_other_are_reported_with_the_depth():
    # Half-widths are 0.05 each, so centres 0.08 apart overlap by 20 mm.
    model, data = _model(TWO_BOXES.format(gap=0.08))
    report = check_start_state(model, data)
    assert report.failed
    assert len(report.overlaps) == 1
    overlap = report.overlaps[0]
    assert {overlap.body_a, overlap.body_b} == {"left", "right"}
    assert overlap.depth_mm == pytest.approx(20.0, abs=0.5)
    assert "20." in report.describe()


def test_resting_contact_is_not_an_overlap():
    """A cup on a table touches at dist ~0; that must not read as a defect."""
    model, data = _model("""
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 0.1"/>
        <body name="cup" pos="0 0 0.05">
          <freejoint/>
          <geom name="cup_g" type="box" size="0.05 0.05 0.05"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    report = check_start_state(model, data)
    assert report.contacts >= 1
    assert not report.failed


def test_the_tolerance_is_what_decides():
    model, data = _model(TWO_BOXES.format(gap=0.0985))  # 1.5 mm of overlap
    assert not check_start_state(model, data).failed          # default 2 mm
    strict = check_start_state(model, data, penetration_tol_m=0.0005)
    assert strict.failed
    assert strict.overlaps[0].depth_mm == pytest.approx(1.5, abs=0.3)


def test_two_robot_links_touching_each_other_are_excused():
    model, data = _model("""
    <mujoco>
      <worldbody>
        <body name="robot0_link1" pos="0 0 0.05">
          <joint name="j1" type="slide" axis="1 0 0"/>
          <geom name="l1" type="box" size="0.05 0.05 0.05"/>
        </body>
        <body name="robot0_link2" pos="0.08 0 0.05">
          <joint name="j2" type="slide" axis="1 0 0"/>
          <geom name="l2" type="box" size="0.05 0.05 0.05"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    report = check_start_state(model, data, ignore_body_prefixes=("robot0", "gripper0"))
    assert not report.failed
    assert report.ignored == ["robot0_link1/robot0_link2"]


def test_a_link_driven_into_an_object_is_still_reported():
    """Both sides must match the prefix to be excused, not just one."""
    model, data = _model("""
    <mujoco>
      <worldbody>
        <body name="robot0_link1" pos="0 0 0.05">
          <geom name="l1" type="box" size="0.05 0.05 0.05"/>
        </body>
        <body name="cup" pos="0.08 0 0.05">
          <freejoint/>
          <geom name="cup_g" type="box" size="0.05 0.05 0.05"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    report = check_start_state(model, data, ignore_body_prefixes=("robot0",))
    assert report.failed
    assert {report.overlaps[0].body_a, report.overlaps[0].body_b} == {
        "robot0_link1", "cup"}


def test_a_named_pair_can_be_excused():
    model, data = _model(TWO_BOXES.format(gap=0.08))
    report = check_start_state(model, data, ignore_pairs=[("left", "right")])
    assert not report.failed
    assert report.ignored == ["left/right"]


def test_the_summary_dict_carries_the_worst_depth():
    model, data = _model(TWO_BOXES.format(gap=0.08))
    payload = check_start_state(model, data).as_dict()
    assert payload["worst_penetration_mm"] == pytest.approx(20.0, abs=0.5)
    assert payload["penetration_tol_mm"] == pytest.approx(
        DEFAULT_PENETRATION_TOL_M * 1000)
    assert payload["overlaps"][0]["depth_mm"] == pytest.approx(20.0, abs=0.5)
