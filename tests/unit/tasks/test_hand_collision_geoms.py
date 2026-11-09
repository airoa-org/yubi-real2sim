# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Which collision geoms belong to which hand, and how wide the set is.

Three separate things can go wrong, and each is pinned on its own:

* the profile can fit the wrong bodies -- pinned by the hand roots, which are
  the one value that differs between rigs;
* the set can be drawn too wide or too narrow -- pinned by the counts;
* the two hands can be mixed together -- pinned by them being disjoint, and by
  the left and right counts differing, which a mixed-up implementation cannot
  reproduce.

The numbers come from compiled models, not from reading the XML: the Duo
numbers from the container environment compiled on franka_duo, and the Flat
numbers from success_criterion.hands as franka_flat box runs record it.

Flat and Duo agree to the geom because both hang the same right-hand and
left-hand YUBI grippers off the same FR3 arms.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS = REPO_ROOT / "robosuite_env" / "assets"

#: rig -> hand -> (hand root body, every collision geom below the wrist,
#: the finger geoms that are not guards)
RIG_HANDS = {
    "franka_flat": {
        "right": ("robot0_right_hand", 178, 44),
        "left": ("robot1_right_hand", 179, 45),
    },
    "duo": {
        "right": ("robot0_right_hand", 178, 44),
        "left": ("robot1_right_hand", 179, 45),
    },
}

#: The gripper each hand carries, and what it holds. 178 = 76 finger geoms +
#: 102 that are not on a finger (the wrist mount and flange, 32, and the
#: gripper body's own guards, 70); 76 = 44 kept + 32 guard_finger_*.
YUBI_GRIPPERS = {
    "yubi_right": (178, 76, 44),
    "yubi_left": (179, 77, 45),
}


def _counts(path):
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(path))

    def on_finger(geom):
        body = int(model.geom_bodyid[geom])
        while body > 0:
            if "finger" in (model.body(body).name or "").lower():
                return True
            body = int(model.body_parentid[body])
        return False

    contact = [
        geom
        for geom in range(int(model.ngeom))
        if int(model.geom_contype[geom]) or int(model.geom_conaffinity[geom])
    ]
    finger = [geom for geom in contact if on_finger(geom)]
    kept = [
        geom
        for geom in finger
        if "guard" not in (model.geom(geom).name or "").lower()
    ]
    return len(contact), len(finger), len(kept)


@pytest.mark.parametrize("gripper", sorted(YUBI_GRIPPERS))
def test_each_yubi_gripper_holds_the_geoms_the_rigs_counted(gripper):
    path = ASSETS / f"{gripper}.xml"
    if not path.is_file():
        pytest.skip(f"{gripper}.xml (gitignored meshes) is not installed")
    assert _counts(path) == YUBI_GRIPPERS[gripper]


def test_the_two_hands_are_not_interchangeable():
    # The left gripper is a mirrored model, not a copy: it carries one more
    # finger pad. An implementation that mixed the hands, or read one hand
    # twice, could not produce this asymmetry.
    for rig, hands in RIG_HANDS.items():
        right_root, right_all, right_fingers = hands["right"]
        left_root, left_all, left_fingers = hands["left"]
        assert right_root != left_root, rig
        assert (right_all, right_fingers) == (178, 44), rig
        assert (left_all, left_fingers) == (179, 45), rig


def test_each_rig_is_told_apart_by_its_hand_roots():
    # The roots say the profile matched the bodies this rig actually has.
    roots = {
        rig: (hands["right"][0], hands["left"][0])
        for rig, hands in RIG_HANDS.items()
    }
    assert roots["duo"] == roots["franka_flat"] == (
        "robot0_right_hand",
        "robot1_right_hand",
    )


def test_narrowing_to_the_fingers_drops_the_guards_and_the_wrist():
    # 178 -> 44 is not a rounding of the set: it drops the wrist mount, the
    # flange and every guard, which is the point of the switch for the container.
    for gripper, (whole, fingers, kept) in YUBI_GRIPPERS.items():
        assert fingers < whole and kept < fingers
        assert whole - fingers == 102, gripper  # mount and flange, body guards
        assert fingers - kept == 32, gripper  # guard_finger_*


def _profile_joints(name):
    from robosuite_env.robots.profiles import get_robot_profile

    profile = get_robot_profile(name)
    return {
        key: (tuple(arm.joint_names), tuple(arm.finger_joint_names))
        for key, arm in profile.arms.items()
    }


def test_fr3_and_fr3_duo_are_the_same_joints_under_two_names():
    # Why a Flat or Duo model resolves with two profiles in rig_profiles:
    # fr3_duo is the fr3 profile renamed. A test that pinned rig_profiles by
    # equality would break the day a franka_flat profile is registered, while
    # the resolver kept working, so membership is what is checked.
    assert _profile_joints("fr3") == _profile_joints("fr3_duo")


@pytest.mark.parametrize(
    ("rig", "expected_profile"),
    [("franka_flat", "fr3"), ("duo", "fr3_duo")],
)
def test_every_rig_has_a_profile_that_claims_it(rig, expected_profile):
    from robosuite_env.robots.profiles import SUPPORTED_ROBOT_NAMES

    assert expected_profile in SUPPORTED_ROBOT_NAMES
    assert rig in RIG_HANDS
