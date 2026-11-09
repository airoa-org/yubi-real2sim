# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the rollout worker command line."""

from __future__ import annotations

import pytest

from robosuite_env.rendering.wrist_render_profiles import (
    FRANKA_PINHOLE_WRIST_RGB_PROFILE,
    ELP_MEAN_WRIST_RGB_PROFILE,
)
from robosuite_env.robots.profiles import SUPPORTED_ROBOT_NAMES
from robosuite_env.rollout.cli import parse


@pytest.mark.parametrize("objects,source,profile", (
    ("cup_plate", "sim", FRANKA_PINHOLE_WRIST_RGB_PROFILE),
    ("cup_plate", "lerobot", ELP_MEAN_WRIST_RGB_PROFILE),
    ("cup_plate", "sim", ELP_MEAN_WRIST_RGB_PROFILE),
))
def test_supported_wrist_profile_configurations(objects, source, profile):
    arguments = parse([
        "--replay", "episode.parquet", "--objects", objects,
        "--policy-image-source", source, "--policy-yubi-render-profile", profile,
    ])
    assert arguments.policy_yubi_render_profile == profile


def test_obs_gs_only_is_parsed_and_excludes_no_obs_gs() -> None:
    arguments = parse(["--replay", "episode.parquet", "--obs-gs-only"])

    assert arguments.obs_gs_only is True
    assert arguments.no_obs_gs is False

    with pytest.raises(SystemExit) as raised:
        parse(
            [
                "--replay",
                "episode.parquet",
                "--no-obs-gs",
                "--obs-gs-only",
            ]
        )
    assert raised.value.code == 2


def test_robot_choices_track_the_profile_registry() -> None:
    """The worker must accept every embodiment the profiles declare.

    Hard-coded ``--robot`` choices would make a run for any other rig die in
    argparse before any profile, asset, or scene work could report a real
    problem. Deriving the list keeps the worker reachable for each embodiment
    the registry gains.
    """

    for name in SUPPORTED_ROBOT_NAMES:
        arguments = parse(["--replay", "episode.parquet", "--robot", name])
        assert arguments.robot == name

    with pytest.raises(SystemExit) as raised:
        parse(["--replay", "episode.parquet", "--robot", "panda"])
    assert raised.value.code == 2


def test_removed_ros_bridge_source_is_rejected(capsys) -> None:
    """The ROS 2 bridge left the release, and the worker source with it."""

    with pytest.raises(SystemExit) as raised:
        parse(["--replay", "episode.parquet", "--ros-bridge"])
    assert raised.value.code == 2
    assert "unrecognized arguments: --ros-bridge" in capsys.readouterr().err
