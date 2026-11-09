# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Named FR3 initial arm-joint pose registry and its schema-v2 binding.

The registry is launcher-only sugar: both spellings must collapse to the same
``--initial-arm-joints-json <path>`` worker argument, so nothing downstream --
the pose loader, the joint ready gate, or the settled init-state contract --
can tell them apart.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from unittest import mock

import pytest
import yaml

from real2sim import launcher
from real2sim.cli import build_arg_parser
from real2sim.config._types import ConfigError
from real2sim.config.initial_arm_joints import (
    INITIAL_ARM_JOINT_POSES,
    format_initial_arm_joint_pose_table,
    initial_arm_joint_pose_names,
    initial_arm_joint_pose_path,
    LEGACY_POSE_NAMES,
    canonical_pose_name,
)
from robosuite_env.robots.fr3.initial_pose import load_initial_arm_joint_pose
from tests._helpers.launcher import base_config
from tests._helpers.repo import REPO_ROOT


@pytest.fixture(autouse=True)
def _isolate_cpu_fixture_from_ambient_cuda(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def _load(config: dict) -> dict:
    directory = Path(tempfile.mkdtemp(prefix=".initial-joints-", dir=REPO_ROOT / "config"))
    try:
        path = directory / "probe.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with mock.patch.dict(
            os.environ, {"REAL2SIM_INPUTS_ROOT": str(REPO_ROOT)}, clear=False
        ):
            return launcher.load_rollout_config(path, timestamp="20260730_120000")
    finally:
        for child in sorted(directory.iterdir()):
            child.unlink()
        directory.rmdir()


def _plan(path: Path) -> dict:
    with mock.patch.dict(
        os.environ, {"REAL2SIM_INPUTS_ROOT": str(REPO_ROOT)}, clear=False
    ):
        return launcher.load_rollout_config(path, timestamp="20260730_120000")


def _cli_value(arguments, option):
    return arguments[arguments.index(option) + 1]


def test_every_registered_pose_resolves_to_a_valid_checked_in_file():
    assert initial_arm_joint_pose_names() == (
        "fr3/duo/home",
        "fr3/duo/home_20260906",
        "fr3/duo/init_high",
        "fr3/flat/grip07",
        "fr3/flat/registered_cup_calibrated",
    )

    for name in initial_arm_joint_pose_names():
        path = initial_arm_joint_pose_path(name)
        assert path.is_file(), name
        # A registered name must never point at an unreadable or off-contract
        # file; the loader is the same one the worker uses.
        load_initial_arm_joint_pose(path)
        assert not INITIAL_ARM_JOINT_POSES[name].startswith("/"), name


def test_grip07_carries_the_real_robot_capture_and_its_raw_gripper():
    pose = load_initial_arm_joint_pose(initial_arm_joint_pose_path("grip07"))

    assert pose.schema_version == 2
    assert pose.grip_by_arm == {"L": 0.7, "R": 0.7}
    # The real robot's left arm controller -> simulator L, its right arm
    # controller -> simulator R. Swapping the two is the failure this pins.
    assert pose.q_by_arm["L"].tolist()[0] == pytest.approx(-0.279842)
    assert pose.q_by_arm["R"].tolist()[0] == pytest.approx(0.813472)


def test_the_historical_pose_stays_schema_1_with_no_gripper():
    pose = load_initial_arm_joint_pose(
        initial_arm_joint_pose_path("registered_cup_calibrated")
    )

    assert pose.schema_version == 1
    assert pose.grip_by_arm is None


def test_a_name_renders_the_same_worker_argument_as_the_path():
    named = copy.deepcopy(base_config())
    parameters = named["simulation"]["parameters"]
    del parameters["initial_arm_joints_json"]
    parameters["initial_arm_joints"] = "grip07"

    by_path = copy.deepcopy(base_config())
    by_path["simulation"]["parameters"]["initial_arm_joints_json"] = (
        "{repo_root}/" + INITIAL_ARM_JOINT_POSES[canonical_pose_name("grip07")]
    )

    named_plan = _load(named)
    path_plan = _load(by_path)
    expected = str(initial_arm_joint_pose_path("grip07"))

    assert (
        named_plan["resolved"]["simulation"]["parameters"]["initial_arm_joints"]
        == "grip07"
    )
    assert (
        named_plan["resolved"]["simulation"]["parameters"]["initial_arm_joints_json"]
        == expected
    )
    assert (
        path_plan["resolved"]["simulation"]["parameters"]["initial_arm_joints"] is None
    )
    named_arguments = named_plan["resolved"]["simulation"]["cli_args"]
    assert named_arguments == path_plan["resolved"]["simulation"]["cli_args"]
    assert _cli_value(named_arguments, "--initial-arm-joints-json") == expected
    # The launcher-only selector must never leak into the worker argv.
    assert "--initial-arm-joints" not in named_arguments


def test_a_name_and_a_path_together_are_rejected():
    config = copy.deepcopy(base_config())
    config["simulation"]["parameters"]["initial_arm_joints"] = "grip07"

    with pytest.raises(ConfigError, match="both select an initial arm joint pose"):
        _load(config)


def test_an_unknown_name_is_rejected_with_the_candidate_list():
    config = copy.deepcopy(base_config())
    parameters = config["simulation"]["parameters"]
    del parameters["initial_arm_joints_json"]
    parameters["initial_arm_joints"] = "grip7"

    with pytest.raises(ConfigError) as caught:
        _load(config)

    assert "not a registered initial arm joint pose" in str(caught.value)
    assert caught.value.allowed == list(initial_arm_joint_pose_names())
    assert "grip07" in caught.value.suggestions


def test_selecting_neither_keeps_the_argument_out_of_the_worker_argv():
    config = copy.deepcopy(base_config())
    del config["simulation"]["parameters"]["initial_arm_joints_json"]

    plan = _load(config)

    assert (
        plan["resolved"]["simulation"]["parameters"]["initial_arm_joints_json"] is None
    )
    assert "--initial-arm-joints-json" not in plan["resolved"]["simulation"]["cli_args"]


def test_the_registry_listing_is_reachable_from_the_launcher_cli(capsys):
    with pytest.raises(SystemExit) as caught:
        build_arg_parser().parse_args(["--list-initial-arm-joints"])

    assert caught.value.code == 0
    printed = capsys.readouterr().out
    assert printed == format_initial_arm_joint_pose_table()
    for name, relative in INITIAL_ARM_JOINT_POSES.items():
        assert name in printed
        assert relative in printed


def test_the_ci_fixture_keeps_the_historical_calibrated_pose():
    parameters = yaml.safe_load(
        (REPO_ROOT / "config/release/fixture_v2.yaml").read_text(encoding="utf-8")
    )["simulation"]["parameters"]

    assert parameters.get("initial_arm_joints") is None
    assert parameters["initial_arm_joints_json"] == "{repo_root}/" + INITIAL_ARM_JOINT_POSES[
        "fr3/flat/registered_cup_calibrated"
    ]


@pytest.mark.parametrize("room", ["a_room", "reading_room"])
def test_the_release_profiles_select_their_pose_by_name(room):
    """A release profile names its start pose; it does not spell out a path."""

    path = REPO_ROOT / f"config/release/fr3/flat/{room}_cup_cup1trim.yaml"
    parameters = yaml.safe_load(path.read_text(encoding="utf-8"))["simulation"][
        "parameters"
    ]

    assert parameters["initial_arm_joints"] == "fr3/flat/grip07"
    assert "initial_arm_joints_json" not in parameters

    arguments = _plan(path)["resolved"]["simulation"]["cli_args"]
    assert _cli_value(arguments, "--initial-arm-joints-json") == str(
        initial_arm_joint_pose_path("grip07")
    )


def test_the_registry_file_list_matches_the_initial_poses_directory():
    root = REPO_ROOT / "assets" / "initial_poses"
    on_disk = {
        path.relative_to(root).with_suffix("").as_posix()
        for path in sorted(root.glob("*/*/*.json"))
    }

    assert on_disk == set(INITIAL_ARM_JOINT_POSES)
    for name, relative in INITIAL_ARM_JOINT_POSES.items():
        assert relative == f"assets/initial_poses/{name}.json"
        payload = json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))
        robot, rig, _ = name.split("/")
        # The file signs the robot and rig its registry name spells.
        assert payload["signature"]["robot"] == robot, name
        assert payload["signature"]["rig"] == rig, name
        assert payload["signature"]["scene"] in payload["signature"]["scenes"], name
        assert payload["units"] == "rad"


def test_every_older_name_still_resolves_to_the_same_file():
    for old, new in LEGACY_POSE_NAMES.items():
        assert initial_arm_joint_pose_path(old) == initial_arm_joint_pose_path(new), old


def test_a_profile_whose_robot_differs_from_the_signature_is_refused():
    config = copy.deepcopy(base_config())
    parameters = config["simulation"]["parameters"]
    del parameters["initial_arm_joints_json"]
    parameters["initial_arm_joints"] = "fr3/duo/init_high"

    with pytest.raises(ConfigError, match="signed for fr3/duo"):
        _load(config)


def test_a_profile_spelling_an_old_pose_path_runs_the_moved_file():
    config = copy.deepcopy(base_config())
    config["simulation"]["parameters"]["initial_arm_joints_json"] = (
        "{repo_root}/config/fr3_grip07_initial_arm_joints.json"
    )

    resolved = _load(config)["resolved"]["simulation"]["parameters"]

    assert resolved["initial_arm_joints_json"] == str(
        initial_arm_joint_pose_path("fr3/flat/grip07")
    )


def test_every_old_pose_path_redirects_to_a_registered_file():
    from robosuite_env.robots.legacy_pose_paths import (
        LEGACY_PATHS_FILE,
        redirect_legacy_pose_path,
    )

    mapping = json.loads(LEGACY_PATHS_FILE.read_text(encoding="utf-8"))["paths"]
    # The two Duo home poses have no legacy path.
    assert set(mapping.values()) == set(INITIAL_ARM_JOINT_POSES.values()) - {
        INITIAL_ARM_JOINT_POSES["fr3/duo/home"],
        INITIAL_ARM_JOINT_POSES["fr3/duo/home_20260906"],
    }
    for old, new in mapping.items():
        assert not (REPO_ROOT / old).exists(), old
        assert redirect_legacy_pose_path(str(REPO_ROOT / old)) == str(REPO_ROOT / new)


def test_the_signature_check_reads_the_robot_as_the_profile_registry_does():
    from real2sim.config.initial_arm_joints import PROFILE_ROBOT_RIG, check_pose_signature
    from robosuite_env.robots.profiles import SUPPORTED_ROBOT_NAMES

    assert set(PROFILE_ROBOT_RIG) == set(SUPPORTED_ROBOT_NAMES)
    duo = str(initial_arm_joint_pose_path("fr3/duo/init_high"))
    check_pose_signature(duo, " FR3_DUO ")
    with pytest.raises(ConfigError, match="signed for fr3/duo"):
        check_pose_signature(duo, " FR3 ")
