# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""``cup_above_plate_test`` reaches the cup_on_plate criterion end to end.

The rigs scored "cup above the plate" two ways: Duo by the cup's
mid-height, Flat by its lowest collision point. The merged criterion keeps
both behind one option, so each link -- profile, launcher argv, worker CLI,
criterion options -- has to carry it, and an absent value must leave every
existing profile's argv unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from real2sim import launcher
from robosuite_env.rollout import cli

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE = REPO_ROOT / "config" / "release"


def _render(**extra):
    source = yaml.safe_load((RELEASE / "fixture_v2.yaml").read_text(encoding="utf-8"))
    source["simulation"]["parameters"].update(extra)
    return launcher._validate_v2_parameters(
        source["simulation"]["parameters"],
        context={
            "repo_root": str(REPO_ROOT),
            "config_dir": str(RELEASE),
            "timestamp": "20990101_000000",
            "robot": "fr3",
        },
        config_dir=RELEASE,
    )


def test_an_absent_test_renders_no_argument():
    resolved, argv = _render()
    assert resolved["cup_above_plate_test"] is None
    assert "--cup-above-plate-test" not in argv


@pytest.mark.parametrize("value", ["middle", "lowest"])
def test_a_stated_test_reaches_the_worker_argv(value):
    resolved, argv = _render(cup_above_plate_test=value)
    assert resolved["cup_above_plate_test"] == value
    index = argv.index("--cup-above-plate-test")
    assert argv[index + 1] == value


def test_an_unknown_test_is_refused():
    with pytest.raises(launcher.ConfigError, match="cup_above_plate_test"):
        _render(cup_above_plate_test="top")


@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], "middle"), (["--cup-above-plate-test", "lowest"], "lowest")],
)
def test_the_worker_cli_parses_it_with_the_criterion_default(argv, expected):
    args = cli.parse(["--policy-batch-server", "127.0.0.1:1", *argv])
    assert args.cup_above_plate_test == expected


def test_the_worker_hands_it_to_the_success_criterion():
    source = (
        REPO_ROOT / "robosuite_env" / "rollout" / "stages" / "init_policy_render.py"
    ).read_text(encoding="utf-8")
    assert '"cup_above_plate_test": str(' in source
    assert 'getattr(state.A, "cup_above_plate_test", "middle")' in source
