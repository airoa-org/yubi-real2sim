# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Per-family bounds on the schema-v2 task-index parameters.

Box packing registers twelve prompt
contracts, but only its six into-box prompts have the required start layout
and direction plumbing, so the launcher refuses the six out-of-box prompts.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from real2sim import launcher
from real2sim.config import parameters


REPO_ROOT = Path(__file__).resolve().parents[3]
# No release profile runs box packing; the parameter bounds are checked on the
# strict CI fixture's parameters, as test_cup_above_plate_test does.
FIXTURE_CONFIG = REPO_ROOT / "config" / "release" / "fixture_v2.yaml"


def _validate(overrides):
    source = yaml.safe_load(FIXTURE_CONFIG.read_text(encoding="utf-8"))
    values = copy.deepcopy(source["simulation"]["parameters"])
    values.update(overrides)
    return launcher._validate_v2_parameters(
        values,
        context={
            "repo_root": str(REPO_ROOT),
            "config_dir": str(FIXTURE_CONFIG.parent),
            "timestamp": "20990101_000000",
            "robot": source["robot"],
        },
        config_dir=FIXTURE_CONFIG.parent,
    )


def test_box_pack_maximum_is_the_last_runnable_into_box_prompt():
    box_pack = pytest.importorskip("robosuite_env.tasks.box_pack")

    into_indices = tuple(
        prompt["task_index"]
        for prompt in box_pack.BOX_PACK_INSTRUCTIONS
        if prompt["direction"] == "in"
    )
    out_indices = tuple(
        prompt["task_index"]
        for prompt in box_pack.BOX_PACK_INSTRUCTIONS
        if prompt["direction"] == "out"
    )

    assert into_indices == tuple(
        range(parameters._TASK_INDEX_MAXIMA["box_pack_task_index"] + 1)
    )
    assert out_indices == (6, 7, 8, 9, 10, 11)


@pytest.mark.parametrize("index", [0, 2, 5])
def test_every_runnable_box_pack_prompt_index_is_accepted(index):
    typed, arguments = _validate({"box_pack_task_index": index})

    assert typed["box_pack_task_index"] == index
    assert arguments[arguments.index("--box-pack-task-index") + 1] == str(index)


@pytest.mark.parametrize("index", [-1, 6, 7, 11, 12])
def test_box_pack_out_and_unknown_indices_name_the_supported_set(index):
    with pytest.raises(
        launcher.ConfigError,
        match=(
            "simulation\\.parameters\\.box_pack_task_index supports only "
            "runnable into-box task indices 0, 1, 2, 3, 4, 5"
        ),
    ) as caught:
        _validate({"box_pack_task_index": index})

    assert caught.value.allowed == [0, 1, 2, 3, 4, 5]
    assert caught.value.got == index
