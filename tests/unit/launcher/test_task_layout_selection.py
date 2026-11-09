# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""A placed-object task must run the layout its profile selected.

Box packing stands its objects at authored poses, and the profile selects which
arrangement, not an ambient ``RS_*`` variable.

That is the quiet failure this pins. A second rig's table and reach are
different -- the shared layouts put their objects where another robot's body
stands -- so a rig needs its own arrangement. Without a config key the run
would load the other rig's layout and nothing would say so: the objects would
simply be somewhere the policy never sees them, and the score would be a real
number measuring the wrong scene.

So the selection has to travel config -> worker argv -> environment, and
preflight has to check the file the run will actually open rather than the
default it would have opened.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
from unittest import mock

import pytest
import yaml

from real2sim import launcher
from real2sim.config._types import ConfigError
from tests._helpers.launcher import base_config
from tests._helpers.repo import REPO_ROOT


#: parameter key -> worker option, for the overridable layouts.
LAYOUT_KEYS = {
    "box_pack_layout_json": "--box-pack-layout-json",
}


@pytest.fixture(autouse=True)
def _isolate_from_ambient_environment(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    for name in ("RS_BOX_PACK_LAYOUT_JSON",):
        monkeypatch.delenv(name, raising=False)


def _load(config: dict) -> dict:
    directory = Path(tempfile.mkdtemp(prefix=".layout-select-", dir=REPO_ROOT / "config"))
    try:
        path = directory / "probe.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with mock.patch.dict(
            os.environ, {"REAL2SIM_INPUTS_ROOT": str(REPO_ROOT)}, clear=False
        ):
            return launcher.load_rollout_config(path, timestamp="20260912_120000")
    finally:
        for child in sorted(directory.iterdir()):
            child.unlink()
        directory.rmdir()


def _cli_value(arguments, option):
    return arguments[arguments.index(option) + 1]


@pytest.mark.parametrize("parameter,option", sorted(LAYOUT_KEYS.items()))
def test_a_selected_layout_reaches_the_worker_argument(parameter, option):
    relative = "robosuite_env/obj_calib/layout_box_pack_duo_manual.json"
    config = copy.deepcopy(base_config())
    config["simulation"]["parameters"][parameter] = "{repo_root}/" + relative

    arguments = _load(config)["resolved"]["simulation"]["cli_args"]

    assert _cli_value(arguments, option) == str(REPO_ROOT / relative)


@pytest.mark.parametrize("parameter,option", sorted(LAYOUT_KEYS.items()))
def test_an_unselected_layout_stays_out_of_the_worker_argv(parameter, option):
    """Absent must mean "the default", not an empty argument.

    Every profile that predates these keys renders an argument vector without
    them, and those runs are what current scores were measured against.
    """
    config = copy.deepcopy(base_config())
    config["simulation"]["parameters"].pop(parameter, None)

    plan = _load(config)

    assert plan["resolved"]["simulation"]["parameters"][parameter] is None
    assert option not in plan["resolved"]["simulation"]["cli_args"]


@pytest.mark.parametrize("option", sorted(LAYOUT_KEYS.values()))
def test_the_worker_accepts_each_layout_option(option):
    """The argument the launcher renders has to be one the worker parses."""

    from robosuite_env.rollout import cli as worker_cli

    source = Path(worker_cli.__file__).read_text(encoding="utf-8")
    assert f'"{option}"' in source, f"the worker has no {option}"


def test_preflight_rejects_a_layout_that_does_not_exist(tmp_path):
    """The check must follow the override.

    Validating the default while the run loads something else is worse than
    not validating: it reports a healthy layout for a file the run never opens.
    """
    from real2sim.preflight import _validate_selected_layout
    from robosuite_env.tasks.box_pack import BOX_PACK_OBJECT_NAMES

    missing = tmp_path / "absent.json"
    plan = {
        "repo_root": str(REPO_ROOT),
        "resolved": {
            "robot": "fr3_duo",
            "simulation": {"parameters": {"box_pack_layout_json": str(missing)}},
        },
    }

    with pytest.raises(ConfigError) as caught:
        _validate_selected_layout(
            plan,
            "box_pack",
            expected_objects=BOX_PACK_OBJECT_NAMES,
            context="task.objects box_pack",
        )

    assert str(missing) in str(caught.value)


def test_preflight_falls_back_to_the_shared_layout():
    from robosuite_env.tasks import box_pack
    from robosuite_env.tasks.layout_selection import select_task_layout

    selection = select_task_layout("box_pack", None, robot="fr3_duo", environ={})

    assert selection.path == Path(box_pack.BOX_PACK_LAYOUT_PATH).resolve()
    assert selection.selected_by == "default"


def test_the_selected_layout_becomes_the_variable_the_task_reads():
    """The worker exports the selection; the task layer already reads it.

    ``robosuite_env/tasks/layout_selection.py`` resolves the box layout from
    ``RS_BOX_PACK_LAYOUT_JSON``. This pins the two halves to the same name, so
    a rename on either side fails here rather than silently reverting every run
    to the shared arrangement.
    """
    worker = (REPO_ROOT / "robosuite_env" / "rollout" / "build" / "model.py").read_text(
        encoding="utf-8"
    )
    selection = (REPO_ROOT / "robosuite_env" / "tasks" / "layout_selection.py").read_text(
        encoding="utf-8"
    )

    for name, reader in (("RS_BOX_PACK_LAYOUT_JSON", selection),):
        assert name in worker, f"the worker never exports {name}"
        assert name in reader, f"the task never reads {name}"
