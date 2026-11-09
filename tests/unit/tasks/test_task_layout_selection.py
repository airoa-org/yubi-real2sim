# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Box runs take their hand-authored layout from the profile.

Each layout is placed for one scene (Duo boards for ``franka_duo``; the
Flat fixture layout for ``franka_flat``). A profile parameter selects the
file (the launcher refuses ``RS_*`` variables), and the launcher preflight, the worker, the environment, the init-state
identity and the run summary all resolve it through
``robosuite_env/tasks/layout_selection.py``:

* order: the profile parameter, then the harness-only override, then the
  default (the Duo layout for box); both set to
  different files is refused;
* a layout placed for another scene than the run's is refused, in preflight and
  in the environment;
* the run summary names the file and the sha256 of the bytes the environment
  validated, and the init-state identity follows the parameter.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from real2sim import launcher, preflight
from real2sim.config._types import ConfigError
from robosuite_env.tasks import box_pack
from robosuite_env.tasks.layout_selection import (
    LAYOUT_OVERRIDE_ENV,
    LAYOUT_PARAMETERS,
    LayoutSelectionError,
    select_task_layout,
)
from tests._helpers.launcher import base_config

REPO_ROOT = Path(__file__).resolve().parents[3]
OBJ_CALIB = REPO_ROOT / "robosuite_env" / "obj_calib"
BOX_DUO = OBJ_CALIB / "layout_box_pack_duo_manual.json"
BOX_FLAT = REPO_ROOT / "tests" / "fixtures" / "layout_box_pack_flat.json"
DUO_SCENE = "franka_duo"
FLAT_SCENE = "franka_flat"
LAYOUT_FLAGS = {"--box-pack-layout-json"}
#: What each task places when the profile selects nothing: the Duo boards.
DEFAULT_DUO_LAYOUT = {"box_pack": BOX_DUO}

@pytest.fixture(autouse=True)
def _no_harness_overrides(monkeypatch):
    for name in LAYOUT_OVERRIDE_ENV.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("RS_BOX_PACK_OBJECT", raising=False)


@pytest.fixture
def launcher_env(monkeypatch, isolated_environ):
    # A Slurm allocation exports CUDA_VISIBLE_DEVICES, which a profile must
    # allowlist; the unit base config does not.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return isolated_environ


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _copy(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


# ---------------------------------------------------------------------------
# One order for all three tasks.


@pytest.mark.parametrize("task", tuple(LAYOUT_PARAMETERS))
def test_the_profile_parameter_wins_then_the_override_then_the_default(task, tmp_path, monkeypatch):
    profile = tmp_path / "profile.json"
    override = tmp_path / "override.json"

    default = select_task_layout(task, None, robot="fr3_duo")
    assert default.selected_by == "default"
    assert default.path == DEFAULT_DUO_LAYOUT[task]

    monkeypatch.setenv(LAYOUT_OVERRIDE_ENV[task], str(override))
    chosen = select_task_layout(task, "", robot="fr3_duo")
    assert (chosen.path, chosen.selected_by) == (override, "harness_override")

    monkeypatch.delenv(LAYOUT_OVERRIDE_ENV[task])
    chosen = select_task_layout(task, str(profile), robot="fr3_duo")
    assert (chosen.path, chosen.selected_by) == (profile, "profile")
    assert chosen.parameter == LAYOUT_PARAMETERS[task]


@pytest.mark.parametrize("task", tuple(LAYOUT_PARAMETERS))
def test_a_profile_layout_and_a_different_override_are_refused_not_ranked(task, tmp_path, monkeypatch):
    monkeypatch.setenv(LAYOUT_OVERRIDE_ENV[task], str(tmp_path / "override.json"))
    with pytest.raises(LayoutSelectionError, match="selected twice"):
        select_task_layout(task, str(tmp_path / "profile.json"), robot="fr3_duo")


@pytest.mark.parametrize("task", tuple(LAYOUT_PARAMETERS))
def test_the_workers_export_of_the_profile_file_is_the_same_selection(task, tmp_path, monkeypatch):
    # The worker exports the profile's file under the override name so the
    # environment reads it; a later resolution sees both, naming one file.
    profile = tmp_path / "layouts" / "profile.json"
    profile.parent.mkdir()
    profile.write_text("{}", encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(profile)
    monkeypatch.setenv(LAYOUT_OVERRIDE_ENV[task], str(alias))

    chosen = select_task_layout(task, str(profile), robot="fr3_duo")

    assert (chosen.path, chosen.selected_by) == (profile, "profile")


def test_defaults_are_the_duo_layouts():
    for robot in ("fr3_duo", "fr3"):
        assert select_task_layout("box_pack", None, robot=robot).path == BOX_DUO
    # Callers catch RuntimeError, so the selection error is one.
    assert issubclass(LayoutSelectionError, RuntimeError)


# ---------------------------------------------------------------------------
# Profile -> resolved config -> worker argv -> worker flag.


def _resolve(root: Path, config: dict) -> dict:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "rollout.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return launcher.load_rollout_config(path, repo_root=root, timestamp="20260910_120000")


@pytest.mark.parametrize("parameter", tuple(LAYOUT_PARAMETERS.values()))
def test_a_layout_parameter_renders_one_absolute_worker_flag(parameter, tmp_path, monkeypatch, launcher_env):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    monkeypatch.setenv("REAL2SIM_INPUTS_ROOT", str(inputs))
    config = base_config()
    environment = config["reproducibility"]["environment"]
    environment["allowlist"].append("REAL2SIM_INPUTS_ROOT")
    environment["required"].append("REAL2SIM_INPUTS_ROOT")
    flag = "--" + parameter.replace("_", "-")

    config["simulation"]["parameters"][parameter] = "{repo_root}/layouts/../layouts/a.json"
    plan = _resolve(tmp_path / "repo", config)
    arguments = plan["resolved"]["simulation"]["cli_args"]
    expected = str(tmp_path / "repo" / "layouts" / "a.json")
    assert plan["resolved"]["simulation"]["parameters"][parameter] == expected
    assert arguments.count(flag) == 1
    assert arguments[arguments.index(flag) + 1] == expected

    config["simulation"]["parameters"][parameter] = "{inputs_root}/layouts/b.json"
    plan = _resolve(tmp_path / "repo2", config)
    arguments = plan["resolved"]["simulation"]["cli_args"]
    assert arguments[arguments.index(flag) + 1] == str(inputs / "layouts" / "b.json")


def test_a_profile_without_layout_parameters_keeps_its_argument_vector(tmp_path, launcher_env):
    plan = _resolve(tmp_path, base_config())
    assert not LAYOUT_FLAGS & set(plan["resolved"]["simulation"]["cli_args"])
    assert all(
        plan["resolved"]["simulation"]["parameters"][parameter] is None
        for parameter in LAYOUT_PARAMETERS.values()
    )


def test_box_layout_list_resolves_and_reaches_runner_without_a_scalar_flag(
    tmp_path, launcher_env
):
    config = base_config()
    config["simulation"]["parameters"]["box_pack_layout_json"] = [
        "{repo_root}/layouts/first.json",
        "{repo_root}/layouts/last.json",
    ]
    plan = _resolve(tmp_path, config)
    expected = [
        str(tmp_path / "layouts" / "first.json"),
        str(tmp_path / "layouts" / "last.json"),
    ]
    assert plan["resolved"]["simulation"]["parameters"]["box_pack_layout_json"] == expected
    assert "--box-pack-layout-json" not in plan["resolved"]["simulation"]["cli_args"]
    environment = launcher.build_runner_environment(plan)
    assert json.loads(environment["BOX_PACK_LAYOUT_JSON_LIST"]) == expected


def test_scalar_box_layout_keeps_the_existing_flag_and_clears_ambient_list(
    tmp_path, monkeypatch, launcher_env
):
    config = base_config()
    monkeypatch.setenv("BOX_PACK_LAYOUT_JSON_LIST", '["/ambient/layout.json"]')
    config["reproducibility"]["environment"]["allowlist"].append(
        "BOX_PACK_LAYOUT_JSON_LIST"
    )
    config["simulation"]["parameters"]["box_pack_layout_json"] = (
        "{repo_root}/layouts/only.json"
    )
    plan = _resolve(tmp_path, config)
    arguments = plan["resolved"]["simulation"]["cli_args"]
    assert arguments[arguments.index("--box-pack-layout-json") + 1] == str(
        tmp_path / "layouts" / "only.json"
    )
    environment = launcher.build_runner_environment(plan)
    assert environment["BOX_PACK_LAYOUT_JSON_LIST"] == ""
    assert json.loads(environment["EXTRA_SIM_ARGS_JSON"]) == arguments


def test_box_layout_list_must_be_nonempty_and_match_repeats(tmp_path, launcher_env):
    for layouts, message in (([], "non-empty"), (["one.json"], "must equal runner.repeats")):
        config = base_config()
        config["simulation"]["parameters"]["box_pack_layout_json"] = layouts
        with pytest.raises(ConfigError, match=message):
            _resolve(tmp_path / f"case-{len(layouts)}", config)


def test_box_layout_list_refuses_even_a_robot_mapped_init_state(
    tmp_path, launcher_env
):
    config = base_config()
    config["simulation"]["parameters"]["box_pack_layout_json"] = [
        "first.json",
        "last.json",
    ]
    config["simulation"]["init_state"] = {
        "fr3": None,
        "fr3_duo": None,
    }
    with pytest.raises(ConfigError, match="requires simulation.init_state: null"):
        _resolve(tmp_path, config)


def test_box_layout_list_selects_run_01_and_the_last_run(tmp_path):
    layouts = [tmp_path / "first.json", tmp_path / "last.json"]
    first = select_task_layout(
        "box_pack", layouts, robot="fr3", environ={}, run_number=1
    )
    last = select_task_layout(
        "box_pack", layouts, robot="fr3", environ={}, run_number=2
    )
    assert (first.path, first.layout_index) == (layouts[0], 0)
    assert (last.path, last.layout_index) == (layouts[1], 1)


def test_a_profile_cannot_allowlist_a_harness_layout_override(tmp_path, launcher_env):
    config = base_config()
    config["reproducibility"]["environment"]["allowlist"].append("RS_BOX_PACK_LAYOUT_JSON")
    with pytest.raises(ConfigError, match="harness-only layout overrides: RS_BOX_PACK_LAYOUT_JSON"):
        _resolve(tmp_path, config)


def test_the_worker_flags_carry_only_the_profile_value(monkeypatch):
    from robosuite_env.rollout.cli import parse

    source = ["--robot", "fr3_duo", "--policy-batch-server", "127.0.0.1:1"]
    for name in LAYOUT_OVERRIDE_ENV.values():
        monkeypatch.setenv(name, "/harness/override.json")
    arguments = parse([*source, "--objects", "box_pack"])
    assert arguments.box_pack_layout_json == ""
    arguments = parse([*source, "--objects", "box_pack", "--box-pack-layout-json", "/profile/box.json"])
    assert arguments.box_pack_layout_json == "/profile/box.json"


# ---------------------------------------------------------------------------
# Preflight validates the layout the profile resolves, on the run's scenes.


def _preflight_plan(tmp_path, task, *, robot, scenes, **parameters):
    task_fields = {"objects": task}
    if task == "box_pack":
        parameters.setdefault("box_pack_task_index", 2)
        task_fields.update(
            instruction=box_pack.box_pack_instruction(parameters["box_pack_task_index"]),
            goal_condition=box_pack.BOX_PACK_GOAL_CONDITION,
            score_condition=box_pack.BOX_PACK_SCORE_CONDITION,
        )
    return {
        "repo_root": REPO_ROOT,
        "resolved": {
            "robot": robot,
            "task": task_fields,
            "simulation": {"parameters": parameters},
            "scenes": [{"tag": tag, "path": f"/scenes/{tag}"} for tag in scenes],
        },
    }


@pytest.mark.parametrize(
    ("task", "parameters", "scene", "placed_for"),
    [
        pytest.param("box_pack", {}, FLAT_SCENE, DUO_SCENE, id="default-box-on-flat"),
        pytest.param(
            "box_pack", {"box_pack_layout_json": str(BOX_FLAT)}, DUO_SCENE, FLAT_SCENE,
            id="flat-box-on-duo",
        ),
    ],
)
def test_preflight_refuses_a_layout_placed_for_another_scene(tmp_path, task, parameters, scene, placed_for):
    plan = _preflight_plan(tmp_path, task, robot="fr3_duo", scenes=(scene,), **parameters)
    with pytest.raises(ConfigError, match=rf"placed for scene '{placed_for}', but this run's scene is '{scene}'"):
        preflight._validate_task_objects(plan)


def test_preflight_accepts_the_flat_box_layout_on_flat_and_refuses_it_on_another_rig(tmp_path):
    """The Flat box layout is Flat's alone; another rig fails closed in preflight."""

    preflight._validate_task_objects(
        _preflight_plan(
            tmp_path, "box_pack", robot="fr3", scenes=(FLAT_SCENE,), box_pack_layout_json=str(BOX_FLAT)
        )
    )
    with pytest.raises(ConfigError, match="placed for the 'fr3' rig, not 'fr3_duo'"):
        preflight._validate_task_objects(
            _preflight_plan(
                tmp_path, "box_pack", robot="fr3_duo", scenes=(FLAT_SCENE,), box_pack_layout_json=str(BOX_FLAT)
            )
        )


def test_preflight_validates_every_box_layout_list_entry(tmp_path):
    missing = tmp_path / "missing-layout.json"
    plan = _preflight_plan(
        tmp_path,
        "box_pack",
        robot="fr3",
        scenes=(FLAT_SCENE,),
        box_pack_layout_json=[str(BOX_FLAT), str(missing)],
    )
    with pytest.raises(ConfigError, match=r"box_pack_layout_json\[1\].*does not exist"):
        preflight._validate_task_objects(plan)


def test_preflight_refuses_a_layout_parameter_for_another_task(tmp_path):
    plan = _preflight_plan(tmp_path, "cup_plate", robot="fr3_duo", scenes=(DUO_SCENE,), box_pack_layout_json=str(BOX_DUO))
    with pytest.raises(ConfigError, match="box_pack_layout_json selects the box_pack layout, but task.objects is 'cup_plate'"):
        preflight._validate_task_objects(plan)


def test_preflight_refuses_a_missing_layout_file(tmp_path):
    plan = _preflight_plan(
        tmp_path, "box_pack", robot="fr3", scenes=(FLAT_SCENE,), box_pack_layout_json=str(tmp_path / "absent.json")
    )
    with pytest.raises(ConfigError, match="simulation.parameters.box_pack_layout_json does not exist"):
        preflight._validate_task_objects(plan)


# ---------------------------------------------------------------------------
# The environment loads the selected file and refuses another scene's layout.


class _FakeObject:
    def __init__(self, name):
        self.name = name
        self.root_body = f"{name}_main"
        self.joints = [f"{name}_joint0"]


@pytest.fixture
def task_envs(monkeypatch):
    pytest.importorskip("robosuite")
    from robosuite_env.tasks import objects as objects_module
    from robosuite_env.tasks.envs.box_pack import configure_box_pack_task

    for name in box_pack.BOX_PACK_OBJECT_NAMES:
        monkeypatch.setitem(objects_module.OBJECT_ASSET_REGISTRY, name, _FakeObject)
    return {"box_pack": configure_box_pack_task}


def _env(scene, robot=None):
    return types.SimpleNamespace(
        table_offset=np.zeros(3),
        gw_scene_tag=scene,
        base_placement=types.SimpleNamespace(robot=robot),
    )


ENV_CASES = [
    pytest.param("box_pack", BOX_FLAT, id="box"),
]


def test_the_worker_hands_every_rig_its_scene_tag_from_the_scene_contract():
    from tests._helpers.worker_ast import BUILD_MODEL_TREE

    run = next(
        node
        for node in BUILD_MODEL_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    # A top-level branch of run(): taken for Duo and Flat alike.
    branches = [
        node
        for node in run.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "state.A.desk == 'scene'"
    ]
    assert len(branches) == 1

    def assignments(body):
        module = ast.Module(body=list(body), type_ignores=[])
        return {ast.unparse(node) for node in ast.walk(module) if isinstance(node, ast.Assign)}

    assert "state.EnvCls.gw_scene_tag = scene_tag(state.SCENE_CONTRACT_REPORT['name'])" in assignments(
        branches[0].body
    )
    assert "state.EnvCls.gw_scene_tag = None" in assignments(branches[0].orelse)


@pytest.mark.parametrize(("task", "layout"), ENV_CASES)
def test_the_env_loads_the_selected_layout_on_its_scene(task, layout, task_envs, monkeypatch):
    monkeypatch.setenv(LAYOUT_OVERRIDE_ENV[task], str(layout))
    env = _env(FLAT_SCENE, robot="fr3")

    task_envs[task](env)

    assert env.task_layout_record == {
        "path": str(layout.resolve()),
        "sha256": _sha256(layout),
        "scene": FLAT_SCENE,
    }


@pytest.mark.parametrize("task", ("box_pack",))
def test_the_env_refuses_a_layout_placed_for_another_scene(task, task_envs):
    # No selection: the Duo default, on a Flat environment.
    with pytest.raises(LayoutSelectionError, match=f"placed for scene '{DUO_SCENE}', but this run's scene is '{FLAT_SCENE}'"):
        task_envs[task](_env(FLAT_SCENE))
    with pytest.raises(LayoutSelectionError, match="names no validated scene"):
        task_envs[task](_env(None))
    env = _env(DUO_SCENE, robot="fr3_duo")
    task_envs[task](env)
    assert env.task_layout_record["sha256"] == _sha256(DEFAULT_DUO_LAYOUT[task])


# ---------------------------------------------------------------------------
# Provenance names the file and digest the environment loaded.


def _state(task, env, **arguments):
    values = {parameter: "" for parameter in LAYOUT_PARAMETERS.values()}
    values.update(arguments)
    return types.SimpleNamespace(A=types.SimpleNamespace(objects=task, **values), env=env, ROBOT="fr3")


def test_provenance_names_the_profile_layout_the_env_loaded(tmp_path, task_envs, monkeypatch):
    from robosuite_env.rollout.task_layout import loaded_task_layout

    layout = _copy(BOX_FLAT, tmp_path / "profile" / "box.json")
    # What the worker does with --box-pack-layout-json before building the env.
    monkeypatch.setenv("RS_BOX_PACK_LAYOUT_JSON", str(layout))
    env = _env(FLAT_SCENE, robot="fr3")
    task_envs["box_pack"](env)

    record = loaded_task_layout(_state("box_pack", env, box_pack_layout_json=str(layout)))

    assert record == {
        "task": "box_pack",
        "parameter": "box_pack_layout_json",
        "configured": str(layout),
        "selected_by": "profile",
        "path": str(layout.resolve()),
        "sha256": _sha256(layout),
        "scene": FLAT_SCENE,
        "running_scene": FLAT_SCENE,
    }
    assert record["sha256"] != _sha256(BOX_DUO)


def test_provenance_names_the_default_only_when_nothing_is_selected(task_envs):
    from robosuite_env.rollout.task_layout import loaded_task_layout

    env = _env(DUO_SCENE, robot="fr3_duo")
    task_envs["box_pack"](env)
    record = loaded_task_layout(_state("box_pack", env))
    assert (record["path"], record["sha256"], record["selected_by"]) == (
        str(BOX_DUO.resolve()),
        _sha256(BOX_DUO),
        "default",
    )
    assert set(record) == {
        "task",
        "parameter",
        "configured",
        "selected_by",
        "path",
        "sha256",
        "scene",
        "running_scene",
    }
    assert loaded_task_layout(_state("cup_plate", env)) is None


def test_scalar_box_layout_summary_keeps_the_base_task_layout_shape(
    task_envs, monkeypatch
):
    from robosuite_env.rollout.task_layout import loaded_task_layout

    monkeypatch.setenv("RS_BOX_PACK_LAYOUT_JSON", str(BOX_FLAT))
    env = _env(FLAT_SCENE, robot="fr3")
    task_envs["box_pack"](env)
    record = loaded_task_layout(
        _state("box_pack", env, box_pack_layout_json=str(BOX_FLAT))
    )
    assert record == {
        "task": "box_pack",
        "parameter": "box_pack_layout_json",
        "configured": str(BOX_FLAT),
        "selected_by": "profile",
        "path": str(BOX_FLAT.resolve()),
        "sha256": _sha256(BOX_FLAT),
        "scene": FLAT_SCENE,
        "running_scene": FLAT_SCENE,
    }


def test_box_layout_summary_records_list_index_and_sweep_offsets(
    tmp_path, task_envs, monkeypatch
):
    from robosuite_env.rollout.task_layout import loaded_task_layout

    layout = _copy(BOX_FLAT, tmp_path / "layout.json")
    payload = json.loads(layout.read_text(encoding="utf-8"))
    payload["source"]["sweep"] = {
        "pen_offset_mm": {"dx": 25, "dy": -25},
        "box_offset_mm": [5, 10],
    }
    layout.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("RS_BOX_PACK_LAYOUT_JSON", str(layout))
    env = _env(FLAT_SCENE, robot="fr3")
    task_envs["box_pack"](env)

    record = loaded_task_layout(
        _state(
            "box_pack",
            env,
            box_pack_layout_json=str(layout),
            box_pack_layout_index=1,
        )
    )
    assert record["index"] == 1
    assert record["path"] == str(layout.resolve())
    assert record["sha256"] == _sha256(layout)
    assert record["pen_offset_mm"] == {"dx": 25.0, "dy": -25.0}
    assert record["box_offset_mm"] == {"dx": 5.0, "dy": 10.0}
    assert record["pen_offset_mm_reason"] is None
    assert record["box_offset_mm_reason"] is None


def test_malformed_list_sweep_offset_is_refused_in_preflight_and_safe_in_summary(
    tmp_path, task_envs, monkeypatch
):
    from robosuite_env.rollout.task_layout import loaded_task_layout

    layout = _copy(BOX_FLAT, tmp_path / "odd-layout.json")
    payload = json.loads(layout.read_text(encoding="utf-8"))
    payload["source"]["sweep"] = {"pen_offset_mm": ["odd", 2]}
    layout.write_text(json.dumps(payload), encoding="utf-8")
    plan = _preflight_plan(
        tmp_path,
        "box_pack",
        robot="fr3",
        scenes=(FLAT_SCENE,),
        box_pack_layout_json=[str(layout), str(BOX_FLAT)],
    )
    with pytest.raises(ConfigError, match="dx and dy must be finite numbers"):
        preflight._validate_task_objects(plan)

    monkeypatch.setenv("RS_BOX_PACK_LAYOUT_JSON", str(layout))
    env = _env(FLAT_SCENE, robot="fr3")
    task_envs["box_pack"](env)
    record = loaded_task_layout(
        _state(
            "box_pack",
            env,
            box_pack_layout_json=str(layout),
            box_pack_layout_index=0,
        )
    )
    assert record["pen_offset_mm"] is None
    assert "finite numbers" in record["pen_offset_mm_reason"]


def test_huge_list_sweep_offset_is_bounded_in_preflight_and_safe_in_summary(
    tmp_path, task_envs, monkeypatch
):
    from robosuite_env.rollout.task_layout import loaded_task_layout

    layout = _copy(BOX_FLAT, tmp_path / "huge-layout.json")
    payload = json.loads(layout.read_text(encoding="utf-8"))
    payload["source"]["sweep"] = {"pen_offset_mm": [10**400, 0]}
    layout.write_text(json.dumps(payload), encoding="utf-8")
    plan = _preflight_plan(
        tmp_path,
        "box_pack",
        robot="fr3",
        scenes=(FLAT_SCENE,),
        box_pack_layout_json=[str(layout), str(BOX_FLAT)],
    )
    with pytest.raises(ConfigError, match="within ±1000 mm"):
        preflight._validate_task_objects(plan)

    monkeypatch.setenv("RS_BOX_PACK_LAYOUT_JSON", str(layout))
    env = _env(FLAT_SCENE, robot="fr3")
    task_envs["box_pack"](env)
    record = loaded_task_layout(
        _state(
            "box_pack",
            env,
            box_pack_layout_json=str(layout),
            box_pack_layout_index=0,
        )
    )
    assert record["pen_offset_mm"] is None
    assert "within ±1000 mm" in record["pen_offset_mm_reason"]


def test_runtime_seal_inputs_include_every_listed_layout(tmp_path, launcher_env):
    layouts = []
    for name in ("first.json", "last.json"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        layouts.append(path)
    config = base_config()
    config["simulation"]["parameters"]["box_pack_layout_json"] = [
        str(path) for path in layouts
    ]
    plan = _resolve(tmp_path / "repo", config)
    sealed_inputs = set(launcher.runtime_config_input_files(plan))
    assert set(layouts) <= sealed_inputs


def test_provenance_refuses_a_layout_that_changed_after_loading(tmp_path, task_envs, monkeypatch):
    from robosuite_env.rollout.task_layout import loaded_task_layout

    layout = _copy(BOX_FLAT, tmp_path / "box.json")
    monkeypatch.setenv("RS_BOX_PACK_LAYOUT_JSON", str(layout))
    env = _env(FLAT_SCENE, robot="fr3")
    task_envs["box_pack"](env)
    layout.write_bytes(layout.read_bytes() + b"\n")

    with pytest.raises(RuntimeError, match="changed after the environment loaded it"):
        loaded_task_layout(_state("box_pack", env, box_pack_layout_json=str(layout)))


def test_provenance_refuses_a_selection_the_env_did_not_load(task_envs):
    from robosuite_env.rollout.task_layout import loaded_task_layout

    env = _env(DUO_SCENE, robot="fr3_duo")
    task_envs["box_pack"](env)
    with pytest.raises(RuntimeError, match="but the environment loaded"):
        loaded_task_layout(_state("box_pack", env, box_pack_layout_json=str(BOX_FLAT)))
    with pytest.raises(RuntimeError, match="recorded no layout"):
        loaded_task_layout(_state("box_pack", types.SimpleNamespace()))


# ---------------------------------------------------------------------------
# The init-state identity follows the profile parameter.


IDENTITY_CASES = [
    pytest.param("box_pack", BOX_DUO, BOX_FLAT, id="box"),
]


@pytest.mark.parametrize(("task", "default", "other"), IDENTITY_CASES)
def test_the_init_identity_changes_when_only_the_layout_parameter_changes(task, default, other):
    from robosuite_env.rollout.state.init_identity import scene_input_identity
    from tests.unit.rollout.test_init_state_identity import _arguments

    parameter = LAYOUT_PARAMETERS[task]
    unset = scene_input_identity(_arguments(objects=task), robot="fr3_duo")
    named_default = scene_input_identity(_arguments(objects=task, **{parameter: str(default)}), robot="fr3_duo")
    selected = scene_input_identity(_arguments(objects=task, **{parameter: str(other)}), robot="fr3_duo")

    assert unset["inputs"]["manual_layout"]["path"] == str(default)
    assert selected["inputs"]["manual_layout"] == {"path": str(other), "sha256": _sha256(other)}
    # Content-addressed: naming the default is the default's identity.
    assert named_default["inputs_sha256"] == unset["inputs_sha256"]
    assert selected["inputs_sha256"] != unset["inputs_sha256"]


def test_two_list_selected_box_layouts_have_different_init_state_identity():
    from robosuite_env.rollout.state.init_identity import scene_input_identity
    from tests.unit.rollout.test_init_state_identity import _arguments

    layouts = [BOX_DUO, BOX_FLAT]
    identities = []
    for run_number in (1, 2):
        selected = select_task_layout(
            "box_pack", layouts, robot="fr3_duo", environ={}, run_number=run_number
        )
        identities.append(
            scene_input_identity(
                _arguments(
                    objects="box_pack",
                    box_pack_layout_json=str(selected.path),
                ),
                robot="fr3_duo",
            )["inputs_sha256"]
        )
    assert identities[0] != identities[1]


def test_the_init_identity_refuses_a_conflicting_harness_override(monkeypatch):
    from robosuite_env.rollout.state.init_identity import InitIdentityError, scene_input_identity
    from tests.unit.rollout.test_init_state_identity import _arguments

    monkeypatch.setenv("RS_BOX_PACK_LAYOUT_JSON", str(BOX_DUO))
    with pytest.raises(InitIdentityError, match="selected twice"):
        scene_input_identity(_arguments(objects="box_pack", box_pack_layout_json=str(BOX_FLAT)), robot="fr3_duo")


def test_the_batch_cache_name_changes_with_only_the_layout_parameter(tmp_path):
    from tests.unit.rollout.test_init_state_identity import _cache_name

    worker = ["--robot", "fr3_duo", "--objects", "box_pack"]
    unset = _cache_name(tmp_path / "unset", worker, {})
    duo = _cache_name(tmp_path / "duo", [*worker, "--box-pack-layout-json", str(BOX_DUO)], {})
    flat = _cache_name(tmp_path / "flat", [*worker, "--box-pack-layout-json", str(BOX_FLAT)], {})
    for completed in (unset, duo, flat):
        assert completed.returncode == 0, completed.stderr

    name = {key: Path(value.stdout.strip()).name for key, value in (("unset", unset), ("duo", duo), ("flat", flat))}
    assert name["duo"] == name["unset"]
    assert name["flat"] != name["unset"]


# ---------------------------------------------------------------------------
# Box Pen: the profile's task index is what makes the env target the pen.


def test_a_box_pen_profile_selects_the_pen_through_its_task_index(tmp_path, launcher_env):
    # No release profile runs box packing; this profile sets what the removed
    # box_pen release profiles set for the task (objects, index 2, wording).
    config = base_config()
    config["task"].update(
        objects="box_pack",
        instruction=box_pack.box_pack_instruction(2),
        goal_condition="object_in_box",
        score_condition="object_in_box_containment",
    )
    config["simulation"]["parameters"]["box_pack_task_index"] = 2
    plan = _resolve(tmp_path, config)
    arguments = plan["resolved"]["simulation"]["cli_args"]

    assert arguments.count("--box-pack-task-index") == 1
    index = int(arguments[arguments.index("--box-pack-task-index") + 1])
    assert index == 2
    assert box_pack.box_pack_target_object(index) == "pen"
    assert box_pack.box_pack_hand(index) == "R"
    assert plan["resolved"]["task"]["instruction"] == box_pack.box_pack_instruction(index)
    assert not any(value.startswith("RS_BOX_PACK") for value in plan["resolved"]["reproducibility"]["environment"])


def test_the_worker_exports_the_prompt_object_from_the_task_index():
    from tests._helpers.worker_ast import BUILD_MODEL_TREE

    statements = {ast.unparse(node) for node in ast.walk(BUILD_MODEL_TREE) if isinstance(node, ast.Assign)}
    assert "_index = int(state.A.box_pack_task_index)" in statements
    assert "os.environ['RS_BOX_PACK_OBJECT'] = box_pack_target_object(_index)" in statements


def test_the_box_env_targets_the_object_the_worker_exported(task_envs, monkeypatch):
    monkeypatch.setenv("RS_BOX_PACK_OBJECT", box_pack.box_pack_target_object(2))
    env = _env(DUO_SCENE, robot="fr3_duo")

    task_envs["box_pack"](env)

    assert env.cube.name == "pen"
    assert env.box_pack_target_object == "pen"
    assert {obj.name for obj in env.extra_task_objects} == {"box", "tape", "lan"}


# ---------------------------------------------------------------------------
# Rigs share a scene without sharing a layout: source.robots names them.


def _flat_box_payload(**source):
    payload = json.loads(BOX_FLAT.read_text(encoding="utf-8"))
    payload["source"].update(source)
    return payload


@pytest.mark.parametrize(
    ("robots", "message"),
    [
        ("fr3", "source.robots must be a non-empty list"),
        ([], "source.robots must be a non-empty list"),
        (["fr3", ""], r"source.robots\[1\] must be a non-empty string"),
        (["fr3", 3], r"source.robots\[1\] must be a non-empty string"),
        (["fr3", "fr3"], "names a rig more than once"),
    ],
)
def test_source_robots_is_validated(robots, message):
    from robosuite_env.tasks import layouts

    with pytest.raises(layouts.LayoutError, match=message):
        layouts.validate_manual_layout(
            _flat_box_payload(robots=robots), expected_objects=box_pack.BOX_PACK_OBJECT_NAMES
        )


def test_source_robots_is_optional_and_kept_when_given():
    from robosuite_env.tasks import layouts

    absent = _flat_box_payload()
    absent["source"].pop("robots")
    canonical = layouts.validate_manual_layout(absent, expected_objects=box_pack.BOX_PACK_OBJECT_NAMES)
    assert "robots" not in canonical["source"]
    shared = layouts.validate_manual_layout(
        _flat_box_payload(robots=["fr3", "fr3_duo"]), expected_objects=box_pack.BOX_PACK_OBJECT_NAMES
    )
    assert shared["source"]["robots"] == ["fr3", "fr3_duo"]


def test_a_layout_naming_no_rig_loads_on_any_rig_and_a_named_one_only_on_its_own():
    from robosuite_env.tasks.layout_selection import require_layout_rig

    selection = select_task_layout("box_pack", str(BOX_FLAT), robot=None, environ={})
    for robot in ("fr3", "fr3_duo", None):
        require_layout_rig({"source": {"scene": FLAT_SCENE}}, robot=robot, selection=selection)
    shared = {"source": {"scene": FLAT_SCENE, "robots": ["fr3", "fr3_duo"]}}
    require_layout_rig(shared, robot="fr3_duo", selection=selection)
    with pytest.raises(LayoutSelectionError, match="placed for the rigs 'fr3', 'fr3_duo', not 'another_rig'"):
        require_layout_rig(shared, robot="another_rig", selection=selection)
    with pytest.raises(LayoutSelectionError, match="names no rig"):
        require_layout_rig(shared, robot=None, selection=selection)


def test_the_flat_box_layout_loads_on_flat_and_another_rigs_env_refuses_it(task_envs, monkeypatch):
    monkeypatch.setenv("RS_BOX_PACK_LAYOUT_JSON", str(BOX_FLAT))
    env = _env(FLAT_SCENE, robot="fr3")
    task_envs["box_pack"](env)
    assert env.task_layout_record["sha256"] == _sha256(BOX_FLAT)
    with pytest.raises(LayoutSelectionError, match="placed for the 'fr3' rig, not 'fr3_duo'"):
        task_envs["box_pack"](_env(FLAT_SCENE, robot="fr3_duo"))


def test_the_flat_box_layout_names_flat_alone():
    payload = json.loads(BOX_FLAT.read_text(encoding="utf-8"))
    assert payload["source"]["robots"] == ["fr3"]
