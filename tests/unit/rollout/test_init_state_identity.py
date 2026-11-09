# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Init states are bound to the scene inputs and seated start they came from.

The deterministic cache name covers HEAD and source and also the layout,
positions and initial-pose JSON contents of every task. An explicit
``init_state`` does not go through the name, and the model signature does not
cover object poses, so it is checked against the seated start: a state saved
before a seating change would otherwise restore sunk starting poses.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from robosuite_env.rollout.state import init_identity
from robosuite_env.rollout.state import init_state
from robosuite_env.rollout.state.init_identity import (
    INIT_IDENTITY_KEY,
    InitIdentityError,
    identity_mismatches,
    init_state_identity,
    scene_input_identity,
    validate_init_state_identity,
)
from tests._helpers.worker_ast import (
    REPO_ROOT,
    RUNNER_SOURCE,
    _shell_function,
    _sim_thread_caps_definitions,
)

OBJ_CALIB = REPO_ROOT / "robosuite_env" / "obj_calib"
INIT_STATE_PATH = REPO_ROOT / "robosuite_env" / "rollout" / "state" / "init_state.py"
_OVERRIDES = (
    "RS_BOX_PACK_LAYOUT_JSON",
    "RS_BASE_PLACEMENT",
    "RS_CUP_XY",
    "RS_PLATE_XY",
    "INIT_STATE_JSON",
)
#: The worker parser requires one action source; the batch runner always
#: passes this one.
_WORKER_SOURCE = ("--policy-batch-server", "127.0.0.1:1")


@pytest.fixture(autouse=True)
def _no_layout_overrides(monkeypatch):
    for name in _OVERRIDES:
        monkeypatch.delenv(name, raising=False)


def _arguments(**overrides):
    values = {
        "objects": "cup_plate",
        "initial_arm_joints_json": "",
        "base_placement": "",
        "box_pack_task_index": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _copy(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def _edit_json(path: Path) -> None:
    """Change one value the way a re-authored layout would."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_edited_for_test"] = True
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# Cache identity: layout, positions and initial-pose contents.
# --------------------------------------------------------------------------


def _task_case(tmp_path: Path, monkeypatch, task: str):
    if task == "box_pack":
        layout = _copy(
            OBJ_CALIB / "layout_box_pack_duo_manual.json", tmp_path / "box.json"
        )
        monkeypatch.setenv("RS_BOX_PACK_LAYOUT_JSON", str(layout))
        return _arguments(objects="box_pack", box_pack_task_index=4), layout
    if task == "positions":
        positions = _copy(
            OBJ_CALIB / "scene_positions_duo.json", tmp_path / "positions.json"
        )
        return _arguments(objects=str(positions)), positions
    if task == "initial_arm_joints":
        pose = _copy(
            REPO_ROOT / "assets" / "initial_poses" / "fr3" / "duo" / "home_20260906.json",
            tmp_path / "initial.json",
        )
        return _arguments(objects="cup_plate", initial_arm_joints_json=str(pose)), pose
    raise AssertionError(task)


@pytest.mark.parametrize(
    "task",
    ("box_pack", "positions", "initial_arm_joints"),
)
def test_a_changed_layout_json_changes_the_identity(tmp_path, monkeypatch, task):
    arguments, data_file = _task_case(tmp_path, monkeypatch, task)
    before = scene_input_identity(arguments, robot="fr3_duo")["inputs_sha256"]
    assert scene_input_identity(arguments, robot="fr3_duo")["inputs_sha256"] == before

    original = data_file.read_bytes()
    _edit_json(data_file)
    assert scene_input_identity(arguments, robot="fr3_duo")["inputs_sha256"] != before

    data_file.write_bytes(original)
    assert scene_input_identity(arguments, robot="fr3_duo")["inputs_sha256"] == before


def test_named_tasks_resolve_the_layout_their_environment_loads(tmp_path, monkeypatch):
    """No file appears in the arguments; the module constant is followed."""

    from robosuite_env.tasks import box_pack

    layout = _copy(OBJ_CALIB / "layout_box_pack_duo_manual.json", tmp_path / "box.json")
    monkeypatch.setattr(box_pack, "BOX_PACK_LAYOUT_PATH", layout)
    identity = scene_input_identity(_arguments(objects="box_pack"), robot="fr3_duo")
    assert identity["inputs"]["manual_layout"]["path"] == str(layout)
    before = identity["inputs_sha256"]
    _edit_json(layout)
    assert (
        scene_input_identity(_arguments(objects="box_pack"), robot="fr3_duo")[
            "inputs_sha256"
        ]
        != before
    )


def test_the_identity_does_not_depend_on_checkout_location(tmp_path):
    first = _copy(OBJ_CALIB / "scene_positions_duo.json", tmp_path / "a" / "p.json")
    second = _copy(first, tmp_path / "b" / "elsewhere.json")
    assert (
        scene_input_identity(_arguments(objects=str(first)), robot="fr3_duo")[
            "inputs_sha256"
        ]
        == scene_input_identity(_arguments(objects=str(second)), robot="fr3_duo")[
            "inputs_sha256"
        ]
    )


def test_an_unreadable_scene_input_fails_closed(tmp_path):
    with pytest.raises(InitIdentityError, match="manual_layout is unreadable"):
        scene_input_identity(
            _arguments(objects="box_pack", box_pack_layout_json=str(tmp_path / "absent.json")),
            robot="fr3_duo",
        )


def _identity_cli(*worker_argv: str) -> str:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "robosuite_env.rollout.state.init_identity",
            "--",
            *worker_argv,
            *_WORKER_SOURCE,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip().splitlines()[-1]


def test_the_identity_cli_tracks_layout_contents(tmp_path):
    poses = _copy(
        OBJ_CALIB / "layout_box_pack_duo_manual.json", tmp_path / "poses.json"
    )
    argv = ("--robot", "fr3_duo", "--objects", "box_pack", "--box-pack-layout-json", str(poses))
    before = _identity_cli(*argv)
    assert before == scene_input_identity(
        _arguments(objects="box_pack", box_pack_layout_json=str(poses)), robot="fr3_duo"
    )["inputs_sha256"]
    _edit_json(poses)
    assert _identity_cli(*argv) != before


def _runner_sim_worker_definition() -> str:
    """Return the runner's own ``SIM_WORKER=(...)`` line, so the test runs its head."""

    definitions = re.findall(r"^SIM_WORKER=\(.*\)$", RUNNER_SOURCE, flags=re.MULTILINE)
    assert len(definitions) == 1, definitions
    return definitions[0]


def _cache_name(
    tmp_path: Path,
    worker_args: list[str],
    environment: dict[str, str],
    *,
    runner_selected_layout: Path | None = None,
):
    """Run the batch runner's own cache-naming function in dry-run mode."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    physics = tmp_path / "physics_contract.sha256"
    physics.write_text("physics\n", encoding="utf-8")
    script = "\n".join(
        (
            "set -euo pipefail",
            _sim_thread_caps_definitions(),
            _shell_function(RUNNER_SOURCE, "cache_tag"),
            _shell_function(RUNNER_SOURCE, "prepare_deterministic_init_state"),
            f'ROOT="{REPO_ROOT}"',
            f'PY="{sys.executable}"',
            "DETERMINISTIC_SIM=1",
            'INIT_STATE_JSON=""',
            "GRIP_TORQUE_MULT=0.5",
            'OBJECTS="${TEST_OBJECTS}"',
            'INSTRUCTION="test instruction"',
            'POLICY_IMAGE_EPISODE=""',
            "INIT_STATE_CACHE_SCHEMA=test_schema",
            f'PHYSICS_CONTRACT_FILE="{physics}"',
            "GRASP_MODE=rule",
            'EXTRA_SIM_ARGS_JSON="[]"',
            'EXTRA_SIM_ARGS=""',
            'GRIPPER_CALIBRATION=""',
            "DETERMINISTIC_SEED=0",
            "SIM_NUM_THREADS=1",
            f'STATE_ROOT="{tmp_path / "state"}"',
            "ROBOT_TAG=fr3_duo",
            "STATE_DIR_MODE=700",
            "STATE_FILE_MODE=600",
            "DRY_RUN=1",
            f'SHARD_RUNTIME_ROOT="{tmp_path / "shard"}"',
            "RENDER_ENABLED=0",
            "REAL2SIM_STRICT_LAUNCH=0",
            f'BG_DIRS=("{tmp_path / "scene"}")',
            "TAGS=(scene)",
            "PORTS=(1)",
            "GS_TOKENS=(token)",
            'read -r -a TEST_WORKER_ARGS <<< "${TEST_WORKER_ARGS_LINE}"',
            _runner_sim_worker_definition(),
            'COMMON_ARGS=("${SIM_WORKER[@]}" "${TEST_WORKER_ARGS[@]}")',
            'if [ -n "${RUNNER_SELECTED_LAYOUT:-}" ]; then',
            '  prepare_deterministic_init_state "${RUNNER_SELECTED_LAYOUT}" 0 0 > /dev/null',
            "else",
            "  prepare_deterministic_init_state > /dev/null",
            "fi",
            'printf "%s\\n" "${INIT_STATE_JSON}"',
        )
    )
    env = dict(os.environ)
    for name in _OVERRIDES:
        env.pop(name, None)
    env.update(environment)
    env["RUNNER_SELECTED_LAYOUT"] = (
        "" if runner_selected_layout is None else str(runner_selected_layout)
    )
    env["TEST_OBJECTS"] = worker_args[worker_args.index("--objects") + 1]
    env["TEST_WORKER_ARGS_LINE"] = " ".join([*worker_args, *_WORKER_SOURCE])
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_runner_selected_box_layouts_produce_distinct_cache_identities(tmp_path):
    first = OBJ_CALIB / "layout_box_pack_duo_manual.json"
    second = REPO_ROOT / "tests" / "fixtures" / "layout_box_pack_flat.json"
    worker = ["--robot", "fr3_duo", "--objects", "box_pack"]
    one = _cache_name(
        tmp_path / "one", worker, {}, runner_selected_layout=first
    )
    two = _cache_name(
        tmp_path / "two", worker, {}, runner_selected_layout=second
    )
    assert one.returncode == 0, one.stderr
    assert two.returncode == 0, two.stderr
    assert Path(one.stdout.strip()).name != Path(two.stdout.strip()).name


@pytest.mark.parametrize("task", ("box_pack_argument", "box_pack"))
def test_the_batch_cache_name_changes_with_layout_contents(tmp_path, task):
    if task == "box_pack_argument":
        data = _copy(
            OBJ_CALIB / "layout_box_pack_duo_manual.json", tmp_path / "poses.json"
        )
        # Same path in the argument vector every time: only the contents move.
        worker_args = ["--robot", "fr3_duo", "--objects", "box_pack", "--box-pack-layout-json", str(data)]
        environment: dict[str, str] = {}
    else:
        data = _copy(OBJ_CALIB / "layout_box_pack_duo_manual.json", tmp_path / "box.json")
        # A named task: no layout file appears in OBJECTS or the arguments.
        worker_args = ["--robot", "fr3_duo", "--objects", "box_pack"]
        environment = {"RS_BOX_PACK_LAYOUT_JSON": str(data)}

    first = _cache_name(tmp_path / "run1", worker_args, environment)
    assert first.returncode == 0, first.stderr
    repeat = _cache_name(tmp_path / "run2", worker_args, environment)
    assert repeat.returncode == 0, repeat.stderr
    assert Path(first.stdout.strip()).name == Path(repeat.stdout.strip()).name

    _edit_json(data)
    changed = _cache_name(tmp_path / "run3", worker_args, environment)
    assert changed.returncode == 0, changed.stderr
    assert Path(changed.stdout.strip()).name != Path(first.stdout.strip()).name


def test_the_batch_cache_refuses_unresolvable_scene_inputs(tmp_path):
    missing = tmp_path / "absent.json"
    completed = _cache_name(
        tmp_path / "run",
        ["--robot", "fr3_duo", "--objects", "box_pack", "--box-pack-layout-json", str(missing)],
        {},
    )
    assert completed.returncode != 0
    assert "cannot resolve the deterministic init-state scene inputs" in completed.stderr


# --------------------------------------------------------------------------
# Loading: an explicit init state must match the current run's seated start.
# --------------------------------------------------------------------------

#: The shape CupPlateLiftGW._seat_layout_placements reports.
SEATED = (
    {"object": "cup", "support": "table", "z_before_m": 0.0, "z_seated_m": 0.034, "lift_m": 0.034},
    {"object": "plate", "support": "table", "z_before_m": 0.0, "z_seated_m": 0.034, "lift_m": 0.034},
)


def _seating(**changes):
    rows = [dict(row) for row in SEATED]
    for row in rows:
        row.update(changes.get(row["object"], {}))
    return rows


def _positions(tmp_path: Path) -> Path:
    return _copy(OBJ_CALIB / "scene_positions_duo.json", tmp_path / "positions.json")


def _state(arguments, seating_report):
    return SimpleNamespace(
        ROBOT="fr3_duo",
        MODEL_SIGNATURE="signature",
        A=arguments,
        env=SimpleNamespace(layout_seating_report=seating_report),
        # Rebound to post-drive poses on the writing path only; the identity
        # must not read it.
        _READY_OBJECT_QPOS={"cup": (7, 6, [9.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0])},
        GRIPPER_CALIBRATION_REPORT={"enabled": False},
        # A deliberately impossible model: reaching the dimension checks
        # proves the identity gate let the state through.
        m=SimpleNamespace(nq=-99, nv=0, nu=0, na=0, nmocap=0),
    )


def _write_payload(path: Path, identity) -> Path:
    payload = {"robot": "fr3_duo", "model_signature": "signature"}
    if identity is not None:
        payload[INIT_IDENTITY_KEY] = identity
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_an_explicit_init_state_recorded_with_different_object_poses_is_refused(tmp_path):
    """A state from before seating: the cup started 4 mm lower than now."""

    arguments = _arguments(objects=str(_positions(tmp_path)))
    recorded = init_state_identity(
        arguments, robot="fr3_duo", seating_report=_seating(cup={"z_seated_m": 0.030})
    )
    path = _write_payload(tmp_path / "old_init_state.json", recorded)

    with pytest.raises(SystemExit, match=r"cup seated z differs by -4\.000 mm"):
        init_state._load_sim_init_state(_state(arguments, _seating()), str(path))


def test_an_explicit_init_state_seated_on_another_support_is_refused(tmp_path):
    arguments = _arguments(objects=str(_positions(tmp_path)))
    recorded = init_state_identity(
        arguments, robot="fr3_duo", seating_report=_seating(cup={"support": "plate"})
    )
    path = _write_payload(tmp_path / "init_state.json", recorded)

    with pytest.raises(SystemExit, match="cup was seated on 'plate', this run seats it on 'table'"):
        init_state._load_sim_init_state(_state(arguments, _seating()), str(path))


def test_an_explicit_init_state_from_another_layout_is_refused(tmp_path):
    positions = _positions(tmp_path)
    arguments = _arguments(objects=str(positions))
    recorded = init_state_identity(arguments, robot="fr3_duo", seating_report=_seating())
    path = _write_payload(tmp_path / "init_state.json", recorded)
    _edit_json(positions)

    with pytest.raises(SystemExit, match="objects_positions: written with"):
        init_state._load_sim_init_state(_state(arguments, _seating()), str(path))


def test_a_legacy_init_state_without_an_identity_is_refused(tmp_path):
    arguments = _arguments(objects=str(_positions(tmp_path)))
    path = _write_payload(tmp_path / "legacy.json", None)

    with pytest.raises(SystemExit, match="records no scene-input identity"):
        init_state._load_sim_init_state(_state(arguments, _seating()), str(path))


def test_a_matching_init_state_passes_the_identity_gate(tmp_path):
    arguments = _arguments(objects=str(_positions(tmp_path)))
    recorded = init_state_identity(arguments, robot="fr3_duo", seating_report=_seating())
    path = _write_payload(tmp_path / "init_state.json", recorded)

    with pytest.raises(SystemExit, match="nq mismatch"):
        init_state._load_sim_init_state(_state(arguments, _seating()), str(path))


def test_float_noise_is_the_same_seated_start(tmp_path):
    arguments = _arguments(objects=str(_positions(tmp_path)))
    recorded = init_state_identity(arguments, robot="fr3_duo", seating_report=_seating())
    current = init_state_identity(
        arguments,
        robot="fr3_duo",
        seating_report=list(reversed(_seating(cup={"z_seated_m": 0.034 + 1e-9}))),
    )
    assert identity_mismatches(recorded, current) == []


@pytest.mark.parametrize(
    ("current_report", "message"),
    (
        (_seating()[:1], "seated objects differ"),
        (None, "seated objects: written with 2 seated objects, this run has no seating"),
    ),
)
def test_a_different_seated_object_set_is_refused(tmp_path, current_report, message):
    arguments = _arguments(objects=str(_positions(tmp_path)))
    recorded = init_state_identity(arguments, robot="fr3_duo", seating_report=_seating())
    current = init_state_identity(
        arguments, robot="fr3_duo", seating_report=current_report
    )
    with pytest.raises(InitIdentityError, match=message):
        validate_init_state_identity(
            {INIT_IDENTITY_KEY: recorded}, current, source="init-state x"
        )


def test_an_edited_identity_record_is_detected(tmp_path):
    arguments = _arguments(objects=str(_positions(tmp_path)))
    recorded = init_state_identity(arguments, robot="fr3_duo", seating_report=_seating())
    current = json.loads(json.dumps(recorded))
    recorded["inputs"]["objects_positions"]["sha256"] = "0" * 64
    failures = identity_mismatches(recorded, current)
    assert any("does not match its inputs" in failure for failure in failures)


def test_the_identity_reads_the_environment_seating_report(tmp_path):
    arguments = _arguments(objects=str(_positions(tmp_path)))
    state = _state(arguments, _seating())
    identity = init_state._current_init_identity(state)
    assert [row["object"] for row in identity["seated_objects"]] == ["cup", "plate"]
    assert "_READY_OBJECT_QPOS" not in json.dumps(identity)

    state.env.layout_seating_report = [{"object": "cup"}]
    with pytest.raises(SystemExit, match="scene identity is unavailable"):
        init_state._current_init_identity(state)


def test_every_written_init_state_records_the_identity():
    tree = ast.parse(INIT_STATE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_sim_state_payload"
    )
    assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "payload"
            and isinstance(target.slice, ast.Name)
            and target.slice.id == "INIT_IDENTITY_KEY"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    call = assignments[0].value
    assert isinstance(call, ast.Call) and call.func.id == "_current_init_identity"
    assert init_identity.INIT_IDENTITY_KEY == INIT_IDENTITY_KEY
