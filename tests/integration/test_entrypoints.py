# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Integration checks for public command-line entrypoints."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import sys
from unittest import mock

import pytest
import yaml

from real2sim import cli as real2sim_cli
from real2sim import launcher
from tests._helpers.launcher import base_config as _base_config


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "robosuite_env"

CONTRACT_MODULES = ("assets", "scene", "worker")


@pytest.mark.parametrize("module_name", CONTRACT_MODULES)
def test_canonical_contract_clis_keep_help_entrypoints(
    module_name: str,
) -> None:
    module = f"robosuite_env.contracts.{module_name}"
    completed = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


# Started with -m by scripts/run_policy_batch.sh, the post-hoc renderer and the
# documented commands; none of them puts the checkout on sys.path any more.
MODULE_ENTRYPOINTS = (
    "robosuite_env.sim",
    "robosuite_env.tools.make_rollout_grid_video",
    "robosuite_env.tools.rerender_third_person",
    "robosuite_env.tools.rewrite_saved_policy_obs_video",
)


@pytest.mark.parametrize("module", MODULE_ENTRYPOINTS)
def test_module_entrypoints_run_from_the_installed_package(module, tmp_path) -> None:
    """Outside the checkout and without PYTHONPATH, only the install finds them."""

    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    # The rollout worker behind robosuite_env.sim imports mujoco at module
    # level and defaults MUJOCO_GL to egl, so even its --help needs a loadable
    # libEGL, which the hosted CI runner lacks. --help renders nothing and this
    # test checks where the modules import from, so run them with GL disabled,
    # as the CI MuJoCo CPU step does.
    environment["MUJOCO_GL"] = "disable"
    completed = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


def test_task_resources_keep_the_central_runtime_paths():
    pytest.importorskip("robosuite")

    from robosuite_env.paths import ASSETS_ROOT, OBJECT_CALIBRATION_ROOT
    from robosuite_env.tasks import objects
    from robosuite_env.robots.base_placement import default_base_placement_path

    assert objects.ASSETS_ROOT == ASSETS_ROOT
    assert default_base_placement_path("fr3").parent == OBJECT_CALIBRATION_ROOT


def test_rollout_cli_defaults_and_flags():
    from robosuite_env.rollout.cli import parse

    with mock.patch.object(sys, "argv", ["sim.py", "--grasp-test"]):
        arguments = parse()
    assert Path(arguments.grip_cfg).is_file()
    assert arguments.joint_ik_nullspace_gain == 0.2
    assert arguments.fr3_joint_kp is None
    assert arguments.fr3_joint_kd is None
    assert arguments.policy_action_clamp is True
    assert arguments.policy_pose_repr == "delta"
    assert arguments.desk == "scene"
    assert "robots/yubi/control/configs" in arguments.grip_cfg

    with mock.patch.object(
        sys, "argv", ["sim.py", "--grasp-test", "--no-policy-action-clamp"]
    ):
        arguments = parse()
    assert arguments.policy_action_clamp is False

    with mock.patch.object(
        sys,
        "argv",
        ["sim.py", "--grasp-test", "--policy-pose-repr", "anchored"],
    ):
        arguments = parse()
    assert arguments.policy_pose_repr == "anchored"

    with mock.patch.object(
        sys,
        "argv",
        [
            "sim.py",
            "--grasp-test",
            "--fr3-joint-kp",
            "400",
            "--fr3-joint-kd",
            "80",
        ],
    ):
        arguments = parse()
    assert arguments.fr3_joint_kp == 400.0
    assert arguments.fr3_joint_kd == 80.0

    for retired in ("--arm-ctrl", "--ee-ctrl"):
        with mock.patch.object(sys, "argv", ["sim.py", "--grasp-test", retired, "x"]):
            with pytest.raises(SystemExit) as raised:
                parse()
        assert raised.value.code == 2


def test_direct_policy_cli_requires_matching_config():
    from robosuite_env.rollout.cli import parse

    with mock.patch.object(sys, "argv", ["sim.py", "--policy", "/tmp/checkpoint"]):
        with pytest.raises(SystemExit) as raised:
            parse()
    assert raised.value.code == 2
    with mock.patch.object(
        sys,
        "argv",
        [
            "sim.py",
            "--policy",
            "/tmp/checkpoint",
            "--config",
            "pi05_yubi_pickplace",
        ],
    ):
        arguments = parse()
    assert arguments.config == "pi05_yubi_pickplace"


def test_rollout_cli_help_groups_and_advanced_help(capsys):
    from robosuite_env.rollout.cli import parse

    groups = (
        "Official/launcher",
        "Source",
        "Robot & IK",
        "Gripper",
        "Timing",
        "Scene/3DGS",
        "Objects & init-state",
        "Render/Output",
        "Success criteria",
        "Debug/Dump",
    )
    with pytest.raises(SystemExit) as raised:
        parse(["--help"])
    assert raised.value.code == 0
    standard_help = capsys.readouterr().out
    for title in groups:
        assert f"{title}:" in standard_help
    for option in ("--franka", "--grip-friction", "--grasp-hand"):
        assert option not in standard_help

    with pytest.raises(SystemExit) as raised:
        parse(["--help-all"])
    assert raised.value.code == 0
    full_help = capsys.readouterr().out
    for title in (*groups, "Grasp-test script"):
        assert f"{title}:" in full_help
    for option in ("--franka", "--grip-friction", "--grasp-hand"):
        assert option in full_help


def test_rollout_cli_reparses_recorded_worker_argv():
    from robosuite_env.rollout.cli import parse

    arguments = parse(
        [
            "--franka",
            "--policy-batch-server",
            "127.0.0.1:8765",
            "--policy-batch-client-id",
            "scene_run_01",
            "--policy-batch-timeout-sec",
            "7200",
            "--frames",
            "1",
            "--rf-overlap",
            "0",
            "--rf-overlap-sampler",
            "single",
            "--rf-overlap-prev-weight",
            "0.5",
            "--rf-overlap-step-shift",
            "0",
            "--gs-background",
            "/scenes/panorama",
            "--gs-port",
            "9900",
            "--gs-server-token",
            "deadbeef",
            "--render-lock",
            "/tmp/lock",
            "--out",
            "/runs/scene/run_01/scene_run_01.mp4",
            "--logdir",
            "/runs/scene/run_01/logs",
            "--summary-path",
            "/runs/scene/run_01/sim_summary.json",
        ]
    )
    assert arguments.robot == "fr3"
    assert arguments.policy_batch_server == "127.0.0.1:8765"
    assert arguments.rf_overlap_prev_weight == 0.5
    assert arguments.rf_overlap_step_shift == 0

from tests._helpers.worker_ast import *  # noqa: F403

def test_sim_module_resolves_worker_from_this_checkout(tmp_path) -> None:
    """A shadow package must not replace the worker the runner starts.

    scripts/run_policy_batch.sh starts ``python -m robosuite_env.sim`` from
    the checkout root, which puts that root first on ``sys.path``. The probe
    starts the module the same way, with the original ``runpy.run_module``,
    while recording where ``sim`` resolves the worker it hands to runpy.
    """

    shadow = tmp_path / "robosuite_env" / "rollout"
    shadow.mkdir(parents=True)
    (tmp_path / "robosuite_env" / "__init__.py").write_text("")
    (shadow / "__init__.py").write_text("")
    (shadow / "worker.py").write_text(
        "raise SystemExit('shadow worker executed')\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(tmp_path)
    probe = """
import importlib.util
import json
from pathlib import Path
import runpy
import sys

def resolve_worker(name, *, run_name=None, alter_sys=False):
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise SystemExit(f"worker module is unresolved: {name}")
    print(json.dumps({
        "module": name,
        "origin": str(Path(spec.origin).resolve()),
        "run_name": run_name,
        "alter_sys": alter_sys,
    }))
    return {}

start_module = runpy.run_module
runpy.run_module = resolve_worker
sys.path[0] = str(Path.cwd())  # what -m puts first; -c puts ""
try:
    start_module("robosuite_env.sim", run_name="__main__", alter_sys=True)
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise
print(json.dumps({"sim": str(Path(importlib.util.find_spec("robosuite_env.sim").origin).resolve())}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    worker, sim = (json.loads(line) for line in completed.stdout.splitlines())
    assert worker == {
        "module": "robosuite_env.rollout.worker",
        "origin": str(WORKER_PATH.resolve()),
        "run_name": "__main__",
        "alter_sys": True,
    }
    assert sim == {"sim": str(SIM_PATH.resolve())}

def test_worker_derives_package_and_repository_roots_after_move() -> None:
    wanted = ("_ROLLOUT_DIR", "_DIR", "REPO")
    assignments: dict[str, ast.Assign] = {}
    for node in WORKER_TREE.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            assignments[target.id] = node

    assert tuple(name for name in wanted if name in assignments) == wanted
    root_module = ast.Module(
        body=[assignments[name] for name in wanted],
        type_ignores=[],
    )
    namespace = {"os": os, "__file__": os.fspath(WORKER_PATH)}
    exec(compile(ast.fix_missing_locations(root_module), str(WORKER_PATH), "exec"), namespace)

    assert Path(namespace["_ROLLOUT_DIR"]).resolve() == WORKER_PATH.parent
    assert Path(namespace["_DIR"]).resolve() == PACKAGE_ROOT
    assert Path(namespace["REPO"]).resolve() == REPO_ROOT


@pytest.fixture
def _launcher_entrypoint_environment(monkeypatch, isolated_environ, request):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    request.instance.isolated_environ = isolated_environ


@pytest.mark.usefixtures("_launcher_entrypoint_environment")
class LauncherEntrypointTest(unittest.TestCase):
    def _write_config(self, root: Path, config: dict) -> Path:
        config_dir = root / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        path = config_dir / "rollout.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def test_cli_dry_run_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._write_config(root, _base_config())
            plan = launcher.load_rollout_config(
                config_path,
                repo_root=root,
                timestamp="20260730_120000",
            )
            with mock.patch.object(
                real2sim_cli,
                "load_rollout_config",
                return_value=plan,
            ):
                self.assertEqual(
                    launcher.main(
                        [
                            "--config",
                            str(config_path),
                            "--dry-run",
                            "--env-file",
                            str(self.isolated_environ),
                        ]
                    ),
                    0,
                )
            self.assertFalse(Path(plan["run_root"]).exists())


def test_artifact_cli_inspects_release_tree_without_manifest_entry(tmp_path):
    artifact = tmp_path / "release-tree"
    artifact.mkdir()
    (artifact / "one.bin").write_bytes(b"one")
    (artifact / "two.bin").write_bytes(b"two")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "real2sim.artifacts",
            "--inspect-path",
            str(artifact),
            "--inspect-kind",
            "tree",
            "--json",
        ],
        cwd=REPO_ROOT,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["kind"] == "tree"
    assert payload["size"] == 6
    assert payload["files"] == 2
    assert len(payload["sha256"]) == 64
