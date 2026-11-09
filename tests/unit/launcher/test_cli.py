# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Command-parsing, reporting, and exit-code contracts for ``real2sim-run``.

The launcher module is a compatibility facade; argument parsing lives in
:mod:`real2sim.cli` and process orchestration in :mod:`real2sim.runner.execute`.
These checks bind the facade to those owners and pin the observable behavior an
operator sees: the exit code, the JSON printed by ``--check-runtime``, and the
warnings printed for retired environment variables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pytest
import yaml

from tests._helpers.launcher import base_config as _base_config


REPO_ROOT = Path(__file__).resolve().parents[3]

from real2sim import cli as real2sim_cli  # noqa: E402
from real2sim import launcher  # noqa: E402
from real2sim.config import schema as config_schema  # noqa: E402
from real2sim.config._types import ConfigError  # noqa: E402
from real2sim.gpu_binding import (  # noqa: E402
    GPU_BINDINGS_ENV,
    resolve_gpu_contract,
)
from real2sim.runner import execute as runner_execute  # noqa: E402
from real2sim.runtime_manifest import (  # noqa: E402
    RuntimeManifestError,
    RuntimeSourceDriftError,
)


@pytest.fixture(autouse=True)
def _isolate_launcher_environment(monkeypatch, isolated_environ, request):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    if request.instance is not None:
        request.instance.isolated_environ = isolated_environ


class LauncherFacadeTest(unittest.TestCase):
    def test_facade_reexports_the_owning_implementations(self):
        self.assertIs(launcher.main, real2sim_cli.main)
        self.assertIs(launcher.build_arg_parser, real2sim_cli.build_arg_parser)
        self.assertIs(launcher.execute, runner_execute.execute)
        self.assertIs(launcher.load_rollout_config, config_schema.load_rollout_config)
        self.assertIs(launcher.ConfigError, ConfigError)

    def test_console_entry_point_still_targets_the_facade(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('real2sim-run = "real2sim.launcher:main"', pyproject)


class LauncherCommandLineTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        environment = dict(os.environ)
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        patcher = mock.patch.dict(os.environ, environment, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write_config(self, root: Path, config: dict) -> Path:
        config_dir = root / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        path = config_dir / "rollout.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def _plan(self, root: Path, config: dict | None = None):
        config_path = self._write_config(root, config or _base_config())
        plan = launcher.load_rollout_config(
            config_path,
            repo_root=root,
            timestamp="20260730_120000",
        )
        return config_path, plan

    def test_validation_modes_are_mutually_exclusive(self):
        parser = launcher.build_arg_parser()
        with self.assertRaises(SystemExit) as raised:
            parser.parse_args(
                ["--config", "config/rollout.yaml", "--dry-run", "--check-runtime"]
            )
        self.assertEqual(raised.exception.code, 2)

    def test_timestamp_must_be_a_run_timestamp(self):
        parser = launcher.build_arg_parser()
        with self.assertRaises(SystemExit) as raised:
            parser.parse_args(
                ["--config", "config/rollout.yaml", "--timestamp", "2026-07-30"]
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(
            parser.parse_args(
                ["--config", "config/rollout.yaml", "--timestamp", "20260730_120000"]
            ).timestamp,
            "20260730_120000",
        )

    def test_dry_run_table_format_renders_the_resolved_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, plan = self._plan(root)
            with mock.patch.object(
                real2sim_cli, "load_rollout_config", return_value=plan
            ), mock.patch("sys.stdout", new_callable=_Capture) as stdout:
                exit_code = launcher.main(
                    [
                        "--config",
                        str(config_path),
                        "--dry-run",
                        "--format",
                        "table",
                        "--env-file",
                        str(self.isolated_environ),
                    ]
                )

        self.assertEqual(exit_code, 0)
        rendered = stdout.text
        self.assertIn("run_root", rendered)
        self.assertNotIn("{", rendered.splitlines()[0])

    def test_dry_run_json_format_is_machine_readable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, plan = self._plan(root)
            with mock.patch.object(
                real2sim_cli, "load_rollout_config", return_value=plan
            ), mock.patch("sys.stdout", new_callable=_Capture) as stdout:
                exit_code = launcher.main(
                    [
                        "--config",
                        str(config_path),
                        "--dry-run",
                        "--env-file",
                        str(self.isolated_environ),
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIsInstance(json.loads(stdout.text), dict)

    def test_check_runtime_prints_the_stable_json_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, plan = self._plan(root)
            # Source capture is covered by provenance tests; this fixture only
            # supplies a config and tests the CLI reporting contract.
            with mock.patch.object(
                real2sim_cli, "capture_launch_runtime_generation"
            ), mock.patch.object(
                real2sim_cli, "capture_runtime_selection"
            ), mock.patch.object(
                real2sim_cli, "runtime_source_selection"
            ), mock.patch.object(
                real2sim_cli, "runtime_config_input_files", return_value=[]
            ), mock.patch.object(
                real2sim_cli, "load_rollout_config", return_value=plan
            ), mock.patch.object(
                real2sim_cli, "validate_launch_preflight", return_value={}
            ) as preflight, mock.patch.object(
                real2sim_cli, "prepare_run_manifest", return_value=None
            ) as manifest, mock.patch(
                "sys.stdout", new_callable=_Capture
            ) as stdout:
                exit_code = launcher.main(
                    [
                        "--config",
                        str(config_path),
                        "--check-runtime",
                        "--env-file",
                        str(self.isolated_environ),
                    ]
                )

        self.assertEqual(exit_code, 0)
        preflight.assert_called_once_with(plan)
        manifest.assert_called_once()
        report = json.loads(stdout.text)
        self.assertEqual(
            sorted(report),
            [
                "check",
                "config",
                "config_sha256",
                "run_manifest_ready",
                "run_root",
                "status",
            ],
        )
        self.assertEqual(report["check"], "runtime")
        self.assertEqual(report["status"], "ok")
        self.assertIs(report["run_manifest_ready"], False)
        self.assertEqual(report["run_root"], str(plan["run_root"]))
        self.assertFalse(Path(plan["run_root"]).exists())

    def test_check_runtime_never_leaks_the_artifact_content_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, plan = self._plan(root)
            report = {"_artifact_verification": object(), "contracts": {}}
            with mock.patch.object(
                real2sim_cli, "capture_launch_runtime_generation"
            ), mock.patch.object(
                real2sim_cli, "capture_runtime_selection"
            ), mock.patch.object(
                real2sim_cli, "runtime_source_selection"
            ), mock.patch.object(
                real2sim_cli, "runtime_config_input_files", return_value=[]
            ), mock.patch.object(
                real2sim_cli, "load_rollout_config", return_value=plan
            ), mock.patch.object(
                real2sim_cli, "validate_launch_preflight", return_value=report
            ), mock.patch.object(
                real2sim_cli,
                "prepare_run_manifest",
                side_effect=RuntimeError("manifest failed"),
            ), mock.patch("sys.stdout", new_callable=_Capture):
                with self.assertRaises(RuntimeError):
                    launcher.main(
                        [
                            "--config",
                            str(config_path),
                            "--check-runtime",
                            "--env-file",
                            str(self.isolated_environ),
                        ]
                    )

        self.assertNotIn("_artifact_verification", report)

    def test_configuration_errors_exit_with_code_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _base_config()
            config["schema_version"] = 1
            config_path = self._write_config(root, config)
            with mock.patch("sys.stderr", new_callable=_Capture) as stderr:
                exit_code = launcher.main(
                    [
                        "--config",
                        str(config_path),
                        "--dry-run",
                        "--env-file",
                        str(self.isolated_environ),
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertIn("configuration error:", stderr.text)

    def test_retired_input_environment_variables_are_reported_and_ignored(self):
        retired = sorted(launcher._LEGACY_INPUT_ENV_NAMES)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, plan = self._plan(root)
            with mock.patch.dict(os.environ, {retired: "/tmp/retired"}, clear=False), \
                    mock.patch.object(
                        real2sim_cli, "load_rollout_config", return_value=plan
                    ), \
                    mock.patch("sys.stdout", new_callable=_Capture), \
                    mock.patch("sys.stderr", new_callable=_Capture) as stderr:
                exit_code = launcher.main(
                    [
                        "--config",
                        str(config_path),
                        "--dry-run",
                        "--env-file",
                        str(self.isolated_environ),
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn(
            "warning: ignoring retired input path variable(s)", stderr.text
        )
        self.assertIn(retired, stderr.text)


class _Capture:
    """Minimal ``sys.stdout``/``sys.stderr`` replacement that records text."""

    def __init__(self) -> None:
        self._parts: list[str] = []

    def write(self, value: str) -> int:
        self._parts.append(value)
        return len(value)

    def flush(self) -> None:
        return None

    @property
    def text(self) -> str:
        return "".join(self._parts)


def _execute_plan(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "rollout.yaml"
    path.write_text(yaml.safe_dump(_base_config(), sort_keys=False), encoding="utf-8")
    return launcher.load_rollout_config(
        path, repo_root=tmp_path, timestamp="20260730_120000"
    )


def _cuda_execute_plan(tmp_path: Path, visible_devices: str):
    plan = _execute_plan(tmp_path)
    plan["resolved"]["reproducibility"]["accelerator_profile"]["kind"] = "cuda"
    plan["resolved"]["reproducibility"]["environment"][
        "CUDA_VISIBLE_DEVICES"
    ] = visible_devices
    plan["resolved"]["runtime"].update(
        {
            "gs_cuda_devices": "0",
            "policy_cuda_device": "0",
            "sim_cuda_devices": "0",
        }
    )
    return plan


@pytest.fixture()
def stubbed_execute(monkeypatch):
    """Neutralize every side effect of ``execute`` except its exit-code rules."""

    monkeypatch.setattr(
        runner_execute, "validate_launch_preflight", lambda plan: {"gpu_bindings": None}
    )
    monkeypatch.setattr(
        runner_execute, "prepare_run_manifest", lambda plan, report: None
    )
    monkeypatch.setattr(
        runner_execute, "persist_config_provenance", lambda plan, manifest: None
    )
    monkeypatch.setattr(
        runner_execute, "build_runner_environment", lambda plan, gpu_bindings=None: {}
    )
    return runner_execute


def _stub_runner(monkeypatch, returncode: int):
    completed = mock.Mock()
    completed.returncode = returncode
    run = mock.Mock(return_value=completed)
    monkeypatch.setattr(runner_execute.subprocess, "run", run)
    return run


@pytest.mark.parametrize("visible_devices", ("5", "3,5"))
def test_execute_seals_unsharded_fixture_gpu_binding_before_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    visible_devices: str,
) -> None:
    plan = _cuda_execute_plan(tmp_path, visible_devices)
    unsealed_environment = runner_execute.build_runner_environment(plan)
    binding = resolve_gpu_contract(
        unsealed_environment,
        nvidia={
            "gpus": [
                {"index": str(index), "uuid": f"GPU-{index}"}
                for index in range(8)
            ]
        },
    )
    preflight_report = {"contracts": {}, "doctor": None}
    monkeypatch.setattr(
        runner_execute,
        "validate_launch_preflight",
        lambda _plan: preflight_report,
    )
    resolve = mock.Mock(return_value=binding)
    monkeypatch.setattr(
        runner_execute,
        "resolve_gpu_contract",
        resolve,
        raising=False,
    )
    prepare = mock.Mock(return_value=None)
    monkeypatch.setattr(runner_execute, "prepare_run_manifest", prepare)
    monkeypatch.setattr(runner_execute, "persist_config_provenance", mock.Mock())
    run = _stub_runner(monkeypatch, 0)
    monkeypatch.setattr(
        runner_execute,
        "summarize_run",
        lambda _plan: (0, {"total_runs": int(plan["expected_rollouts"])}, {}),
    )

    assert runner_execute.execute(plan) == 0

    resolve.assert_called_once_with(unsealed_environment)
    manifest_report = prepare.call_args.args[1]
    assert manifest_report["gpu_bindings"] == binding
    runner_environment = run.call_args.kwargs["env"]
    assert json.loads(runner_environment[GPU_BINDINGS_ENV]) == binding


def test_execute_preserves_preflight_sealed_gpu_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _cuda_execute_plan(tmp_path, "5")
    environment = runner_execute.build_runner_environment(plan)
    binding = resolve_gpu_contract(
        environment,
        nvidia={"gpus": [{"index": "5", "uuid": "GPU-5"}]},
    )
    preflight_report = {
        "contracts": {},
        "doctor": {"status": "ok"},
        "gpu_bindings": binding,
    }
    monkeypatch.setattr(
        runner_execute,
        "validate_launch_preflight",
        lambda _plan: preflight_report,
    )
    resolve = mock.Mock(side_effect=AssertionError("replaced preflight binding"))
    monkeypatch.setattr(
        runner_execute,
        "resolve_gpu_contract",
        resolve,
        raising=False,
    )
    prepare = mock.Mock(return_value=None)
    monkeypatch.setattr(runner_execute, "prepare_run_manifest", prepare)
    monkeypatch.setattr(runner_execute, "persist_config_provenance", mock.Mock())
    run = _stub_runner(monkeypatch, 0)
    monkeypatch.setattr(
        runner_execute,
        "summarize_run",
        lambda _plan: (0, {"total_runs": int(plan["expected_rollouts"])}, {}),
    )

    assert runner_execute.execute(plan) == 0

    resolve.assert_not_called()
    assert prepare.call_args.args[1]["gpu_bindings"] == binding
    assert (
        json.loads(run.call_args.kwargs["env"][GPU_BINDINGS_ENV])
        == binding
    )


def test_execute_returns_the_runner_exit_code_first(
    tmp_path, monkeypatch, stubbed_execute
):
    plan = _execute_plan(tmp_path)
    _stub_runner(monkeypatch, 3)
    monkeypatch.setattr(
        runner_execute, "summarize_run", lambda plan: (4, {"total_runs": 0}, {})
    )

    assert runner_execute.execute(plan) == 3


def test_execute_returns_the_summary_exit_code_when_the_runner_succeeds(
    tmp_path, monkeypatch, stubbed_execute
):
    plan = _execute_plan(tmp_path)
    _stub_runner(monkeypatch, 0)
    monkeypatch.setattr(
        runner_execute, "summarize_run", lambda plan: (4, {"total_runs": 2}, {})
    )

    assert runner_execute.execute(plan) == 4


def test_execute_reports_a_short_rollout_count(tmp_path, monkeypatch, stubbed_execute):
    plan = _execute_plan(tmp_path)
    _stub_runner(monkeypatch, 0)
    monkeypatch.setattr(
        runner_execute, "summarize_run", lambda plan: (0, {"total_runs": 1}, {})
    )

    assert runner_execute.execute(plan) == 5


def test_execute_returns_zero_only_for_a_complete_run(
    tmp_path, monkeypatch, stubbed_execute
):
    plan = _execute_plan(tmp_path)
    _stub_runner(monkeypatch, 0)
    monkeypatch.setattr(
        runner_execute,
        "summarize_run",
        lambda plan: (0, {"total_runs": int(plan["expected_rollouts"])}, {}),
    )

    assert runner_execute.execute(plan) == 0


def test_execute_reports_a_runner_that_cannot_start(
    tmp_path, monkeypatch, stubbed_execute
):
    plan = _execute_plan(tmp_path)
    monkeypatch.setattr(
        runner_execute.subprocess,
        "run",
        mock.Mock(side_effect=OSError("Exec format error")),
    )
    monkeypatch.setattr(
        runner_execute, "summarize_run", lambda plan: (0, {"total_runs": 0}, {})
    )

    assert runner_execute.execute(plan) == 127


def test_execute_seals_and_rechecks_the_runtime_source_manifest(
    tmp_path, monkeypatch, stubbed_execute
):
    plan = _execute_plan(tmp_path)
    manifest_path = tmp_path / "runtime_source_manifest.json"
    monkeypatch.setattr(
        runner_execute,
        "seal_runtime_source_manifest",
        lambda plan, baseline: (manifest_path, "a" * 64),
    )
    verify = mock.Mock()
    monkeypatch.setattr(runner_execute, "verify_runtime_manifest", verify)
    environments = []

    def capture_environment(plan, gpu_bindings=None):
        environment: dict[str, str] = {}
        environments.append(environment)
        return environment

    monkeypatch.setattr(
        runner_execute, "build_runner_environment", capture_environment
    )
    _stub_runner(monkeypatch, 0)
    monkeypatch.setattr(
        runner_execute,
        "summarize_run",
        lambda plan: (0, {"total_runs": int(plan["expected_rollouts"])}, {}),
    )

    assert runner_execute.execute(plan, runtime_source_baseline={"generation": 1}) == 0
    verify.assert_called_once_with(manifest_path, expected_sha256="a" * 64)
    assert environments[0][runner_execute.RUNTIME_SOURCE_MANIFEST_ENV] == str(
        manifest_path
    )
    assert (
        environments[0][runner_execute.RUNTIME_SOURCE_MANIFEST_SHA256_ENV] == "a" * 64
    )


def test_execute_still_fails_when_sealing_itself_breaks(
    tmp_path, monkeypatch, stubbed_execute
):
    # Only drift is survivable. A malformed baseline, an unreadable config or a
    # manifest that cannot be written are sealing failures, and starting an
    # unsealed run over one of those would skip verification for the rest of
    # the batch on the strength of a bug.
    plan = _execute_plan(tmp_path)
    monkeypatch.setattr(
        runner_execute,
        "seal_runtime_source_manifest",
        mock.Mock(side_effect=RuntimeManifestError("could not write the manifest")),
    )
    run = _stub_runner(monkeypatch, 0)

    with pytest.raises(RuntimeManifestError, match="could not write the manifest"):
        runner_execute.execute(plan, runtime_source_baseline={"generation": 1})
    run.assert_not_called()


def test_execute_still_fails_when_the_manifest_is_unusable_at_finalization(
    tmp_path, monkeypatch, stubbed_execute
):
    # The rollouts are finished by the time the seal is rechecked, so
    # a worktree edited while they ran is reported, not used to discard them.
    plan = _execute_plan(tmp_path)
    monkeypatch.setattr(
        runner_execute,
        "seal_runtime_source_manifest",
        lambda plan, baseline: (tmp_path / "manifest.json", "b" * 64),
    )
    monkeypatch.setattr(
        runner_execute,
        "verify_runtime_manifest",
        mock.Mock(side_effect=RuntimeManifestError("manifest is invalid")),
    )
    _stub_runner(monkeypatch, 0)
    summarize = mock.Mock(
        return_value=(0, {"total_runs": int(plan["expected_rollouts"])}, {})
    )
    monkeypatch.setattr(runner_execute, "summarize_run", summarize)

    with pytest.raises(RuntimeManifestError, match="manifest is invalid"):
        runner_execute.execute(plan, runtime_source_baseline={"generation": 1})
    summarize.assert_not_called()


def test_execute_keeps_a_finished_run_whose_runtime_seal_broke_mid_run(
    tmp_path, monkeypatch, stubbed_execute, capsys
):
    # The rollouts are already done by the time the seal is re-verified, so a
    # source change here costs nothing but the record, and the finished run is
    # kept rather than thrown away with exit 2.
    plan = _execute_plan(tmp_path)
    monkeypatch.setattr(
        runner_execute,
        "seal_runtime_source_manifest",
        lambda plan, baseline: (tmp_path / "manifest.json", "b" * 64),
    )
    monkeypatch.setattr(
        runner_execute,
        "verify_runtime_manifest",
        mock.Mock(side_effect=RuntimeSourceDriftError("generation changed")),
    )
    _stub_runner(monkeypatch, 0)
    summarize = mock.Mock(return_value=(0, {"total_runs": 2}, {}))
    monkeypatch.setattr(runner_execute, "summarize_run", summarize)

    assert runner_execute.execute(plan, runtime_source_baseline={"generation": 1}) == 0
    summarize.assert_called_once()
    assert "generation changed" in capsys.readouterr().err


def test_execute_runs_unsealed_when_the_source_tree_changed_before_launch(
    tmp_path, monkeypatch, stubbed_execute, capsys
):
    # A worktree edited between capture and seal no longer refuses
    # the launch. The run goes ahead without a sealed manifest -- nothing is
    # passed to the runner as if it were sealed, nothing is verified against a
    # manifest that does not exist -- and the warning names the cost.
    plan = _execute_plan(tmp_path)
    monkeypatch.setattr(
        runner_execute,
        "seal_runtime_source_manifest",
        mock.Mock(side_effect=RuntimeSourceDriftError("worktree drifted")),
    )
    run = _stub_runner(monkeypatch, 0)
    monkeypatch.setattr(
        runner_execute,
        "summarize_run",
        mock.Mock(return_value=(0, {"total_runs": 2}, {})),
    )

    assert runner_execute.execute(plan, runtime_source_baseline={"generation": 1}) == 0
    run.assert_called_once()
    message = capsys.readouterr().err
    assert "worktree drifted" in message
    # The warning has to say what is lost, or a missing post-hoc video later
    # looks like a render failure instead of this.
    assert "without a sealed runtime manifest" in message
    assert "post-hoc" in message
    # An unsealed run must not advertise a manifest to the runner -- and it has
    # to say the omission was deliberate. Leaving the variables merely absent is
    # what the runner reads as "derive one", which re-seals the tree that had
    # already moved and reports the run verified.
    environment = run.call_args.kwargs.get("env") or run.call_args[1]["env"]
    assert runner_execute.RUNTIME_SOURCE_MANIFEST_ENV not in environment
    assert runner_execute.RUNTIME_SOURCE_MANIFEST_SHA256_ENV not in environment
    assert environment[runner_execute.RUNTIME_SOURCE_UNSEALED_ENV] == "1"
