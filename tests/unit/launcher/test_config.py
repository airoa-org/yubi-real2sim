# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pytest

from real2sim.config.keys import _ROBOT_CLI_NAMES
import yaml

from tests._helpers.launcher import base_config as _base_config


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_SCRIPT_PATH = REPO_ROOT / "scripts" / "run_policy_batch.sh"
FIXTURE = REPO_ROOT / "config" / "release" / "fixture_v2.yaml"

from real2sim import launcher  # noqa: E402
from real2sim import artifacts as real2sim_artifacts  # noqa: E402
from real2sim import artifacts_binding  # noqa: E402
from real2sim.artifacts import ManifestError  # noqa: E402
from real2sim.cli import _print_configuration_error  # noqa: E402
from real2sim.config.schema import _runtime_port  # noqa: E402
from real2sim.runtime_manifest import RuntimeManifestError  # noqa: E402



class RunPolicyFromConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        # CPU fixture/config tests must not inherit the scheduler's GPU
        # visibility boundary. Tests of the GPU contract set it explicitly.
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

    def _load_temp(self, config: dict):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        path = self._write_config(root, config)
        return launcher.load_rollout_config(
            path,
            repo_root=root,
            timestamp="20260730_120000",
        )

    def test_schema_rejects_noncanonical_or_duplicate_gpu_selectors(self):
        cases = {
            "gs_cuda_devices": "00",
            "policy_cuda_device": "01",
            "sim_cuda_devices": "0,0",
        }
        for field, value in cases.items():
            with self.subTest(field=field, value=value):
                config = _base_config()
                config["runtime"][field] = value
                with self.assertRaisesRegex(
                    launcher.ConfigError,
                    "canonical non-negative GPU selector|duplicate GPU selector",
                ):
                    self._load_temp(config)

    def test_bundled_configs_are_portable_and_have_expected_profiles(self):
        # The strict fixture is the only file directly in config/release;
        # the shipped profiles are under config/release/<robot>/<rig>/
        # (see docs/architecture.md's "Supported production profile" table).
        expected = {
            "fixture_v2.yaml": ("fr3", 1, 1),
        }
        portable_root = REPO_ROOT / "portable-fixtures"
        environment = {
            "REAL2SIM_INPUTS_ROOT": str(portable_root),
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            for filename, (robot, rollouts, parallel) in expected.items():
                with self.subTest(config=filename):
                    path = REPO_ROOT / "config" / "release" / filename
                    text = path.read_text(encoding="utf-8")
                    self.assertNotIn("/" + "home/", text)
                    self.assertNotIn("/" + "groups/", text)
                    plan = launcher.load_rollout_config(
                        path,
                        timestamp="20260730_120000",
                    )
                    self.assertEqual(plan["resolved"]["robot"], robot)
                    self.assertEqual(
                        plan["resolved"]["policy"]["config_name"],
                        "pi05_yubi_pickplace",
                    )
                    self.assertEqual(plan["expected_rollouts"], rollouts)
                    self.assertEqual(
                        plan["resolved"]["runner"]["parallel_repeats"],
                        parallel,
                    )
                    if plan["resolved"]["runner"]["policy_batch_size"] >= 10:
                        render = plan["resolved"]["render"]
                        self.assertFalse(render["enabled"])
                        self.assertEqual(
                            render["forced_posthoc"], render["requested_enabled"]
                        )
                        self.assertEqual(
                            render["posthoc_required"], render["forced_posthoc"]
                        )
                        self.assertEqual(
                            plan["resolved"]["simulation"]["cli_args"].count(
                                "--no-render"
                            ),
                            1,
                        )
                    self.assertTrue(
                        str(plan["run_root"]).startswith(str(REPO_ROOT / "outputs"))
                    )
            release_profiles = sorted((REPO_ROOT / "config" / "release").glob("*/*/*.yaml"))
            self.assertTrue(release_profiles)
            for path in release_profiles:
                with self.subTest(config=str(path.relative_to(REPO_ROOT))):
                    text = path.read_text(encoding="utf-8")
                    self.assertNotIn("/" + "home/", text)
                    self.assertNotIn("/" + "groups/", text)
                    plan = launcher.load_rollout_config(path, timestamp="20260730_120000")
                    self.assertEqual(plan["expected_rollouts"], 10)
                    # One 300-frame horizon and one 7200 s wall limit for every release profile.
                    self.assertEqual(plan["resolved"]["runner"]["frames"], 300)
                    self.assertEqual(plan["resolved"]["runner"]["max_wall_sec"], 7200)
                    self.assertTrue(
                        str(plan["run_root"]).startswith(str(REPO_ROOT / "outputs"))
                    )

    def test_schema_v2_gs_composite_mode_defaults_to_depth_and_accepts_foreground(self):
        # gs_composite_mode follows the same optional-parameter pattern as
        # policy_action_clamp: absent from simulation.parameters, it falls
        # back to the robosuite_env/rollout/cli.py --gs-composite-mode
        # default ("depth"), and can be pinned to "foreground" explicitly.
        source = copy.deepcopy(_base_config())
        source["simulation"]["parameters"].pop("gs_composite_mode", None)
        default_plan = self._load_temp(source)
        default_simulation = default_plan["resolved"]["simulation"]
        self.assertEqual(default_simulation["parameters"]["gs_composite_mode"], "depth")
        cli_args = default_simulation["cli_args"]
        self.assertEqual(cli_args[cli_args.index("--gs-composite-mode") + 1], "depth")

        source["simulation"]["parameters"]["gs_composite_mode"] = "foreground"
        foreground_plan = self._load_temp(source)
        foreground_simulation = foreground_plan["resolved"]["simulation"]
        self.assertEqual(
            foreground_simulation["parameters"]["gs_composite_mode"], "foreground"
        )
        cli_args = foreground_simulation["cli_args"]
        self.assertEqual(
            cli_args[cli_args.index("--gs-composite-mode") + 1], "foreground"
        )

    def test_schema_v2_obs_gs_only_defaults_false_and_is_mutually_exclusive(self):
        source = copy.deepcopy(_base_config())
        source["simulation"]["parameters"].pop("obs_gs_only", None)
        default_plan = self._load_temp(source)
        default_simulation = default_plan["resolved"]["simulation"]
        self.assertFalse(default_simulation["parameters"]["obs_gs_only"])
        self.assertNotIn("--obs-gs-only", default_simulation["cli_args"])

        source["simulation"]["parameters"]["obs_gs_only"] = True
        gs_only_plan = self._load_temp(source)
        gs_only_simulation = gs_only_plan["resolved"]["simulation"]
        self.assertTrue(gs_only_simulation["parameters"]["obs_gs_only"])
        self.assertIn("--obs-gs-only", gs_only_simulation["cli_args"])

        source["simulation"]["parameters"]["no_obs_gs"] = True
        with self.assertRaisesRegex(launcher.ConfigError, "mutually exclusive"):
            self._load_temp(source)

    def test_schema_v2_gs_color_transform_defaults_round_trip_and_rejects_invalid(self):
        source = copy.deepcopy(_base_config())
        source["simulation"]["parameters"].pop("gs_color_gain_rgb", None)
        source["simulation"]["parameters"].pop("gs_color_gamma_rgb", None)
        default_plan = self._load_temp(source)
        default_simulation = default_plan["resolved"]["simulation"]
        self.assertEqual(
            default_simulation["parameters"]["gs_color_gain_rgb"],
            [1.0, 1.0, 1.0],
        )
        self.assertEqual(
            default_simulation["parameters"]["gs_color_gamma_rgb"],
            [1.0, 1.0, 1.0],
        )
        default_cli_args = default_simulation["cli_args"]
        gain_index = default_cli_args.index("--gs-color-gain-rgb")
        self.assertEqual(
            default_cli_args[gain_index + 1 : gain_index + 4],
            ["1.0", "1.0", "1.0"],
        )
        gamma_index = default_cli_args.index("--gs-color-gamma-rgb")
        self.assertEqual(
            default_cli_args[gamma_index + 1 : gamma_index + 4],
            ["1.0", "1.0", "1.0"],
        )

        source["simulation"]["parameters"]["gs_color_gain_rgb"] = [
            1.12,
            1.0,
            0.86,
        ]
        source["simulation"]["parameters"]["gs_color_gamma_rgb"] = [
            0.9,
            1.0,
            1.1,
        ]
        explicit_plan = self._load_temp(source)
        explicit_simulation = explicit_plan["resolved"]["simulation"]
        self.assertEqual(
            explicit_simulation["parameters"]["gs_color_gain_rgb"],
            [1.12, 1.0, 0.86],
        )
        self.assertEqual(
            explicit_simulation["parameters"]["gs_color_gamma_rgb"],
            [0.9, 1.0, 1.1],
        )
        explicit_cli_args = explicit_simulation["cli_args"]
        gain_index = explicit_cli_args.index("--gs-color-gain-rgb")
        self.assertEqual(
            explicit_cli_args[gain_index + 1 : gain_index + 4],
            ["1.12", "1.0", "0.86"],
        )
        gamma_index = explicit_cli_args.index("--gs-color-gamma-rgb")
        self.assertEqual(
            explicit_cli_args[gamma_index + 1 : gamma_index + 4],
            ["0.9", "1.0", "1.1"],
        )

        invalid_values = (
            ("gs_color_gain_rgb", [1.0, 1.0]),
            ("gs_color_gamma_rgb", [1.0, 1.0, 1.0, 1.0]),
            ("gs_color_gain_rgb", [0.0, 1.0, 1.0]),
            ("gs_color_gamma_rgb", [-0.1, 1.0, 1.0]),
            ("gs_color_gain_rgb", ["warm", 1.0, 1.0]),
            ("gs_color_gamma_rgb", [1.0, None, 1.0]),
            ("gs_color_gain_rgb", [4.1, 1.0, 1.0]),
            ("gs_color_gamma_rgb", [1.0, 1.0, 4.1]),
        )
        for key, invalid in invalid_values:
            with self.subTest(key=key, value=invalid):
                bad = copy.deepcopy(_base_config())
                bad["simulation"]["parameters"][key] = invalid
                with self.assertRaisesRegex(
                    launcher.ConfigError,
                    rf"simulation\.parameters\.{key}",
                ):
                    self._load_temp(bad)

    def test_environment_expansion_applies_to_extra_arguments(self):
        config = _base_config()
        config["reproducibility"]["environment"]["allowlist"].append(
            "REAL2SIM_TEST_JOINTS"
        )
        config["simulation"]["parameters"]["initial_arm_joints_json"] = (
            "${REAL2SIM_TEST_JOINTS}"
        )
        expected = "/tmp/registered/initial_arm_joints.json"
        with mock.patch.dict(
            os.environ, {"REAL2SIM_TEST_JOINTS": expected}, clear=False
        ):
            plan = self._load_temp(config)

        self.assertEqual(
            plan["resolved"]["simulation"]["parameters"]["initial_arm_joints_json"],
            expected,
        )
        arguments = plan["resolved"]["simulation"]["cli_args"]
        self.assertEqual(
            arguments[arguments.index("--initial-arm-joints-json") + 1], expected
        )

    def test_environment_expansion_and_dotenv_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / ".env"
            from_file = root / "from-file"
            from_process = root / "from-process"
            env_file.write_text(
                f"REAL2SIM_TEST_PATH={from_file}\n"
                "export QUOTED_VALUE='two words'\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {"REAL2SIM_TEST_PATH": str(from_process)},
                clear=False,
            ):
                launcher.load_env_file(env_file)
                self.assertEqual(
                    os.environ["REAL2SIM_TEST_PATH"], str(from_process)
                )
                self.assertEqual(os.environ["QUOTED_VALUE"], "two words")
                self.assertEqual(
                    launcher._render_template(
                        "${REAL2SIM_TEST_PATH}/{robot}",
                        "test",
                        {"robot": "fr3"},
                    ),
                    str(from_process / "fr3"),
                )
            with self.assertRaisesRegex(
                launcher.ConfigError, "unset environment variable"
            ):
                launcher._render_template(
                    "${REAL2SIM_VARIABLE_THAT_DOES_NOT_EXIST}",
                    "test",
                    {},
                )

    def test_schema_rejects_cluster_sections_unknown_keys_and_bad_counts(self):
        for section, key, value in (
            ("runtime", "group_id", "legacy-group"),
            (None, "issue", {"publish": False}),
            ("policy", "password", "forbidden"),
        ):
            config = _base_config()
            if section is None:
                config[key] = value
            else:
                config[section][key] = value
            with self.subTest(key=key), self.assertRaises(launcher.ConfigError):
                self._load_temp(config)

        config = _base_config()
        config["runner"]["expected_rollouts"] = 3
        with self.assertRaisesRegex(
            launcher.ConfigError, "len\\(scenes\\) \\* repeats"
        ):
            self._load_temp(config)

        config = _base_config()
        del config["policy"]["config_name"]
        with self.assertRaisesRegex(launcher.ConfigError, "config_name"):
            self._load_temp(config)

        for key, value, message in (
            ("build_blend", False, r"render\.build_blend is retired"),
            ("desk_texture", "/tmp/table.png", "forbids render.desk_texture"),
        ):
            config = _base_config()
            config["render"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                launcher.ConfigError, message
            ):
                self._load_temp(config)

    def test_typed_controller_selectors_expand_once_and_validate_combinations(self):
        plan = self._load_temp(_base_config())
        args = plan["resolved"]["simulation"]["cli_args"]
        self.assertEqual(args.count("--fr3-controller"), 1)
        self.assertNotIn("--fr3-contact-control", args)
        self.assertEqual(args.count("--grip-ctrl"), 1)
        self.assertEqual(args.count("--dynamic-grip-cbpc"), 1)
        self.assertNotIn("--grasp-mode", args)

    def test_fr3_nullspace_selector(self):
        # Default: the selector is optional and resolves to home without
        # touching the worker argument vector.
        config = self._load_temp(_base_config())
        self.assertEqual(config["resolved"]["simulation"]["fr3_nullspace"], "home")
        self.assertNotIn("--fr3-nullspace", config["resolved"]["simulation"]["cli_args"])

        # The retired arm-controller values are rejected before null-space use.
        wrong_arm = _base_config()
        wrong_arm["simulation"]["fr3_controller"] = "osc"
        wrong_arm["simulation"]["fr3_nullspace"] = "off"
        with self.assertRaisesRegex(launcher.ConfigError, "must be one of: .*joint_ik"):
            self._load_temp(wrong_arm)

        # off: explicit selector reaches the worker; the configured gain is
        # kept as-is in YAML and forced to 0 at runtime.
        off = _base_config()
        off["simulation"]["fr3_controller"] = "joint_ik"
        off["simulation"]["fr3_nullspace"] = "off"
        loaded = self._load_temp(off)
        self.assertEqual(loaded["resolved"]["simulation"]["fr3_nullspace"], "off")
        cli_args = loaded["resolved"]["simulation"]["cli_args"]
        self.assertEqual(cli_args[cli_args.index("--fr3-nullspace") + 1], "off")

        # home with a zero gain would be a silent no-op.
        noop = _base_config()
        noop["simulation"]["fr3_controller"] = "joint_ik"
        noop["simulation"]["fr3_nullspace"] = "home"
        noop["simulation"]["parameters"]["joint_ik_nullspace_gain"] = 0.0
        with self.assertRaisesRegex(launcher.ConfigError, "positive"):
            self._load_temp(noop)

        bad_value = _base_config()
        bad_value["simulation"]["fr3_controller"] = "joint_ik"
        bad_value["simulation"]["fr3_nullspace"] = "maybe"
        with self.assertRaisesRegex(launcher.ConfigError, "must be one of"):
            self._load_temp(bad_value)

    def test_impedance_source_parameters_and_physics_clock_reach_worker(self):
        config = _base_config()
        config["simulation"]["fr3_controller"] = "cartesian_impedance"
        parameters = config["simulation"]["parameters"]
        parameters.update(fr3_impedance={}, policy_action_clamp=False)
        sim = self._load_temp(config)["resolved"]["simulation"]
        argv = sim["cli_args"]
        gains = json.loads(argv[argv.index("--fr3-impedance")+1])
        self.assertEqual(gains["stiffness_linear"], 1500)
        self.assertEqual(gains["tau_limit"], [40,40,40,8,8,8,8])
        self.assertEqual(argv[argv.index("--mujoco-timestep")+1], "0.001")
        self.assertIn("--no-policy-action-clamp", argv)
        self.assertEqual(sim["parameters"]["mujoco_timestep"], .001)
        parameters["mujoco_timestep"] = .002
        sim = self._load_temp(config)["resolved"]["simulation"]
        self.assertEqual(sim["parameters"]["mujoco_timestep"], .002)
        parameters["policy_action_clamp"] = True
        with self.assertRaisesRegex(launcher.ConfigError, "cmd_twist"):
            self._load_temp(config)

    def test_cmd_twist_controller_without_torque_hook_is_rejected(self):
        config = _base_config()
        config["simulation"]["fr3_controller"] = "cartesian_impedance"
        config["simulation"]["parameters"].update(fr3_impedance={}, policy_action_clamp=False)
        with mock.patch("real2sim.fr3_controllers.FR3_IMPEDANCE_HOOK_CONTROLLERS", frozenset()):
            with self.assertRaisesRegex(launcher.ConfigError, "has no torque hook"):
                self._load_temp(config)

    def test_optional_fr3_joint_gains_are_strict_and_omitted_by_default(self):
        default_plan = self._load_temp(_base_config())
        default_simulation = default_plan["resolved"]["simulation"]
        self.assertNotIn("fr3_joint_kp", default_simulation["parameters"])
        self.assertNotIn("fr3_joint_kd", default_simulation["parameters"])
        self.assertNotIn("--fr3-joint-kp", default_simulation["cli_args"])
        self.assertNotIn("--fr3-joint-kd", default_simulation["cli_args"])

        configured = _base_config()
        configured["simulation"]["parameters"]["fr3_joint_kp"] = 400
        configured["simulation"]["parameters"]["fr3_joint_kd"] = 80
        configured_plan = self._load_temp(configured)
        configured_simulation = configured_plan["resolved"]["simulation"]
        self.assertEqual(configured_simulation["parameters"]["fr3_joint_kp"], 400.0)
        self.assertEqual(configured_simulation["parameters"]["fr3_joint_kd"], 80.0)
        arguments = configured_simulation["cli_args"]
        self.assertEqual(arguments[arguments.index("--fr3-joint-kp") + 1], "400.0")
        self.assertEqual(arguments[arguments.index("--fr3-joint-kd") + 1], "80.0")

        invalid_values = (
            ("fr3_joint_kp", 0),
            ("fr3_joint_kp", -1),
            ("fr3_joint_kp", True),
            ("fr3_joint_kp", float("inf")),
            ("fr3_joint_kd", -1),
            ("fr3_joint_kd", True),
            ("fr3_joint_kd", float("nan")),
        )
        for key, invalid in invalid_values:
            with self.subTest(key=key, value=invalid):
                bad = _base_config()
                bad["simulation"]["parameters"][key] = invalid
                with self.assertRaisesRegex(
                    launcher.ConfigError,
                    rf"simulation\.parameters\.{key}",
                ):
                    self._load_temp(bad)

    def test_base_placement_is_a_typed_path_parameter(self):
        """The decisive scene input belongs in the argv, not only in the env.

        Carried only as an environment variable, the Duo mount calibration
        would leave the recorded run command silent about which bases a run
        used.
        """

        default_plan = self._load_temp(_base_config())
        default_simulation = default_plan["resolved"]["simulation"]
        self.assertIsNone(default_simulation["parameters"]["base_placement"])
        self.assertNotIn("--base-placement", default_simulation["cli_args"])

        configured = _base_config()
        configured["simulation"]["parameters"]["base_placement"] = (
            "{repo_root}/robosuite_env/obj_calib/fr3_base_duo.json"
        )
        arguments = self._load_temp(configured)["resolved"]["simulation"]["cli_args"]
        rendered = Path(arguments[arguments.index("--base-placement") + 1])
        self.assertTrue(rendered.is_absolute())
        self.assertEqual(
            rendered.parts[-3:],
            ("robosuite_env", "obj_calib", "fr3_base_duo.json"),
        )

    def test_fr3_joint_gains_require_dynamic_physics(self):
        """Only the dynamic JOINT_POSITION controller ever reads them.

        A kinematic profile carrying the gains in its official argv is refused,
        since robosuite_env/rollout/build/model.py would drop them without a
        word.
        """

        for key in ("fr3_joint_kp", "fr3_joint_kd"):
            with self.subTest(key=key):
                kinematic = _base_config()
                kinematic["simulation"]["parameters"]["physics"] = "kinematic"
                kinematic["simulation"]["parameters"][key] = 400
                with self.assertRaisesRegex(
                    launcher.ConfigError, "requires physics: dynamic"
                ):
                    self._load_temp(kinematic)

    def test_schema_v2_fr3_runs_bind_the_tracked_fr3_arm_bundle(self):
        # FR3 ships its arm geometry as a tracked repository bundle next to the
        # shared YUBI hands and the task objects.
        self.assertEqual(
            artifacts_binding._schema_v2_asset_bundle_ids(
                self._load_temp(_base_config())
            ),
            ("yubi_right", "yubi_left", "fr3", "cup", "plate"),
        )

    def test_schema_v2_rejects_an_unregistered_embodiment(self):
        config = _base_config()
        config["robot"] = "ur5"
        with self.assertRaisesRegex(
            launcher.ConfigError,
            "robot must be one of: " + ", ".join(sorted(_ROBOT_CLI_NAMES)),
        ):
            self._load_temp(config)

    def test_runner_environment_is_repo_scoped_and_forwards_gpu_settings(self):
        config = _base_config()
        config["runner"].update(
            mode="fast_parallel",
            repeats=4,
            policy_batch_size=2,
            expected_rollouts=4,
        )
        config["runtime"]["render_lock_root"] = "{repo_root}/node-render-locks"
        # schema-v2 only forwards process environment variables the config
        # itself declares in reproducibility.environment.allowlist; nothing
        # is inherited ambiently the way schema-v1 used to allow.
        config["reproducibility"]["environment"]["allowlist"].append(
            "OPENPI_DATA_HOME"
        )
        with mock.patch.dict(
            os.environ,
            {
                "OPENPI_DATA_HOME": "relative-openpi-cache",
                "OPENPI_SRC": str(REPO_ROOT / "test-openpi"),
            },
            clear=False,
        ):
            plan = self._load_temp(config)
            environment = launcher.build_runner_environment(plan)
        self.assertTrue(
            environment["OPENPI_DATA_HOME"].endswith("relative-openpi-cache")
        )
        self.assertTrue(Path(environment["OPENPI_DATA_HOME"]).is_absolute())
        # OpenPI runs from the installed, locked environment: the runner no
        # longer names a source checkout for it.
        self.assertNotIn("OPENPI_SRC", environment)
        self.assertEqual(environment["GS_CUDA_DEVICES"], "0,2")
        self.assertEqual(environment["POLICY_CUDA_DEVICE"], "1")
        self.assertEqual(environment["SIM_CUDA_DEVICES"], "3,5")
        self.assertEqual(environment["GRASP_MODE"], "rule")
        self.assertEqual(
            environment["RENDER_LOCK_ROOT"],
            str(Path(plan["repo_root"]) / "node-render-locks"),
        )
        self.assertEqual(
            environment["BATCH_CLIENT_ORDER"].split(","),
            [
                "test_scene_run_01",
                "test_scene_run_02",
                "test_scene_run_03",
                "test_scene_run_04",
            ],
        )
        self.assertEqual(
            json.loads(environment["EXTRA_SIM_ARGS_JSON"]),
            plan["resolved"]["simulation"]["cli_args"],
        )
        forbidden = {
            "GROUP_ID",
            "USER_ID",
            "WORK_ROOT",
            "SHARED_ROOT",
        }
        self.assertFalse(forbidden.intersection(environment))

    def test_ambient_behavior_env_vars_are_rejected_unless_allowlisted(self):
        # Schema v1 silently forwarded most of the ambient process
        # environment (RS_*-prefixed variables excepted); schema v2 is
        # fail-closed and rejects any behavior-changing variable that is
        # present in the process environment but not explicitly allowlisted.
        config = _base_config()
        with mock.patch.dict(
            os.environ, {"RS_CUP_XY": "do-not-forward"}, clear=False
        ):
            with self.assertRaisesRegex(
                launcher.ConfigError,
                "behavior-changing process environment variables",
            ):
                self._load_temp(config)





    def test_render_plan_matrix(self):
        batch10 = {
            "mode": "fast_parallel",
            "repeats": 10,
            "policy_batch_size": 10,
            "expected_rollouts": 10,
        }
        live30 = {
            "mode": "sequential_waves",
            "repeats": 30,
            "policy_batch_size": 1,
            "expected_rollouts": 30,
        }
        # name, runner, enabled (None means omitted), third_person, and the
        # complete normalized render contract plus --no-render cardinality.
        cases = (
            ("b10", batch10, True, None, (True, False, True, True, True, "1", 1)),
            ("b10-no-third-person", batch10, True, False, (True, False, True, False, False, "0", 1)),
            ("explicit-opt-out", None, False, None, (False, False, False, False, True, "0", 1)),
            ("live-below-threshold", live30, True, None, (True, True, False, False, True, "0", 0)),
            ("b10-default", batch10, None, None, (True, False, True, True, True, "1", 1)),
            ("b10-opt-out", batch10, False, None, (False, False, False, False, True, "0", 1)),
        )
        for name, runner, enabled, third_person, expected in cases:
            source = _base_config()
            if runner is not None:
                source["runner"].update(runner)
            if enabled is None:
                source["render"].pop("enabled", None)
            else:
                source["render"]["enabled"] = enabled
            if third_person is not None:
                source["render"]["third_person"] = third_person
            with self.subTest(name=name):
                plan = self._load_temp(source)
                render = plan["resolved"]["render"]
                observed = (
                    render["requested_enabled"], render["enabled"],
                    render["forced_posthoc"], render["posthoc_required"],
                    render["third_person"],
                    launcher.build_runner_environment(plan)["POSTHOC_RENDER_REQUIRED"],
                    plan["resolved"]["simulation"]["cli_args"].count("--no-render"),
                )
                self.assertEqual(observed, expected)

        invalid = _base_config()
        invalid["render"]["third_person"] = "yes"
        with self.assertRaisesRegex(
            launcher.ConfigError, "render.third_person must be a YAML boolean"
        ):
            self._load_temp(invalid)

    def test_schema_v2_posthoc_requires_a_supported_grid_cardinality(self):
        source = yaml.safe_load(
            (REPO_ROOT / "config" / "release" / "fixture_v2.yaml").read_text(
                encoding="utf-8"
            )
        )
        source["runner"].update(
            {
                "mode": "fast_parallel",
                "policy_batch_size": 11,
                "repeats": 11,
                "expected_rollouts": 11,
            }
        )
        source["render"]["enabled"] = True
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "config") as temporary:
            config_path = self._write_config(Path(temporary), source)
            with self.assertRaisesRegex(
                launcher.ConfigError,
                "video-producing schema-v2 runs require a checked-in",
            ):
                launcher.load_rollout_config(
                    config_path,
                    repo_root=REPO_ROOT,
                    timestamp="20990101_000000",
                )

    def test_schema_v2_thirty_rollouts_selects_six_by_six_grid(self):
        source = yaml.safe_load(
            (REPO_ROOT / "config" / "release" / "fixture_v2.yaml").read_text(
                encoding="utf-8"
            )
        )
        source["runner"].update(
            {
                "mode": "fast_parallel",
                "policy_batch_size": 10,
                "repeats": 30,
                "expected_rollouts": 30,
            }
        )
        source["render"]["enabled"] = True
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "config") as temporary:
            config_path = self._write_config(Path(temporary), source)
            plan = launcher.load_rollout_config(
                config_path,
                repo_root=REPO_ROOT,
                timestamp="20990101_000000",
            )

        expected = str(
            REPO_ROOT
            / "config"
            / "render_templates"
            / "rollout_grid_clear_headers_6x6.json"
        )
        self.assertEqual(plan["resolved"]["render"]["grid_template"], expected)
        self.assertEqual(
            launcher.build_runner_environment(plan)["GRID_TEMPLATE"], expected
        )



    def test_schema_v2_environment_does_not_invent_gpu_zero(self):
        plan = launcher.load_rollout_config(
            REPO_ROOT / "config" / "release" / "fixture_v2.yaml",
            timestamp="20990101_000000",
        )

        environment = launcher.build_runner_environment(plan)

        self.assertEqual(environment["GS_CUDA_DEVICES"], "")
        self.assertEqual(environment["POLICY_CUDA_DEVICE"], "")
        self.assertEqual(environment["SIM_CUDA_DEVICES"], "")
        self.assertEqual(
            environment["RENDER_LOCK_ROOT"],
            f"/tmp/real2sim-render-locks-{os.getuid()}",
        )
        self.assertEqual(
            environment["INIT_STATE_CACHE_SCHEMA"],
            "initv5_named_task_object_settle",
        )
        self.assertEqual(environment["MIMICGEN_INIT_EPISODE_OFFSET"], "0")
        self.assertEqual(environment["WHITE_DESK_TEX"], "")
        self.assertEqual(
            environment["GRID_TEMPLATE"],
            str(
                REPO_ROOT
                / "config"
                / "render_templates"
                / "rollout_grid_clear_headers_4x3.json"
            ),
        )
        for key in (
            "INIT_STATE_ROOT",
            "MIMICGEN_INIT_DATASET_ROOT",
            "MIMICGEN_INIT_STATE_LOG_ROOT",
            "POLICY_NOISE_BANK_DIR",
            "POLICY_INFERENCE_MANIFEST",
        ):
            self.assertEqual(environment[key], "")

    def test_schema_v2_rejects_unsupported_batch_and_subframe_modes(self):
        fixture = REPO_ROOT / "config/release/fixture_v2.yaml"
        for mode, message in (
            ("paired", "exact policy model batch size"),
            ("subframes", "dynamic_subframes: 1"),
        ):
            source = yaml.safe_load(fixture.read_text(encoding="utf-8"))
            if mode == "paired":
                source["simulation"].update(
                    rf_overlap=4,
                    rf_overlap_sampler="paired_weighted",
                    rf_overlap_prev_weight=None,
                )
            else:
                source["simulation"]["parameters"]["dynamic_subframes"] = 2
            with tempfile.TemporaryDirectory(dir=REPO_ROOT / "config") as temporary:
                config_path = self._write_config(Path(temporary), source)
                with self.subTest(mode=mode), self.assertRaisesRegex(
                    launcher.ConfigError, message
                ):
                    launcher.load_rollout_config(
                        config_path, repo_root=REPO_ROOT, timestamp="20990101_000000"
                    )

    def test_schema_v2_policy_action_clamp_is_optional_and_can_be_disabled(self):
        source = yaml.safe_load(
            (REPO_ROOT / "config" / "release" / "fixture_v2.yaml").read_text(
                encoding="utf-8"
            )
        )
        source["simulation"]["parameters"].pop("policy_action_clamp", None)
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "config") as temporary:
            config_path = self._write_config(Path(temporary), source)
            enabled = launcher.load_rollout_config(
                config_path,
                repo_root=REPO_ROOT,
                timestamp="20990101_000000",
            )
            source["simulation"]["parameters"]["policy_action_clamp"] = False
            config_path = self._write_config(Path(temporary), source)
            disabled = launcher.load_rollout_config(
                config_path,
                repo_root=REPO_ROOT,
                timestamp="20990101_000001",
            )

        enabled_simulation = enabled["resolved"]["simulation"]
        disabled_simulation = disabled["resolved"]["simulation"]
        self.assertTrue(enabled_simulation["parameters"]["policy_action_clamp"])
        self.assertIn("--policy-action-clamp", enabled_simulation["cli_args"])
        self.assertFalse(disabled_simulation["parameters"]["policy_action_clamp"])
        self.assertIn("--no-policy-action-clamp", disabled_simulation["cli_args"])

    def test_schema_v2_rejects_duplicate_scene_ids(self):
        plan = launcher.load_rollout_config(
            REPO_ROOT / "config" / "release" / "fixture_v2.yaml",
            timestamp="20990101_000002",
        )
        second = dict(plan["resolved"]["scenes"][0])
        second["tag"] = "second_scene"
        second["path"] = "/tmp/second-scene"
        plan["resolved"]["scenes"].append(second)
        asset_report = {
            "manifest_path": REPO_ROOT / "robosuite_env" / "assets" / "manifest.json",
            "contract_digest": "a" * 64,
        }
        common = {
            "id": "duplicate-id",
            "manifest_path": REPO_ROOT / "background.json",
            "ply_sha256": "d" * 64,
            "w2gs_matrix": np.eye(4),
            "collision_directory": REPO_ROOT / "table" / "coacd",
            "collision_files": (REPO_ROOT / "table" / "coacd" / "part.obj",),
            "contract_digest": "b" * 64,
            "collision_tree_sha256": "c" * 64,
        }
        reports = [
            {**common, "name": "tiny_fixture_scene"},
            {**common, "name": "second_scene"},
        ]

        with mock.patch.object(
            artifacts_binding, "validate_repository_assets", return_value=asset_report
        ), mock.patch.object(
            artifacts_binding, "validate_scene_contract", side_effect=reports
        ), self.assertRaisesRegex(launcher.ConfigError, "duplicate scene id"):
            launcher._validate_schema_v2_contracts(plan)

    def test_schema_v2_rejects_any_table_path_outside_the_scene_contract(self):
        source = yaml.safe_load(
            (REPO_ROOT / "config" / "release" / "fixture_v2.yaml").read_text(
                encoding="utf-8"
            )
        )
        source["simulation"]["parameters"]["desk"] = "/tmp/legacy-desk.obj"
        with self.assertRaisesRegex(
            launcher.ConfigError,
            "requires simulation.parameters.desk: scene",
        ):
            launcher._validate_v2_parameters(
                source["simulation"]["parameters"],
                context={
                    "repo_root": str(REPO_ROOT),
                    "config_dir": str(REPO_ROOT / "config" / "release"),
                    "timestamp": "20990101_000000",
                    "robot": "fr3",
                },
                config_dir=REPO_ROOT / "config" / "release",
            )

    def test_schema_v2_rejects_gs_transform_outside_scene_contract(self):
        source = yaml.safe_load(
            (REPO_ROOT / "config" / "release" / "fixture_v2.yaml").read_text(
                encoding="utf-8"
            )
        )
        source["simulation"]["parameters"]["gs_calibration_world_offset"] = [
            0.0,
            0.0,
            0.8,
        ]
        with self.assertRaisesRegex(
            launcher.ConfigError,
            "world-to-GS transform is authoritative",
        ):
            launcher._validate_v2_parameters(
                source["simulation"]["parameters"],
                context={
                    "repo_root": str(REPO_ROOT),
                    "config_dir": str(REPO_ROOT / "config" / "release"),
                    "timestamp": "20990101_000000",
                    "robot": "fr3",
                },
                config_dir=REPO_ROOT / "config" / "release",
            )

    def test_extra_args_cannot_select_official_execution(self):
        # schema-v2 forbids simulation.extra_args entirely, so a config can
        # never hand-select official-execution-only flags such as
        # --strict-assets or --official-run-manifest.
        for flag in (
            "--grip-ctrl",
            "--gs",
            "--gs-transform",
            "--gs-rot",
            "--gs-pivot",
            "--gs-pivz",
            "--strict-assets",
            "--official-run-manifest",
        ):
            config = _base_config()
            config["simulation"]["extra_args"] = [flag]
            with self.subTest(flag=flag), self.assertRaisesRegex(
                launcher.ConfigError,
                "forbids simulation.extra_args",
            ):
                self._load_temp(config)




def test_config_error_preserves_message_and_exposes_context():
    error = launcher.ConfigError(
        "invalid value",
        path="simulation.parameters.physics",
        allowed=("dynamic", "kinematic"),
        got="quantum",
    )

    assert str(error) == "invalid value"
    assert error.message == "invalid value"
    assert error.path == "simulation.parameters.physics"
    assert error.allowed == ("dynamic", "kinematic")
    assert error.got == "quantum"

def test_cli_unknown_key_reports_path_and_close_match(capsys, isolated_environ):
    config = yaml.safe_load(
        (REPO_ROOT / "config" / "release" / "fixture_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["runner"]["repeatsx"] = config["runner"]["repeats"]
    with tempfile.TemporaryDirectory(dir=REPO_ROOT / "config") as temporary:
        config_path = Path(temporary) / "unknown-key.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        assert launcher.main(["--config", str(config_path), "--dry-run", "--env-file", str(isolated_environ)]) == 2

    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == "configuration error: runner contains unknown keys: repeatsx"
    assert lines[1].startswith("  at: runner")
    assert "got: repeatsx" in lines[1]
    assert "suggestions: repeats" in lines[1]


def test_cli_refuses_unknown_task(capsys, isolated_environ):
    config = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    config["task"]["objects"] = "unknown_task"
    with tempfile.TemporaryDirectory(dir=REPO_ROOT / "config") as temporary:
        config_path = Path(temporary) / "unknown-task.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        assert launcher.main(["--config", str(config_path), "--dry-run", "--env-file", str(isolated_environ)]) == 2

    lines = capsys.readouterr().err.splitlines()
    assert lines[0].startswith("configuration error: unknown task.objects 'unknown_task'")
    assert lines[1].startswith("  at: task.objects")


def test_raw_config_resolution_and_table_projection_match_launcher_contract():
    config_path = REPO_ROOT / "config" / "release" / "fixture_v2.yaml"
    raw_bytes, raw_config = launcher.load_raw_config(config_path)

    assert raw_bytes == config_path.read_bytes()
    resolved = launcher.resolve_plan(
        raw_config,
        path=config_path,
        timestamp="20000101_000000",
    )
    wrapped = launcher.load_rollout_config(
        config_path,
        timestamp="20000101_000000",
    )
    assert resolved == wrapped

    table = launcher.format_dry_run_table(resolved)
    assert "checkpoint" in table
    assert "scene[" in table
    assert "controller.fr3_controller" in table
    assert "action_horizon" in table
    assert "server" in table
    assert "gs_cuda_devices" in table
    assert "render.posthoc_required" in table
    assert "run_root" in table

def test_cli_timestamp_is_validated_and_passed_to_plan(capsys, isolated_environ):
    config_path = REPO_ROOT / "config" / "release" / "fixture_v2.yaml"
    assert launcher.main(
        [
            "--config",
            str(config_path),
            "--dry-run",
            "--timestamp",
            "20000101_010203",
            "--env-file",
            str(isolated_environ),
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["run_root"].endswith("_20000101_010203")

def test_cli_timestamp_rejects_worker_incompatible_value():
    parser = launcher.build_arg_parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(
            [
                "--config",
                "config/release/fixture_v2.yaml",
                "--dry-run",
                "--timestamp",
                "2026-08-23T01:02:03",
            ]
        )
    assert error.value.code == 2

def test_runtime_port_templates_resolve_and_are_visible_in_dry_run():
    source = REPO_ROOT / "config" / "release" / "fixture_v2.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["runtime"]["server_port"] = "${TEST_SERVER_PORT}"
    config["runtime"]["gs_port_base"] = "${TEST_GS_PORT_BASE}"
    environment = config["reproducibility"]["environment"]
    environment["allowlist"].extend(["TEST_SERVER_PORT", "TEST_GS_PORT_BASE"])
    environment["values"].update(
        {"TEST_SERVER_PORT": "9074", "TEST_GS_PORT_BASE": "9321"}
    )

    with tempfile.TemporaryDirectory(dir=REPO_ROOT / "config") as temporary:
        config_path = Path(temporary) / "port-templates.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        plan = launcher.load_rollout_config(
            config_path,
            timestamp="20000101_000000",
        )

    runtime = plan["resolved"]["runtime"]
    assert runtime["server_port"] == 9074
    assert runtime["gs_port_base"] == 9321
    environment_view = launcher.dry_run_view(plan)["environment"]
    assert environment_view["SERVER_HOST"] == "127.0.0.1"
    assert environment_view["SERVER_PORT"] == "9074"
    assert environment_view["GS_PORT_BASE"] == "9321"
    assert environment_view["GS_START_TIMEOUT_SEC"] == "60"
    # a config that says nothing about the far field keeps the black it has
    # always rendered
    assert environment_view["GS_BG_COLOR"] == "0.0 0.0 0.0"


def test_runtime_gs_background_color_reaches_the_batch_runner():
    """Carving splats turns holes into far field, so its colour is sealed."""

    source = REPO_ROOT / "config" / "release" / "fixture_v2.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["runtime"]["gs_background_color"] = [0.6835, 0.6937, 0.6377]
    with tempfile.TemporaryDirectory(dir=REPO_ROOT / "config") as temporary:
        config_path = Path(temporary) / "far-field.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        plan = launcher.load_rollout_config(
            config_path, timestamp="20000101_000000"
        )
    assert plan["resolved"]["runtime"]["gs_background_color"] == [
        0.6835,
        0.6937,
        0.6377,
    ]
    view = launcher.dry_run_view(plan)["environment"]
    assert view["GS_BG_COLOR"] == "0.6835 0.6937 0.6377"


@pytest.mark.parametrize(
    "value", ([0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 0.0, -0.1], "black")
)
def test_runtime_gs_background_color_rejects_a_colour_outside_the_unit_cube(value):
    source = REPO_ROOT / "config" / "release" / "fixture_v2.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["runtime"]["gs_background_color"] = value
    with tempfile.TemporaryDirectory(dir=REPO_ROOT / "config") as temporary:
        config_path = Path(temporary) / "far-field-bad.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        with pytest.raises(Exception, match="gs_background_color"):
            launcher.load_rollout_config(
                config_path, timestamp="20000101_000000"
            )

@pytest.fixture(autouse=True)
def _isolate_cpu_fixture_from_ambient_cuda(
    monkeypatch, isolated_environ, request
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    if request.instance is not None:
        request.instance.isolated_environ = isolated_environ

def test_configuration_error_context_is_conditional(capsys):
    _print_configuration_error(launcher.ConfigError("plain error"))
    _print_configuration_error(RuntimeManifestError("runtime error"))
    _print_configuration_error(
        launcher.ConfigError("structured", path="runner", allowed={"z", "a"}, got="x")
    )

    assert capsys.readouterr().err.splitlines() == [
        "configuration error: plain error",
        "configuration error: runtime error",
        "configuration error: structured",
        "  at: runner  allowed: a, z  got: x",
    ]

def test_schema_diagnostics_use_plain_sorted_allowed_names(capsys, isolated_environ):
    config = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    config["robot"] = "fr33"
    with tempfile.TemporaryDirectory(dir=REPO_ROOT / "config") as temporary:
        path = Path(temporary) / "bad-robot.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        assert launcher.main(
            [
                "--config",
                str(path),
                "--dry-run",
                "--env-file",
                str(isolated_environ),
            ]
        ) == 2

    lines = capsys.readouterr().err.splitlines()
    # Every supported embodiment is listed, comma-separated and sorted, with
    # no Python repr leaking into the diagnostic.
    allowed = ", ".join(sorted(_ROBOT_CLI_NAMES))
    assert lines[1] == f"  at: robot  allowed: {allowed}  got: fr33"
    assert "PosixPath" not in lines[1]
    assert "{" not in lines[1]

def test_schema_v2_close_matches_exclude_rejected_keys(capsys, isolated_environ):
    config = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    config["runner"]["parallel_repeat"] = 2
    with tempfile.TemporaryDirectory(dir=REPO_ROOT / "config") as temporary:
        path = Path(temporary) / "bad-runner-key.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        assert launcher.main(
            [
                "--config",
                str(path),
                "--dry-run",
                "--env-file",
                str(isolated_environ),
            ]
        ) == 2

    stderr = capsys.readouterr().err
    assert "suggestions: parallel_repeats" not in stderr

def test_resolve_plan_uses_loaded_bytes_after_disk_changes():
    raw_on_disk = FIXTURE.read_bytes()
    with tempfile.TemporaryDirectory(dir=REPO_ROOT / "config") as temporary:
        path = Path(temporary) / "provenance.yaml"
        path.write_bytes(raw_on_disk)
        raw_bytes, raw_config = launcher.load_raw_config(path)
        raw_config["description"] = "in-memory mapping"
        path.write_bytes(raw_on_disk + b"\n# changed after load\n")

        plan = launcher.resolve_plan(
            raw_config,
            path=path,
            timestamp="20000101_000000",
        )

    assert plan["raw_bytes"] == raw_bytes
    assert plan["config_sha256"] == hashlib.sha256(raw_bytes).hexdigest()
    assert plan["resolved"]["description"] == "in-memory mapping"

@pytest.mark.parametrize("value", ["8765", "8_765"])
def test_runtime_port_rejects_non_template_strings(value):
    with pytest.raises(launcher.ConfigError, match=r"runtime\.server_port"):
        _runtime_port(
            value,
            "runtime.server_port",
            {"_environment": {}},
        )

def test_runtime_port_accepts_braced_environment_template_and_checks_range():
    context = {"_environment": {"REAL2SIM_SERVER_PORT": "8765"}}
    assert (
        _runtime_port(
            "${REAL2SIM_SERVER_PORT}",
            "runtime.server_port",
            context,
        )
        == 8765
    )

    with pytest.raises(launcher.ConfigError, match=r"runtime\.server_port"):
        _runtime_port(
            "${REAL2SIM_SERVER_PORT}",
            "runtime.server_port",
            {"_environment": {"REAL2SIM_SERVER_PORT": "65536"}},
        )


def test_schema_v2_has_no_raw_extra_args_and_builds_complete_provenance():
    plan = launcher.load_rollout_config(
        REPO_ROOT / "config" / "release" / "fixture_v2.yaml",
        timestamp="20260802_000000",
    )
    assert plan["resolved"]["schema_version"] == 2
    assert "extra_args" not in plan["resolved"]["simulation"]
    assert plan["resolved"]["simulation"]["parameters"]["physics"] == "dynamic"
    assert plan["resolved"]["simulation"]["parameters"]["policy_wrist_contrast"] == 1.0
    assert plan["resolved"]["simulation"]["parameters"][
        "gs_calibration_world_offset"
    ] == [0.0, 0.0, 0.0]
    assert "--physics" in plan["resolved"]["simulation"]["cli_args"]
    assert "--grasp-mode" not in plan["resolved"]["simulation"]["cli_args"]
    assert launcher.build_runner_environment(plan)["GRASP_MODE"] == "rule"

    run_manifest = launcher.prepare_run_manifest(plan)
    assert run_manifest is not None
    assert run_manifest["mode"] == "fixture"
    assert run_manifest["code"]["commit"]
    if run_manifest["superproject"] is not None:
        assert run_manifest["superproject"]["gitlink_commit"] == run_manifest["code"]["commit"]
    assert run_manifest["package_locks"][0]["sha256"]
    assert run_manifest["preflight"] is None
    assert {item["id"] for item in run_manifest["artifacts"]} == {
        "tiny-checkpoint-v1",
        "tiny-scene-v2",
        "tiny-registered-desk-v1",
        "tiny-policy-episode-v1",
        "tiny-repository-assets-v1",
    }
    assert "PATH" in run_manifest["resolved_environment"]
    assert run_manifest["host"]["nvidia"] is None
    # Which machine ran it, at the same grain as the GPUs it already names by
    # UUID. Without this a finished run cannot be matched against what else was
    # on its node.
    assert run_manifest["host"]["hostname"] == platform.node()

def test_schema_v2_rejects_missing_typed_parameter():
    source = yaml.safe_load(
        (REPO_ROOT / "config" / "release" / "fixture_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    del source["simulation"]["parameters"]["substeps"]
    with pytest.raises(launcher.ConfigError, match="missing required keys: substeps"):
        launcher._validate_v2_parameters(
            source["simulation"]["parameters"],
            context={
                "repo_root": str(REPO_ROOT),
                "config_dir": str(REPO_ROOT / "config" / "release"),
                "timestamp": "20260802_000000",
                "robot": "fr3",
            },
            config_dir=REPO_ROOT / "config" / "release",
        )

def test_schema_v2_rejects_unallowed_behavior_environment():
    with mock.patch.dict(os.environ, {"RS_CUP_XY": "0.1 0.2"}, clear=False):
        with pytest.raises(launcher.ConfigError, match="not allowed"):
            launcher.load_rollout_config(
                REPO_ROOT / "config" / "release" / "fixture_v2.yaml",
                timestamp="20260802_000000",
            )

def test_schema_v2_child_environment_drops_unrelated_process_values():
    with mock.patch.dict(os.environ, {"UNRELATED_HOST_VALUE": "must-not-pass"}, clear=False):
        plan = launcher.load_rollout_config(
            REPO_ROOT / "config" / "release" / "fixture_v2.yaml",
            timestamp="20260802_000000",
        )
        environment = launcher.build_runner_environment(plan)
    assert "UNRELATED_HOST_VALUE" not in environment
    assert environment["REAL2SIM_STRICT_LAUNCH"] == "1"

def test_schema_v2_template_uses_resolved_allowlist_not_ambient_environment():
    context = {
        "repo_root": str(REPO_ROOT),
        "config_dir": str(REPO_ROOT / "config"),
        "timestamp": "20260802_000000",
        "robot": "fr3",
        "_environment": {"DECLARED_PATH": "/declared/value"},
    }
    with mock.patch.dict(
        os.environ,
        {"DECLARED_PATH": "/ambient/value", "UNDECLARED_PATH": "/ambient/only"},
        clear=False,
    ):
        assert (
            launcher._render_template(
                "${DECLARED_PATH}/scene", "test.path", context
            )
            == "/declared/value/scene"
        )
        with pytest.raises(launcher.ConfigError, match="UNDECLARED_PATH"):
            launcher._render_template(
                "${UNDECLARED_PATH}/scene", "test.path", context
            )

def test_retired_gpu_ids_environment_contract_is_rejected(
    monkeypatch, tmp_path: Path
):
    fixture = REPO_ROOT / "config" / "release" / "fixture_v2.yaml"
    monkeypatch.setenv("REAL2SIM_GPU_IDS", "0")
    with pytest.raises(launcher.ConfigError, match="REAL2SIM_GPU_IDS is retired"):
        launcher.load_rollout_config(fixture)

    monkeypatch.delenv("REAL2SIM_GPU_IDS")
    source = yaml.safe_load(fixture.read_text(encoding="utf-8"))
    source["reproducibility"]["environment"]["allowlist"].append(
        "REAL2SIM_GPU_IDS"
    )
    stale = tmp_path / "stale-gpu-environment.yaml"
    stale.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    with pytest.raises(launcher.ConfigError, match="REAL2SIM_GPU_IDS is retired"):
        launcher.load_rollout_config(stale, repo_root=REPO_ROOT)

def test_retired_portable_policy_endpoint_is_rejected(
    monkeypatch, tmp_path: Path
):
    fixture = REPO_ROOT / "config" / "release" / "fixture_v2.yaml"
    # An empty value selected the local checkpoint before the bridge was removed.
    monkeypatch.setenv("PORTABLE_POLICY_SERVER", "")
    launcher.load_rollout_config(fixture)

    monkeypatch.setenv("PORTABLE_POLICY_SERVER", "ws://127.0.0.1:8000")
    with pytest.raises(
        launcher.ConfigError, match="PORTABLE_POLICY_SERVER is retired"
    ):
        launcher.load_rollout_config(fixture)

    monkeypatch.delenv("PORTABLE_POLICY_SERVER")
    source = yaml.safe_load(fixture.read_text(encoding="utf-8"))
    source["reproducibility"]["environment"]["allowlist"].append(
        "PORTABLE_POLICY_SERVER"
    )
    stale = tmp_path / "stale-policy-transport.yaml"
    stale.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    with pytest.raises(
        launcher.ConfigError, match="PORTABLE_POLICY_SERVER is retired"
    ):
        launcher.load_rollout_config(stale, repo_root=REPO_ROOT)

def test_cuda_visibility_can_only_be_captured_from_the_process(monkeypatch):
    source = yaml.safe_load(
        (
            REPO_ROOT / "config" / "release" / "fixture_v2.yaml"
        ).read_text(encoding="utf-8")
    )
    source["reproducibility"]["environment"]["allowlist"].append(
        "CUDA_VISIBLE_DEVICES"
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-interactive")

    with tempfile.TemporaryDirectory(dir=REPO_ROOT / "config") as temporary:
        path = Path(temporary) / "capture-cvd.yaml"
        path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
        plan = launcher.load_rollout_config(
            path,
            timestamp="20260824_000000",
        )
        assert plan["resolved"]["reproducibility"]["environment"][
            "CUDA_VISIBLE_DEVICES"
        ] == "GPU-interactive"

        source["reproducibility"]["environment"]["values"][
            "CUDA_VISIBLE_DEVICES"
        ] = "GPU-from-yaml"
        path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
        with pytest.raises(
            launcher.ConfigError,
            match="CUDA_VISIBLE_DEVICES.*only be captured from the process",
        ):
            launcher.load_rollout_config(path)

@pytest.mark.parametrize(
    "assignment",
    (
        "CUDA_VISIBLE_DEVICES=GPU-from-env-file",
        "export CUDA_VISIBLE_DEVICES='GPU-from-env-file'",
        "CUDA_VISIBLE_DEVICES=",
        "  CUDA_VISIBLE_DEVICES = GPU-from-env-file  # stale dotenv value",
    ),
)
@pytest.mark.parametrize("ambient_value", [None, "GPU-from-shell"])
def test_cuda_visibility_from_env_file_is_rejected_before_mutation(
    monkeypatch,
    tmp_path: Path,
    assignment: str,
    ambient_value: str | None,
):
    monkeypatch.delenv("REAL2SIM_INPUTS_ROOT", raising=False)
    if ambient_value is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ambient_value)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "REAL2SIM_INPUTS_ROOT=/inputs\n"
        f"{assignment}\n",
        encoding="utf-8",
    )

    with pytest.raises(launcher.ConfigError) as excinfo:
        launcher.load_env_file(env_file)

    message = str(excinfo.value)
    assert f"{env_file.resolve()}:2" in message
    assert "CUDA_VISIBLE_DEVICES is forbidden in environment files" in message
    assert "delete this assignment" in message
    assert "set CUDA_VISIBLE_DEVICES in the invoking shell" in message
    assert "REAL2SIM_INPUTS_ROOT" not in os.environ
    if ambient_value is None:
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
    else:
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ambient_value

@pytest.mark.parametrize(
    ("assignment", "message"),
    (
        (
            "export\tCUDA_VISIBLE_DEVICES=GPU-from-env-file",
            "invalid environment variable name",
        ),
        (
            "'CUDA_VISIBLE_DEVICES'=GPU-from-env-file",
            "invalid environment variable name",
        ),
        ("CUDA_VISIBLE_DEVICES", "expected KEY=VALUE"),
        ("export CUDA_VISIBLE_DEVICES", "expected KEY=VALUE"),
    ),
)
def test_unsupported_dotenv_cuda_syntax_fails_atomically(
    monkeypatch,
    tmp_path: Path,
    assignment: str,
    message: str,
):
    monkeypatch.delenv("REAL2SIM_INPUTS_ROOT", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-from-shell")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "REAL2SIM_INPUTS_ROOT=/inputs\n"
        f"{assignment}\n",
        encoding="utf-8",
    )

    with pytest.raises(launcher.ConfigError, match=message):
        launcher.load_env_file(env_file)

    assert "REAL2SIM_INPUTS_ROOT" not in os.environ
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-from-shell"

def test_dotenv_comments_and_empty_unrelated_values_are_preserved(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.delenv("REAL2SIM_TEST_EMPTY", raising=False)
    monkeypatch.delenv("REAL2SIM_TEST_VALUE", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "  # CUDA_VISIBLE_DEVICES=GPU-comment-only\n"
        "\n"
        "REAL2SIM_TEST_EMPTY=\n"
        "  REAL2SIM_TEST_VALUE = kept  # unrelated trailing comment\n",
        encoding="utf-8",
    )

    launcher.load_env_file(env_file)

    assert os.environ["REAL2SIM_TEST_EMPTY"] == ""
    assert os.environ["REAL2SIM_TEST_VALUE"] == "kept"

def test_unsupported_multiline_dotenv_value_is_atomic_and_not_scanned_as_a_key(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.delenv("REAL2SIM_INPUTS_ROOT", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-from-shell")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "REAL2SIM_INPUTS_ROOT=/inputs\n"
        'REAL2SIM_MULTILINE="first line\n'
        "CUDA_VISIBLE_DEVICES=body text only\n"
        'last line"\n',
        encoding="utf-8",
    )

    with pytest.raises(launcher.ConfigError) as excinfo:
        launcher.load_env_file(env_file)

    assert "unterminated quoted value" in str(excinfo.value)
    assert "CUDA_VISIBLE_DEVICES is forbidden" not in str(excinfo.value)
    assert "REAL2SIM_INPUTS_ROOT" not in os.environ
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-from-shell"

def _inputs_root_config(tmp_path: Path, **overrides) -> Path:
    """Write a fixture-shaped v2 profile under <tmp>/config that uses {inputs_root}."""

    source = yaml.safe_load(
        (REPO_ROOT / "config" / "release" / "fixture_v2.yaml").read_text(encoding="utf-8")
    )
    source["policy"]["checkpoint"] = "{inputs_root}/checkpoints/pi05_cup"
    source["scenes"] = [{"tag": "tiny_fixture_scene", "path": "{inputs_root}/scenes/tiny"}]
    source["reproducibility"]["support_matrix"] = str(
        REPO_ROOT / "config" / "support-matrix.yaml"
    )
    environment = source["reproducibility"]["environment"]
    environment["allowlist"] = list(environment["allowlist"]) + ["REAL2SIM_INPUTS_ROOT"]
    environment["required"] = list(environment["required"]) + ["REAL2SIM_INPUTS_ROOT"]
    for key, value in overrides.items():
        section, _, field = key.partition(".")
        if field:
            source[section][field] = value
        else:
            source[section] = value
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    path = config_dir / "inputs_root.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return path

def test_inputs_root_is_normalized_once_into_the_plan(tmp_path: Path):
    config_path = _inputs_root_config(tmp_path)
    root = tmp_path / "inputs"
    messy = str(tmp_path / "other" / ".." / "inputs") + "/"
    with mock.patch.dict(os.environ, {"REAL2SIM_INPUTS_ROOT": messy}, clear=False):
        plan = launcher.load_rollout_config(
            config_path, repo_root=tmp_path, timestamp="20260822_000000"
        )
    resolved = plan["resolved"]
    assert resolved["reproducibility"]["inputs_root"] == str(root)
    assert resolved["reproducibility"]["environment"]["REAL2SIM_INPUTS_ROOT"] == str(root)
    assert resolved["policy"]["checkpoint"] == str(root / "checkpoints" / "pi05_cup")
    assert resolved["scenes"] == [{"tag": "tiny_fixture_scene", "path": str(root / "scenes" / "tiny")}]
    # The child environment and the artifact role paths see the same string.
    assert launcher.build_runner_environment(plan)["REAL2SIM_INPUTS_ROOT"] == str(root)
    context = {
        "repo_root": str(REPO_ROOT), "config_dir": str(REPO_ROOT / "config"),
        "timestamp": "20260822_000000", "robot": "fr3",
        "_environment": {"REAL2SIM_INPUTS_ROOT": str(root)}, "inputs_root": str(root),
    }
    assert launcher._render_template(
        "{inputs_root}/checkpoints/pi05_cup", "policy.checkpoint", context
    ) == str(root / "checkpoints/pi05_cup")

def test_missing_inputs_root_is_a_config_error_that_points_at_env_example(tmp_path: Path):
    config_path = _inputs_root_config(tmp_path)
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("REAL2SIM_INPUTS_ROOT", None)
        for name in real2sim_artifacts._LEGACY_INPUT_ENV_NAMES:
            os.environ.pop(name, None)
        with pytest.raises(launcher.ConfigError) as excinfo:
            launcher.load_rollout_config(config_path, repo_root=tmp_path)
        assert "required environment variables are unset: REAL2SIM_INPUTS_ROOT" in str(excinfo.value)
        assert ".env.example" in str(excinfo.value)
        # A stale .env that still carries the retired names gets the migration hint.
        os.environ["REAL2SIM_PINHOLE_WHITE_SCENE"] = "/old/scene"
        with pytest.raises(launcher.ConfigError, match="REAL2SIM_PINHOLE_WHITE_SCENE is retired:"):
            launcher.load_rollout_config(config_path, repo_root=tmp_path)
    # Allowlisted but not required and unset: the template site reports it.
    optional = _inputs_root_config(tmp_path)
    source = yaml.safe_load(optional.read_text(encoding="utf-8"))
    source["reproducibility"]["environment"]["required"] = ["HOME", "PATH"]
    optional.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("REAL2SIM_INPUTS_ROOT", None)
        with pytest.raises(launcher.ConfigError, match=r"scenes\[0\].path uses \{inputs_root\} but REAL2SIM_INPUTS_ROOT is unset"):
            launcher.load_rollout_config(optional, repo_root=tmp_path)

        context = {
            "repo_root": str(REPO_ROOT), "config_dir": str(REPO_ROOT / "config"),
            "timestamp": "20260822_000000", "robot": "fr3",
            "_environment": {}, "inputs_root": None,
        }
        with pytest.raises(launcher.ConfigError) as excinfo:
            launcher._render_template("{inputs_root}/scenes/x", "scenes[0].path", context)
        message = str(excinfo.value)
        assert "scenes[0].path uses {inputs_root} but REAL2SIM_INPUTS_ROOT is unset" in message
        assert ".env.example" in message
        assert "REAL2SIM_CHECKPOINT" not in message
        os.environ["REAL2SIM_CHECKPOINT"] = "/old/checkpoint"
        with pytest.raises(launcher.ConfigError, match="REAL2SIM_CHECKPOINT is retired"):
            launcher._render_template("{inputs_root}/scenes/x", "scenes[0].path", context)

def test_relative_or_tilde_inputs_root_is_rejected(tmp_path: Path):
    config_path = _inputs_root_config(tmp_path)
    for value in ("relative/inputs", "~/inputs", "  "):
        with mock.patch.dict(os.environ, {"REAL2SIM_INPUTS_ROOT": value}, clear=False):
            with pytest.raises(launcher.ConfigError, match="REAL2SIM_INPUTS_ROOT"):
                launcher.load_rollout_config(config_path, repo_root=tmp_path)

def test_inputs_root_field_requires_the_variable_to_be_allowlisted(tmp_path: Path):
    config_path = _inputs_root_config(tmp_path)
    source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    environment = source["reproducibility"]["environment"]
    environment["allowlist"] = [n for n in environment["allowlist"] if n != "REAL2SIM_INPUTS_ROOT"]
    environment["required"] = [n for n in environment["required"] if n != "REAL2SIM_INPUTS_ROOT"]
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    with mock.patch.dict(os.environ, {"REAL2SIM_INPUTS_ROOT": str(tmp_path)}, clear=False):
        with pytest.raises(launcher.ConfigError, match="not in reproducibility.environment.allowlist"):
            launcher.load_rollout_config(config_path, repo_root=tmp_path)

def test_profiles_without_inputs_root_do_not_need_the_variable():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("REAL2SIM_INPUTS_ROOT", None)
        plan = launcher.load_rollout_config(
            REPO_ROOT / "config" / "release" / "fixture_v2.yaml", timestamp="20260822_000000"
        )
    assert plan["resolved"]["reproducibility"]["inputs_root"] is None
    assert "REAL2SIM_INPUTS_ROOT" not in plan["resolved"]["reproducibility"]["environment"]

def test_runtime_paths_require_an_existing_symlink_free_inputs_root(tmp_path: Path):
    config_path = _inputs_root_config(tmp_path)
    missing = tmp_path / "missing-inputs"
    with mock.patch.dict(os.environ, {"REAL2SIM_INPUTS_ROOT": str(missing)}, clear=False):
        plan = launcher.load_rollout_config(
            config_path, repo_root=tmp_path, timestamp="20260822_000000"
        )
    with pytest.raises(launcher.ConfigError, match="REAL2SIM_INPUTS_ROOT is not an existing directory"):
        launcher.validate_runtime_paths(plan)
    real = tmp_path / "real-inputs"
    real.mkdir()
    link = tmp_path / "linked-inputs"
    link.symlink_to(real, target_is_directory=True)
    with mock.patch.dict(os.environ, {"REAL2SIM_INPUTS_ROOT": str(link)}, clear=False):
        plan = launcher.load_rollout_config(
            config_path, repo_root=tmp_path, timestamp="20260822_000000"
        )
    with pytest.raises(launcher.ConfigError, match="symlinks are rejected"):
        launcher.validate_runtime_paths(plan)

def test_resolve_local_path_expands_inputs_root_only_when_allowlisted(tmp_path: Path):
    entry = {"id": "example-tree", "local_path": "{inputs_root}/checkpoints/example"}
    allowlist = ("REAL2SIM_INPUTS_ROOT", "REAL2SIM_POLICY_EPISODE")
    root = tmp_path / "inputs"
    messy = str(tmp_path / "x" / ".." / "inputs")
    resolved = real2sim_artifacts.resolve_local_path(
        entry,
        repo_root=REPO_ROOT,
        environment_allowlist=allowlist,
        environment={"REAL2SIM_INPUTS_ROOT": messy},
    )
    assert resolved == root / "checkpoints" / "example"
    with pytest.raises(ManifestError, match="not in the manifest environment_allowlist"):
        real2sim_artifacts.resolve_local_path(
            entry,
            repo_root=REPO_ROOT,
            environment_allowlist=("REAL2SIM_POLICY_EPISODE",),
            environment={"REAL2SIM_INPUTS_ROOT": str(root)},
        )
    with pytest.raises(ManifestError) as excinfo:
        real2sim_artifacts.resolve_local_path(
            entry,
            repo_root=REPO_ROOT,
            environment_allowlist=allowlist,
            environment={"REAL2SIM_CHECKPOINT": "/old"},
        )
    assert "artifact example-tree local_path uses {inputs_root} but REAL2SIM_INPUTS_ROOT is unset" in str(
        excinfo.value
    )
    assert "REAL2SIM_CHECKPOINT is retired" in str(excinfo.value)
    with pytest.raises(ManifestError, match="must be an absolute path"):
        real2sim_artifacts.resolve_local_path(
            entry,
            repo_root=REPO_ROOT,
            environment_allowlist=allowlist,
            environment={"REAL2SIM_INPUTS_ROOT": "relative/inputs"},
        )
    # {repo_root} entries are untouched by the root.
    fixture = {"id": "fixture", "local_path": "{repo_root}/tests/fixtures/artifacts/checkpoint"}
    assert real2sim_artifacts.resolve_local_path(
        fixture, repo_root=REPO_ROOT, environment_allowlist=allowlist, environment={}
    ) == REPO_ROOT.resolve() / "tests" / "fixtures" / "artifacts" / "checkpoint"

if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
# Grid-template selection, launcher-owned child environment, and the
# cross-field arithmetic the schema enforces before any process starts.
# --------------------------------------------------------------------------


def _fixture_source(**runner_overrides) -> dict:
    source = yaml.safe_load(
        (REPO_ROOT / "config" / "release" / "fixture_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    source["runner"].update(runner_overrides)
    return source


def _load_repo_local(source: dict):
    with tempfile.TemporaryDirectory(dir=REPO_ROOT / "config") as temporary:
        config_dir = Path(temporary) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        path = config_dir / "rollout.yaml"
        path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
        return launcher.load_rollout_config(
            path, repo_root=REPO_ROOT, timestamp="20990101_000000"
        )


def test_b10_ten_rollouts_selects_the_policy_eval_grid_template():
    source = _fixture_source(
        mode="fast_parallel", policy_batch_size=10, repeats=10, expected_rollouts=10
    )
    source["render"]["enabled"] = True

    plan = _load_repo_local(source)

    expected = str(
        REPO_ROOT / "config" / "render_templates" / "rollout_grid_policy_eval_4x3.json"
    )
    assert plan["resolved"]["render"]["forced_posthoc"] is True
    assert plan["resolved"]["render"]["grid_template"] == expected
    assert launcher.build_runner_environment(plan)["GRID_TEMPLATE"] == expected


def test_below_the_posthoc_threshold_keeps_the_clear_headers_template():
    source = _fixture_source(
        mode="fast_parallel", policy_batch_size=5, repeats=10, expected_rollouts=10
    )
    source["render"]["enabled"] = True

    plan = _load_repo_local(source)

    assert plan["resolved"]["render"]["forced_posthoc"] is False
    assert plan["resolved"]["render"]["grid_template"] == str(
        REPO_ROOT
        / "config"
        / "render_templates"
        / "rollout_grid_clear_headers_4x3.json"
    )


def test_official_run_manifest_environment_is_launcher_owned():
    plan = launcher.load_rollout_config(
        REPO_ROOT / "config" / "release" / "fixture_v2.yaml",
        timestamp="20990101_000000",
    )
    forged = {
        "REAL2SIM_STRICT_LAUNCH": "0",
        "REAL2SIM_OFFICIAL_RUN_MANIFEST": "/tmp/forged-run-manifest.json",
    }

    with mock.patch.dict(os.environ, forged, clear=False):
        environment = launcher.build_runner_environment(plan)

    assert environment["REAL2SIM_STRICT_LAUNCH"] == "1"
    assert environment["REAL2SIM_OFFICIAL_RUN_MANIFEST"] == str(
        Path(plan["run_root"]) / "run_manifest.json"
    )


def test_batch_client_order_follows_the_runner_mode():
    sequential = launcher.load_rollout_config(
        REPO_ROOT / "config" / "release" / "fixture_v2.yaml",
        timestamp="20990101_000000",
    )
    tags = [scene["tag"] for scene in sequential["resolved"]["scenes"]]
    assert launcher.build_runner_environment(sequential)[
        "BATCH_CLIENT_ORDER"
    ] == ",".join(tags)

    parallel = _load_repo_local(
        _fixture_source(
            mode="fast_parallel",
            policy_batch_size=2,
            repeats=2,
            expected_rollouts=2,
        )
    )
    assert launcher.build_runner_environment(parallel)["BATCH_CLIENT_ORDER"] == ",".join(
        f"{tag}_run_{index:02d}" for tag in tags for index in (1, 2)
    )


@pytest.mark.parametrize(
    ("render_overrides", "message"),
    (
        (
            {"wrist": False, "wrist_gs": True},
            r"render\.wrist_gs: true requires render\.wrist: true",
        ),
        (
            {"wrist": True, "wrist_gs": False, "clean_video": True},
            r"render\.clean_video: true requires render\.wrist: false",
        ),
    ),
)
def test_render_cross_field_rules_fail_closed(render_overrides, message):
    source = _fixture_source()
    source["render"].update(render_overrides)

    with pytest.raises(launcher.ConfigError, match=message):
        _load_repo_local(source)


@pytest.mark.parametrize(
    ("runner_overrides", "message"),
    (
        (
            {"parallel_repeats": 2},
            "schema_version: 2 uses runner.policy_batch_size",
        ),
        (
            {
                "mode": "sequential_waves",
                "policy_batch_size": 2,
                "repeats": 2,
                "expected_rollouts": 2,
            },
            r"sequential_waves requires runner\.policy_batch_size: len\(scenes\)",
        ),
        (
            {
                "mode": "fast_parallel",
                "policy_batch_size": 4,
                "repeats": 2,
                "expected_rollouts": 2,
            },
            "must be divisible by and >= runner.policy_batch_size",
        ),
        (
            {
                "mode": "fast_parallel",
                "policy_batch_size": 2,
                "repeats": 2,
                "expected_rollouts": 3,
            },
            r"len\(scenes\) \* repeats",
        ),
    ),
)
def test_batch_arithmetic_is_validated_before_launch(runner_overrides, message):
    source = _fixture_source(**runner_overrides)

    with pytest.raises(launcher.ConfigError, match=message):
        _load_repo_local(source)


@pytest.mark.parametrize(
    ("simulation_overrides", "policy_overrides", "message"),
    (
        (
            {"rf_overlap": 24},
            {},
            r"simulation\.rf_overlap must not exceed policy\.exec_steps",
        ),
        (
            {"rf_overlap_step_shift": 2},
            {},
            "single overlap sampling requires simulation.rf_overlap_step_shift: 0",
        ),
        (
            {"rf_overlap_prev_weight": 0.5},
            {},
            "single overlap sampling requires simulation.rf_overlap_prev_weight: null",
        ),
    ),
)
def test_overlap_arithmetic_is_validated_before_launch(
    simulation_overrides, policy_overrides, message
):
    source = _fixture_source()
    source["simulation"].update(simulation_overrides)
    source["policy"].update(policy_overrides)

    with pytest.raises(launcher.ConfigError, match=message):
        _load_repo_local(source)


def test_rollout_config_must_live_under_the_repository_config_directory(tmp_path):
    outside = tmp_path / "rollout.yaml"
    outside.write_text(
        yaml.safe_dump(_fixture_source(), sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(launcher.ConfigError, match="must live inside this repository"):
        launcher.load_rollout_config(
            outside, repo_root=REPO_ROOT, timestamp="20990101_000000"
        )

def _fixture_with_environment_values(values, allowlist=()):
    config = yaml.safe_load(
        (REPO_ROOT / "config" / "release" / "fixture_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    environment = config["reproducibility"]["environment"]
    environment["allowlist"].extend(allowlist)
    environment["values"].update(values)
    return config


def _load_with_environment_values(values, allowlist=()):
    config = _fixture_with_environment_values(values, allowlist)
    with tempfile.TemporaryDirectory(dir=REPO_ROOT / "config") as temporary:
        config_path = Path(temporary) / "environment-values.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        return launcher.load_rollout_config(
            config_path, timestamp="20000101_000000"
        )


def test_environment_values_expand_repo_root():
    """A sealed environment may name a file inside the checkout portably."""
    plan = _load_with_environment_values(
        {"TEST_LOCAL_ASSET": "{repo_root}/config/asset.json"},
        allowlist=["TEST_LOCAL_ASSET"],
    )
    resolved = plan["resolved"]["reproducibility"]["environment"]
    assert resolved["TEST_LOCAL_ASSET"] == str(REPO_ROOT / "config" / "asset.json")
    assert "{repo_root}" not in resolved["TEST_LOCAL_ASSET"]


def test_environment_values_expand_config_dir():
    plan = _load_with_environment_values(
        {"TEST_LOCAL_ASSET": "{config_dir}/asset.json"},
        allowlist=["TEST_LOCAL_ASSET"],
    )
    resolved = plan["resolved"]["reproducibility"]["environment"]
    assert resolved["TEST_LOCAL_ASSET"].endswith("/asset.json")
    assert "{config_dir}" not in resolved["TEST_LOCAL_ASSET"]


def test_environment_values_reject_an_unsupported_template_field():
    """inputs_root is resolved from this very section, so it is not offered."""
    with pytest.raises(launcher.ConfigError, match="unsupported template field"):
        _load_with_environment_values(
            {"TEST_LOCAL_ASSET": "{inputs_root}/asset.json"},
            allowlist=["TEST_LOCAL_ASSET"],
        )


def test_environment_values_reject_an_environment_reference():
    """Defining the sealed environment in terms of itself is circular."""
    with pytest.raises(launcher.ConfigError, match="in terms of itself"):
        _load_with_environment_values(
            {"TEST_LOCAL_ASSET": "${HOME}/asset.json"},
            allowlist=["TEST_LOCAL_ASSET"],
        )


def test_environment_values_leave_a_plain_value_alone():
    plan = _load_with_environment_values(
        {"TEST_LOCAL_ASSET": "/opt/shared/asset.json"},
        allowlist=["TEST_LOCAL_ASSET"],
    )
    resolved = plan["resolved"]["reproducibility"]["environment"]
    assert resolved["TEST_LOCAL_ASSET"] == "/opt/shared/asset.json"
