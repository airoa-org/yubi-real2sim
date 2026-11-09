# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Compatibility facade for the Real2Sim launcher implementation."""

from __future__ import annotations

import subprocess
import sys

from real2sim.artifacts import _LEGACY_INPUT_ENV_NAMES
from real2sim.artifacts_binding import (
    _artifact_role_paths,
    _validate_schema_v2_contracts,
)
from real2sim.cli import build_arg_parser, main
from real2sim.config._types import ConfigError
from real2sim.config.parameters import _validate_v2_parameters
from real2sim.config.paths import _render_template
from real2sim.config.reproducibility import _load_support_profile
from real2sim.config.schema import load_raw_config, load_rollout_config, resolve_plan
from real2sim.env_file import load_env_file
from real2sim.gpu_binding import resolve_gpu_contract
from real2sim.launch_provenance import prepare_run_manifest, runtime_config_input_files
from real2sim.preflight import (
    DoctorError,
    _check_policy_python_runtime,
    _validate_task_objects,
    run_doctor,
    validate_joint_trajectory_runtime,
    validate_launch_preflight,
    validate_production_preflight,
    validate_runtime_paths,
)
from real2sim.reporting.completion import rollout_completion_errors
from real2sim.reporting.dry_run import dry_run_view, format_dry_run_table
from real2sim.reporting.posthoc_provenance import (
    _BATCH_POLICY_LINEAGE_FILENAMES,
    _CORE_RENDERER_SOURCE_PATHS,
    _file_sha256,
    _file_state,
    _tracked_diff_identity,
)
from real2sim.runner.argv import _official_static_sim_argv
from real2sim.runner.environment import (
    RUNTIME_SOURCE_MANIFEST_ENV,
    RUNTIME_SOURCE_MANIFEST_SHA256_ENV,
    build_all_shard_environments,
    build_runner_environment,
)
from real2sim.runner.execute import execute

__all__ = [
    "_artifact_role_paths",
    "_BATCH_POLICY_LINEAGE_FILENAMES",
    "build_all_shard_environments",
    "build_arg_parser",
    "build_runner_environment",
    "_check_policy_python_runtime",
    "ConfigError",
    "_CORE_RENDERER_SOURCE_PATHS",
    "DoctorError",
    "dry_run_view",
    "execute",
    "_file_sha256",
    "_file_state",
    "format_dry_run_table",
    "_LEGACY_INPUT_ENV_NAMES",
    "load_env_file",
    "load_raw_config",
    "load_rollout_config",
    "_load_support_profile",
    "_official_static_sim_argv",
    "prepare_run_manifest",
    "_render_template",
    "resolve_gpu_contract",
    "resolve_plan",
    "rollout_completion_errors",
    "run_doctor",
    "runtime_config_input_files",
    "RUNTIME_SOURCE_MANIFEST_ENV",
    "RUNTIME_SOURCE_MANIFEST_SHA256_ENV",
    "subprocess",
    "sys",
    "_tracked_diff_identity",
    "validate_joint_trajectory_runtime",
    "validate_launch_preflight",
    "validate_production_preflight",
    "validate_runtime_paths",
    "_validate_schema_v2_contracts",
    "_validate_task_objects",
    "_validate_v2_parameters",
]

if __name__ == "__main__":
    raise SystemExit(main())
