# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Run an FR3 policy evaluation from one portable YAML configuration.

The YAML owns the policy, task, scenes, simulation, rendering, runtime, and
output settings. Environment-specific paths are expanded from ``${VAR}``
references after loading the optional repository-local ``.env`` file.

``--shards`` overrides ``runner.shards`` without editing YAML. Its compact
syntax is one or more space-separated ``JOB:GPU[,GPU...]:START-STOP`` entries,
for example ``--shards '1001:0,1,2,3:1-50 1002:0,1:51-100'``. The GPU
list is used for GS and simulation, its first selector runs policy inference,
and all other shard settings use the runner/runtime defaults.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from real2sim._bootstrap import REPO_ROOT
from real2sim.artifacts import (
    _LEGACY_INPUT_ENV_NAMES,
    legacy_input_env_message,
)
from real2sim.config._types import ConfigError
from real2sim.config.initial_arm_joints import format_initial_arm_joint_pose_table
from real2sim.config.schema import load_rollout_config
from real2sim.config.shards import parse_shards_cli
from real2sim.env_file import load_env_file
from real2sim.launch_provenance import (
    capture_launch_runtime_generation,
    capture_runtime_selection,
    prepare_run_manifest,
    runtime_config_input_files,
    runtime_source_selection,
)
from real2sim.preflight import validate_launch_preflight
from real2sim.reporting.dry_run import dry_run_view, format_dry_run_table
from real2sim.runner.execute import execute
from real2sim.runtime_manifest import RuntimeManifestError


def _format_error_value(value: Any) -> str:
    if value is None:
        return "<unknown>"
    if isinstance(value, Mapping):
        value = list(value.keys())
    if isinstance(value, (set, frozenset, list, tuple)):
        return ", ".join(sorted(str(item) for item in value)) or "<none>"
    return str(value)


def _print_configuration_error(error: Exception) -> None:
    print(f"configuration error: {error}", file=sys.stderr)
    path = getattr(error, "path", None)
    allowed = getattr(error, "allowed", None)
    got = getattr(error, "got", None)
    if path is None and allowed is None and got is None:
        return
    suggestions = getattr(error, "suggestions", ())
    detail = (
        f"  at: {path or '<unknown>'}"
        f"  allowed: {_format_error_value(allowed)}"
        f"  got: {_format_error_value(got)}"
    )
    if suggestions:
        detail += f"  suggestions: {', '.join(suggestions)}"
    print(detail, file=sys.stderr)


def _timestamp_value(value: str) -> str:
    if re.fullmatch(r"[0-9]{8}_[0-9]{6}", value) is None:
        raise argparse.ArgumentTypeError(
            "--timestamp must match YYYYMMDD_HHMMSS"
        )
    return value


class _ListInitialArmJointPoses(argparse.Action):
    """Print the registry and exit, the way ``--help`` short-circuits parsing."""

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):  # noqa: V107  # argparse.Action.__call__ signature
        print(format_initial_arm_joint_pose_table(), end="")
        parser.exit(0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-initial-arm-joints",
        action=_ListInitialArmJointPoses,
        help=(
            "print every name accepted by "
            "simulation.parameters.initial_arm_joints and exit"
        ),
    )
    parser.add_argument("--config", required=True, type=Path, help="rollout YAML under config/")
    parser.add_argument(
        "--env-file",
        type=Path,
        help="environment file (default: .env at the repository root, when present)",
    )
    validation_mode = parser.add_mutually_exclusive_group()
    validation_mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the YAML schema and print the resolved plan without runtime checks",
    )
    validation_mode.add_argument(
        "--check-runtime",
        action="store_true",
        help="validate runtime paths and print a JSON success result without writing or executing",
    )
    parser.add_argument(
        "--format",
        choices=("json", "table"),
        default="json",
        help="format for --dry-run output (default: json)",
    )
    parser.add_argument(
        "--timestamp",
        type=_timestamp_value,
        help="resolved run timestamp (YYYYMMDD_HHMMSS; default: current time)",
    )
    parser.add_argument(
        "--shards",
        type=parse_shards_cli,
        help=(
            "override runner.shards with space-separated "
            "JOB:GPU[,GPU...]:START-STOP entries"
        ),
    )
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        env_file = args.env_file
        if env_file is not None:
            load_env_file(env_file)
        elif (REPO_ROOT / ".env").is_file():
            load_env_file(REPO_ROOT / ".env")
        stale_input_names = sorted(_LEGACY_INPUT_ENV_NAMES.intersection(os.environ))
        if stale_input_names:
            print(
                "warning: ignoring retired input path variable(s) from .env / the "
                "process environment: " + legacy_input_env_message(stale_input_names),
                file=sys.stderr,
            )
        runtime_source_baseline = None
        if not args.dry_run and not args.check_runtime:
            runtime_source_baseline = capture_launch_runtime_generation(args.config)
        plan = load_rollout_config(
            args.config,
            timestamp=args.timestamp,
            shards_override=args.shards,
        )
        if args.dry_run:
            if args.format == "table":
                print(format_dry_run_table(plan), end="")
            else:
                print(json.dumps(dry_run_view(plan), indent=2, sort_keys=True))
            return 0
        if args.check_runtime:
            # Read every runtime file the launch will seal, exactly as the
            # launch reads it (no symlinks, regular files, stable while read),
            # so a profile that would die at its seal is refused here.
            capture_launch_runtime_generation(args.config)
            capture_runtime_selection(
                runtime_source_selection(
                    plan["repo_root"],
                    extra_files=runtime_config_input_files(plan),
                )
            )
            preflight_report = validate_launch_preflight(plan)
            try:
                run_manifest = prepare_run_manifest(plan, preflight_report)
            finally:
                preflight_report.pop("_artifact_verification", None)
            print(
                json.dumps(
                    {
                        "check": "runtime",
                        "config": str(plan["config_path"]),
                        "config_sha256": plan["config_sha256"],
                        "run_root": str(plan["run_root"]),
                        "run_manifest_ready": run_manifest is not None,
                        "status": "ok",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        return execute(
            plan,
            runtime_source_baseline=runtime_source_baseline,
        )
    except (ConfigError, RuntimeManifestError) as exc:
        _print_configuration_error(exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
