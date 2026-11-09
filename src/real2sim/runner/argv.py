# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Argument-vector helpers shared by the runner and worker contract."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from real2sim.config._types import ConfigError

#: Module scripts/run_policy_batch.sh starts each rollout worker with, as
#: ``<python> -m robosuite_env.sim <worker arguments>``.
SIM_WORKER_MODULE = "robosuite_env.sim"


def worker_command_head_length(argv: Sequence[str]) -> int | None:
    """Return how many leading tokens of a recorded worker command start it.

    The batch runner records ``<python> -m robosuite_env.sim <arguments>``.
    Runs recorded before that started the same worker by file path,
    ``<python> robosuite_env/sim.py <arguments>``; both forms are accepted so
    those runs still validate and replay. Any other head returns ``None``.
    """

    if len(argv) >= 3 and argv[1] == "-m" and argv[2] == SIM_WORKER_MODULE:
        return 3
    if len(argv) >= 2 and argv[1].endswith("sim.py"):
        return 2
    return None


def _extra_argument_occurrences(arguments: Sequence[str], option: str) -> int:
    """Count exact and ``--option=value`` occurrences without changing arguments."""

    return sum(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )

def _extra_argument_value(
    arguments: Sequence[str], option: str
) -> str | None:
    """Return the last exact or ``--option=value`` occurrence."""

    value = None
    for index, argument in enumerate(arguments):
        if argument == option:
            if index + 1 >= len(arguments):
                raise ConfigError(f"{option} requires a value")
            value = arguments[index + 1]
        elif argument.startswith(f"{option}="):
            value = argument.split("=", 1)[1]
    return value

def _official_static_sim_argv(environment: Mapping[str, str]) -> list[str]:
    """Mirror the schema-v2 batch runner's launcher-owned COMMON_ARGS."""

    arguments = [
        "--robot", environment["ROBOT"],
        "--policy-batch-server", f"{environment['SERVER_HOST']}:{environment['SERVER_PORT']}",
        "--config", environment["CONFIG"],
        "--config-src", environment["CONFIG_SRC"],
        "--objects", environment["OBJECTS"],
        "--instruction", environment["INSTRUCTION"],
        "--ah", environment["ACTION_HORIZON"],
        "--exec", environment["EXEC_STEPS"],
        "--rf-overlap", environment["RF_OVERLAP"],
        "--rf-overlap-sampler", environment["RF_OVERLAP_SAMPLER"],
        "--rf-overlap-step-shift", environment["RF_OVERLAP_STEP_SHIFT"],
        "--grip-torque-mult", environment["GRIP_TORQUE_MULT"],
        "--grasp-mode", environment["GRASP_MODE"],
        "--deterministic-seed", environment["DETERMINISTIC_SEED"],
        "--frames", environment["FRAMES"],
        "--max-wall-sec", environment["MAX_WALL_SEC"],
        "--policy-batch-timeout-sec", environment["MAX_WALL_SEC"],
        "--goal-condition", environment["GOAL_CONDITION"],
        "--score-condition", environment["SCORE_CONDITION"],
        "--view-azimuth", environment["VIEW_AZIMUTH"],
        "--view-elevation", environment["VIEW_ELEVATION"],
        "--view-distance", environment["VIEW_DISTANCE"],
        "--strict-assets",
        "--official-run-manifest", environment["REAL2SIM_OFFICIAL_RUN_MANIFEST"],
    ]
    for variable, option in (
        ("RF_OVERLAP_PREV_WEIGHT", "--rf-overlap-prev-weight"),
        ("POLICY_IMAGE_EPISODE", "--policy-image-episode"),
        ("GRIPPER_CALIBRATION", "--gripper-calibration"),
        ("VIEW_LOOKAT", "--view-lookat"),
    ):
        if environment[variable]:
            arguments.extend((option, environment[variable]))
    if environment["RENDER_WRIST"] != "1":
        arguments.append("--no-wrist")
    if environment["WRIST_GS"] == "1":
        arguments.append("--wrist-gs")
    if environment["SHOW_ORIGIN"] != "1":
        arguments.append("--no-origin")
    if environment["CLEAN_VIDEO"] == "1":
        arguments.append("--clean-video")
    try:
        extra_arguments = json.loads(environment["EXTRA_SIM_ARGS_JSON"])
    except (json.JSONDecodeError, TypeError) as exc:
        raise ConfigError("internal schema-v2 simulator arguments are invalid") from exc
    if not isinstance(extra_arguments, list) or any(
        not isinstance(item, str) for item in extra_arguments
    ):
        raise ConfigError("internal schema-v2 simulator arguments must be strings")
    arguments.extend(extra_arguments)
    return arguments

def _argv_value(argv: Sequence[str], option: str) -> str | None:
    value: str | None = None
    prefix = f"{option}="
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == option:
            if index + 1 >= len(argv):
                return None
            value = argv[index + 1]
            index += 2
            continue
        if token.startswith(prefix):
            value = token[len(prefix) :]
        index += 1
    return value
