# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Typed controller selector resolution."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from real2sim.config._types import ConfigError, _string
from real2sim.fr3_controllers import FR3_CONTROLLERS
from real2sim.runner.argv import _extra_argument_occurrences, _extra_argument_value

_CONTROLLER_SELECTORS = (
    (
        "fr3_controller",
        "--fr3-controller",
        frozenset(FR3_CONTROLLERS),
        "joint_ik",
    ),
    (
        "fr3_nullspace",
        "--fr3-nullspace",
        frozenset({"home", "off"}),
        "home",
    ),
    (
        "yubi_controller",
        "--grip-ctrl",
        frozenset({"cbpc", "kinematic"}),
        "cbpc",
    ),
)

def _resolve_controller_selectors(
    simulation: Mapping[str, Any], extra_args: Sequence[str]
) -> tuple[dict[str, str], list[str]]:
    """Resolve legacy CLI selectors and synthesize only explicitly typed ones.

    Existing configs keep their extra-argument vector exactly as written. A
    typed selector is appended as one option/value pair and cannot also appear
    in ``simulation.extra_args``. Explicitly selecting the historical CbPC
    backend also enables its dynamic 2 ms hook; legacy configs still control
    that auxiliary flag themselves.
    """

    effective_args = list(extra_args)
    controllers: dict[str, str] = {}
    for key, option, choices, default in _CONTROLLER_SELECTORS:
        if key in simulation:
            if _extra_argument_occurrences(extra_args, option):
                raise ConfigError(
                    f"simulation.{key} duplicates simulation.extra_args option {option}"
                )
            value = _string(simulation[key], f"simulation.{key}")
            effective_args.extend((option, value))
        else:
            value = _extra_argument_value(extra_args, option) or default
        if value not in choices:
            raise ConfigError(
                f"simulation.{key} must be one of: {', '.join(sorted(choices))}"
            )
        controllers[key] = value

    dynamic_cbpc_count = _extra_argument_occurrences(
        effective_args, "--dynamic-grip-cbpc"
    )
    if "yubi_controller" in simulation:
        if controllers["yubi_controller"] == "cbpc":
            if not dynamic_cbpc_count:
                effective_args.append("--dynamic-grip-cbpc")
        elif dynamic_cbpc_count:
            raise ConfigError(
                "simulation.yubi_controller "
                f"{controllers['yubi_controller']} cannot be combined with "
                "--dynamic-grip-cbpc"
            )

    return controllers, effective_args

def _validate_nullspace_selection(
    controllers: Mapping[str, str], parameters: Mapping[str, Any]
) -> None:
    """Reject a silent no-op: ``home`` posture mode with a zero gain.

    ``fr3_nullspace: off`` is the explicit way to disable the posture
    objective; the worker then forces the effective gain to 0 and records
    both the configured and the effective value in ``sim_summary.json``.
    """

    gain = float(parameters["joint_ik_nullspace_gain"])
    if controllers["fr3_nullspace"] == "home" and gain <= 0.0:
        raise ConfigError(
            "simulation.fr3_nullspace home requires a positive "
            "simulation.parameters.joint_ik_nullspace_gain; set "
            "simulation.fr3_nullspace: off to disable the posture objective"
        )
