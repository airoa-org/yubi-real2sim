# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Path, template, and runner filename contracts."""

from __future__ import annotations

import os
from pathlib import Path
import re
import string
from typing import Any, Mapping, Sequence

from real2sim._bootstrap import REPO_ROOT
from real2sim.artifacts import (
    INPUTS_ROOT_ENV,
    INPUTS_ROOT_FIELD,
    _LEGACY_INPUT_ENV_NAMES,
    legacy_input_env_message,
)
from real2sim.config._types import ConfigError, _string

RUNNERS = {
    "fast_parallel": REPO_ROOT / "scripts" / "run_policy_batch.sh",
    "sequential_waves": REPO_ROOT / "scripts" / "run_policy_batch.sh",
}

SUMMARY_TOOL = REPO_ROOT / "robosuite_env" / "tools" / "summarize_policy_rollouts.py"

RESOLVED_CONFIG_FILE = "resolved_rollout_config.json"

CONFIG_SNAPSHOT_FILE = "rollout_config.yaml"

RUN_MANIFEST_FILE = "run_manifest.json"

POSTHOC_RENDER_BATCH_THRESHOLD = 10

POLICY_EVAL_GRID_TEMPLATE = (
    REPO_ROOT / "config" / "render_templates" / "rollout_grid_policy_eval_4x3.json"
)

GRID_TEMPLATES = {
    10: REPO_ROOT
    / "config"
    / "render_templates"
    / "rollout_grid_clear_headers_4x3.json",
    20: REPO_ROOT
    / "config"
    / "render_templates"
    / "rollout_grid_clear_headers_5x5.json",
    30: REPO_ROOT
    / "config"
    / "render_templates"
    / "rollout_grid_clear_headers_6x6.json",
    50: REPO_ROOT
    / "config"
    / "render_templates"
    / "rollout_grid_clear_headers_10x7.json",
    100: REPO_ROOT
    / "config"
    / "render_templates"
    / "rollout_grid_clear_headers_10x12.json",
}

_TEMPLATE_FIELDS = frozenset(
    {
        "repo_root",
        "config_dir",
        "timestamp",
        "robot",
        # The machine-local input root from REAL2SIM_INPUTS_ROOT
        # (reproducibility.environment.allowlist). Canonical uses:
        # policy.checkpoint "{inputs_root}/checkpoints/<name>" and
        # scenes[].path "{inputs_root}/scenes/<scene>". output.* must not use it.
        "inputs_root",
    }
)

_ENV_REFERENCE_RE = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|"
    r"(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)

def _inputs_root_unset_message(path: str) -> str:
    message = (
        f"{path} uses {INPUTS_ROOT_FIELD} but {INPUTS_ROOT_ENV} is unset or not in "
        "reproducibility.environment.allowlist. Set it to the absolute input "
        "root in the repository .env file (see .env.example) and list it in "
        "the profile allowlist"
    )
    stale = _LEGACY_INPUT_ENV_NAMES.intersection(os.environ)
    if stale:
        message += ". " + legacy_input_env_message(stale)
    return message

def _uses_inputs_root_field(value: Any) -> bool:
    """Return whether any string in a raw config mentions ``{inputs_root}``."""

    if isinstance(value, Mapping):
        return any(_uses_inputs_root_field(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_uses_inputs_root_field(child) for child in value)
    return isinstance(value, str) and INPUTS_ROOT_FIELD in value

def _render_template(value: Any, path: str, context: Mapping[str, Any]) -> str:
    template = _string(value, path)
    environment_value = context.get("_environment", os.environ)
    if not isinstance(environment_value, Mapping):
        raise ConfigError("internal template environment must be a mapping")
    template_environment = {
        str(key): str(item) for key, item in environment_value.items()
    }
    missing_environment = sorted(
        {
            match.group("braced") or match.group("plain")
            for match in _ENV_REFERENCE_RE.finditer(template)
            if (match.group("braced") or match.group("plain"))
            not in template_environment
        }
    )
    if missing_environment:
        message = (
            f"{path} references unset environment variable(s): "
            + ", ".join(missing_environment)
            + ". Set them in the repository .env file (see .env.example) "
            "or in the process environment"
        )
        legacy = _LEGACY_INPUT_ENV_NAMES.intersection(missing_environment)
        if legacy:
            message += ". " + legacy_input_env_message(legacy)
        raise ConfigError(message)
    expanded = _ENV_REFERENCE_RE.sub(
        lambda match: template_environment[
            match.group("braced") or match.group("plain")
        ],
        template,
    )
    for _literal, field_name, format_spec, conversion in string.Formatter().parse(expanded):
        if field_name is None:
            continue
        if field_name not in _TEMPLATE_FIELDS:
            raise ConfigError(f"{path} uses unsupported template field {{{field_name}}}")
        if format_spec or conversion:
            raise ConfigError(f"{path} must not use template format specs or conversions")
        if field_name == "inputs_root" and context.get("inputs_root") is None:
            raise ConfigError(_inputs_root_unset_message(path))
    try:
        return expanded.format_map(context)
    except (KeyError, ValueError) as exc:
        raise ConfigError(f"could not render {path}: {exc}") from exc

def _resolved_path(value: Any, path: str, context: Mapping[str, Any], config_dir: Path) -> Path:
    rendered = _render_template(value, path, context)
    candidate = Path(rendered).expanduser()
    if not candidate.is_absolute():
        candidate = config_dir / candidate
    # Normalize ``.`` / ``..`` without resolving symlinks.  In particular, a
    # virtualenv's ``bin/python`` is commonly a symlink; replacing it with the
    # base interpreter silently drops that virtualenv's sys.prefix and packages.
    return Path(os.path.abspath(candidate))
