# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Reproducibility profile and environment validation."""

from __future__ import annotations

import os
from pathlib import Path
import re
import string
from typing import Any, Mapping, Sequence

from real2sim._bootstrap import yaml
from real2sim.artifacts import (
    INPUTS_ROOT_ENV,
    INPUTS_ROOT_FIELD,
    _LEGACY_INPUT_ENV_NAMES,
    ManifestError,
    legacy_input_env_message,
    resolve_inputs_root,
)
from real2sim.config._types import (
    ConfigError,
    _choice,
    _mapping,
    _string,
    _string_list,
    _unknown_key_error,
)
from real2sim.config.keys import (
    _BEHAVIOR_ENV_NAMES,
    _REPRO_ENVIRONMENT_KEYS,
    _REPRODUCIBILITY_KEYS,
    _SECRET_WORDS,
    RETIRED_GPU_ENV_NAME,
    RETIRED_POLICY_TRANSPORT_ENV_NAME,
)
from real2sim.config.paths import (
    _ENV_REFERENCE_RE,
    _resolved_path,
    _uses_inputs_root_field,
)

def _looks_secret_like(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    words = set(normalized.split("_"))
    return (
        "api_key" in normalized
        or "apikey" in normalized
        or bool(words.intersection(_SECRET_WORDS))
    )


def _retired_gpu_environment_message() -> str:
    return (
        f"{RETIRED_GPU_ENV_NAME} is retired; set CUDA_VISIBLE_DEVICES to select "
        "host GPUs or set runtime.*_cuda_devices in YAML"
    )


def _retired_policy_transport_message() -> str:
    return (
        f"{RETIRED_POLICY_TRANSPORT_ENV_NAME} is retired: the portable WebSocket "
        "policy bridge was removed, and this run would start the local "
        "policy.checkpoint instead of the requested remote policy; unset it"
    )


# Retired variables a profile must not name, each with what replaced it.
_RETIRED_ENVIRONMENT_MESSAGES = {
    RETIRED_GPU_ENV_NAME: _retired_gpu_environment_message,
    RETIRED_POLICY_TRANSPORT_ENV_NAME: _retired_policy_transport_message,
}


def _reject_retired_environment_names(names: Any, prefix: str) -> None:
    for name, message in _RETIRED_ENVIRONMENT_MESSAGES.items():
        if name in names:
            raise ConfigError(prefix + message())

def reject_secret_like_keys(value: Any, *, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if _looks_secret_like(key):
                raise ConfigError(f"secret-like key is forbidden at {path}.{key}")
            reject_secret_like_keys(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            reject_secret_like_keys(child, path=f"{path}[{index}]")

def _environment_references(value: Any) -> set[str]:
    references: set[str] = set()
    if isinstance(value, Mapping):
        for child in value.values():
            references.update(_environment_references(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            references.update(_environment_references(child))
    elif isinstance(value, str):
        references.update(
            match.group("braced") or match.group("plain")
            for match in _ENV_REFERENCE_RE.finditer(value)
        )
    return references

def _load_support_profile(path: Path, name: str) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not read support matrix {path}: {exc}") from exc
    if not isinstance(loaded, dict) or loaded.get("schema_version") != 1:
        raise ConfigError("reproducibility.support_matrix must use schema_version: 1")
    profiles = loaded.get("profiles")
    if not isinstance(profiles, dict) or name not in profiles:
        raise ConfigError(f"unknown reproducibility.accelerator_profile: {name}")
    profile = profiles[name]
    required = {"kind", "architecture", "python", "cuda", "driver", "compiler"}
    if not isinstance(profile, dict) or set(profile) != required:
        raise ConfigError(
            f"support profile {name} must contain exactly: {', '.join(sorted(required))}"
        )
    if profile["kind"] not in {"cpu", "cuda"}:
        raise ConfigError(f"support profile {name}.kind must be cpu or cuda")
    if re.fullmatch(r"\d+\.\d+\.\d+", str(profile["python"])) is None:
        raise ConfigError(
            f"support profile {name}.python must be major.minor.patch"
        )
    return {"name": name, **profile}

_ENVIRONMENT_VALUE_FIELDS = frozenset({"repo_root", "config_dir"})


def _environment_value(raw: Any, path: str, context: Mapping[str, Any]) -> str:
    """Render one ``reproducibility.environment.values`` entry.

    A sealed environment often has to name something inside the checkout --
    a calibration file selected by an env var, for example -- and writing
    those as absolute paths pins the config to one clone. ``{repo_root}`` and
    ``{config_dir}`` are substituted here so it need not be.

    Only those two fields are offered. ``{inputs_root}`` is itself resolved
    from this section and is not known yet, and ``$VAR`` references are refused
    outright: the environment being described is the very thing they would have
    to read, so the lookup is circular.
    """

    value = _string(raw, path, allow_empty=True)
    if _ENV_REFERENCE_RE.search(value):
        raise ConfigError(
            f"{path} must not reference environment variables; the sealed "
            "environment cannot be defined in terms of itself",
            path=path,
        )
    for _literal, field, spec, conversion in string.Formatter().parse(value):
        if field is None:
            continue
        if field not in _ENVIRONMENT_VALUE_FIELDS:
            raise ConfigError(
                f"{path} uses unsupported template field {{{field}}}",
                path=path,
                allowed=sorted(_ENVIRONMENT_VALUE_FIELDS),
                got=field,
            )
        if spec or conversion:
            raise ConfigError(
                f"{path} must not use template format specs or conversions",
                path=path,
            )
    try:
        return value.format_map(context)
    except (KeyError, ValueError) as exc:  # pragma: no cover - guarded above
        raise ConfigError(f"could not render {path}: {exc}", path=path) from exc


def _validate_reproducibility(
    value: Any,
    *,
    raw_config: Mapping[str, Any],
    context: Mapping[str, Any],
    config_dir: Path,
) -> dict[str, Any]:
    if RETIRED_GPU_ENV_NAME in os.environ:
        raise ConfigError(_retired_gpu_environment_message())
    # Before the bridge was removed an empty value selected the local checkpoint.
    if os.environ.get(RETIRED_POLICY_TRANSPORT_ENV_NAME):
        raise ConfigError(_retired_policy_transport_message())
    section = _mapping(value, "reproducibility")
    unknown = sorted(set(section) - _REPRODUCIBILITY_KEYS)
    missing = sorted(_REPRODUCIBILITY_KEYS - set(section))
    if unknown:
        raise _unknown_key_error(
            f"reproducibility contains unknown keys: {', '.join(unknown)}",
            path="reproducibility",
            allowed=sorted(_REPRODUCIBILITY_KEYS),
            got=unknown,
        )
    if missing:
        raise ConfigError(
            f"reproducibility is missing required keys: {', '.join(missing)}",
            path="reproducibility",
            allowed=sorted(_REPRODUCIBILITY_KEYS),
            got=missing,
        )
    mode = _choice(section["mode"], "reproducibility.mode", ("fixture", "production"))

    raw_environment = _mapping(section["environment"], "reproducibility.environment")
    unknown_environment = sorted(set(raw_environment) - _REPRO_ENVIRONMENT_KEYS)
    missing_environment = sorted(_REPRO_ENVIRONMENT_KEYS - set(raw_environment))
    if unknown_environment or missing_environment:
        raise ConfigError(
            "invalid reproducibility.environment keys: "
            f"unknown={unknown_environment}, missing={missing_environment}",
            path="reproducibility.environment",
            allowed=sorted(_REPRO_ENVIRONMENT_KEYS),
            got=unknown_environment or missing_environment,
        )
    allowlist = _string_list(
        raw_environment["allowlist"], "reproducibility.environment.allowlist"
    )
    required = _string_list(
        raw_environment["required"],
        "reproducibility.environment.required",
        allow_empty=True,
    )
    for path_name, names in (("allowlist", allowlist), ("required", required)):
        invalid = sorted(
            name
            for name in names
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", name) is None
        )
        if invalid:
            raise ConfigError(
                f"reproducibility.environment.{path_name} has invalid names: "
                + ", ".join(invalid)
            )
    for path_name, names in (("allowlist", allowlist), ("required", required)):
        _reject_retired_environment_names(
            names,
            f"reproducibility.environment.{path_name} names a retired variable: ",
        )
        legacy = _LEGACY_INPUT_ENV_NAMES.intersection(names)
        if legacy:
            raise ConfigError(
                f"reproducibility.environment.{path_name} names retired variables: "
                + legacy_input_env_message(legacy)
            )
        # The layout overrides are harness-only: a profile selects its layout
        # with the parameter, so the resolved config names the file.
        from robosuite_env.tasks.layout_selection import (
            LAYOUT_OVERRIDE_ENV,
            LAYOUT_PARAMETERS,
        )

        layout_overrides = sorted(set(LAYOUT_OVERRIDE_ENV.values()).intersection(names))
        if layout_overrides:
            raise ConfigError(
                f"reproducibility.environment.{path_name} names harness-only "
                f"layout overrides: {', '.join(layout_overrides)}; select a layout "
                "with simulation.parameters."
                + ", ".join(sorted(LAYOUT_PARAMETERS.values()))
            )
    if not {"HOME", "PATH"}.issubset(allowlist):
        raise ConfigError("reproducibility.environment.allowlist must include HOME and PATH")
    if not set(required).issubset(allowlist):
        raise ConfigError("reproducibility.environment.required must be a subset of allowlist")
    raw_values = _mapping(raw_environment["values"], "reproducibility.environment.values")
    if "CUDA_VISIBLE_DEVICES" in raw_values:
        raise ConfigError(
            "reproducibility.environment.values.CUDA_VISIBLE_DEVICES is forbidden; "
            "CUDA_VISIBLE_DEVICES may only be captured from the process environment"
        )
    _reject_retired_environment_names(
        raw_values, "reproducibility.environment.values names a retired variable: "
    )
    legacy_values = _LEGACY_INPUT_ENV_NAMES.intersection(raw_values)
    if legacy_values:
        raise ConfigError(
            "reproducibility.environment.values names retired variables: "
            + legacy_input_env_message(legacy_values)
        )
    if not set(raw_values).issubset(allowlist):
        raise ConfigError("reproducibility.environment.values keys must be in allowlist")
    values: dict[str, str] = {}
    for key, raw in raw_values.items():
        if _looks_secret_like(key):
            raise ConfigError(f"secret-like environment key is forbidden: {key}")
        values[key] = _environment_value(
            raw, f"reproducibility.environment.values.{key}", context
        )

    referenced = _environment_references(raw_config)
    _reject_retired_environment_names(
        referenced,
        "schema v2 configuration references a retired environment variable: ",
    )
    legacy_references = _LEGACY_INPUT_ENV_NAMES.intersection(referenced)
    if legacy_references:
        raise ConfigError(
            "schema v2 configuration references retired environment variables: "
            + legacy_input_env_message(legacy_references)
        )
    forbidden_references = sorted(referenced - set(allowlist))
    if forbidden_references:
        raise ConfigError(
            "schema v2 configuration references environment variables outside "
            "reproducibility.environment.allowlist: "
            + ", ".join(forbidden_references)
        )
    behavior_overrides = {
        key
        for key in os.environ
        if key in _BEHAVIOR_ENV_NAMES or key.startswith("RS_")
    }
    forbidden_overrides = sorted(behavior_overrides - set(allowlist))
    if forbidden_overrides:
        raise ConfigError(
            "behavior-changing process environment variables are not allowed by the config: "
            + ", ".join(forbidden_overrides)
        )
    resolved_environment = {
        key: values[key] if key in values else os.environ[key]
        for key in allowlist
        if key in values or key in os.environ
    }
    missing_required = sorted(set(required) - set(resolved_environment))
    if missing_required:
        message = "required environment variables are unset: " + ", ".join(missing_required)
        if INPUTS_ROOT_ENV in missing_required:
            message += ". Set it to the absolute input root in the repository .env file (see .env.example)"
            stale = _LEGACY_INPUT_ENV_NAMES.intersection(os.environ)
            if stale:
                message += ". " + legacy_input_env_message(stale)
        raise ConfigError(message)
    # Normalize the single input root once so the child environment, the run
    # manifest, artifact resolution, and every {inputs_root} template see the
    # same absolute string (abspath collapses . and .. without following
    # symlinks, like _resolved_path).
    inputs_root: str | None = None
    if INPUTS_ROOT_ENV in resolved_environment:
        try:
            normalized_root = resolve_inputs_root(resolved_environment)
        except ManifestError as exc:
            raise ConfigError(str(exc)) from exc
        assert normalized_root is not None
        inputs_root = str(normalized_root)
        resolved_environment[INPUTS_ROOT_ENV] = inputs_root
    if _uses_inputs_root_field(raw_config) and INPUTS_ROOT_ENV not in allowlist:
        raise ConfigError(
            f"configuration uses {INPUTS_ROOT_FIELD} but {INPUTS_ROOT_ENV} is not in "
            "reproducibility.environment.allowlist (it must be allowlisted so the "
            "runner and artifact verification receive the same root)"
        )
    resolved_context = dict(context)
    resolved_context["_environment"] = resolved_environment
    resolved_context["inputs_root"] = inputs_root

    raw_artifacts = _mapping(section["artifacts"], "reproducibility.artifacts")
    if not raw_artifacts:
        raise ConfigError("reproducibility.artifacts must not be empty")
    artifacts: dict[str, str] = {}
    for role, artifact_id in raw_artifacts.items():
        if re.fullmatch(r"[a-z][a-z0-9_.:-]{2,127}", role) is None:
            raise ConfigError(f"invalid reproducibility artifact role: {role}")
        artifacts[role] = _string(
            artifact_id, f"reproducibility.artifacts.{role}"
        )

    def resolved_paths(raw: Any, field: str) -> list[str]:
        return [
            str(
                _resolved_path(
                    item,
                    f"{field}[{index}]",
                    resolved_context,
                    config_dir,
                )
            )
            for index, item in enumerate(_string_list(raw, field))
        ]

    manifest_path = _resolved_path(
        section["artifact_manifest"],
        "reproducibility.artifact_manifest",
        resolved_context,
        config_dir,
    )
    support_matrix = _resolved_path(
        section["support_matrix"],
        "reproducibility.support_matrix",
        resolved_context,
        config_dir,
    )
    accelerator_name = _string(
        section["accelerator_profile"], "reproducibility.accelerator_profile"
    )
    accelerator = _load_support_profile(support_matrix, accelerator_name)
    if mode == "production" and accelerator["kind"] != "cuda":
        raise ConfigError("production rollout requires a CUDA accelerator profile")
    return {
        "mode": mode,
        "production": mode == "production",
        "artifact_manifest": str(manifest_path),
        "artifacts": artifacts,
        "lockfiles": resolved_paths(section["lockfiles"], "reproducibility.lockfiles"),
        "source_dependencies": resolved_paths(
            section["source_dependencies"], "reproducibility.source_dependencies"
        ),
        "support_matrix": str(support_matrix),
        "accelerator_profile": accelerator,
        "environment": dict(sorted(resolved_environment.items())),
        "inputs_root": inputs_root,
    }
