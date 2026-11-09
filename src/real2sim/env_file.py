# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Small repository-local environment file loader."""

from __future__ import annotations

import os
from pathlib import Path
import re

from real2sim.config._types import ConfigError

def load_env_file(path: str | Path, *, override: bool = False) -> None:
    """Load a small shell-compatible ``KEY=VALUE`` file without executing it."""

    env_path = Path(path).expanduser().resolve()
    if not env_path.is_file():
        raise ConfigError(f"environment file not found: {env_path}")
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"could not read environment file {env_path}: {exc}") from exc

    assignments: dict[str, tuple[int, str]] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ConfigError(
                f"{env_path}:{line_number}: expected KEY=VALUE"
            )
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
            raise ConfigError(
                f"{env_path}:{line_number}: invalid environment variable name"
            )
        value = raw_value.strip()
        if value[:1] in {"'", '"'}:
            quote = value[0]
            if len(value) < 2 or value[-1] != quote:
                raise ConfigError(
                    f"{env_path}:{line_number}: unterminated quoted value"
                )
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        if override:
            assignments[key] = (line_number, value)
        else:
            assignments.setdefault(key, (line_number, value))

    # Parsing is deliberately non-mutating. GPU visibility is an allocation
    # boundary owned by the invoking process, so validate every normalized key
    # before committing any otherwise-valid dotenv assignment to os.environ.
    for key, (line_number, _value) in assignments.items():
        if key == "CUDA_VISIBLE_DEVICES":
            raise ConfigError(
                f"{env_path}:{line_number}: CUDA_VISIBLE_DEVICES is forbidden in "
                "environment files; delete this assignment and set "
                "CUDA_VISIBLE_DEVICES in the invoking shell before running real2sim-run"
            )

    for key, (_line_number, value) in assignments.items():
        if override or key not in os.environ:
            os.environ[key] = value
