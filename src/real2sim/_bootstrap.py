# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Repository root discovery used by Real2Sim submodules.

The module only locates the checkout; it does not edit ``sys.path``. The
``real2sim``, ``robosuite_env`` and ``scene`` packages resolve from the
installed (editable) distribution.
"""

from __future__ import annotations

from pathlib import Path


try:
    import yaml  # noqa: F401 -- dependency check
except ImportError as exc:  # pragma: no cover - exercised only in a broken env.
    raise SystemExit(
        "PyYAML is required; run scripts/bootstrap.sh to sync the locked environment"
    ) from exc


def _discover_repo_root() -> Path:
    source_candidate = Path(__file__).resolve().parents[2]
    cwd = Path.cwd().resolve()
    candidates = (source_candidate, cwd, *cwd.parents)
    for candidate in candidates:
        if (
            (candidate / "scripts" / "run_policy_batch.sh").is_file()
            and (candidate / "robosuite_env").is_dir()
        ):
            return candidate
    raise RuntimeError(
        "Real2Sim repository resources were not found; run the installed "
        "entrypoint from a Real2Sim checkout"
    )

REPO_ROOT = _discover_repo_root()
