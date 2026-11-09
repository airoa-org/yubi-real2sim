#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Validate machine-readable evidence produced by the clean-clone verifier.

This helper intentionally uses only the Python standard library.  The verifier
always invokes it with the Python executable created inside the clone under
test; keeping validation here avoids shell pipelines that can hide the exit
status of the command being checked.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Sequence


SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
RUN_EVIDENCE_FILES = (
    "resolved_rollout_config.json",
    "run_manifest.json",
    "rollout_success_summary.json",
    "rollout_success_summary.csv",
    "rollout_success_summary.png",
    "outputs.csv",
    "rollout_grid_10.mp4",
)


class AcceptanceValidationError(ValueError):
    """Raised when an acceptance artifact is absent or internally inconsistent."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcceptanceValidationError(
            f"could not read JSON object {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise AcceptanceValidationError(f"JSON evidence must contain an object: {path}")
    return value


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _require_below(path: str | Path, root: str | Path, label: str) -> Path:
    candidate = _resolved(path)
    parent = _resolved(root)
    try:
        candidate.relative_to(parent)
    except ValueError as exc:
        raise AcceptanceValidationError(
            f"{label} must be below {parent}, got {candidate}"
        ) from exc
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_success_threshold(
    min_successes: int, *, expected_rollouts: int
) -> None:
    if (
        isinstance(min_successes, bool)
        or not isinstance(min_successes, int)
        or min_successes < 1
        or min_successes > expected_rollouts
    ):
        raise AcceptanceValidationError(
            "minimum successes must be an integer from 1 through "
            f"{expected_rollouts}, got {min_successes!r}"
        )


def validate_identity(repo_root: Path) -> None:
    repo = repo_root.resolve()
    expected_prefix = repo / ".venv"
    if Path.cwd().resolve() != repo:
        raise AcceptanceValidationError(
            f"validator cwd is not the clone: expected {repo}, got {Path.cwd().resolve()}"
        )
    if Path(sys.prefix).resolve() != expected_prefix:
        raise AcceptanceValidationError(
            f"Python prefix is not clone-local: expected {expected_prefix}, got {sys.prefix}"
        )
    try:
        import real2sim
        from real2sim._bootstrap import REPO_ROOT
    except ImportError as exc:
        raise AcceptanceValidationError(
            f"clone-local real2sim import failed: {exc}"
        ) from exc
    if Path(REPO_ROOT).resolve() != repo:
        raise AcceptanceValidationError(
            f"real2sim repository root is not the clone: {REPO_ROOT}"
        )
    module_path = Path(real2sim.__file__).resolve()
    _require_below(module_path, repo, "real2sim module")


def validate_artifacts(path: Path, expected_ids: Sequence[str]) -> None:
    payload = _load_object(path)
    if payload.get("status") != "ok":
        raise AcceptanceValidationError(f"artifact status is not ok: {path}")
    records = payload.get("artifacts")
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise AcceptanceValidationError(f"artifact records are malformed: {path}")
    observed = [item.get("id") for item in records]
    if len(observed) != len(set(observed)):
        raise AcceptanceValidationError(f"artifact records contain duplicate ids: {path}")
    if set(observed) != set(expected_ids) or len(observed) != len(expected_ids):
        raise AcceptanceValidationError(
            f"artifact ids differ: expected {sorted(expected_ids)}, got {sorted(map(str, observed))}"
        )
    for record in records:
        digest = record.get("sha256")
        size = record.get("size")
        files = record.get("files")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise AcceptanceValidationError(
                f"artifact {record.get('id')} has an invalid SHA-256"
            )
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise AcceptanceValidationError(
                f"artifact {record.get('id')} has an invalid size"
            )
        if isinstance(files, bool) or not isinstance(files, int) or files < 1:
            raise AcceptanceValidationError(
                f"artifact {record.get('id')} has an invalid file count"
            )


def validate_status(path: Path, kind: str, repo_root: Path | None) -> None:
    payload = _load_object(path)
    if payload.get("status") != "ok":
        raise AcceptanceValidationError(f"{kind} status is not ok: {path}")
    if kind == "runtime":
        if payload.get("check") != "runtime" or payload.get("run_manifest_ready") is not True:
            raise AcceptanceValidationError(f"runtime result is incomplete: {path}")
        if repo_root is None:
            raise AcceptanceValidationError("runtime validation requires --repo-root")
        _require_below(str(payload.get("config", "")), repo_root, "runtime config")
        _require_below(str(payload.get("run_root", "")), repo_root / "outputs", "run root")


def validate_dry_run(
    path: Path,
    *,
    repo_root: Path,
    mode: str,
    expected_rollouts: int,
    run_root_output: Path,
) -> None:
    payload = _load_object(path)
    if payload.get("schema_version") != 2:
        raise AcceptanceValidationError(f"dry-run schema version is not 2: {path}")
    if payload.get("reproducibility_mode") != mode:
        raise AcceptanceValidationError(
            f"dry-run mode differs: expected {mode}, got {payload.get('reproducibility_mode')}"
        )
    if payload.get("expected_rollouts") != expected_rollouts:
        raise AcceptanceValidationError(
            "dry-run expected_rollouts differs: "
            f"expected {expected_rollouts}, got {payload.get('expected_rollouts')}"
        )
    _require_below(str(payload.get("config", "")), repo_root / "config", "dry-run config")
    _require_below(str(payload.get("runner", "")), repo_root, "dry-run runner")
    run_root = _require_below(
        str(payload.get("run_root", "")), repo_root / "outputs", "dry-run run root"
    )
    run_root_output.parent.mkdir(parents=True, exist_ok=True)
    run_root_output.write_text(str(run_root) + "\n", encoding="utf-8")


def validate_summary(
    run_root: Path,
    *,
    expected_rollouts: int,
    min_successes: int,
    evidence_output: Path,
) -> None:
    _validate_success_threshold(
        min_successes, expected_rollouts=expected_rollouts
    )
    root = run_root.resolve()
    if not root.is_dir():
        raise AcceptanceValidationError(f"run root is missing: {root}")
    paths = [root / name for name in RUN_EVIDENCE_FILES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise AcceptanceValidationError(
            "completed run is missing required evidence: " + ", ".join(missing)
        )
    summary = _load_object(root / "rollout_success_summary.json")
    if summary.get("total_runs") != expected_rollouts:
        raise AcceptanceValidationError(
            f"summary total_runs differs: expected {expected_rollouts}, "
            f"got {summary.get('total_runs')}"
        )
    if summary.get("official_success_field") != "new_final_success":
        raise AcceptanceValidationError("summary does not use new_final_success")
    success_count = summary.get("new_final_success_count")
    if (
        isinstance(success_count, bool)
        or not isinstance(success_count, int)
        or success_count < 0
        or success_count > expected_rollouts
    ):
        raise AcceptanceValidationError(
            "summary new_final_success_count is invalid: "
            f"{success_count!r}"
        )
    success_rate = summary.get("new_final_success_rate")
    if (
        isinstance(success_rate, bool)
        or not isinstance(success_rate, (int, float))
        or not math.isfinite(float(success_rate))
    ):
        raise AcceptanceValidationError(
            f"summary new_final_success_rate is invalid: {success_rate!r}"
        )
    expected_rate = success_count / expected_rollouts
    if not math.isclose(
        float(success_rate), expected_rate, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AcceptanceValidationError(
            "summary success count and rate are inconsistent: "
            f"count={success_count}, rate={success_rate}, expected={expected_rate}"
        )
    minimum_rate = min_successes / expected_rollouts
    if success_count < min_successes or float(success_rate) < minimum_rate:
        raise AcceptanceValidationError(
            "rollout success threshold was not met: "
            f"required {min_successes}/{expected_rollouts}, "
            f"observed {success_count}/{expected_rollouts}"
        )

    copied = evidence_output / "run"
    copied.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for source in paths:
        destination = copied / source.name
        shutil.copy2(source, destination)
        records.append(
            {
                "name": source.name,
                "source": str(source),
                "size": source.stat().st_size,
                "sha256": _sha256_file(source),
            }
        )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_root": str(root),
        "expected_rollouts": expected_rollouts,
        "min_successes": min_successes,
        "new_final_success_count": success_count,
        "new_final_success_rate": float(success_rate),
        "files": records,
    }
    (evidence_output / "run-evidence.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_result(
    *,
    output: Path,
    mode: str,
    commit: str,
    repo_root: Path,
    run_root: Path | None,
    min_successes: int | None,
) -> None:
    if COMMIT_RE.fullmatch(commit) is None:
        raise AcceptanceValidationError(f"result commit is invalid: {commit!r}")
    if mode == "full":
        if min_successes is None:
            raise AcceptanceValidationError(
                "full acceptance result requires minimum successes"
            )
        _validate_success_threshold(min_successes, expected_rollouts=10)
    elif min_successes is not None:
        raise AcceptanceValidationError(
            "fixture acceptance result must not set minimum successes"
        )
    payload = {
        "schema_version": 1,
        "status": "ok",
        "mode": mode,
        "commit": commit,
        "repo_root": str(repo_root.resolve()),
        "run_root": str(run_root.resolve()) if run_root is not None else None,
        "min_successes": min_successes,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    identity = subparsers.add_parser("identity")
    identity.add_argument("--repo-root", type=Path, required=True)

    artifacts = subparsers.add_parser("artifacts")
    artifacts.add_argument("--input", type=Path, required=True)
    artifacts.add_argument("--expected", action="append", default=[], required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--input", type=Path, required=True)
    status.add_argument("--kind", choices=("doctor", "runtime"), required=True)
    status.add_argument("--repo-root", type=Path)

    dry_run = subparsers.add_parser("dry-run")
    dry_run.add_argument("--input", type=Path, required=True)
    dry_run.add_argument("--repo-root", type=Path, required=True)
    dry_run.add_argument("--mode", choices=("fixture", "production"), required=True)
    dry_run.add_argument("--expected-rollouts", type=int, required=True)
    dry_run.add_argument("--run-root-output", type=Path, required=True)

    summary = subparsers.add_parser("summary")
    summary.add_argument("--run-root", type=Path, required=True)
    summary.add_argument("--expected-rollouts", type=int, required=True)
    summary.add_argument("--min-successes", type=int, required=True)
    summary.add_argument("--evidence-output", type=Path, required=True)

    result = subparsers.add_parser("result")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--mode", choices=("fixture", "full"), required=True)
    result.add_argument("--commit", required=True)
    result.add_argument("--repo-root", type=Path, required=True)
    result.add_argument("--run-root", type=Path)
    result.add_argument("--min-successes", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "identity":
            validate_identity(args.repo_root)
        elif args.command == "artifacts":
            validate_artifacts(args.input, args.expected)
        elif args.command == "status":
            validate_status(args.input, args.kind, args.repo_root)
        elif args.command == "dry-run":
            validate_dry_run(
                args.input,
                repo_root=args.repo_root,
                mode=args.mode,
                expected_rollouts=args.expected_rollouts,
                run_root_output=args.run_root_output,
            )
        elif args.command == "summary":
            validate_summary(
                args.run_root,
                expected_rollouts=args.expected_rollouts,
                min_successes=args.min_successes,
                evidence_output=args.evidence_output,
            )
        elif args.command == "result":
            write_result(
                output=args.output,
                mode=args.mode,
                commit=args.commit,
                repo_root=args.repo_root,
                run_root=args.run_root,
                min_successes=args.min_successes,
            )
        else:  # pragma: no cover - argparse owns this branch.
            raise AssertionError(args.command)
    except (AcceptanceValidationError, OSError) as exc:
        print(f"acceptance validation error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
