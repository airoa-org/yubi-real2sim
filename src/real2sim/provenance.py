# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Build a fail-closed run manifest for production Real2Sim evaluations."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from .artifacts import ArtifactRecord
from .locked_imports import (
    FULL,
    OPENPI_PACKAGES,
    ROBOSUITE_PACKAGES,
    LockedImportError,
    LockedPackage,
    installed_records,
    require_locked_imports,
)

#: Git dependencies a run imports from their locked, installed distributions.
LOCKED_RUN_IMPORTS: tuple[LockedPackage, ...] = OPENPI_PACKAGES + ROBOSUITE_PACKAGES
# Source checkouts that runs do not use: their packages install from uv.lock
# and are recorded by collect_locked_imports. A profile may still list them;
# they are recorded as retired and ignored, so its YAML bytes stay unchanged
# for replay.
RETIRED_SOURCE_DEPENDENCIES: tuple[str, ...] = ("third_party/openpi", "third_party/robosuite")


SHA_RE = re.compile(r"[0-9a-f]{40}")
GAUSSIAN_WRAPPING_GITLINK = "third_party/GaussianWrapping"
GITHUB_SCP_REMOTE_RE = re.compile(
    r"git@github\.com:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?"
)


class ProvenanceError(ValueError):
    """Raised when required production provenance cannot be collected."""


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: int = 15,
    environment: Mapping[str, str] | None = None,
) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=None if environment is None else dict(environment),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _safe_remote(value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith("git@"):
        return value
    parsed = urlsplit(value)
    if not parsed.scheme:
        return value
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def _supported_production_remote(value: str | None) -> bool:
    """Accept public HTTPS and canonical authenticated GitHub SSH transports."""

    if value is None or value.startswith("https://"):
        return True
    if GITHUB_SCP_REMOTE_RE.fullmatch(value) is not None:
        return True
    parsed = urlsplit(value)
    return parsed.scheme == "ssh" and parsed.hostname in {
        "github.com",
        "ssh.github.com",
    }


def _inside_gaussian_wrapping(root: Path, path: str) -> bool:
    """Whether submodule ``path`` below ``root`` is nested in the gitlinked GaussianWrapping checkout.

    Bootstrap fetches the GaussianWrapping gitlink without its nested
    submodules, because nothing builds or imports them (THIRD_PARTY_NOTICES
    B.1). Their state is recorded but does not gate a production run.
    ``root`` is the Real2Sim checkout or, as a source dependency, that
    GaussianWrapping checkout itself.
    """

    target = (root / path).resolve()
    checkouts = [(root / GAUSSIAN_WRAPPING_GITLINK).resolve()]
    parent = _run(("git", "rev-parse", "--show-superproject-working-tree"), cwd=root)
    if parent and (Path(parent) / GAUSSIAN_WRAPPING_GITLINK).resolve() == root.resolve():
        checkouts.append(root.resolve())
    return any(checkout in target.parents for checkout in checkouts)


def collect_git_submodules(root: Path, *, production: bool) -> list[dict[str, str]]:
    output = _run(("git", "submodule", "status", "--recursive"), cwd=root)
    if output is None:
        if production and (root / ".gitmodules").is_file():
            raise ProvenanceError(f"could not inspect submodules below {root}")
        return []
    records: list[dict[str, str]] = []
    for line in output.splitlines():
        # ``_run`` strips the captured output, which removes the leading
        # space that ``git submodule status`` uses as the "pinned" marker on
        # the first line. A line that starts directly with the 40-hex commit
        # is therefore a pinned submodule, not a malformed record.
        if len(line) >= 40 and SHA_RE.fullmatch(line[:40]) is not None:
            line = " " + line
        if len(line) < 42:
            raise ProvenanceError(f"unexpected git submodule status below {root}: {line!r}")
        marker = line[0]
        commit = line[1:41]
        path = line[42:].split(" ", 1)[0]
        if marker not in {" ", "-", "+", "U"} or SHA_RE.fullmatch(commit) is None:
            raise ProvenanceError(f"unexpected git submodule status below {root}: {line!r}")
        if production and marker != " " and not _inside_gaussian_wrapping(root, path):
            raise ProvenanceError(
                f"production submodule is missing, conflicted, or not at its gitlink: {root / path}"
            )
        records.append(
            {
                "path": path,
                "commit": commit,
                "status": {
                    " ": "pinned",
                    "-": "missing",
                    "+": "different-commit",
                    "U": "conflicted",
                }[marker],
            }
        )
    return records


def collect_superproject(root: Path, *, production: bool) -> dict[str, Any] | None:
    parent_text = _run(("git", "rev-parse", "--show-superproject-working-tree"), cwd=root)
    if not parent_text:
        return None
    parent = Path(parent_text)
    parent_commit = _run(("git", "rev-parse", "HEAD"), cwd=parent)
    parent_status = _run(("git", "status", "--porcelain=v1", "--untracked-files=all"), cwd=parent)
    relative = root.resolve().relative_to(parent.resolve()).as_posix()
    gitlink_line = _run(("git", "ls-tree", "HEAD", "--", relative), cwd=parent)
    fields = gitlink_line.split() if gitlink_line else []
    gitlink = fields[2] if len(fields) >= 4 and fields[0] == "160000" else None
    if production:
        if parent_commit is None or SHA_RE.fullmatch(parent_commit) is None:
            raise ProvenanceError(f"superproject commit is unavailable: {parent}")
        if parent_status:
            raise ProvenanceError(f"production run requires a clean superproject: {parent}")
        code_commit = _run(("git", "rev-parse", "HEAD"), cwd=root)
        if gitlink != code_commit:
            raise ProvenanceError(
                f"superproject gitlink {gitlink} does not select Real2Sim commit {code_commit}"
            )
    remote = _safe_remote(
        _run(("git", "config", "--get", "remote.origin.url"), cwd=parent)
    )
    if production and not _supported_production_remote(remote):
        raise ProvenanceError(
            "production superproject remote must use HTTPS or canonical "
            f"GitHub SSH: {parent}: {remote}"
        )
    return {
        "path": str(parent.resolve()),
        "commit": parent_commit,
        "dirty": bool(parent_status) if parent_status is not None else None,
        "remote": remote,
        "gitlink_path": relative,
        "gitlink_commit": gitlink,
    }


def collect_git_repository(path: Path, *, production: bool) -> dict[str, Any]:
    root_text = _run(("git", "rev-parse", "--show-toplevel"), cwd=path)
    commit = _run(("git", "rev-parse", "HEAD"), cwd=path)
    if root_text is None or commit is None or SHA_RE.fullmatch(commit) is None:
        if production:
            raise ProvenanceError(f"required Git provenance is unavailable for {path}")
        return {"path": str(path.resolve()), "commit": None, "dirty": None, "remote": None}
    root = Path(root_text)
    status = _run(("git", "status", "--porcelain=v1", "--untracked-files=all"), cwd=root)
    if status is None:
        raise ProvenanceError(f"could not inspect Git working tree {root}")
    dirty = bool(status)
    if production and dirty:
        raise ProvenanceError(f"production run requires a clean Git tree: {root}")
    remote = _safe_remote(
        _run(("git", "config", "--get", "remote.origin.url"), cwd=root)
    )
    if production and not _supported_production_remote(remote):
        raise ProvenanceError(
            "production Git remote must use HTTPS or canonical GitHub SSH: "
            f"{root}: {remote}"
        )
    return {
        "path": str(root.resolve()),
        "commit": commit,
        "dirty": dirty,
        "remote": remote,
        "submodules": collect_git_submodules(root, production=production),
    }


def collect_source_dependencies(
    repo_root: Path,
    paths: Sequence[str],
    *,
    production: bool,
    gaussian_wrapping_repo: str | None = None,
) -> list[dict[str, Any]]:
    """``gaussian_wrapping_repo`` is the checkout the GS server imports; production requires it at the gitlink."""

    gaussian = (repo_root / gaussian_wrapping_repo).resolve() if gaussian_wrapping_repo else None
    retired = {(repo_root / path).resolve() for path in RETIRED_SOURCE_DEPENDENCIES}
    records: list[dict[str, Any]] = []
    listed = False
    for raw in paths:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if candidate.resolve() in retired:
            # Nothing runs from this checkout any more (see
            # RETIRED_SOURCE_DEPENDENCIES); collect_locked_imports records
            # what the run imports instead. A leftover directory is ignored.
            records.append(
                {"path": str(candidate), "retired": True, "commit": None, "dirty": None, "remote": None}
            )
            continue
        if not candidate.is_dir():
            if production:
                raise ProvenanceError(f"source dependency is missing: {candidate}")
            records.append({"path": str(candidate), "commit": None, "dirty": None, "remote": None})
            continue
        record = collect_git_repository(candidate, production=production)
        if candidate.resolve() == gaussian:
            listed = True
            gitlink = _run(("git", "rev-parse", f"HEAD:{GAUSSIAN_WRAPPING_GITLINK}"), cwd=repo_root)
            if production and record["commit"] != gitlink:
                raise ProvenanceError(
                    f"{candidate} is at {record['commit']}, but {GAUSSIAN_WRAPPING_GITLINK} pins {gitlink}"
                )
        records.append(record)
    if production and not listed:
        raise ProvenanceError(
            "production run requires runtime.gaussian_wrapping_repo in "
            f"reproducibility.source_dependencies: {gaussian}"
        )
    return records


def collect_locked_imports(
    repo_root: Path,
    *,
    production: bool,
    packages: Sequence[LockedPackage] = LOCKED_RUN_IMPORTS,
    launch_records: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Record the installed Git dependencies this run imports.

    These are not source checkouts: each is a distribution installed from the
    commit ``uv.lock`` pins. A production run requires the ``FULL`` check
    (``real2sim.locked_imports``). ``launch_records`` are the records the
    launcher preflight's full check returned in this process
    (``real2sim.preflight.validate_locked_imports``); they are recorded as
    they are, with ``checked_in: "launcher preflight"``, so the installed
    files are hashed once per launch. Without them the full check runs here
    (``checked_in: "run manifest"``). Each production record keeps the
    guard's ``check: "full"`` and ``checks`` and adds ``verified: true``. A
    fixture run records whatever is installed, unchecked (``verified: false``,
    ``check: null``), or ``None`` fields if absent.
    """

    if not production:
        return [
            {**record, "verified": False, "check": None, "checks": [], "checked_in": None}
            for record in installed_records(packages)
        ]
    if launch_records is not None:
        expected = [(package.module, package.distribution) for package in packages]
        observed = [
            (record.get("module"), record.get("distribution")) for record in launch_records
        ]
        if observed != expected:
            raise ProvenanceError(
                f"the launcher's locked import records cover {observed}, not {expected}"
            )
        if any(record.get("check") != FULL for record in launch_records):
            raise ProvenanceError(
                "the launcher's locked import records did not come from the full check"
            )
        return [
            {**record, "verified": True, "checked_in": "launcher preflight"}
            for record in launch_records
        ]
    try:
        records = require_locked_imports(repo_root, packages, mode=FULL)
    except LockedImportError as exc:
        raise ProvenanceError(f"locked import check failed: {exc}") from exc
    return [{**record, "verified": True, "checked_in": "run manifest"} for record in records]


def collect_lockfiles(
    repo_root: Path,
    paths: Sequence[str],
    *,
    production: bool,
) -> list[dict[str, Any]]:
    if production and not paths:
        raise ProvenanceError("production run requires at least one package lockfile")
    records: list[dict[str, Any]] = []
    for raw in paths:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        try:
            payload = candidate.read_bytes()
        except OSError as exc:
            if production:
                raise ProvenanceError(f"required lockfile is unreadable: {candidate}: {exc}") from exc
            continue
        if production:
            text = payload.decode("utf-8", errors="replace")
            if candidate.name == "uv.lock":
                if 'hash = "sha256:' not in text:
                    raise ProvenanceError(f"uv lock contains no distribution hashes: {candidate}")
            elif candidate.suffix == ".txt":
                requirement_blocks = re.split(r"(?m)(?=^[A-Za-z0-9_.-]+(?:\[[^]]+\])?==)", text)
                unhashed = [
                    block.splitlines()[0]
                    for block in requirement_blocks
                    if re.match(r"^[A-Za-z0-9_.-]+(?:\[[^]]+\])?==", block)
                    and "--hash=sha256:" not in block
                ]
                if unhashed:
                    raise ProvenanceError(
                        f"requirements lock has unhashed entries ({', '.join(unhashed[:3])}): {candidate}"
                    )
            else:
                raise ProvenanceError(f"unsupported production lock format: {candidate}")
        records.append(
            {
                "path": str(candidate.resolve()),
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return records


def collect_nvidia(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    query = _run(
        (
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ),
        cwd=Path.cwd(),
        environment=environment,
    )
    if query is None:
        return None
    gpus: list[dict[str, str]] = []
    for line in query.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 5:
            raise ProvenanceError(f"unexpected nvidia-smi output: {line!r}")
        gpus.append(
            dict(zip(("index", "name", "uuid", "driver_version", "compute_capability"), fields))
        )
    if not gpus:
        return None
    tool_version = _run(
        ("nvidia-smi", "--version"),
        cwd=Path.cwd(),
        environment=environment,
    )
    return {"gpus": gpus, "nvidia_smi_available": True, "tool_version": tool_version}


#: Environment variables that name the job a batch scheduler is running.
#: Scheduler job variables are matched by suffix (<PREFIX>_JOBID,
#: <PREFIX>_JOB_ID, <PREFIX>_QUEUE or <PREFIX>_NODEFILE) rather than by
#: scheduler name.
_SCHEDULER_NAME_SUFFIXES = ("_JOBID", "_JOB_ID", "_QUEUE", "_NODEFILE")


def _scheduler_identity() -> dict[str, str]:
    """The submitting job, as its scheduler names it in this process."""

    return {
        name: value
        for name, value in sorted(os.environ.items())
        if value and name.isupper() and name.endswith(_SCHEDULER_NAME_SUFFIXES)
    }


def build_run_manifest(
    *,
    repo_root: Path,
    config: Mapping[str, Any],
    environment: Mapping[str, str],
    artifact_manifest_path: str,
    artifact_manifest_sha256: str,
    artifacts: Sequence[ArtifactRecord],
    lockfiles: Sequence[str],
    source_dependencies: Sequence[str],
    accelerator_profile: Mapping[str, Any],
    production: bool,
    gaussian_wrapping_repo: str | None = None,
    launch_locked_imports: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    code = collect_git_repository(repo_root, production=production)
    superproject = collect_superproject(repo_root, production=production)
    dependency_records = collect_source_dependencies(
        repo_root,
        source_dependencies,
        production=production,
        gaussian_wrapping_repo=gaussian_wrapping_repo,
    )
    lock_records = collect_lockfiles(repo_root, lockfiles, production=production)
    locked_imports = collect_locked_imports(
        repo_root, production=production, launch_records=launch_locked_imports
    )
    accelerator_kind = str(accelerator_profile.get("kind", ""))
    if accelerator_kind not in {"cpu", "cuda"}:
        raise ProvenanceError("accelerator profile kind must be cpu or cuda")
    nvidia = collect_nvidia(environment) if accelerator_kind == "cuda" else None
    if production and accelerator_kind == "cuda" and nvidia is None:
        raise ProvenanceError("production CUDA run requires readable GPU and driver provenance")
    if production and not artifacts:
        raise ProvenanceError("production run requires verified artifact provenance")
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "production" if production else "fixture",
        "config": dict(config),
        "code": code,
        "superproject": superproject,
        "source_dependencies": dependency_records,
        "locked_imports": locked_imports,
        "package_locks": lock_records,
        "artifact_manifest": {
            "path": artifact_manifest_path,
            "sha256": artifact_manifest_sha256,
        },
        "artifacts": [record.as_dict() for record in artifacts],
        "resolved_environment": dict(sorted(environment.items())),
        "host": {
            # Which machine, by name and by scheduler job. The GPUs are
            # already recorded by UUID, so the host they sit in belongs here at
            # the same grain -- without it a finished run cannot be matched
            # against what else ran on the same host.
            #
            # This is the host that built the manifest, which is the host that
            # ran the work for an ordinary run. A sharded run dispatches its
            # workers to other machines, and their names are not here.
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "python_executable": sys.executable,
            "nvidia": nvidia,
            **({"scheduler": scheduler} if (scheduler := _scheduler_identity()) else {}),
        },
        "accelerator_profile": dict(accelerator_profile),
    }


def write_run_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
