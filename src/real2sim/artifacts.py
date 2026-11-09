# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Strict external-artifact manifest loading and content verification.

The tree digest is deliberately independent of archive metadata and filesystem
timestamps.  It hashes one UTF-8 record per regular file, sorted by POSIX
relative path::

    <relative-path> NUL <decimal-size> NUL <file-sha256> LF

Symlinks and special files are rejected.  A ``collection`` uses the same
algorithm over the explicitly listed repository-relative files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from real2sim._bootstrap import yaml


SHA256_RE = re.compile(r"[0-9a-f]{64}")
ENV_RE = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|"
    r"(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)
ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{2,95}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
ALLOWED_ACCESS = frozenset({"public", "generated", "fixture"})
ALLOWED_KINDS = frozenset({"file", "tree", "collection"})
ALLOWED_SCHEMES = frozenset(
    {"https", "s3", "gs", "archive", "generated", "fixture"}
)
REQUIRED_ENTRY_FIELDS = frozenset(
    {
        "id",
        "name",
        "version",
        "uri",
        "size",
        "sha256",
        "license",
        "access",
        "created_at",
        "generator_commit",
        "status",
        "kind",
        "local_path",
    }
)
OPTIONAL_ENTRY_FIELDS = frozenset({"files", "description"})

# The single machine-local path root declared in ``.env``. Manifest
# ``local_path`` fields reference it through the ``{inputs_root}`` template
# field (``{inputs_root}/checkpoints/<name>``, ``{inputs_root}/scenes/<scene>``,
# ``{inputs_root}/registered-world/<bundle>``); scene ``background.json`` files
# use the plain ``${REAL2SIM_INPUTS_ROOT}`` environment spelling because the
# scene contract expands environment references, not template fields.
INPUTS_ROOT_ENV = "REAL2SIM_INPUTS_ROOT"
INPUTS_ROOT_FIELD = "{inputs_root}"
# Retired per-input path variables. They are no longer accepted anywhere; the
# names survive only so diagnostics can point a stale .env / profile /
# manifest at the replacement.
_LEGACY_INPUT_ENV_NAMES = frozenset(
    {
        "REAL2SIM_CHECKPOINT",
        "REAL2SIM_PINHOLE_WHITE_SCENE",
        "REAL2SIM_PINHOLE_WHITE_DESK",
        "REAL2SIM_PINHOLE_WHITE_DESK_BUNDLE",
    }
)
LEGACY_INPUT_ENV_HINT = (
    ".env carries only REAL2SIM_INPUTS_ROOT; use {inputs_root}/checkpoints/<name> "
    "(scenes: {inputs_root}/scenes/<scene>, table collision: "
    "${REAL2SIM_INPUTS_ROOT}/registered-world/<bundle>/coacd); "
    "see docs/configuration.md#inputs-root"
)


def legacy_input_env_message(names: Iterable[str]) -> str:
    """Return the retirement diagnostic for one or more legacy variable names."""

    listed = sorted(set(names))
    verb = "is" if len(listed) == 1 else "are"
    return f"{', '.join(listed)} {verb} retired: {LEGACY_INPUT_ENV_HINT}"


def resolve_inputs_root(
    environment: Mapping[str, str] | None = None,
    *,
    repo_root: Path | None = None,
) -> Path | None:
    """Return the normalized ``REAL2SIM_INPUTS_ROOT`` or ``None`` when unset.

    The value must be absolute: ``robosuite_env.contracts.scene`` expands the
    raw ``${REAL2SIM_INPUTS_ROOT}`` reference in several separate processes, so
    nothing can normalize a relative value on their behalf. ``.`` / ``..``
    are collapsed without resolving symlinks (the same normalization the
    launcher applies to every other path). Existence is not checked here;
    ``--check-runtime`` and artifact verification enforce it.
    """

    del repo_root  # reserved: the root is never repo-relative
    env = os.environ if environment is None else environment
    raw = env.get(INPUTS_ROOT_ENV)
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        raise ManifestError(f"{INPUTS_ROOT_ENV} is set but empty; see .env.example")
    if value.startswith("~") or not os.path.isabs(value):
        raise ManifestError(
            f"{INPUTS_ROOT_ENV} must be an absolute path (no '~', no relative path): "
            f"{value!r}; see .env.example"
        )
    return Path(os.path.abspath(value))


class ManifestError(ValueError):
    """Raised when a manifest or installed artifact violates its contract."""


@dataclass(frozen=True)
class ContentDigest:
    size: int
    sha256: str
    files: int


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    name: str
    version: str
    uri: str
    license: str
    access: str
    created_at: str
    generator_commit: str | None
    kind: str
    path: str
    size: int
    sha256: str
    files: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.artifact_id,
            "name": self.name,
            "version": self.version,
            "uri": self.uri,
            "license": self.license,
            "access": self.access,
            "created_at": self.created_at,
            "generator_commit": self.generator_commit,
            "kind": self.kind,
            "path": self.path,
            "size": self.size,
            "sha256": self.sha256,
            "files": self.files,
        }


@dataclass(frozen=True)
class ArtifactPathState:
    """Filesystem identity used to detect changes after content verification."""

    relative_path: str
    kind: str
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int  # noqa: V107  # ArtifactPathState equality compares it


@dataclass(frozen=True)
class ArtifactFilesystemState:
    artifact_id: str
    paths: tuple[ArtifactPathState, ...]


@dataclass(frozen=True)
class ArtifactVerification:
    """Immutable proof produced by one in-process artifact verification."""

    manifest_path: str
    manifest_sha256: str
    repo_root: str
    artifact_ids: tuple[str, ...]
    records: tuple[ArtifactRecord, ...]
    filesystem_states: tuple[ArtifactFilesystemState, ...]


def _undeclared_artifact(artifact_id: str) -> ManifestError:
    return ManifestError(f"artifact is not declared in the manifest: {artifact_id}")


def sha256_file(path: Path) -> ContentDigest:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise ManifestError(f"could not hash artifact file {path}: {exc}") from exc
    return ContentDigest(size=size, sha256=digest.hexdigest(), files=1)


def _hash_records(root: Path, relative_paths: Iterable[Path]) -> ContentDigest:
    digest = hashlib.sha256()
    total_size = 0
    file_count = 0
    seen: set[str] = set()
    for relative in sorted(relative_paths, key=lambda item: item.as_posix()):
        relative_text = relative.as_posix()
        if relative.is_absolute() or relative_text in {"", "."} or ".." in relative.parts:
            raise ManifestError(f"artifact member must be a safe relative path: {relative}")
        if relative_text in seen:
            raise ManifestError(f"duplicate artifact member: {relative_text}")
        seen.add(relative_text)
        candidate = root / relative
        if candidate.is_symlink():
            raise ManifestError(f"artifact symlinks are forbidden: {candidate}")
        if not candidate.is_file():
            raise ManifestError(f"artifact member is not a regular file: {candidate}")
        item = sha256_file(candidate)
        digest.update(relative_text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.size).encode("ascii"))
        digest.update(b"\0")
        digest.update(item.sha256.encode("ascii"))
        digest.update(b"\n")
        total_size += item.size
        file_count += 1
    if file_count == 0:
        raise ManifestError(f"artifact contains no regular files: {root}")
    return ContentDigest(size=total_size, sha256=digest.hexdigest(), files=file_count)


def sha256_tree(path: Path) -> ContentDigest:
    if path.is_symlink() or not path.is_dir():
        raise ManifestError(f"artifact tree is not a directory: {path}")
    relative_paths: list[Path] = []
    try:
        for candidate in path.rglob("*"):
            if candidate.is_symlink():
                raise ManifestError(f"artifact symlinks are forbidden: {candidate}")
            if candidate.is_file():
                relative_paths.append(candidate.relative_to(path))
            elif not candidate.is_dir():
                raise ManifestError(f"artifact special files are forbidden: {candidate}")
    except OSError as exc:
        raise ManifestError(f"could not inspect artifact tree {path}: {exc}") from exc
    return _hash_records(path, relative_paths)


def sha256_collection(root: Path, members: Sequence[str]) -> ContentDigest:
    if root.is_symlink() or not root.is_dir():
        raise ManifestError(f"artifact collection root is not a directory: {root}")
    if not isinstance(members, list) or not members or not all(isinstance(item, str) for item in members):
        raise ManifestError("collection artifact files must be a non-empty string list")
    return _hash_records(root, [Path(item) for item in members])


def manifest_sha256(path: Path) -> str:
    return sha256_file(path).sha256


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} must be a non-empty string")
    result = value.strip()
    if "\0" in result or "\n" in result or "\r" in result:
        raise ManifestError(f"{field} must not contain NUL or newlines")
    return result


def _validate_uri(value: Any, field: str) -> str:
    uri = _nonempty_string(value, field)
    parsed = urlsplit(uri)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ManifestError(
            f"{field} must use one of these schemes: {', '.join(sorted(ALLOWED_SCHEMES))}"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ManifestError(f"{field} must not embed credentials, query strings, or fragments")
    if parsed.scheme == "https" and not parsed.netloc:
        raise ManifestError(f"{field} is not an absolute HTTPS URI")
    return uri


def _validate_created_at(value: Any, field: str) -> str:
    text = _nonempty_string(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestError(f"{field} must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ManifestError(f"{field} must include a timezone")
    return text


def _validate_collection_members(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) for item in value
    ):
        raise ManifestError(f"{field} must be a non-empty string list")

    members: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        member_field = f"{field}[{index}]"
        member = _nonempty_string(item, member_field)
        relative = Path(member)
        canonical = relative.as_posix()
        if (
            relative.is_absolute()
            or canonical in {"", "."}
            or ".." in relative.parts
            or canonical != member
        ):
            raise ManifestError(
                f"{member_field} must be a canonical POSIX relative path without '..'"
            )
        if canonical in seen:
            raise ManifestError(f"{field} contains duplicate member: {canonical}")
        seen.add(canonical)
        members.append(canonical)
    return members


def _validate_entry(raw: Any, index: int) -> dict[str, Any]:
    field_root = f"artifacts[{index}]"
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise ManifestError(f"{field_root} must be a mapping with string keys")
    unknown = sorted(set(raw) - REQUIRED_ENTRY_FIELDS - OPTIONAL_ENTRY_FIELDS)
    missing = sorted(REQUIRED_ENTRY_FIELDS - set(raw))
    if unknown:
        raise ManifestError(f"{field_root} contains unknown keys: {', '.join(unknown)}")
    if missing:
        raise ManifestError(f"{field_root} is missing keys: {', '.join(missing)}")

    entry = dict(raw)
    artifact_id = _nonempty_string(entry["id"], f"{field_root}.id")
    if ID_RE.fullmatch(artifact_id) is None:
        raise ManifestError(f"{field_root}.id is not a safe artifact identifier")
    for key in ("name", "version", "license", "local_path"):
        entry[key] = _nonempty_string(entry[key], f"{field_root}.{key}")
    entry["uri"] = _validate_uri(entry["uri"], f"{field_root}.uri")
    entry["created_at"] = _validate_created_at(
        entry["created_at"], f"{field_root}.created_at"
    )
    if entry["access"] not in ALLOWED_ACCESS:
        raise ManifestError(f"{field_root}.access must be one of {sorted(ALLOWED_ACCESS)}")
    if entry["kind"] not in ALLOWED_KINDS:
        raise ManifestError(f"{field_root}.kind must be one of {sorted(ALLOWED_KINDS)}")
    if entry["status"] not in {"ready", "pending"}:
        raise ManifestError(f"{field_root}.status must be ready or pending")

    generator_commit = entry["generator_commit"]
    if generator_commit is not None:
        generator_commit = _nonempty_string(
            generator_commit, f"{field_root}.generator_commit"
        )
        if generator_commit != "not-applicable" and COMMIT_RE.fullmatch(generator_commit) is None:
            raise ManifestError(
                f"{field_root}.generator_commit must be a full Git SHA, not-applicable, or null"
            )
    entry["generator_commit"] = generator_commit

    if entry["status"] == "ready":
        if isinstance(entry["size"], bool) or not isinstance(entry["size"], int) or entry["size"] < 0:
            raise ManifestError(f"{field_root}.size must be a non-negative integer")
        if not isinstance(entry["sha256"], str) or SHA256_RE.fullmatch(entry["sha256"]) is None:
            raise ManifestError(f"{field_root}.sha256 must be 64 lowercase hexadecimal characters")
    else:
        if entry["size"] is not None or entry["sha256"] is not None:
            raise ManifestError(f"{field_root} pending entries must use null size and sha256")
        if entry["access"] == "public":
            raise ManifestError(f"{field_root} pending artifacts cannot claim public access")

    if entry["kind"] == "collection":
        entry["files"] = _validate_collection_members(
            entry.get("files"), f"{field_root}.files"
        )
    elif "files" in entry:
        raise ManifestError(f"{field_root}.files is only valid for collection artifacts")
    return entry


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        raw_bytes = manifest_path.read_bytes()
        loaded = yaml.safe_load(raw_bytes)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ManifestError(f"could not read artifact manifest {manifest_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ManifestError("artifact manifest must contain a mapping")
    unknown = sorted(set(loaded) - {"schema_version", "environment_allowlist", "artifacts"})
    missing = sorted({"schema_version", "environment_allowlist", "artifacts"} - set(loaded))
    if unknown or missing:
        raise ManifestError(f"invalid manifest keys: unknown={unknown}, missing={missing}")
    if loaded["schema_version"] != 1:
        raise ManifestError("only artifact manifest schema_version: 1 is supported")
    allowlist = loaded["environment_allowlist"]
    if not isinstance(allowlist, list) or not all(
        isinstance(item, str) and re.fullmatch(r"[A-Z][A-Z0-9_]*", item)
        for item in allowlist
    ):
        raise ManifestError("environment_allowlist must be a list of uppercase variable names")
    if len(set(allowlist)) != len(allowlist):
        raise ManifestError("environment_allowlist contains duplicate names")
    artifacts_raw = loaded["artifacts"]
    if not isinstance(artifacts_raw, list) or not artifacts_raw:
        raise ManifestError("artifacts must be a non-empty list")
    artifacts: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(artifacts_raw):
        entry = _validate_entry(raw, index)
        if entry["id"] in artifacts:
            raise ManifestError(f"duplicate artifact id: {entry['id']}")
        artifacts[entry["id"]] = entry
    return {
        "path": manifest_path,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "environment_allowlist": tuple(allowlist),
        "artifacts": artifacts,
    }


def resolve_local_path(
    entry: Mapping[str, Any],
    *,
    repo_root: Path,
    environment_allowlist: Sequence[str],
    environment: Mapping[str, str] | None = None,
) -> Path:
    env = os.environ if environment is None else environment
    template = str(entry["local_path"])
    referenced = {
        match.group("braced") or match.group("plain") for match in ENV_RE.finditer(template)
    }
    legacy = sorted(referenced.intersection(_LEGACY_INPUT_ENV_NAMES))
    if legacy:
        raise ManifestError(
            f"artifact {entry['id']} local_path references retired variables: "
            + legacy_input_env_message(legacy)
        )
    forbidden = sorted(referenced - set(environment_allowlist))
    if forbidden:
        raise ManifestError(
            f"artifact {entry['id']} local_path references variables outside the manifest allowlist: "
            + ", ".join(forbidden)
        )
    missing = sorted(name for name in referenced if name not in env)
    if missing:
        raise ManifestError(
            f"artifact {entry['id']} local_path references unset variables: "
            + ", ".join(missing)
        )
    expanded = template
    for name in sorted(referenced, key=len, reverse=True):
        expanded = expanded.replace(f"${{{name}}}", env[name]).replace(f"${name}", env[name])
    expanded = expanded.replace("{repo_root}", str(repo_root.resolve()))
    if INPUTS_ROOT_FIELD in expanded:
        if INPUTS_ROOT_ENV not in environment_allowlist:
            raise ManifestError(
                f"artifact {entry['id']} local_path uses {INPUTS_ROOT_FIELD} but "
                f"{INPUTS_ROOT_ENV} is not in the manifest environment_allowlist"
            )
        try:
            inputs_root = resolve_inputs_root(env)
        except ManifestError as exc:
            raise ManifestError(f"artifact {entry['id']}: {exc}") from exc
        if inputs_root is None:
            message = (
                f"artifact {entry['id']} local_path uses {INPUTS_ROOT_FIELD} but "
                f"{INPUTS_ROOT_ENV} is unset; set it in the repository .env file "
                "(see .env.example)"
            )
            stale = sorted(_LEGACY_INPUT_ENV_NAMES.intersection(env))
            if stale:
                message += ". " + legacy_input_env_message(stale)
            raise ManifestError(message)
        expanded = expanded.replace(INPUTS_ROOT_FIELD, str(inputs_root))
    if "{" in expanded or "}" in expanded:
        raise ManifestError(f"artifact {entry['id']} local_path has an unsupported template field")
    candidate = Path(expanded).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return Path(os.path.abspath(candidate))


def verify_artifact(
    manifest: Mapping[str, Any],
    artifact_id: str,
    *,
    repo_root: Path,
    environment: Mapping[str, str] | None = None,
) -> ArtifactRecord:
    try:
        entry = manifest["artifacts"][artifact_id]
    except KeyError as exc:
        raise _undeclared_artifact(artifact_id) from exc
    if entry["status"] != "ready":
        raise ManifestError(f"artifact {artifact_id} is pending and cannot be used")
    path = resolve_local_path(
        entry,
        repo_root=repo_root,
        environment_allowlist=manifest["environment_allowlist"],
        environment=environment,
    )
    if entry["kind"] == "file":
        if path.is_symlink() or not path.is_file():
            raise ManifestError(f"artifact file is missing or not regular: {path}")
        observed = sha256_file(path)
    elif entry["kind"] == "tree":
        observed = sha256_tree(path)
    else:
        observed = sha256_collection(path, entry["files"])
    if observed.size != entry["size"]:
        raise ManifestError(
            f"artifact {artifact_id} size mismatch: expected {entry['size']}, got {observed.size}"
        )
    if observed.sha256 != entry["sha256"]:
        raise ManifestError(
            f"artifact {artifact_id} SHA-256 mismatch: expected {entry['sha256']}, got {observed.sha256}"
        )
    return ArtifactRecord(
        artifact_id=artifact_id,
        name=entry["name"],
        version=entry["version"],
        uri=entry["uri"],
        license=entry["license"],
        access=entry["access"],
        created_at=entry["created_at"],
        generator_commit=entry["generator_commit"],
        kind=entry["kind"],
        path=str(path),
        size=observed.size,
        sha256=observed.sha256,
        files=observed.files,
    )


def verify_artifacts(
    manifest: Mapping[str, Any],
    artifact_ids: Sequence[str],
    *,
    repo_root: Path,
    environment: Mapping[str, str] | None = None,
) -> list[ArtifactRecord]:
    if not artifact_ids:
        raise ManifestError("at least one required artifact id must be selected")
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ManifestError("required artifact ids contain duplicates")
    return [
        verify_artifact(
            manifest,
            artifact_id,
            repo_root=repo_root,
            environment=environment,
        )
        for artifact_id in artifact_ids
    ]


def _artifact_path_state(path: Path, *, root: Path) -> ArtifactPathState:
    try:
        observed = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ManifestError(f"could not stat artifact path {path}: {exc}") from exc
    if stat.S_ISREG(observed.st_mode):
        kind = "file"
    elif stat.S_ISDIR(observed.st_mode):
        kind = "directory"
    else:
        raise ManifestError(f"artifact path is not a regular file or directory: {path}")
    relative_path = "." if path == root else path.relative_to(root).as_posix()
    return ArtifactPathState(
        relative_path=relative_path,
        kind=kind,
        device=observed.st_dev,
        inode=observed.st_ino,
        mode=observed.st_mode,
        size=observed.st_size,
        mtime_ns=observed.st_mtime_ns,
        ctime_ns=observed.st_ctime_ns,
    )


def _artifact_filesystem_state(
    manifest: Mapping[str, Any],
    artifact_id: str,
    *,
    repo_root: Path,
    environment: Mapping[str, str] | None,
) -> ArtifactFilesystemState:
    try:
        entry = manifest["artifacts"][artifact_id]
    except KeyError as exc:
        raise _undeclared_artifact(artifact_id) from exc
    root = resolve_local_path(
        entry,
        repo_root=repo_root,
        environment_allowlist=manifest["environment_allowlist"],
        environment=environment,
    )
    if entry["kind"] == "file":
        candidates = [root]
    elif entry["kind"] == "tree":
        if root.is_symlink() or not root.is_dir():
            raise ManifestError(f"artifact tree is not a directory: {root}")
        try:
            candidates = [root, *root.rglob("*")]
        except OSError as exc:
            raise ManifestError(f"could not inspect artifact tree {root}: {exc}") from exc
    else:
        if root.is_symlink() or not root.is_dir():
            raise ManifestError(f"artifact collection root is not a directory: {root}")
        # A collection digest intentionally ignores unlisted siblings, so its
        # metadata seal must cover only the explicitly selected files as well.
        candidates = [root / member for member in entry["files"]]
    states = tuple(
        _artifact_path_state(candidate, root=root)
        for candidate in sorted(
            candidates,
            key=lambda item: "." if item == root else item.relative_to(root).as_posix(),
        )
    )
    return ArtifactFilesystemState(artifact_id=artifact_id, paths=states)


def _artifact_filesystem_states(
    manifest: Mapping[str, Any],
    artifact_ids: Sequence[str],
    *,
    repo_root: Path,
    environment: Mapping[str, str] | None,
) -> tuple[ArtifactFilesystemState, ...]:
    return tuple(
        _artifact_filesystem_state(
            manifest,
            artifact_id,
            repo_root=repo_root,
            environment=environment,
        )
        for artifact_id in artifact_ids
    )


def verify_artifacts_with_snapshot(
    manifest: Mapping[str, Any],
    artifact_ids: Sequence[str],
    *,
    repo_root: Path,
    environment: Mapping[str, str] | None = None,
) -> ArtifactVerification:
    """Hash artifacts once and seal the result with filesystem metadata.

    Metadata is captured before and after hashing so a file that changes while
    it is being read cannot be accepted with a digest from a different state.
    """

    selected_ids = tuple(artifact_ids)
    if not selected_ids:
        raise ManifestError("at least one required artifact id must be selected")
    if len(set(selected_ids)) != len(selected_ids):
        raise ManifestError("required artifact ids contain duplicates")
    for artifact_id in selected_ids:
        try:
            entry = manifest["artifacts"][artifact_id]
        except KeyError as exc:
            raise _undeclared_artifact(artifact_id) from exc
        if entry["status"] != "ready":
            raise ManifestError(f"artifact {artifact_id} is pending and cannot be used")
    resolved_root = repo_root.resolve()
    before = _artifact_filesystem_states(
        manifest,
        selected_ids,
        repo_root=resolved_root,
        environment=environment,
    )
    records = tuple(
        verify_artifacts(
            manifest,
            selected_ids,
            repo_root=resolved_root,
            environment=environment,
        )
    )
    after = _artifact_filesystem_states(
        manifest,
        selected_ids,
        repo_root=resolved_root,
        environment=environment,
    )
    if before != after:
        raise ManifestError("artifact files changed while their content was verified")
    return ArtifactVerification(
        manifest_path=str(Path(manifest["path"]).resolve()),
        manifest_sha256=str(manifest["sha256"]),
        repo_root=str(resolved_root),
        artifact_ids=selected_ids,
        records=records,
        filesystem_states=after,
    )


def reuse_artifact_verification(
    verification: ArtifactVerification,
    manifest: Mapping[str, Any],
    artifact_ids: Sequence[str],
    *,
    repo_root: Path,
    environment: Mapping[str, str] | None = None,
) -> list[ArtifactRecord]:
    """Reuse an in-process verification after strict manifest and stat checks."""

    selected_ids = tuple(artifact_ids)
    resolved_root = repo_root.resolve()
    if verification.manifest_path != str(Path(manifest["path"]).resolve()):
        raise ManifestError("artifact verification belongs to a different manifest path")
    if verification.manifest_sha256 != manifest["sha256"]:
        raise ManifestError("artifact manifest changed after doctor verification")
    if verification.repo_root != str(resolved_root):
        raise ManifestError("artifact verification belongs to a different repository root")
    if verification.artifact_ids != selected_ids:
        raise ManifestError("artifact selection changed after doctor verification")
    if (
        len(verification.records) != len(selected_ids)
        or len(verification.filesystem_states) != len(selected_ids)
    ):
        raise ManifestError("artifact verification record count is inconsistent")

    for artifact_id, record in zip(selected_ids, verification.records):
        try:
            entry = manifest["artifacts"][artifact_id]
        except KeyError as exc:
            raise _undeclared_artifact(artifact_id) from exc
        if entry["status"] != "ready":
            raise ManifestError(f"artifact {artifact_id} is pending and cannot be used")
        expected_path = resolve_local_path(
            entry,
            repo_root=resolved_root,
            environment_allowlist=manifest["environment_allowlist"],
            environment=environment,
        )
        expected = {
            "artifact_id": artifact_id,
            "name": entry["name"],
            "version": entry["version"],
            "uri": entry["uri"],
            "license": entry["license"],
            "access": entry["access"],
            "created_at": entry["created_at"],
            "generator_commit": entry["generator_commit"],
            "kind": entry["kind"],
            "path": str(expected_path),
            "size": entry["size"],
            "sha256": entry["sha256"],
        }
        observed = {
            field: getattr(record, field)
            for field in expected
        }
        if observed != expected or record.files < 1:
            raise ManifestError(
                f"artifact verification record no longer matches the manifest: {artifact_id}"
            )

    current_states = _artifact_filesystem_states(
        manifest,
        selected_ids,
        repo_root=resolved_root,
        environment=environment,
    )
    if current_states != verification.filesystem_states:
        raise ManifestError("artifact files changed after doctor verification")
    for artifact_id, record, filesystem_state in zip(
        selected_ids,
        verification.records,
        verification.filesystem_states,
    ):
        if filesystem_state.artifact_id != artifact_id or record.files != sum(
            path.kind == "file" for path in filesystem_state.paths
        ):
            raise ManifestError(
                f"artifact verification record count is inconsistent: {artifact_id}"
            )
    return list(verification.records)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.yaml"))
    parser.add_argument("--artifact", action="append", default=[], dest="artifacts")
    parser.add_argument("--all-ready", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--inspect-path",
        type=Path,
        help="Compute canonical size/SHA-256 metadata for a file or extracted tree.",
    )
    parser.add_argument("--inspect-kind", choices=("file", "tree"))
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if (args.inspect_path is None) != (args.inspect_kind is None):
        print(
            "artifact inspection requires both --inspect-path and --inspect-kind",
            file=os.sys.stderr,
        )
        return 2
    if args.inspect_path is not None:
        if args.artifacts or args.all_ready:
            print(
                "artifact inspection cannot be combined with --artifact or --all-ready",
                file=os.sys.stderr,
            )
            return 2
        inspected_path = args.inspect_path.expanduser().resolve()
        try:
            if args.inspect_kind == "file":
                if inspected_path.is_symlink() or not inspected_path.is_file():
                    raise ManifestError(
                        f"artifact file is missing or not regular: {inspected_path}"
                    )
                observed = sha256_file(inspected_path)
            else:
                observed = sha256_tree(inspected_path)
        except ManifestError as exc:
            print(f"artifact inspection error: {exc}", file=os.sys.stderr)
            return 2
        payload = {
            "path": str(inspected_path),
            "kind": args.inspect_kind,
            "size": observed.size,
            "sha256": observed.sha256,
            "files": observed.files,
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                f"{args.inspect_kind} {observed.size} {observed.sha256} "
                f"{observed.files} {inspected_path}"
            )
        return 0
    try:
        manifest = load_manifest(args.manifest)
        selected = list(args.artifacts)
        if args.all_ready:
            selected.extend(
                artifact_id
                for artifact_id, entry in manifest["artifacts"].items()
                if entry["status"] == "ready" and artifact_id not in selected
            )
        records = verify_artifacts(manifest, selected, repo_root=args.repo_root.resolve())
    except ManifestError as exc:
        print(f"artifact verification error: {exc}", file=os.sys.stderr)
        return 2
    payload = {
        "manifest": str(manifest["path"]),
        "manifest_sha256": manifest["sha256"],
        "artifacts": [record.as_dict() for record in records],
        "status": "ok",
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for record in records:
            print(f"ok {record.artifact_id} {record.sha256} {record.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
