# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Install the pinned release inputs selected by one or more release profiles.

The inputs are extracted folders in two gated Hugging Face repositories: the
policy checkpoints in a model repository and the scenes and meshes in a
dataset repository. ``artifacts/release_downloads.json`` names each repository
once (``huggingface``), with the revision the installer reads and the repository
folder of each destination, and pins every input's extracted tree (the
SHA-256 of each file's path, size and SHA-256). A download that differs from
its pin is never installed. Downloading needs a Hugging Face login
(``.venv/bin/hf auth login``) with the terms of both repositories accepted.

Typical use, from the checkout after ``scripts/bootstrap.sh``::

    .venv/bin/real2sim-download-release \\
        --profile config/release/fr3/flat/a_room_cup_cup1trim.yaml --dry-run

``--repo-root`` defaults to the checkout of an editable install, or else to
the working directory or its nearest ancestor that holds the release catalog.

The PaliGemma tokenizer is intentionally absent. The pinned OpenPI runtime
fetches it anonymously from ``gs://big_vision/paligemma_tokenizer.model`` on
first use and caches it under ``OPENPI_DATA_HOME``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import ssl
import sys
import time
from typing import Iterator, Mapping
import urllib.parse

import yaml

from real2sim.artifacts import ManifestError, sha256_tree
from robosuite_env.contracts.assets import runtime_bundle_ids


RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 6
CHUNK = 8 * 1024 * 1024

HF_PAGE = "https://huggingface.co"
# The catalog holds a revision with this prefix until the repository is
# published at a commit; the installer refuses to read an unpinned repository
# unless --allow-unpinned is given.
UNSET_PREFIX = "UNSET"
UNPINNED_REVISION = "main"
HF_REPO_TYPES = {"model": "", "dataset": "datasets/"}
_HF_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
HF_REPO_ID = re.compile(rf"{_HF_NAME}/{_HF_NAME}")

ENV_INPUTS_ROOT = "REAL2SIM_INPUTS_ROOT"
LEGACY_ENV_NAMES = (
    "REAL2SIM_CHECKPOINT",
    "REAL2SIM_PINHOLE_WHITE_SCENE",
    "REAL2SIM_PINHOLE_WHITE_SCENE_Z0",
    "REAL2SIM_PINHOLE_WHITE_DESK",
    "REAL2SIM_PINHOLE_WHITE_DESK_BUNDLE",
)
PLACEHOLDER_MARKER = "/ABSOLUTE/PATH"


class ReleaseAssetError(RuntimeError):
    """A failure that must stop installation."""


RELEASE_CATALOG = PurePosixPath("artifacts/release_downloads.json")


def default_repo_root() -> Path:
    """The checkout whose release catalog and profiles the installer reads."""
    editable_checkout = Path(__file__).resolve().parents[2]
    cwd = Path.cwd().resolve()
    for candidate in (editable_checkout, cwd, *cwd.parents):
        if (candidate / RELEASE_CATALOG).is_file():
            return candidate
    raise ReleaseAssetError(
        f"{RELEASE_CATALOG} was not found; run from a Real2Sim checkout or pass --repo-root"
    )


def _log(message: str) -> None:
    print(f"[release-assets] {message}", flush=True)


def revision_env(key: str) -> str:
    """The environment variable that overrides one repository's revision."""
    return f"REAL2SIM_HF_{key.upper()}_REVISION"


@dataclass(frozen=True)
class HubRepo:
    """One release repository on the Hugging Face Hub, at the revision to read."""

    key: str
    repo_id: str
    repo_type: str
    revision: str
    folders: tuple[tuple[str, str], ...]

    @property
    def page(self) -> str:
        return f"{HF_PAGE}/{HF_REPO_TYPES.get(self.repo_type, '')}{self.repo_id}"

    def tree_url(self, folder: str) -> str:
        revision = urllib.parse.quote(self.revision, safe="")
        return f"{self.page}/tree/{revision}/{urllib.parse.quote(folder, safe='/')}"

    def problems(self) -> list[str]:
        """Why this repository cannot be read; empty when it can."""
        found = []
        if not HF_REPO_ID.fullmatch(self.repo_id):
            found.append(f"{self.repo_id!r} is not a Hugging Face repository id (OWNER/NAME)")
        if self.repo_type not in HF_REPO_TYPES:
            found.append(f"unknown Hugging Face repository type {self.repo_type!r}")
        if not self.revision or self.revision.startswith(UNSET_PREFIX):
            found.append(
                f"{self.repo_id} has no pinned revision ({self.revision!r}); set "
                f"huggingface.{self.key}.revision in {RELEASE_CATALOG} to its release "
                f"commit, pass --hf-revision {self.key}=COMMIT or set "
                f"{revision_env(self.key)}, or pass --allow-unpinned to read its "
                f"{UNPINNED_REVISION} branch (the tree pins are still verified)"
            )
        return found

    def locate(self, destination: str) -> str | None:
        """The repository folder that holds ``destination``, if this repository has it."""
        for prefix, folder in self.folders:
            if destination.startswith(prefix) and destination != prefix:
                return folder + destination[len(prefix):]
        return None


def hub_repos(
    catalog: dict,
    *,
    revisions: Mapping[str, str] | None = None,
    allow_unpinned: bool = False,
    environ: Mapping[str, str] = os.environ,
) -> dict[str, HubRepo]:
    """The catalog's repositories, each at its pinned or overridden revision.

    A revision comes from ``revisions`` (``--hf-revision``), then the
    environment, then the catalog. ``allow_unpinned`` reads the default branch
    of a repository whose revision is still unset.
    """
    revisions = revisions or {}
    unknown = sorted(set(revisions) - set(catalog["huggingface"]))
    if unknown:
        raise ReleaseAssetError(f"--hf-revision names no catalog repository: {', '.join(unknown)}")
    repos = {}
    for key, entry in catalog["huggingface"].items():
        revision = revisions.get(key) or environ.get(revision_env(key)) or entry["revision"]
        if allow_unpinned and (not revision or revision.startswith(UNSET_PREFIX)):
            revision = UNPINNED_REVISION
        repos[key] = HubRepo(key=key, repo_id=entry["repo_id"], repo_type=entry["repo_type"],
                             revision=revision, folders=tuple(entry["folders"].items()))
    return repos


def locate(pin: dict, repos: Mapping[str, HubRepo]) -> tuple[HubRepo, str]:
    """The repository and folder that hold one pinned input."""
    found = [(repo, folder) for repo in repos.values()
             if (folder := repo.locate(pin["destination"])) is not None]
    if len(found) != 1:
        raise ReleaseAssetError(
            f"{pin['destination']} maps to {len(found)} Hugging Face repositories; expected 1")
    return found[0]


def _hub_snapshot(**kwargs) -> str:
    """``huggingface_hub.snapshot_download``, imported only when a download runs.

    The repository is looked up first. snapshot_download swallows a refused or
    failed lookup and returns a non-empty ``local_dir`` as it is, so without
    this check a gated repository would look like a damaged download.
    """
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise ReleaseAssetError(
            "huggingface_hub is not installed; run scripts/bootstrap.sh"
        ) from exc
    HfApi(endpoint=kwargs.get("endpoint")).repo_info(
        repo_id=kwargs["repo_id"], repo_type=kwargs["repo_type"], revision=kwargs["revision"],
        timeout=kwargs.get("etag_timeout"))
    return snapshot_download(**kwargs)


# requests' connection, timeout and dropped-transfer errors, matched by name so
# that the installer imports neither requests nor huggingface_hub.
_TRANSIENT_NAMES = frozenset({"ConnectionError", "Timeout", "ChunkedEncodingError"})
# requests' TLS and proxy errors. They derive from its ConnectionError and are
# retried like it, because a handshake cut short or a proxy answering 503 can
# pass. Only the causes that _setup_failure names stop the download at once.
_TLS_PROXY_NAMES = frozenset({"SSLError", "ProxyError"})
_TLS_PROXY_HINT = ("check the proxy variables (HTTPS_PROXY, NO_PROXY), the CA bundle "
                   "(REQUESTS_CA_BUNDLE) and the system clock")


def _wrapped(exc: BaseException):
    """``exc`` and every exception it wraps, each once.

    requests raises its ``SSLError`` or ``ProxyError`` with urllib3's
    ``MaxRetryError`` in ``args``. That error keeps its cause in ``reason`` and
    ``__cause__``, and urllib3's ``SSLError`` holds the ``ssl`` error in ``args``.
    """
    seen: set[int] = set()
    pending = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        links = (current.__cause__, current.__context__, getattr(current, "reason", None),
                 *current.args)
        pending.extend(link for link in links if isinstance(link, BaseException))


def _setup_failure(exc: BaseException, status: int | None) -> str | None:
    """The local TLS or proxy setup problem behind ``exc`` that no retry can change.

    These are a certificate that fails verification (an untrusted CA, a wrong
    host name or a wrong clock) and a proxy that asks for authentication.
    """
    if status == 407:
        return "the proxy requires authentication (HTTP 407)"
    for error in _wrapped(exc):
        if (isinstance(error, ssl.SSLCertVerificationError)
                or type(error).__name__ == "CertificateError"):
            return "its TLS certificate could not be verified"
        # http.client reports the status of a refused CONNECT only in its message.
        if isinstance(error, OSError) and "Tunnel connection failed: 407" in str(error):
            return "the proxy requires authentication (HTTP 407)"
    return None


def _tls_or_proxy(exc: BaseException) -> bool:
    return any(_TLS_PROXY_NAMES & {cls.__name__ for cls in type(error).__mro__}
               for error in _wrapped(exc))


def _hub_failure(exc: BaseException, repo: HubRepo) -> ReleaseAssetError | None:
    """A final error for a Hub failure, or None when retrying may help.

    Only connection errors, timeouts and the HTTP statuses in ``RETRY_STATUS``
    (408, 429, 500, 502, 503 and 504) are retried; any other status is final.
    huggingface_hub's HTTP errors derive from ``requests.HTTPError`` and so from
    ``OSError``; they are classified by status before the connection check. TLS
    and proxy errors are connection errors to requests and are retried, unless
    ``_setup_failure`` finds a cause that a retry cannot change.
    """
    names = {cls.__name__ for cls in type(exc).__mro__}
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    access = (
        f"{repo.page} refused access (HTTP {status or '401/403'}): {exc}. The repository "
        f"is gated. Log in with `.venv/bin/hf auth login` (or set HF_TOKEN) and accept its "
        f"terms on {repo.page}, then rerun."
    )
    if names & {"GatedRepoError", "RepositoryNotFoundError"} or status in {401, 403}:
        return ReleaseAssetError(access)
    if "RevisionNotFoundError" in names:
        return ReleaseAssetError(f"{repo.repo_id} has no revision {repo.revision!r}: {exc}")
    if "LocalEntryNotFoundError" in names:
        # A file lookup failed and nothing was cached; the cause says why.
        cause = exc.__cause__
        return _hub_failure(cause, repo) if cause is not None else ReleaseAssetError(str(exc))
    if "OfflineModeIsEnabled" in names:
        return ReleaseAssetError(f"{exc}; unset HF_HUB_OFFLINE to download")
    setup_problem = _setup_failure(exc, status)
    if setup_problem is not None:
        return ReleaseAssetError(
            f"{repo.page} could not be reached because {setup_problem}: {exc}. This is not "
            f"retried; {_TLS_PROXY_HINT}, then rerun.")
    if status is not None:
        if status in RETRY_STATUS:
            return None
        return ReleaseAssetError(f"{repo.page} returned HTTP {status}: {exc}")
    if "HTTPError" in names:
        return ReleaseAssetError(f"{repo.page} request failed: {exc}")
    if isinstance(exc, (ConnectionError, TimeoutError)) or names & _TRANSIENT_NAMES:
        return None
    return ReleaseAssetError(str(exc))


def download_tree(repo: HubRepo, folder: str, local_dir: Path, *, timeout: float) -> Path:
    """Download one repository folder into ``local_dir``, resuming partial files."""
    _log(f"downloading {repo.tree_url(folder)}")
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            _hub_snapshot(
                repo_id=repo.repo_id, repo_type=repo.repo_type, revision=repo.revision,
                allow_patterns=[f"{folder}/*"], local_dir=str(local_dir),
                etag_timeout=timeout, endpoint=os.environ.get("HF_ENDPOINT") or None,
            )
            break
        except ReleaseAssetError:
            raise
        except Exception as exc:  # noqa: BLE001 - classified below; unknown errors stop
            failure = _hub_failure(exc, repo)
            if failure is not None:
                raise failure from exc
            last = exc
        if attempt + 1 < MAX_ATTEMPTS:
            delay = min(2**attempt, 30)
            _log(f"download attempt {attempt + 1}/{MAX_ATTEMPTS} failed ({last}); "
                 f"retrying in {delay}s")
            time.sleep(delay)
    else:
        hint = ""
        if last is not None and _tls_or_proxy(last):
            hint = f". The last attempt failed with a TLS or proxy error: {_TLS_PROXY_HINT}"
        raise ReleaseAssetError(
            f"{repo.tree_url(folder)} failed after {MAX_ATTEMPTS} attempts: {last}{hint}")
    tree = local_dir / folder
    if not tree.is_dir():
        raise ReleaseAssetError(f"{repo.tree_url(folder)} holds no files")
    return tree


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_file_content(left: Path, right: Path) -> bool:
    left_stat = left.stat()
    right_stat = right.stat()
    if left_stat.st_size != right_stat.st_size:
        return False
    if (left_stat.st_dev, left_stat.st_ino) == (right_stat.st_dev, right_stat.st_ino):
        return True
    return _sha256_file(left) == _sha256_file(right)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------
def _iter_files(directory: Path) -> Iterator[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ReleaseAssetError(f"refusing symlink in install source: {path}")
        if path.is_file():
            yield path


def _install_tree_verbatim(src_dir: Path, dst_dir: Path, *, label: str) -> list[str]:
    """Install an exact tree without changing any source bytes."""
    source_files = {
        path.relative_to(src_dir).as_posix(): path for path in _iter_files(src_dir)
    }
    if not source_files:
        raise ReleaseAssetError(f"{label}: source tree is empty: {src_dir}")

    if dst_dir.is_symlink() or (dst_dir.exists() and not dst_dir.is_dir()):
        raise ReleaseAssetError(f"{label}: destination is not a real directory: {dst_dir}")

    if not dst_dir.exists():
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_dir), str(dst_dir))
        return sorted(source_files)

    destination_files = {
        path.relative_to(dst_dir).as_posix(): path for path in _iter_files(dst_dir)
    }
    extras = sorted(set(destination_files) - set(source_files))
    differing = sorted(
        name
        for name in set(source_files).intersection(destination_files)
        if not _same_file_content(source_files[name], destination_files[name])
    )
    if extras or differing:
        detail = []
        if differing:
            detail.append("differing files: " + ", ".join(differing[:5]))
        if extras:
            detail.append("extra files: " + ", ".join(extras[:5]))
        raise ReleaseAssetError(
            f"{label}: {dst_dir} is not byte-identical ({'; '.join(detail)}); "
            f"remove {dst_dir} and rerun the same profile to reinstall it"
        )

    written: list[str] = []
    for name in sorted(set(source_files) - set(destination_files)):
        destination = dst_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_files[name], destination)
        written.append(name)
    return written


# ---------------------------------------------------------------------------
# .env
# ---------------------------------------------------------------------------
def _assignment_name(line: str) -> str | None:
    stripped = line.lstrip()
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :]
    name, sep, _ = stripped.partition("=")
    return name.strip() if sep else None


def update_env(
    env_file: Path,
    env_example: Path,
    values: dict[str, str],
    *,
    remove: tuple[str, ...] = (),
) -> None:
    source = env_file if env_file.is_file() else env_example
    lines = source.read_text(encoding="utf-8").splitlines() if source.is_file() else []
    lines = [line for line in lines if _assignment_name(line) not in remove]
    seen: dict[str, int] = {}
    for index, line in enumerate(lines):
        name = _assignment_name(line)
        if name in values and name not in seen:
            seen[name] = index
    for name, value in values.items():
        line = f"{name}={value}"
        if name in seen:
            lines[seen[name]] = line
        else:
            lines.append(line)
    env_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = env_file.with_name(env_file.name + ".release-assets.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, env_file)


def _env_value(env_file: Path, name: str) -> str | None:
    if not env_file.is_file():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :]
        key, sep, value = stripped.partition("=")
        if sep and key.strip() == name:
            value = value.strip()
            if value[:1] in {"'", '"'} and len(value) >= 2 and value[-1] == value[0]:
                value = value[1:-1]
            return value
    return None


def _configured_inputs_root(env_file: Path) -> Path | None:
    value = _env_value(env_file, ENV_INPUTS_ROOT)
    if not value or PLACEHOLDER_MARKER in value:
        return None
    if not Path(value).is_absolute():
        raise ReleaseAssetError(
            f"{env_file} sets {ENV_INPUTS_ROOT}={value!r}, which is not absolute"
        )
    return Path(os.path.abspath(value))


# ---------------------------------------------------------------------------
# Profile selection and installation
# ---------------------------------------------------------------------------


def select_archives(profiles: list[Path], repo_root: Path, catalog: dict) -> list[str]:
    selected = set()
    for profile in profiles:
        profile = profile if profile.is_absolute() else repo_root / profile
        try:
            relative = profile.resolve().relative_to(repo_root / "config/release")
        except ValueError as exc:
            raise ReleaseAssetError("--profile must be under config/release/") from exc
        if len(relative.parts) != 3:
            raise ReleaseAssetError(f"expected <robot>/<rig>/<profile>.yaml: {profile}")
        data = yaml.safe_load(profile.read_text())
        scenes = data.get("scenes")
        if not isinstance(scenes, list) or len(scenes) != 1 or not isinstance(scenes[0], dict):
            raise ReleaseAssetError(f"{profile}: expected exactly one scene")
        path = str(scenes[0].get("path", ""))
        scene = PurePosixPath(path).name
        pin = catalog["archives"].get(scene)
        if (
            pin is None
            or not scene.startswith(relative.parts[1] + "_")
            or not pin["destination"].startswith("{inputs_root}/scenes/")
            or path != pin["destination"]
        ):
            raise ReleaseAssetError(f"{profile}: no pinned {relative.parts[1]} scene download for {path}")
        checkpoint = data["policy"]["checkpoint"]
        name = PurePosixPath(checkpoint).name
        if checkpoint != "{inputs_root}/checkpoints/" + name or name not in catalog["archives"]:
            raise ReleaseAssetError(f"{profile}: no pinned download for {checkpoint}")
        # Keep task/robot interpretation shared with runtime preflight. The
        # catalog only maps logical bundles to published archives; an empty
        # list explicitly denotes a bundle whose payloads are tracked in Git.
        bundles = runtime_bundle_ids(data.get("robot", "fr3"), data["task"]["objects"])
        downloads = catalog.get("runtime_bundles", {})
        if not isinstance(downloads, dict):
            raise ReleaseAssetError("catalog runtime_bundles must map bundle names to archive lists")
        unavailable = [bundle for bundle in bundles if bundle not in downloads]
        if unavailable:
            raise ReleaseAssetError(
                f"{profile}: no pinned downloads for runtime bundles: "
                + ", ".join(unavailable)
            )
        for bundle in bundles:
            archives = downloads[bundle]
            if not isinstance(archives, list) or any(
                not isinstance(archive, str) or archive not in catalog["archives"]
                for archive in archives
            ):
                raise ReleaseAssetError(
                    f"{profile}: invalid pinned downloads for runtime bundle {bundle!r}"
                )
            selected.update(archives)
        selected.update((scene, name))
    return sorted(selected, key=lambda name: catalog["archives"][name]["tree"]["size"])


def real_path(path: Path) -> Path:
    path = Path(os.path.abspath(path.expanduser()))
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ReleaseAssetError(f"runtime paths must not contain symlinks: {parent}")
    return path


def matches_tree(path: Path, pin: dict) -> bool:
    if not path.exists():
        return False
    try:
        digest = sha256_tree(path)
    except ManifestError:
        return False
    return digest.sha256 == pin["sha256"] and digest.size == pin["size"]


def install(name: str, pin: dict, target: Path, cache: Path, *, timeout: float,
            repos: Mapping[str, HubRepo]) -> None:
    """Install one pinned input from its Hugging Face repository."""
    real_path(target)
    if target.exists():
        _log(f"checking installed {name}")
        if matches_tree(target, pin["tree"]):
            _log(f"already verified: {target}")
            return
    repo, folder = locate(pin, repos)
    problems = repo.problems()
    if problems:
        raise ReleaseAssetError(f"cannot download {name}: " + "; ".join(problems))
    local_dir = cache / repo.key
    source = download_tree(repo, folder, local_dir, timeout=timeout)
    _log(f"hashing {name} ...")
    if not matches_tree(source, pin["tree"]):
        shutil.rmtree(source)
        raise ReleaseAssetError(
            f"{name}: the files downloaded from {repo.tree_url(folder)} do not match "
            f"the pinned tree (sha256 {pin['tree']['sha256']}, {pin['tree']['size']:,} "
            "bytes); nothing was installed")
    _install_tree_verbatim(source, target, label=name)
    # A fresh install moved the downloaded folder into place; completing a
    # partial install copied only the missing files. Drop what is left and its
    # resume metadata so each input is stored once.
    shutil.rmtree(source, ignore_errors=True)
    shutil.rmtree(local_dir / ".cache/huggingface/download" / folder, ignore_errors=True)
    _log(f"installed and verified: {target}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, action="append", required=True,
                        help="release profile YAML; repeat to install several profiles into one input root")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--inputs-root", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--no-env", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hf-revision", action="append", default=[], metavar="REPO=REV",
                        help="read catalog repository REPO (policies or assets) at revision REV")
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="read a repository whose catalog revision is unset at its main branch")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args(argv)
    revisions = {}
    for item in args.hf_revision:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise ReleaseAssetError(f"--hf-revision expects REPO=REV, got {item!r}")
        revisions[key] = value
    repo = real_path(args.repo_root or default_repo_root())
    env_file = args.env_file or repo / ".env"
    configured = _configured_inputs_root(env_file)
    environment_root = os.environ.get(ENV_INPUTS_ROOT)
    root = real_path(args.inputs_root or (Path(environment_root) if environment_root else None)
                     or configured or repo / "local-inputs")
    if configured is not None and root != configured and not args.no_env:
        raise ReleaseAssetError(f"{env_file} uses {configured}; reuse it or pass --no-env")
    catalog = json.loads((repo / "artifacts/release_downloads.json").read_text())
    names = select_archives(args.profile, repo, catalog)
    cache = root / "release-assets/downloads"
    targets = {name: real_path(Path(catalog["archives"][name]["destination"].format(
        inputs_root=root, repo_root=repo))) for name in names}
    repos = hub_repos(catalog, revisions=revisions, allow_unpinned=args.allow_unpinned)
    located = {name: locate(catalog["archives"][name], repos) for name in names}
    # Loop names other than ``repo``: that name holds the checkout root, which
    # the .env update below reads.
    for hub in sorted({hub for hub, _ in located.values()}, key=lambda hub: hub.key):
        print(f"Hugging Face {hub.repo_type} {hub.repo_id} at revision {hub.revision}")
        for problem in hub.problems():
            print(f"  NOT DOWNLOADABLE: {problem}")
        if hub.revision == UNPINNED_REVISION:
            print(f"  WARNING: {hub.repo_id} is read unpinned; only the tree pins guard it")
    for name in names:
        pin = catalog["archives"][name]
        hub, folder = located[name]
        print(f"  {name}: {pin['tree']['size']:,} bytes -> {targets[name]}")
        print(f"    {hub.tree_url(folder)}; tree sha256 {pin['tree']['sha256']}")
    if args.dry_run:
        return 0
    for name in names:
        install(name, catalog["archives"][name], targets[name], cache,
                timeout=args.timeout, repos=repos)
    if not args.no_env:
        update_env(env_file, repo / ".env.example", {ENV_INPUTS_ROOT: str(root)},
                          remove=LEGACY_ENV_NAMES)
    receipt = root / "release-assets/INSTALLED.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(receipt.read_text()) if receipt.exists() else {"archives": {}}
    source = {"huggingface": {repo.key: {"repo_id": repo.repo_id, "revision": repo.revision}
                              for repo in sorted({r for r, _ in located.values()},
                                                 key=lambda repo: repo.key)}}
    previous.update({"updated_at": datetime.now(timezone.utc).isoformat(),
                     "source": source, "inputs_root": str(root)})
    previous["archives"].update({name: catalog["archives"][name] for name in names})
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(previous, indent=2) + "\n")
    temporary.replace(receipt)
    print("Inputs verified. Run real2sim-run --config <same profile> --check-runtime.")
    return 0


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (ReleaseAssetError, ManifestError, OSError, ValueError, KeyError,
            yaml.YAMLError) as exc:
        print(f"[release-assets] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
