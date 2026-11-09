# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Canonical provenance primitives that bind the renderer code identity.

These definitions are the single source of truth shared by the renderer-side
capture/recheck code (:mod:`robosuite_env.rerender`) and the launcher-side
verification code (``real2sim.launcher``).  Both consumers import from here so
the two sides can never drift apart:

* :data:`CORE_RENDERER_SOURCE_PATHS` names the small, auditable renderer
  surface whose exact bytes are fingerprinted into every provenance record.
  The set is self-referential — this module is itself listed, because its
  contents define what the renderer identity binds.
* :data:`BATCH_POLICY_LINEAGE_FILENAMES` names the immutable policy lineage
  records published at a batch root.
* :func:`file_state`, :func:`sha256_file`, and
  :func:`git_tracked_diff_identity` are the primitive fingerprinting
  operations both sides must compute identically for records to match.
* :func:`official_generation_check` is the one rule that decides whether a
  renderer may stand in for an official run's own code. The post-hoc
  renderer applies it to its checkout before rendering and records the
  result in the rerender provenance; completion reporting verifies that
  record against the run's seal (:func:`official_generation_check_errors`)
  and does not read the checkout, so a derivative the renderer accepts is
  never rejected afterwards because the checkout changed.
"""


from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]


# Policy launchers publish these immutable lineage records at the batch root
# (two levels above ``<scene>/run_NN``).  Older batches may not have every
# record, so provenance binds the exact presence set as well as the bytes of
# every file that does exist.
BATCH_POLICY_LINEAGE_FILENAMES = (
    "resolved_rollout_config.json",
    "policy_batch_server_ready.json",
    "policy_inference_manifest.jsonl",
    "rollout_config.yaml",
)


# This deliberately names the small, auditable renderer surface instead of
# pretending that a commit plus a boolean dirty flag identifies the running
# implementation.  File records bind exact bytes (and replacement metadata),
# while the repository-wide tracked diff below binds edits to indirect tracked
# dependencies that are not listed here.
CORE_RENDERER_SOURCE_PATHS = (
    "robosuite_env/sim.py",
    "robosuite_env/contracts/provenance.py",
    # Owns the GS server argv and the /health identity check that the rerender
    # pipeline (rerender/gs_server.py) uses, so it is part of the fingerprinted
    # surface.
    "robosuite_env/gs_launch.py",
    "robosuite_env/policy/policy_io.py",
    "robosuite_env/rendering/rgb_color_calibration.py",
    "robosuite_env/rendering/wrist_fisheye.py",
    "robosuite_env/rendering/fisheye_model.py",
    "robosuite_env/rendering/wrist_render_profiles.py",
    "robosuite_env/rerender/__init__.py",
    "robosuite_env/rerender/batch.py",
    "robosuite_env/rerender/cadence.py",
    "robosuite_env/rerender/cli.py",
    "robosuite_env/rerender/common.py",
    "robosuite_env/rerender/gpu.py",
    "robosuite_env/rerender/gs_server.py",
    "robosuite_env/rerender/provenance.py",
    "robosuite_env/rerender/registration.py",
    "robosuite_env/rerender/replay_argv.py",
    "robosuite_env/rerender/run.py",
    "robosuite_env/rerender/sources.py",
    "robosuite_env/rerender/validation.py",
    "robosuite_env/rollout/outputs.py",
    "robosuite_env/rollout/replay_policy_insets.py",
    "robosuite_env/rollout/state/mujoco.py",
    "robosuite_env/rollout/worker.py",
    "robosuite_env/tools/rerender_third_person.py",
    "robosuite_env/tools/validate_rollout_video.py",
    "scene/gs_background_client.py",
    "scene/gs_background_protocol.py",
    "scene/gs_background_server.py",
)


def file_state(path: Path) -> dict[str, int]:
    """Capture the canonical replacement-detection state of a regular file."""

    observed = path.stat()
    return {
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "mode": int(observed.st_mode),
        "size": int(observed.st_size),
        "mtime_ns": int(observed.st_mtime_ns),
        "ctime_ns": int(observed.st_ctime_ns),
    }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's exact bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_tracked_diff_identity(commit: str) -> dict[str, Any]:
    """Hash the exact tracked patch from ``commit`` to index/worktree.

    Failures propagate as :class:`OSError` or
    :class:`subprocess.CalledProcessError`; callers wrap them into their own
    domain errors.
    """

    result = subprocess.run(
        [
            "git",
            "diff",
            "--no-ext-diff",
            "--binary",
            "--full-index",
            commit,
            "--",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return {
        "base_commit": commit,
        "sha256": hashlib.sha256(result.stdout).hexdigest(),
        "size_bytes": len(result.stdout),
    }


# The launcher seals the exact bytes of every runtime source a generation
# used in this file at the run root, next to ``run_manifest.json``
# (``real2sim.runtime_manifest.MANIFEST_FILE``).
RUNTIME_SOURCE_MANIFEST_FILENAME = "runtime_source_manifest.json"

#: The renderer's checkout is the run's own clean commit.
RULE_GENERATION_COMMIT = "generation_commit"
#: The renderer's checkout is a later one that still holds every sealed source.
RULE_SEALED_SOURCES = "sealed_sources"

#: Keys of the record of the official generation check: what the renderer
#: checked before rendering, plus ``unchanged_at_write``, its recheck of the
#: same checkout when it wrote the provenance.
OFFICIAL_GENERATION_CHECK_KEYS = frozenset(
    {
        "rule",
        "checkout_root",
        "commit",
        "dirty",
        "run_seal",
        "coverage",
        "files",
        "unchanged_at_write",
    }
)

_SHA256_HEX = frozenset("0123456789abcdef")


def _read_runtime_source_seal(
    run_root: Path,
) -> tuple[dict[str, Any], dict[str, str]] | None:
    """The run's seal and a record of the exact bytes it was read from."""

    sealed = Path(run_root) / RUNTIME_SOURCE_MANIFEST_FILENAME
    try:
        if sealed.is_symlink() or not sealed.is_file():
            return None
        data = sealed.read_bytes()
        payload = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload, {
        "path": str(sealed.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def sealed_file_digests(run_root: Path) -> dict[str, str]:
    """The SHA-256 the run's seal records for each file, by its absolute path.

    The keys are the paths exactly as the launcher sealed them (absolute,
    unresolved). A run without a readable seal, or a malformed record, gives
    no entry, so a caller that needs a digest fails closed.
    """

    sealed = _read_runtime_source_seal(run_root)
    files = sealed[0].get("files") if sealed is not None else None
    if not isinstance(files, list):
        return {}
    digests: dict[str, str] = {}
    for record in files:
        if not isinstance(record, dict):
            continue
        path = record.get("path")
        digest = record.get("sha256")
        if isinstance(path, str) and _is_sha256(digest):
            digests[path] = digest
    return digests


def _gaussian_wrapping_gitlink() -> str:
    from real2sim.provenance import GAUSSIAN_WRAPPING_GITLINK

    return GAUSSIAN_WRAPPING_GITLINK


def _git_output(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def gaussian_wrapping_identity(checkout_root: Path) -> dict[str, Any]:
    """The commit and local-change state of a checkout's GaussianWrapping checkout.

    The post-hoc renderer's GS server imports GaussianWrapping from the
    gitlinked checkout (``robosuite_env.gs_launch.GS_DEFAULT_GW_REPO``), so
    the renderer identity records it. ``commit`` and ``dirty`` are ``None``
    when that directory is not the top level of a Git checkout. ``dirty``
    counts untracked files, as the launcher's production check does.
    """

    path = Path(checkout_root) / _gaussian_wrapping_gitlink()
    identity: dict[str, Any] = {"path": str(path), "commit": None, "dirty": None}
    try:
        top = _git_output(path, "rev-parse", "--show-toplevel")
        if Path(top).resolve() != path.resolve():
            return identity
        commit = _git_output(path, "rev-parse", "HEAD")
        status = _git_output(path, "status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError):
        return identity
    return {"path": str(path), "commit": commit, "dirty": bool(status)}


def _launch_gaussian_wrapping(run_manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The run manifest's record of the gitlinked GaussianWrapping checkout at launch.

    ``build_run_manifest`` records it under ``source_dependencies`` (a
    production launch requires it clean and at the gitlink).
    """

    code = run_manifest.get("code")
    repo_path = code.get("path") if isinstance(code, Mapping) else None
    dependencies = run_manifest.get("source_dependencies")
    if not isinstance(repo_path, str) or not isinstance(dependencies, list):
        return None
    expected = str(Path(repo_path) / _gaussian_wrapping_gitlink())
    for dependency in dependencies:
        if isinstance(dependency, Mapping) and dependency.get("path") == expected:
            return dependency
    return None


def gaussian_wrapping_problem(
    run_manifest: Any, renderer_code: Mapping[str, Any], checkout_root: Path
) -> str | None:
    """Why the renderer's GaussianWrapping checkout cannot stand in for the run's, or ``None``.

    The run's record must name a clean commit, and the renderer identity's
    record of ``checkout_root``'s GaussianWrapping checkout must be clean at
    that commit. A later gitlink, or a local edit, is a different renderer.
    """

    launch = (
        _launch_gaussian_wrapping(run_manifest)
        if isinstance(run_manifest, Mapping)
        else None
    )
    if (
        launch is None
        or not _is_commit(launch.get("commit"))
        or launch.get("dirty") is not False
    ):
        return "the run manifest records no clean GaussianWrapping checkout at its gitlink"
    rendered = renderer_code.get("gaussian_wrapping")
    if not isinstance(rendered, Mapping) or rendered.get("path") != str(
        Path(checkout_root) / _gaussian_wrapping_gitlink()
    ):
        return (
            "the renderer identity records no GaussianWrapping checkout in the "
            "checked checkout"
        )
    if rendered.get("commit") != launch.get("commit"):
        return (
            "the renderer's GaussianWrapping checkout is at "
            f"{rendered.get('commit')}, the run's was at {launch.get('commit')}"
        )
    if rendered.get("dirty") is not False:
        return "the renderer's GaussianWrapping checkout has local changes"
    return None


def _has_seal(run_root: Path) -> bool:
    sealed = Path(run_root) / RUNTIME_SOURCE_MANIFEST_FILENAME
    return sealed.exists() or sealed.is_symlink()


def _sealed_hashes_under(
    payload: Mapping[str, Any], repo_path: Any
) -> dict[str, str] | None:
    if not isinstance(repo_path, str) or not repo_path:
        return None
    files = payload.get("files")
    if not isinstance(files, list):
        return None
    root = Path(repo_path)
    hashes: dict[str, str] = {}
    for record in files:
        if not isinstance(record, dict):
            return None
        path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            return None
        try:
            relative = Path(path).relative_to(root).as_posix()
        except ValueError:
            continue
        hashes[relative] = digest
    return hashes


def _coverage_names_core(coverage: Mapping[str, Any], renderer_code: Mapping[str, Any]) -> bool:
    """Does a recorded coverage name every core renderer file the renderer recorded?

    Completion checks the renderer's recorded coverage against the seal
    rather than the coverage the reporting checkout would require, so a pull
    between render and report that adds a tree or file does not refuse a
    render its renderer accepted. This keeps the recorded coverage from being
    empty or narrower than the renderer's own core files.
    """

    from real2sim.runtime_manifest import tree_selects

    core = renderer_code.get("core_source_files")
    try:
        files = set(coverage["files"])
        trees = [(tree["path"], tuple(tree["suffixes"])) for tree in coverage["trees"]]
    except (KeyError, TypeError):
        return False
    if not isinstance(core, Mapping) or not core:
        return False
    for relative_path in core:
        if not isinstance(relative_path, str):
            return False
        if relative_path in files:
            continue
        path = Path(relative_path)
        if not any(
            path.is_relative_to(tree_path) and tree_selects(path, Path(tree_path), suffixes)
            for tree_path, suffixes in trees
            if isinstance(tree_path, str)
        ):
            return False
    return True


def _paths_in_commit(root: Path, commit: str) -> frozenset[str] | None:
    """The paths ``commit`` tracks, read from Git objects in ``root``; never fetches."""

    try:
        listed = subprocess.run(
            ["git", "ls-tree", "-r", "-z", "--name-only", commit],
            cwd=root,
            check=True,
            capture_output=True,
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return frozenset(os.fsdecode(item) for item in listed.split(b"\0") if item)


def _sealed_files_outside_commit(
    hashes: Mapping[str, str], root: Path, commit: str
) -> dict[str, str] | None:
    """Check the sealed files ``commit`` does not track against their sealed digests.

    A clean checkout at the run's commit holds the tracked files that commit
    pins. The seal also holds files Git ignores or does not track, such as
    the archive meshes installed under ``robosuite_env/assets`` and the run's
    config; these must still hash to the seal. Returns their digests, or
    ``None`` on any difference.
    """

    tracked = _paths_in_commit(root, commit)
    if tracked is None:
        return None
    observed: dict[str, str] = {}
    for relative_path, digest in sorted(hashes.items()):
        if relative_path in tracked:
            continue
        candidate = root / relative_path
        try:
            if candidate.is_symlink() or sha256_file(candidate) != digest:
                return None
        except OSError:
            return None
        observed[relative_path] = digest
    return observed


def _core_sources_match(
    hashes: Mapping[str, str],
    renderer_code: Mapping[str, Any],
    paths: Any = CORE_RENDERER_SOURCE_PATHS,
) -> bool:
    """Every core renderer file in ``paths`` is recorded and hashes to the seal."""

    observed = renderer_code.get("core_source_files")
    if not isinstance(observed, Mapping) or not paths:
        return False
    for relative_path in paths:
        record = observed.get(relative_path)
        digest = hashes.get(relative_path)
        if (
            not isinstance(record, Mapping)
            or not isinstance(digest, str)
            or record.get("sha256") != digest
        ):
            return False
    return True


def _checkout_holds_sealed_sources(
    hashes: Mapping[str, str], selection: Mapping[str, Any], root: Path
) -> dict[str, str] | None:
    """The digest of every sealed file in ``root``, or ``None`` on any difference.

    Every sealed file must hash the same in ``root``, and no file the seal's
    tree specs select there (``real2sim.runtime_manifest.tree_members``, the
    rule sealing uses) may be unsealed, because a new task, stage, robot
    module or asset in a sealed tree can change what the worker loads.
    """

    from real2sim.runtime_manifest import RuntimeManifestError, tree_members

    observed: dict[str, str] = {}
    for relative_path, digest in sorted(hashes.items()):
        candidate = root / relative_path
        if candidate.is_symlink() or not candidate.is_file():
            return None
        try:
            actual = sha256_file(candidate)
        except OSError:
            return None
        if actual != digest:
            return None
        observed[relative_path] = actual
    trees = selection.get("trees")
    if not isinstance(trees, list):
        return None
    for tree in trees:
        if not isinstance(tree, dict):
            return None
        tree_path = tree.get("path")
        suffixes = tree.get("suffixes")
        if not isinstance(tree_path, str) or not isinstance(suffixes, list):
            return None
        if not (root / tree_path).is_dir():
            return None
    try:
        members = tree_members(root, trees)
    except RuntimeManifestError:
        return None
    if any(relative not in hashes for relative in members):
        return None
    return observed


def official_generation_check(
    run_root: Path,
    run_manifest: Any,
    renderer_code: Mapping[str, Any],
    *,
    checkout_root: Path,
) -> dict[str, Any] | None:
    """May the checkout at ``checkout_root`` render for the official run at ``run_root``?

    ``run_manifest`` is the run's ``run_manifest.json`` and ``renderer_code``
    the renderer's identity of ``checkout_root``. Under either rule below, the
    checkout's GaussianWrapping checkout, which renders the 3DGS background,
    must be clean and at the commit the run recorded for it at launch
    (:func:`gaussian_wrapping_problem`). The checkout qualifies when it is the
    run's clean commit and every sealed
    file that commit does not track -- the gitignored archive meshes
    installed under ``robosuite_env/assets``, a run config -- still hashes to
    the seal (a run without a seal has only its commit). Later commits that
    leave every sealed runtime source unchanged are not evidence that the
    replay changed, so a moved-on checkout also qualifies when the seal
    covers every file and tree this checkout's runtime reads
    (``real2sim.runtime_manifest.required_coverage``; a seal written before a
    data tree joined it does not), each recorded core renderer file hashes to
    the seal, every other file the seal lists under the run's ``code.path``
    hashes the same in ``checkout_root``, and nothing unsealed sits in a
    sealed tree. Without a usable seal only the commit rule applies.

    Returns ``None`` when the checkout does not qualify, or else the record
    of the check: the rule, the checkout and its commit, the seal's bytes and
    the digest of every file compared (for the commit rule, the sealed files
    the commit does not track), and for a moved-on checkout the coverage
    required. The renderer stores it in the rerender provenance and
    completion reporting verifies it against the run's seal
    (:func:`official_generation_check_errors`) instead of reading whatever
    checkout exists when the report is made.

    Whether the manifest itself names a clean commit is a separate question
    that callers answer before relying on this one.
    """

    from real2sim.runtime_manifest import required_coverage, selection_covers

    manifest_code = (
        run_manifest.get("code") if isinstance(run_manifest, Mapping) else None
    )
    if not isinstance(manifest_code, Mapping):
        return None
    root = Path(checkout_root).resolve()
    if gaussian_wrapping_problem(run_manifest, renderer_code, root) is not None:
        return None
    commit = manifest_code.get("commit")
    base = {
        "checkout_root": str(root),
        "commit": renderer_code.get("commit"),
        "dirty": renderer_code.get("dirty"),
    }
    if (
        isinstance(commit, str)
        and renderer_code.get("commit") == commit
        and renderer_code.get("dirty") is False
    ):
        if not _has_seal(run_root):
            # A run launched before the seal existed has only its commit.
            return {
                "rule": RULE_GENERATION_COMMIT,
                **base,
                "run_seal": None,
                "coverage": None,
                "files": {},
            }
        sealed = _read_runtime_source_seal(run_root)
        if sealed is None:
            return None
        payload, seal_record = sealed
        hashes = _sealed_hashes_under(payload, manifest_code.get("path"))
        outside = (
            None if hashes is None else _sealed_files_outside_commit(hashes, root, commit)
        )
        if outside is None:
            return None
        return {
            "rule": RULE_GENERATION_COMMIT,
            **base,
            "run_seal": seal_record,
            "coverage": None,
            "files": outside,
        }
    sealed = _read_runtime_source_seal(run_root)
    if sealed is None:
        return None
    payload, seal_record = sealed
    hashes = _sealed_hashes_under(payload, manifest_code.get("path"))
    selection = payload.get("selection")
    coverage = required_coverage()
    if (
        not hashes
        or not isinstance(selection, dict)
        or not selection_covers(selection, coverage)
        or not _core_sources_match(hashes, renderer_code)
    ):
        return None
    observed = _checkout_holds_sealed_sources(hashes, selection, root)
    if observed is None:
        return None
    return {
        "rule": RULE_SEALED_SOURCES,
        **base,
        "run_seal": seal_record,
        "coverage": coverage,
        "files": observed,
    }


def official_generation_files_unchanged(record: Mapping[str, Any]) -> bool:
    """Does the recorded checkout still hold every file the check compared?

    The renderer asks this when it writes the provenance, after rendering.
    """

    root = Path(str(record.get("checkout_root")))
    files = record.get("files")
    if not isinstance(files, Mapping):
        return False
    for relative_path, digest in files.items():
        candidate = root / relative_path
        try:
            if candidate.is_symlink() or sha256_file(candidate) != digest:
                return False
        except OSError:
            return False
    return True


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= _SHA256_HEX


def official_generation_check_errors(
    run_root: Path,
    run_manifest: Any,
    renderer_code: Mapping[str, Any],
    record: Any,
) -> list[str]:
    """Verify a renderer's recorded check against the official run's seal.

    This reads the run's seal and manifest, never a checkout's working tree,
    so completion reporting reaches the renderer's answer however the
    checkout changed after the render. A missing or malformed record, one
    that disagrees with the renderer's identity, the manifest or the seal,
    or one whose checkout changed while the rerender ran is refused. This
    is a consistency check: a record written to agree with the seal and the
    renderer identity is not detected.
    """

    from real2sim.runtime_manifest import selection_covers

    if not isinstance(record, Mapping) or set(record) != OFFICIAL_GENERATION_CHECK_KEYS:
        return ["the rerender has no valid record of its official generation check"]
    errors: list[str] = []
    checkout_root = record.get("checkout_root")
    if not isinstance(checkout_root, str) or not Path(checkout_root).is_absolute():
        errors.append("the official generation check names no absolute checkout")
    else:
        problem = gaussian_wrapping_problem(
            run_manifest, renderer_code, Path(checkout_root)
        )
        if problem is not None:
            errors.append(problem)
    manifest_code = (
        run_manifest.get("code") if isinstance(run_manifest, Mapping) else None
    )
    if record.get("commit") != renderer_code.get("commit") or record.get(
        "dirty"
    ) is not renderer_code.get("dirty"):
        errors.append(
            "the official generation check names a different checkout commit "
            "than the renderer code"
        )
    if record.get("unchanged_at_write") is not True:
        errors.append("the renderer checkout changed while the rerender ran")
    if not isinstance(manifest_code, Mapping):
        return [*errors, "the run manifest has no code record"]
    rule = record.get("rule")
    if rule == RULE_GENERATION_COMMIT:
        if not (
            record.get("commit") == manifest_code.get("commit")
            and record.get("dirty") is False
            and record.get("coverage") is None
        ):
            errors.append(
                "the official generation check claims the run's clean commit, "
                "but the record does not show it"
            )
            return errors
        if not _has_seal(run_root):
            if record.get("run_seal") is not None or record.get("files") != {}:
                errors.append(
                    "the official generation check names a seal the run does not have"
                )
            return errors
        sealed = _read_runtime_source_seal(run_root)
        if sealed is None:
            return [*errors, "the run has no readable runtime source seal"]
        payload, seal_record = sealed
        if record.get("run_seal") != seal_record:
            errors.append("the official generation check was made against other seal bytes")
        hashes = _sealed_hashes_under(payload, manifest_code.get("path"))
        # The files the commit does not track are read from Git objects of the
        # checked checkout, which a later pull or edit does not change.
        tracked = (
            _paths_in_commit(Path(checkout_root), record["commit"])
            if isinstance(checkout_root, str) and _is_commit(record.get("commit"))
            else None
        )
        if hashes is None or tracked is None:
            errors.append(
                "the sealed files outside the run's commit could not be listed"
            )
        elif record.get("files") != {
            path: digest for path, digest in hashes.items() if path not in tracked
        }:
            errors.append(
                "the official generation check did not compare exactly the sealed "
                "files outside the run's commit, or found them different"
            )
        return errors
    if rule != RULE_SEALED_SOURCES:
        return [*errors, f"the official generation check has an unknown rule {rule!r}"]
    sealed = _read_runtime_source_seal(run_root)
    if sealed is None:
        return [*errors, "the run has no readable runtime source seal"]
    payload, seal_record = sealed
    if record.get("run_seal") != seal_record:
        errors.append("the official generation check was made against other seal bytes")
    hashes = _sealed_hashes_under(payload, manifest_code.get("path"))
    files = record.get("files")
    if (
        not hashes
        or not isinstance(files, Mapping)
        or dict(files) != hashes
        or not all(_is_sha256(digest) for digest in files.values())
    ):
        errors.append(
            "the official generation check did not compare exactly the sealed "
            "sources, or found them different"
        )
    selection = payload.get("selection")
    coverage = record.get("coverage")
    if (
        not isinstance(selection, dict)
        or not isinstance(coverage, Mapping)
        or not selection_covers(selection, coverage)
        or not _coverage_names_core(coverage, renderer_code)
    ):
        errors.append(
            "the run's seal does not cover what the renderer's runtime reads"
        )
    # The renderer's own core list, as it recorded it: completion does not
    # substitute the core list or coverage of the code that reports.
    recorded_core = renderer_code.get("core_source_files")
    if not hashes or not _core_sources_match(
        hashes,
        renderer_code,
        tuple(recorded_core) if isinstance(recorded_core, Mapping) else (),
    ):
        errors.append("the recorded core renderer files do not hash to the seal")
    return errors
