# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
from real2sim import runtime_manifest
from real2sim.provenance import (
    ProvenanceError,
    collect_git_repository,
    collect_lockfiles,
)


class RuntimeSourceManifestTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[dict, Path, Path]:
        package = root / "runtime"
        package.mkdir()
        entrypoint = root / "launch.sh"
        module = package / "worker.py"
        entrypoint.write_text("#!/bin/sh\n", encoding="utf-8")
        module.write_text("GENERATION = 1\n", encoding="utf-8")
        selection = {
            "repo_root": str(root.resolve()),
            "files": ["launch.sh"],
            "external_files": [],
            "trees": [{"path": "runtime", "suffixes": [".py"]}],
        }
        return selection, entrypoint, module

    def test_default_selection_covers_formal_portable_entrypoints(self):
        required = {
            "robosuite_env/tools/summarize_policy_rollouts.py",
            "scripts/run_policy_batch.sh",
            "src/real2sim/launcher.py",
            "src/real2sim/runtime_manifest.py",
        }

        self.assertTrue(required.issubset(runtime_manifest.DEFAULT_FILES))
        self.assertIn(
            "robosuite_env/tools/rerender_third_person.py",
            runtime_manifest.DEFAULT_FILES,
        )
        self.assertIn(
            ("robosuite_env/rerender", (".py",)),
            runtime_manifest.DEFAULT_TREE_SPECS,
        )
        # The shared provenance primitives module must stay inside a scanned
        # tree so the runtime manifest binds its bytes.
        self.assertIn(
            ("robosuite_env/contracts", (".py",)),
            runtime_manifest.DEFAULT_TREE_SPECS,
        )
        real2sim_source_files = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / "src" / "real2sim").rglob("*.py")
        }
        excluded = {
            "src/real2sim/doctor.py",
            "src/real2sim/portable_tools.py",
            "src/real2sim/release_assets.py",
        }
        self.assertTrue(
            (real2sim_source_files - excluded).issubset(runtime_manifest.DEFAULT_FILES)
        )
        self.assertTrue(excluded.isdisjoint(runtime_manifest.DEFAULT_FILES))
        self.assertNotIn(
            ("src/real2sim", (".py",)),
            runtime_manifest.DEFAULT_TREE_SPECS,
        )

    def test_atomic_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _entrypoint, _module = self._fixture(root)
            capture = runtime_manifest.capture_selection(selection)
            manifest = root / "output" / "runtime_source_manifest.json"

            digest = runtime_manifest.write_manifest(manifest, capture)
            report = runtime_manifest.verify_manifest(
                manifest,
                expected_sha256=digest,
            )

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["file_count"], 2)
            self.assertEqual(
                hashlib.sha256(manifest.read_bytes()).hexdigest(), digest
            )
            self.assertEqual(
                list(manifest.parent.glob(f".{manifest.name}.tmp-*")), []
            )

    def test_verify_fails_closed_on_content_or_state_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _entrypoint, module = self._fixture(root)
            capture = runtime_manifest.capture_selection(selection)
            manifest = root / "runtime_source_manifest.json"
            digest = runtime_manifest.write_manifest(manifest, capture)

            module.write_text("GENERATION = 2\n", encoding="utf-8")

            # The precise type matters: the batch runner only survives drift,
            # and the superclass would also match a corrupt manifest.
            with self.assertRaisesRegex(
                runtime_manifest.RuntimeSourceDriftError,
                "critical runtime file (state|content) changed",
            ):
                runtime_manifest.verify_manifest(
                    manifest,
                    expected_sha256=digest,
                )

    def test_verify_rejects_new_import_file_in_selected_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _entrypoint, _module = self._fixture(root)
            capture = runtime_manifest.capture_selection(selection)
            manifest = root / "runtime_source_manifest.json"
            digest = runtime_manifest.write_manifest(manifest, capture)
            (root / "runtime" / "late_module.py").write_text(
                "GENERATION = 2\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(
                runtime_manifest.RuntimeSourceDriftError,
                "critical runtime file set changed",
            ):
                runtime_manifest.verify_manifest(
                    manifest,
                    expected_sha256=digest,
                )

    def test_a_moved_tree_is_drift_and_a_broken_manifest_is_not(self):
        # The batch runner survives drift and must still stop on anything else.
        # The two raise different types with different exit codes, so the shell
        # does not treat a corrupt manifest as a harmless edit.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _entrypoint, module = self._fixture(root)
            capture = runtime_manifest.capture_selection(selection)
            manifest = root / "runtime_source_manifest.json"
            digest = runtime_manifest.write_manifest(manifest, capture)

            module.write_text("GENERATION = 2\n", encoding="utf-8")
            with self.assertRaises(runtime_manifest.RuntimeSourceDriftError):
                runtime_manifest.verify_manifest(manifest, expected_sha256=digest)
            self.assertEqual(
                runtime_manifest.main(
                    [
                        "verify",
                        "--manifest",
                        str(manifest),
                        "--expected-sha256",
                        digest,
                    ]
                ),
                3,
            )
            module.write_text("GENERATION = 1\n", encoding="utf-8")

            # Adding a file to a selected tree is what a commit landing in a
            # shared worktree usually looks like. It is drift too, not a hard
            # stop.
            late = root / "runtime" / "late_module.py"
            late.write_text("GENERATION = 2\n", encoding="utf-8")
            with self.assertRaises(runtime_manifest.RuntimeSourceDriftError):
                runtime_manifest.verify_manifest(manifest, expected_sha256=digest)
            self.assertEqual(
                runtime_manifest.main(
                    [
                        "verify",
                        "--manifest",
                        str(manifest),
                        "--expected-sha256",
                        digest,
                    ]
                ),
                3,
            )
            late.unlink()

            manifest.write_text("{ not json", encoding="utf-8")
            broken = hashlib.sha256(manifest.read_bytes()).hexdigest()
            with self.assertRaises(runtime_manifest.RuntimeManifestError) as raised:
                runtime_manifest.verify_manifest(manifest, expected_sha256=broken)
            self.assertNotIsInstance(
                raised.exception, runtime_manifest.RuntimeSourceDriftError
            )
            self.assertEqual(
                runtime_manifest.main(
                    [
                        "verify",
                        "--manifest",
                        str(manifest),
                        "--expected-sha256",
                        broken,
                    ]
                ),
                2,
            )

    def test_drift_is_a_runtime_manifest_error_so_callers_still_catch_it(self):
        self.assertTrue(
            issubclass(
                runtime_manifest.RuntimeSourceDriftError,
                runtime_manifest.RuntimeManifestError,
            )
        )

    def test_deleting_or_replacing_a_sealed_source_is_drift(self):
        # Discovery reaches the path before the file-set comparison does; these
        # must still read as drift, not as "unreadable"/"not a regular file"
        # with exit 2 -- a hard stop for an ordinary rm or checkout.
        for mutate in (
            lambda path: path.unlink(),
            lambda path: (path.unlink(), path.symlink_to("/etc/hostname")),
            lambda path: (path.unlink(), path.mkdir()),
            lambda path: path.chmod(0o000),
        ):
            with self.subTest(mutate=mutate):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    selection, _entrypoint, module = self._fixture(root)
                    capture = runtime_manifest.capture_selection(selection)
                    manifest = root / "runtime_source_manifest.json"
                    digest = runtime_manifest.write_manifest(manifest, capture)

                    mutate(module)
                    # No cleanup: a 0o000 file is still removable through its
                    # directory, and chmod on the symlink case would follow the
                    # link out of the temporary tree.
                    with self.assertRaises(runtime_manifest.RuntimeSourceDriftError):
                        runtime_manifest.verify_manifest(
                            manifest, expected_sha256=digest
                        )
                    self.assertEqual(
                        runtime_manifest.main(
                            [
                                "verify",
                                "--manifest",
                                str(manifest),
                                "--expected-sha256",
                                digest,
                            ]
                        ),
                        3,
                    )

    def test_the_same_unreadable_path_is_drift_only_while_verifying(self):
        # The classification is a property of the operation, not of the file.
        # Sealing an unreadable path is a configuration error that must stop the
        # launch; finding the same path unreadable against an existing manifest
        # is an edit, because the record proves it was readable when sealed.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, entrypoint, _module = self._fixture(root)
            capture = runtime_manifest.capture_selection(selection)
            manifest = root / "runtime_source_manifest.json"
            digest = runtime_manifest.write_manifest(manifest, capture)
            # An explicitly listed file, not a tree member: deleting a tree
            # member only seals a smaller set, which is not an error either way.
            entrypoint.unlink()

            with self.assertRaises(runtime_manifest.RuntimeManifestError) as sealing:
                runtime_manifest.capture_selection(selection)
            self.assertNotIsInstance(
                sealing.exception, runtime_manifest.RuntimeSourceDriftError
            )

            with self.assertRaises(runtime_manifest.RuntimeSourceDriftError):
                runtime_manifest.verify_manifest(manifest, expected_sha256=digest)

    def test_sealing_stays_fail_closed_through_the_production_seal_call(self):
        # seal_runtime_source_manifest verifies its baseline through the same
        # verify_capture that post-seal verification uses. That function does
        # not decide on its own that tree reads are drift: an unreadable path at
        # seal time reported as drift would let the launcher start the batch
        # unsealed, skipping verification for the rest of the run. The choice
        # belongs to the caller.
        from real2sim import launch_provenance

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _entrypoint, module = self._fixture(root)
            baseline = runtime_manifest.capture_selection(selection)
            module.chmod(0o000)
            try:
                with self.assertRaises(
                    runtime_manifest.RuntimeManifestError
                ) as raised:
                    launch_provenance.seal_runtime_source_manifest(
                        {"config_path": root / "missing.yaml", "run_root": root},
                        baseline,
                    )
                self.assertNotIsInstance(
                    raised.exception, runtime_manifest.RuntimeSourceDriftError
                )
            finally:
                module.chmod(0o644)

    def test_an_unreadable_subtree_is_never_silently_sealed(self):
        # Path.rglob suppresses PermissionError and simply omits the subtree.
        # Sealing would then record a generation with a directory missing from
        # it and call that a complete capture, and no later verification would
        # notice files added beneath it.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _entrypoint, _module = self._fixture(root)
            inner = root / "runtime" / "inner"
            inner.mkdir()
            (inner / "deep.py").write_text("GENERATION = 1\n", encoding="utf-8")
            inner.chmod(0o000)
            try:
                with self.assertRaisesRegex(
                    runtime_manifest.RuntimeManifestError, "could not be listed"
                ):
                    runtime_manifest.capture_selection(selection)
            finally:
                inner.chmod(0o755)

    def test_a_close_that_fails_is_not_a_successful_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _entrypoint, _module = self._fixture(root)
            with mock.patch.object(
                os, "close", side_effect=OSError(9, "Bad file descriptor")
            ):
                with self.assertRaisesRegex(
                    runtime_manifest.RuntimeManifestError, "could not be closed"
                ):
                    runtime_manifest.capture_selection(selection)

    def test_a_capture_that_moved_before_sealing_is_not_drift(self):
        # The file-set, state and content comparisons do not raise drift
        # directly, which would ignore the caller's answer: a tree that moved
        # between being captured and being sealed would reach the launcher's
        # drift catch and start the batch unsealed. Nothing in the module
        # decides; only the drift context converts.
        from real2sim import launch_provenance

        for label, mutate in (
            ("content", lambda root, module: module.write_text("G=2\n")),
            ("state", lambda root, module: module.chmod(0o600)),
            ("added", lambda root, module: (root / "runtime" / "late.py").write_text("G=2\n")),
            ("removed", lambda root, module: module.unlink()),
        ):
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    selection, _entrypoint, module = self._fixture(root)
                    baseline = runtime_manifest.capture_selection(selection)
                    mutate(root, module)

                    with self.assertRaises(
                        runtime_manifest.RuntimeManifestError
                    ) as raised:
                        launch_provenance.seal_runtime_source_manifest(
                            {"config_path": root / "missing.yaml", "run_root": root},
                            baseline,
                        )
                    self.assertNotIsInstance(
                        raised.exception, runtime_manifest.RuntimeSourceDriftError
                    )

    def test_a_read_that_fails_part_way_through_still_has_an_exit_code(self):
        # A raw OSError must not escape both the drift context and main's
        # handlers: the command would exit neither 2 nor 3, and the shell,
        # which switches on the exit code, could not tell drift from corruption.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _entrypoint, _module = self._fixture(root)
            capture = runtime_manifest.capture_selection(selection)
            manifest = root / "runtime_source_manifest.json"
            digest = runtime_manifest.write_manifest(manifest, capture)

            with mock.patch.object(os, "read", side_effect=OSError(5, "I/O error")):
                with self.assertRaisesRegex(
                    runtime_manifest.RuntimeManifestError, "could not be read"
                ):
                    runtime_manifest.verify_manifest(
                        manifest, expected_sha256=digest
                    )
                self.assertEqual(
                    runtime_manifest.main(
                        [
                            "verify",
                            "--manifest",
                            str(manifest),
                            "--expected-sha256",
                            digest,
                        ]
                    ),
                    2,
                )

    def test_a_manifest_rewritten_underfoot_is_not_drift(self):
        # verify_manifest reads the manifest before it enters the drift
        # context, so a record that is itself unreadable stays a hard error.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "runtime_source_manifest.json"
            with self.assertRaises(runtime_manifest.RuntimeManifestError) as raised:
                runtime_manifest.verify_manifest(missing, expected_sha256="a" * 64)
            self.assertNotIsInstance(
                raised.exception, runtime_manifest.RuntimeSourceDriftError
            )
            self.assertEqual(
                runtime_manifest.main(
                    ["verify", "--manifest", str(missing), "--expected-sha256", "a" * 64]
                ),
                2,
            )

    def test_verify_checks_detached_manifest_digest_before_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _entrypoint, _module = self._fixture(root)
            capture = runtime_manifest.capture_selection(selection)
            manifest = root / "runtime_source_manifest.json"
            digest = runtime_manifest.write_manifest(manifest, capture)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["created_at_utc"] = "tampered"
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                runtime_manifest.RuntimeManifestError,
                "runtime manifest content changed",
            ):
                runtime_manifest.verify_manifest(
                    manifest,
                    expected_sha256=digest,
                )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_capture_rejects_symlinked_critical_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "runtime"
            package.mkdir()
            target = root / "target.py"
            target.write_text("GENERATION = 1\n", encoding="utf-8")
            (package / "worker.py").symlink_to(target)
            selection = {
                "repo_root": str(root.resolve()),
                "files": [],
                "external_files": [],
                "trees": [{"path": "runtime", "suffixes": [".py"]}],
            }

            with self.assertRaisesRegex(
                runtime_manifest.RuntimeManifestError,
                "must not (be a symlink|contain symlinks)",
            ):
                runtime_manifest.capture_selection(selection)


def test_production_rejects_unhashed_requirements_lock(tmp_path):
    lock = tmp_path / "requirements-lock.txt"
    lock.write_text("example==1.2.3\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="unhashed"):
        collect_lockfiles(tmp_path, [str(lock)], production=True)

def test_production_provenance_accepts_canonical_github_ssh(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Real2Sim test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    (tmp_path / "tracked.txt").write_text("pinned\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    remote = "git@github.com:example-org/real2sim.git"
    subprocess.run(
        ["git", "-C", str(tmp_path), "remote", "add", "origin", remote],
        check=True,
    )

    assert collect_git_repository(tmp_path, production=True)["remote"] == remote

def test_pinned_submodule_survives_stripped_status_output(tmp_path: Path):
    # ``git submodule status`` marks a pinned submodule with a leading space
    # that the stripped subprocess capture removes from the first line. A
    # production run with an initialized submodule must still be accepted.
    from real2sim import provenance

    pinned = "d1dc83afd89ded4379851257fe5d85632d31d5ec"
    other = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
    stripped = f"{pinned} third_party/aloha (d1dc83a)\n {other} third_party/libero (f78abd6)"
    with mock.patch.object(provenance, "_run", return_value=stripped):
        records = provenance.collect_git_submodules(tmp_path, production=True)
    assert [(r["path"], r["commit"], r["status"]) for r in records] == [
        ("third_party/aloha", pinned, "pinned"),
        ("third_party/libero", other, "pinned"),
    ]
    with mock.patch.object(provenance, "_run", return_value=f"-{pinned} third_party/aloha"):
        with pytest.raises(ProvenanceError, match="missing, conflicted"):
            provenance.collect_git_submodules(tmp_path, production=True)


@pytest.mark.parametrize("marker", ("-", "+"))
def test_production_records_but_does_not_require_gaussian_wrapping_nested_submodules(
    tmp_path: Path, marker: str
):
    # Bootstrap does not fetch GaussianWrapping's nested submodules; nothing
    # builds or imports them. Only they are exempt, not a look-alike sibling.
    from real2sim import provenance

    commit = "d1dc83afd89ded4379851257fe5d85632d31d5ec"
    nested = f"{provenance.GAUSSIAN_WRAPPING_GITLINK}/submodules/nvdiffrast"
    with mock.patch.object(provenance, "_run", return_value=f"{marker}{commit} {nested}"):
        records = provenance.collect_git_submodules(tmp_path, production=True)
    assert records == [
        {"path": nested, "commit": commit,
         "status": {"-": "missing", "+": "different-commit"}[marker]},
    ]
    for path in (provenance.GAUSSIAN_WRAPPING_GITLINK, "third_party/GaussianWrapping2/nvdiffrast"):
        with mock.patch.object(provenance, "_run", return_value=f"{marker}{commit} {path}"):
            with pytest.raises(ProvenanceError, match="missing, conflicted"):
                provenance.collect_git_submodules(tmp_path, production=True)

@pytest.mark.parametrize("name", ("GaussianWrapping", "gw"))
def test_production_gaussian_wrapping_checkout_must_match_the_gitlink(tmp_path: Path, name: str):
    from real2sim import provenance

    runtime = f"third_party/{name}"
    source = tmp_path / runtime
    source.mkdir(parents=True)

    def git(cwd: Path, *args: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.name=Real2Sim test", "-c", "user.email=test@example.invalid",
             "-C", str(cwd), *args],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    git(source, "init", "-q")
    git(source, "remote", "add", "origin", "https://github.com/airoa-org/GaussianWrapping.git")
    (source / "tracked.txt").write_text("pinned\n")
    git(source, "add", ".")
    git(source, "commit", "-qm", "pinned")
    pinned = git(source, "rev-parse", "HEAD")
    git(tmp_path, "init", "-q")
    git(tmp_path, "update-index", "--add", "--cacheinfo", f"160000,{pinned},third_party/GaussianWrapping")
    git(tmp_path, "commit", "-qm", "pin")

    [record] = provenance.collect_source_dependencies(
        tmp_path, [runtime], production=True, gaussian_wrapping_repo=runtime
    )
    assert record["commit"] == pinned

    (source / "tracked.txt").write_text("next\n")
    git(source, "commit", "-qam", "next")
    with pytest.raises(ProvenanceError, match=f"third_party/GaussianWrapping pins {pinned}"):
        provenance.collect_source_dependencies(
            tmp_path, [runtime], production=True, gaussian_wrapping_repo=runtime
        )
    [record] = provenance.collect_source_dependencies(
        tmp_path, [runtime], production=False, gaussian_wrapping_repo=runtime
    )
    assert record["commit"] != pinned


def test_production_requires_the_runtime_gaussian_wrapping_checkout_as_a_dependency(tmp_path: Path):
    from real2sim import provenance

    with pytest.raises(ProvenanceError, match="runtime.gaussian_wrapping_repo"):
        provenance.collect_source_dependencies(
            tmp_path, [], production=True, gaussian_wrapping_repo="third_party/GaussianWrapping"
        )
    assert provenance.collect_source_dependencies(tmp_path, [], production=False) == []


if __name__ == "__main__":
    unittest.main()
