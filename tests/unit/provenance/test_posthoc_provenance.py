# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The post-hoc rerender provenance chain of an official run.

``real2sim.reporting.posthoc_provenance`` is what makes a post-hoc canonical
video admissible as evidence: the video must be a replay of the exact state log
the official run wrote, produced by the exact renderer code the official run
ran, against the exact scene the launcher bound.  Every one of those bindings
is a stat-and-hash seal recorded in ``rerender/provenance.json`` and re-checked
here after the fact.

The fixture below is a complete, *passing* official run built on the same
launcher authority as ``tests/unit/launcher/test_worker_contract.py``: a real
``run_manifest.json`` from a resolved plan, a real
``official_worker_contract.json``, and an ``execution_class: official``
generation summary.  The tamper matrix then breaks one link at a time and
pins the diagnostic each break must produce, so a future refactor cannot
quietly stop checking a link.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Iterator

import pytest

from real2sim.reporting.posthoc_provenance import (
    _BATCH_POLICY_LINEAGE_FILENAMES,
    _CORE_RENDERER_SOURCE_PATHS,
    _RERENDER_PROVENANCE_SCHEMA_VERSION,
    _recorded_run_argv,
    _validate_posthoc_provenance,
)
from robosuite_env.contracts import provenance as provenance_contract
from robosuite_env.contracts.worker import write_worker_contract
from robosuite_env.rerender import sources as rerender_sources
from robosuite_env.rerender.common import RerenderError
from tests._helpers.repo import REPO_ROOT
from tests._helpers.sealed_checkout import (
    copy_covered_files,
    private_copy,
    gaussian_wrapping,
    renderer_code as sealed_renderer_code,
    run_manifest as run_manifest_with_gaussian_wrapping,
    replace,
    seal,
)
from tests.unit.launcher.test_worker_contract import (
    _authority,
    _rollout_argv,
    _state_path,
)


FIXTURE_SCENE = REPO_ROOT / "tests" / "fixtures" / "artifacts" / "scene"
SCENE_TAG = "tiny_fixture_scene"
FRAMES = 16
IDENTITY_MATRIX = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
WORLD_OFFSET = [0.0, 0.0, 0.0]
DECODED_SHA256 = "d" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state(path: Path) -> dict[str, int]:
    observed = path.stat()
    return {
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "mode": int(observed.st_mode),
        "size": int(observed.st_size),
        "mtime_ns": int(observed.st_mtime_ns),
        "ctime_ns": int(observed.st_ctime_ns),
    }


def _sealed(path: Path) -> dict[str, Any]:
    """The path + bytes + inode seal every provenance file record carries."""

    return {"path": str(path.resolve()), "sha256": _sha256(path), "state": _state(path)}


def _committed(relative: str, commit: str) -> dict[str, Any]:
    """A core renderer file record whose hash is the file's blob at ``commit``.

    Completion checks a clean render's core files against its commit, so the
    fixture does not depend on edits in this working tree.
    """

    blob = subprocess.run(
        ["git", "cat-file", "blob", f"{commit}:{relative}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    path = REPO_ROOT.resolve() / relative
    return {
        "path": str(path),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "state": _state(path),
    }


def _hashed(path: Path) -> dict[str, Any]:
    """A seal without the stat snapshot, used for launcher-owned records."""

    return {"path": str(path.resolve()), "sha256": _sha256(path)}


class _OfficialRerender:
    """One official rollout whose post-hoc video passes every provenance check."""

    def __init__(
        self,
        run_dir: Path,
        official_manifest_path: Path,
        run_manifest: dict,
        *,
        renderer_commit: str | None = None,
    ):
        self.run_dir = run_dir
        self.official_manifest_path = official_manifest_path
        self.run_manifest = run_manifest
        # The checkout the renderer ran from; a moved-on rerender differs from
        # the generation commit recorded in the manifest.
        self.renderer_commit = (
            str(run_manifest["code"]["commit"])
            if renderer_commit is None
            else renderer_commit
        )
        self.rerender_dir = run_dir / "rerender"
        self.summary = run_dir / "sim_summary.json"
        self.state_log = run_dir / "logs" / "sim_log.jsonl"
        self.run_command = run_dir / "RUN_COMMAND.md"
        self.worker_contract = run_dir / "official_worker_contract.json"
        self.video = run_dir / f"{SCENE_TAG}_{run_dir.name}.mp4"
        self.rerender_summary = self.rerender_dir / "sim_summary.json"
        self.validation_path = self.rerender_dir / "video_validation.json"
        self.provenance_path = self.rerender_dir / "provenance.json"
        self.scene_manifest = FIXTURE_SCENE / "background.json"
        self.scene_ply = FIXTURE_SCENE / "point_cloud.ply"
        self.scene_w2gs = FIXTURE_SCENE / "world_to_gs.txt"
        self.scene_payload = json.loads(
            self.scene_manifest.read_text(encoding="utf-8")
        )

    # -- fixture construction ------------------------------------------------

    def build(self) -> None:
        self.rerender_dir.mkdir(parents=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)

        registration = {
            "gs_w2gs": str(self.scene_w2gs.resolve()),
            "gs_w2gs_raw": IDENTITY_MATRIX,
            "gs_w2gs_effective": IDENTITY_MATRIX,
            "gs_calibration_world_offset_m": WORLD_OFFSET,
        }
        self.summary.write_text(
            json.dumps(
                {
                    "execution_class": "official",
                    "official_run_manifest": str(
                        self.official_manifest_path.resolve()
                    ),
                    "official_worker_contract": str(
                        self.worker_contract.resolve()
                    ),
                    "frames": FRAMES,
                    "rendered_frames": 0,
                    "wall_stopped_early": False,
                    **registration,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.state_log.write_text(
            "".join(json.dumps({"frame": frame}) + "\n" for frame in range(FRAMES)),
            encoding="utf-8",
        )
        self.run_command.write_text(
            "# Run command\n\n```bash\n"
            f"{sys.executable} robosuite_env/sim.py --no-render\n```\n",
            encoding="utf-8",
        )
        self.video.write_bytes(b"official-posthoc-video")
        self.rerender_summary.write_text(
            json.dumps(
                {
                    "frames": FRAMES,
                    "rendered_frames": FRAMES,
                    **registration,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.validation = {
            "ok": True,
            "video": str(self.video.resolve()),
            "summary": str(self.rerender_summary.resolve()),
            "decoded_frames": FRAMES,
            "decoded_sha256": DECODED_SHA256,
            "file_size_bytes": self.video.stat().st_size,
            "rerender_summary_sha256": _sha256(self.rerender_summary),
            "gs_registration_exact_match": True,
        }
        self.validation_path.write_text(
            json.dumps(self.validation, sort_keys=True), encoding="utf-8"
        )
        self.write_provenance(self.baseline_provenance())

    def baseline_provenance(self) -> dict[str, Any]:
        commit = self.code_commit
        state_log_record = _sealed(self.state_log)
        state_log_record["rows"] = FRAMES
        registration_file = _sealed(self.scene_w2gs)
        return {
            "schema_version": _RERENDER_PROVENANCE_SCHEMA_VERSION,
            "kind": "real2sim-rerender-provenance",
            "mode": "official",
            "created_at": "2026-08-26T00:00:00Z",
            "source": {
                "generation_summary": _sealed(self.summary),
                "state_log": state_log_record,
                "run_command": _sealed(self.run_command),
                "batch_policy_lineage": {
                    "batch_root": str(self.run_dir.parents[1].resolve()),
                    "optional_filenames": list(_BATCH_POLICY_LINEAGE_FILENAMES),
                    "files": {
                        name: _sealed(self.run_dir.parents[1] / name)
                        for name in _BATCH_POLICY_LINEAGE_FILENAMES
                        if (self.run_dir.parents[1] / name).is_file()
                    },
                },
                "expected_frames": FRAMES,
                "gs_registration": {
                    "kind": "file",
                    "path": str(self.scene_w2gs.resolve()),
                    "file": registration_file,
                    "raw_matrix": IDENTITY_MATRIX,
                    "effective_matrix": IDENTITY_MATRIX,
                    "calibration_world_offset_m": WORLD_OFFSET,
                },
                "run_manifest": _hashed(self.official_manifest_path),
                "worker_contract": _hashed(self.worker_contract),
                "code_commit": commit,
                "official_generation_check": {
                    "rule": provenance_contract.RULE_GENERATION_COMMIT,
                    "checkout_root": str(REPO_ROOT.resolve()),
                    "commit": self.renderer_commit,
                    "dirty": False,
                    "run_seal": None,
                    "coverage": None,
                    "files": {},
                    "unchanged_at_write": True,
                },
            },
            "renderer": {
                "sanitized_argv": [
                    sys.executable,
                    "robosuite_env/sim.py",
                    "--state-log-replay",
                    str(self.state_log.resolve()),
                    "--out",
                    str(self.video.resolve()),
                    "--summary-path",
                    str(self.rerender_summary.resolve()),
                ],
                "code": {
                    "commit": self.renderer_commit,
                    "dirty": False,
                    "core_source_files": {
                        relative: _committed(relative, self.renderer_commit)
                        if self.renderer_commit == _head_commit()
                        else _sealed(REPO_ROOT / relative)
                        for relative in _CORE_RENDERER_SOURCE_PATHS
                    },
                    "tracked_diff": {
                        "base_commit": self.renderer_commit,
                        "sha256": hashlib.sha256(b"").hexdigest(),
                        "size_bytes": 0,
                    },
                    "gaussian_wrapping": gaussian_wrapping(REPO_ROOT.resolve()),
                },
            },
            "scene": {
                "tag": SCENE_TAG,
                "id": self.scene_payload["id"],
                "contract_digest": self.scene_digest,
                "manifest": _sealed(self.scene_manifest),
                "ply": _sealed(self.scene_ply),
                "w2gs": {
                    "kind": "file",
                    "path": str(self.scene_w2gs.resolve()),
                    "file": registration_file,
                    "raw_matrix": IDENTITY_MATRIX,
                },
            },
            "output": {
                "video": _sealed(self.video),
                "validation": _sealed(self.validation_path),
                "rerender_summary": _sealed(self.rerender_summary),
                "decoded_frames": FRAMES,
                "decoded_sha256": DECODED_SHA256,
            },
        }

    @property
    def code_commit(self) -> str:
        return str(self.run_manifest["code"]["commit"])

    scene_digest = "e" * 64

    def write_provenance(self, payload: dict[str, Any]) -> None:
        self.provenance_path.write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )

    # -- the call under test -------------------------------------------------

    def errors(self, validation: dict[str, Any] | None = None) -> list[str]:
        return _validate_posthoc_provenance(
            provenance_path=self.provenance_path,
            validation_path=self.validation_path,
            validation=self.validation if validation is None else validation,
            expected_video=self.video,
            expected_summary=self.summary,
            expected_frames=FRAMES,
            run_dir=self.run_dir,
            scene_tag_value=SCENE_TAG,
            official_manifest_path=self.official_manifest_path,
            run_manifest=self.run_manifest,
            expected_scene_id=self.scene_payload["id"],
            expected_scene_digest=self.scene_digest,
            expected_scene_manifest=self.scene_manifest,
            expected_scene_ply_sha256=self.scene_payload["ply_sha256"],
        )


@pytest.fixture(autouse=True)
def _isolate_cpu_fixture_from_ambient_cuda(monkeypatch):
    # The CPU fixture profile forbids ambient behavior-changing variables.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def _head_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@contextlib.contextmanager
def _built_official_rerender(
    tmp_path: Path,
    *,
    code: dict[str, Any] | None = None,
    renderer_commit: str | None = None,
) -> Iterator[_OfficialRerender]:
    with _authority(tmp_path) as (manifest_path, manifest, _resolved):
        # build_run_manifest() records the generation commit here; the post-hoc
        # renderer must prove it ran that exact code.
        manifest["code"] = (
            {"commit": _head_commit(), "dirty": False, "path": str(REPO_ROOT.resolve())}
            if code is None
            else code
        )
        # build_run_manifest() records the GaussianWrapping checkout at launch.
        manifest["source_dependencies"] = run_manifest_with_gaussian_wrapping(
            manifest["code"]
        )["source_dependencies"]
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        run_dir = manifest_path.parent / SCENE_TAG / "run_01"
        run_dir.mkdir(parents=True)
        contract = run_dir / "official_worker_contract.json"
        write_worker_contract(
            contract,
            manifest_path,
            _rollout_argv(
                manifest,
                contract,
                state_path=_state_path(manifest["resolved_environment"]),
            ),
        )

        fixture = _OfficialRerender(
            run_dir, manifest_path, manifest, renderer_commit=renderer_commit
        )
        fixture.build()
        yield fixture


@pytest.fixture
def official_rerender(tmp_path: Path) -> Iterator[_OfficialRerender]:
    with _built_official_rerender(tmp_path) as fixture:
        yield fixture


def test_a_complete_official_rerender_passes_every_provenance_check(
    official_rerender: _OfficialRerender,
):
    assert official_rerender.errors() == []


#: A generation commit the checkout has since moved past.  Neither side looks
#: it up in git; they only compare it with the renderer's own commit.
_MOVED_ON_GENERATION_COMMIT = "b" * 40
_CODE_REJECTION = "official rerender cannot stand in for the run's code"


#: A rendering module the replay worker runs that is not a core renderer file.
_NON_CORE_RENDERING_SOURCE = "robosuite_env/rendering/mujoco_kb4.py"


@pytest.mark.parametrize(
    "seal_case", ["matching", "stale", "missing", "stale-non-core", "unsealed-non-core"]
)
def test_moved_on_official_rerender_gets_one_answer_from_renderer_and_completion(
    tmp_path: Path, seal_case: str
):
    """The renderer's start check and completion reporting must agree.

    A run generated at an older commit is rerendered from a later checkout.
    The renderer accepts it only when the seal covers what the replay reads,
    every core renderer file matches the launcher's runtime seal and every
    other sealed source is unchanged in the checkout, and it records that
    check; completion reporting must accept the render it records, and must
    reject a record made against any seal but the run's. ``stale-non-core``
    is a checkout whose replay-side rendering module differs from what the
    run sealed, and ``unsealed-non-core`` one that holds a file in a sealed
    tree that the run did not have.
    """

    from real2sim import runtime_manifest

    with _built_official_rerender(
        tmp_path,
        code={
            "commit": _MOVED_ON_GENERATION_COMMIT,
            "dirty": False,
            "path": str(REPO_ROOT),
        },
        renderer_commit=_head_commit(),
    ) as fixture:
        run_root = fixture.official_manifest_path.parent
        seal_path = run_root / runtime_manifest.MANIFEST_FILE
        # The seal the launcher really writes for the current sources.
        capture = runtime_manifest.capture_selection(
            runtime_manifest.default_selection(REPO_ROOT)
        )
        provenance = fixture.baseline_provenance()
        renderer_code = provenance["renderer"]["code"]
        renderer_code["core_source_files"] = {
            relative: _sealed(REPO_ROOT / relative)
            for relative in _CORE_RENDERER_SOURCE_PATHS
        }
        assert renderer_code["commit"] != _MOVED_ON_GENERATION_COMMIT
        contract_argv = json.loads(
            fixture.worker_contract.read_text(encoding="utf-8")
        )["argv"]
        recorded_argv = [
            sys.executable,
            "robosuite_env/sim.py",
            *contract_argv,
            "--official-worker-contract",
            str(fixture.worker_contract),
        ]
        summary = json.loads(fixture.summary.read_text(encoding="utf-8"))

        def renderer_start() -> dict[str, Any]:
            _authority_record, records = rerender_sources._validate_official_source(
                fixture.run_dir, recorded_argv, summary, renderer_code
            )
            return records

        # The check a renderer records against the run's genuine seal.
        seal_path.write_text(json.dumps(capture), encoding="utf-8")
        records = renderer_start()
        assert records["code_commit"] == _MOVED_ON_GENERATION_COMMIT
        record = records["official_generation_check"]
        assert record["rule"] == provenance_contract.RULE_SEALED_SOURCES
        provenance["source"]["official_generation_check"] = {
            **record,
            "unchanged_at_write": True,
        }
        fixture.write_provenance(provenance)

        if seal_case == "stale":
            stale_path = str(REPO_ROOT / _CORE_RENDERER_SOURCE_PATHS[0])
            stale = [r for r in capture["files"] if r["path"] == stale_path]
            assert len(stale) == 1
            stale[0]["sha256"] = "0" * 64
        if seal_case in {"stale-non-core", "unsealed-non-core"}:
            assert _NON_CORE_RENDERING_SOURCE not in _CORE_RENDERER_SOURCE_PATHS
            non_core_path = str(REPO_ROOT / _NON_CORE_RENDERING_SOURCE)
            non_core = [r for r in capture["files"] if r["path"] == non_core_path]
            assert len(non_core) == 1
            if seal_case == "stale-non-core":
                non_core[0]["sha256"] = "0" * 64
            else:
                capture["files"].remove(non_core[0])
        if seal_case == "missing":
            seal_path.unlink()
        elif seal_case != "matching":
            seal_path.write_text(json.dumps(capture), encoding="utf-8")

        completion_errors = fixture.errors()

        if seal_case == "matching":
            assert completion_errors == []
        else:
            with pytest.raises(RerenderError, match="requires clean checkout"):
                renderer_start()
            assert completion_errors, "completion accepted a check of another seal"
            assert all(
                _CODE_REJECTION in error for error in completion_errors
            ), completion_errors


@pytest.fixture(scope="module")
def pristine_checkout(tmp_path_factory) -> Path:
    return copy_covered_files(tmp_path_factory.mktemp("pristine") / "checkout")


@contextlib.contextmanager
def _moved_on_render_of_a_copy(tmp_path: Path, pristine: Path):
    """An official run sealed in a copy of this checkout, rerendered from it.

    Yields the fixture, the copy and the provenance carrying the check the
    renderer records for it.
    """

    checkout = private_copy(pristine, tmp_path / "checkout").resolve()
    (tmp_path / "run").mkdir()
    manifest_code = {
        "commit": _MOVED_ON_GENERATION_COMMIT,
        "dirty": False,
        "path": str(checkout),
    }
    with _built_official_rerender(
        tmp_path / "run", code=manifest_code, renderer_commit="c" * 40
    ) as fixture:
        run_root = fixture.official_manifest_path.parent
        seal(checkout, run_root)
        code = sealed_renderer_code(checkout, commit="c" * 40)
        record = provenance_contract.official_generation_check(
            run_root, fixture.run_manifest, code, checkout_root=checkout
        )
        assert record is not None
        provenance = fixture.baseline_provenance()
        provenance["renderer"]["code"] = code
        provenance["source"]["official_generation_check"] = {
            **record,
            "unchanged_at_write": True,
        }
        fixture.write_provenance(provenance)
        yield fixture, checkout, provenance


def test_completion_keeps_the_renderers_answer_when_the_checkout_changes_afterwards(
    tmp_path: Path, pristine_checkout: Path
):
    with _moved_on_render_of_a_copy(tmp_path, pristine_checkout) as (
        fixture,
        checkout,
        _provenance,
    ):
        assert fixture.errors() == []

        # A pull or an edit between rendering and reporting: a sealed asset, a
        # core renderer file and the launcher change, and a task module appears.
        replace(checkout / "robosuite_env/assets/box.xml", b"<mujoco/>\n")
        replace(checkout / "robosuite_env/gs_launch.py", b"# edited\n")
        replace(checkout / "src/real2sim/launcher.py", b"# edited\n")
        (checkout / "robosuite_env/tasks/new_task.py").write_text("", encoding="utf-8")

        run_root = fixture.official_manifest_path.parent
        assert (
            provenance_contract.official_generation_check(
                run_root,
                fixture.run_manifest,
                sealed_renderer_code(checkout, commit="c" * 40),
                checkout_root=checkout,
            )
            is None
        ), "a renderer starting now must refuse"
        assert fixture.errors() == []


def _forge_file_digest(record: dict[str, Any]) -> None:
    relative = "robosuite_env/assets/box.xml"
    record["files"][relative] = "0" * 64


def _forge_seal_bytes(record: dict[str, Any]) -> None:
    record["run_seal"]["sha256"] = "0" * 64


def _forge_coverage(record: dict[str, Any]) -> None:
    record["coverage"] = {"files": [], "trees": []}


def _forge_commit_rule(record: dict[str, Any]) -> None:
    record["rule"] = provenance_contract.RULE_GENERATION_COMMIT


def _forge_commit(record: dict[str, Any]) -> None:
    record["commit"] = "d" * 40


def _forge_changed_while_rendering(record: dict[str, Any]) -> None:
    record["unchanged_at_write"] = False


def _forge_missing_key(record: dict[str, Any]) -> None:
    del record["files"]


def _forge_coverage_without_a_core_file(record: dict[str, Any]) -> None:
    record["coverage"]["files"].remove("robosuite_env/gs_launch.py")


FORGED_RECORDS = (
    (_forge_file_digest, "did not compare exactly the sealed sources"),
    (_forge_seal_bytes, "made against other seal bytes"),
    (_forge_coverage, "does not cover what the renderer's runtime reads"),
    (
        _forge_coverage_without_a_core_file,
        "does not cover what the renderer's runtime reads",
    ),
    (_forge_commit_rule, "claims the run's clean commit"),
    (_forge_commit, "names a different checkout commit"),
    (_forge_changed_while_rendering, "changed while the rerender ran"),
    (_forge_missing_key, "has no valid record of its official generation check"),
)


@pytest.mark.parametrize(
    ("forge", "expected"),
    FORGED_RECORDS,
    ids=[forge.__name__.removeprefix("_forge_") for forge, _ in FORGED_RECORDS],
)
def test_an_inconsistent_official_generation_check_is_refused(
    tmp_path: Path, pristine_checkout: Path, forge, expected: str
):
    with _moved_on_render_of_a_copy(tmp_path, pristine_checkout) as (
        fixture,
        _checkout,
        provenance,
    ):
        forged = copy.deepcopy(provenance)
        forge(forged["source"]["official_generation_check"])
        fixture.write_provenance(forged)

        errors = fixture.errors()
        assert any(
            _CODE_REJECTION in error and expected in error for error in errors
        ), errors


def _break_schema(payload: dict[str, Any]) -> None:
    payload["schema_version"] = _RERENDER_PROVENANCE_SCHEMA_VERSION - 1


def _break_mode(payload: dict[str, Any]) -> None:
    # A diagnostic rerender is not admissible as an official canonical video.
    payload["mode"] = "diagnostic"


def _break_generation_summary_hash(payload: dict[str, Any]) -> None:
    payload["source"]["generation_summary"]["sha256"] = "0" * 64


def _break_state_log_rows(payload: dict[str, Any]) -> None:
    payload["source"]["state_log"]["rows"] = FRAMES - 1


def _break_run_manifest_binding(payload: dict[str, Any]) -> None:
    payload["source"]["run_manifest"]["sha256"] = "0" * 64


def _break_worker_contract_binding(payload: dict[str, Any]) -> None:
    payload["source"]["worker_contract"]["sha256"] = "0" * 64


def _break_code_commit(payload: dict[str, Any]) -> None:
    payload["source"]["code_commit"] = "0" * 40


def _break_renderer_cleanliness(payload: dict[str, Any]) -> None:
    payload["renderer"]["code"]["dirty"] = True


def _break_renderer_source_file(payload: dict[str, Any]) -> None:
    relative = _CORE_RENDERER_SOURCE_PATHS[0]
    payload["renderer"]["code"]["core_source_files"][relative]["sha256"] = "0" * 64


def _break_tracked_diff(payload: dict[str, Any]) -> None:
    payload["renderer"]["code"]["tracked_diff"]["sha256"] = "0" * 64


def _break_replay_source(payload: dict[str, Any]) -> None:
    argv = payload["renderer"]["sanitized_argv"]
    argv[argv.index("--state-log-replay") + 1] = str(
        Path(argv[argv.index("--state-log-replay") + 1]).with_name("other.jsonl")
    )


def _break_argv_purity(payload: dict[str, Any]) -> None:
    # A canonical video may not be produced by a run that was told not to render.
    payload["renderer"]["sanitized_argv"].append("--no-render")


def _break_view(payload: dict[str, Any]) -> None:
    payload["renderer"]["view_overrides"] = {"view_azimuth": 90}


def _break_scene_tag(payload: dict[str, Any]) -> None:
    payload["scene"]["tag"] = "some_other_scene"


def _break_scene_id(payload: dict[str, Any]) -> None:
    payload["scene"]["id"] = "some-other-scene-v2"


def _break_scene_digest(payload: dict[str, Any]) -> None:
    payload["scene"]["contract_digest"] = "f" * 64


def _break_scene_ply(payload: dict[str, Any]) -> None:
    payload["scene"]["ply"]["sha256"] = "0" * 64


def _break_scene_registration_kind(payload: dict[str, Any]) -> None:
    payload["scene"]["w2gs"] = {
        "kind": "identity",
        "path": None,
        "file": None,
        "raw_matrix": None,
    }


def _break_registration_agreement(payload: dict[str, Any]) -> None:
    payload["scene"]["w2gs"]["raw_matrix"] = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _break_output_video_seal(payload: dict[str, Any]) -> None:
    payload["output"]["video"]["sha256"] = "0" * 64


def _break_output_frame_count(payload: dict[str, Any]) -> None:
    payload["output"]["decoded_frames"] = FRAMES - 1


def _break_decoded_hash(payload: dict[str, Any]) -> None:
    payload["output"]["decoded_sha256"] = "a" * 64


def _break_rerender_summary_seal(payload: dict[str, Any]) -> None:
    payload["output"]["rerender_summary"]["sha256"] = "0" * 64


TAMPER_CASES: tuple[tuple[str, Callable[[dict[str, Any]], None], str], ...] = (
    ("schema downgrade", _break_schema, "rerender provenance schema_version must be"),
    ("diagnostic mode", _break_mode, "rerender provenance mode="),
    (
        "generation summary bytes",
        _break_generation_summary_hash,
        "generation summary provenance SHA-256",
    ),
    ("state log row count", _break_state_log_rows, "state log provenance row count"),
    (
        "official run manifest",
        _break_run_manifest_binding,
        "official run manifest provenance SHA-256",
    ),
    (
        "official worker contract",
        _break_worker_contract_binding,
        "official worker contract provenance SHA-256",
    ),
    (
        "generation code commit",
        _break_code_commit,
        "rerender source code commit differs from the official run",
    ),
    (
        "dirty renderer checkout",
        _break_renderer_cleanliness,
        "official rerender cannot stand in for the run's code",
    ),
    (
        "renderer source bytes",
        _break_renderer_source_file,
        "renderer source robosuite_env/sim.py provenance SHA-256 does not match commit",
    ),
    (
        "tracked diff",
        _break_tracked_diff,
        "rerender tracked diff provenance is not empty for a clean renderer checkout",
    ),
    (
        "replayed a different state log",
        _break_replay_source,
        "rerender sanitized argv has an invalid --state-log-replay binding",
    ),
    (
        "generation-only options retained",
        _break_argv_purity,
        "rerender sanitized argv retains forbidden generation options",
    ),
    (
        "third-person view override",
        _break_view,
        "rerender used third-person view overrides",
    ),
    ("scene tag", _break_scene_tag, "rerender provenance scene tag is invalid"),
    (
        "scene id",
        _break_scene_id,
        "rerender provenance scene id differs from launch preflight",
    ),
    (
        "scene contract digest",
        _break_scene_digest,
        "rerender provenance scene contract differs from launch preflight",
    ),
    ("scene PLY", _break_scene_ply, "rerender scene PLY differs from launch preflight"),
    (
        "scene registration kind",
        _break_scene_registration_kind,
        "rerender scene GS registration path differs from manifest",
    ),
    (
        "registration disagreement",
        _break_registration_agreement,
        "rerender scene GS registration differs from generation",
    ),
    (
        "video bytes",
        _break_output_video_seal,
        "rerender video provenance SHA-256",
    ),
    (
        "decoded frame count",
        _break_output_frame_count,
        "rerender provenance decoded frame count is invalid",
    ),
    (
        "decoded video hash",
        _break_decoded_hash,
        "rerender provenance decoded SHA-256 is invalid",
    ),
    (
        "rerender summary bytes",
        _break_rerender_summary_seal,
        "rerender summary provenance SHA-256",
    ),
)


@pytest.mark.parametrize(
    ("label", "mutate", "expected"),
    TAMPER_CASES,
    ids=[case[0] for case in TAMPER_CASES],
)
def test_every_broken_provenance_link_is_reported(
    official_rerender: _OfficialRerender,
    label: str,
    mutate: Callable[[dict[str, Any]], None],
    expected: str,
):
    tampered = copy.deepcopy(official_rerender.baseline_provenance())
    mutate(tampered)
    official_rerender.write_provenance(tampered)

    errors = official_rerender.errors()
    assert any(expected in error for error in errors), errors


def test_replacing_a_sealed_file_in_place_is_caught_by_its_stat_snapshot(
    official_rerender: _OfficialRerender,
):
    # A same-length rewrite keeps every recorded byte count intact, so only the
    # inode/mtime seal separates the sealed video from a substituted one.
    original = official_rerender.video.read_bytes()
    official_rerender.video.unlink()
    official_rerender.video.write_bytes(original)

    errors = official_rerender.errors()
    assert any(
        "rerender video provenance stat snapshot" in error for error in errors
    ), errors


def test_a_truncated_state_log_fails_even_with_a_consistent_row_count(
    official_rerender: _OfficialRerender,
):
    # The row count is checked against both the record and the run's frame
    # budget, so re-sealing a short log does not launder it.
    official_rerender.state_log.write_text(
        "".join(json.dumps({"frame": frame}) + "\n" for frame in range(FRAMES - 1)),
        encoding="utf-8",
    )
    payload = official_rerender.baseline_provenance()
    payload["source"]["state_log"]["rows"] = FRAMES - 1
    official_rerender.write_provenance(payload)

    errors = official_rerender.errors()
    assert any(
        f"state log has {FRAMES - 1} rows, expected {FRAMES}" in error
        for error in errors
    ), errors


def test_a_missing_validation_attestation_is_rejected(
    official_rerender: _OfficialRerender,
):
    validation = dict(official_rerender.validation)
    validation.pop("gs_registration_exact_match")

    errors = official_rerender.errors(validation=validation)
    assert any(
        "video validation has no exact GS registration attestation" in error
        for error in errors
    ), errors


def test_a_validation_that_disagrees_with_the_rerender_summary_is_rejected(
    official_rerender: _OfficialRerender,
):
    validation = dict(official_rerender.validation)
    validation["rerender_summary_sha256"] = "0" * 64

    errors = official_rerender.errors(validation=validation)
    assert any(
        "rerender summary provenance differs from video validation" in error
        for error in errors
    ), errors


def test_an_unreadable_provenance_record_short_circuits(
    official_rerender: _OfficialRerender,
):
    official_rerender.provenance_path.write_text("{", encoding="utf-8")

    assert official_rerender.errors() == [
        "rerender provenance is missing or unreadable: JSONDecodeError"
    ]


def test_an_unknown_top_level_key_is_rejected_outright(
    official_rerender: _OfficialRerender,
):
    payload = official_rerender.baseline_provenance()
    payload["extra"] = True
    official_rerender.write_provenance(payload)

    assert official_rerender.errors() == [
        "rerender provenance has an invalid top-level structure"
    ]


def test_partial_playback_cadence_provenance_is_rejected(
    official_rerender: _OfficialRerender,
):
    # The cadence contract is optional, but it is a producer-consumer pair: it
    # must appear in the source, the top level, and the video validation, or in
    # none of them.
    payload = official_rerender.baseline_provenance()
    payload["policy_action_playback_contract"] = {"source_frames": FRAMES}
    official_rerender.write_provenance(payload)

    errors = official_rerender.errors()
    assert any(
        "policy-action playback provenance must be present" in error
        for error in errors
    ), errors


def test_completion_checks_the_renderers_recorded_coverage_and_core_list(
    tmp_path: Path, pristine_checkout: Path, monkeypatch
):
    """A pull between render and report that widens what the code reads.

    The reporting process's coverage and core list grow; the render its
    renderer accepted against its own recorded coverage and core list stays
    accepted.
    """

    from real2sim import runtime_manifest

    with _moved_on_render_of_a_copy(tmp_path, pristine_checkout) as (
        fixture,
        _checkout,
        _provenance,
    ):
        widened = runtime_manifest.required_coverage()
        widened["files"] = sorted([*widened["files"], "config/added-later.yaml"])
        widened["trees"] = [*widened["trees"], {"path": "zz_added_later", "suffixes": ["*"]}]
        monkeypatch.setattr(runtime_manifest, "required_coverage", lambda: widened)
        added_core = (*_CORE_RENDERER_SOURCE_PATHS, "robosuite_env/added_later.py")
        monkeypatch.setattr(provenance_contract, "CORE_RENDERER_SOURCE_PATHS", added_core)
        monkeypatch.setattr(
            "real2sim.reporting.posthoc_provenance._CORE_RENDERER_SOURCE_PATHS",
            added_core,
        )

        assert fixture.errors() == []


def test_completion_refuses_a_renderer_with_no_core_files(
    tmp_path: Path, pristine_checkout: Path
):
    with _moved_on_render_of_a_copy(tmp_path, pristine_checkout) as (
        fixture,
        _checkout,
        provenance,
    ):
        provenance["renderer"]["code"]["core_source_files"] = {}
        fixture.write_provenance(provenance)

        errors = fixture.errors()
        assert any("core source provenance is invalid" in error for error in errors), errors
        assert any("do not hash to the seal" in error for error in errors), errors


@pytest.mark.parametrize(
    ("head", "accepted"),
    (
        ("robosuite_env/sim.py", True),  # runs recorded before the -m launch
        ("-m robosuite_env.sim", True),
        ("-m robosuite_env.other", False),
    ),
)
def test_recorded_run_argv_accepts_both_worker_heads(tmp_path, head, accepted):
    run_command = tmp_path / "RUN_COMMAND.md"
    run_command.write_text(
        f"# Run command\n\n```bash\nMUJOCO_GL=egl /venv/bin/py {head} --frames 3\n```\n",
        encoding="utf-8",
    )
    if not accepted:
        with pytest.raises(ValueError, match="unexpected recorded command head"):
            _recorded_run_argv(run_command)
        return
    assert _recorded_run_argv(run_command) == ["/venv/bin/py", *head.split(), "--frames", "3"]
