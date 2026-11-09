# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0


import json

from pathlib import Path



import numpy as np

import pytest



import robosuite_env.tools.rerender_third_person as rerender


from robosuite_env.rerender import gs_server as rerender_gs_server

from robosuite_env.rerender import provenance as rerender_provenance


from robosuite_env.rerender import sources as rerender_sources
from robosuite_env.contracts import provenance as provenance_contract


from real2sim.runner.argv import worker_command_head_length
from tests._helpers.sealed_checkout import gaussian_wrapping, run_manifest
from tests._helpers.rollouts import (
    LEGACY_WORKER_HEAD,
    MODULE_WORKER_HEAD,
    RUN_COMMAND,
    use_module_worker_head,
    _configure_complete_source,
    _configure_render_state_source,
    run_dir,  # noqa: F401 -- pytest fixture
)

def test_prepare_source_run_accepts_only_complete_sequential_input(
    run_dir: Path,
) -> None:
    _configure_complete_source(run_dir, 3)

    source = rerender.prepare_source_run(
        run_dir,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )

    assert source.expected_frames == 3
    assert [row["frame"] for row in source.state_rows] == [0, 1, 2]
    assert source.official is False
    assert set(source.source_records) == {
        "generation_summary",
        "state_log",
        "run_command",
        "batch_policy_lineage",
        "expected_frames",
        "gs_registration",
    }
    assert source.source_records["state_log"]["rows"] == 3
    assert source.source_records["batch_policy_lineage"]["files"] == {}


def test_prepare_source_run_requires_the_full_frame_budget(
    run_dir: Path,
) -> None:
    """A rollout may not be shorter than its --frames budget."""

    _configure_complete_source(run_dir, 3)
    command_path = run_dir / "RUN_COMMAND.md"
    command_path.write_text(
        command_path.read_text(encoding="utf-8").replace("--frames 3", "--frames 9"),
        encoding="utf-8",
    )

    with pytest.raises(rerender.RerenderError, match="expected final --frames=9"):
        rerender.prepare_source_run(
            run_dir,
            renderer_code={"commit": "a" * 40, "dirty": True},
        )


def test_prepare_source_run_captures_exact_file_registration(run_dir: Path) -> None:
    source = rerender.prepare_source_run(
        run_dir,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )

    registration = source.source_records["gs_registration"]
    expected_path = Path(source.summary["gs_w2gs"]).resolve()
    assert registration == {
        "kind": "file",
        "path": str(expected_path),
        "file": rerender._file_record(expected_path, include_state=True),
        "raw_matrix": np.eye(4).tolist(),
        "effective_matrix": np.eye(4).tolist(),
        "calibration_world_offset_m": [0.0, 0.0, 0.0],
    }

@pytest.mark.parametrize("registration_error", ("raw_file", "effective_offset"))
def test_prepare_source_run_rejects_raw_matrix_that_differs_from_file(
    run_dir: Path,
    registration_error: str,
) -> None:
    summary_path = run_dir / "sim_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if registration_error == "raw_file":
        changed = np.eye(4)
        changed[0, 3] = 0.125
        summary["gs_w2gs_raw"] = changed.tolist()
        summary["gs_w2gs_effective"] = changed.tolist()
        match = "differs from its registered file"
    else:
        summary["gs_calibration_world_offset_m"] = [0.1, 0.0, 0.0]
        match = r"effective GS matrix does not equal raw @ Translation\(offset\)"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(rerender.RerenderError, match=match):
        rerender.prepare_source_run(
            run_dir,
            renderer_code={"commit": "a" * 40, "dirty": True},
        )

def test_registration_capture_preserves_explicit_transform_and_inline_matrix(
    run_dir: Path,
) -> None:
    command_path = run_dir / "RUN_COMMAND.md"
    explicit_path = Path(
        json.loads((run_dir / "sim_summary.json").read_text())["gs_w2gs"]
    ).resolve()
    command_path.write_text(
        command_path.read_text(encoding="utf-8").replace(
            "--no-render", f"--gs-transform {explicit_path} --no-render"
        ),
        encoding="utf-8",
    )
    explicit_source = rerender.prepare_source_run(
        run_dir,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )
    explicit_current = rerender._current_gs_registration(
        {"w2gs_path": None, "w2gs_matrix": np.eye(4)}, explicit_source, {}
    )
    assert explicit_current["kind"] == "file"
    assert explicit_current["path"] == str(explicit_path)

    command_path.write_text(RUN_COMMAND, encoding="utf-8")
    summary_path = run_dir / "sim_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["gs_w2gs"] = "inline-matrix"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    inline_source = rerender.prepare_source_run(
        run_dir,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )
    inline_current = rerender._current_gs_registration(
        {"w2gs_path": None, "w2gs_matrix": np.eye(4)},
        inline_source,
        {"mat_w2gs": np.eye(4).tolist()},
    )
    rerender._assert_registration_matches(
        inline_source.source_records["gs_registration"],
        inline_current,
        label="inline test",
        compare_effective=False,
    )
    assert inline_current["kind"] == "inline"
    with pytest.raises(rerender.RerenderError, match="registration source differs"):
        rerender._assert_registration_matches(
            inline_source.source_records["gs_registration"],
            explicit_current,
            label="inline-to-file drift",
            compare_effective=False,
        )

    # Inline and identity registrations can share the identity matrix while
    # still representing different source kinds; that drift must fail closed.
    summary_path = run_dir / "sim_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["gs_w2gs"] = "inline-matrix"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    inline_source = rerender.prepare_source_run(
        run_dir,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )
    current_identity = rerender._current_gs_registration(
        {"w2gs_path": None, "w2gs_matrix": np.eye(4)},
        inline_source,
        {},
    )
    with pytest.raises(rerender.RerenderError, match="differs from generation"):
        rerender._assert_registration_matches(
            inline_source.source_records["gs_registration"],
            current_identity,
            label="inline-to-identity drift",
            compare_effective=False,
        )

    summary.update(
        {
            "gs_w2gs": None,
            "gs_w2gs_raw": None,
            "gs_w2gs_effective": None,
        }
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    identity_source = rerender.prepare_source_run(
        run_dir,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )
    current_inline = rerender._current_gs_registration(
        {"w2gs_path": None, "w2gs_matrix": np.eye(4)},
        identity_source,
        {"mat_w2gs": np.eye(4).tolist()},
    )
    with pytest.raises(rerender.RerenderError, match="differs from generation"):
        rerender._assert_registration_matches(
            identity_source.source_records["gs_registration"],
            current_inline,
            label="identity-to-inline drift",
            compare_effective=False,
        )

def test_prepare_source_run_accepts_valid_complete_render_state(
    run_dir: Path,
) -> None:
    contract = _configure_render_state_source(run_dir, frames=3)

    source = rerender.prepare_source_run(
        run_dir,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )

    validation = source.source_records["render_state_log_validation"]
    assert validation["rows_written"] == 3
    assert validation["complete"] is True
    assert validation["model_signature"] == contract["model_signature"]
    assert validation["inventory_sha256"] == contract["inventory_sha256"]
    assert validation["target"] == "cup"
    assert len(validation["task_object_names"]) == 2

    # Legacy state logs remain valid when no render-state contract was
    # recorded. Reset this fixture to that older summary/log shape.
    summary_path = run_dir / "sim_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("render_state_log")
    summary["objects"] = "cup_plate"
    summary["frames"] = 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    command_path = run_dir / "RUN_COMMAND.md"
    command_path.write_text(
        command_path.read_text(encoding="utf-8").replace(
            "--frames 3", "--frames 1"
        ),
        encoding="utf-8",
    )
    (run_dir / "logs" / "sim_log.jsonl").write_text(
        '{"frame": 0}\n', encoding="utf-8"
    )

    legacy_source = rerender.prepare_source_run(
        run_dir,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )
    assert "render_state_log_validation" not in legacy_source.source_records

def test_prepare_source_run_rejects_render_state_rows_without_contract(
    run_dir: Path,
) -> None:
    _configure_render_state_source(run_dir, frames=1)
    summary_path = run_dir / "sim_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    del summary["render_state_log"]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(
        rerender.RerenderError,
        match="without a summary contract",
    ):
        rerender.prepare_source_run(
            run_dir,
            renderer_code={"commit": "a" * 40, "dirty": True},
        )

def test_prepare_source_run_binds_optional_batch_policy_lineage(
    run_dir: Path,
) -> None:
    batch_root = run_dir.parents[1]
    contents = {
        "resolved_rollout_config.json": b'{"config": "resolved"}\n',
        "policy_batch_server_ready.json": b'{"ready": true}\n',
        "policy_inference_manifest.jsonl": b'{"request": 1}\n',
        "rollout_config.yaml": b"repeat: 30\n",
    }
    for filename, content in contents.items():
        (batch_root / filename).write_bytes(content)

    source = rerender.prepare_source_run(
        run_dir,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )

    lineage = source.source_records["batch_policy_lineage"]
    assert lineage["batch_root"] == str(batch_root.resolve())
    assert set(lineage["files"]) == set(contents)
    for filename, content in contents.items():
        assert lineage["files"][filename]["sha256"] == rerender.hashlib.sha256(
            content
        ).hexdigest()
        assert "state" in lineage["files"][filename]

def test_batch_policy_lineage_recheck_detects_content_and_presence_changes(
    run_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Changed bytes are reported and the render goes on. A file appearing or
    # vanishing is refused.
    batch_root = run_dir.parents[1]
    config = batch_root / "resolved_rollout_config.json"
    config.write_text("{}\n", encoding="utf-8")
    lineage = rerender._capture_batch_policy_lineage(run_dir)

    config.write_text('{"changed": true}\n', encoding="utf-8")
    rerender._assert_batch_policy_lineage_current(lineage)
    warned = capsys.readouterr().err
    assert "policy lineage" in warned and "changed" in warned
    assert "rendering anyway" in warned

    config.write_text("{}\n", encoding="utf-8")
    lineage = rerender._capture_batch_policy_lineage(run_dir)
    (batch_root / "rollout_config.yaml").write_text("repeat: 30\n", encoding="utf-8")
    with pytest.raises(rerender.RerenderError, match="presence changed"):
        rerender._assert_batch_policy_lineage_current(lineage)

@pytest.mark.parametrize(
    ("authority_args", "is_strict"),
    (
        ("--official-scene-tag scene", False),
        ("--strict-assets", True),
        ("--official-run-manifest /runs/run_manifest.json", True),
        ("--official-worker-contract /runs/worker_contract.json", True),
    ),
)
def test_prepare_source_run_rejects_diagnostic_strict_authority(
    run_dir: Path,
    authority_args: str,
    is_strict: bool,
) -> None:
    command_path = run_dir / "RUN_COMMAND.md"
    command_path.write_text(
        command_path.read_text(encoding="utf-8").replace(
            "--no-render", f"--no-render {authority_args}"
        ),
        encoding="utf-8",
    )

    if is_strict:
        with pytest.raises(
            rerender.RerenderError,
            match="diagnostic source has inconsistent official authority",
        ):
            rerender.prepare_source_run(
                run_dir,
                renderer_code={"commit": "a" * 40, "dirty": True},
            )
    else:
        source = rerender.prepare_source_run(
            run_dir,
            renderer_code={"commit": "a" * 40, "dirty": True},
        )
        assert source.summary["execution_class"] == "diagnostic"
        assert source.official is False
        assert source.official_authority is None

@pytest.mark.parametrize("execution_class", (None, "unknown", True, [], {}))
def test_prepare_source_run_rejects_invalid_execution_class(
    run_dir: Path,
    execution_class: object,
) -> None:
    summary_path = run_dir / "sim_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if execution_class is None:
        summary.pop("execution_class")
    else:
        summary["execution_class"] = execution_class
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(rerender.RerenderError, match="execution_class must be"):
        rerender.prepare_source_run(
            run_dir,
            renderer_code={"commit": "a" * 40, "dirty": True},
        )

@pytest.mark.parametrize(
    ("failure", "match"),
    (
        ("summary_frames", "summary frames=2"),
        ("wall_stopped", "wall_stopped_early"),
        ("wall_unattested", "wall_stopped_early"),
        ("truncated_log", "has 2 rows, expected 3"),
        ("nonsequential_log", "frame=2, expected 1"),
    ),
)
def test_prepare_source_run_rejects_incomplete_or_inconsistent_generation(
    run_dir: Path,
    failure: str,
    match: str,
) -> None:
    summary_path = run_dir / "sim_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if failure == "summary_frames":
        summary["frames"] = 2
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    elif failure in {"wall_stopped", "wall_unattested"}:
        if failure == "wall_stopped":
            summary["wall_stopped_early"] = True
        else:
            summary.pop("wall_stopped_early")
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    else:
        _configure_complete_source(run_dir, 3)
        rows = (
            '{"frame": 0}\n{"frame": 1}\n'
            if failure == "truncated_log"
            else '{"frame": 0}\n{"frame": 2}\n{"frame": 1}\n'
        )
        (run_dir / "logs" / "sim_log.jsonl").write_text(
            rows, encoding="utf-8"
        )

    with pytest.raises(rerender.RerenderError, match=match):
        rerender.prepare_source_run(
            run_dir,
            renderer_code={"commit": "a" * 40, "dirty": True},
        )

@pytest.mark.parametrize("module_head", (False, True), ids=("file-head", "module-head"))
def test_parse_run_command_splits_env_and_argv(run_dir: Path, module_head: bool) -> None:
    if module_head:
        use_module_worker_head(run_dir)
    environment, argv = rerender.parse_run_command(run_dir)
    assert environment["MUJOCO_GL"] == "egl"
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    head = (MODULE_WORKER_HEAD if module_head else LEGACY_WORKER_HEAD).split()
    assert argv[: len(head)] == head
    assert worker_command_head_length(argv) == len(head)
    assert argv[len(head)] == "--robot"


def test_parse_run_command_refuses_another_command_head(run_dir: Path) -> None:
    command_path = run_dir / "RUN_COMMAND.md"
    command_path.write_text(
        command_path.read_text(encoding="utf-8").replace(
            LEGACY_WORKER_HEAD, "/venv/bin/python -m robosuite_env.other"
        ),
        encoding="utf-8",
    )
    with pytest.raises(rerender.RerenderError, match="unexpected command head"):
        rerender.parse_run_command(run_dir)

def _configure_official_source(run: Path) -> tuple[Path, Path, list[str]]:
    commit = "c" * 40
    manifest_path = run.parents[1] / "run_manifest.json"
    worker_path = run / "official_worker_contract.json"
    scene_manifest = run.parents[1] / "scene_source" / "background.json"
    command_path = run / "RUN_COMMAND.md"
    command = command_path.read_text(encoding="utf-8")
    command = command.replace(
        "--no-render",
        "--no-render --strict-assets "
        f"--official-run-manifest {manifest_path} "
        "--official-scene-tag scene",
    )
    command = command.replace(
        "\n```", f" --official-worker-contract {worker_path}\n```"
    )
    command_path.write_text(command, encoding="utf-8")
    _, argv = rerender.parse_run_command(run)
    contract_argv, removed = rerender._without_option(
        argv[worker_command_head_length(argv):], "--official-worker-contract"
    )
    assert removed == 1
    manifest = {
        "code": {
            "commit": commit,
            "dirty": False,
            "path": str(Path(rerender.REPO_ROOT).resolve()),
        },
        "source_dependencies": [gaussian_wrapping(Path(rerender.REPO_ROOT).resolve())],
        "preflight": {
            "contracts": {
                "scenes": {
                    "scene": {
                        "id": "scene-id",
                        "manifest_path": str(scene_manifest),
                        "contract_digest": "d" * 64,
                        "ply_sha256": "e" * 64,
                    }
                }
            }
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    worker_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_manifest_sha256": rerender._sha256_file(manifest_path),
                "argv": contract_argv,
            }
        ),
        encoding="utf-8",
    )
    summary_path = run / "sim_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "execution_class": "official",
            "official_run_manifest": str(manifest_path),
            "official_worker_contract": str(worker_path),
            "scene_contract": {
                "tag": "scene",
                "id": "scene-id",
                "sha256": "d" * 64,
                "manifest": str(scene_manifest),
            },
        }
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return manifest_path, worker_path, contract_argv

@pytest.mark.parametrize("module_head", (False, True), ids=("file-head", "module-head"))
def test_prepare_source_run_binds_official_launcher_authority(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, module_head: bool
) -> None:
    if module_head:
        use_module_worker_head(run_dir)
    manifest_path, worker_path, contract_argv = _configure_official_source(run_dir)
    assert contract_argv[:2] == ["--robot", "FR3"]
    calls = []

    def fake_validate(contract, manifest, argv):
        calls.append((Path(contract), Path(manifest), list(argv)))
        return {"argv": list(argv)}

    monkeypatch.setattr(rerender_sources, "validate_worker_contract", fake_validate)

    expanded_renderer_identity = {
        "commit": "c" * 40,
        "dirty": False,
        "core_source_files": {"example.py": {"sha256": "a" * 64}},
        "tracked_diff": {
            "base_commit": "c" * 40,
            "sha256": "b" * 64,
            "size_bytes": 0,
        },
        "gaussian_wrapping": gaussian_wrapping(Path(rerender.REPO_ROOT).resolve()),
    }
    source = rerender.prepare_source_run(
        run_dir,
        renderer_code=expanded_renderer_identity,
    )

    assert source.official is True
    assert source.source_records["code_commit"] == "c" * 40
    assert set(source.source_records["run_manifest"]) == {"path", "sha256"}
    assert set(source.source_records["worker_contract"]) == {"path", "sha256"}
    assert source.renderer_code == expanded_renderer_identity
    assert calls == [(worker_path.resolve(), manifest_path.resolve(), contract_argv)]

def test_git_code_identity_binds_exact_core_sources_and_tracked_diff() -> None:
    identity = rerender._git_code_identity()

    assert rerender._COMMIT_RE.fullmatch(identity["commit"])
    assert isinstance(identity["dirty"], bool)
    assert set(identity["core_source_files"]) == set(
        rerender._CORE_RENDERER_SOURCE_PATHS
    )
    for relative_path, record in identity["core_source_files"].items():
        assert record["path"] == str((rerender.REPO_ROOT / relative_path).resolve())
        assert rerender._SHA256_RE.fullmatch(record["sha256"])
        assert "state" in record
    assert identity["tracked_diff"]["base_commit"] == identity["commit"]
    assert rerender._SHA256_RE.fullmatch(identity["tracked_diff"]["sha256"])
    assert identity["tracked_diff"]["size_bytes"] >= 0
    rerender._assert_renderer_code_current(identity)

def test_renderer_code_recheck_warns_on_core_hash_or_tracked_diff_drift(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Drift is reported, not refused; a malformed identity -- one
    # the check cannot be made against at all -- still raises.
    identity = rerender._git_code_identity()
    bad_hash_identity = json.loads(json.dumps(identity))
    first_path = rerender._CORE_RENDERER_SOURCE_PATHS[0]
    bad_hash_identity["core_source_files"][first_path]["sha256"] = "0" * 64
    rerender._assert_renderer_code_current(bad_hash_identity)
    warned = capsys.readouterr().err
    assert "renderer source" in warned and "changed" in warned

    observed_diff = dict(identity["tracked_diff"])
    monkeypatch.setattr(
        rerender_provenance,
        "_git_tracked_diff_identity",
        lambda _commit: {**observed_diff, "sha256": "f" * 64},
    )
    rerender._assert_renderer_code_current(identity)
    assert "tracked source diff changed" in capsys.readouterr().err

    no_diff_identity = json.loads(json.dumps(identity))
    no_diff_identity.pop("tracked_diff")
    with pytest.raises(rerender.RerenderError, match="no tracked source diff"):
        rerender._assert_renderer_code_current(no_diff_identity)

def test_provenance_primitives_are_shared_with_launcher() -> None:
    """Renderer and launcher must consume the SAME canonical objects.

    Both import the shared module in robosuite_env.contracts.provenance
    rather than keeping copies that could drift, so this guard asserts object
    identity rather than mere equality.
    """

    from real2sim import launcher
    from robosuite_env.contracts import provenance as provenance_contract

    canonical_paths = provenance_contract.CORE_RENDERER_SOURCE_PATHS
    assert "robosuite_env/rendering/fisheye_model.py" in canonical_paths
    assert rerender._CORE_RENDERER_SOURCE_PATHS is canonical_paths
    assert launcher._CORE_RENDERER_SOURCE_PATHS is canonical_paths

    canonical_lineage = provenance_contract.BATCH_POLICY_LINEAGE_FILENAMES
    assert rerender._BATCH_POLICY_LINEAGE_FILENAMES is canonical_lineage
    assert launcher._BATCH_POLICY_LINEAGE_FILENAMES is canonical_lineage

    assert rerender._file_state is provenance_contract.file_state
    assert launcher._file_state is provenance_contract.file_state
    assert rerender._sha256_file is provenance_contract.sha256_file
    assert launcher._file_sha256 is provenance_contract.sha256_file
    assert (
        launcher._tracked_diff_identity
        is provenance_contract.git_tracked_diff_identity
    )

def test_prepare_source_run_rejects_dirty_official_checkout(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_official_source(run_dir)
    monkeypatch.setattr(rerender_sources, "validate_worker_contract", lambda *_: {})

    with pytest.raises(rerender.RerenderError, match="requires clean checkout"):
        rerender.prepare_source_run(
            run_dir,
            renderer_code={"commit": "c" * 40, "dirty": True},
        )

def _seal_core_sources(run: Path, repo_root: Path, *, tamper: str | None = None) -> None:
    """Write the run's runtime source seal the launcher captures for this checkout.

    ``tamper`` names one sealed file whose recorded hash no longer matches.
    """

    from real2sim import runtime_manifest

    capture = runtime_manifest.capture_selection(
        runtime_manifest.default_selection(repo_root)
    )
    if tamper is not None:
        tampered = [
            record
            for record in capture["files"]
            if record["path"] == str(Path(repo_root).resolve() / tamper)
        ]
        assert len(tampered) == 1
        tampered[0]["sha256"] = "f" * 64
    manifest_path = run.parents[1] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["code"]["path"] = str(Path(repo_root).resolve())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (run.parents[1] / "runtime_source_manifest.json").write_text(
        json.dumps(capture), encoding="utf-8"
    )


def test_prepare_source_run_accepts_moved_checkout_with_sealed_core_bytes(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Later commits are fine while every core renderer file is unchanged."""

    _configure_official_source(run_dir)
    repo_root = Path(rerender.REPO_ROOT)
    _seal_core_sources(run_dir, repo_root)
    # The worker contract binds the manifest bytes, which _seal_core_sources
    # rewrote, so re-point it at the manifest it now has to match.
    worker_path = run_dir / "official_worker_contract.json"
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    worker["run_manifest_sha256"] = rerender._sha256_file(
        run_dir.parents[1] / "run_manifest.json"
    )
    worker_path.write_text(json.dumps(worker), encoding="utf-8")
    monkeypatch.setattr(rerender_sources, "validate_worker_contract", lambda *_: {})

    renderer_code = {
        "commit": "a" * 40,
        "dirty": True,
        "core_source_files": {
            relative_path: {
                "path": str((repo_root / relative_path).resolve()),
                "sha256": rerender._sha256_file(repo_root / relative_path),
            }
            for relative_path in rerender._CORE_RENDERER_SOURCE_PATHS
        },
        "gaussian_wrapping": gaussian_wrapping(repo_root.resolve()),
    }
    # Reaching past the checkout rule is the assertion: any later failure is
    # about this fixture's other authorities, not the commit comparison.
    try:
        rerender.prepare_source_run(run_dir, renderer_code=renderer_code)
    except rerender.RerenderError as exc:
        assert "requires clean checkout" not in str(exc)


def test_launcher_runtime_seal_covers_every_core_renderer_source(
    tmp_path: Path,
) -> None:
    """The seal the launcher really writes must let an unchanged renderer through.

    The fixture above copies CORE_RENDERER_SOURCE_PATHS into its seal, so it
    cannot notice a core renderer file the launcher never hashes; this one
    captures the launcher's own default runtime selection instead.
    """

    from real2sim import runtime_manifest

    repo_root = Path(rerender.REPO_ROOT).resolve()
    capture = runtime_manifest.capture_selection(
        runtime_manifest.default_selection(repo_root)
    )
    (tmp_path / provenance_contract.RUNTIME_SOURCE_MANIFEST_FILENAME).write_text(
        json.dumps(capture), encoding="utf-8"
    )
    assert (
        provenance_contract.RUNTIME_SOURCE_MANIFEST_FILENAME
        == runtime_manifest.MANIFEST_FILE
    )

    record = provenance_contract.official_generation_check(
        tmp_path,
        run_manifest({"commit": "b" * 40, "dirty": False, "path": str(repo_root)}),
        {
            "commit": "a" * 40,
            "dirty": True,
            "core_source_files": rerender._core_renderer_source_records(),
            "gaussian_wrapping": gaussian_wrapping(repo_root),
        },
        checkout_root=repo_root,
    )
    assert record is not None
    assert sorted(set(rerender._CORE_RENDERER_SOURCE_PATHS) - set(record["files"])) == []


def test_prepare_source_run_rejects_moved_checkout_with_changed_core_bytes(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A core renderer file that differs from the seal still fails closed."""

    _configure_official_source(run_dir)
    repo_root = Path(rerender.REPO_ROOT)
    tampered = rerender._CORE_RENDERER_SOURCE_PATHS[0]
    _seal_core_sources(run_dir, repo_root, tamper=tampered)
    monkeypatch.setattr(rerender_sources, "validate_worker_contract", lambda *_: {})

    renderer_code = {
        "commit": "a" * 40,
        "dirty": True,
        "core_source_files": {
            relative_path: {
                "path": str((repo_root / relative_path).resolve()),
                "sha256": rerender._sha256_file(repo_root / relative_path),
            }
            for relative_path in rerender._CORE_RENDERER_SOURCE_PATHS
        },
    }
    with pytest.raises(rerender.RerenderError, match="requires clean checkout"):
        rerender.prepare_source_run(run_dir, renderer_code=renderer_code)


@pytest.mark.parametrize(
    "non_core",
    [
        "robosuite_env/rendering/mujoco_kb4.py",
        "robosuite_env/assets/box.xml",
        "robosuite_env/assets/yubi_right/visual_mj/dyn_l_finger_r__geometry0.obj",
        "config/support-matrix.yaml",
        "robosuite_env/tools/__init__.py",
        "uv.lock",
    ],
)
def test_prepare_source_run_rejects_moved_checkout_with_changed_non_core_bytes(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, non_core: str
) -> None:
    """The replay worker runs and reads more than the core renderer files.

    A moved-on checkout whose MuJoCo scene renderer, object model, gripper
    mesh, support matrix, tools package or dependency lock differs from what
    the run sealed is refused even though every core renderer file still
    matches. The last three are the repository files outside the sealed trees
    that a traced same-commit rerender read.
    """

    assert non_core not in rerender._CORE_RENDERER_SOURCE_PATHS
    _configure_official_source(run_dir)
    repo_root = Path(rerender.REPO_ROOT)
    _seal_core_sources(run_dir, repo_root, tamper=non_core)
    monkeypatch.setattr(rerender_sources, "validate_worker_contract", lambda *_: {})

    renderer_code = {
        "commit": "a" * 40,
        "dirty": True,
        "core_source_files": {
            relative_path: {
                "path": str((repo_root / relative_path).resolve()),
                "sha256": rerender._sha256_file(repo_root / relative_path),
            }
            for relative_path in rerender._CORE_RENDERER_SOURCE_PATHS
        },
    }
    with pytest.raises(rerender.RerenderError, match="requires clean checkout"):
        rerender.prepare_source_run(run_dir, renderer_code=renderer_code)


def test_prepare_source_run_rejects_run_command_worker_contract_drift(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, worker_path, _ = _configure_official_source(run_dir)
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    worker["argv"] = worker["argv"][:-2]
    worker_path.write_text(json.dumps(worker), encoding="utf-8")
    monkeypatch.setattr(
        rerender_sources,
        "validate_worker_contract",
        lambda *_: (_ for _ in ()).throw(AssertionError("must fail before validation")),
    )

    with pytest.raises(rerender.RerenderError, match="RUN_COMMAND differs"):
        rerender.prepare_source_run(
            run_dir,
            renderer_code={"commit": "c" * 40, "dirty": False},
        )

def test_scene_ply_resolves_relative_path_from_scene_directory(
    tmp_path: Path,
) -> None:
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    ply = scene_dir / "point_cloud.ply"
    ply.write_bytes(b"ply\n")
    (scene_dir / "background.json").write_text(
        json.dumps({"ply_path": "point_cloud.ply"}),
        encoding="utf-8",
    )

    assert rerender._scene_ply(scene_dir) == ply.resolve()

def test_scene_validation_reuses_authenticated_server_ply_digest(
    run_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = rerender.prepare_source_run(
        run_dir,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )
    scene_dir = tmp_path / "scene_source"
    scene_dir.mkdir()
    ply = scene_dir / "point_cloud.ply"
    ply.write_bytes(b"large-ply-placeholder")
    manifest = scene_dir / "background.json"
    manifest.write_text(
        json.dumps({"schema_version": 2, "ply_path": ply.name}),
        encoding="utf-8",
    )
    manifest_path, manifest_record, ply_path, ply_state = (
        rerender._capture_scene_inputs(scene_dir)
    )
    server_digest = "f" * 64
    calls = []

    def fake_validate(path, *, strict, preverified_ply_sha256):
        calls.append((Path(path), strict, preverified_ply_sha256))
        registration = source.source_records["gs_registration"]
        return {
            "id": "scene-id",
            "manifest_path": manifest_path,
            "ply_path": ply_path,
            "ply_sha256": server_digest,
            "contract_digest": "d" * 64,
            "w2gs_path": Path(registration["path"]),
            "w2gs_matrix": np.asarray(registration["raw_matrix"]),
        }

    monkeypatch.setattr(rerender_gs_server, "validate_scene_contract", fake_validate)

    identity = rerender._validated_scene_identity(
        scene_tag="scene",
        scene_dir=scene_dir,
        sources=[source],
        health={
            "status": "ok",
            "ready_token": "token",
            "ply_path": str(ply.resolve()),
            "ply_sha256": server_digest,
        },
        manifest_path=manifest_path,
        manifest_record=manifest_record,
        ply_path=ply_path,
        ply_state=ply_state,
    )

    assert calls == [(scene_dir, False, server_digest)]
    assert identity["ply"] == {
        "path": str(ply.resolve()),
        "sha256": server_digest,
        "state": ply_state,
    }
    assert identity["w2gs"]["file"] == source.source_records["gs_registration"][
        "file"
    ]

def test_scene_validation_rejects_different_w2gs_path_even_with_same_matrix(
    run_dir: Path,
    tmp_path: Path,
) -> None:
    source = rerender.prepare_source_run(
        run_dir,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )
    scene_dir = tmp_path / "different_registration_scene"
    scene_dir.mkdir()
    ply = scene_dir / "point_cloud.ply"
    ply.write_bytes(b"ply\n")
    current_w2gs = scene_dir / "mujoco_to_gs.npy"
    np.save(current_w2gs, np.eye(4, dtype=np.float64))
    (scene_dir / "background.json").write_text(
        json.dumps(
            {
                "name": "scene",
                "ply_path": ply.name,
                "w2gs": current_w2gs.name,
            }
        ),
        encoding="utf-8",
    )
    manifest_path, manifest_record, ply_path, ply_state = (
        rerender._capture_scene_inputs(scene_dir)
    )

    with pytest.raises(
        rerender.RerenderError,
        match="registration path or SHA-256 differs from generation",
    ):
        rerender._validated_scene_identity(
            scene_tag="scene",
            scene_dir=scene_dir,
            sources=[source],
            health={
                "status": "ok",
                "ready_token": "token",
                "ply_path": str(ply.resolve()),
                "ply_sha256": rerender._sha256_file(ply),
            },
            manifest_path=manifest_path,
            manifest_record=manifest_record,
            ply_path=ply_path,
            ply_state=ply_state,
        )

def _write_file_form_scene_with_env_collision(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    matrix: np.ndarray,
) -> tuple[Path, Path]:
    """Write a schema-v2 scene using file-form w2gs and an env-reference collision dir."""

    from robosuite_env.contracts.scene import compute_tree_sha256

    bundle = root / "desk_bundle"
    collision = bundle / "coacd"
    collision.mkdir(parents=True)
    for index in range(2):
        (collision / f"part_{index:03d}.obj").write_text(
            f"v {index} 0 0\nv {index} 1 0\nv {index} 0 1\nf 1 2 3\n",
            encoding="utf-8",
        )
    monkeypatch.setenv("RERENDER_TEST_DESK_BUNDLE", str(bundle))

    scene_dir = root / "file_form_scene"
    scene_dir.mkdir()
    ply = scene_dir / "point_cloud.ply"
    ply.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
    w2gs = scene_dir / "gs_w2gs.npy"
    np.save(w2gs, np.asarray(matrix, dtype=np.float64))
    (scene_dir / "background.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "id": "file-form-scene-0123456789ab",
                "name": "file_form_scene",
                "ply_path": ply.name,
                "ply_sha256": rerender._sha256_file(ply),
                "w2gs": w2gs.name,
                "w2gs_sha256": rerender._sha256_file(w2gs),
                # Header-only fixture PLY: below TABLE_Z_MIN_VERTICES, so the
                # declared table height is accepted without measurement.
                "calibration_table_z_m": 0.0,
                "calibration_table_z_method": "ply-plane-fit-v1",
                "table": {
                    "appearance": "embedded_in_3dgs",
                    "collision": {
                        "directory": "${RERENDER_TEST_DESK_BUNDLE}/coacd",
                        "glob": "*.obj",
                        "count": 2,
                        "tree_sha256": compute_tree_sha256(
                            collision, sorted(collision.glob("*.obj"))
                        ),
                        "frame": "mujoco_world",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return scene_dir, w2gs

def _scene_identity_for(run: Path, scene_dir: Path) -> dict:
    source = rerender.prepare_source_run(
        run,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )
    manifest_path, manifest_record, ply_path, ply_state = (
        rerender._capture_scene_inputs(scene_dir)
    )
    return rerender._validated_scene_identity(
        scene_tag="file_form_scene",
        scene_dir=scene_dir,
        sources=[source],
        health={
            "status": "ok",
            "ready_token": "token",
            "ply_path": str(ply_path.resolve()),
            "ply_sha256": rerender._sha256_file(ply_path),
        },
        manifest_path=manifest_path,
        manifest_record=manifest_record,
        ply_path=ply_path,
        ply_state=ply_state,
    )

def test_scene_validation_accepts_file_form_w2gs_with_env_collision_dir(
    run_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Uniform scale 3.5 times a 90-degree rotation about Z plus a translation:
    # a valid similarity transform, unlike the identity used elsewhere.
    matrix = np.array(
        [
            [0.0, -3.5, 0.0, 0.93],
            [3.5, 0.0, 0.0, 3.72],
            [0.0, 0.0, 3.5, 0.21],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    scene_dir, w2gs = _write_file_form_scene_with_env_collision(
        tmp_path, monkeypatch, matrix
    )
    # Generation record as the official preflight + summary now produce it for a
    # file-form contract: the registered path, not the inline sentinel.
    summary_path = run_dir / "sim_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "gs_w2gs": str(w2gs.resolve()),
            "gs_w2gs_raw": matrix.tolist(),
            "gs_w2gs_effective": matrix.tolist(),
        }
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    identity = _scene_identity_for(run_dir, scene_dir)

    assert identity["w2gs"]["kind"] == "file"
    assert identity["w2gs"]["path"] == str(w2gs.resolve())
    assert identity["w2gs"]["file"]["sha256"] == rerender._sha256_file(w2gs)
    assert identity["w2gs"]["raw_matrix"] == matrix.tolist()

    # A generation record that collapses the same scene into "inline-matrix"
    # must still be rejected because it lost the file identity.
    summary["gs_w2gs"] = "inline-matrix"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(
        rerender.RerenderError, match="registration source differs from generation"
    ):
        _scene_identity_for(run_dir, scene_dir)

def test_scene_validation_rejects_changed_file_form_matrix(
    run_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = np.diag([1.0, 1.0, 1.0, 1.0])
    scene_dir, w2gs = _write_file_form_scene_with_env_collision(
        tmp_path, monkeypatch, matrix
    )
    summary_path = run_dir / "sim_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "gs_w2gs": str(w2gs.resolve()),
            "gs_w2gs_raw": matrix.tolist(),
            "gs_w2gs_effective": matrix.tolist(),
        }
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    source = rerender.prepare_source_run(
        run_dir,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )
    source_registration = source.source_records["gs_registration"]
    assert source_registration["kind"] == "file"

    # Re-register the scene with a different matrix, keeping the contract valid.
    changed = matrix.copy()
    changed[0, 3] = 0.5
    np.save(w2gs, changed)
    manifest_path = scene_dir / "background.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["w2gs_sha256"] = rerender._sha256_file(w2gs)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    scene_manifest_path, manifest_record, ply_path, ply_state = (
        rerender._capture_scene_inputs(scene_dir)
    )

    with pytest.raises(rerender.RerenderError, match="raw GS matrix differs"):
        rerender._validated_scene_identity(
            scene_tag="file_form_scene",
            scene_dir=scene_dir,
            sources=[source],
            health={
                "status": "ok",
                "ready_token": "token",
                "ply_path": str(ply_path.resolve()),
                "ply_sha256": rerender._sha256_file(ply_path),
            },
            manifest_path=scene_manifest_path,
            manifest_record=manifest_record,
            ply_path=ply_path,
            ply_state=ply_state,
        )


def test_renderer_code_recheck_reports_a_moved_gaussian_wrapping_checkout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    identity = rerender._git_code_identity()
    assert identity["gaussian_wrapping"] == provenance_contract.gaussian_wrapping_identity(
        rerender.REPO_ROOT
    )
    assert rerender_provenance._assert_renderer_code_current(identity) is True

    moved = {**identity["gaussian_wrapping"], "commit": "f" * 40, "dirty": False}
    monkeypatch.setattr(
        rerender_provenance, "gaussian_wrapping_identity", lambda _root: moved
    )

    assert rerender_provenance._assert_renderer_code_current(identity) is False
    assert "GaussianWrapping checkout changed" in capsys.readouterr().err
