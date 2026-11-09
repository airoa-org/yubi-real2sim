# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Runs recorded before the condition inputs moved to assets/release/ still replay.

A recorded worker argv names the per-condition base placement and gripper
calibration by absolute checkout path. The previous directory name is matched
by digest only, so these tests stand in a synthetic name for it. The moved
file replaces the recorded one only when its bytes hash to the digest the
run's runtime source seal recorded for the recorded path.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import robosuite_env.rerender.replay_argv as replay_argv
import robosuite_env.tools.rerender_third_person as rerender
from real2sim.runtime_manifest import capture_selection, write_manifest
from robosuite_env.contracts.provenance import RUNTIME_SOURCE_MANIFEST_FILENAME
from robosuite_env.rerender.common import REPO_ROOT, RerenderError
from tests._helpers.rollouts import (  # noqa: F401 -- run_dir is a pytest fixture
    _configure_complete_source,
    run_dir,
)


STAND_IN = "pre-move"
RUN_COMMIT = "c0ffee" + "0" * 34
CONDITION = Path("fr3") / "flat" / "cup"


@pytest.fixture
def stand_in_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        replay_argv,
        "_PREVIOUS_CONDITION_ASSET_DIR_SHA256",
        hashlib.sha256(STAND_IN.encode()).hexdigest(),
    )


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    root = (tmp_path / "checkout").resolve()
    root.mkdir()
    return root


def _run(tmp_path: Path, *, commit: str | None = RUN_COMMIT) -> Path:
    """A run directory two levels below its run root, with run_manifest.json."""

    run = tmp_path / "batch" / "scene" / "run_01"
    run.mkdir(parents=True)
    if commit is not None:
        (run.parents[1] / "run_manifest.json").write_text(
            json.dumps({"code": {"commit": commit, "dirty": False}}),
            encoding="utf-8",
        )
    return run


def _seal(run_dir: Path, checkout: Path) -> None:
    """Seal the checkout's assets/ tree as the launcher does, at the run root."""

    selection = {
        "repo_root": str(checkout),
        "files": [],
        "external_files": [],
        "trees": [{"path": "assets", "suffixes": ["*"]}],
    }
    write_manifest(
        run_dir.parents[1] / RUNTIME_SOURCE_MANIFEST_FILENAME,
        capture_selection(selection),
    )


def _recorded_then_moved(checkout: Path, run_dir: Path | None) -> tuple[Path, Path]:
    """Record the pre-move file, seal the run (if any), then move the file."""

    recorded = checkout / "assets" / STAND_IN / CONDITION / "base_placement.json"
    recorded.parent.mkdir(parents=True)
    recorded.write_text('{"x": 0.5}\n', encoding="utf-8")
    if run_dir is not None:
        _seal(run_dir, checkout)
    moved = checkout / "assets" / "release" / CONDITION / "base_placement.json"
    moved.parent.mkdir(parents=True)
    shutil.move(recorded, moved)
    return recorded, moved


def _condition_file(root: Path) -> Path:
    moved = root / "assets" / "release" / "fr3" / "duo" / "cup" / "base_placement.json"
    moved.parent.mkdir(parents=True)
    moved.write_text("{}\n", encoding="utf-8")
    return moved


def test_a_pre_move_path_maps_to_the_moved_file(tmp_path: Path, stand_in_digest) -> None:
    moved = _condition_file(tmp_path)
    recorded = tmp_path / "assets" / STAND_IN / "fr3" / "duo" / "cup" / "base_placement.json"

    assert replay_argv._moved_condition_asset(str(recorded), tmp_path) == str(moved)


def test_the_sealed_bytes_replace_a_pre_move_path(
    tmp_path: Path, checkout: Path, stand_in_digest
) -> None:
    run = _run(tmp_path)
    recorded, moved = _recorded_then_moved(checkout, run)

    assert replay_argv._with_moved_condition_assets(
        ["sim.py", "--base-placement", str(recorded), f"--gripper-calibration={recorded}"],
        run,
        checkout,
    ) == ["sim.py", "--base-placement", str(moved), f"--gripper-calibration={moved}"]


def test_a_moved_file_edited_after_the_run_is_refused(
    tmp_path: Path, checkout: Path, stand_in_digest
) -> None:
    run = _run(tmp_path)
    recorded, moved = _recorded_then_moved(checkout, run)
    moved.write_text('{"x": 0.6}\n', encoding="utf-8")

    for argv in (
        ["sim.py", "--base-placement", str(recorded)],
        ["sim.py", f"--base-placement={recorded}"],
    ):
        with pytest.raises(
            RerenderError,
            match=rf"has SHA-256 {hashlib.sha256(moved.read_bytes()).hexdigest()} "
            rf"and the run sealed [0-9a-f]{{64}}.*run's own commit {RUN_COMMIT}",
        ):
            replay_argv._with_moved_condition_assets(argv, run, checkout)


def test_a_path_the_seal_does_not_record_is_refused(
    tmp_path: Path, checkout: Path, stand_in_digest
) -> None:
    run = _run(tmp_path)
    # The seal predates the condition file, so it holds a digest for another
    # asset only.
    (checkout / "assets").mkdir()
    (checkout / "assets" / "other.json").write_text("{}\n", encoding="utf-8")
    _seal(run, checkout)
    recorded, _ = _recorded_then_moved(checkout, None)

    with pytest.raises(
        RerenderError,
        match=rf"seal records no SHA-256 for {recorded}.*run's own commit {RUN_COMMIT}",
    ):
        replay_argv._with_moved_condition_assets(
            ["sim.py", "--base-placement", str(recorded)], run, checkout
        )


def test_a_run_without_a_seal_or_commit_is_refused(
    tmp_path: Path, checkout: Path, stand_in_digest
) -> None:
    run = _run(tmp_path, commit=None)
    recorded, _ = _recorded_then_moved(checkout, None)

    with pytest.raises(
        RerenderError,
        match=r"seal records no SHA-256.*run_manifest\.json records no commit",
    ):
        replay_argv._with_moved_condition_assets(
            ["sim.py", "--base-placement", str(recorded)], run, checkout
        )


@pytest.mark.parametrize(
    "recorded",
    [
        # Not the previous directory.
        "assets/other/fr3/duo/cup/base_placement.json",
        # The previous directory, but the moved file does not exist.
        f"assets/{STAND_IN}/fr3/duo/box_pen/base_placement.json",
        # Too short to name a condition file.
        f"assets/{STAND_IN}",
    ],
)
def test_other_missing_paths_stay_as_recorded(
    tmp_path: Path, stand_in_digest, recorded: str
) -> None:
    _condition_file(tmp_path)
    value = str(tmp_path / recorded)
    assert replay_argv._moved_condition_asset(value, tmp_path) == value
    # Nothing to substitute, so no seal is needed.
    run = _run(tmp_path / "runs", commit=None)
    assert replay_argv._with_moved_condition_assets(["sim.py", value], run, tmp_path) == [
        "sim.py",
        value,
    ]


def test_existing_relative_and_foreign_paths_stay_as_recorded(
    tmp_path: Path, stand_in_digest
) -> None:
    _condition_file(tmp_path)
    present = tmp_path / "assets" / STAND_IN / "fr3" / "duo" / "cup" / "base_placement.json"
    present.parent.mkdir(parents=True)
    present.write_text("{}\n", encoding="utf-8")
    foreign = tmp_path.parent / "elsewhere" / "assets" / STAND_IN / "fr3" / "duo" / "cup" / "base_placement.json"
    for value in (
        str(present),
        f"assets/{STAND_IN}/fr3/duo/cup/base_placement.json",
        str(foreign),
    ):
        assert replay_argv._moved_condition_asset(value, tmp_path) == value


def _write_seal_records(run_dir: Path, digests: dict[Path, str]) -> None:
    """A seal at the run root recording ``digests`` for paths in this checkout.

    The checkout's previous directory is a stand-in that never existed, so its
    records are written directly in the launcher's record format.
    """

    state = {
        "device": 0,
        "inode": 0,
        "mode": 0o100644,
        "size": 0,
        "mtime_ns": 0,
        "ctime_ns": 0,
    }
    (run_dir.parents[1] / RUNTIME_SOURCE_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {"path": str(path), "sha256": digest, "state": state}
                    for path, digest in sorted(digests.items())
                ],
            }
        ),
        encoding="utf-8",
    )


def _pre_move_argv(run_dir: Path) -> tuple[list[str], Path, Path, Path, Path]:
    placement = REPO_ROOT / "assets" / "release" / CONDITION / "base_placement.json"
    calibration = REPO_ROOT / "assets" / "release" / CONDITION / "gripper_calib_rad.json"
    assert placement.is_file() and calibration.is_file()
    previous = REPO_ROOT / "assets" / STAND_IN / CONDITION
    _, recorded = rerender.parse_run_command(run_dir)
    argv = [
        *recorded,
        "--base-placement",
        str(previous / "base_placement.json"),
        f"--gripper-calibration={previous / 'gripper_calib_rad.json'}",
    ]
    return (
        argv,
        previous / "base_placement.json",
        placement,
        previous / "gripper_calib_rad.json",
        calibration,
    )


def _build(argv: list[str], run_dir: Path) -> list[str]:
    return rerender.build_replay_argv(
        argv,
        run_dir=run_dir,
        gs_port=9999,
        gs_token="feedface",
        render_lock="/tmp/rerender.lock",
        wrist_insets="off",
    )


def test_build_replay_argv_points_a_pre_move_run_at_the_sealed_release_inputs(
    run_dir: Path, stand_in_digest, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The checked-in condition files are the targets; only the recorded
    # directory name is the stand-in. The run sealed their current bytes.
    argv, old_placement, placement, old_calibration, calibration = _pre_move_argv(run_dir)
    _write_seal_records(
        run_dir,
        {
            old_placement: hashlib.sha256(placement.read_bytes()).hexdigest(),
            old_calibration: hashlib.sha256(calibration.read_bytes()).hexdigest(),
        },
    )
    hashed: list[Path] = []
    original_sha256 = replay_argv.sha256_file

    def recording_sha256(path: Path) -> str:
        hashed.append(path)
        return original_sha256(path)

    monkeypatch.setattr(replay_argv, "sha256_file", recording_sha256)

    replay = _build(argv, run_dir)

    # The byte comparison ran for both recorded files before substitution.
    assert hashed == [placement, calibration]
    assert replay[replay.index("--base-placement") + 1] == str(placement)
    assert f"--gripper-calibration={calibration}" in replay
    assert not any(STAND_IN in token for token in replay)


def test_build_replay_argv_refuses_release_inputs_that_differ_from_the_seal(
    run_dir: Path, stand_in_digest
) -> None:
    argv, old_placement, placement, old_calibration, calibration = _pre_move_argv(run_dir)
    (run_dir.parents[1] / "run_manifest.json").write_text(
        json.dumps({"code": {"commit": RUN_COMMIT}}), encoding="utf-8"
    )
    _write_seal_records(
        run_dir,
        {
            old_placement: hashlib.sha256(placement.read_bytes()).hexdigest(),
            # The run used other bytes than the release file now holds.
            old_calibration: "0" * 64,
        },
    )

    with pytest.raises(
        RerenderError,
        match=rf"{calibration} has SHA-256 [0-9a-f]{{64}} and the run sealed 0{{64}}.*"
        rf"run's own commit {RUN_COMMIT}",
    ):
        _build(argv, run_dir)
    assert not (run_dir / "rerender").exists()


def test_source_preflight_refuses_an_unsealed_pre_move_path(
    run_dir: Path, stand_in_digest
) -> None:
    # The same guard stops the run before any GS server or GPU starts.
    _configure_complete_source(run_dir, 1)
    previous = REPO_ROOT / "assets" / STAND_IN / CONDITION / "base_placement.json"
    command_path = run_dir / "RUN_COMMAND.md"
    command_path.write_text(
        command_path.read_text(encoding="utf-8").replace(
            "--robot FR3", f"--robot FR3 --base-placement {previous}"
        ),
        encoding="utf-8",
    )

    with pytest.raises(RerenderError, match=rf"seal records no SHA-256 for {previous}"):
        rerender.prepare_source_run(
            run_dir, renderer_code={"commit": "a" * 40, "dirty": True}
        )
    assert not (run_dir / "rerender").exists()


def test_the_previous_name_is_not_a_current_asset_directory() -> None:
    current = {
        hashlib.sha256(path.name.encode()).hexdigest()
        for path in (REPO_ROOT / "assets").iterdir()
    }
    assert replay_argv._PREVIOUS_CONDITION_ASSET_DIR_SHA256 not in current
    assert (REPO_ROOT / "assets" / "release").is_dir()
