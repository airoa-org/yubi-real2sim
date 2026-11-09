# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Retired hue provenance must never silently change recorded replay visuals."""

import json
import shlex

import pytest

from robosuite_env.rerender.common import RerenderError
from robosuite_env.rerender.sources import prepare_source_run
from robosuite_env.rollout.cli import parse
from tests._helpers.rollouts import replayed_worker_arguments, run_dir  # noqa: F401 -- pytest fixture
from tests.unit.rerender.test_retired_options import replay_command


HUE_OPTION = "--plate-texture-hue-shift-opencv-bins"
HUE_REPORT = "plate_visual_texture_hue_ablation"
SHIFT_FIELDS = (
    "requested_hue_shift_opencv_bins",
    "hue_shift_opencv_bins",
    "hue_shift_degrees",
)
HASH_FIELDS = (
    "source_sha256",
    "result_sha256",
    "source_rgb_sha256",
    "effective_rgb_sha256",
)


def no_hue_summary(pixel_hash="a" * 64):
    # Fields and aliases emitted by the original apply_plate_texture_hue_shift.
    return {
        HUE_REPORT: {
            **dict.fromkeys(SHIFT_FIELDS, 0),
            "enabled": False,
            "changed": False,
            **dict.fromkeys(HASH_FIELDS, pixel_hash),
        }
    }


@pytest.mark.parametrize("task", ["cup_plate", "box_pack"])
@pytest.mark.parametrize("equals", [False, True])
@pytest.mark.parametrize(
    "values", [[], ["0"], ["+0"], ["-0"], ["179", "0"], ["-179", "179", "0"]]
)
def test_retired_hue_last_zero_replays_through_worker_parser(
    tmp_path, task, equals, values
):
    options = []
    for value in values:
        options += [f"{HUE_OPTION}={value}"] if equals else [HUE_OPTION, value]
    # Both compatibility migrations compose in the production replay path.
    options += ["--success-quiet-translation-mm", "2"]
    parsed_argv = replay_command(tmp_path, options, task=task)
    parsed = parse(replayed_worker_arguments(parsed_argv))
    assert parsed.objects == task
    assert parsed.state_log_replay == str(tmp_path / "logs" / "sim_log.jsonl")
    assert not any(token.partition("=")[0] == HUE_OPTION for token in parsed_argv)
    assert not hasattr(parsed, "plate_texture_hue_shift_opencv_bins")


@pytest.mark.parametrize("equals", [False, True])
@pytest.mark.parametrize(
    "value", ["", "oops", "nan", "inf", "-inf", "0.0", "1e0", "-180", "180"]
)
def test_retired_hue_validates_every_occurrence(tmp_path, equals, value):
    invalid = [f"{HUE_OPTION}={value}"] if equals else [HUE_OPTION, value]
    with pytest.raises(RerenderError, match=r"integer in \[-179, 179\]"):
        replay_command(tmp_path, [*invalid, HUE_OPTION, "0"])
    assert not (tmp_path / "rerender").exists()


@pytest.mark.parametrize("tail", [[], ["--frames", "1"], [HUE_OPTION, "0"]])
def test_retired_hue_rejects_missing_values(tmp_path, tail):
    with pytest.raises(RerenderError, match="missing its value"):
        replay_command(tmp_path, [HUE_OPTION, *tail])
    assert not (tmp_path / "rerender").exists()


@pytest.mark.parametrize("value", ["-179", "-1", "1", "179"])
@pytest.mark.parametrize("equals", [False, True])
def test_retired_hue_rejects_effective_nonzero_after_zero(tmp_path, equals, value):
    effective = [f"{HUE_OPTION}={value}"] if equals else [HUE_OPTION, value]
    with pytest.raises(RerenderError, match="nonzero.*original renderer"):
        replay_command(
            tmp_path, [HUE_OPTION, "0", *effective], summary=no_hue_summary()
        )
    assert not (tmp_path / "rerender").exists()


@pytest.mark.parametrize("pixel_hash", [None, "a" * 64])
@pytest.mark.parametrize("options", [[], [HUE_OPTION, "0"]])
def test_retired_hue_accepts_consistent_noop_summary(tmp_path, pixel_hash, options):
    summary = no_hue_summary(pixel_hash)
    parsed = parse(replayed_worker_arguments(replay_command(tmp_path, options, summary=summary)))
    assert parsed.objects == "cup_plate"


@pytest.mark.parametrize("field", SHIFT_FIELDS)
def test_retired_hue_rejects_nonzero_summary_without_cli_option(tmp_path, field):
    summary = no_hue_summary()
    summary[HUE_REPORT][field] = 1
    with pytest.raises(RerenderError, match=f"nonzero {field}"):
        replay_command(tmp_path, [], summary=summary)
    assert not (tmp_path / "rerender").exists()


@pytest.mark.parametrize("report", [None, False, [], "0", {}])
def test_retired_hue_rejects_malformed_summary_report(tmp_path, report):
    with pytest.raises(RerenderError, match=HUE_REPORT):
        replay_command(tmp_path, [], summary={HUE_REPORT: report})


@pytest.mark.parametrize("field", SHIFT_FIELDS)
@pytest.mark.parametrize("value", [None, "0", 0.0, False, float("nan"), float("inf")])
def test_retired_hue_rejects_malformed_summary_shifts(tmp_path, field, value):
    summary = no_hue_summary()
    summary[HUE_REPORT][field] = value
    with pytest.raises(RerenderError, match=f"{field} must be an integer"):
        replay_command(tmp_path, [], summary=summary)


@pytest.mark.parametrize("field", ["enabled", "changed"])
@pytest.mark.parametrize("value", [None, "false", 0, 1, True])
def test_retired_hue_rejects_inconsistent_summary_booleans(tmp_path, field, value):
    summary = no_hue_summary()
    summary[HUE_REPORT][field] = value
    with pytest.raises(RerenderError, match=f"{field} must be false"):
        replay_command(tmp_path, [], summary=summary)


@pytest.mark.parametrize("field", HASH_FIELDS)
@pytest.mark.parametrize("value", [None, "b" * 64, "bad", 0])
def test_retired_hue_rejects_malformed_or_changed_pixel_hashes(tmp_path, field, value):
    summary = no_hue_summary()
    summary[HUE_REPORT][field] = value
    with pytest.raises(RerenderError, match="pixel hashes disagree|must be a SHA-256"):
        replay_command(tmp_path, [], summary=summary)


@pytest.mark.parametrize("field", [*SHIFT_FIELDS, "enabled", "changed", *HASH_FIELDS])
def test_retired_hue_rejects_incomplete_report(tmp_path, field):
    summary = no_hue_summary()
    del summary[HUE_REPORT][field]
    with pytest.raises(RerenderError, match=field):
        replay_command(tmp_path, [], summary=summary)


@pytest.mark.parametrize("options", [[HUE_OPTION, "0"], [f"{HUE_OPTION}=0"]])
def test_retired_hue_remains_unsupported_for_live_worker(options):
    with pytest.raises(SystemExit) as error:
        parse(options)
    assert error.value.code == 2


def test_retired_options_preserve_unknown_task_guard(tmp_path):
    with pytest.raises(RerenderError, match="unknown task.objects 'unknown_task'"):
        replay_command(tmp_path, [HUE_OPTION, "0"], task="unknown_task")
    assert not (tmp_path / "rerender").exists()


@pytest.mark.parametrize("source", ["command", "summary", "quiet"])
def test_source_preflight_rejects_retired_input_before_rendering(run_dir, source):
    if source == "summary":
        summary_path = run_dir / "sim_summary.json"
        summary = json.loads(summary_path.read_text())
        summary.update(no_hue_summary())
        summary[HUE_REPORT]["hue_shift_opencv_bins"] = 1
        summary_path.write_text(json.dumps(summary))
    else:
        options = (
            [HUE_OPTION, "1"]
            if source == "command"
            else ["--success-quiet-rotation-deg", "nan"]
        )
        command_path = run_dir / "RUN_COMMAND.md"
        command_path.write_text(
            command_path.read_text().replace(
                "--no-render", f"{shlex.join(options)} --no-render"
            )
        )
    with pytest.raises(RerenderError, match="nonzero|finite number"):
        prepare_source_run(run_dir, renderer_code={"commit": "a" * 40, "dirty": True})
    assert not (run_dir / "rerender").exists()


def test_source_preflight_preserves_original_command_and_zero_summary(run_dir):
    command_path = run_dir / "RUN_COMMAND.md"
    command_path.write_text(
        command_path.read_text().replace(
            "--no-render", f"{HUE_OPTION}=179 {HUE_OPTION}=0 --no-render"
        )
    )
    summary_path = run_dir / "sim_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(no_hue_summary())
    summary_path.write_text(json.dumps(summary))
    source = prepare_source_run(
        run_dir, renderer_code={"commit": "a" * 40, "dirty": True}
    )
    assert f"{HUE_OPTION}=179" in source.argv
    assert f"{HUE_OPTION}=0" in source.argv
    assert source.summary[HUE_REPORT] == summary[HUE_REPORT]
