# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""A box-pack profile is held to its own checkpoint's wording of the task.

The checkpoints under test were trained on different wordings of the same
instruction: the box baselines on the dataset's tasks.jsonl string ("with
right gripper") and the pi05_box_ct checkpoint on the same task with the
article. So a profile may depart from the task default, but only towards what
its checkpoint carries, and preflight decides that from prompt.yaml.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

from real2sim.artifacts import ManifestError  # noqa: E402
from real2sim.config._types import ConfigError  # noqa: E402
from real2sim.preflight import (  # noqa: E402
    _checkpoint_prompt_instructions,
    _require_trained_instruction,
)

from robosuite_env.tasks.box_pack import (  # noqa: E402
    BOX_PACK_INSTRUCTIONS_BY_CHECKPOINT,
    box_pack_instruction,
    box_pack_instruction_for_checkpoint,
)


BASELINE_NAME = "pi05_box_bs"
BOX_CT_NAME = "pi05_box_ct"


def _staged_checkpoint(request, name):
    # Checkpoints live under the staged input root given by the shared
    # real_inputs_root fixture. The production resolver rejects a relative or
    # empty REAL2SIM_INPUTS_ROOT; here that only means no checkpoint is
    # reachable, so skip instead of failing collection.
    try:
        root = request.getfixturevalue("real_inputs_root")
    except ManifestError as error:
        pytest.skip(f"REAL2SIM_INPUTS_ROOT is not usable: {error}")
    checkpoint = root / "checkpoints" / name
    if not checkpoint.is_dir():
        pytest.skip(f"the {name} checkpoint is not installed")
    return checkpoint


@pytest.fixture
def baseline(request):
    return _staged_checkpoint(request, BASELINE_NAME)


@pytest.fixture
def box_ct(request):
    return _staged_checkpoint(request, BOX_CT_NAME)


PEN_PLAIN = (
    "Pick up the ball point pen on the table with right gripper and put it in "
    "the box."
)
PEN_ARTICLE = (
    "Pick up the ball point pen on the table with the right gripper and put it "
    "in the box."
)
TAPE_PLAIN = "Pick up the tape on the table with right gripper and put it in the box."


def _plan(checkpoint: Path):
    return {"resolved": {"policy": {"checkpoint": str(checkpoint)}}}


def _write_prompt(directory: Path, document) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "prompt.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    return directory


def _check(checkpoint: Path, instruction: str) -> None:
    _require_trained_instruction(_plan(checkpoint), instruction, context="test")


def test_schema_one_lists_its_lane_strings_directly(tmp_path):
    checkpoint = _write_prompt(
        tmp_path / "schema1", {"lanes": {"plain": {"strings": [PEN_ARTICLE]}}}
    )
    listed, decoded = _checkpoint_prompt_instructions(checkpoint)
    assert list(listed.values()) == [PEN_ARTICLE] and decoded == {}
    _check(checkpoint, PEN_ARTICLE)


def test_schema_two_nests_its_lane_strings_under_text(tmp_path):
    checkpoint = _write_prompt(
        tmp_path / "schema2",
        {
            "lanes": {
                "action": {
                    "pa": [
                        {
                            "variant": "pa",
                            "weight": 0.4,
                            "strings": [{"text": PEN_ARTICLE}],
                        }
                    ]
                }
            }
        },
    )
    listed, _ = _checkpoint_prompt_instructions(checkpoint)
    assert list(listed.values()) == [PEN_ARTICLE]
    _check(checkpoint, PEN_ARTICLE)


def test_a_checkpoint_that_lists_no_lane_strings_is_refused(tmp_path):
    checkpoint = _write_prompt(
        tmp_path / "empty", {"lanes": {"action": {"pa": []}}, "run": {"backend": "x"}}
    )
    with pytest.raises(ConfigError, match="lists no lane strings"):
        _check(checkpoint, PEN_PLAIN)


def test_a_checkpoint_without_prompt_yaml_is_not_second_guessed(tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    _check(bare, PEN_PLAIN)


def test_a_decoded_row_outranks_the_lane_wording(tmp_path):
    # The listing is formatted; the decoded row is what was tokenised.
    checkpoint = _write_prompt(
        tmp_path / "decoded_wins",
        {
            "lanes": {"plain": {"strings": [PEN_ARTICLE]}},
            "sample": {"rows": [{"decoded": f"Task: {PEN_PLAIN}, State: 1 2 3"}]},
        },
    )
    _check(checkpoint, PEN_PLAIN)
    with pytest.raises(ConfigError, match="decoded training row"):
        _check(checkpoint, PEN_ARTICLE)


def test_an_instruction_with_no_decoded_row_keeps_the_profile_wording(tmp_path):
    # Decoded rows are samples, so their silence is not evidence: the listing
    # alone may not overrule the profile.
    checkpoint = _write_prompt(
        tmp_path / "no_decoded_row",
        {
            "lanes": {"plain": {"strings": [PEN_ARTICLE, TAPE_PLAIN]}},
            "sample": {"rows": [{"decoded": f"Task: {TAPE_PLAIN}, State: 1 2 3"}]},
        },
    )
    _check(checkpoint, PEN_PLAIN)


def test_an_instruction_the_checkpoint_never_carried_is_refused(tmp_path):
    checkpoint = _write_prompt(
        tmp_path / "unknown", {"lanes": {"plain": {"strings": [PEN_ARTICLE]}}}
    )
    with pytest.raises(ConfigError, match="does not list"):
        _check(checkpoint, "Pick up the stapler and put it in the box.")


def test_a_malformed_lane_string_does_not_admit_its_clean_form(tmp_path):
    # The baselines carry "Pick up pairs of yellow socks of with the right
    # gripper ..." -- a typo in the data. Matching is exact, so the corrected
    # wording is not admitted by the presence of the broken one.
    broken = (
        "Pick up pairs of yellow socks of with the right gripper and place in "
        "the container."
    )
    checkpoint = _write_prompt(tmp_path / "typo", {"lanes": {"plain": {"strings": [broken]}}})
    _check(checkpoint, broken)
    with pytest.raises(ConfigError, match="does not list"):
        _check(
            checkpoint,
            "Pick up pairs of yellow socks with the right gripper and place in "
            "the container.",
        )


def test_phase_bound_wording_is_refused(tmp_path):
    phase = PEN_ARTICLE.rstrip(".") + ", releasing it."
    checkpoint = _write_prompt(
        tmp_path / "phase",
        {
            "lanes": {"plain": {"strings": [phase]}},
            "sample": {"rows": [{"decoded": f"Task: {phase}, State: 1 2 3"}]},
        },
    )
    with pytest.raises(ConfigError, match="phase-bound"):
        _check(checkpoint, phase)


def _box_plan(tmp_path, instruction, task_index=2, checkpoint=None):
    from robosuite_env.tasks import box_pack

    return {
        "repo_root": REPO_ROOT,
        "resolved": {
            "robot": "fr3",
            "task": {
                "objects": "box_pack",
                "instruction": instruction,
                "goal_condition": box_pack.BOX_PACK_GOAL_CONDITION,
                "score_condition": box_pack.BOX_PACK_SCORE_CONDITION,
            },
            "policy": {"checkpoint": str(checkpoint or tmp_path / "absent")},
            "simulation": {"parameters": {"box_pack_task_index": task_index}},
            "scenes": [{"tag": "franka_flat", "path": "/scenes/franka_flat"}],
        },
    }


@pytest.fixture
def only_the_instruction(monkeypatch):
    # The layout check has its own tests.
    from real2sim import preflight

    monkeypatch.setattr(preflight, "_validate_selected_layout", lambda *a, **k: ({}, {}))


def test_a_profile_may_not_swap_the_task_its_index_names(tmp_path, only_the_instruction):
    from real2sim import preflight

    # Index 2 is the pen, right hand. The tape wording is a real instruction of
    # this checkpoint, and still not this task.
    with pytest.raises(ConfigError, match="does not describe its task"):
        preflight._validate_task_objects(_box_plan(tmp_path, TAPE_PLAIN))
    with pytest.raises(ConfigError, match="does not describe its task"):
        preflight._validate_task_objects(
            _box_plan(tmp_path, box_pack_instruction(3))  # the pen, other hand
        )
    preflight._validate_task_objects(_box_plan(tmp_path, PEN_PLAIN))
    preflight._validate_task_objects(_box_plan(tmp_path, PEN_ARTICLE))


def test_the_out_of_box_direction_is_bound_too(tmp_path, only_the_instruction):
    from real2sim import preflight

    preflight._validate_task_objects(
        _box_plan(tmp_path, box_pack_instruction(8), task_index=8)
    )
    with pytest.raises(ConfigError, match="does not describe its task"):
        preflight._validate_task_objects(_box_plan(tmp_path, PEN_PLAIN, task_index=8))


def test_a_decoded_row_does_not_add_to_the_allowed_set(tmp_path):
    # Rule: decoded rows correct the listing's wording, they do not extend it.
    checkpoint = _write_prompt(
        tmp_path / "decoded_only",
        {
            "lanes": {"plain": {"strings": [PEN_ARTICLE]}},
            "sample": {"rows": [{"decoded": f"Task: {TAPE_PLAIN}, State: 1 2"}]},
        },
    )
    with pytest.raises(ConfigError, match="does not list"):
        _check(checkpoint, TAPE_PLAIN)


def test_only_the_article_and_one_full_stop_are_ignored(tmp_path):
    checkpoint = _write_prompt(
        tmp_path / "tight", {"lanes": {"plain": {"strings": [PEN_ARTICLE]}}}
    )
    _check(checkpoint, PEN_PLAIN)
    for near_miss in (
        PEN_PLAIN.replace("Pick up", "Pick  up"),
        PEN_PLAIN + ".",
        PEN_PLAIN.replace("the box.", "the  box."),
    ):
        with pytest.raises(ConfigError, match="does not list"):
            _check(checkpoint, near_miss)


def test_phase_bound_wording_is_refused_with_no_prompt_record(tmp_path):
    # The guard does not depend on the checkpoint: one instruction per episode.
    bare = tmp_path / "bare_phase"
    bare.mkdir()
    phase = PEN_ARTICLE.rstrip(".") + ", releasing it."
    with pytest.raises(ConfigError, match="phase-bound"):
        _check(bare, phase)
    with pytest.raises(ConfigError, match="phase-bound"):
        _require_trained_instruction({"resolved": {}}, phase, context="test")


def test_a_prompt_yaml_that_is_not_a_mapping_is_refused(tmp_path):
    checkpoint = tmp_path / "malformed"
    checkpoint.mkdir()
    (checkpoint / "prompt.yaml").write_text("- lanes:\n    plain:\n      strings: [x]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a mapping"):
        _check(checkpoint, PEN_PLAIN)


def test_the_task_default_is_the_datasets_own_wording():
    # tasks.jsonl index 9 is the pen, right hand, and carries no article. The
    # baselines were trained from it, so the default must not drift.
    assert box_pack_instruction(2) == PEN_PLAIN
    assert box_pack_instruction(4) == TAPE_PLAIN


def test_the_baseline_checkpoint_keeps_the_plain_pen_and_tape_wordings(baseline):
    # Its decoded rows carry the tape verbatim and no pen row at all, so the
    # tape wording is pinned and the pen wording is left alone.
    listed, decoded = _checkpoint_prompt_instructions(baseline)
    assert listed and TAPE_PLAIN in decoded.values()
    _check(baseline, PEN_PLAIN)
    _check(baseline, TAPE_PLAIN)
    assert box_pack_instruction_for_checkpoint(str(baseline), 2) == PEN_PLAIN
    assert box_pack_instruction_for_checkpoint(str(baseline), 4) == TAPE_PLAIN


def test_the_box_ct_checkpoint_requires_its_own_pen_wording(box_ct):
    _check(box_ct, PEN_ARTICLE)
    with pytest.raises(ConfigError, match="decoded training row"):
        _check(box_ct, PEN_PLAIN)
    assert box_pack_instruction_for_checkpoint(str(box_ct), 2) == PEN_ARTICLE


def test_the_box_ct_wordings_come_from_its_pa_lane(box_ct):
    # The runtime sends one instruction in the Task slot, which is the pa
    # lane's form. The sht lane's short "Pack objects in the box (...)" label
    # and the pa_contact lane's phase suffixes must not be chosen instead.
    document = yaml.safe_load((box_ct / "prompt.yaml").read_text(encoding="utf-8"))
    lanes = document["lanes"]["action"]
    pa_strings = {
        row["text"]
        for entry in lanes["pa"]
        for row in entry["strings"]
        if isinstance(row, dict) and "text" in row
    }
    for instruction in BOX_PACK_INSTRUCTIONS_BY_CHECKPOINT[box_ct.name].values():
        assert instruction in pa_strings
        assert not instruction.startswith("Pack objects in the box")


def test_the_worker_refuses_what_the_launcher_would_refuse():
    # The pair that must agree: a profile carrying another task's wording must be
    # stopped by the worker even though the launcher is not there to see it.
    from robosuite_env.tasks.box_pack import (
        box_pack_instruction,
        require_box_pack_instruction,
    )

    require_box_pack_instruction(box_pack_instruction(2), 2)
    with pytest.raises(SystemExit):
        require_box_pack_instruction(box_pack_instruction(4), 2)
    with pytest.raises(SystemExit):
        require_box_pack_instruction("Pack the box.", 2)


def test_a_worker_that_loads_its_own_checkpoint_is_held_to_its_wording():
    # When the worker does know the checkpoint -- it loaded the policy itself
    # rather than talking to a batch server -- it makes the exact check, and
    # the other checkpoint's wording is refused.
    from robosuite_env.tasks.box_pack import (
        box_pack_instruction,
        box_pack_instruction_for_checkpoint,
        require_box_pack_instruction,
    )

    box_ct = "/somewhere/pi05_box_ct"
    require_box_pack_instruction(
        box_pack_instruction_for_checkpoint(box_ct, 2), 2, checkpoint=box_ct
    )
    with pytest.raises(SystemExit):
        require_box_pack_instruction(
            box_pack_instruction(2), 2, checkpoint=box_ct
        )


def test_every_registered_wording_still_names_its_own_task():
    # Widening what the worker accepts must not let one task's prompt through
    # under another's index: that is the whole of what the worker can check.
    from real2sim import preflight
    from robosuite_env.tasks.box_pack import box_pack_instructions_for_task

    for task_index in range(12):
        for wording in box_pack_instructions_for_task(task_index):
            preflight._require_instruction_names_the_task(
                wording, task_index, context="test"
            )
        other = 3 if task_index != 3 else 2
        for wording in box_pack_instructions_for_task(other):
            if wording in box_pack_instructions_for_task(task_index):
                continue
            with pytest.raises(ConfigError):
                preflight._require_instruction_names_the_task(
                    wording, task_index, context="test"
                )


def test_only_a_checkpoint_the_table_knows_can_hold_a_prompt():
    # The table's answer for an unregistered checkpoint is the task default,
    # which is a guess. Enforcing a guess is how a correct run dies minutes in
    # on a node, so the worker asks first whether the table knows the name.
    from robosuite_env.tasks.box_pack import (
        box_pack_checkpoint_is_registered as registered,
    )

    assert registered("/inputs/checkpoints/pi05_box_ct")
    assert registered("pi05_box_ct/")
    # The baselines carry the default wording and are deliberately not listed.
    assert not registered("/inputs/checkpoints/pi05_box_bs")
    # A remote policy server names an endpoint, not a directory.
    assert not registered("tcp://policy-host:9000")
    assert not registered("") and not registered(None)


def test_the_served_checkpoint_catches_a_mispaired_prompt():
    # What a batch worker can check once the server tells it what it loaded:
    # the pi05_box_ct checkpoint served against the task default is a pairing
    # no preflight saw, because a direct batch launch runs none.
    from robosuite_env.tasks.box_pack import (
        box_pack_instruction,
        box_pack_instruction_for_checkpoint,
        require_box_pack_instruction,
    )

    served = "/inputs/checkpoints/pi05_box_ct"
    require_box_pack_instruction(
        box_pack_instruction_for_checkpoint(served, 2), 2, checkpoint=served
    )
    with pytest.raises(SystemExit, match="does not match the checkpoint"):
        require_box_pack_instruction(box_pack_instruction(2), 2, checkpoint=served)


def test_a_registered_checkpoint_gets_the_workers_exact_check(tmp_path, only_the_instruction):
    # The pi05_box_ct checkpoint carries no decoded tape row, and the lane key
    # ignores the article, so its prompt.yaml alone cannot tell "with right
    # gripper" from "with the right gripper" for task 4. The worker can, once
    # the policy server names the checkpoint; preflight must refuse the same
    # wording up front instead of letting the run die on the node.
    from real2sim import preflight
    from robosuite_env.tasks import box_pack

    served = tmp_path / "pi05_box_ct"
    served.mkdir()
    registered = box_pack.box_pack_instruction_for_checkpoint(str(served), 4)
    assert "with the right gripper" in registered
    preflight._validate_task_objects(
        _box_plan(tmp_path, registered, task_index=4, checkpoint=served)
    )
    plain = box_pack.box_pack_instruction(4)
    assert plain != registered
    with pytest.raises(ConfigError, match="registered checkpoint"):
        preflight._validate_task_objects(
            _box_plan(tmp_path, plain, task_index=4, checkpoint=served)
        )
