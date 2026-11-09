#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Box-packing task: the six LeRobot instructions, objects, and conditions.

Twelve independent single-instruction tasks, not a chain. Two templates,

    Pick up the {objects} on the table with {hands} gripper and put it in the box.
    Take the {objects} from the box with {hands} gripper and place it on the table.

filled from objects {LAN cable, ball point pen, tape} and hands {left, right}.
Six object/direction pairs, each in the two hand spellings the policy was
trained on, is twelve prompts -- the same twelve the training dataset carries.

Every ``text`` below is verbatim from the dataset's ``meta/tasks.jsonl``. A
prompt that is not character-identical to the training text is a different
task as far as the policy is concerned, so these strings are the contract and
must never be paraphrased, re-cased, or re-punctuated here.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ..contracts.assets import runtime_geometry_for_object
from ..paths import OBJECT_CALIBRATION_ROOT
from .layout_selection import box_pack_layout_path


#: Logical scene objects, in the order the task contract reports them. Each
#: object compiles under its own name (``tape_joint0``, ``box_joint0``, ...).
#: The env installs the prompt's object as ``env.cube`` (the target role) and
#: sets ``env.plate = None``; the rollout worker resolves the target from
#: ``env.cube``'s own joint, never from a fixed ``cube_joint0``.
BOX_PACK_OBJECT_NAMES = ("box", "tape", "pen", "lan")

#: The receptacle every instruction fills.
BOX_PACK_RECEPTACLE = "box"

BOX_PACK_GOAL_CONDITION = "object_in_box"
BOX_PACK_SCORE_CONDITION = "object_in_box_containment"
BOX_PACK_LAYOUT_PATH = OBJECT_CALIBRATION_ROOT / "layout_box_pack_duo_manual.json"

#: task_index -> the exact prompt, the object it moves, and the hand it names.
#: The dataset assigns a separate index to each hand, so the hand is part of
#: the task rather than a runtime choice.
BOX_PACK_INSTRUCTIONS = (
    {
        "task_index": 0,
        "direction": "in",
        "object": "lan",
        "hand": "R",
        "text": (
            "Pick up the LAN cable on the table with right gripper and "
            "put it in the box."
        ),
    },
    {
        "task_index": 1,
        "direction": "in",
        "object": "lan",
        "hand": "L",
        "text": (
            "Pick up the LAN cable on the table with left gripper and "
            "put it in the box."
        ),
    },
    {
        "task_index": 2,
        "direction": "in",
        "object": "pen",
        "hand": "R",
        "text": (
            "Pick up the ball point pen on the table with right gripper "
            "and put it in the box."
        ),
    },
    {
        "task_index": 3,
        "direction": "in",
        "object": "pen",
        "hand": "L",
        "text": (
            "Pick up the ball point pen on the table with left gripper "
            "and put it in the box."
        ),
    },
    {
        "task_index": 4,
        "direction": "in",
        "object": "tape",
        "hand": "R",
        "text": (
            "Pick up the tape on the table with right gripper and put "
            "it in the box."
        ),
    },
    {
        "task_index": 5,
        "direction": "in",
        "object": "tape",
        "hand": "L",
        "text": (
            "Pick up the tape on the table with left gripper and put it "
            "in the box."
        ),
    },
    {
        "task_index": 6,
        "direction": "out",
        "object": "lan",
        "hand": "R",
        "text": (
            "Take the LAN cable from the box with right gripper and "
            "place it on the table."
        ),
    },
    {
        "task_index": 7,
        "direction": "out",
        "object": "lan",
        "hand": "L",
        "text": (
            "Take the LAN cable from the box with left gripper and "
            "place it on the table."
        ),
    },
    {
        "task_index": 8,
        "direction": "out",
        "object": "pen",
        "hand": "R",
        "text": (
            "Take the ball point pen from the box with right gripper "
            "and place it on the table."
        ),
    },
    {
        "task_index": 9,
        "direction": "out",
        "object": "pen",
        "hand": "L",
        "text": (
            "Take the ball point pen from the box with left gripper and "
            "place it on the table."
        ),
    },
    {
        "task_index": 10,
        "direction": "out",
        "object": "tape",
        "hand": "R",
        "text": (
            "Take the tape from the box with right gripper and place it "
            "on the table."
        ),
    },
    {
        "task_index": 11,
        "direction": "out",
        "object": "tape",
        "hand": "L",
        "text": (
            "Take the tape from the box with left gripper and place it "
            "on the table."
        ),
    },
)
#: The dataset's own high-level label. It is a run label only: no episode
#: carries it as a prompt, so it must never reach ``obs["prompt"]``.
BOX_PACK_RUN_LABEL = "Pack objects in the box (LAN cable, ball point pen, tape)"


def box_pack_direction(task_index: int) -> str:
    """Return ``in`` for a prompt that fills the box, ``out`` for one that empties it."""

    return _record(task_index)["direction"]


def box_pack_task_indices() -> tuple[int, ...]:
    """Return every registered task index, in order."""

    return tuple(record["task_index"] for record in BOX_PACK_INSTRUCTIONS)


def box_pack_instruction(task_index: int) -> str:
    """Return one exact LeRobot prompt."""

    return _record(task_index)["text"]

#: What each packable object is called inside an instruction. Any wording of a
#: task has to name its object and its hand, whichever checkpoint it came from.
BOX_PACK_OBJECT_LABELS = {
    "lan": "LAN cable",
    "pen": "ball point pen",
    "tape": "tape",
}


#: How individual checkpoints word these instructions, where that differs from
#: the default above. The default is what the box baselines were fed; the
#: pi05_box_ct checkpoint was fed the same tasks with the article ("with the
#: right gripper"). A profile driving such a checkpoint carries the wording
#: from this table in ``task.instruction``, and src/real2sim/preflight.py holds
#: it to the checkpoint's own decoded prompt rows.
#:
#: Provenance per entry: the string appears verbatim in that checkpoint's own
#: prompt.yaml, in a decoded sample row where one exists and in its plain
#: prompt list otherwise. Its phase-suffixed variants are deliberately not
#: used, and neither is its short "Goal:" form: the pi05_box_ct
#: wrapper is "Goal: {goal}, Task: {prompt}, State: {state};" with exclusive
#: combination, and the runtime sends its instruction in the Task slot.
BOX_PACK_INSTRUCTIONS_BY_CHECKPOINT = {
    "pi05_box_ct": {
        # decoded: "Pick up the ball point pen on the table with the right
        # gripper and put it in the box." (the same checkpoint also carries a
        # ", releasing it." variant, which is phase-bound and not used).
        2: (
            "Pick up the ball point pen on the table with the right gripper "
            "and put it in the box."
        ),
        # the plain prompt list of the same checkpoint carries this verbatim; its
        # decoded rows happen to sample no tape-into-box episode.
        4: (
            "Pick up the tape on the table with the right gripper and put it "
            "in the box."
        ),
    },
}


def box_pack_checkpoint_is_registered(checkpoint) -> bool:
    """Does the table know this checkpoint's wording, by name?

    Only then may a worker hold a prompt to it. For a checkpoint the table has
    never heard of, the table's answer is the task default, which is a guess:
    enforcing it would refuse a profile whose wording the launcher had checked
    against that checkpoint's own prompt.yaml and found correct -- the run dies
    minutes in, on a node, for being right.
    """

    if not checkpoint:
        return False
    name = str(checkpoint).rstrip("/").rsplit("/", 1)[-1]
    return name in BOX_PACK_INSTRUCTIONS_BY_CHECKPOINT


def require_box_pack_instruction(instruction, task_index, *, checkpoint=None):
    """The check the simulation worker makes on the prompt it was handed.

    With a checkpoint in hand -- a worker that loads the policy itself -- this
    is exact: the wording has to be the one that checkpoint carries. Without
    one, which is every batch run, the policy server holds the checkpoint and
    this process is only given an address, so the question narrows to whether
    the prompt belongs to this task at all: the right object, the right hand,
    the right direction. The launcher's preflight makes the strict check in
    that case (src/real2sim/preflight.py::_require_trained_instruction).
    """

    index = int(task_index)
    if checkpoint:
        expected = box_pack_instruction_for_checkpoint(str(checkpoint), index)
        if instruction != expected:
            raise SystemExit(
                "[sim] box pack instruction does not match the checkpoint this "
                f"worker loaded for task_index {index}: expected {expected!r}, "
                f"got {instruction!r}"
            )
        return
    allowed = box_pack_instructions_for_task(index)
    if instruction not in allowed:
        raise SystemExit(
            "[sim] box pack instruction does not match the selected task_index "
            f"{index}: expected one of {sorted(allowed)!r}, got {instruction!r}"
        )


def box_pack_instructions_for_task(task_index: int) -> frozenset:
    """Every wording a profile may carry for this task, over all checkpoints.

    The worker checks the prompt it was handed against this. It cannot check
    the tighter thing -- that the wording is the one *its own* checkpoint was
    trained on -- because in a batch run the checkpoint is loaded by the policy
    server and the worker is never told which it is; the launcher's preflight
    holds that end (src/real2sim/preflight.py::_require_trained_instruction).
    What the worker still catches is a prompt that belongs to another task:
    the wrong object, the wrong hand, the wrong direction.
    """

    index = int(task_index)
    return frozenset(
        {box_pack_instruction(index)}
        | {
            wordings[index]
            for wordings in BOX_PACK_INSTRUCTIONS_BY_CHECKPOINT.values()
            if index in wordings
        }
    )


def box_pack_instruction_for_checkpoint(checkpoint: str, task_index: int) -> str:
    """The instruction to give ``checkpoint``, which may not be the default."""

    name = str(checkpoint).rstrip("/").rsplit("/", 1)[-1]
    return BOX_PACK_INSTRUCTIONS_BY_CHECKPOINT.get(name, {}).get(
        int(task_index), box_pack_instruction(int(task_index))
    )


def box_pack_target_object(task_index: int) -> str:
    """Return the logical object a task index moves into the box."""

    return _record(task_index)["object"]


def box_pack_hand(task_index: int) -> str:
    """Return the gripper the instruction names, ``R`` or ``L``."""

    return _record(task_index)["hand"]


def box_pack_task_contract(task_index: int) -> dict[str, Any]:
    """Return one task index's full contract, geometry included."""

    record = deepcopy(_record(task_index))
    record["receptacle"] = BOX_PACK_RECEPTACLE
    record["goal_condition"] = BOX_PACK_GOAL_CONDITION
    record["score_condition"] = BOX_PACK_SCORE_CONDITION
    record["distractors"] = tuple(
        name
        for name in BOX_PACK_OBJECT_NAMES
        if name not in (record["object"], BOX_PACK_RECEPTACLE)
    )
    record["geometry"] = {
        name: dict(runtime_geometry_for_object(name)["source_collision_geometry"])
        for name in (record["object"], BOX_PACK_RECEPTACLE)
    }
    return record


def _record(task_index: int) -> Mapping[str, Any]:
    if isinstance(task_index, bool) or not isinstance(task_index, int):
        raise ValueError("box pack task_index must be an integer")
    for record in BOX_PACK_INSTRUCTIONS:
        if record["task_index"] == task_index:
            return record
    raise ValueError(
        "box pack task_index must be in the inclusive range 0..11, got "
        f"{task_index}"
    )


__all__ = [
    "BOX_PACK_LAYOUT_PATH",
    "BOX_PACK_GOAL_CONDITION",
    "BOX_PACK_INSTRUCTIONS",
    "BOX_PACK_OBJECT_NAMES",
    "BOX_PACK_RECEPTACLE",
    "BOX_PACK_RUN_LABEL",
    "BOX_PACK_SCORE_CONDITION",
    "box_pack_direction",
    "box_pack_layout_path",
    "box_pack_hand",
    "box_pack_instruction",
    "box_pack_target_object",
    "box_pack_task_contract",
    "box_pack_task_indices",
]
