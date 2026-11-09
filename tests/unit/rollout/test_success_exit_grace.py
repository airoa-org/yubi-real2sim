# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Stopping a rollout once the criterion has reported success.

Rule: run the frame budget, but once success is seen,
simulate a short grace window and then stop rather than running the budget out.
The frame budgets themselves are hard ceilings -- nothing extends a rollout
past them.

The grace frames are ordinary frames: stepped, logged and scored like any
other. That matters because most of the criteria are not latched, so a success
that does not survive the grace window is still visible in the log, and the
run's official field still comes from the last frame.
"""

from __future__ import annotations

import types


from robosuite_env.rollout.stages.run_frames import _success_exit_after_frame


def _state(*, grace=30, criterion=True):
    return types.SimpleNamespace(
        A=types.SimpleNamespace(
            success_exit_grace_frames=grace,
        ),
        _SUCCESS_CRITERION=object() if criterion else None,
        _RECS=[],
        _SUCCESS_FIRST_FRAME=None,
    )


def _frame(state, index, success):
    """Append the record the loop would have written, then ask for a stop."""

    state._RECS.append({"frame": index, "success": {"success": success}})
    return _success_exit_after_frame(state, index)


def test_a_rollout_that_never_succeeds_is_never_stopped_early():
    state = _state()

    for index in range(200):
        assert _frame(state, index, False) is None
    assert state._SUCCESS_FIRST_FRAME is None


def test_it_stops_exactly_the_grace_window_after_the_first_success():
    state = _state(grace=30)

    for index in range(10):
        assert _frame(state, index, False) is None
    assert _frame(state, 10, True) is None, "the success frame itself keeps going"
    for index in range(11, 40):
        assert _frame(state, index, True) is None, f"frame {index} is inside the window"

    assert _frame(state, 40, True) == 10
    assert state._SUCCESS_FIRST_FRAME == 10


def test_the_window_is_measured_from_the_first_success_not_the_latest():
    """A criterion that flickers does not keep pushing the stop frame back."""

    state = _state(grace=10)

    _frame(state, 5, True)
    for index in range(6, 15):
        _frame(state, index, index % 2 == 0)

    assert _frame(state, 15, False) == 5, "still stops, ten frames after frame 5"


def test_a_success_that_does_not_survive_the_window_still_stops():
    """The stop is not a claim that the run succeeded -- scoring is separate."""

    state = _state(grace=5)

    assert _frame(state, 3, True) is None
    for index in range(4, 8):
        _frame(state, index, False)

    assert _frame(state, 8, False) == 3


def test_zero_grace_disables_the_early_exit():
    state = _state(grace=0)

    for index in range(100):
        assert _frame(state, index, True) is None


def test_a_run_without_a_success_criterion_is_never_stopped_early():
    state = _state(criterion=False)

    for index in range(100):
        assert _success_exit_after_frame(state, index) is None


def test_a_frame_with_no_success_block_asks_the_criterion_directly():
    """Some frame paths record no success block; absence is not failure."""

    asked = []

    class _Criterion:
        def evaluate(self):
            asked.append(True)
            return {"success": True}

    state = _state()
    state._SUCCESS_CRITERION = _Criterion()
    state._RECS.append({"frame": 0})

    assert _success_exit_after_frame(state, 0) is None
    assert asked, "the criterion was not consulted"
    assert state._SUCCESS_FIRST_FRAME == 0
