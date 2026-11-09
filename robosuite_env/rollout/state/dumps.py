# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Optional init-state and per-frame state dump helpers."""

from __future__ import annotations

import os
import re


def state_dump_client_tag(state):
    tag = state.A.policy_batch_client_id or ""
    if not tag:
        try:
            tag = os.path.basename(os.path.dirname(os.path.abspath(state.LOGDIR)))
        except Exception:
            tag = "sim"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_") or "sim"


def state_dump_path(state, label, frame=None):
    clean_label = re.sub(
        r"[^A-Za-z0-9_.-]+", "_", str(label)
    ).strip("_") or "state"
    if frame is None:
        fname = f"{state_dump_client_tag(state)}_{clean_label}.json"
    else:
        fname = (
            f"{state_dump_client_tag(state)}_f{int(frame):04d}_{clean_label}.json"
        )
    return os.path.join(state.A.state_dump_dir, fname)


def dump_state_once(state, label, frame=None):
    if not state.A.state_dump_dir:
        return
    key = (str(label), int(frame) if frame is not None else None)
    if key in state._STATE_DUMPED:
        return
    state._STATE_DUMPED.add(key)
    from robosuite_env.rollout.state.init_state import _write_sim_init_state

    _write_sim_init_state(state, state_dump_path(state, label, frame), label)


def maybe_dump_record_state(state, rec):
    frame = rec.get("frame")
    if frame is not None and int(frame) in state.STATE_DUMP_FRAMES:
        dump_state_once(state, "frame", int(frame))
    if (
        state.A.state_dump_on_success
        and isinstance(rec.get("success"), dict)
        and rec["success"].get("success")
    ):
        if not state._STATE_SUCCESS_DUMPED:
            state._STATE_SUCCESS_DUMPED = True
            dump_state_once(
                state,
                "success_first",
                int(frame) if frame is not None else None,
            )


__all__ = [
    "dump_state_once",
    "maybe_dump_record_state",
    "state_dump_client_tag",
    "state_dump_path",
]
