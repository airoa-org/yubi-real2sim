# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Replay parquet and state-log source helpers."""

from __future__ import annotations

import json
import os

import mujoco
import numpy as np

from robosuite_env.rollout.control.pose_math import _quat_normalize as quat_normalize
from robosuite_env.rollout.replay_policy_insets import (
    ReplayPolicyInsetError,
    load_replay_policy_insets,
)
from robosuite_env.rollout.state.render_state import (
    validate_render_state_row,
    validate_render_state_rows,
)

def replay_integrate_relative_delta(p, q_xyzw, rel):
    """Shared replay/policy relative integrator in canonical hand-root world coordinates.

    q_xyzw and rel[3:7] use the parquet / policy XYZW convention. The returned
    position is a world-frame target produced by the verified replay integrator.
    """
    from robosuite_env.policy.rel2abs import apply_delta_local
    p2, q2 = apply_delta_local(p, q_xyzw, rel)
    return np.asarray(p2, float), np.asarray(q2, float)

def load_replay(path, cap):
    import pandas as pd
    df = pd.read_parquet(path); g=lambda k:np.stack(df[k].values).astype(np.float32)
    def pick(*ks):
        for k in ks:
            if k in df.columns: return g(k)
        raise KeyError(ks)
    relR=pick("observation.pose.right_hand_root.relative","action.pose.right_hand_root.relative")
    relL=pick("observation.pose.left_hand_root.relative","action.pose.left_hand_root.relative")
    try: grip=pick("observation.joint_states","action.joint_states")[:,:2]
    except Exception: grip=np.zeros((len(relR),2),np.float32)
    aR=pick("observation.pose.right_hand_root.absolute"); aL=pick("observation.pose.left_hand_root.absolute")
    n=len(relR); n=min(n,cap) if cap else n
    obj={}
    for key in (
        "observation.object.cup_bottom",
        "observation.object.cup_center",
        "observation.object.plate_center",
        "observation.object.cup_quat",
        "observation.object.plate_quat",
    ):
        if key in df.columns:
            obj[key]=g(key)[:n]
    return relR[:n], relL[:n], grip[:n], n, aR[:n], aL[:n], obj

def _force_replay_object_poses(state,t):
    if not (state.A.replay and state.A.replay_object_source == "parquet" and state.REPLAY_OBJ):
        return False
    idx = min(int(t), len(next(iter(state.REPLAY_OBJ.values()))) - 1)
    cup_pos = state.REPLAY_OBJ.get("observation.object.cup_bottom")
    if cup_pos is None:
        cup_pos = state.REPLAY_OBJ.get("observation.object.cup_center")
    if cup_pos is not None:
        state.d.qpos[state.CUPa:state.CUPa+3] = np.asarray(cup_pos[idx], float)[:3]
        cup_quat = state.REPLAY_OBJ.get("observation.object.cup_quat")
        state.d.qpos[state.CUPa+3:state.CUPa+7] = quat_normalize(
            np.asarray(cup_quat[idx], float)[:4] if cup_quat is not None else [1.0, 0.0, 0.0, 0.0]
        )
        state.d.qvel[state.cdof:state.cdof+6] = 0.0
    plate_pos = state.REPLAY_OBJ.get("observation.object.plate_center")
    if state.PLATEa is not None and plate_pos is not None:
        state.d.qpos[state.PLATEa:state.PLATEa+3] = np.asarray(plate_pos[idx], float)[:3]
        plate_quat = state.REPLAY_OBJ.get("observation.object.plate_quat")
        if state.PLATEa + 7 <= len(state.d.qpos):
            state.d.qpos[state.PLATEa+3:state.PLATEa+7] = quat_normalize(
                np.asarray(plate_quat[idx], float)[:4] if plate_quat is not None else [1.0, 0.0, 0.0, 0.0]
            )
        if state.PLATEd is not None:
            state.d.qvel[state.PLATEd:state.PLATEd+6] = 0.0
    mujoco.mj_forward(state.m, state.d)
    return True


def load_state_log_replay(state):
    """Load and validate the exact render-state replay source."""

    state.STATE_ROWS = []
    with open(state.A.state_log_replay) as stream:
        for line in stream:
            line = line.strip()
            if line:
                state.STATE_ROWS.append(json.loads(line))
    if not state.STATE_ROWS:
        raise SystemExit(
            f"[sim] --state-log-replay log is empty: {state.A.state_log_replay}"
        )
    try:
        state._STATE_RENDER_VALIDATION = validate_render_state_rows(
            state.STATE_ROWS,
            (
                state._STATE_LOG_SOURCE_SUMMARY.get("render_state_log")
                if isinstance(state._STATE_LOG_SOURCE_SUMMARY, dict)
                else None
            ),
            expected_contract=state._RENDER_STATE_CONTRACT,
            required=False,
        )
    except ValueError as exc:
        raise SystemExit(f"[sim] invalid render-state replay source: {exc}") from exc
    state._STATE_RENDER_QPOS = (
        [
            np.asarray(
                validate_render_state_row(
                    row, state._RENDER_STATE_CONTRACT, required=True
                )["qpos"],
                dtype=np.float64,
            )
            for row in state.STATE_ROWS
        ]
        if state._STATE_RENDER_VALIDATION is not None
        else None
    )
    state._STATE_REPLAY_TARGET_BASELINE_SOURCE = None
    source_settle = (
        state._STATE_LOG_SOURCE_SUMMARY.get("dynamic_object_settle")
        if isinstance(state._STATE_LOG_SOURCE_SUMMARY, dict)
        else None
    )
    source_settle_objects = (
        source_settle.get("objects") if isinstance(source_settle, dict) else None
    )
    source_target_state = (
        source_settle_objects.get(state._TASK_OBJECT_TARGET)
        if isinstance(source_settle_objects, dict)
        else None
    )
    source_target_qpos = (
        np.asarray(source_target_state.get("qpos", []), dtype=np.float64)
        if isinstance(source_target_state, dict)
        else np.asarray([])
    )
    if source_target_qpos.shape == (7,) and np.all(np.isfinite(source_target_qpos)):
        state.cup = source_target_qpos[:3].copy()
        state._STATE_REPLAY_TARGET_BASELINE_SOURCE = (
            "source_summary.dynamic_object_settle.objects."
            f"{state._TASK_OBJECT_TARGET}.qpos"
        )
    elif state._STATE_RENDER_QPOS is not None:
        state.cup = np.asarray(
            state._STATE_RENDER_QPOS[0][state.CUPa : state.CUPa + 3],
            dtype=np.float64,
        ).copy()
        state._STATE_REPLAY_TARGET_BASELINE_SOURCE = "first_render_state_row"
    else:
        legacy_target_pos = state.STATE_ROWS[0].get("cup_pos")
        if legacy_target_pos is not None and len(legacy_target_pos) >= 3:
            state.cup = np.asarray(legacy_target_pos[:3], dtype=np.float64).copy()
            state._STATE_REPLAY_TARGET_BASELINE_SOURCE = "first_legacy_state_row"
    state._STATE_RENDER_ROWS_VALIDATED = (
        int(state._STATE_RENDER_VALIDATION["rows_written"])
        if state._STATE_RENDER_VALIDATION is not None
        else 0
    )
    state._STATE_RENDER_ROWS_APPLIED = 0
    state._STATE_RENDER_MAX_APPLY_ERROR = 0.0
    if state.A.state_replay_policy_obs_dir:
        try:
            state._STATE_REPLAY_POLICY_INSETS = load_replay_policy_insets(
                state.STATE_ROWS, state.A.state_replay_policy_obs_dir
            )
        except ReplayPolicyInsetError as exc:
            raise SystemExit(
                f"[sim] invalid saved replay policy wrist sequence: {exc}"
            ) from exc
        print(
            "[sim] saved replay policy wrist insets ready: "
            f"frames={len(state._STATE_REPLAY_POLICY_INSETS.frames)} "
            f"unique_pairs={state._STATE_REPLAY_POLICY_INSETS.unique_pairs} "
            f"dir={os.path.abspath(state.A.state_replay_policy_obs_dir)}",
            flush=True,
        )
    state.N = (
        len(state.STATE_ROWS)
        if state.A.frames <= 0
        else min(state.A.frames, len(state.STATE_ROWS))
    )
    print(
        f"[sim] state-log-replay: {state.A.state_log_replay} "
        f"frames={len(state.STATE_ROWS)} render_frames={state.N}",
        flush=True,
    )

class ReplayActionSource:
    """Callable adapter over replay-relative poses and gripper samples."""

    def __init__(self, rel_right, rel_left, grip):
        self._rel_right = rel_right
        self._rel_left = rel_left
        self._grip = grip

    def __call__(self, t):
        return (
            self._rel_right[t],
            self._rel_left[t],
            float(self._grip[t, 1]),
            float(self._grip[t, 0]),
        )


__all__ = [
    "ReplayActionSource",
    "_force_replay_object_poses",
    "load_state_log_replay",
    "load_replay",
    "replay_integrate_relative_delta",
]
