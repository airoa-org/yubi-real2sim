# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Policy deployment anchor and homogeneous-frame transforms."""

from __future__ import annotations

import os

import numpy as np

from robosuite_env.rollout.control.pose_math import qmul
from robosuite_env.rollout.setup.frames import (
    _T_from_xyzw,
    _T_inv,
    _q_from_R,
)

def _anchor_table_origin(state):
    raw=str(state.A.anchor_table_origin).replace(",", " ").split()
    if len(raw)!=3:
        raise SystemExit("[sim] --anchor-table-origin expects exactly 3 floats: 'X Y Z'")
    try:
        return [float(x) for x in raw]
    except ValueError as e:
        raise SystemExit(f"[sim] invalid --anchor-table-origin {state.A.anchor_table_origin!r}: {e}") from e

def load_policy_deploy_ready(state):
    # init from the episode actually used for obs (so sim init pose matches --policy-image-episode);
    # Uses the explicit episode, then the optional REAL2SIM_POLICY_EPISODE.
    _init_ep = getattr(state.A, "policy_image_episode", "") or state.DEFAULT_POLICY_EPISODE
    _init_ep=os.path.abspath(os.path.expanduser(os.path.expandvars(str(_init_ep))))
    if not os.path.isfile(_init_ep):
        _message=f"[sim] policy-anchor deploy episode not found: {_init_ep}"
        print(_message+"; using hand-written ready pose",flush=True)
        state.POLICY_ANCHOR_EFFECTIVE="ready_fallback"
        return None
    try:
        import pandas as _pd
        df=_pd.read_parquet(_init_ep)
    except Exception as exc:
        _message=f"[sim] could not read policy-anchor deploy episode {_init_ep}: {exc}"
        print(_message+"; using hand-written ready pose",flush=True)
        state.POLICY_ANCHOR_EFFECTIVE="ready_fallback"
        return None
    _hand_features=(
        "observation.pose.right_hand_root.absolute",
        "observation.pose.left_hand_root.absolute",
    )
    _required_features=("observation.pose.hmd.absolute",)+_hand_features
    _missing=[feature for feature in _required_features if feature not in df.columns]
    if len(df)==0 or _missing:
        _reason=(
            "contains no rows"
            if len(df)==0
            else f"is missing required columns {_missing}"
        )
        _message=f"[sim] policy-anchor deploy episode {_init_ep} {_reason}"
        print(_message+"; using hand-written ready pose",flush=True)
        state.POLICY_ANCHOR_EFFECTIVE="ready_fallback"
        return None
    frame0={}
    for _feature in _required_features:
        try:
            _pose=np.asarray(df[_feature].iloc[0],float).reshape(-1)
        except Exception as exc:
            _message=(
                f"[sim] policy-anchor deploy episode {_init_ep} has an unreadable "
                f"frame-0 {_feature}: {exc}"
            )
            print(_message+"; using hand-written ready pose",flush=True)
            state.POLICY_ANCHOR_EFFECTIVE="ready_fallback"
            return None
        if _pose.size<7 or not np.all(np.isfinite(_pose[:7])):
            _message=(
                f"[sim] policy-anchor deploy episode {_init_ep} frame-0 "
                f"{_feature} must contain at least seven finite pose values"
            )
            print(_message+"; using hand-written ready pose",flush=True)
            state.POLICY_ANCHOR_EFFECTIVE="ready_fallback"
            return None
        frame0[_feature]=_pose
    state.POLICY_ANCHOR_EFFECTIVE="deploy"
    state.POLICY_ANCHOR_EPISODE=_init_ep
    if "observation.joint_states" in df:
        js=np.asarray(df["observation.joint_states"].iloc[0],float)
        if js.shape[0]>=2:
            state.POLICY_INIT_GRIP={"L":float(np.clip(js[0],0,0.7854)),"R":float(np.clip(js[1],0,0.7854))}
            print(f"[sim] policy-anchor deploy init grip: L={state.POLICY_INIT_GRIP['L']:.4f} R={state.POLICY_INIT_GRIP['R']:.4f}",flush=True)

    # Franka YUBI deployment lerobot.transform_to_reference:
    # odom -> table_origin, then inverse(quest_hmd -> table_origin), then inverse(frame0 hmd).
    state.ANCHOR_TABLE_ORIGIN=_anchor_table_origin(state)
    T_odom_table=_T_from_xyzw(state.ANCHOR_TABLE_ORIGIN,[0.0,0.0,-0.7071067811865475,0.7071067811865476])
    T_quest_hmd_table=_T_from_xyzw(
        [1.2247249000943632,0.020947262038982534,-0.006677898234565015],
        [0.4914741803575705,-0.5132614856606474,-0.4902425683324312,0.5046563202557766],
    )
    hmd0=frame0["observation.pose.hmd.absolute"]
    T_hmd0=_T_from_xyzw(hmd0[:3],hmd0[3:7])
    T_dataset_odom=T_odom_table@_T_inv(T_quest_hmd_table)@_T_inv(T_hmd0)
    # sim world is the table_origin frame: table surface z=0, matching robosuite_env table_z=0.
    T_dataset_sim=_T_inv(T_odom_table)@T_dataset_odom
    out={}
    dbg={}
    for key,feat in (("R","observation.pose.right_hand_root.absolute"),("L","observation.pose.left_hand_root.absolute")):
        raw=frame0[feat]
        T_hand_dataset=_T_from_xyzw(raw[:3],raw[3:7])
        T_hand_sim=T_dataset_sim@T_hand_dataset
        qcanon=_q_from_R(T_hand_sim[:3,:3])
        qgrip=qmul(qcanon,state.Q_CANON_TO_ROBOT); qgrip/=np.linalg.norm(qgrip)+1e-12
        out[key]=(T_hand_sim[:3,3].copy(),qgrip)
        dbg[key]=np.round(T_hand_sim[:3,3],4).tolist()
    table_check=np.round((_T_inv(T_odom_table)@np.r_[T_odom_table[:3,3],1.0])[:3],6).tolist()
    print(f"[sim] policy-anchor deploy: sim world = table_origin; table_origin check {table_check}",flush=True)
    print(f"[sim] policy-anchor deploy targets TCP sim: R={dbg['R']} L={dbg['L']}",flush=True)
    return out

__all__ = [
    "_T_from_xyzw",
    "_T_inv",
    "_anchor_table_origin",
    "_q_from_R",
    "load_policy_deploy_ready",
]
