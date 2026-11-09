# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""LeRobot observation sources used by policy rollouts."""

from __future__ import annotations

import os

import numpy as np

def _lerobot_video_paths(episode_parquet):
    ep = os.path.abspath(os.path.expanduser(os.path.expandvars(str(episode_parquet))))
    if not os.path.isfile(ep):
        raise SystemExit(f"[sim] --policy-image-episode not found: {ep}")
    chunk_dir = os.path.dirname(ep)
    data_dir = os.path.dirname(chunk_dir)
    root = os.path.dirname(data_dir)
    chunk = os.path.basename(chunk_dir)
    stem = os.path.splitext(os.path.basename(ep))[0]
    if os.path.basename(data_dir) != "data":
        # Expected layout is <root>/data/<chunk>/<episode>.parquet. Keep the error explicit
        # because silently guessing the video root is a common source of stale-image tests.
        raise SystemExit(f"[sim] cannot resolve LeRobot root from episode path: {ep}")
    vids = {}
    for key, side in (("observation.image.left", "L"), ("observation.image.right", "R")):
        path = os.path.join(root, "videos", chunk, key, stem + ".mp4")
        if not os.path.isfile(path):
            raise SystemExit(f"[sim] LeRobot {side} video not found: {path}")
        vids[side] = path
    return vids

def load_lerobot_policy_images(episode_parquet):
    import av
    paths = _lerobot_video_paths(episode_parquet)
    out = {}
    for side, path in paths.items():
        frames = []
        container = av.open(path)
        try:
            for fr in container.decode(video=0):
                frames.append(np.asarray(fr.to_image()))  # RGB
        finally:
            container.close()
        if not frames:
            raise SystemExit(f"[sim] LeRobot {side} video decoded zero frames: {path}")
        out[side] = frames
    n = min(len(out["L"]), len(out["R"]))
    if len(out["L"]) != len(out["R"]):
        print(f"[sim] warning: LeRobot wrist video length mismatch L={len(out['L'])} R={len(out['R'])}; truncating to {n}", flush=True)
    out["L"] = out["L"][:n]
    out["R"] = out["R"][:n]
    print(f"[sim] LeRobot policy images loaded: frames={n} L={paths['L']} R={paths['R']}", flush=True)
    return out

def load_lerobot_policy_poses(episode_parquet):
    """Real per-frame proprioception obs (teacher-forcing): the exact columns the policy consumes.

    Returns rel[n,7] (observation.pose.left_hand_root_to_right_hand_root.absolute, XYZW) and
    joint[n,2] (observation.joint_states), fed verbatim so the policy sees the on-manifold real obs.
    """
    import pandas as pd
    ep = os.path.abspath(os.path.expanduser(os.path.expandvars(str(episode_parquet))))
    if not os.path.isfile(ep):
        raise SystemExit(f"[sim] --policy-image-episode not found: {ep}")
    df = pd.read_parquet(ep)
    relcol = "observation.pose.left_hand_root_to_right_hand_root.absolute"
    jscol = "observation.joint_states"
    for col in (relcol, jscol):
        if col not in df.columns:
            raise SystemExit(f"[sim] LeRobot pose column missing in {ep}: {col}")
    rel = np.stack(df[relcol].values).astype(np.float32)
    js = np.stack(df[jscol].values).astype(np.float32)
    if rel.shape[1] != 7 or js.shape[1] != 2:
        raise SystemExit(f"[sim] unexpected LeRobot pose shapes rel={rel.shape} joint={js.shape} (want [n,7],[n,2])")
    n = min(len(rel), len(js))
    print(f"[sim] LeRobot policy poses loaded: frames={n} rel[0]={np.round(rel[0],4).tolist()} joint[0]={np.round(js[0],4).tolist()}", flush=True)
    return {"rel": rel[:n], "joint": js[:n]}

__all__ = ["load_lerobot_policy_images", "load_lerobot_policy_poses"]
