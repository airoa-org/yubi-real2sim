#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Relative -> absolute pose integration for the YUBI hand_root stream (socks_cup convention).

VERIFIED against ground-truth `observation.pose.*_hand_root.absolute` (error ~0): the per-frame
relative delta is expressed in the hand's OWN (LOCAL) frame, NOT world:

    abs_pos[t] = abs_pos[t-1] + R(abs_q[t-1]) @ dpos      # position delta rotated by current orientation
    abs_q[t]   = abs_q[t-1] (x) dq                        # orientation delta = LOCAL (right) quaternion compose

(A world-additive `abs_pos[t] = abs_pos[t-1] + dpos` form is wrong here: ~372 mm drift.)

All quaternions are XYZW (the parquet convention). The policy emits deltas in THIS same convention,
so policy rollout must integrate the same way.
"""
import numpy as np


def quat_xyzw_to_mat(q):
    x, y, z, w = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def qmul_xyzw(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def apply_delta_local(p, q_xyzw, rel):
    """One step: (abs pos[3], abs quat[xyzw]) + relative[dx,dy,dz, dqx,dqy,dqz,dqw] -> new (pos, quat)."""
    dp = np.asarray(rel[0:3], float)
    dq = np.asarray(rel[3:7], float); dq = dq / (np.linalg.norm(dq) + 1e-12)
    p_new = np.asarray(p, float) + quat_xyzw_to_mat(q_xyzw) @ dp
    q_new = qmul_xyzw(np.asarray(q_xyzw, float), dq)
    return p_new, q_new / (np.linalg.norm(q_new) + 1e-12)
