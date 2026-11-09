# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Coordinate-frame constants and small pose helpers for a rollout."""

from __future__ import annotations

import mujoco
import numpy as np

from robosuite_env.rollout.control import pose_math as _pose_math
from robosuite_env.policy import policy_io as pio


def xy2wx(q):
    return np.array([q[3], q[0], q[1], q[2]])


def _qmat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, float).reshape(9))
    return q / (np.linalg.norm(q) + 1e-12)


def _axis_rot(ax, deg):
    h = np.radians(deg) / 2
    q = [np.cos(h), 0.0, 0.0, 0.0]
    q[1 + ax] = np.sin(h)
    return _pose_math._matq(q)


def _se3(p, q):
    return np.asarray(p, float), np.asarray(q, float)


def wx2xy_(q):
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def xy2wx_(q):
    return np.array([q[3], q[0], q[1], q[2]], float)


def _body_se3(state, body_id):
    return state.d.xpos[body_id].copy(), state.d.xquat[body_id].copy()


def _T_from_xyzw(position, quat_xyzw):
    transform = np.eye(4)
    transform[:3, :3] = _pose_math._matq(xy2wx_(quat_xyzw))
    transform[:3, 3] = np.asarray(position, float)
    return transform


def _T_inv(transform):
    inverse = np.eye(4)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -(transform[:3, :3].T @ transform[:3, 3])
    return inverse


def _q_from_R(rotation):
    quat = _qmat(rotation)
    return quat / (np.linalg.norm(quat) + 1e-12)


def _quat_distance_deg(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a /= np.linalg.norm(a) + 1e-12
    b /= np.linalg.norm(b) + 1e-12
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def build_frame_constants(state):
    """Return the historical world and hand-root frame constants."""

    T = np.array(state.A.world_offset, float)
    EE_OFF = np.asarray(pio.FR3_EEF_TO_HAND_ROOT_M, float)
    ROOT_TCP_OFF = np.asarray(pio.FR3_GRIPPER_ROOT_TO_HAND_ROOT_M, float)
    R_CANON_TO_FRANKA = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    ) @ _axis_rot(2, 90)
    Q_CANON_TO_FRANKA = _qmat(R_CANON_TO_FRANKA)
    Q_CANON_TO_ROBOT = Q_CANON_TO_FRANKA.copy()
    Q_ROBOT_TO_CANON = _pose_math.qconj(Q_CANON_TO_ROBOT)
    return (
        T,
        EE_OFF,
        ROOT_TCP_OFF,
        R_CANON_TO_FRANKA,
        Q_CANON_TO_FRANKA,
        Q_CANON_TO_ROBOT,
        Q_ROBOT_TO_CANON,
    )


def ee_tgt(state, key):
    p, q = _pose_math.policy_tcp_pose(state, key)
    return [p, q]


__all__ = [
    "_T_from_xyzw",
    "_T_inv",
    "_axis_rot",
    "_body_se3",
    "_qmat",
    "_q_from_R",
    "_quat_distance_deg",
    "_se3",
    "build_frame_constants",
    "ee_tgt",
    "wx2xy_",
    "xy2wx",
    "xy2wx_",
]
