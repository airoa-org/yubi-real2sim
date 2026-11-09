# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Shared quaternion and pose helpers for the rollout control paths."""

import mujoco
import numpy as np

def _matq(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.ascontiguousarray(q, float)); return R.reshape(3, 3)


def qmul(a, b):
    w0,x0,y0,z0=a; w1,x1,y1,z1=b
    return np.array([w0*w1-x0*x1-y0*y1-z0*z1, w0*x1+x0*w1+y0*z1-z0*y1, w0*y1-x0*z1+y0*w1+z0*x1, w0*z1+x0*y1-y0*x1+z0*w1])


def qconj(q): return np.array([q[0],-q[1],-q[2],-q[3]])


def _quat_normalize(q):
    q=np.asarray(q,float)
    return q/(np.linalg.norm(q)+1e-12)


def eef_pose(state,key):
    eb=state.ARM[key]["eef"]; return state.d.xpos[eb].copy(), state.d.xquat[eb].copy()


def policy_tcp_pose(state,key):
    a=state.ARM[key]; pb=a["policy"]; q=state.d.xquat[pb].copy()
    return state.d.xpos[pb].copy()+_matq(q)@a["policy_offset"], q


def policy_tcp_error(state,key,tp,tq):
    cp,cq=policy_tcp_pose(state,key); qe=qmul(np.asarray(tq),qconj(cq))
    return float(np.linalg.norm(np.asarray(tp)-cp)), float(np.linalg.norm(2*qe[1:]))


def _rlist(v,n=5): return [round(float(x),n) for x in np.asarray(v).ravel()]
