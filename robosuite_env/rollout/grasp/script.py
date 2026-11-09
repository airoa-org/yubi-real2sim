# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Scripted grasp-test trajectory."""

import numpy as np


class GraspTestScript:
    def __init__(self, state, phase_bounds, grasp_quat, pre, grasp, top):
        self._state = state
        self._phase_bounds = phase_bounds
        self._grasp_quat = grasp_quat
        self._pre = pre
        self._grasp = grasp
        self._top = top

    @staticmethod
    def _lerp(a, b, u):
        return a + (b - a) * float(np.clip(u, 0.0, 1.0))

    def _phase_of(self, t):
        for name, (start, end) in self._phase_bounds.items():
            if start <= t < end:
                return name, (t - start) / max(1, end - start)
        return "drop", 1.0

    def __call__(self, t):
        phase, u = self._phase_of(t)
        q = self._grasp_quat
        if phase == "settle":
            p, grip = self._pre, self._state.A.open_grip
        elif phase == "descend":
            p, grip = self._lerp(self._pre, self._grasp, u), self._state.A.open_grip
        elif phase == "close":
            p, grip = self._grasp, self._lerp(
                self._state.A.open_grip, self._state.A.close_grip, u
            )
        elif phase == "lift":
            p, grip = self._lerp(self._grasp, self._top, u), self._state.A.close_grip
        elif phase == "hold":
            p, grip = self._top, self._state.A.close_grip
        elif phase == "open":
            p, grip = self._top, self._lerp(
                self._state.A.close_grip, self._state.A.open_grip, u
            )
        else:
            p, grip = self._top, self._state.A.open_grip
        return p, q, float(grip), phase


__all__ = ["GraspTestScript"]
