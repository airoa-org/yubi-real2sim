#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Robosuite integration for the current-limited YUBI torque controller."""
from __future__ import annotations

from collections.abc import Mapping


class DynamicYubiQfrcHook:
    """Apply one CbPC update immediately before every MuJoCo physics step."""

    _SIDES = (("R", "right"), ("L", "left"))

    def __init__(self, original_pre_action, data, arms, controller, expected_dt=None):
        self.original_pre_action = original_pre_action
        self.data = data
        self.arms = arms
        self.controller = controller
        self.expected_dt = float(expected_dt) if expected_dt is not None else None
        self.tick_count = 0
        self.target_update_count = 0
        self.cadence_violation_count = 0
        self.last_time = None
        self.latest_torque = {"R": 0.0, "L": 0.0}
        self._samples = []

    def _finger_state(self, key):
        refs = self.arms[key]
        q = sum(float(self.data.qpos[j]) for j in refs["fj"]) / len(refs["fj"])
        qd = sum(float(self.data.qvel[j]) for j in refs["fd"]) / len(refs["fd"])
        return q, qd

    def set_targets(self, targets: Mapping[str, float], t=None):
        """Queue one source-action target update for each supplied hand."""
        now = float(self.data.time if t is None else t)
        for key, side in self._SIDES:
            if key in targets:
                getattr(self.controller, side).set_action(float(targets[key]), now)
        self.target_update_count += 1

    def apply(self, t=None):
        """Compute and assign finger forces for the next MuJoCo step."""
        now = float(self.data.time if t is None else t)
        if self.last_time is not None and self.expected_dt is not None:
            delta = now - self.last_time
            tol = max(1e-9, abs(self.expected_dt) * 1e-6)
            if delta < -tol or abs(delta - self.expected_dt) > tol:
                self.cadence_violation_count += 1
        self.last_time = now

        sample = {"sim_time_s": now, "sides": {}}
        for key, side in self._SIDES:
            hand = getattr(self.controller, side)
            q, qd = self._finger_state(key)
            tau = float(hand.compute_torque(now, q, qd))
            dofs = self.arms[key]["fd"]
            jaw_tau = tau / len(dofs)
            for dof in dofs:
                self.data.qfrc_applied[dof] = jaw_tau
            self.latest_torque[key] = tau

            telemetry = hand.telemetry
            torque_limit = hand.cfg.resolve_torque_limit(opening=float(telemetry.get("e", 0.0)) > 0.0)
            sample["sides"][key] = {
                "q": q,
                "qd": qd,
                "goal": float(telemetry.get("q_user_goal", q)),
                "q_ref": float(telemetry.get("q_ref", q)),
                "error": float(telemetry.get("e", 0.0)),
                "tau_raw": float(telemetry.get("tau_raw", tau)),
                "tau_cmd": tau,
                "jaw_tau": jaw_tau,
                "torque_limit": None if torque_limit is None else float(torque_limit),
                "current_equiv_A": telemetry.get("current_equiv_A"),
                "saturated": bool(telemetry.get("saturated", False)),
                "sat_ratio": float(telemetry.get("sat_ratio", 0.0)),
            }
        self._samples.append(sample)
        self.tick_count += 1

    def __call__(self, action, policy_step=False):
        self.original_pre_action(action, policy_step)
        self.apply()

    def drain_samples(self):
        samples = self._samples
        self._samples = []
        return samples

    def reset_timing(self, clear_samples=False, reset_counts=False):
        self.last_time = None
        self.cadence_violation_count = 0
        if reset_counts:
            self.tick_count = 0
            self.target_update_count = 0
        if clear_samples:
            self._samples = []
