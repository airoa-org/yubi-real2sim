#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""YubiCurrentLimitedPositionController — emulate the EXTERNAL behaviour of the real YUBI gripper
(Dynamixel Operating Mode 5: Current-based Position Control) in MuJoCo.

Pipeline (per spec):
  policy action -> normalized_width|user_angle_rad -> per-side user goal
  -> command-rate sample-and-hold -> optional delay queue -> optional motion profile
  -> position PID (anti-windup) -> clip to current-derived torque limit -> (optional 1st-order lag)
  -> torque to the MuJoCo MOTOR actuator(s).

Contact behaviour is emergent: on contact the position error persists, tau saturates at the limit,
and the MuJoCo contact solver balances it -> the jaw stops short of the goal WITHOUT any joint
freeze / weld / target-reset / torque-off. Remove the object and the jaw resumes toward the goal.

Per-hand state is fully isolated (target, integrator, delay queue, telemetry).
"""
from __future__ import annotations
import math
from collections import deque
from .config import YubiConfig, SideConfig


class YubiHandController:
    """Single hand. Works in USER motor-space (0 rad = closed, + = open)."""

    def __init__(self, cfg: YubiConfig, side: str, control_dt: float, warn=print):
        assert side in ("left", "right")
        self.cfg = cfg
        self.side = side
        self.s: SideConfig = cfg.left if side == "left" else cfg.right
        self.dt = float(control_dt)            # physics/control step the controller is ticked at
        self.warn = warn
        self.reset()

    # ---------- lifecycle ----------
    def reset(self, q_user0: float = 0.0):
        self._held_goal = float(q_user0)
        self._delay = deque()                  # (apply_time, q_user_goal)
        self._integ = 0.0
        self._q_ref = float(q_user0)
        self._qd_ref = 0.0
        self._tau_lag = 0.0
        self._last_action = None
        self.telemetry = {}

    # ---------- action -> user goal ----------
    def action_to_user_goal(self, action: float) -> float:
        m = self.cfg.action_mode
        if m == "normalized_width":
            u = min(1.0, max(0.0, float(action)))
            return u * self.s.max_open_user_rad
        if m == "user_angle_rad":
            return float(action)
        raise ValueError(f"unknown action_mode {m!r}")

    def set_action(self, action: float, t: float):
        """Apply a NEW policy command at time t (call at command_rate; sample-and-hold handled here)."""
        self._last_action = action
        g = self.action_to_user_goal(action)
        lo, hi = 0.0, self.s.max_open_user_rad     # safe user range
        if g < lo - 1e-9 or g > hi + 1e-9:
            self.warn(f"[yubi:{self.side}] user goal {g:.3f} out of safe range [{lo},{hi}] -> clipped")
        g = min(hi, max(lo, g))
        self._delay.append((t + self.s.command_delay_s, g))

    # ---------- internal ----------
    def _pop_delayed(self, t: float) -> float:
        while self._delay and self._delay[0][0] <= t + 1e-12:
            self._held_goal = self._delay.popleft()[1]
        return self._held_goal

    def _profile(self, q_goal: float):
        c = self.cfg.controller
        if c.velocity_limit_rad_s is None:        # disabled/unlimited
            self._q_ref = q_goal; self._qd_ref = 0.0; return
        vmax = c.velocity_limit_rad_s
        dq = q_goal - self._q_ref
        if c.acceleration_limit_rad_s2 is None:   # velocity-limited
            step = max(-vmax * self.dt, min(vmax * self.dt, dq))
            self._q_ref += step; self._qd_ref = step / self.dt
        else:                                      # velocity-and-acceleration-limited
            amax = c.acceleration_limit_rad_s2
            v_des = max(-vmax, min(vmax, dq / self.dt))
            dv = max(-amax * self.dt, min(amax * self.dt, v_des - self._qd_ref))
            self._qd_ref = max(-vmax, min(vmax, self._qd_ref + dv))
            self._q_ref += self._qd_ref * self.dt

    # ---------- main: torque for this step ----------
    def compute_torque(self, t: float, q_user: float, qd_user: float) -> float:
        c = self.cfg.controller
        if c.kp is None or c.kd is None:
            raise RuntimeError("controller.kp/kd are uncalibrated (None) - set them or use a calibrated config")
        goal = self._pop_delayed(t)
        self._profile(goal)
        e = self._q_ref - q_user
        edot = self._qd_ref - qd_user
        # integral (anti-windup applied after saturation test)
        if c.ki:
            self._integ += e * self.dt
        tau_raw = c.kp * e + c.kd * edot + c.ki * self._integ
        opening = e > 0.0                          # user-space: increasing q = opening
        lim = self.cfg.resolve_torque_limit(opening=opening)
        if lim is None:
            raise RuntimeError("torque limit uncalibrated - set torque_limit_Nm or effective_torque_per_amp")
        tau_clip = max(-lim, min(lim, tau_raw))
        saturated = abs(tau_raw) > lim + 1e-12
        # anti-windup: stop integrating when clamped & integral drives further into the clamp
        if c.ki and saturated and (math.copysign(1, tau_raw) == math.copysign(1, e)):
            self._integ -= e * self.dt
            ilim = c.integral_limit
            if ilim is not None:
                self._integ = max(-ilim, min(ilim, self._integ))
        # 1st-order current/torque response lag
        if c.current_response_time_constant_s and c.current_response_time_constant_s > 0:
            a = self.dt / (c.current_response_time_constant_s + self.dt)
            self._tau_lag += a * (tau_clip - self._tau_lag)
            tau = self._tau_lag
        else:
            tau = tau_clip
        etpa = c.effective_torque_per_amp
        self.telemetry = dict(
            t=t, action=self._last_action, q_user_goal=goal, q_ref=self._q_ref, qd_ref=self._qd_ref,
            q=q_user, qd=qd_user, e=e, tau_raw=tau_raw, tau_cmd=tau,
            saturated=saturated, sat_ratio=abs(tau_raw) / lim if lim else float("nan"),
            integ=self._integ, current_equiv_A=(tau / etpa) if etpa else None)
        return tau


class BimanualYubiController:
    """Two fully-independent hand controllers (separate target / integrator / delay / telemetry)."""

    def __init__(self, cfg: YubiConfig, control_dt: float, warn=print):
        self.cfg = cfg
        self.left = YubiHandController(cfg, "left", control_dt, warn)
        self.right = YubiHandController(cfg, "right", control_dt, warn)

    def reset(self, q_user0=0.0):
        self.left.reset(q_user0); self.right.reset(q_user0)

    def telemetry(self):
        return {"left": self.left.telemetry, "right": self.right.telemetry}
