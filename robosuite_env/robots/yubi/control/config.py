#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Config for the YUBI current-limited position controller (Dynamixel Current-based Position Mode
emulation). Every parameter carries a SOURCE classification:

  REPO_CONFIRMED            - fixed by the real-robot repo (do not change)
  DERIVED_FROM_REPO         - computed from repo-confirmed values
  REAL_MEASUREMENT_REQUIRED - must be measured on hardware before calibrated use
  SIM_TUNED                 - tuned in MuJoCo against measured target curves
  UNCALIBRATED_PLACEHOLDER  - provisional, NOT calibrated; running calibrated modes must error out

`null`/None means the value is not yet known. Modes that need it must raise, never silently guess.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json

SOURCE = {  # parameter -> source classification (for the final report / audit)
    "current_limit_A": "REPO_CONFIRMED",          # bimanual Goal Current = 400 mA
    "direction": "REPO_CONFIRMED",
    "position_offset_rad": "REPO_CONFIRMED",       # L 0.120, R 0.186 (real raw<->user)
    "max_open_user_rad": "REPO_CONFIRMED",         # L 0.8, R 1.0
    "command_rate_hz": "REPO_CONFIRMED",           # 10 Hz direct / 15 Hz alt runner
    "feedback_rate_hz": "REPO_CONFIRMED",          # 60 Hz
    "kp": "SIM_TUNED",                             # Dynamixel internal P unknown -> tune in sim
    "kd": "SIM_TUNED",
    "ki": "SIM_TUNED",
    "effective_torque_per_amp": "REAL_MEASUREMENT_REQUIRED",
    "torque_limit_Nm": "UNCALIBRATED_PLACEHOLDER",
    "velocity_limit_rad_s": "REAL_MEASUREMENT_REQUIRED",
    "acceleration_limit_rad_s2": "REAL_MEASUREMENT_REQUIRED",
    "current_response_time_constant_s": "REAL_MEASUREMENT_REQUIRED",
    "command_delay_s": "REAL_MEASUREMENT_REQUIRED",
    "user_to_joint_scale": "SIM_TUNED",            # maps user motor angle -> MuJoCo finger-joint angle
}


@dataclass
class SideConfig:
    # --- REPO_CONFIRMED real-robot values ---
    direction: float = -1.0
    position_offset_rad: float = 0.120     # left default; right=0.186
    max_open_user_rad: float = 0.8         # left default; right=1.0
    # --- REAL_MEASUREMENT_REQUIRED ---
    command_delay_s: float = 0.0           # shared-bus / motion-start delay (0 until measured)
    # --- model wiring (which MuJoCo joints/actuators this hand drives) ---
    motor_joints: list = field(default_factory=list)      # finger joint names (mirrored jaws)
    motor_actuators: list = field(default_factory=list)   # corresponding motor-actuator names
    user_to_joint_scale: float = 1.0       # q_joint = clip(scale * q_user, 0, joint_max); SIM_TUNED
    joint_max_rad: float = 0.7854          # MuJoCo finger-joint upper range


@dataclass
class ControllerConfig:
    kp: float | None = None
    ki: float = 0.0
    kd: float | None = None
    # current->torque. Provide ONE of: torque_limit_Nm (direct) OR effective_torque_per_amp.
    torque_limit_Nm: float | None = None
    effective_torque_per_amp: float | None = None
    # correction factor s_tau (spec §4.4): tau_max = base * torque_scale. Lumps gear efficiency /
    # voltage drop / mechanism friction / sim grip-force tuning. 1.0 = nominal. (applies to close/open too)
    torque_scale: float = 1.0
    # optional asymmetric clip (close/open). None -> use the symmetric resolved limit for both.
    torque_limit_close_Nm: float | None = None
    torque_limit_open_Nm: float | None = None
    # motion profile (None disables that stage)
    velocity_limit_rad_s: float | None = None
    acceleration_limit_rad_s2: float | None = None
    # 1st-order current/torque response lag (0 = instantaneous)
    current_response_time_constant_s: float = 0.0
    # anti-windup integral clamp (rad*s); None -> auto (= torque_limit/ki when ki>0)
    integral_limit: float | None = None


@dataclass
class YubiConfig:
    action_mode: str = "normalized_width"          # "normalized_width" | "user_angle_rad"
    command_rate_hz: float = 10.0
    feedback_rate_hz: float = 60.0
    current_limit_A: float = 0.4                    # REPO_CONFIRMED Goal Current
    strict_real_driver_wrap_compat: bool = False    # do NOT mimic raw angle-wrap by default
    model_joint_space: str = "user"                # "user" (MuJoCo joint = user-space) | "raw"
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    left: SideConfig = field(default_factory=lambda: SideConfig(
        direction=-1.0, position_offset_rad=0.120, max_open_user_rad=0.8,
        motor_joints=["L_finger_joint_left", "L_finger_joint_right"],
        motor_actuators=["L_finger_l", "L_finger_r"]))
    right: SideConfig = field(default_factory=lambda: SideConfig(
        direction=-1.0, position_offset_rad=0.186, max_open_user_rad=1.0,
        motor_joints=["R_finger_joint_left", "R_finger_joint_right"],
        motor_actuators=["R_finger_l", "R_finger_r"]))

    def resolve_torque_limit(self, opening: bool) -> float | None:
        """tau limit [Nm] for the given direction. None => UNCALIBRATED (caller must error)."""
        c = self.controller
        s = c.torque_scale if c.torque_scale is not None else 1.0
        base = c.torque_limit_Nm
        if base is None and c.effective_torque_per_amp is not None:
            base = c.effective_torque_per_amp * self.current_limit_A
        if opening and c.torque_limit_open_Nm is not None:
            return c.torque_limit_open_Nm * s
        if (not opening) and c.torque_limit_close_Nm is not None:
            return c.torque_limit_close_Nm * s
        return base * s if base is not None else None

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "YubiConfig":
        d = dict(d)
        ctrl = ControllerConfig(**d.pop("controller", {}))
        left = SideConfig(**d.pop("left", {})); right = SideConfig(**d.pop("right", {}))
        return YubiConfig(controller=ctrl, left=left, right=right, **d)

    @staticmethod
    def load(path: str) -> "YubiConfig":
        with open(path) as f:
            if path.endswith((".yaml", ".yml")):
                import yaml; d = yaml.safe_load(f)
            else:
                d = json.load(f)
        return YubiConfig.from_dict(d.get("yubi", d))

    def save(self, path: str):
        d = {"yubi": self.to_dict()}
        with open(path, "w") as f:
            if path.endswith((".yaml", ".yml")):
                import yaml; yaml.safe_dump(d, f, sort_keys=False)
            else:
                json.dump(d, f, indent=2)


def needs_calibration(cfg: YubiConfig) -> list:
    """Return the list of required-but-missing calibrated parameters (empty => runnable)."""
    miss = []
    c = cfg.controller
    if c.kp is None: miss.append("controller.kp")
    if c.kd is None: miss.append("controller.kd")
    if cfg.resolve_torque_limit(opening=True) is None or cfg.resolve_torque_limit(opening=False) is None:
        miss.append("controller.torque_limit_Nm OR effective_torque_per_amp")
    return miss
