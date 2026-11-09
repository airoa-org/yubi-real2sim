#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""FR3 (Franka Research 3) as a robosuite robot, built from this repository's FR3 model. Default gripper = YUBI."""
import os

import numpy as np

from ...paths import ASSETS_ROOT, CONTROLLERS_ROOT
from .joint_gains import apply_fr3_joint_gain_overrides
from ..yubi import model as _yubi_model  # noqa: F401 -- register YUBI models
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.robots import register_robot_class

FR3_CONTROLLER_CONFIGS = {
    "joint_ik": str(CONTROLLERS_ROOT / "default_fr3_joint_ik.json"),
    "cartesian_impedance": str(CONTROLLERS_ROOT / "default_fr3_joint_ik.json"),  # Ready drive only.
}
# Canonical FR3 home posture (the ready posture of the deployed
# robot's initial-pose manager).  Keep this independent from rollout
# ready targets: Cartesian ready poses and scene-specific initial-joint JSONs
# are starting conditions, while this vector is the posture objective used to
# select among redundant IK solutions.
FR3_HOME_QPOS = np.array(
    [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398],
    dtype=np.float64,
)
FR3_HOME_QPOS.setflags(write=False)


def load_fr3_controller_config(
    mode="joint_ik",
    *,
    joint_kp: float | None = None,
    joint_kd: float | None = None,
):
    """Load a repo-local FR3 controller config.

    ``joint_ik`` accepts absolute seven-joint targets produced by the
    simulator's rate-independent whole-pose IK and trajectory generator.
    ``joint_kd`` is passed through robosuite's direct ``kd`` config key; it is
    not converted to ``damping_ratio``. If it is omitted, robosuite keeps the
    JSON behavior ``kd = 2 * sqrt(kp) * damping_ratio`` unchanged.
    """
    from robosuite.controllers import load_composite_controller_config

    try:
        controller_path = FR3_CONTROLLER_CONFIGS[mode]
    except KeyError as exc:
        raise ValueError(
            f"unknown FR3 controller mode {mode!r}; expected one of {sorted(FR3_CONTROLLER_CONFIGS)}"
        ) from exc
    config = load_composite_controller_config(controller=controller_path, robot="FR3")
    if joint_kp is not None or joint_kd is not None:
        apply_fr3_joint_gain_overrides(
            config,
            joint_kp=joint_kp,
            joint_kd=joint_kd,
        )
    return config


@register_robot_class("FixedBaseRobot")  # noqa: V102  # robosuite robot registry
class FR3(ManipulatorModel):
    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(str(ASSETS_ROOT / "fr3_robot.xml"), idn=idn)

    @property  # noqa: V106  # Robot.load_model reads it
    def default_base(self):
        # real rig: FR3 is clamped directly to the desk (no pedestal). NullMount = no base.
        # set FR3_BASE=RethinkMount to restore the robosuite floor pedestal.
        return os.environ.get("FR3_BASE", "NullMount")

    @property  # noqa: V106  # Robot.load_model reads it
    def default_gripper(self):
        return {"right": "YubiGripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_fr3"}   # repo-local config loaded via load_fr3_controller_config()

    @property  # noqa: V106  # Robot.load_model reads it
    def init_qpos(self):
        # Return a copy so controller code cannot mutate the shared home
        # posture while robosuite randomizes or initializes one robot.
        return FR3_HOME_QPOS.copy()

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property  # noqa: V106  # ManipulatorModel.__init__ reads it
    def arm_type(self):
        return "single"


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from ...tasks.envs.cup_plate import CupPlateLift
    cfg = load_fr3_controller_config()
    env = CupPlateLift(robots=["FR3", "FR3"], controller_configs=cfg,
                       has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, control_freq=20)
    env.reset()
    m = env.sim.model._model
    print("[fr3] LOADED. action_dim=", env.action_dim, "nq=", m.nq)
    print("[fr3] gripper geoms present:", any("fing" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") for g in range(m.ngeom)))
