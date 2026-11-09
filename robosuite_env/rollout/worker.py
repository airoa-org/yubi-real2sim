#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Production FR3 rollout worker entry point."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
_ROLLOUT_DIR = os.path.dirname(os.path.abspath(__file__))
_DIR = os.path.dirname(_ROLLOUT_DIR)
REPO = os.path.dirname(_DIR)
_OPENPI_CACHE = os.path.join(REPO, "runtime-cache", "openpi")
if os.path.isdir(_OPENPI_CACHE):
    os.environ.setdefault("OPENPI_DATA_HOME", _OPENPI_CACHE)
# OpenPI, openpi_client and the YUBI deploy config import from the installed,
# locked environment; the policy stage refuses anything else before a policy
# is loaded or a batch server is contacted.

import numpy as np  # noqa: E402,F401
import mujoco  # noqa: E402,F401
import cv2 as cv  # noqa: E402

cv.setNumThreads(
    max(
        1,
        int(
            os.environ.get(
                "OPENCV_FOR_THREADS_NUM", os.environ.get("OMP_NUM_THREADS", "1")
            )
        ),
    )
)

from robosuite_env.rollout.build.controllers import run as setup_controllers  # noqa: E402
from robosuite_env.rollout.build.model import run as build_model_env  # noqa: E402
from robosuite_env.rollout.session import RolloutSession  # noqa: E402
from robosuite_env.rollout.setup.args import run as validate_args  # noqa: E402
from robosuite_env.rollout.stages.finalize_outputs import run as finalize_outputs  # noqa: E402
from robosuite_env.rollout.stages.init_policy_render import run as init_policy_render  # noqa: E402
from robosuite_env.rollout.stages.run_frames import run as run_frames  # noqa: E402
from robosuite_env.rollout.stages.settle_and_ready import run as settle_and_ready  # noqa: E402
from robosuite_env.rollout.stages.setup_grasp_and_gs import run as setup_grasp_and_gs  # noqa: E402
from real2sim.locked_imports import CHEAP, ROBOSUITE_PACKAGES, require_locked_imports  # noqa: E402


def main():
    """Run the complete rollout lifecycle in its established stage order."""

    session = RolloutSession()
    session._ROLLOUT_DIR = _ROLLOUT_DIR
    session._DIR = _DIR
    session.REPO = REPO
    validate_args(session)
    # robosuite must be the installed, locked distribution before any model is
    # built. The worker runs the cheap check (the imported file is one RECORD
    # lists, direct_url.json names the locked commit); the launcher preflight
    # ran the full one. sim_summary.json reports the records with the policy
    # stage's OpenPI records.
    session.LOCKED_IMPORTS = [
        {**record, "checked_in": "rollout worker"}
        for record in require_locked_imports(REPO, ROBOSUITE_PACKAGES, mode=CHEAP)
    ]
    build_model_env(session)
    setup_controllers(session)
    setup_grasp_and_gs(session)
    settle_and_ready(session)
    init_policy_render(session)
    run_frames(session)
    finalize_outputs(session)


if __name__ == "__main__":
    main()
