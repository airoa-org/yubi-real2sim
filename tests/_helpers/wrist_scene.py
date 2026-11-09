# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Production YUBI geometry used by CPU wrist-renderer contract tests.

Keep camera-constructor coverage independent of any replay CLI or data reader.
Only the scene fixture lives here; rendering and pixel composition are production code.
"""
from pathlib import Path
from types import SimpleNamespace

import robosuite_env


def build_wrist_test_env():
    import mujoco

    assets = Path(robosuite_env.__file__).parent / "assets"
    spec = mujoco.MjSpec()
    spec.modelname = "wrist_contract_test"
    spec.option.gravity = [0.0, 0.0, 0.0]
    spec.visual.global_.offwidth = 1024
    spec.visual.global_.offheight = 1024
    for side in ("left", "right"):
        hand = mujoco.MjSpec.from_file(str(assets / f"yubi_{side}.xml"))
        mount = spec.worldbody.add_body(name=f"{side}_mount", mocap=True)
        mount.add_frame().attach_body(hand.body("right_gripper"), f"gripper0_{side}_", "")
    model = spec.compile()
    model.stat.extent = 1.0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(sim=SimpleNamespace(
        model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data),
    ))
