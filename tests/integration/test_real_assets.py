# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Integration tests that require installed real input or robot assets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robosuite_env.contracts.scene import (
    TABLE_Z_METHOD,
    TABLE_Z_TOLERANCE_M,
    measure_gs_table_plane_z,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_real_public_scene_table_plane_sits_at_runtime_table_z(
    real_inputs_root: Path,
) -> None:
    """Regression on an installed public scene's baked table height."""

    scene = real_inputs_root / "scenes" / "flat_reading_room"
    manifest = scene / "background.json"
    if not manifest.is_file():
        pytest.skip("public scene flat_reading_room is not installed")

    spec = json.loads(manifest.read_text(encoding="utf-8"))
    ply = scene / spec["ply_path"]
    if "mat_w2gs" in spec:
        baked = np.asarray(spec["mat_w2gs"], dtype=np.float64)
    else:
        baked = np.load(scene / spec["w2gs"])

    measured = measure_gs_table_plane_z(ply, baked)
    assert -TABLE_Z_TOLERANCE_M <= measured <= TABLE_Z_TOLERANCE_M
    assert abs(measured - spec["calibration_table_z_m"]) <= TABLE_Z_TOLERANCE_M
    assert spec["calibration_table_z_method"] == TABLE_Z_METHOD


def test_real_fr3_escapes_joint_limit_seed_via_home_restart(
    real_inputs_root: Path,
    mujoco,
) -> None:
    # Requiring real_inputs_root records this as an assets test even though the
    # installed FR3 bundle itself is mirrored into the repository asset root.
    assert real_inputs_root.is_dir()

    from robosuite_env.robots.fr3.absolute_ik import AbsoluteFR3IK
    from robosuite_env.robots.fr3.model import FR3_HOME_QPOS

    asset_root = REPO_ROOT / "robosuite_env" / "assets"
    if not (asset_root / "fr3_collision" / "link7.stl").is_file():
        pytest.skip("optional FR3 collision asset bundle is not installed")
    model = mujoco.MjModel.from_xml_path(str(asset_root / "fr3_robot.xml"))
    joint_ids = np.asarray(
        [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index}"
            )
            for index in range(1, 8)
        ],
        dtype=int,
    )
    qpos_indices = model.jnt_qposadr[joint_ids].astype(int)
    dof_indices = model.jnt_dofadr[joint_ids].astype(int)
    eef_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "right_hand"
    )
    target_q = np.asarray(
        [
            -0.2319213,
            -0.50599167,
            0.13027652,
            -2.34556454,
            -0.09900418,
            1.33814939,
            0.19832539,
        ]
    )
    seed_q = np.asarray(
        [
            -0.22052811,
            -0.36498602,
            -0.19666453,
            -0.1169,
            -0.05536904,
            1.08767305,
            0.81220416,
        ]
    )
    target_data = mujoco.MjData(model)
    target_data.qpos[qpos_indices] = target_q
    mujoco.mj_forward(model, target_data)
    solver = AbsoluteFR3IK(
        model,
        {
            "R": {
                "qpos_indices": qpos_indices,
                "dof_indices": dof_indices,
                "joint_ranges": model.jnt_range[joint_ids].copy(),
                "eef_body_id": eef_body_id,
            }
        },
        np.zeros(3),
        iterations=100,
        nullspace_gain=0.2,
        home_q=FR3_HOME_QPOS,
    )

    result = solver.solve(
        "R",
        target_data.xpos[eef_body_id].copy(),
        target_data.xquat[eef_body_id].copy(),
        seed_q=seed_q,
        continuity_q=seed_q,
    )

    assert result.converged, result
    assert result.iterations > 0
    assert result.pos_error_mm < 0.5
    assert result.ori_error_deg < 0.5
    assert result.min_limit_margin_rad > 0.1
