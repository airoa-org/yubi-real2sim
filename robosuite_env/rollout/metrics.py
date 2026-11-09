# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Per-frame geometry metrics required by rollout logs and summaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from robosuite_env.rendering.wrist_render_profiles import get_wrist_render_profile


def _rounded(values: Any, digits: int = 5) -> list[float]:
    return [round(float(value), digits) for value in np.asarray(values).ravel()]


@dataclass
class FrameMetricsCollector:
    """Collect the stable per-frame geometry schema without worker globals."""

    mujoco: Any
    model: Any
    data: Any
    arm: Mapping[str, Mapping[str, Any]]
    cup_qpos_address: int
    plate_qpos_address: int | None
    render_profile_name: str
    eef_pose: Callable[[str], tuple[np.ndarray, np.ndarray]]
    policy_tcp_pose: Callable[[str], tuple[np.ndarray, np.ndarray]]

    def collect(
        self,
        targets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, Any]:
        data = self.data
        cup_address = int(self.cup_qpos_address)
        cup_position = data.qpos[cup_address : cup_address + 3].copy()
        output: dict[str, Any] = {
            "cup_pos": _rounded(cup_position),
            "arms_metric": {},
        }
        if cup_address + 7 <= len(data.qpos):
            output["cup_quat"] = _rounded(
                data.qpos[cup_address + 3 : cup_address + 7].copy()
            )
        if self.plate_qpos_address is not None:
            plate_address = int(self.plate_qpos_address)
            output["plate_pos"] = _rounded(
                data.qpos[plate_address : plate_address + 3].copy()
            )
            if plate_address + 7 <= len(data.qpos):
                output["plate_quat"] = _rounded(
                    data.qpos[plate_address + 3 : plate_address + 7].copy()
                )

        camera_profile = get_wrist_render_profile(self.render_profile_name)
        for side in ("R", "L"):
            eef_position, eef_quaternion = self.eef_pose(side)
            tcp_position, tcp_quaternion = self.policy_tcp_pose(side)
            target_position = np.asarray(targets[side][0], dtype=float)
            camera_name = (
                f"{self.arm[side]['gripper_prefix']}_wrist_fisheye_front"
            )
            camera_id = self.mujoco.mj_name2id(
                self.model,
                self.mujoco.mjtObj.mjOBJ_CAMERA,
                camera_name,
            )
            camera_pose = None
            if camera_id >= 0:
                camera_rotation = np.asarray(
                    data.cam_xmat[camera_id], dtype=np.float64
                ).reshape(3, 3)
                camera_forward = -camera_rotation[:, 2]
                camera_position = np.asarray(
                    data.cam_xpos[camera_id], dtype=np.float64
                ).copy()
                camera_position -= (
                    camera_forward * camera_profile.camera_backward_m
                )
                camera_pose = {
                    "name": camera_name,
                    "position": _rounded(camera_position),
                    "right": _rounded(camera_rotation[:, 0]),
                    "up": _rounded(camera_rotation[:, 1]),
                    "forward": _rounded(camera_forward),
                }
            metrics = {
                "tcp_pos": _rounded(tcp_position),
                "eef_pos": _rounded(eef_position),
                "tcp_quat": _rounded(tcp_quaternion),
                "eef_quat": _rounded(eef_quaternion),
                "tcp_cup_dist_mm": round(
                    float(np.linalg.norm(tcp_position - cup_position)) * 1000.0,
                    3,
                ),
                "eef_cup_dist_mm": round(
                    float(np.linalg.norm(eef_position - cup_position)) * 1000.0,
                    3,
                ),
                "target_tcp_res_mm": round(
                    float(np.linalg.norm(tcp_position - target_position))
                    * 1000.0,
                    3,
                ),
                "target_eef_res_mm": round(
                    float(np.linalg.norm(eef_position - target_position))
                    * 1000.0,
                    3,
                ),
                "target_pos": _rounded(target_position),
            }
            if camera_pose is not None:
                metrics["wrist_front_camera"] = camera_pose
            output["arms_metric"][side] = metrics
        return output


__all__ = ["FrameMetricsCollector"]
