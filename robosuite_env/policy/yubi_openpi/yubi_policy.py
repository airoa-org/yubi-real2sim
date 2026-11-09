# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

# Portions of this file are adapted from openpi
# (https://github.com/Physical-Intelligence/openpi, commit e6b0441,
# src/openpi/policies/libero_policy.py), Copyright the openpi authors,
# licensed under the Apache License, Version 2.0. See THIRD_PARTY_NOTICES.
# _parse_image is copied from that file and changed by AIRoA to transpose only
# 3-dimensional channel-first arrays; YubiInputs and YubiOutputs follow its
# Inputs/Outputs transform pattern for the YUBI observation and action layout.

import dataclasses

import einops
import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class YubiInputs(transforms.DataTransformFn):
    """Inputs for the yubi policy.

    State = left-to-right relative pose (7) + finger joints (2), padded to ``action_dim``.
    Actions = concat(left.relative (7), right.relative (7), joints (2)) per timestep,
    padded to ``action_dim`` along the last axis.
    """

    action_dim: int = 32

    def __call__(self, data: dict) -> dict:
        left_image = _parse_image(data["observation.image.left"])
        right_image = _parse_image(data["observation.image.right"])

        rel_pose = np.asarray(data["observation.pose.left_hand_root_to_right_hand_root.absolute"], dtype=np.float32)

        joints = np.asarray(data["observation.joint_states"], dtype=np.float32)
        state = np.concatenate([rel_pose, joints], axis=-1)
        state = transforms.pad_to_dim(state, self.action_dim)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": np.zeros_like(left_image),
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Action sequence per timestep is built from the umi dataset's
        # per-step relative pose observations (which are TRUE deltas
        # dx,dy,dz,dqx,dqy,dqz,dqw) and the finger joint actions.
        if "observation.pose.left_hand_root.relative" in data:
            left_act = np.asarray(data["observation.pose.left_hand_root.relative"], dtype=np.float32)
            right_act = np.asarray(data["observation.pose.right_hand_root.relative"], dtype=np.float32)
            joint_act = np.asarray(data["action.joint_states"], dtype=np.float32)
            actions = np.concatenate([left_act, right_act, joint_act], axis=-1)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim, axis=-1)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class YubiOutputs(transforms.DataTransformFn):
    """Outputs for the yubi policy.

    Returns a flat ``actions`` array (not split per-key) so the deployed
    policy runtime's action extraction can consume it without changes.
    Layout per timestep: ``[left_relative(7), right_relative(7), finger(2)]``.
    The publish-side ``policy.action.output_names`` in the deployment
    configuration must match this ordering.
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        # Drop the model's internal padding back to the 16-dim publish vector.
        return {"actions": actions[:, :16]}
