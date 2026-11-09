# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Minimal OpenPI configuration for the YUBI rollout policy."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import difflib

from typing_extensions import override

import openpi.models.pi0_config as pi0_config
import openpi.training.config as base_config
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as transforms

from . import yubi_policy


@dataclasses.dataclass(frozen=True)
class LeRobotYubiDataConfig(base_config.DataConfigFactory):
    """Map the YUBI LeRobot schema to OpenPI model inputs and outputs."""

    default_prompt: str | None = None
    action_sequence_keys: Sequence[str] = (
        "observation.pose.left_hand_root.relative",
        "observation.pose.right_hand_root.relative",
        "action.joint_states",
    )

    def resolve_norm_stats_asset_id(self) -> str | None:
        """Return ``create_base_config``'s asset id without building transforms."""

        asset_id = self.assets.asset_id or self.repo_id
        return asset_id if isinstance(asset_id, str) else None

    @override
    def create(
        self,
        assets_dirs,
        model_config,
    ) -> base_config.DataConfig:
        repack = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation.image.left": "observation.image.left",
                        "observation.image.right": "observation.image.right",
                        (
                            "observation.pose."
                            "left_hand_root_to_right_hand_root.absolute"
                        ): (
                            "observation.pose."
                            "left_hand_root_to_right_hand_root.absolute"
                        ),
                        "observation.pose.left_hand_root.relative": (
                            "observation.pose.left_hand_root.relative"
                        ),
                        "observation.pose.right_hand_root.relative": (
                            "observation.pose.right_hand_root.relative"
                        ),
                        "observation.joint_states": "observation.joint_states",
                        "action.joint_states": "action.joint_states",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = transforms.Group(
            inputs=[yubi_policy.YubiInputs()],
            outputs=[yubi_policy.YubiOutputs()],
        )
        model_transforms = base_config.ModelTransformFactory(
            default_prompt=self.default_prompt
        )(model_config)
        configured = self.create_base_config(assets_dirs, model_config)
        expected_asset_id = self.resolve_norm_stats_asset_id()
        if configured.asset_id != expected_asset_id:
            raise ValueError(
                "YUBI lightweight norm-stats asset resolution disagrees with "
                f"DataConfig.asset_id ({expected_asset_id!r} != "
                f"{configured.asset_id!r})"
            )
        return dataclasses.replace(
            configured,
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


_PI05_YUBI_PICKPLACE_ASSET_ID = (
    "yubi/fix_actions_tf/frame_souce_table/"
    "socks_cup_penholder_box_merged"
)

# Norm-stats asset of the released cup checkpoint (pi05_cup). The checkpoint
# stores it at ``assets/pi05_cup/norm_stats.json``.
_PI05_CUP_ASSET_ID = "pi05_cup"

# The box baseline checkpoint (released as pi05_box_bs). Its prompt.yaml
# records config_name pi05_box_bs and this repo_id; the box instructions
# this repository pins appear in its task table verbatim.
_PI05_BOX_BS_ASSET_ID = "pi05_box_bs"

# The contact-dimension box checkpoint (pi05_box_ct): the released archive
# stores its norm stats at ``assets/pi05_box_ct/norm_stats.json``.
_PI05_BOX_CT_ASSET_ID = "pi05_box_ct"


_CONFIGS = {
    "pi05_yubi_pickplace": base_config.TrainConfig(
        name="pi05_yubi_pickplace",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
        ),
        data=LeRobotYubiDataConfig(
            repo_id=_PI05_YUBI_PICKPLACE_ASSET_ID,
            assets=base_config.AssetsConfig(
                asset_id=_PI05_YUBI_PICKPLACE_ASSET_ID,
            ),
            base_config=base_config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=200_000,
        batch_size=64,
        num_workers=8,
    ),
    "pi05_cup": base_config.TrainConfig(
        name="pi05_cup",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
        ),
        data=LeRobotYubiDataConfig(
            repo_id=_PI05_CUP_ASSET_ID,
            assets=base_config.AssetsConfig(
                asset_id=_PI05_CUP_ASSET_ID,
            ),
            base_config=base_config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=10_000,
        batch_size=8,
        num_workers=8,
    ),
    "pi05_box_bs": base_config.TrainConfig(
        name="pi05_box_bs",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
        ),
        data=LeRobotYubiDataConfig(
            repo_id=_PI05_BOX_BS_ASSET_ID,
            assets=base_config.AssetsConfig(
                asset_id=_PI05_BOX_BS_ASSET_ID,
            ),
            base_config=base_config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=150_000,
        batch_size=64,
        num_workers=8,
    ),
    # The contact-dimension box checkpoint: the two-camera pi0.5 stack with
    # action horizon 32, against the norm stats that run wrote.
    "pi05_box_ct": base_config.TrainConfig(
        name="pi05_box_ct",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
        ),
        data=LeRobotYubiDataConfig(
            repo_id=_PI05_BOX_CT_ASSET_ID,
            assets=base_config.AssetsConfig(
                asset_id=_PI05_BOX_CT_ASSET_ID,
            ),
            base_config=base_config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=150_000,
        batch_size=64,
        num_workers=8,
    ),
}


def get_config(config_name: str) -> base_config.TrainConfig:
    """Return one rollout configuration with an actionable unknown-name error."""

    try:
        return _CONFIGS[config_name]
    except KeyError as exc:
        suggestions = difflib.get_close_matches(
            config_name, _CONFIGS.keys(), n=1
        )
        hint = f" Did you mean {suggestions[0]!r}?" if suggestions else ""
        raise ValueError(
            f"Unknown YUBI config {config_name!r}.{hint}"
        ) from exc
