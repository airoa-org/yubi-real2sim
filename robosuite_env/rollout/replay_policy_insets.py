# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Resolve exact saved policy wrist observations for state-log video replay.

This module is deliberately independent of the rollout worker so the replay
contract can be validated before any MuJoCo or EGL context is created.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import cv2
import numpy as np

from robosuite_env.policy.policy_io import (
    PolicyImageDisplayTransform,
    prepare_policy_wrist_display_image,
)
from robosuite_env.rendering.wrist_render_profiles import get_wrist_render_profile


class ReplayPolicyInsetError(ValueError):
    """The state log cannot be paired with a complete saved wrist sequence."""


@dataclass(frozen=True)
class ReplayPolicyInsetSchedule:
    """Exact policy images and both supported display forms for every frame."""

    frames: tuple[tuple[np.ndarray, np.ndarray], ...]
    raw_display_frames: tuple[tuple[np.ndarray, np.ndarray], ...]
    display_frames: tuple[tuple[np.ndarray, np.ndarray], ...]
    display_transforms: tuple[
        tuple[PolicyImageDisplayTransform, PolicyImageDisplayTransform], ...
    ]
    observation_frames: tuple[int, ...]
    unique_pairs: int


def _recorded_obs_log(row: Any, frame_index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ReplayPolicyInsetError(
            f"state-log frame {frame_index} is not a JSON object"
        )
    policy = row.get("policy")
    obs_log = policy.get("obs_log") if isinstance(policy, dict) else None
    if not isinstance(obs_log, dict):
        raise ReplayPolicyInsetError(
            f"state-log frame {frame_index} has no policy.obs_log object"
        )
    return obs_log


def _recorded_primary_path(row: Any, frame_index: int, side: str) -> str:
    obs_log = _recorded_obs_log(row, frame_index)
    paths = obs_log.get("paths") if isinstance(obs_log, dict) else None
    value = paths.get(f"primary_{side}") if isinstance(paths, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ReplayPolicyInsetError(
            f"state-log frame {frame_index} has no policy.obs_log.paths.primary_{side}"
        )
    return value


def _recorded_observation_frame(
    row: Any, obs_log: dict[str, Any], frame_index: int
) -> int:
    value = obs_log.get("observation_frame", obs_log.get("frame"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReplayPolicyInsetError(
            f"state-log frame {frame_index} has invalid saved observation "
            f"frame {value!r}"
        )
    if obs_log.get("frame", value) != value:
        raise ReplayPolicyInsetError(
            f"state-log frame {frame_index} obs_log.frame and "
            "obs_log.observation_frame disagree"
        )
    policy = row.get("policy") if isinstance(row, dict) else None
    active_value = policy.get("observation_frame") if isinstance(policy, dict) else None
    if active_value is not None and active_value != value:
        raise ReplayPolicyInsetError(
            f"state-log frame {frame_index} active observation frame "
            f"{active_value!r} differs from obs_log {value}"
        )
    return int(value)


def _validate_recorded_filename(
    recorded: str, *, side: str, observation_frame: int, frame_index: int
) -> None:
    name = Path(recorded).name
    match = re.fullmatch(
        rf"obs_{side}(?:_[A-Za-z0-9_-]+)?_f(\d+)(?:_i\d+)?\.png",
        name,
    )
    if match is None:
        raise ReplayPolicyInsetError(
            f"state-log frame {frame_index} primary_{side} has an invalid "
            f"saved-observation filename: {name!r}"
        )
    file_frame = int(match.group(1))
    if file_frame != observation_frame:
        raise ReplayPolicyInsetError(
            f"state-log frame {frame_index} primary_{side} filename frame "
            f"{file_frame} differs from obs_log observation frame "
            f"{observation_frame}"
        )


def _strict_source_hw(value: Any, *, frame_index: int, side: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) not in (2, 3):
        raise ReplayPolicyInsetError(
            f"state-log frame {frame_index} image_source_shapes.{side} must be "
            "[height,width] or [height,width,channels]"
        )
    height, width = value[:2]
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or height <= 0
        or width <= 0
    ):
        raise ReplayPolicyInsetError(
            f"state-log frame {frame_index} image_source_shapes.{side} has "
            f"invalid dimensions {value!r}"
        )
    if len(value) == 3 and value[2] != 3:
        raise ReplayPolicyInsetError(
            f"state-log frame {frame_index} image_source_shapes.{side} must "
            f"declare 3 RGB channels, got {value[2]!r}"
        )
    return int(height), int(width)


def _display_contract(
    obs_log: dict[str, Any], frame_index: int
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Resolve display geometry from logged facts, never image pixels."""

    resize_mode = obs_log.get("policy_image_resize")
    if resize_mode not in ("pad", "stretch"):
        raise ReplayPolicyInsetError(
            f"state-log frame {frame_index} has invalid or missing "
            f"policy_image_resize={resize_mode!r}"
        )

    shapes = obs_log.get("image_source_shapes")
    if shapes is not None:
        if not isinstance(shapes, dict):
            raise ReplayPolicyInsetError(
                f"state-log frame {frame_index} image_source_shapes must be an object"
            )
        left_hw = _strict_source_hw(
            shapes.get("L"), frame_index=frame_index, side="L"
        )
        right_hw = _strict_source_hw(
            shapes.get("R"), frame_index=frame_index, side="R"
        )
        return resize_mode, left_hw, right_hw

    # Backward-compatible, but still exact: historical simulated observations
    # identify the immutable renderer profile whose declared raster determines
    # the OpenPI padding. Unknown profiles and external image sources fail
    # closed because their aspect ratio cannot be reconstructed reliably.
    if obs_log.get("image_source") != "sim":
        raise ReplayPolicyInsetError(
            f"state-log frame {frame_index} has no image_source_shapes for "
            f"image_source={obs_log.get('image_source')!r}"
        )
    profile_name = obs_log.get("policy_yubi_render_profile")
    try:
        source_hw = get_wrist_render_profile(profile_name).source_image_hw
    except (TypeError, ValueError) as exc:
        raise ReplayPolicyInsetError(
            f"state-log frame {frame_index} cannot resolve wrist source raster "
            f"from profile {profile_name!r}: {exc}"
        ) from exc
    return resize_mode, source_hw, source_hw


def _resolve_recorded_path(policy_obs_dir: Path, recorded: str) -> Path:
    recorded_path = Path(recorded).expanduser()
    # The supplied directory is authoritative. This also makes copied or
    # relocated rollout roots work when the JSON retains an old absolute path.
    relocated = policy_obs_dir / recorded_path.name
    if relocated.is_file():
        return relocated.resolve()
    raise ReplayPolicyInsetError(
        f"saved policy wrist image is missing from authoritative directory: "
        f"{relocated} (recorded path is informational only: {recorded_path})"
    )


def _load_rgb_224(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ReplayPolicyInsetError(
            f"saved policy wrist image cannot be decoded: {path}"
        )
    if image_bgr.shape != (224, 224, 3):
        raise ReplayPolicyInsetError(
            "saved policy wrist image must be 224x224 RGB: "
            f"{path} has shape {tuple(image_bgr.shape)}"
        )
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb.setflags(write=False)
    return image_rgb


def load_replay_policy_insets(
    state_rows: Iterable[Any], policy_obs_dir: str | Path
) -> ReplayPolicyInsetSchedule:
    """Validate and decode the active saved policy wrist pair for every frame.

    Repeated frames share one cached pair, matching live ``--policy-obs-inset``
    behavior where the most recent policy observation is held until replan.
    """

    directory = Path(policy_obs_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ReplayPolicyInsetError(
            f"saved policy observation directory does not exist: {directory}"
        )

    rows = list(state_rows)
    if not rows:
        raise ReplayPolicyInsetError("state log is empty")

    image_cache: dict[Path, np.ndarray] = {}
    pair_cache: dict[tuple[Path, Path], tuple[np.ndarray, np.ndarray]] = {}
    raw_display_cache: dict[
        tuple[Path, Path], tuple[np.ndarray, np.ndarray]
    ] = {}
    display_cache: dict[
        tuple[Path, Path, str, tuple[int, int], tuple[int, int]],
        tuple[
            tuple[np.ndarray, np.ndarray],
            tuple[PolicyImageDisplayTransform, PolicyImageDisplayTransform],
        ],
    ] = {}
    pair_contracts: dict[
        tuple[Path, Path], tuple[str, tuple[int, int], tuple[int, int]]
    ] = {}
    frame_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    raw_display_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    display_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    display_transforms: list[
        tuple[PolicyImageDisplayTransform, PolicyImageDisplayTransform]
    ] = []
    observation_frames: list[int] = []
    for frame_index, row in enumerate(rows):
        obs_log = _recorded_obs_log(row, frame_index)
        observation_frame = _recorded_observation_frame(
            row, obs_log, frame_index
        )
        if observation_frame > frame_index:
            raise ReplayPolicyInsetError(
                f"state-log frame {frame_index} references future policy "
                f"observation frame {observation_frame}"
            )
        if observation_frames and observation_frame < observation_frames[-1]:
            raise ReplayPolicyInsetError(
                f"state-log frame {frame_index} moves policy observation "
                f"backwards from {observation_frames[-1]} to {observation_frame}"
            )
        left_recorded = _recorded_primary_path(row, frame_index, "L")
        right_recorded = _recorded_primary_path(row, frame_index, "R")
        _validate_recorded_filename(
            left_recorded,
            side="L",
            observation_frame=observation_frame,
            frame_index=frame_index,
        )
        _validate_recorded_filename(
            right_recorded,
            side="R",
            observation_frame=observation_frame,
            frame_index=frame_index,
        )
        left = _resolve_recorded_path(
            directory, left_recorded
        )
        right = _resolve_recorded_path(
            directory, right_recorded
        )
        pair_key = (left, right)
        pair = pair_cache.get(pair_key)
        if pair is None:
            for path in pair_key:
                if path not in image_cache:
                    image_cache[path] = _load_rgb_224(path)
            pair = (image_cache[left], image_cache[right])
            pair_cache[pair_key] = pair
        raw_display_pair = raw_display_cache.get(pair_key)
        if raw_display_pair is None:
            raw_display_pair = (
                cv2.resize(
                    pair[0],
                    (180, 135),
                    interpolation=cv2.INTER_LINEAR,
                ),
                cv2.resize(
                    pair[1],
                    (180, 135),
                    interpolation=cv2.INTER_LINEAR,
                ),
            )
            for image in raw_display_pair:
                image.setflags(write=False)
            raw_display_cache[pair_key] = raw_display_pair
        contract = _display_contract(obs_log, frame_index)
        previous_contract = pair_contracts.setdefault(pair_key, contract)
        if previous_contract != contract:
            raise ReplayPolicyInsetError(
                f"state-log frame {frame_index} changes display metadata for "
                f"saved pair {left.name}/{right.name}: "
                f"{previous_contract!r} != {contract!r}"
            )
        resize_mode, left_source_hw, right_source_hw = contract
        display_key = (
            left,
            right,
            resize_mode,
            left_source_hw,
            right_source_hw,
        )
        display_record = display_cache.get(display_key)
        if display_record is None:
            try:
                display_left, transform_left = prepare_policy_wrist_display_image(
                    pair[0], resize_mode=resize_mode, source_hw=left_source_hw
                )
                display_right, transform_right = prepare_policy_wrist_display_image(
                    pair[1], resize_mode=resize_mode, source_hw=right_source_hw
                )
            except ValueError as exc:
                raise ReplayPolicyInsetError(
                    f"state-log frame {frame_index} has invalid display geometry: {exc}"
                ) from exc
            display_left.setflags(write=False)
            display_right.setflags(write=False)
            display_record = (
                (display_left, display_right),
                (transform_left, transform_right),
            )
            display_cache[display_key] = display_record
        frame_pairs.append(pair)
        raw_display_pairs.append(raw_display_pair)
        display_pairs.append(display_record[0])
        display_transforms.append(display_record[1])
        observation_frames.append(observation_frame)

    return ReplayPolicyInsetSchedule(
        frames=tuple(frame_pairs),
        raw_display_frames=tuple(raw_display_pairs),
        display_frames=tuple(display_pairs),
        display_transforms=tuple(display_transforms),
        observation_frames=tuple(observation_frames),
        unique_pairs=len(pair_cache),
    )
