# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Deterministic rollout quiet-window gate for free task objects.

The simulator advances MuJoCo at the arm controller rate while task objects
settle under gravity.  This module contains the small, simulator-independent
part of that gate so threshold handling and consecutive-sample semantics can
be unit tested without constructing the full dual-FR3 model.  Instantaneous
MuJoCo contact accelerations remain available as telemetry, but callers may
exclude them from acceptance because solver chatter is not net object motion.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Mapping, Sequence, TypedDict

import numpy as np


STATIC_DESK_SUPPORT = "static_desk"


class ObjectSupportResolution(TypedDict):
    """Reachability and one canonical path to the static desk."""

    supported: bool
    support_path: tuple[str, ...]


def resolve_transitive_support(
    object_names: Sequence[str],
    directly_supported: Iterable[str],
    adjacency: Mapping[str, Iterable[str]],
) -> dict[str, ObjectSupportResolution]:
    """Resolve load-bearing object-contact paths to the static desk.

    ``object_names`` defines both output order and every graph tie break.
    ``adjacency`` is an undirected object-contact graph and must contain
    exactly one entry for every object.  Objects named by
    ``directly_supported`` are roots with an implicit edge to
    :data:`STATIC_DESK_SUPPORT`.

    A supported object's path starts at that object and ends at
    ``static_desk``.  An unsupported object has an empty path.  Breadth-first
    traversal makes the selected path shortest; input order makes equal-length
    choices deterministic even when callers provide sets as neighbor lists.
    """

    if isinstance(object_names, (str, bytes)) or not isinstance(
        object_names, Sequence
    ):
        raise ValueError("object_names must be an ordered sequence of names")
    names = tuple(object_names)
    if not names:
        raise ValueError("object_names must contain at least one object")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("object_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("object_names must not contain duplicate names")
    if STATIC_DESK_SUPPORT in names:
        raise ValueError(
            f"{STATIC_DESK_SUPPORT!r} is reserved for the static support root"
        )

    if isinstance(directly_supported, (str, bytes)) or not isinstance(
        directly_supported, Iterable
    ):
        raise ValueError("directly_supported must be an iterable of names")
    direct_names = tuple(directly_supported)
    if any(not isinstance(name, str) or not name for name in direct_names):
        raise ValueError("directly_supported must contain non-empty strings")
    if len(set(direct_names)) != len(direct_names):
        raise ValueError("directly_supported must not contain duplicate names")

    name_set = set(names)
    unknown_direct = set(direct_names) - name_set
    if unknown_direct:
        raise ValueError(
            "directly_supported contains unknown object names: "
            f"{sorted(unknown_direct)!r}"
        )

    if not isinstance(adjacency, Mapping):
        raise ValueError("adjacency must be a mapping")
    adjacency_key_names = tuple(adjacency)
    if any(
        not isinstance(name, str) or not name for name in adjacency_key_names
    ):
        raise ValueError("adjacency keys must be non-empty strings")
    adjacency_keys = set(adjacency_key_names)
    unknown_keys = adjacency_keys - name_set
    missing_keys = name_set - adjacency_keys
    if unknown_keys:
        raise ValueError(
            "adjacency contains unknown object names: "
            f"{sorted(unknown_keys)!r}"
        )
    if missing_keys:
        raise ValueError(
            "adjacency is missing object names: "
            f"{sorted(missing_keys)!r}"
        )

    neighbor_sets: dict[str, set[str]] = {}
    for name in names:
        neighbors = adjacency[name]
        if isinstance(neighbors, (str, bytes)) or not isinstance(
            neighbors, Iterable
        ):
            raise ValueError(
                f"adjacency[{name!r}] must be an iterable of names"
            )
        neighbor_names = tuple(neighbors)
        if any(
            not isinstance(neighbor, str) or not neighbor
            for neighbor in neighbor_names
        ):
            raise ValueError(
                f"adjacency[{name!r}] must contain non-empty strings"
            )
        if len(set(neighbor_names)) != len(neighbor_names):
            raise ValueError(
                f"adjacency[{name!r}] must not contain duplicate names"
            )
        if name in neighbor_names:
            raise ValueError(f"adjacency must not contain self edge {name!r}")
        unknown_neighbors = set(neighbor_names) - name_set
        if unknown_neighbors:
            raise ValueError(
                f"adjacency[{name!r}] contains unknown object names: "
                f"{sorted(unknown_neighbors)!r}"
            )
        neighbor_sets[name] = set(neighbor_names)

    for name in names:
        for neighbor in neighbor_sets[name]:
            if name not in neighbor_sets[neighbor]:
                raise ValueError(
                    "adjacency must be symmetric; missing reverse edge "
                    f"{neighbor!r} -> {name!r}"
                )

    order = {name: index for index, name in enumerate(names)}
    direct_set = set(direct_names)
    roots = [name for name in names if name in direct_set]
    parents: dict[str, str | None] = {name: None for name in roots}
    frontier = deque(roots)
    while frontier:
        current = frontier.popleft()
        for neighbor in sorted(
            neighbor_sets[current], key=order.__getitem__
        ):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            frontier.append(neighbor)

    result: dict[str, ObjectSupportResolution] = {}
    for name in names:
        if name not in parents:
            result[name] = {"supported": False, "support_path": ()}
            continue
        path = [name]
        cursor = name
        while parents[cursor] is not None:
            cursor = parents[cursor]
            path.append(cursor)
        path.append(STATIC_DESK_SUPPORT)
        result[name] = {
            "supported": True,
            "support_path": tuple(path),
        }
    return result


@dataclass(frozen=True)
class FreeJointMotion:
    """Velocity and acceleration magnitudes for one MuJoCo free joint."""

    linear_speed_m_s: float
    angular_speed_rad_s: float
    linear_acceleration_m_s2: float
    angular_acceleration_rad_s2: float

    @classmethod
    def from_vectors(
        cls,
        qvel: Sequence[float],
        qacc: Sequence[float],
    ) -> "FreeJointMotion":
        velocity = np.asarray(qvel, dtype=np.float64)
        acceleration = np.asarray(qacc, dtype=np.float64)
        if velocity.shape != (6,) or acceleration.shape != (6,):
            raise ValueError(
                "free-joint qvel and qacc must each have shape (6,), "
                f"got {velocity.shape} and {acceleration.shape}"
            )
        if not np.all(np.isfinite(velocity)) or not np.all(
            np.isfinite(acceleration)
        ):
            raise ValueError("free-joint motion contains non-finite values")
        return cls(
            linear_speed_m_s=float(np.linalg.norm(velocity[:3])),
            angular_speed_rad_s=float(np.linalg.norm(velocity[3:])),
            linear_acceleration_m_s2=float(np.linalg.norm(acceleration[:3])),
            angular_acceleration_rad_s2=float(np.linalg.norm(acceleration[3:])),
        )


@dataclass(frozen=True)
class QuietThresholds:
    """Maximum motion magnitudes accepted by the settle gate."""

    linear_speed_m_s: float
    angular_speed_rad_s: float
    linear_acceleration_m_s2: float
    angular_acceleration_rad_s2: float

    def __post_init__(self) -> None:
        values = (
            self.linear_speed_m_s,
            self.angular_speed_rad_s,
            self.linear_acceleration_m_s2,
            self.angular_acceleration_rad_s2,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError(
                "object-settle thresholds must be finite and non-negative"
            )

    def accepts(
        self,
        motion: FreeJointMotion,
        *,
        include_acceleration: bool = True,
    ) -> bool:
        velocity_quiet = bool(
            motion.linear_speed_m_s <= self.linear_speed_m_s
            and motion.angular_speed_rad_s <= self.angular_speed_rad_s
        )
        if not include_acceleration:
            return velocity_quiet
        return bool(
            velocity_quiet
            and motion.linear_acceleration_m_s2
            <= self.linear_acceleration_m_s2
            and motion.angular_acceleration_rad_s2
            <= self.angular_acceleration_rad_s2
        )


class ConsecutiveQuietWindow:
    """Require every object to be quiet for consecutive controller samples."""

    def __init__(
        self,
        thresholds: QuietThresholds,
        required_samples: int,
        *,
        include_acceleration: bool = True,
    ) -> None:
        if int(required_samples) != required_samples or required_samples <= 0:
            raise ValueError("required_samples must be a positive integer")
        self.thresholds = thresholds
        self.required_samples = int(required_samples)
        self.include_acceleration = bool(include_acceleration)
        self.consecutive_samples = 0

    def reset(self) -> None:
        self.consecutive_samples = 0

    def observe(
        self,
        motions: Mapping[str, FreeJointMotion],
        *,
        extra_quiet: bool = True,
    ) -> bool:
        if not motions:
            raise ValueError("object-settle gate requires at least one object")
        quiet = bool(extra_quiet) and all(
            self.thresholds.accepts(
                motion,
                include_acceleration=self.include_acceleration,
            )
            for motion in motions.values()
        )
        self.consecutive_samples = (
            self.consecutive_samples + 1 if quiet else 0
        )
        return self.consecutive_samples >= self.required_samples
