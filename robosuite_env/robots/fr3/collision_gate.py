# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Process-local arm-to-desk collision gate for FR3 joint targets.

The policy-facing IK solver and the live MuJoCo simulation must not share
``MjData``.  This module therefore evaluates candidate arm configurations in
its own scratch state.  Only caller-supplied arm-link / desk geometry pairs
are queried; gripper/object contacts and grasp rules are deliberately outside
the gate.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Mapping

import mujoco
import numpy as np


@dataclass(frozen=True)
class ArmDeskCollisionCheck:
    """Minimum signed distance for one arm configuration."""

    side: str
    safe: bool
    minimum_distance_m: float
    clearance_m: float
    arm_geom_id: int
    arm_geom_name: str
    desk_geom_id: int
    desk_geom_name: str
    evaluated_pairs: int
    collision_contact_pairs: int
    ignored_zero_distance_pairs: int


@dataclass(frozen=True)
class ArmDeskTargetFilter:
    """Result of clipping one requested target to a safe path prefix."""

    side: str
    accepted: bool
    clipped: bool
    reason: str | None
    selected_fraction: float
    requested_target: np.ndarray
    selected_target: np.ndarray
    requested_check: ArmDeskCollisionCheck
    selected_check: ArmDeskCollisionCheck
    blocked_fraction: float | None
    blocked_check: ArmDeskCollisionCheck | None
    collision_checks: int


class ArmDeskCollisionGate:
    """Check and line-search FR3 arm targets against fixed desk geometry."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        arm_qpos_indices: Mapping[str, Any],
        arm_geom_ids: Mapping[str, Any],
        desk_geom_ids: Any,
        clearance_m: float,
        distance_query_limit_m: float = 2.0,
    ) -> None:
        if not isinstance(arm_qpos_indices, Mapping) or not arm_qpos_indices:
            raise ValueError("arm_qpos_indices must be a non-empty mapping")
        if set(arm_qpos_indices) != set(arm_geom_ids):
            raise ValueError(
                "arm_qpos_indices and arm_geom_ids must have identical sides"
            )
        self.arm_qpos_indices = {
            str(side): self._indices(
                values,
                upper=int(model.nq),
                name=f"arm_qpos_indices[{side!r}]",
            )
            for side, values in arm_qpos_indices.items()
        }
        self.arm_geom_ids = {
            str(side): self._indices(
                arm_geom_ids[side],
                upper=int(model.ngeom),
                name=f"arm_geom_ids[{side!r}]",
            )
            for side in arm_qpos_indices
        }
        self.desk_geom_ids = self._indices(
            desk_geom_ids,
            upper=int(model.ngeom),
            name="desk_geom_ids",
        )
        if not math.isfinite(float(clearance_m)) or float(clearance_m) < 0.0:
            raise ValueError("clearance_m must be finite and non-negative")
        if (
            not math.isfinite(float(distance_query_limit_m))
            or float(distance_query_limit_m) <= float(clearance_m)
        ):
            raise ValueError(
                "distance_query_limit_m must be finite and exceed clearance_m"
            )
        self.clearance_m = float(clearance_m)
        self.distance_query_limit_m = float(distance_query_limit_m)
        desk_set = set(int(value) for value in self.desk_geom_ids)
        for side, geom_ids in self.arm_geom_ids.items():
            overlap = desk_set.intersection(int(value) for value in geom_ids)
            if overlap:
                raise ValueError(
                    f"arm and desk geometry sets overlap for {side}: "
                    f"{sorted(overlap)}"
                )
        # Collision margin belongs to MjModel, not MjData. A private model
        # copy lets scratch collision detection expose contacts throughout
        # the requested clearance shell without changing live simulation
        # physics, rendering, or grasp contacts.
        self.model = copy.copy(model)
        gated_geom_ids = set(int(value) for value in self.desk_geom_ids)
        for geom_ids in self.arm_geom_ids.values():
            gated_geom_ids.update(int(value) for value in geom_ids)
        for geom_id in gated_geom_ids:
            self.model.geom_margin[geom_id] = max(
                float(self.model.geom_margin[geom_id]),
                self.clearance_m,
            )
            self.model.geom_gap[geom_id] = 0.0
        self._data = mujoco.MjData(self.model)

    @staticmethod
    def _indices(value: Any, *, upper: int, name: str) -> np.ndarray:
        try:
            indices = np.asarray(value, dtype=np.int64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a one-dimensional index list") from exc
        if indices.size == 0:
            raise ValueError(f"{name} must not be empty")
        if (
            np.any(indices < 0)
            or np.any(indices >= int(upper))
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError(
                f"{name} must contain unique indices in [0, {int(upper)})"
            )
        return indices.copy()

    @staticmethod
    def _finite_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
        try:
            vector = np.asarray(value, dtype=np.float64).reshape(shape)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must have shape {shape}") from exc
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values")
        return vector.copy()

    def _prepare(self, base_qpos: Any | None) -> None:
        if base_qpos is None:
            qpos = np.asarray(self.model.qpos0, dtype=np.float64)
        else:
            qpos = self._finite_vector(
                base_qpos,
                (int(self.model.nq),),
                "base_qpos",
            )
        self._data.qpos[:] = qpos
        self._data.qvel[:] = 0.0
        self._data.qacc_warmstart[:] = 0.0
        if int(self.model.na):
            self._data.act[:] = 0.0

    def _geom_name(self, geom_id: int) -> str:
        return (
            mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                int(geom_id),
            )
            or f"geom_{int(geom_id)}"
        )

    def _check_prepared(self, side: str, q: np.ndarray) -> ArmDeskCollisionCheck:
        self._data.qpos[self.arm_qpos_indices[side]] = q
        mujoco.mj_fwdPosition(self.model, self._data)
        mujoco.mj_collision(self.model, self._data)
        arm_set = set(int(value) for value in self.arm_geom_ids[side])
        desk_set = set(int(value) for value in self.desk_geom_ids)
        contact_distances: dict[tuple[int, int], float] = {}
        for contact_index in range(int(self._data.ncon)):
            contact = self._data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in arm_set and geom2 in desk_set:
                pair = (geom1, geom2)
            elif geom2 in arm_set and geom1 in desk_set:
                pair = (geom2, geom1)
            else:
                continue
            contact_distances[pair] = min(
                contact_distances.get(pair, float("inf")),
                float(contact.dist),
            )
        minimum_distance = float("inf")
        minimum_pair: tuple[int, int] | None = None
        evaluated_pairs = 0
        ignored_zero_distance_pairs = 0
        for arm_geom_id in self.arm_geom_ids[side]:
            for desk_geom_id in self.desk_geom_ids:
                pair = (int(arm_geom_id), int(desk_geom_id))
                distance = float(
                    mujoco.mj_geomDistance(
                        self.model,
                        self._data,
                        int(arm_geom_id),
                        int(desk_geom_id),
                        self.distance_query_limit_m,
                        None,
                    )
                )
                if not math.isfinite(distance):
                    raise RuntimeError(
                        "MuJoCo returned non-finite arm/desk distance"
                    )
                evaluated_pairs += 1
                contact_distance = contact_distances.get(pair)
                if contact_distance is not None:
                    distance = float(contact_distance)
                elif abs(distance) <= 1e-12:
                    # MuJoCo's convex-mesh distance query can return an
                    # ambiguous exact zero for widely separated pairs. If
                    # the same scratch model, with an inflated collision
                    # margin, generated no matching contact, that zero is
                    # not evidence of a clearance violation.
                    ignored_zero_distance_pairs += 1
                    continue
                if distance < minimum_distance:
                    minimum_distance = distance
                    minimum_pair = pair
        if minimum_pair is None:
            minimum_pair = (
                int(self.arm_geom_ids[side][0]),
                int(self.desk_geom_ids[0]),
            )
            minimum_distance = self.clearance_m
        tolerance = 1e-9
        arm_geom_id, desk_geom_id = minimum_pair
        return ArmDeskCollisionCheck(
            side=side,
            safe=bool(minimum_distance >= self.clearance_m - tolerance),
            minimum_distance_m=minimum_distance,
            clearance_m=self.clearance_m,
            arm_geom_id=arm_geom_id,
            arm_geom_name=self._geom_name(arm_geom_id),
            desk_geom_id=desk_geom_id,
            desk_geom_name=self._geom_name(desk_geom_id),
            evaluated_pairs=evaluated_pairs,
            collision_contact_pairs=len(contact_distances),
            ignored_zero_distance_pairs=ignored_zero_distance_pairs,
        )

    def check(
        self,
        side: str,
        q: Any,
        *,
        base_qpos: Any | None = None,
    ) -> ArmDeskCollisionCheck:
        """Evaluate one target without modifying any caller-owned ``MjData``."""

        if side not in self.arm_qpos_indices:
            raise ValueError(
                f"unknown side {side!r}; expected one of "
                f"{sorted(self.arm_qpos_indices)}"
            )
        target = self._finite_vector(
            q,
            self.arm_qpos_indices[side].shape,
            "q",
        )
        self._prepare(base_qpos)
        return self._check_prepared(side, target)

    def check_path(
        self,
        side: str,
        q_sequence: Any,
        *,
        base_qpos: Any | None = None,
    ) -> tuple[ArmDeskCollisionCheck, ...]:
        """Evaluate an ordered command path in one prepared scratch state."""

        if side not in self.arm_qpos_indices:
            raise ValueError(
                f"unknown side {side!r}; expected one of "
                f"{sorted(self.arm_qpos_indices)}"
            )
        try:
            sequence = np.asarray(q_sequence, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("q_sequence must be a finite two-dimensional array") from exc
        expected_width = int(self.arm_qpos_indices[side].size)
        if sequence.ndim != 2 or sequence.shape[1:] != (expected_width,):
            raise ValueError(
                "q_sequence must have shape "
                f"(N, {expected_width}), got {sequence.shape}"
            )
        if not np.all(np.isfinite(sequence)):
            raise ValueError("q_sequence must contain only finite values")
        self._prepare(base_qpos)
        return tuple(
            self._check_prepared(side, np.asarray(target, dtype=np.float64))
            for target in sequence
        )

    def filter_target(
        self,
        side: str,
        requested_target: Any,
        safe_reference: Any,
        *,
        base_qpos: Any | None = None,
        path_samples: int = 16,
        bisection_iterations: int = 8,
        retreat_rad: float = 0.0,
    ) -> ArmDeskTargetFilter:
        """Keep the largest checked collision-free prefix in joint space.

        Prefix sampling prevents accepting a target beyond the first detected
        collision interval. Bisection then places the selected endpoint close
        to the requested clearance boundary. If no positive safe progress is
        available, callers should continue their previous validated target.
        """

        if side not in self.arm_qpos_indices:
            raise ValueError(
                f"unknown side {side!r}; expected one of "
                f"{sorted(self.arm_qpos_indices)}"
            )
        if (
            isinstance(path_samples, bool)
            or not isinstance(path_samples, (int, np.integer))
            or int(path_samples) < 1
        ):
            raise ValueError("path_samples must be a positive integer")
        if (
            isinstance(bisection_iterations, bool)
            or not isinstance(bisection_iterations, (int, np.integer))
            or int(bisection_iterations) < 0
        ):
            raise ValueError(
                "bisection_iterations must be a non-negative integer"
            )
        if not math.isfinite(float(retreat_rad)) or float(retreat_rad) < 0.0:
            raise ValueError("retreat_rad must be finite and non-negative")
        shape = self.arm_qpos_indices[side].shape
        requested = self._finite_vector(
            requested_target,
            shape,
            "requested_target",
        )
        reference = self._finite_vector(
            safe_reference,
            shape,
            "safe_reference",
        )
        self._prepare(base_qpos)
        checks = 0
        requested_check = self._check_prepared(side, requested)
        checks += 1
        reference_check = self._check_prepared(side, reference)
        checks += 1
        if not reference_check.safe:
            return ArmDeskTargetFilter(
                side=side,
                accepted=False,
                clipped=False,
                reason="safe_reference_below_desk_clearance",
                selected_fraction=0.0,
                requested_target=requested,
                selected_target=reference.copy(),
                requested_check=requested_check,
                selected_check=reference_check,
                blocked_fraction=0.0,
                blocked_check=reference_check,
                collision_checks=checks,
            )

        delta = requested - reference
        lower_fraction = 0.0
        lower_check = reference_check
        upper_fraction = 1.0
        upper_check = requested_check
        first_unsafe_found = False
        for sample_index in range(1, int(path_samples) + 1):
            fraction = sample_index / float(path_samples)
            if sample_index == int(path_samples):
                sample_check = requested_check
            else:
                sample_check = self._check_prepared(
                    side,
                    reference + fraction * delta,
                )
                checks += 1
            if sample_check.safe:
                lower_fraction = fraction
                lower_check = sample_check
            else:
                upper_fraction = fraction
                upper_check = sample_check
                first_unsafe_found = True
                break

        if not first_unsafe_found:
            return ArmDeskTargetFilter(
                side=side,
                accepted=True,
                clipped=False,
                reason=None,
                selected_fraction=1.0,
                requested_target=requested,
                selected_target=requested.copy(),
                requested_check=requested_check,
                selected_check=requested_check,
                blocked_fraction=None,
                blocked_check=None,
                collision_checks=checks,
            )

        for _ in range(int(bisection_iterations)):
            middle_fraction = 0.5 * (lower_fraction + upper_fraction)
            middle_check = self._check_prepared(
                side,
                reference + middle_fraction * delta,
            )
            checks += 1
            if middle_check.safe:
                lower_fraction = middle_fraction
                lower_check = middle_check
            else:
                upper_fraction = middle_fraction
                upper_check = middle_check

        maximum_delta = float(np.max(np.abs(delta), initial=0.0))
        if maximum_delta > 0.0 and lower_fraction > 0.0:
            boundary_fraction = lower_fraction
            boundary_check = lower_check
            retreat_fraction = max(
                0.0,
                lower_fraction - float(retreat_rad) / maximum_delta,
            )
            if retreat_fraction == 0.0:
                lower_fraction = 0.0
                lower_check = reference_check
            elif float(retreat_rad) > 0.0:
                retreat_check = self._check_prepared(
                    side,
                    reference + retreat_fraction * delta,
                )
                checks += 1
                if retreat_check.safe:
                    lower_fraction = retreat_fraction
                    lower_check = retreat_check
                else:
                    # A non-convex mesh can expose another clearance shell
                    # while moving back from the first boundary. The
                    # bisection result itself is known safe, so retain it
                    # instead of aborting the rollout.
                    lower_fraction = boundary_fraction
                    lower_check = boundary_check
        selected = reference + lower_fraction * delta
        positive_progress = lower_fraction > 1e-9
        return ArmDeskTargetFilter(
            side=side,
            accepted=positive_progress,
            clipped=positive_progress,
            reason=(
                "arm_desk_collision_clipped"
                if positive_progress
                else "arm_desk_collision_hold"
            ),
            selected_fraction=float(lower_fraction),
            requested_target=requested,
            selected_target=selected,
            requested_check=requested_check,
            selected_check=lower_check,
            blocked_fraction=float(upper_fraction),
            blocked_check=upper_check,
            collision_checks=checks,
        )


__all__ = [
    "ArmDeskCollisionCheck",
    "ArmDeskCollisionGate",
    "ArmDeskTargetFilter",
]
