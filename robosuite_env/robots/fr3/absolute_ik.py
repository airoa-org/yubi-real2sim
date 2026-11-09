# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Scratch-state whole-pose inverse kinematics for simulated FR3 arms.

The solver deliberately owns its :class:`mujoco.MjData`.  Solving IK therefore
does not overwrite the live simulator's qpos, qvel, contacts, or warm-start
state.  Targets are expressed at the policy TCP; MuJoCo's body Jacobian is
evaluated at the configured EEF after applying the fixed EEF-to-TCP transform.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import mujoco
import numpy as np


_POSITION_TOLERANCE_M = 5e-4
_ORIENTATION_TOLERANCE_RAD = math.radians(0.5)
_FINITE_EPSILON = 1e-12


@dataclass(frozen=True)
class IKResult:
    """Result of one absolute whole-pose IK solve.

    ``q`` is always finite and hard-clipped to the configured physical joint
    limits.  When ``converged`` is false it is the best finite iterate found,
    and ``fallback_reason`` describes why the requested tolerance was not met.
    """

    q: np.ndarray
    converged: bool
    iterations: int
    pos_error_mm: float
    ori_error_deg: float
    min_limit_margin_rad: float
    condition_number: float
    home_distance_rad: float
    fallback_reason: str | None


@dataclass(frozen=True)
class _ArmSpec:
    qpos_indices: np.ndarray
    dof_indices: np.ndarray
    joint_ranges: np.ndarray
    eef_body_id: int


@dataclass(frozen=True)
class _RigidTransform:
    position: np.ndarray
    quaternion_wxyz: np.ndarray


def _finite_array(value: Any, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape {shape}") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _normalize_quaternion(value: Any, *, name: str) -> np.ndarray:
    quaternion = _finite_array(value, (4,), name=name)
    norm = float(np.linalg.norm(quaternion))
    if norm <= _FINITE_EPSILON:
        raise ValueError(f"{name} must have non-zero norm")
    return quaternion / norm


def _quaternion_matrix(quaternion_wxyz: Any) -> np.ndarray:
    w, x, y, z = _normalize_quaternion(
        quaternion_wxyz, name="quaternion_wxyz"
    )
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _matrix_quaternion(matrix: Any) -> np.ndarray:
    matrix = _finite_array(matrix, (3, 3), name="rotation matrix")
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(
        quaternion,
        np.ascontiguousarray(matrix, dtype=np.float64).reshape(9),
    )
    return _normalize_quaternion(quaternion, name="converted quaternion")


def _quaternion_conjugate(quaternion_wxyz: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            quaternion_wxyz[0],
            -quaternion_wxyz[1],
            -quaternion_wxyz[2],
            -quaternion_wxyz[3],
        ],
        dtype=np.float64,
    )


def _quaternion_multiply(
    left_wxyz: np.ndarray, right_wxyz: np.ndarray
) -> np.ndarray:
    lw, lx, ly, lz = left_wxyz
    rw, rx, ry, rz = right_wxyz
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _quaternion_error_rotvec(
    target_wxyz: np.ndarray, current_wxyz: np.ndarray
) -> np.ndarray:
    """World-frame rotation vector taking ``current`` to ``target``."""

    error = _quaternion_multiply(
        target_wxyz, _quaternion_conjugate(current_wxyz)
    )
    error /= max(float(np.linalg.norm(error)), _FINITE_EPSILON)
    if error[0] < 0.0:
        error = -error
    vector_norm = float(np.linalg.norm(error[1:]))
    if vector_norm <= _FINITE_EPSILON:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, max(float(error[0]), 0.0))
    return error[1:] * (angle / vector_norm)


def _orientation_error_rad(
    target_wxyz: np.ndarray, actual_wxyz: np.ndarray
) -> float:
    dot = float(np.clip(abs(np.dot(target_wxyz, actual_wxyz)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def _mapping_value(spec: Any, name: str) -> Any:
    if isinstance(spec, Mapping):
        try:
            return spec[name]
        except KeyError as exc:
            raise ValueError(f"arm spec is missing {name!r}") from exc
    try:
        return getattr(spec, name)
    except AttributeError as exc:
        raise ValueError(f"arm spec is missing {name!r}") from exc


def _coerce_arm_spec(model: mujoco.MjModel, side: str, raw: Any) -> _ArmSpec:
    qpos_indices = np.asarray(
        _mapping_value(raw, "qpos_indices"), dtype=np.int64
    ).reshape(-1)
    dof_indices = np.asarray(
        _mapping_value(raw, "dof_indices"), dtype=np.int64
    ).reshape(-1)
    try:
        joint_ranges = np.asarray(
            _mapping_value(raw, "joint_ranges"), dtype=np.float64
        ).reshape(-1, 2)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"arm_specs[{side!r}].joint_ranges must have shape (7, 2)"
        ) from exc
    try:
        eef_body_id = int(_mapping_value(raw, "eef_body_id"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"arm_specs[{side!r}].eef_body_id must be an integer"
        ) from exc

    if len(qpos_indices) != 7 or len(dof_indices) != 7:
        raise ValueError(
            f"arm_specs[{side!r}] must describe exactly seven FR3 joints"
        )
    if joint_ranges.shape != (7, 2):
        raise ValueError(
            f"arm_specs[{side!r}].joint_ranges must have shape (7, 2)"
        )
    if not np.all(np.isfinite(joint_ranges)):
        raise ValueError(
            f"arm_specs[{side!r}].joint_ranges must contain finite values"
        )
    if np.any(joint_ranges[:, 0] > joint_ranges[:, 1]):
        raise ValueError(
            f"arm_specs[{side!r}] contains an inverted joint range"
        )
    if (
        np.any(qpos_indices < 0)
        or np.any(qpos_indices >= model.nq)
        or len(np.unique(qpos_indices)) != 7
    ):
        raise ValueError(
            f"arm_specs[{side!r}].qpos_indices must be seven unique valid indices"
        )
    if (
        np.any(dof_indices < 0)
        or np.any(dof_indices >= model.nv)
        or len(np.unique(dof_indices)) != 7
    ):
        raise ValueError(
            f"arm_specs[{side!r}].dof_indices must be seven unique valid indices"
        )
    if eef_body_id < 0 or eef_body_id >= model.nbody:
        raise ValueError(
            f"arm_specs[{side!r}].eef_body_id is outside the MuJoCo model"
        )
    return _ArmSpec(
        qpos_indices=qpos_indices.copy(),
        dof_indices=dof_indices.copy(),
        joint_ranges=joint_ranges.copy(),
        eef_body_id=eef_body_id,
    )


def _coerce_transform(raw: Any, *, name: str) -> _RigidTransform:
    """Accept a translation, 4x4 matrix, or position/quaternion mapping."""

    if isinstance(raw, Mapping):
        position_key = "position" if "position" in raw else "pos"
        quaternion_key = (
            "quaternion_wxyz" if "quaternion_wxyz" in raw else "quat_wxyz"
        )
        if position_key not in raw or quaternion_key not in raw:
            raise ValueError(
                f"{name} mapping requires position and quaternion_wxyz"
            )
        return _RigidTransform(
            position=_finite_array(raw[position_key], (3,), name=f"{name}.position"),
            quaternion_wxyz=_normalize_quaternion(
                raw[quaternion_key], name=f"{name}.quaternion_wxyz"
            ),
        )

    array = np.asarray(raw, dtype=np.float64)
    if array.shape == (3,):
        return _RigidTransform(
            position=_finite_array(array, (3,), name=name),
            quaternion_wxyz=np.asarray(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            ),
        )
    if array.shape == (4, 4):
        transform = _finite_array(array, (4, 4), name=name)
        if not np.allclose(
            transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0
        ):
            raise ValueError(f"{name} must be a homogeneous rigid transform")
        rotation = transform[:3, :3]
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0.0
        ) or not math.isclose(
            float(np.linalg.det(rotation)), 1.0, abs_tol=1e-7
        ):
            raise ValueError(f"{name} rotation must be orthonormal")
        return _RigidTransform(
            position=transform[:3, 3].copy(),
            quaternion_wxyz=_matrix_quaternion(rotation),
        )
    raise ValueError(
        f"{name} must be a 3-vector, 4x4 transform, or transform mapping"
    )


class AbsoluteFR3IK:
    """Damped-least-squares absolute IK using an isolated MuJoCo state."""

    def __init__(
        self,
        model: mujoco.MjModel,
        arm_specs: Mapping[str, Any],
        eef_to_tcp: Any,
        iterations: int = 50,
        damping: float = 1e-5,
        max_step_deg: float = 5,
        nullspace_gain: float = 0,
        home_q: Any | None = None,
        nullspace_tolerance_rad: float = 1e-5,
    ):
        if not isinstance(arm_specs, Mapping) or not arm_specs:
            raise ValueError("arm_specs must be a non-empty side-to-spec mapping")
        self.model = model
        self.arm_specs = {
            str(side): _coerce_arm_spec(model, str(side), spec)
            for side, spec in arm_specs.items()
        }

        side_transform_mapping = (
            isinstance(eef_to_tcp, Mapping)
            and set(self.arm_specs).issubset(eef_to_tcp)
        )
        if side_transform_mapping:
            self.eef_to_tcp = {
                side: _coerce_transform(
                    eef_to_tcp[side], name=f"eef_to_tcp[{side!r}]"
                )
                for side in self.arm_specs
            }
        else:
            shared_transform = _coerce_transform(eef_to_tcp, name="eef_to_tcp")
            self.eef_to_tcp = {
                side: shared_transform for side in self.arm_specs
            }

        try:
            self.iterations = int(iterations)
        except (TypeError, ValueError) as exc:
            raise ValueError("iterations must be a positive integer") from exc
        if self.iterations <= 0 or self.iterations != iterations:
            raise ValueError("iterations must be a positive integer")
        self.damping = float(damping)
        self.max_step_rad = math.radians(float(max_step_deg))
        self.nullspace_gain = float(nullspace_gain)
        self.nullspace_tolerance_rad = float(nullspace_tolerance_rad)
        if not math.isfinite(self.damping) or self.damping < 0.0:
            raise ValueError("damping must be finite and non-negative")
        if not math.isfinite(self.max_step_rad) or self.max_step_rad <= 0.0:
            raise ValueError("max_step_deg must be finite and positive")
        if (
            not math.isfinite(self.nullspace_gain)
            or self.nullspace_gain < 0.0
        ):
            raise ValueError("nullspace_gain must be finite and non-negative")
        if (
            not math.isfinite(self.nullspace_tolerance_rad)
            or self.nullspace_tolerance_rad <= 0.0
        ):
            raise ValueError(
                "nullspace_tolerance_rad must be finite and positive"
            )

        if home_q is None:
            if self.nullspace_gain > 0.0:
                raise ValueError(
                    "home_q is required when nullspace_gain is positive"
                )
            self.home_q: dict[str, np.ndarray] | None = None
        else:
            side_home_mapping = (
                isinstance(home_q, Mapping)
                and set(self.arm_specs).issubset(home_q)
            )
            self.home_q = {}
            for side, spec in self.arm_specs.items():
                raw_home = home_q[side] if side_home_mapping else home_q
                resolved_home = _finite_array(
                    raw_home, (7,), name=f"home_q[{side!r}]"
                )
                lower = spec.joint_ranges[:, 0]
                upper = spec.joint_ranges[:, 1]
                outside = np.flatnonzero(
                    (resolved_home < lower) | (resolved_home > upper)
                )
                if outside.size:
                    joints = ", ".join(
                        f"j{int(index) + 1}" for index in outside
                    )
                    raise ValueError(
                        f"home_q[{side!r}] is outside joint limits at {joints}"
                    )
                self.home_q[side] = resolved_home

        # This is intentionally never exposed to, copied from, or aliased with
        # the environment's live MjData.
        self._data = mujoco.MjData(model)
        self._jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
        self._jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)

    def _target_eef_pose(
        self,
        side: str,
        target_tcp_position: np.ndarray,
        target_tcp_quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        transform = self.eef_to_tcp[side]
        tcp_rotation = _quaternion_matrix(target_tcp_quaternion)
        eef_to_tcp_rotation = _quaternion_matrix(transform.quaternion_wxyz)
        eef_rotation = tcp_rotation @ eef_to_tcp_rotation.T
        eef_position = target_tcp_position - eef_rotation @ transform.position
        return eef_position, _matrix_quaternion(eef_rotation)

    def _tcp_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        spec = self.arm_specs[side]
        transform = self.eef_to_tcp[side]
        eef_position = self._data.xpos[spec.eef_body_id].copy()
        eef_quaternion = _normalize_quaternion(
            self._data.xquat[spec.eef_body_id], name="MuJoCo EEF quaternion"
        )
        eef_rotation = _quaternion_matrix(eef_quaternion)
        tcp_position = eef_position + eef_rotation @ transform.position
        tcp_quaternion = _quaternion_multiply(
            eef_quaternion, transform.quaternion_wxyz
        )
        tcp_quaternion /= max(
            float(np.linalg.norm(tcp_quaternion)), _FINITE_EPSILON
        )
        return tcp_position, tcp_quaternion

    def _set_arm(self, spec: _ArmSpec, q: np.ndarray) -> None:
        self._data.qpos[spec.qpos_indices] = q
        self._data.qvel[spec.dof_indices] = 0.0
        mujoco.mj_forward(self.model, self._data)

    def _metrics(
        self,
        side: str,
        target_tcp_position: np.ndarray,
        target_tcp_quaternion: np.ndarray,
    ) -> tuple[float, float]:
        actual_position, actual_quaternion = self._tcp_pose(side)
        return (
            float(np.linalg.norm(target_tcp_position - actual_position)),
            _orientation_error_rad(
                target_tcp_quaternion, actual_quaternion
            ),
        )

    def _arm_jacobian(self, spec: _ArmSpec) -> np.ndarray:
        self._jacobian_position.fill(0.0)
        self._jacobian_rotation.fill(0.0)
        mujoco.mj_jacBody(
            self.model,
            self._data,
            self._jacobian_position,
            self._jacobian_rotation,
            spec.eef_body_id,
        )
        return np.vstack(
            [
                self._jacobian_position[:, spec.dof_indices],
                self._jacobian_rotation[:, spec.dof_indices],
            ]
        )

    def _condition_number(self, jacobian: np.ndarray) -> float:
        try:
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
        except np.linalg.LinAlgError:
            return float("inf")
        if (
            len(singular_values) == 0
            or not np.all(np.isfinite(singular_values))
            or singular_values[-1] <= _FINITE_EPSILON
        ):
            return float("inf")
        return float(singular_values[0] / singular_values[-1])

    def _inverse_and_structural_nullspace(
        self, jacobian: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a damped task inverse and the exact 7-DoF redundancy.

        The task update uses the same SVD as the structural null-space basis.
        Unlike ``I - J#J`` with a damped inverse, the returned projector is a
        true orthogonal projector and therefore does not leak the posture
        objective into the first-order Cartesian task.
        """

        rows, columns = jacobian.shape
        if rows != 6 or columns != 7:
            raise ValueError(
                f"FR3 Jacobian must have shape (6, 7), got {jacobian.shape}"
            )
        u, singular_values, vh = np.linalg.svd(
            jacobian, full_matrices=True
        )
        denominator = singular_values * singular_values + self.damping
        filters = np.divide(
            singular_values,
            denominator,
            out=np.zeros_like(singular_values),
            where=denominator > _FINITE_EPSILON,
        )
        pseudo_inverse = (vh[:rows].T * filters) @ u.T

        # A 6D task on seven independent FR3 joints always has one structural
        # redundancy.  Do not promote weak task singular directions into the
        # posture objective near a kinematic singularity.
        null_basis = vh[rows:].T
        nullspace = null_basis @ null_basis.T
        return pseudo_inverse, nullspace

    @staticmethod
    def _step_to_joint_limits(
        q: np.ndarray,
        delta: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> float:
        """Largest scale in [0, 1] that keeps ``q + scale * delta`` valid."""

        scale = 1.0
        positive = delta > _FINITE_EPSILON
        negative = delta < -_FINITE_EPSILON
        if np.any(positive):
            scale = min(
                scale,
                float(np.min((upper[positive] - q[positive]) / delta[positive])),
            )
        if np.any(negative):
            scale = min(
                scale,
                float(np.min((lower[negative] - q[negative]) / delta[negative])),
            )
        return float(np.clip(scale, 0.0, 1.0))

    @staticmethod
    def _discard_outward_limit_motion(
        q: np.ndarray,
        delta: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        """Keep useful free-joint motion when one joint is on a hard limit."""

        bounded = delta.copy()
        at_lower = q <= lower + _FINITE_EPSILON
        at_upper = q >= upper - _FINITE_EPSILON
        bounded[at_lower & (bounded < 0.0)] = 0.0
        bounded[at_upper & (bounded > 0.0)] = 0.0
        return bounded

    @staticmethod
    def _home_cost(
        q: np.ndarray,
        home: np.ndarray,
    ) -> float:
        distance = q - home
        return 0.5 * float(distance @ distance)

    def solve(
        self,
        side: str,
        target_tcp_pos: Any,
        target_quat_wxyz: Any,
        seed_q: Any,
        continuity_q: Any | None = None,
        *,
        _allow_home_restart: bool = True,
    ) -> IKResult:
        """Solve one policy-TCP target without modifying live simulation data."""

        try:
            spec = self.arm_specs[side]
        except KeyError as exc:
            raise ValueError(
                f"unknown arm side {side!r}; expected one of "
                f"{sorted(self.arm_specs)}"
            ) from exc
        target_position = _finite_array(
            target_tcp_pos, (3,), name="target_tcp_pos"
        )
        target_quaternion = _normalize_quaternion(
            target_quat_wxyz, name="target_quat_wxyz"
        )
        seed = _finite_array(seed_q, (7,), name="seed_q")
        continuity = (
            None
            if continuity_q is None
            else _finite_array(continuity_q, (7,), name="continuity_q")
        )
        lower = spec.joint_ranges[:, 0]
        upper = spec.joint_ranges[:, 1]

        # A previous valid solution is the best branch-continuous numerical
        # seed.  The measured/current seed remains the default for the first
        # target in a sequence.  It is deliberately not the posture objective:
        # continuity chooses where the numerical search starts, while home_q
        # chooses among redundant solutions.
        q = np.clip(continuity if continuity is not None else seed, lower, upper)
        continuity_reference = q.copy()
        home = (
            q.copy()
            if self.home_q is None or self.nullspace_gain == 0.0
            else self.home_q[side]
        )
        target_eef_position, target_eef_quaternion = self._target_eef_pose(
            side, target_position, target_quaternion
        )

        best_feasible_q: np.ndarray | None = None
        best_feasible_key: tuple[float, ...] | None = None
        best_infeasible_q = q.copy()
        best_infeasible_key: tuple[float, ...] | None = None
        iterations_used = 0
        condition_number = float("inf")
        fallback_reason: str | None = None

        for iteration in range(self.iterations + 1):
            self._set_arm(spec, q)
            current_eef_position = self._data.xpos[spec.eef_body_id].copy()
            current_eef_quaternion = _normalize_quaternion(
                self._data.xquat[spec.eef_body_id],
                name="MuJoCo EEF quaternion",
            )
            position_error = target_eef_position - current_eef_position
            orientation_error = _quaternion_error_rotvec(
                target_eef_quaternion, current_eef_quaternion
            )
            if not np.all(np.isfinite(position_error)) or not np.all(
                np.isfinite(orientation_error)
            ):
                fallback_reason = "nonfinite_forward_kinematics"
                break

            position_norm, orientation_norm = self._metrics(
                side, target_position, target_quaternion
            )
            jacobian = self._arm_jacobian(spec)
            jacobian_is_finite = bool(np.all(np.isfinite(jacobian)))
            condition_number = (
                self._condition_number(jacobian)
                if jacobian_is_finite
                else float("inf")
            )
            feasible = bool(
                position_norm <= _POSITION_TOLERANCE_M
                and orientation_norm <= _ORIENTATION_TOLERANCE_RAD
            )
            normalized_position = position_norm / _POSITION_TOLERANCE_M
            normalized_orientation = (
                orientation_norm / _ORIENTATION_TOLERANCE_RAD
            )
            task_score = (
                normalized_position * normalized_position
                + normalized_orientation * normalized_orientation
            )
            home_cost = self._home_cost(q, home)
            min_margin = float(np.min(np.minimum(q - lower, upper - q)))
            continuity_distance = float(
                (q - continuity_reference) @ (q - continuity_reference)
            )
            if feasible:
                candidate_key = (
                    home_cost,
                    task_score,
                    -min_margin,
                    continuity_distance,
                )
                if (
                    best_feasible_key is None
                    or candidate_key < best_feasible_key
                ):
                    best_feasible_key = candidate_key
                    best_feasible_q = q.copy()
            else:
                candidate_key = (
                    max(normalized_position, normalized_orientation),
                    task_score,
                    home_cost,
                )
                if (
                    best_infeasible_key is None
                    or candidate_key < best_infeasible_key
                ):
                    best_infeasible_key = candidate_key
                    best_infeasible_q = q.copy()

            # With no secondary objective, retain the historical immediate
            # task-tolerance exit.
            if feasible and self.nullspace_gain == 0.0:
                break
            if iteration == self.iterations:
                fallback_reason = "max_iterations"
                break

            if not jacobian_is_finite:
                fallback_reason = "nonfinite_jacobian"
                break
            try:
                pseudo_inverse, nullspace = (
                    self._inverse_and_structural_nullspace(jacobian)
                )
            except np.linalg.LinAlgError:
                fallback_reason = "singular_damped_system"
                break

            task_delta = pseudo_inverse @ np.concatenate(
                [position_error, orientation_error]
            )
            posture_delta = (
                self.nullspace_gain * nullspace @ (home - q)
            )
            if (
                feasible
                and float(np.max(np.abs(posture_delta), initial=0.0))
                <= self.nullspace_tolerance_rad
            ):
                break
            delta = task_delta + posture_delta
            if not np.all(np.isfinite(delta)):
                fallback_reason = "nonfinite_update"
                break
            # A single joint exactly on a bound must not zero the common step
            # scale and discard valid inward/free-joint task motion.
            delta = self._discard_outward_limit_motion(
                q, delta, lower, upper
            )
            maximum = float(np.max(np.abs(delta), initial=0.0))
            if maximum <= _FINITE_EPSILON:
                fallback_reason = "joint_limits"
                break
            if maximum > self.max_step_rad:
                delta *= self.max_step_rad / maximum
            limit_scale = self._step_to_joint_limits(
                q, delta, lower, upper
            )
            if limit_scale <= _FINITE_EPSILON:
                fallback_reason = "joint_limits"
                break

            if feasible:
                # An instantaneous null-space step can produce second-order
                # Cartesian drift.  Keep the accepted iterate inside the task
                # tolerance and require strict progress toward home.
                accepted = False
                step_scale = limit_scale
                while step_scale >= (1.0 / 64.0):
                    trial_q = q + step_scale * delta
                    self._set_arm(spec, trial_q)
                    trial_position, trial_orientation = self._metrics(
                        side, target_position, target_quaternion
                    )
                    if (
                        trial_position <= _POSITION_TOLERANCE_M
                        and trial_orientation <= _ORIENTATION_TOLERANCE_RAD
                        and self._home_cost(trial_q, home)
                        < home_cost - _FINITE_EPSILON
                    ):
                        q = trial_q
                        accepted = True
                        break
                    step_scale *= 0.5
                if not accepted:
                    break
            else:
                q = q + limit_scale * delta
            iterations_used = iteration + 1

        # A feasible pose always outranks an infeasible pose, even when the
        # latter is closer to home.  Among feasible solutions home distance is
        # the primary selection objective; otherwise Cartesian error remains
        # primary so unreachable targets fail predictably.
        converged = best_feasible_q is not None
        q = (
            best_feasible_q.copy()
            if best_feasible_q is not None
            else best_infeasible_q.copy()
        )
        q = np.clip(q, lower, upper)
        self._set_arm(spec, q)
        position_norm, orientation_norm = self._metrics(
            side, target_position, target_quaternion
        )
        final_jacobian = self._arm_jacobian(spec)
        condition_number = (
            self._condition_number(final_jacobian)
            if np.all(np.isfinite(final_jacobian))
            else float("inf")
        )
        min_margin = float(np.min(np.minimum(q - lower, upper - q)))
        if fallback_reason == "max_iterations" and min_margin <= 1e-9:
            fallback_reason = "joint_limits"
        result = IKResult(
            q=q.copy(),
            converged=converged,
            iterations=iterations_used,
            pos_error_mm=position_norm * 1000.0,
            ori_error_deg=math.degrees(orientation_norm),
            min_limit_margin_rad=max(0.0, min_margin),
            condition_number=condition_number,
            home_distance_rad=float(np.linalg.norm(q - home)),
            fallback_reason=(
                None if converged else fallback_reason or "solver_failure"
            ),
        )
        # Bound-constrained IK can settle on a poor local branch when a live
        # joint starts exactly at its limit.  Retry once from the preferred
        # posture; this both escapes that branch and preserves the intended
        # home-biased solution selection.  Keep the better Cartesian fallback
        # when neither start reaches tolerance.
        if (
            not result.converged
            and _allow_home_restart
            and self.nullspace_gain > 0.0
            and self.home_q is not None
            and not np.allclose(
                continuity_reference,
                home,
                atol=self.nullspace_tolerance_rad,
                rtol=0.0,
            )
        ):
            restarted = self.solve(
                side,
                target_position,
                target_quaternion,
                seed_q=home,
                continuity_q=home,
                _allow_home_restart=False,
            )
            if restarted.converged:
                return restarted
            result_score = max(
                result.pos_error_mm / (_POSITION_TOLERANCE_M * 1000.0),
                result.ori_error_deg / math.degrees(_ORIENTATION_TOLERANCE_RAD),
            )
            restarted_score = max(
                restarted.pos_error_mm / (_POSITION_TOLERANCE_M * 1000.0),
                restarted.ori_error_deg
                / math.degrees(_ORIENTATION_TOLERANCE_RAD),
            )
            if (restarted_score, restarted.home_distance_rad) < (
                result_score,
                result.home_distance_rad,
            ):
                return restarted
        return result


__all__ = ["AbsoluteFR3IK", "IKResult"]
