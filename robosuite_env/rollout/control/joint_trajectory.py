# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Small, deterministic rollout joint-space trajectory helpers.

The policy produces one Cartesian target every 100 ms. IK stays outside the
high-rate MuJoCo loop. Runtime targets use a stateful jerk-limited Ruckig
trajectory so position, velocity, and acceleration remain continuous across
10 Hz replans; the minimum-jerk helper remains available for the ready drive.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


RUCKIG_VERSION = "0.19.4"
MINIMUM_JERK_MAX_VELOCITY_COEFFICIENT = 1.875
MINIMUM_JERK_MAX_ACCELERATION_COEFFICIENT = 10.0 / np.sqrt(3.0)


def minimum_jerk_blend(u):
    """Return ``10u^3 - 15u^4 + 6u^5`` for clamped normalized time."""

    value = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


@dataclass(frozen=True)
class MinimumJerkSegment:
    commands: np.ndarray
    requested_goal: np.ndarray
    executed_goal: np.ndarray
    scale: float
    duration_s: float
    max_velocity_rad_s: float
    max_acceleration_rad_s2: float


@dataclass(frozen=True)
class JerkLimitedTargetUpdate:
    accepted: bool
    reason: str | None
    requested_target: np.ndarray
    active_target: np.ndarray
    trajectory_duration_s: float | None
    predicted_position_min: np.ndarray | None
    predicted_position_max: np.ndarray | None
    result_code: int | None


@dataclass(frozen=True)
class JerkLimitedTargetPreview:
    update: JerkLimitedTargetUpdate
    sample_times_s: np.ndarray
    positions: np.ndarray


@dataclass(frozen=True)
class JerkLimitedSample:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    active_target: np.ndarray
    result_code: int
    finished: bool
    new_calculation: bool
    trajectory_duration_s: float
    predicted_position_min: np.ndarray
    predicted_position_max: np.ndarray


def bounded_minimum_jerk_segment(
    start,
    goal,
    *,
    steps: int,
    duration_s: float,
    max_velocity_rad_s: float,
    max_acceleration_rad_s2: float,
) -> MinimumJerkSegment:
    """Build a common-scaled, joint-limit-agnostic minimum-jerk segment.

    A single scale is used for every joint so the Cartesian path implied by
    the IK solution is not distorted joint-by-joint.  Callers remain
    responsible for clipping ``start`` and ``goal`` to the robot's hard joint
    limits before invoking this function.
    """

    start = np.asarray(start, dtype=np.float64).reshape(-1)
    goal = np.asarray(goal, dtype=np.float64).reshape(start.shape)
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    if not np.isfinite(max_velocity_rad_s) or max_velocity_rad_s <= 0.0:
        raise ValueError(
            f"max_velocity_rad_s must be positive, got {max_velocity_rad_s}"
        )
    if not np.isfinite(max_acceleration_rad_s2) or max_acceleration_rad_s2 <= 0.0:
        raise ValueError(
            f"max_acceleration_rad_s2 must be positive, got {max_acceleration_rad_s2}"
        )
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
        raise ValueError("start and goal must contain only finite values")

    delta = goal - start
    max_delta = float(np.max(np.abs(delta), initial=0.0))
    scale = 1.0
    if max_delta > 0.0:
        velocity_scale = (
            float(max_velocity_rad_s)
            * float(duration_s)
            / (MINIMUM_JERK_MAX_VELOCITY_COEFFICIENT * max_delta)
        )
        acceleration_scale = (
            float(max_acceleration_rad_s2)
            * float(duration_s) ** 2
            / (MINIMUM_JERK_MAX_ACCELERATION_COEFFICIENT * max_delta)
        )
        scale = float(np.clip(min(velocity_scale, acceleration_scale), 0.0, 1.0))

    executed_goal = start + scale * delta
    blend = minimum_jerk_blend(
        np.arange(1, int(steps) + 1, dtype=np.float64) / float(steps)
    )
    commands = start[None, :] + blend[:, None] * (executed_goal - start)[None, :]
    executed_delta = float(
        np.max(np.abs(executed_goal - start), initial=0.0)
    )
    continuous_vmax = (
        MINIMUM_JERK_MAX_VELOCITY_COEFFICIENT
        * executed_delta
        / float(duration_s)
    )
    continuous_amax = (
        MINIMUM_JERK_MAX_ACCELERATION_COEFFICIENT
        * executed_delta
        / float(duration_s) ** 2
    )
    return MinimumJerkSegment(
        commands=commands,
        requested_goal=goal.copy(),
        executed_goal=executed_goal,
        scale=scale,
        duration_s=float(duration_s),
        max_velocity_rad_s=float(continuous_vmax),
        max_acceleration_rad_s2=float(continuous_amax),
    )


class StatefulJerkLimitedTrajectory:
    """State-preserving 50 Hz joint trajectory backed by local Ruckig OTG.

    The Community edition does not enforce position limits. Every proposed
    target is therefore calculated offline first, and its continuous position
    extrema must remain inside the supplied hard joint range before the live
    online generator is allowed to adopt it. An unsafe retarget leaves the
    previous, already-validated trajectory active.
    """

    def __init__(
        self,
        position,
        *,
        delta_time_s: float,
        max_velocity_rad_s,
        max_acceleration_rad_s2,
        max_jerk_rad_s3,
        position_lower,
        position_upper,
        velocity=None,
        acceleration=None,
        position_tolerance_rad: float = 1e-9,
    ) -> None:
        try:
            import ruckig
        except ImportError as exc:
            raise RuntimeError(
                f"joint_ik requires ruckig=={RUCKIG_VERSION}"
            ) from exc
        if getattr(ruckig, "__version__", None) != RUCKIG_VERSION:
            raise RuntimeError(
                f"joint_ik requires ruckig=={RUCKIG_VERSION}, got "
                f"{getattr(ruckig, '__version__', None)!r}"
            )

        self._ruckig_module = ruckig
        self.position = self._state_vector(position, "position")
        self.degrees_of_freedom = int(self.position.size)
        self.velocity = self._state_vector(
            np.zeros_like(self.position) if velocity is None else velocity,
            "velocity",
        )
        self.acceleration = self._state_vector(
            (
                np.zeros_like(self.position)
                if acceleration is None
                else acceleration
            ),
            "acceleration",
        )
        self.max_velocity = self._positive_limit_vector(
            max_velocity_rad_s, "max_velocity_rad_s"
        )
        self.max_acceleration = self._positive_limit_vector(
            max_acceleration_rad_s2, "max_acceleration_rad_s2"
        )
        self.max_jerk = self._positive_limit_vector(
            max_jerk_rad_s3, "max_jerk_rad_s3"
        )
        self.position_lower = self._state_vector(
            position_lower, "position_lower"
        )
        self.position_upper = self._state_vector(
            position_upper, "position_upper"
        )
        if np.any(self.position_lower >= self.position_upper):
            raise ValueError("position_lower must be smaller than position_upper")
        if (
            not np.isfinite(delta_time_s)
            or float(delta_time_s) <= 0.0
        ):
            raise ValueError(
                f"delta_time_s must be positive, got {delta_time_s}"
            )
        if (
            not np.isfinite(position_tolerance_rad)
            or float(position_tolerance_rad) < 0.0
        ):
            raise ValueError(
                "position_tolerance_rad must be finite and non-negative"
            )
        self.delta_time_s = float(delta_time_s)
        self.position_tolerance_rad = float(position_tolerance_rad)
        self._assert_state_within_limits()

        self._otg = ruckig.Ruckig(
            self.degrees_of_freedom, self.delta_time_s
        )
        self._input = self._make_input(self.position)
        self._output = ruckig.OutputParameter(self.degrees_of_freedom)
        self.active_target = self.position.copy()
        self.target_rejection_count = 0
        initial = self._calculate_candidate(self.active_target)
        if not initial.accepted:
            raise ValueError(
                "initial state has no joint-limit-safe braking trajectory: "
                f"{initial.reason}"
            )
        self.last_target_update = initial

    def _state_vector(self, value, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if hasattr(self, "position") and vector.shape != self.position.shape:
            raise ValueError(
                f"{name} must have shape {self.position.shape}, "
                f"got {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values")
        return vector.copy()

    def _positive_limit_vector(self, value, name: str) -> np.ndarray:
        raw = np.asarray(value, dtype=np.float64)
        if raw.ndim == 0:
            vector = np.full(self.position.shape, float(raw), dtype=np.float64)
        else:
            vector = self._state_vector(raw, name)
        if not np.all(np.isfinite(vector)) or np.any(vector <= 0.0):
            raise ValueError(
                f"{name} must contain only finite positive values"
            )
        return vector

    @property
    def ruckig_version(self) -> str:
        """Return the version of the module backing this live trajectory."""

        return str(self._ruckig_module.__version__)

    def _assert_state_within_limits(self) -> None:
        tolerance = self.position_tolerance_rad
        if np.any(self.position < self.position_lower - tolerance) or np.any(
            self.position > self.position_upper + tolerance
        ):
            raise ValueError("initial position lies outside hard joint limits")
        if np.any(np.abs(self.velocity) > self.max_velocity + 1e-12):
            raise ValueError("initial velocity exceeds max_velocity_rad_s")
        if np.any(
            np.abs(self.acceleration) > self.max_acceleration + 1e-12
        ):
            raise ValueError(
                "initial acceleration exceeds max_acceleration_rad_s2"
            )

    def _make_input(self, target):
        ruckig = self._ruckig_module
        parameters = ruckig.InputParameter(self.degrees_of_freedom)
        parameters.current_position = self.position.tolist()  # noqa: V101  # ruckig.InputParameter field
        parameters.current_velocity = self.velocity.tolist()  # noqa: V101  # ruckig.InputParameter field
        parameters.current_acceleration = self.acceleration.tolist()  # noqa: V101  # ruckig.InputParameter field
        parameters.target_position = np.asarray(
            target, dtype=np.float64
        ).tolist()
        parameters.target_velocity = np.zeros(  # noqa: V101  # ruckig.InputParameter field
            self.degrees_of_freedom, dtype=np.float64
        ).tolist()
        parameters.target_acceleration = np.zeros(  # noqa: V101  # ruckig.InputParameter field
            self.degrees_of_freedom, dtype=np.float64
        ).tolist()
        parameters.max_velocity = self.max_velocity.tolist()
        parameters.max_acceleration = self.max_acceleration.tolist()
        parameters.max_jerk = self.max_jerk.tolist()
        parameters.control_interface = ruckig.ControlInterface.Position  # noqa: V101  # ruckig.InputParameter field
        parameters.synchronization = ruckig.Synchronization.Time  # noqa: V101  # ruckig.InputParameter field
        parameters.duration_discretization = (  # noqa: V101  # ruckig.InputParameter field
            ruckig.DurationDiscretization.Discrete
        )
        return parameters

    @staticmethod
    def _constant_jerk_segment_extrema(
        position: float,
        velocity: float,
        acceleration: float,
        jerk: float,
        duration_s: float,
    ) -> tuple[float, float]:
        """Return exact position extrema on one constant-jerk interval."""

        values = [
            float(position),
            float(
                position
                + velocity * duration_s
                + 0.5 * acceleration * duration_s**2
                + jerk * duration_s**3 / 6.0
            ),
        ]
        roots: list[float] = []
        machine_epsilon = np.finfo(np.float64).eps
        time_tolerance = 64.0 * machine_epsilon * max(
            1.0,
            abs(float(duration_s)),
        )
        if float(jerk) == 0.0:
            if float(acceleration) != 0.0:
                roots.append(-float(velocity) / float(acceleration))
        else:
            discriminant = (
                float(acceleration) ** 2
                - 2.0 * float(jerk) * float(velocity)
            )
            discriminant_tolerance = 64.0 * machine_epsilon * max(
                1.0,
                float(acceleration) ** 2,
                abs(2.0 * float(jerk) * float(velocity)),
            )
            if discriminant >= -discriminant_tolerance:
                root_term = np.sqrt(max(0.0, discriminant))
                roots.extend(
                    (
                        (-float(acceleration) - root_term) / float(jerk),
                        (-float(acceleration) + root_term) / float(jerk),
                    )
                )
        for root in roots:
            if (
                -time_tolerance
                <= root
                <= float(duration_s) + time_tolerance
            ):
                time_s = float(np.clip(root, 0.0, duration_s))
                values.append(
                    float(
                        position
                        + velocity * time_s
                        + 0.5 * acceleration * time_s**2
                        + jerk * time_s**3 / 6.0
                    )
                )
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                "Ruckig profile produced non-finite segment extrema"
            )
        return min(values), max(values)

    @classmethod
    def _brake_profile_extrema(cls, brake) -> tuple[float, float]:
        """Evaluate every public constant-jerk brake interval exactly."""

        durations = np.asarray(brake.t, dtype=np.float64).reshape(-1)
        jerks = np.asarray(brake.j, dtype=np.float64).reshape(-1)
        if durations.shape != jerks.shape or durations.size == 0:
            raise RuntimeError("Ruckig returned an invalid brake profile")
        position = float(brake.p[0])
        velocity = float(brake.v[0])
        acceleration = float(brake.a[0])
        minimum = maximum = position
        for duration_s, jerk in zip(durations, jerks):
            if not np.isfinite(duration_s) or duration_s < 0.0:
                raise RuntimeError(
                    "Ruckig returned an invalid brake duration"
                )
            segment_min, segment_max = cls._constant_jerk_segment_extrema(
                position,
                velocity,
                acceleration,
                float(jerk),
                float(duration_s),
            )
            minimum = min(minimum, segment_min)
            maximum = max(maximum, segment_max)
            position = float(
                position
                + velocity * duration_s
                + 0.5 * acceleration * duration_s**2
                + jerk * duration_s**3 / 6.0
            )
            velocity = float(
                velocity
                + acceleration * duration_s
                + 0.5 * jerk * duration_s**2
            )
            acceleration = float(acceleration + jerk * duration_s)
        return minimum, maximum

    @classmethod
    def _main_profile_extrema(cls, profile) -> tuple[float, float]:
        """Evaluate all seven public Ruckig profile intervals exactly."""

        durations = np.asarray(profile.t, dtype=np.float64).reshape(-1)
        jerks = np.asarray(profile.j, dtype=np.float64).reshape(-1)
        positions = np.asarray(profile.p, dtype=np.float64).reshape(-1)
        velocities = np.asarray(profile.v, dtype=np.float64).reshape(-1)
        accelerations = np.asarray(profile.a, dtype=np.float64).reshape(-1)
        if (
            durations.shape != jerks.shape
            or positions.size != durations.size + 1
            or velocities.size != durations.size + 1
            or accelerations.size != durations.size + 1
        ):
            raise RuntimeError("Ruckig returned an invalid motion profile")
        minimum = float(np.min(positions))
        maximum = float(np.max(positions))
        for index, (duration_s, jerk) in enumerate(
            zip(durations, jerks)
        ):
            if not np.isfinite(duration_s) or duration_s < 0.0:
                raise RuntimeError(
                    "Ruckig returned an invalid profile duration"
                )
            segment_min, segment_max = cls._constant_jerk_segment_extrema(
                float(positions[index]),
                float(velocities[index]),
                float(accelerations[index]),
                float(jerk),
                float(duration_s),
            )
            minimum = min(minimum, segment_min)
            maximum = max(maximum, segment_max)
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise RuntimeError("Ruckig returned non-finite motion extrema")
        return minimum, maximum

    def _trajectory_extrema(self, trajectory):
        """Return continuous extrema, including interior velocity roots.

        Ruckig 0.19.4 Community's ``position_extrema`` omits some interior
        roots where velocity changes sign inside a constant-acceleration
        interval. The public profile coefficients are pinned by version and
        let us evaluate those cubic position intervals exactly.
        """

        bounds = trajectory.position_extrema
        if len(bounds) != self.degrees_of_freedom:
            raise RuntimeError(
                "Ruckig returned the wrong number of position extrema"
            )
        minimum = np.asarray([bound.min for bound in bounds], dtype=np.float64)
        maximum = np.asarray([bound.max for bound in bounds], dtype=np.float64)
        profiles = trajectory.profiles
        for section_index, section in enumerate(profiles):
            if len(section) != self.degrees_of_freedom:
                raise RuntimeError(
                    "Ruckig returned the wrong number of motion profiles"
                )
            for index, profile in enumerate(section):
                profile_min, profile_max = self._main_profile_extrema(
                    profile
                )
                minimum[index] = min(minimum[index], profile_min)
                maximum[index] = max(maximum[index], profile_max)
                if (
                    section_index == 0
                    and float(profile.brake.duration) > 0.0
                ):
                    brake_min, brake_max = self._brake_profile_extrema(
                        profile.brake
                    )
                    minimum[index] = min(minimum[index], brake_min)
                    maximum[index] = max(maximum[index], brake_max)
        if not np.all(np.isfinite(minimum)) or not np.all(
            np.isfinite(maximum)
        ):
            raise RuntimeError("Ruckig returned non-finite position extrema")
        return minimum, maximum

    def _extrema_are_safe(self, minimum, maximum) -> bool:
        tolerance = self.position_tolerance_rad
        return bool(
            np.all(minimum >= self.position_lower - tolerance)
            and np.all(maximum <= self.position_upper + tolerance)
        )

    def _calculate_candidate_trajectory(self, target):
        requested = self._state_vector(target, "target")
        if np.any(requested < self.position_lower) or np.any(
            requested > self.position_upper
        ):
            return (
                JerkLimitedTargetUpdate(
                    accepted=False,
                    reason="target_outside_hard_joint_limits",
                    requested_target=requested,
                    active_target=self.active_target.copy(),
                    trajectory_duration_s=None,
                    predicted_position_min=None,
                    predicted_position_max=None,
                    result_code=None,
                ),
                None,
            )
        parameters = self._make_input(requested)
        trajectory = self._ruckig_module.Trajectory(
            self.degrees_of_freedom
        )
        try:
            result = self._otg.calculate(parameters, trajectory)
        except self._ruckig_module.RuckigError as exc:
            return (
                JerkLimitedTargetUpdate(
                    accepted=False,
                    reason=f"ruckig_calculation_error:{exc}",
                    requested_target=requested,
                    active_target=self.active_target.copy(),
                    trajectory_duration_s=None,
                    predicted_position_min=None,
                    predicted_position_max=None,
                    result_code=None,
                ),
                None,
            )
        result_code = int(result)
        if result_code < 0:
            return (
                JerkLimitedTargetUpdate(
                    accepted=False,
                    reason=f"ruckig_result:{result_code}",
                    requested_target=requested,
                    active_target=self.active_target.copy(),
                    trajectory_duration_s=None,
                    predicted_position_min=None,
                    predicted_position_max=None,
                    result_code=result_code,
                ),
                None,
            )
        minimum, maximum = self._trajectory_extrema(trajectory)
        if not self._extrema_are_safe(minimum, maximum):
            return (
                JerkLimitedTargetUpdate(
                    accepted=False,
                    reason="predicted_position_extrema_outside_hard_limits",
                    requested_target=requested,
                    active_target=self.active_target.copy(),
                    trajectory_duration_s=float(trajectory.duration),
                    predicted_position_min=minimum,
                    predicted_position_max=maximum,
                    result_code=result_code,
                ),
                trajectory,
            )
        return (
            JerkLimitedTargetUpdate(
                accepted=True,
                reason=None,
                requested_target=requested,
                active_target=requested.copy(),
                trajectory_duration_s=float(trajectory.duration),
                predicted_position_min=minimum,
                predicted_position_max=maximum,
                result_code=result_code,
            ),
            trajectory,
        )

    def _calculate_candidate(self, target) -> JerkLimitedTargetUpdate:
        update, _trajectory = self._calculate_candidate_trajectory(target)
        return update

    def preview_target(
        self,
        target,
        *,
        sample_period_s: float,
        horizon_s: float | None = None,
    ) -> JerkLimitedTargetPreview:
        """Sample a candidate Ruckig profile without adopting or advancing it."""

        if (
            not np.isfinite(sample_period_s)
            or float(sample_period_s) <= 0.0
        ):
            raise ValueError("sample_period_s must be finite and positive")
        if horizon_s is not None and (
            not np.isfinite(horizon_s) or float(horizon_s) <= 0.0
        ):
            raise ValueError("horizon_s must be finite and positive")
        update, trajectory = self._calculate_candidate_trajectory(target)
        if not update.accepted or trajectory is None:
            return JerkLimitedTargetPreview(
                update=update,
                sample_times_s=np.empty(0, dtype=np.float64),
                positions=np.empty(
                    (0, self.degrees_of_freedom),
                    dtype=np.float64,
                ),
            )
        duration_s = float(trajectory.duration)
        sampled_duration_s = min(
            duration_s,
            float(horizon_s) if horizon_s is not None else duration_s,
        )
        if sampled_duration_s <= 0.0:
            times = np.empty(0, dtype=np.float64)
        else:
            period_s = float(sample_period_s)
            sample_count = int(
                np.floor(sampled_duration_s / period_s + 1e-12)
            )
            times = (
                np.arange(1, sample_count + 1, dtype=np.float64)
                * period_s
            )
            if (
                times.size == 0
                or sampled_duration_s - float(times[-1]) > 1e-12
            ):
                times = np.append(times, sampled_duration_s)
        positions = np.empty(
            (len(times), self.degrees_of_freedom),
            dtype=np.float64,
        )
        for index, time_s in enumerate(times):
            state = trajectory.at_time(float(time_s))
            positions[index] = np.asarray(
                state[0],
                dtype=np.float64,
            ).reshape(self.position.shape)
        return JerkLimitedTargetPreview(
            update=update,
            sample_times_s=times,
            positions=positions,
        )

    def set_target(self, target) -> JerkLimitedTargetUpdate:
        """Adopt a safe target or continue the previous safe trajectory."""
        update = self._calculate_candidate(target)
        if update.accepted:
            self.active_target = update.active_target.copy()
            self._input.target_position = self.active_target.tolist()
            self._input.target_velocity = np.zeros(  # noqa: V101  # ruckig.InputParameter field
                self.degrees_of_freedom, dtype=np.float64
            ).tolist()
            self._input.target_acceleration = np.zeros(  # noqa: V101  # ruckig.InputParameter field
                self.degrees_of_freedom, dtype=np.float64
            ).tolist()
        else:
            self.target_rejection_count += 1
        self.last_target_update = update
        return update

    def reject_target(
        self,
        target,
        *,
        reason: str,
    ) -> JerkLimitedTargetUpdate:
        """Record an external safety rejection without changing the plan.

        IK convergence and workspace collision checks live outside Ruckig.
        Their rejection must preserve the current q / qdot / qddot and the
        previous validated active target exactly, rather than commanding an
        abrupt stop or adopting the rejected best-effort target.
        """

        requested = self._state_vector(target, "target")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        update = JerkLimitedTargetUpdate(
            accepted=False,
            reason=reason.strip(),
            requested_target=requested,
            active_target=self.active_target.copy(),
            trajectory_duration_s=None,
            predicted_position_min=None,
            predicted_position_max=None,
            result_code=None,
        )
        self.target_rejection_count += 1
        self.last_target_update = update
        return update

    def step(self) -> JerkLimitedSample:
        """Advance exactly one control period without resetting q/qd/qdd."""
        result = self._otg.update(self._input, self._output)
        result_code = int(result)
        if result_code < 0:
            raise RuntimeError(f"Ruckig online update failed: {result_code}")
        position = np.asarray(self._output.new_position, dtype=np.float64)
        velocity = np.asarray(self._output.new_velocity, dtype=np.float64)
        acceleration = np.asarray(
            self._output.new_acceleration, dtype=np.float64
        )
        jerk = np.asarray(self._output.new_jerk, dtype=np.float64)
        minimum, maximum = self._trajectory_extrema(self._output.trajectory)
        if not self._extrema_are_safe(minimum, maximum):
            raise RuntimeError(
                "Ruckig online trajectory violated prevalidated joint limits"
            )
        tolerance = self.position_tolerance_rad
        if np.any(position < self.position_lower - tolerance) or np.any(
            position > self.position_upper + tolerance
        ):
            raise RuntimeError("Ruckig command lies outside hard joint limits")
        if np.any(np.abs(velocity) > self.max_velocity + 1e-9):
            raise RuntimeError("Ruckig command exceeded velocity limit")
        if np.any(np.abs(acceleration) > self.max_acceleration + 1e-8):
            raise RuntimeError("Ruckig command exceeded acceleration limit")
        if np.any(np.abs(jerk) > self.max_jerk + 1e-7):
            raise RuntimeError("Ruckig command exceeded jerk limit")

        self._output.pass_to_input(self._input)
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.acceleration = acceleration.copy()
        return JerkLimitedSample(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            active_target=self.active_target.copy(),
            result_code=result_code,
            finished=bool(result == self._ruckig_module.Result.Finished),
            new_calculation=bool(self._output.new_calculation),
            trajectory_duration_s=float(self._output.trajectory.duration),
            predicted_position_min=minimum,
            predicted_position_max=maximum,
        )
