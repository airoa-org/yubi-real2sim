# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Pure timing helpers shared by rollout runtime components and tests."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicTiming:
    requested_control_repeats: int
    control_repeats: int
    requested_subframes: int
    subframes: int
    render_repeats: tuple[int, ...]


@dataclass(frozen=True)
class PlaybackClock:
    """Exact one-record-per-frame clock for saved state replay."""

    frame_frequency_hz: float
    frame_dt_s: float
    output_fps: int


def exact_state_replay_playback_clock(frame_frequency_hz: float) -> PlaybackClock:
    """Resolve a lossless state-row playback clock.

    A state log already contains one final state per source action.  Unlike a
    live simulation frame, its presentation period must not be quantized to a
    multiple of the MuJoCo timestep.  MP4 encoding in this pipeline uses an
    integer FPS, so non-integral rates are rejected instead of rounded.
    """

    if isinstance(frame_frequency_hz, bool):
        raise ValueError("state replay frame rate must be a positive finite number")
    frequency = float(frame_frequency_hz)
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("state replay frame rate must be a positive finite number")
    nearest = int(round(frequency))
    if not math.isclose(frequency, float(nearest), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("state replay frame rate must be an integer")
    return PlaybackClock(
        frame_frequency_hz=frequency,
        frame_dt_s=1.0 / frequency,
        output_fps=nearest,
    )


def control_ticks_for_source_action(
    action_index: int,
    source_frequency_hz: float,
    *,
    control_frequency_hz: float = 50.0,
) -> int:
    """Return fixed-rate controller ticks assigned to one source action.

    A phase accumulator is represented as the difference between two cumulative
    floors. This preserves the exact average source period when the source rate
    does not divide the controller rate (for example, 15 Hz on a 50 Hz control
    loop yields ``3, 3, 4`` ticks repeatedly).
    """

    if isinstance(action_index, bool) or not isinstance(action_index, int):
        raise ValueError("action index must be a non-negative integer")
    if action_index < 0:
        raise ValueError("action index must be a non-negative integer")
    source = float(source_frequency_hz)
    control = float(control_frequency_hz)
    if not math.isfinite(source) or source <= 0.0:
        raise ValueError("source frequency must be a positive finite number")
    if not math.isfinite(control) or control <= 0.0:
        raise ValueError("control frequency must be a positive finite number")
    if source > control:
        raise ValueError("source frequency must not exceed control frequency")

    ticks_per_action = control / source
    epsilon = 1e-12
    start = math.floor(action_index * ticks_per_action + epsilon)
    stop = math.floor((action_index + 1) * ticks_per_action + epsilon)
    ticks = int(stop - start)
    if ticks < 1:
        raise AssertionError("fixed-rate phase accumulator emitted no control tick")
    return ticks


def nearest_control_ticks_for_source_action(
    action_index: int,
    source_frequency_hz: float,
    *,
    control_frequency_hz: float = 50.0,
) -> int:
    """Assign one source action using cumulative-nearest controller time.

    Every ideal source boundary is quantized to the nearest complete controller
    tick with explicit half-up rounding. Taking adjacent boundary differences
    preserves the long-run source cadence while keeping boundary phase error to
    at most half a controller period. The floor-based helper above remains the
    deployed schedule for fixed-rate control loops; this nearest-boundary
    variant is for FR3 policy rows.
    """

    # Reuse the established validation, including the guarantee that the
    # source rate cannot demand more than one action per controller tick.
    control_ticks_for_source_action(
        action_index,
        source_frequency_hz,
        control_frequency_hz=control_frequency_hz,
    )
    ticks_per_action = float(control_frequency_hz) / float(source_frequency_hz)
    epsilon = 1e-12

    def cumulative_ticks(boundary_index: int) -> int:
        return int(
            math.floor(boundary_index * ticks_per_action + 0.5 + epsilon)
        )

    ticks = cumulative_ticks(action_index + 1) - cumulative_ticks(action_index)
    if ticks < 1:
        raise AssertionError(
            "nearest fixed-rate phase accumulator emitted no control tick"
        )
    return ticks


def constant_control_ticks_per_source_action(
    source_frequency_hz: float,
    *,
    control_frequency_hz: float = 50.0,
) -> int | None:
    """Return an exact scalar tick count, or ``None`` for phase schedules.

    This is intended for legacy summary fields that can represent divisor
    rates such as 5 Hz or 10 Hz, but must not misreport a non-divisor schedule
    such as a 50/15 Hz ``3, 3, 4`` cadence as a constant three ticks.
    """

    control_ticks_for_source_action(
        0,
        source_frequency_hz,
        control_frequency_hz=control_frequency_hz,
    )
    ratio = float(control_frequency_hz) / float(source_frequency_hz)
    nearest = int(round(ratio))
    if math.isclose(ratio, float(nearest), rel_tol=0.0, abs_tol=1e-12):
        return nearest
    return None


def resolve_dynamic_timing(
    exec_steps: int,
    requested_control_repeats: int,
    requested_subframes: int,
    *,
    action_hz: float = 0.0,
    control_hz: float = 20.0,
) -> DynamicTiming:
    if requested_control_repeats < 0:
        raise ValueError("dynamic control repeats must be >= 0")
    if action_hz < 0:
        raise ValueError("action rate must be >= 0")
    if control_hz <= 0:
        raise ValueError("dynamic control rate must be > 0")
    if requested_control_repeats > 0:
        control_repeats = requested_control_repeats
    elif action_hz > 0:
        repeat_ratio = float(control_hz) / float(action_hz)
        control_repeats = int(round(repeat_ratio))
        if control_repeats < 1 or abs(repeat_ratio - control_repeats) > 1e-6:
            raise ValueError(
                "dynamic control rate must be an integer multiple of action rate "
                f"(control_hz={control_hz:g}, action_hz={action_hz:g})"
            )
    else:
        # Policy chunk length (--exec) is an inference scheduling parameter, not
        # a physical hold duration. One controller interval per source action is
        # the safe default when no action rate was requested explicitly.
        control_repeats = 1
    subframes = min(max(1, int(requested_subframes)), control_repeats)
    render_repeats = tuple(
        sorted(
            {
                min(
                    control_repeats - 1,
                    max(0, int(round((index + 1) * control_repeats / float(subframes))) - 1),
                )
                for index in range(subframes)
            }
        )
    )
    return DynamicTiming(
        requested_control_repeats=int(requested_control_repeats),
        control_repeats=control_repeats,
        requested_subframes=int(requested_subframes),
        subframes=subframes,
        render_repeats=render_repeats,
    )
