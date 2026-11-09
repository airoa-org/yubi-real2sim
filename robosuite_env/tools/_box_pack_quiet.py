# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Displacement-window rule for box-pack log rescoring.

These constants and PoseQuietWindow implement the displacement-window rest test.
The live rollout success criteria do not import this analysis-only helper.
"""

from __future__ import annotations

from collections import deque
import math

import numpy as np

DEFAULT_WINDOW_SEC = 2.0
DEFAULT_TRANSLATION_M = 0.002
DEFAULT_ROTATION_RAD = math.radians(2.0)


class PoseQuietWindow:
    """Keeps the last ``window_sec`` of poses for one or more named bodies."""

    def __init__(
        self,
        *,
        window_sec: float = DEFAULT_WINDOW_SEC,
        translation_m: float = DEFAULT_TRANSLATION_M,
        rotation_rad: float = DEFAULT_ROTATION_RAD,
        fallback_hz: float = 50.0,
    ) -> None:
        for label, value, positive in (
            ("window_sec", window_sec, True),
            ("translation_m", translation_m, False),
            ("rotation_rad", rotation_rad, False),
            ("fallback_hz", fallback_hz, True),
        ):
            invalid = not math.isfinite(value) or (
                value <= 0.0 if positive else value < 0.0
            )
            if invalid:
                relation = "positive" if positive else "non-negative"
                raise ValueError(f"{label} must be finite and {relation}")
        self.window_sec = float(window_sec)
        self.translation_m = float(translation_m)
        self.rotation_rad = float(rotation_rad)
        self.fallback_hz = float(fallback_hz)
        self._samples: dict[str, deque] = {}
        self._ticks: dict[str, int] = {}

    def reset(self, name: str | None = None) -> None:
        if name is None:
            self._samples.clear()
            self._ticks.clear()
            return
        self._samples.pop(name, None)
        self._ticks.pop(name, None)

    def update(self, name, *, time_s, position, quaternion=None) -> dict:
        """Record one pose of ``name`` and report its movement over the window."""

        # A model without a clock still has an order; count this name's own
        # ticks at the nominal rate, so one body's samples cannot advance
        # another body's clock.
        ticks = self._ticks.get(name, 0) + 1
        self._ticks[name] = ticks
        now = ticks / self.fallback_hz if time_s is None else float(time_s)
        # Copied, not referenced: MuJoCo's d.xpos and d.xquat hand out views
        # onto its own arrays, and a window of views would every one of them
        # read as the current pose, so nothing would ever have moved.
        point = np.array(position, dtype=float).reshape(3)
        rotation = (
            None if quaternion is None
            else np.array(quaternion, dtype=float).reshape(4)
        )
        samples = self._samples.setdefault(name, deque())
        if samples and samples[-1][0] >= now:
            # A repeated or rewound clock is the same instant, not a new one:
            # replace it, so a criterion that ticks twice on one frame neither
            # grows the window without bound nor pretends time passed.
            samples[-1] = (now, point, rotation)
        else:
            samples.append((now, point, rotation))
        # Keep the newest sample at or before the cutoff, not just the ones
        # after it: that sample is the evidence that the body has held still
        # for the whole window. Dropping it would let three samples at 0.0,
        # 1.0 and 2.1 s certify a two-second window that only spans 1.1 s.
        horizon = now - self.window_sec
        while len(samples) > 2 and samples[1][0] <= horizon:
            samples.popleft()
        return self.report(name)

    def report(self, name) -> dict:
        """Movement over the window: the largest excursion from the pose now."""

        samples = self._samples.get(name)
        if not samples:
            return {
                "tracked_sec": 0.0,
                "window_sec": self.window_sec,
                "window_samples": 0,
                "translation_mm": None,
                "rotation_deg": None,
                "translation_limit_mm": round(self.translation_m * 1000.0, 6),
                "rotation_limit_deg": round(math.degrees(self.rotation_rad), 6),
                "quiet": False,
                "test": "displacement over the window, not instantaneous speed",
            }
        now, point, rotation = samples[-1]
        # The span actually covered by the samples in hand, which is what the
        # displacement below is measured over.
        span = now - samples[0][0]
        # The largest excursion, not the distance between the ends: a body that
        # travels and comes back has not been still.
        translation = max(
            float(np.linalg.norm(sample[1] - point)) for sample in samples
        )
        angle = 0.0
        if rotation is not None:
            for sample in samples:
                if sample[2] is None:
                    continue
                dot = abs(float(np.dot(sample[2], rotation)))
                norms = float(np.linalg.norm(sample[2]) * np.linalg.norm(rotation))
                if norms <= 0.0:
                    continue
                angle = max(
                    angle, 2.0 * math.acos(max(-1.0, min(1.0, dot / norms)))
                )
        quiet = bool(
            span >= self.window_sec - 1e-12
            and translation <= self.translation_m + 1e-12
            and angle <= self.rotation_rad + 1e-12
        )
        return {
            "tracked_sec": round(float(span), 6),
            "window_sec": self.window_sec,
            "window_samples": len(samples),
            "translation_mm": round(translation * 1000.0, 6),
            "rotation_deg": round(math.degrees(angle), 6),
            "translation_limit_mm": round(self.translation_m * 1000.0, 6),
            "rotation_limit_deg": round(math.degrees(self.rotation_rad), 6),
            "quiet": quiet,
            "test": "displacement over the window, not instantaneous speed",
        }

    def quiet(self, name) -> bool:
        return bool(self.report(name)["quiet"])
