# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Pure rollout helpers for interpreting policy gripper targets."""
from __future__ import annotations

import math
from collections.abc import Mapping


# Widest angle either jaw can be commanded to. The policy target is clipped to
# it before it reaches the controller, so it also bounds the rule-grasp release
# edge: a threshold above this value can never be met. This module has no
# imports of its own, so both the control and build sides can read it without
# creating a cycle.
YUBI_JAW_MAX_RAD = 0.7854


def classify_policy_gripper_intent(
    target_rad: float,
    actual_rad: float,
    epsilon_rad: float,
) -> dict[str, float | str]:
    """Classify the commanded jaw direction from target tracking error.

    Yubi jaw angles increase while opening, so a target below the measured
    angle is a closing command and a target above it is an opening command.
    """
    target = float(target_rad)
    actual = float(actual_rad)
    epsilon = float(epsilon_rad)
    if not all(math.isfinite(value) for value in (target, actual, epsilon)):
        raise ValueError("gripper target, actual angle, and epsilon must be finite")
    if epsilon < 0.0:
        raise ValueError("gripper intent epsilon must be >= 0")

    error = target - actual
    on_lower_boundary = math.isclose(error, -epsilon, rel_tol=1e-12, abs_tol=1e-12)
    on_upper_boundary = math.isclose(error, epsilon, rel_tol=1e-12, abs_tol=1e-12)
    if error < -epsilon and not on_lower_boundary:
        direction = "closing"
    elif error > epsilon and not on_upper_boundary:
        direction = "opening"
    else:
        direction = "neutral"
    return {
        "value": round(error, 9),
        "direction": direction,
        "source": "policy_target_error",
        "target_rad": round(target, 9),
        "actual_rad": round(actual, 9),
        "epsilon_rad": round(epsilon, 9),
    }


def latch_policy_gripper_intents(
    targets_rad: Mapping[str, float],
    actuals_rad: Mapping[str, float],
    epsilon_rad: float,
) -> dict[str, dict[str, float | str | bool]]:
    """Snapshot policy intent once for every hand in a source action."""
    latched = {}
    for hand,target in targets_rad.items():
        intent = classify_policy_gripper_intent(
            target,
            actuals_rad[hand],
            epsilon_rad,
        )
        intent.update({"latched": True, "sample_phase": "action_start"})
        latched[hand] = intent
    return latched


def evaluate_contact_angle_grasp(
    policy_target_rad: float,
    contact_angles_rad: Mapping[str, float],
    actual_angles_rad: Mapping[str, float],
    epsilon_rad: float,
    release_epsilon_rad: float | None = None,
    jaw_max_rad: float | None = None,
) -> dict[str, object]:
    """Evaluate the contact-angle rule-grasp predicate.

    A grasp may start only when the scalar policy target commands both fingers
    below their measured inner-pad contact angles.  Once active, the grasp no
    longer holds when the policy target commands a finger above its saved
    contact angle.  Increasing YUBI joint angle means opening.

    Both edges test the same quantity -- the policy target against the contact
    angle frozen at latch time -- so the pair forms one hysteresis band
    (``-epsilon`` to latch, ``+release_epsilon`` to let go).  Releasing on the
    *measured* angle instead cannot work while the grasp is held kinematically:
    the held object stops yielding, the fingers are pushed open by the contact
    they can no longer compress, and the measured angle crosses the release
    threshold on its own.  Latching would then be what ends the grasp.  The
    measured angles stay in the result for telemetry only.

    ``jaw_max_rad`` caps the release edge at the widest angle the jaws can be
    commanded to.  Without it a grasp latched near that limit is unreleasable:
    the target is clipped to the limit before it reaches ``contact +
    release_epsilon``, so the policy can ask for the jaws to open as far as they
    go and the object still never lets go.  Capping makes a full-open command
    always release, and by construction it can only ever lower the threshold.
    """
    target = float(policy_target_rad)
    epsilon = float(epsilon_rad)
    release_epsilon = (
        epsilon
        if release_epsilon_rad is None
        else float(release_epsilon_rad)
    )
    if epsilon < 0.0 or not math.isfinite(epsilon):
        raise ValueError("gripper contact-angle epsilon must be finite and >= 0")
    if release_epsilon < 0.0 or not math.isfinite(release_epsilon):
        raise ValueError(
            "gripper release epsilon must be finite and >= 0"
        )
    if not math.isfinite(target):
        raise ValueError("gripper policy target must be finite")
    jaw_max = None if jaw_max_rad is None else float(jaw_max_rad)
    if jaw_max is not None and not math.isfinite(jaw_max):
        raise ValueError("gripper jaw maximum must be finite")

    contact_keys = set(contact_angles_rad)
    actual_keys = set(actual_angles_rad)
    if not contact_keys or contact_keys != actual_keys:
        raise ValueError("contact and actual finger angles must have the same non-empty keys")

    contacts: dict[str, float] = {}
    actuals: dict[str, float] = {}
    closing_sides: dict[str, bool] = {}
    opened_sides: dict[str, bool] = {}
    release_thresholds: dict[str, float] = {}
    releasable_sides: dict[str, bool] = {}
    for side in sorted(contact_keys):
        contact = float(contact_angles_rad[side])
        actual = float(actual_angles_rad[side])
        if not math.isfinite(contact) or not math.isfinite(actual):
            raise ValueError("gripper contact and actual angles must be finite")
        contacts[side] = contact
        actuals[side] = actual
        release_at = contact + release_epsilon
        if jaw_max is not None:
            release_at = min(release_at, jaw_max)
        release_thresholds[side] = release_at
        # A latch is only meaningful if the policy can still command the jaws
        # past its release edge. Latched at the stop, contact + release_epsilon
        # is clipped back onto the stop and nothing short of the exact maximum
        # ever lets go -- a grasp with the jaws wide open that cannot be undone.
        releasable_sides[side] = release_at > contact + epsilon
        closing_sides[side] = target < contact - epsilon
        # Opening past the release edge normally has to be strict. At the jaw
        # stop it cannot be: the policy target is clipped to the same stop, so
        # a strict > could never be satisfied and the grasp would never let go.
        at_jaw_stop = jaw_max is not None and release_at >= jaw_max
        opened_sides[side] = (
            target >= release_at if at_jaw_stop else target > release_at
        )

    return {
        "policy_target_rad": target,
        "contact_angles_rad": contacts,
        "actual_angles_rad": actuals,
        "epsilon_rad": epsilon,
        "release_epsilon_rad": release_epsilon,
        "jaw_max_rad": jaw_max,
        "release_thresholds_rad": release_thresholds,
        "closing_below_contact": all(closing_sides.values()),
        "release_reachable": all(releasable_sides.values()),
        "releasable_sides": releasable_sides,
        "opened_above_contact": any(opened_sides.values()),
        "closing_sides": closing_sides,
        "opened_sides": opened_sides,
    }
