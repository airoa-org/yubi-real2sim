# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Small registry for simulation success criteria.

Adding a new task should be a local change here plus a new criterion class with
``evaluate()`` returning a JSON-serializable dict.
"""

from .box_pack import BoxPackCriterion
from .cup_on_plate import CupOnPlateCriterion


_CRITERIA = {
    "object_in_box": BoxPackCriterion,
    "cup_on_plate": CupOnPlateCriterion,
}

_OFF = {"", "none", "off", "false", "0", "no"}


def available_criteria():
    return tuple(sorted(_CRITERIA))


def make_success_criterion(name, *, mujoco, model, data, options=None):
    raw = "none" if name is None else str(name).strip()
    lower = raw.lower()
    if lower in _OFF:
        return None
    raw = "cup_on_plate" if lower == "auto" else lower
    try:
        cls = _CRITERIA[raw]
    except KeyError as exc:
        raise ValueError(f"unknown success criterion {raw!r}; choices: auto, none, {', '.join(available_criteria())}") from exc
    options = options or {}
    return cls(mujoco=mujoco, model=model, data=data, options=options)


def summarize_success_records(records, criterion=None):
    pairs = [(int(r.get("frame", i)), r.get("success")) for i, r in enumerate(records) if isinstance(r.get("success"), dict)]
    if not pairs:
        return {}

    frames = [p[0] for p in pairs]
    evals = [p[1] for p in pairs]
    ok = [bool(e.get("success")) for e in evals]
    first = next((frame for frame, v in zip(frames, ok) if v), None)
    last = next((frame for frame, v in zip(reversed(frames), reversed(ok)) if v), None)
    run = best_run = 0
    for v in ok:
        run = run + 1 if v else 0
        best_run = max(best_run, run)

    final = evals[-1]
    final_success = bool(final.get("success"))
    out = {
        "new_final_success": final_success,
        "success": final_success,
        "success_ever": any(ok),
        "success_first_frame": first,
        "success_last_frame": last,
        "success_hold_frames": best_run,
        "success_final": final,
    }
    if any("strict_success" in e for e in evals):
        strict_ok = [bool(e.get("strict_success")) for e in evals]
        strict_first = next((frame for frame, v in zip(frames, strict_ok) if v), None)
        strict_last = next((frame for frame, v in zip(reversed(frames), reversed(strict_ok)) if v), None)
        strict_run = strict_best_run = 0
        for v in strict_ok:
            strict_run = strict_run + 1 if v else 0
            strict_best_run = max(strict_best_run, strict_run)
        out.update(
            {
                "strict_success": bool(final.get("strict_success")),
                "strict_success_ever": any(strict_ok),
                "strict_success_first_frame": strict_first,
                "strict_success_last_frame": strict_last,
                "strict_success_hold_frames": strict_best_run,
            }
        )
    if any("partial_success" in e for e in evals):
        partial = [bool(e.get("partial_success")) for e in evals]
        out["partial_success"] = bool(final.get("partial_success"))
        out["partial_success_ever"] = any(partial)
    score_pairs = [
        (frame, e.get("score"))
        for frame, e in zip(frames, evals)
        if isinstance(e.get("score"), dict) and e["score"].get("enabled", True)
    ]
    if score_pairs:
        score_frames = [p[0] for p in score_pairs]
        scores = [int(p[1].get("points", 0)) for p in score_pairs]
        max_scores = [int(p[1].get("max_points", 0)) for p in score_pairs]
        out.update(
            {
                "score_total": int(sum(scores)),
                "score_max_total": int(sum(max_scores)),
                "score_final": int(scores[-1]),
                "score_best": int(max(scores)),
                "score_mean": round(float(sum(scores)) / max(len(scores), 1), 4),
                "score_frames": int(len(scores)),
                "score_one_point_frames": [int(frame) for frame, score in zip(score_frames, scores) if score == 1],
                "score_two_point_frames": [int(frame) for frame, score in zip(score_frames, scores) if score >= 2],
                "score_condition": score_pairs[-1][1].get("condition"),
            }
        )
    if criterion is not None:
        out["success_criterion"] = criterion.describe()
        note = getattr(criterion, "contract_note", None)
        if note:
            # What changed about the bar itself, so a reader comparing two runs
            # is told rather than left to infer it from the describe() keys.
            out["success_criterion_note"] = str(note)
    return out
