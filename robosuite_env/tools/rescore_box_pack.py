# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Score a recorded box-pack run again from its state log, without running it again.

Two rest tests are available. ``old`` is instantaneous speed under a limit for
a run of samples; ``new`` is displacement over a window (see
robosuite_env/tools/_box_pack_quiet.py). They do not select or change the
current rollout success criterion. Every input either rule needs is in the
run's own state log.

Each row of ``logs/sim_log.jsonl`` carries, for one frame:

* the scored object's pose (``cup_pos`` / ``cup_quat`` -- the name is from the
  rig's first task and means the object under test, whichever it is);
* the criterion's conditions for that frame, including whether the object was
  placed, clear of the robot and free of a rule-grasp weld;
* the geometric dwell counter as it stood (``metrics.stability``), and whether
  the run's speed test passed (``conditions.object_velocity_quiet``).

The containment geometry is not recomputed here -- that would need the model,
and this tool would have to say so rather than guess. What is recomputed is the
dwell, and it has to be: the counter the run kept only advanced on frames the
old rule already called still, speed test and all, so reusing it would carry
the speed test into the new score. The dwell is therefore counted again from
the geometric conditions alone -- placed, clear of the robot, no weld -- held
continuously for the same two seconds.

Two stages, and the first is the point:

* ``--rule old`` replays the rule the run was scored under. If that does not
  reproduce the verdicts the run recorded, this path is not sound for that run
  and the answer is to re-run it. ``--verify`` does exactly this comparison.
* ``--rule new`` applies the window.

The state log holds one row per frame (10 Hz), while the criterion ticked at
the control rate (50 Hz), so the window here is sampled five times more coarsely
than it would be live. It is a window in simulation time either way, and a 2 s
window spans 21 rows rather than 101 ticks. Two differences from a fresh run
follow, and both can only make a run look stiller here than it was: a wobble
that begins and ends inside one frame is invisible, and the logged pose is
rounded to five decimals (robosuite_env/rollout/metrics.py), so a verdict
within a micrometre of the 2 mm line is not decidable from the log at all.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from robosuite_env.tools._box_pack_quiet import (
    DEFAULT_ROTATION_RAD,
    DEFAULT_TRANSLATION_M,
    DEFAULT_WINDOW_SEC,
    PoseQuietWindow,
)

#: The state log names the scored object after the rig's first task.
OBJECT_POSITION_KEY = "cup_pos"
OBJECT_ORIENTATION_KEY = "cup_quat"


def _rows(run_dir: Path):
    log = run_dir / "logs" / "sim_log.jsonl"
    if not log.is_file():
        raise FileNotFoundError(f"{run_dir}: no state log at {log}")
    with log.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{log}:{number}: {error}") from error


def _run_completed(run_dir: Path) -> bool:
    """Did this run finish, or is its log the prefix of one that was killed?

    A run that never finished writes no sim_summary.json, but it does leave a
    state log, and that log scores perfectly happily -- at whatever frame the
    run was cut off. Two batches of the same config, killed at different
    points, then look like the same config giving different answers. That is
    not a difference in the run; it is a difference in when it was stopped.
    """

    return (run_dir / "sim_summary.json").is_file()


def _frame_hz(run_dir: Path, default=10.0):
    summary = run_dir / "sim_summary.json"
    if summary.is_file():
        recorded = json.loads(summary.read_text(encoding="utf-8"))
        for key in ("frame_hz_effective", "frame_hz"):
            value = recorded.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
    return float(default)


def rescore(
    run_dir: Path,
    *,
    rule: str,
    window_sec: float = DEFAULT_WINDOW_SEC,
    translation_m: float = DEFAULT_TRANSLATION_M,
    rotation_rad: float = DEFAULT_ROTATION_RAD,
):
    """Re-decide one run's success, frame by frame, from its state log."""

    frame_hz = _frame_hz(run_dir)
    window = PoseQuietWindow(
        window_sec=window_sec,
        translation_m=translation_m,
        rotation_rad=rotation_rad,
        fallback_hz=frame_hz,
    )
    required_samples = max(1, math.ceil(window_sec * frame_hz - 1e-12))
    frames = 0
    scored_rows = 0
    held_samples = 0
    final_success = False
    recorded_final_success = False
    window_translation_mm = None
    success_frames = []
    recorded_success_frames = []
    disagreements = []
    largest_translation_mm = 0.0
    for row in _rows(run_dir):
        scored = row.get("success")
        if not isinstance(scored, dict):
            continue
        frames += 1
        frame = int(row.get("frame", frames - 1))
        recorded_stability = (scored.get("metrics") or {}).get("stability") or {}
        if {"consecutive_samples", "required_samples"} <= set(recorded_stability):
            scored_rows += 1
        conditions = scored.get("conditions") or {}
        stability = (scored.get("metrics") or {}).get("stability") or {}
        # What the run itself decided, to compare against.
        if bool(scored.get("success")):
            recorded_success_frames.append(frame)
        recorded_final_success = bool(scored.get("success"))

        now = frame / frame_hz
        hold = (
            bool(conditions.get("object_placed"))
            and bool(conditions.get("object_clear_of_robot"))
            and bool(conditions.get("object_rule_grasp_inactive"))
        )
        # The dwell counts samples, as the criterion's own counter does: the
        # first held sample is one, so two seconds at 10 Hz is twenty rows.
        # Counting seconds from the first held sample would ask for one sample
        # more than a live run needs.
        held_samples = held_samples + 1 if hold else 0

        if rule == "old":
            # The counter the run kept, against the count it wanted. Its own
            # speed test is in the row, so this needs nothing recomputed.
            resting = int(stability.get("consecutive_samples", 0)) >= int(
                stability.get("required_samples", 1)
            )
        else:
            # Updated on every row, held or not, exactly as the criterion
            # does. Resetting it on a geometric failure would report 0.0 mm of
            # movement for an object that is in fact sliding off the table.
            report = window.update(
                "target",
                time_s=now,
                position=row[OBJECT_POSITION_KEY],
                quaternion=row.get(OBJECT_ORIENTATION_KEY),
            )
            if report["translation_mm"] is not None:
                window_translation_mm = float(report["translation_mm"])
                largest_translation_mm = max(
                    largest_translation_mm, window_translation_mm
                )
            resting = (
                hold
                and held_samples >= required_samples
                and bool(report["quiet"])
            )

        success = bool(
            resting
            and conditions.get("moved_only_by_named_hand")
            and conditions.get("box_on_table")
        )
        if success:
            success_frames.append(frame)
        final_success = success
        if rule == "old" and success != bool(scored.get("success")):
            disagreements.append(frame)

    return {
        "run": str(run_dir),
        "rule": rule,
        "completed": _run_completed(run_dir),
        "frames": frames,
        "frame_hz": frame_hz,
        # The batch counts the last frame's verdict (new_final_success), so
        # that is the headline; holding it at some point is worth knowing too,
        # and the two differ for a run that settles and then creeps.
        "success": final_success,
        "success_ever": bool(success_frames),
        "first_success_frame": success_frames[0] if success_frames else None,
        "success_frames": len(success_frames),
        "recorded_success": recorded_final_success,
        "recorded_success_ever": bool(recorded_success_frames),
        "recorded_success_frames": len(recorded_success_frames),
        # Nothing to reproduce is not the same as reproducing everything: a
        # truncated log, or one without the counter fields, must not pass the
        # gate this tool exists to be.
        "scored_rows": scored_rows,
        "reproduces_the_recorded_verdict": (
            None if rule != "old" else bool(scored_rows) and not disagreements
        ),
        "disagreeing_frames": disagreements[:10],
        "largest_window_translation_mm": (
            None if rule == "old" else round(largest_translation_mm, 4)
        ),
        "final_window_translation_mm": (
            None
            if rule == "old" or window_translation_mm is None
            else round(window_translation_mm, 4)
        ),
    }


def _run_dirs(paths):
    for raw in paths:
        path = Path(raw)
        if (path / "logs" / "sim_log.jsonl").is_file():
            yield path
            continue
        found = sorted(
            candidate.parent.parent
            for candidate in path.glob("*/run_*/logs/sim_log.jsonl")
        )
        if not found:
            found = sorted(
                candidate.parent.parent
                for candidate in path.glob("run_*/logs/sim_log.jsonl")
            )
        if not found:
            raise FileNotFoundError(f"{path}: no runs with a state log below it")
        yield from found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "runs",
        nargs="+",
        help="run directories, or a batch directory holding them",
    )
    parser.add_argument(
        "--rule",
        choices=("old", "new"),
        default="new",
        help="the rest test to apply (default: new)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="score under the old rule and check it reproduces what the run recorded",
    )
    parser.add_argument(
        "--include-unfinished",
        action="store_true",
        help="score runs that never finished, whose logs stop where they were killed",
    )
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument(
        "--translation-mm", type=float, default=DEFAULT_TRANSLATION_M * 1000.0
    )
    parser.add_argument("--rotation-deg", type=float, default=2.0)
    parser.add_argument("--json", type=Path, help="write the full report here")
    arguments = parser.parse_args(argv)

    rule = "old" if arguments.verify else arguments.rule
    reports = [
        rescore(
            run,
            rule=rule,
            window_sec=arguments.window_sec,
            translation_m=arguments.translation_mm / 1000.0,
            rotation_rad=math.radians(arguments.rotation_deg),
        )
        for run in _run_dirs(arguments.runs)
    ]
    unfinished = [report for report in reports if not report["completed"]]
    if unfinished and not arguments.include_unfinished:
        print(
            f"{len(unfinished)} of {len(reports)} runs never finished (no "
            "sim_summary.json). Their logs stop wherever the run was killed, "
            "so scoring them compares stopping points, not policies. Pass "
            "--include-unfinished to score them anyway."
        )
        for report in unfinished:
            print(f"  unfinished: {Path(report['run']).name} ({report['frames']} frames)")
        reports = [report for report in reports if report["completed"]]
        if not reports:
            print("nothing left to score")
            return 1
    scored = sum(report["success"] for report in reports)
    ever = sum(report["success_ever"] for report in reports)
    recorded = sum(report["recorded_success"] for report in reports)
    for report in reports:
        name = Path(report["run"]).name
        line = (
            f"{name}: {rule} rule -> {'success' if report['success'] else 'failure'}"
            f" (the run recorded {'success' if report['recorded_success'] else 'failure'})"
        )
        if rule == "new" and report["final_window_translation_mm"] is not None:
            line += f", moved {report['final_window_translation_mm']} mm over the last window"
        if report["success_ever"] and not report["success"]:
            line += f" (held it earlier, from frame {report['first_success_frame']})"
        if rule == "old" and not report["reproduces_the_recorded_verdict"]:
            line += (
                ", NO SCORED ROWS"
                if not report["scored_rows"]
                else f", DISAGREES at frames {report['disagreeing_frames']}"
            )
        print(line)
    print(
        f"{scored}/{len(reports)} at the last frame under the {rule} rule "
        f"({ever}/{len(reports)} at some frame); {recorded}/{len(reports)} as recorded"
    )
    if arguments.verify:
        sound = all(report["reproduces_the_recorded_verdict"] for report in reports)
        print(
            "the state log reproduces every recorded verdict"
            if sound
            else "the state log does NOT reproduce every recorded verdict: re-run instead"
        )
    if arguments.json:
        arguments.json.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    if arguments.verify:
        return 0 if sound else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
