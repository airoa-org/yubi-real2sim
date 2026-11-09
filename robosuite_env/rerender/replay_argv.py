# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Recorded worker argv parsing and state-log replay argv rewriting."""


from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
from pathlib import Path

from robosuite_env.rerender.cadence import _fr3_policy_action_playback_contract
from real2sim.config.paths import RUN_MANIFEST_FILE
from real2sim.config.task_objects import validate_task_objects
from real2sim.runner.argv import SIM_WORKER_MODULE, worker_command_head_length
from real2sim.wrist_profiles import WRIST_BACKENDS
from robosuite_env.contracts.provenance import sealed_file_digests, sha256_file
from robosuite_env.robots.profiles import normalize_robot_name
from robosuite_env.rerender.common import (
    REPO_ROOT,
    RerenderError,
    _canonical_video_path,
    _last_option_value,
)
from robosuite_env.rollout.replay.recorded_wrist import (
    UNSUPPORTED_RECORDED_WRIST_BACKEND,
    RecordedWristError,
    recorded_gs_kb4_near_cull,
    recorded_wrist_backend,
    recorded_wrist_profile,
)


# Options that only make sense for the live policy rollout.
_DROP_WITH_VALUE = (
    "--policy",
    "--replay",
    "--ik-replay",
    "--policy-batch-server",
    "--policy-batch-client-id",
    "--policy-batch-timeout-sec",
    "--policy-action-hz",
    # Retired worker options may still appear in recorded RUN_COMMAND.md files.
    "--blender-export",
    "--arm-ctrl",
    "--ee-ctrl",
    # Retired; keep RUN_COMMAND.md files recorded before the retirement replayable.
    "--fr3-contact-control",
    "--ik-url",
    # State-log replay owns the complete trajectory. Policy scheduling and
    # generation-time initialization are both irrelevant and some combinations
    # are explicitly rejected when no live policy source is present.
    "--rf-overlap",
    "--action-overlap",
    "--rf-overlap-sampler",
    "--rf-overlap-prev-weight",
    "--rf-overlap-step-shift",
    "--policy-image-source",
    "--policy-pose-source",
    "--policy-image-episode",
    "--policy-image-index",
    "--init-state-in",
    "--init-state-out",
    "--mimicgen-init-episode",
    "--mimicgen-init-state-log-root",
    "--dump-obs",
    "--anchor-dump",
    "--state-dump-dir",
    "--state-dump-frames",
    # These are launcher-owned attestations for the original official worker.
    # A state-log replay is a diagnostic derivative and cannot truthfully reuse
    # the recorded worker's manifest or worker contract.
    "--official-run-manifest",
    "--official-worker-contract",
    "--official-scene-tag",
)


_DROP_FLAGS = (
    "--defer-overview-render",
    "--no-render",
    "--strict-assets",
    "--policy-image-loop",
    "--policy-freeze-left",
    "--init-state-only",
    "--state-dump-on-rule-grasp",
    "--state-dump-on-success",
    "--grasp-test",
)


# Options the replay pass must own.
_REWRITE_WITH_VALUE = (
    "--gs-port",
    "--gs-server-token",
    "--out",
    "--logdir",
    "--summary-path",
    "--render-lock",
    "--max-wall-sec",
    "--dynamic-subframes",
    "--state-log-replay",
    "--state-replay-policy-obs-dir",
    "--state-replay-policy-obs-display",
)


# Recorded third-person viewpoint options a caller may override post hoc.
_VIEW_OVERRIDE_OPTIONS = (
    "--view-azimuth",
    "--view-elevation",
    "--view-distance",
    "--view-lookat",
)


def _without_option(argv: list[str], option: str) -> tuple[list[str], int]:
    """Remove all separate/equals occurrences of one value-taking option."""

    result: list[str] = []
    removed = 0
    index = 0
    prefix = f"{option}="
    while index < len(argv):
        token = argv[index]
        if token == option:
            if index + 1 >= len(argv):
                raise RerenderError(f"{option} is missing its value")
            removed += 1
            index += 2
            continue
        if token.startswith(prefix):
            if not token[len(prefix) :]:
                raise RerenderError(f"{option} is missing its value")
            removed += 1
            index += 1
            continue
        result.append(token)
        index += 1
    return result, removed


def _recorded_option_values(argv: list[str], option: str) -> list[str]:
    """Read every occurrence before dropping a retired value-taking option."""

    values: list[str] = []
    for index, token in enumerate(argv):
        if token == option:
            if index + 1 == len(argv) or argv[index + 1].startswith("--"):
                raise RerenderError(f"{option} is missing its value")
            values.append(argv[index + 1])
        elif token.startswith(f"{option}="):
            values.append(token.partition("=")[2])
    return values


def _without_retired_quiet_options(argv: list[str]) -> list[str]:
    # These success thresholds never affect the saved trajectory or cup/box
    # visuals. Validate all occurrences as the old parser did before removing
    # them; never consume the next option as a missing numeric value.
    for option in (
        "--success-quiet-translation-mm",
        "--success-quiet-rotation-deg",
    ):
        for raw in _recorded_option_values(argv, option):
            try:
                value = float(raw)
            except ValueError as exc:
                raise RerenderError(f"{option} must be a finite number: {raw!r}") from exc
            if not math.isfinite(value):
                raise RerenderError(f"{option} must be a finite number: {raw!r}")
        argv, _ = _without_option(argv, option)
    return argv


def _validate_recorded_hue_summary(source_summary: dict | None) -> None:
    key = "plate_visual_texture_hue_ablation"
    if source_summary is None or key not in source_summary:
        return
    report = source_summary[key]
    prefix = f"recorded {key} cannot be replayed"
    if not isinstance(report, dict):
        raise RerenderError(f"{prefix}: expected an object")
    # These are the original report's requested, modulo-180 applied, and
    # degree shifts. A nonzero shift is unsupported even if changed is false
    # (e.g. a monochrome texture); state rows cannot restore texture bytes.
    for field in (
        "requested_hue_shift_opencv_bins",
        "hue_shift_opencv_bins",
        "hue_shift_degrees",
    ):
        value = report.get(field)
        if type(value) is not int:
            raise RerenderError(f"{prefix}: {field} must be an integer")
        if value != 0:
            raise RerenderError(
                f"{prefix}: nonzero {field}={value}; "
                "use the source run's original renderer"
            )
    for field in ("enabled", "changed"):
        if report.get(field) is not False:
            raise RerenderError(f"{prefix}: {field} must be false")

    # The old report always included both names for each pixel hash. A no-op
    # has four equal hashes, or four nulls when the plate texture is absent.
    hashes: list[str | None] = []
    for field in (
        "source_sha256",
        "result_sha256",
        "source_rgb_sha256",
        "effective_rgb_sha256",
    ):
        if field not in report:
            raise RerenderError(f"{prefix}: missing {field}")
        value = report[field]
        if value is not None:
            if not isinstance(value, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}", value
            ):
                raise RerenderError(f"{prefix}: {field} must be a SHA-256 or null")
            value = value.lower()
        hashes.append(value)
    if len(set(hashes)) != 1:
        raise RerenderError(f"{prefix}: source/result pixel hashes disagree")


def normalize_recorded_replay_options(
    argv: list[str], source_summary: dict | None = None
) -> list[str]:
    """Validate retired options, removing only values inert in state replay."""

    recorded_robot = _last_option_value(argv, "--robot")
    if recorded_robot is not None:
        try:
            normalize_robot_name(recorded_robot)
        except ValueError as exc:
            raise RerenderError(f"recorded --robot: {exc}") from exc
    argv = _without_retired_quiet_options(argv)
    option = "--plate-texture-hue-shift-opencv-bins"
    values: list[int] = []
    for raw in _recorded_option_values(argv, option):
        try:
            value = int(raw)
        except ValueError as exc:
            raise RerenderError(
                f"{option} must be an integer in [-179, 179]: {raw!r}"
            ) from exc
        if not -179 <= value <= 179:
            raise RerenderError(f"{option} must be an integer in [-179, 179]: {raw!r}")
        values.append(value)
    # argparse validated every occurrence, with only the last value taking
    # effect. An earlier nonzero value overridden by zero did not alter pixels.
    if values and values[-1] != 0:
        raise RerenderError(
            f"recorded nonzero {option}={values[-1]} cannot be replayed; "
            "use the source run's original renderer"
        )
    _validate_recorded_hue_summary(source_summary)
    argv, _ = _without_option(argv, option)
    return argv


def parse_run_command(run_dir: Path) -> tuple[dict[str, str], list[str]]:
    """Return (env, argv) recorded in the run's RUN_COMMAND.md bash fence."""

    text = (run_dir / "RUN_COMMAND.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    try:
        start = next(
            index for index, line in enumerate(lines) if line.strip() == "```bash"
        )
        stop = next(
            index
            for index in range(start + 1, len(lines))
            if lines[index].strip() == "```"
        )
    except StopIteration as exc:
        raise RerenderError(f"{run_dir}: RUN_COMMAND.md has no bash fence") from exc
    tokens = shlex.split(" ".join(lines[start + 1 : stop]))
    environment: dict[str, str] = {}
    index = 0
    while (
        index < len(tokens)
        and "=" in tokens[index]
        and not tokens[index].startswith("-")
    ):
        key, _, value = tokens[index].partition("=")
        if not key.isidentifier():
            break
        environment[key] = value
        index += 1
    argv = tokens[index:]
    if worker_command_head_length(argv) is None:
        raise RerenderError(f"{run_dir}: unexpected command head {argv[:3]}")
    return environment, argv


def resolve_view_overrides(arguments: argparse.Namespace) -> dict[str, str]:
    """Map requested third-person viewpoint overrides to worker CLI values.

    Every value defaults to ``None`` meaning "keep the recorded viewpoint".
    The returned mapping contains only the explicitly requested overrides,
    keyed by the worker option name.
    """

    overrides: dict[str, str] = {}
    for attribute, option in (
        ("view_azimuth", "--view-azimuth"),
        ("view_elevation", "--view-elevation"),
        ("view_distance", "--view-distance"),
    ):
        value = getattr(arguments, attribute, None)
        if value is None:
            continue
        try:
            overrides[option] = f"{float(value):g}"
        except (TypeError, ValueError) as exc:
            raise RerenderError(f"{option} must be a number: {value!r}") from exc
    lookat = getattr(arguments, "view_lookat", None)
    if lookat is not None:
        text = str(lookat).replace(",", " ").strip()
        if text:
            parts = text.split()
            try:
                valid = len(parts) == 3 and all(
                    math.isfinite(float(part)) for part in parts
                )
            except ValueError:
                valid = False
            if not valid:
                raise RerenderError(
                    f"--view-lookat expects three numbers 'x y z': {lookat!r}"
                )
            text = " ".join(parts)
        # An explicit empty string restores the worker's default lookat.
        overrides["--view-lookat"] = text
    return overrides


def _recorded_wrist_profile(source_summary: dict | None, sim_log: Path) -> str:
    """The run's recorded profile (``recorded_wrist.recorded_wrist_profile``), as a rerender error."""

    try:
        return recorded_wrist_profile(source_summary, sim_log)
    except RecordedWristError as exc:
        raise RerenderError(str(exc)) from exc


def _recorded_wrist_backend(source_summary: dict | None, sim_log: Path) -> str:
    """The run's recorded backend (``recorded_wrist.recorded_wrist_backend``), as a rerender error."""

    try:
        return recorded_wrist_backend(source_summary, sim_log)
    except RecordedWristError as exc:
        raise RerenderError(str(exc)) from exc


def _recorded_gs_kb4_near_cull(source_summary: dict | None, sim_log: Path) -> str:
    """The run's recorded KB4 near cull (``recorded_wrist.recorded_gs_kb4_near_cull``), as a rerender error."""

    try:
        return recorded_gs_kb4_near_cull(source_summary, sim_log)
    except RecordedWristError as exc:
        raise RerenderError(str(exc)) from exc


# The per-condition inputs (base placement, gripper calibration) moved from
# assets/<previous name>/ to assets/release/ unchanged. A recorded worker argv
# names them by absolute checkout path, so a run recorded before the move still
# names the previous directory, which is matched by its SHA-256. The moved file
# stands in for the recorded one only when its bytes hash to the digest the
# run's seal recorded for the recorded path, so a later edit of a release
# condition file is never replayed as the run's input.
_PREVIOUS_CONDITION_ASSET_DIR_SHA256 = (
    "ec6f5fcd3b94d3370066645801cac7522045fa74ead7367b1302cda3b8a99745"
)


def _moved_condition_asset(value: str, repo_root: Path = REPO_ROOT) -> str:
    """Return the assets/release/ path a recorded pre-move condition file moved to.

    Only a missing absolute path inside this checkout's previous condition
    directory is mapped, and only to an existing file; every other value is
    returned unchanged, so a replay still fails on a genuinely missing input.
    The mapping compares names only: :func:`_with_moved_condition_assets`
    checks the bytes before using it.
    """

    path = Path(value)
    if not path.is_absolute() or path.exists():
        return value
    try:
        parts = path.relative_to(repo_root).parts
    except ValueError:
        return value
    if (
        len(parts) < 3
        or parts[0] != "assets"
        or hashlib.sha256(parts[1].encode("utf-8")).hexdigest()
        != _PREVIOUS_CONDITION_ASSET_DIR_SHA256
    ):
        return value
    moved = repo_root.joinpath("assets", "release", *parts[2:])
    return str(moved) if moved.is_file() else value


def _run_commit(run_root: Path) -> str | None:
    """The code commit the run's run_manifest.json records, if any."""

    try:
        manifest = json.loads(
            (run_root / RUN_MANIFEST_FILE).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError):
        return None
    code = manifest.get("code") if isinstance(manifest, dict) else None
    commit = code.get("commit") if isinstance(code, dict) else None
    return commit if isinstance(commit, str) and commit else None


def _with_moved_condition_assets(
    argv: list[str], run_dir: Path, repo_root: Path = REPO_ROOT
) -> list[str]:
    """Point recorded pre-move condition paths at their sealed assets/release/ bytes.

    Every value (``--opt=value`` too) that :func:`_moved_condition_asset`
    maps is replaced only when the moved file's SHA-256 equals the digest the
    run's runtime source seal (at the run root, two levels above ``run_dir``)
    recorded for the recorded path. A different digest, or no recorded one,
    is refused and names the run's own commit, where the recorded path exists.
    """

    run_root = run_dir.parents[1]
    sealed: dict[str, str] | None = None

    def substitute(value: str) -> str:
        nonlocal sealed
        moved = _moved_condition_asset(value, repo_root)
        if moved == value:
            return value
        if sealed is None:
            sealed = sealed_file_digests(run_root)
        recorded = sealed.get(str(Path(value)))
        actual = sha256_file(Path(moved))
        if recorded == actual:
            return moved
        commit = _run_commit(run_root)
        replay_at = (
            f"replay it at the run's own commit {commit}"
            if commit is not None
            else f"its {RUN_MANIFEST_FILE} records no commit to replay it at"
        )
        reason = (
            f"the run's seal records no SHA-256 for {value}"
            if recorded is None
            else f"{moved} has SHA-256 {actual} and the run sealed {recorded}"
        )
        raise RerenderError(
            f"{run_dir}: recorded condition file {value} moved to {moved}, "
            f"but {reason}, so the moved file is not the run's input; {replay_at}"
        )

    rewritten = []
    for token in argv:
        option, separator, value = token.partition("=")
        if separator and option.startswith("--"):
            rewritten.append(option + separator + substitute(value))
        else:
            rewritten.append(substitute(token))
    return rewritten


def build_replay_argv(
    argv: list[str],
    *,
    run_dir: Path,
    gs_port: int,
    gs_token: str,
    render_lock: str,
    wrist_insets: str = "saved",
    saved_wrist_display: str = "raw224",
    policy_obs_dir: Path | None = None,
    source_summary: dict | None = None,
    view_overrides: dict[str, str] | None = None,
) -> list[str]:
    """Rewrite a recorded worker argv into a state-log replay render command."""

    recorded_objects = _last_option_value(argv, "--objects")
    if recorded_objects is not None:
        try:
            validate_task_objects(recorded_objects)
        except ValueError as exc:
            raise RerenderError(f"{run_dir}: {exc}") from exc
    argv = normalize_recorded_replay_options(argv, source_summary)
    argv = _with_moved_condition_assets(argv, run_dir)
    sim_log = run_dir / "logs" / "sim_log.jsonl"
    if not sim_log.is_file():
        raise RerenderError(f"{run_dir}: missing {sim_log}")
    rerender_dir = run_dir / "rerender"
    (rerender_dir / "logs").mkdir(parents=True, exist_ok=True)
    cadence_contract = _fr3_policy_action_playback_contract(argv, source_summary)

    view_overrides = dict(view_overrides or {})
    unknown_view_options = sorted(
        set(view_overrides) - set(_VIEW_OVERRIDE_OPTIONS)
    )
    if unknown_view_options:
        raise RerenderError(
            f"unknown view override option(s): {unknown_view_options}"
        )

    rewritten: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        option = token.partition("=")[0]
        if option in _DROP_WITH_VALUE:
            index += 1 if "=" in token else 2
            continue
        if option in _DROP_FLAGS:
            index += 1
            continue
        if option in _REWRITE_WITH_VALUE:
            index += 1 if "=" in token else 2
            continue
        # Only an explicitly requested override strips the recorded viewpoint;
        # untouched view options replay exactly as recorded.
        if option in view_overrides:
            index += 1 if "=" in token else 2
            continue
        if cadence_contract is not None and option in (
            "--frame-hz",
            "--dynamic-control-repeats",
        ):
            index += 1 if "=" in token else 2
            continue
        # The recorded no-video command commonly contains --no-wrist. Replay
        # owns the final saved/off choice and adds the appropriate option once.
        if token == "--no-wrist":
            index += 1
            continue
        if token in (
            "--policy-obs-inset",
            "--wrist-gs",
        ):
            index += 1
            continue
        rewritten.append(token)
        index += 1
    rewritten += [
        "--state-log-replay",
        str(sim_log),
        # State-log rendering must use a fresh overview EGL context after any
        # wrist contexts are closed. The inline path can lose robot visuals
        # and return its first overview readback upside down.
        "--defer-overview-render",
        "--gs-port",
        str(gs_port),
        "--gs-server-token",
        gs_token,
        "--render-lock",
        render_lock,
        "--max-wall-sec",
        "0",
        # A state-log row is already one final source frame. Generation-time
        # dynamic subframes have no control-loop ticks to resample in replay.
        "--dynamic-subframes",
        "1",
        # Override INIT_STATE_JSON inherited from a recorded worker. Logged
        # qpos/object states, not the generation initializer, drive replay.
        "--init-state-in",
        "",
        "--out",
        str(_canonical_video_path(run_dir)),
        "--logdir",
        str(rerender_dir / "logs"),
        "--summary-path",
        str(rerender_dir / "sim_summary.json"),
    ]
    if not any(token.partition("=")[0] == "--policy-yubi-render-profile" for token in argv):
        rewritten += ["--policy-yubi-render-profile", _recorded_wrist_profile(source_summary, sim_log)]
    if cadence_contract is not None:
        rewritten += [
            "--frame-hz",
            f"{cadence_contract['replay_frame_hz']:g}",
            # State replay performs no controller steps, but an explicit one
            # keeps the legacy dynamic timing resolver from deriving a
            # live-policy integer ratio from --frame-hz.
            "--dynamic-control-repeats",
            "1",
        ]
    if wrist_insets == "saved":
        if policy_obs_dir is None:
            raise RerenderError("saved wrist insets require a policy observation directory")
        if saved_wrist_display not in ("aspect-fit", "raw224"):
            raise RerenderError(
                "unknown saved wrist display mode: "
                f"{saved_wrist_display}"
            )
        rewritten += [
            "--state-replay-policy-obs-dir",
            str(policy_obs_dir),
            "--state-replay-policy-obs-display",
            saved_wrist_display,
        ]
    elif wrist_insets == "off":
        rewritten.append("--no-wrist")
    else:
        raise RerenderError(f"unknown resolved wrist inset mode: {wrist_insets}")
    for option in _VIEW_OVERRIDE_OPTIONS:
        if option in view_overrides:
            rewritten += [option, view_overrides[option]]
    if not any(token.partition("=")[0] == "--policy-wrist-backend" for token in argv):
        rewritten += ["--policy-wrist-backend", _recorded_wrist_backend(source_summary, sim_log)]
    elif _last_option_value(argv, "--policy-wrist-backend") not in WRIST_BACKENDS:
        raise RerenderError(f"{run_dir}: {UNSUPPORTED_RECORDED_WRIST_BACKEND}")
    if not any(token.partition("=")[0] == "--gs-kb4-near-cull" for token in argv):
        rewritten += ["--gs-kb4-near-cull", _recorded_gs_kb4_near_cull(source_summary, sim_log)]
    # Start the replay worker as a module, also for a run recorded with the
    # file-path head. The replay runs from the checkout root, so robosuite_env
    # resolves from that checkout (the target of its editable install) without
    # a sys.path insert.
    head_length = worker_command_head_length(rewritten)
    if head_length is None:
        raise RerenderError(f"{run_dir}: unexpected command head {rewritten[:3]}")
    return [rewritten[0], "-m", SIM_WORKER_MODULE, *rewritten[head_length:]]
