# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Runtime, production, and policy preflight checks."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

from real2sim._bootstrap import REPO_ROOT
from real2sim.artifacts import INPUTS_ROOT_ENV
from real2sim.config._types import ConfigError
from real2sim.config.paths import SUMMARY_TOOL
from real2sim.doctor import DoctorError, run_doctor
from real2sim.locked_imports import LockedImportError
from real2sim.provenance import ProvenanceError
from real2sim.runner.argv import (
    _extra_argument_occurrences,
    _extra_argument_value,
)
from real2sim.runner.environment import build_runner_environment
from real2sim.artifacts_binding import _validate_schema_v2_contracts

REQUIRED_RUCKIG_VERSION = "0.19.4"


def _validate_base_placement(robot: str, base_placement: str | None) -> None:
    """Load the calibration that will place this rig, before any GPU work.

    Inside the worker a wrong or malformed file puts the bases somewhere that
    still looks plausible from most camera angles, and the run completes with a
    number. Refusing here is the only place it is cheap.
    """

    from robosuite_env.robots.base_placement import (
        BasePlacementError,
        resolve_robot_base_placement,
    )

    if base_placement is not None and not Path(base_placement).is_file():
        raise ConfigError(
            f"simulation.parameters.base_placement is not a file: {base_placement}",
            path="simulation.parameters.base_placement",
            got=base_placement,
        )
    try:
        resolve_robot_base_placement(robot, base_placement)
    except (OSError, BasePlacementError) as exc:
        raise ConfigError(
            f"simulation.parameters.base_placement is not usable for {robot}: {exc}",
            path="simulation.parameters.base_placement",
            got=base_placement,
        ) from exc

def _require_task_contract(
    plan: Mapping[str, Any],
    expected: Mapping[str, str],
    context: str,
) -> None:
    """Refuse a profile whose prompt or success wiring drifted from the task."""

    task = plan["resolved"]["task"]
    mismatches = {
        key: {"expected": value, "actual": task.get(key)}
        for key, value in expected.items()
        if task.get(key) != value
    }
    if mismatches:
        raise ConfigError(
            f"{context} requires its exact instruction and pinned success "
            f"contract: {mismatches}"
        )


#: Phase suffixes the pi05_box_ct checkpoint's pa_contact lane carries
#: ("..., releasing it."). Those are meant to be switched per phase; a profile
#: sends one instruction for the whole episode, so it takes the plain form.
_PROMPT_PHASE_SUFFIX_RE = re.compile(
    r",\s*(approaching|gripping|holding|releasing)\s+it\.?$", re.IGNORECASE
)


def _prompt_instruction_key(text: str) -> str:
    """Normalise what varies between two records of one instruction.

    "with the right gripper" against "with right gripper", and a trailing full
    stop: the difference between a checkpoint's lane listing and the rows it
    tokenised. The key ignores both so one instruction can be recognised in
    either, and the exact wording is then taken from the decoded row.
    """

    normalised = str(text).strip()
    if normalised.endswith("."):
        normalised = normalised[:-1]
    for hand in ("right", "left"):
        normalised = normalised.replace(
            f"with the {hand} gripper", f"with {hand} gripper"
        )
    return normalised


def _checkpoint_prompt_instructions(
    checkpoint: Path,
) -> tuple[dict[str, str], dict[str, str]] | None:
    """Return ({key: lane text}, {key: decoded text}) from prompt.yaml.

    ``None`` when the checkpoint carries no prompt.yaml. Both prompt schemas
    are read: schema 1 lists lane strings directly, schema 2 nests them under
    ``text`` keys inside weighted lanes.
    """

    prompt_path = checkpoint / "prompt.yaml"
    if not prompt_path.is_file():
        return None
    import yaml

    try:
        document = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"checkpoint prompt.yaml is not readable YAML: {error}")

    listed: dict[str, str] = {}
    decoded: dict[str, str] = {}

    if not isinstance(document, Mapping):
        raise ConfigError(
            f"checkpoint prompt.yaml is not a mapping: {prompt_path}"
        )

    def note(store: dict[str, str], text: str) -> None:
        cleaned = str(text).strip()
        if cleaned:
            store.setdefault(_prompt_instruction_key(cleaned), cleaned)

    def read_lane_strings(node: Any) -> None:
        """``strings:`` is a list of plain strings (schema 1) or of {text:} rows."""

        if not isinstance(node, Sequence) or isinstance(node, (str, bytes)):
            return
        for row in node:
            if isinstance(row, str):
                note(listed, row)
            elif isinstance(row, Mapping) and isinstance(row.get("text"), str):
                note(listed, row["text"])

    def walk_lanes(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key == "strings":
                    read_lane_strings(value)
                    continue
                walk_lanes(value)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for item in node:
                walk_lanes(item)

    def walk_samples(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key == "decoded" and isinstance(value, str):
                    # One fed row: "Goal: ..., Task: <instruction>, State: ...".
                    match = re.search(r"Task:\s*(.+?)(?:,\s*State:|$)", value, re.S)
                    if match is not None:
                        note(decoded, match.group(1))
                    continue
                walk_samples(value)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for item in node:
                walk_samples(item)

    walk_lanes(document.get("lanes"))
    walk_samples(document.get("sample"))
    return listed, decoded


def _require_trained_instruction(
    plan: Mapping[str, Any],
    instruction: str,
    *,
    context: str,
) -> None:
    """Hold a profile's instruction to the wording its checkpoint was trained on.

    The task default is not right for every checkpoint: the box baselines were
    fed "with right gripper" -- the wording of the dataset's own tasks.jsonl --
    and the pi05_box_ct checkpoint "with the right gripper". A profile may
    therefore depart from the default, but only towards what its checkpoint
    carries. The rule:

    * the allowed set is the checkpoint's ``lanes`` listing, in either schema;
    * a checkpoint whose listing is empty is refused rather than waved through;
    * where a lane and a decoded row disagree over one instruction, the decoded
      row wins, because that is the string that was tokenised;
    * a decoded row missing is not evidence against an instruction, since those
      rows are a handful of samples rather than the whole set.
    """

    if _PROMPT_PHASE_SUFFIX_RE.search(instruction):
        # Refused whatever the checkpoint records: a profile sends one
        # instruction for a whole episode.
        raise ConfigError(
            f"{context} names a phase-bound instruction ({instruction!r}). "
            "Those belong to the pa_contact lane and are meant to change with "
            "the phase, while a profile sends one instruction for the whole "
            "episode. Use the plain form.",
            path="task.instruction",
        )
    policy = plan["resolved"].get("policy") or {}
    raw_checkpoint = policy.get("checkpoint")
    if not raw_checkpoint:
        # Nothing binds a checkpoint here (a layout-only check, say), so there
        # is no prompt record to hold the wording to.
        return
    checkpoint = Path(str(raw_checkpoint)).expanduser()
    prompts = _checkpoint_prompt_instructions(checkpoint)
    if prompts is None:
        return
    listed, decoded = prompts
    if not listed:
        raise ConfigError(
            f"{context} cannot be checked against its checkpoint: "
            f"{checkpoint / 'prompt.yaml'} lists no lane strings, so there is "
            "no set of trained instructions to hold it to.",
            path="task.instruction",
        )
    key = _prompt_instruction_key(instruction)
    if key not in listed:
        # The decoded rows correct the listing's wording; they do not add
        # instructions to it. Decoded rows may differ in wording from the
        # listing.
        raise ConfigError(
            f"{context} names an instruction its checkpoint does not list: "
            f"{instruction!r} is absent from the lanes of "
            f"{checkpoint / 'prompt.yaml'}.",
            path="task.instruction",
        )
    exact = decoded.get(key)
    if exact is None:
        # The lane listing is formatted: it writes "with the right gripper"
        # where the dataset's own tasks.jsonl and the rows fed from it write
        # "with right gripper". With no decoded row for this instruction there
        # is nothing to correct the listing against, and a missing row is not
        # evidence against the profile, so its wording stands.
        return
    if instruction != exact:
        raise ConfigError(
            f"{context} words its instruction differently from the decoded "
            f"training row of its checkpoint: the profile says {instruction!r} "
            f"and the checkpoint was fed {exact!r}. Use the checkpoint's "
            "wording.",
            path="task.instruction",
        )


def _require_instruction_names_the_task(
    instruction: str,
    task_index: int,
    *,
    context: str,
) -> None:
    """Keep a wording bound to the object and hand its task index moves.

    A profile may word the task as its checkpoint does, but not swap the task:
    box_pack_task_index picks the object, the hand and the direction that the
    environment and the success criterion use, and the instruction is what the
    policy is told.
    """

    from robosuite_env.tasks.box_pack import (
        BOX_PACK_INSTRUCTIONS,
        BOX_PACK_OBJECT_LABELS,
    )

    entry = next(
        (row for row in BOX_PACK_INSTRUCTIONS if row["task_index"] == task_index),
        None,
    )
    if entry is None:
        raise ConfigError(
            f"{context}: no box-pack task carries index {task_index}",
            path="simulation.parameters.box_pack_task_index",
        )
    label = BOX_PACK_OBJECT_LABELS[entry["object"]]
    hand = "right" if entry["hand"] == "R" else "left"
    required = {
        f"the {label}": f"the object of task {task_index} ({label})",
        f"{hand} gripper": f"the {hand} hand this task uses",
    }
    if entry["direction"] == "in":
        required["in the box"] = "putting the object into the box"
    else:
        required["from the box"] = "taking the object out of the box"
    missing = [why for phrase, why in required.items() if phrase not in instruction]
    if missing:
        raise ConfigError(
            f"{context} names an instruction that does not describe its task: "
            f"{instruction!r} never mentions {', '.join(missing)}.",
            path="task.instruction",
        )


def _validate_selected_layout(
    plan: Mapping[str, Any],
    task: str,
    *,
    expected_objects: Sequence[str],
    context: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the hand-authored layout this run resolves, on this run's scenes.

    The worker, the environment, the init-state identity and the run summary
    resolve the layout through the same ``select_task_layout``. A launcher run
    never carries the harness-only ``RS_*_LAYOUT_JSON`` overrides (the
    reproducibility check refuses them), so none is consulted here.
    """

    from robosuite_env.tasks.layout_selection import (
        LAYOUT_PARAMETERS,
        LayoutError,
        load_selected_layout,
        layout_sweep_offset_mm,
        require_layout_rig,
        require_layout_scene,
        select_task_layout,
    )

    parameter = LAYOUT_PARAMETERS[task]
    field = f"simulation.parameters.{parameter}"
    parameters = plan["resolved"]["simulation"].get("parameters") or {}
    configured = parameters.get(parameter)
    configured_values = configured if isinstance(configured, list) else [configured]
    result: tuple[dict[str, Any], dict[str, Any]] | None = None
    for index, configured_value in enumerate(configured_values):
        entry_field = f"{field}[{index}]" if isinstance(configured, list) else field
        try:
            selection = select_task_layout(
                task,
                configured_value,
                robot=plan["resolved"]["robot"],
                environ={},
            )
        except LayoutError as exc:
            raise ConfigError(
                f"{context} layout selection failed: {exc}", path=entry_field
            ) from exc
        if not selection.path.is_file():
            raise ConfigError(
                f"{entry_field} does not exist: {selection.path}"
                if selection.selected_by == "profile"
                else f"{context} default layout does not exist: {selection.path}",
                path=entry_field,
                got=str(selection.path),
            )
        try:
            layout, payload, _record = load_selected_layout(
                selection, expected_objects=expected_objects
            )
        except LayoutError as exc:
            raise ConfigError(
                f"{context} layout failed: {exc}", path=entry_field
            ) from exc
        if isinstance(configured, list):
            for offset_key in ("pen_offset_mm", "box_offset_mm"):
                _offset, reason = layout_sweep_offset_mm(payload, offset_key)
                if reason is not None:
                    raise ConfigError(
                        f"{context} layout failed: {reason}",
                        path=entry_field,
                        got=str(selection.path),
                    )
        for scene in plan["resolved"]["scenes"]:
            try:
                require_layout_scene(
                    layout, running_scene=scene["tag"], selection=selection
                )
            except LayoutError as exc:
                raise ConfigError(
                    f"{context} layout is not for scene {scene['tag']!r}: {exc}",
                    path=entry_field,
                    got=str(selection.path),
                ) from exc
        robot = plan["resolved"]["robot"]
        try:
            require_layout_rig(payload, robot=robot, selection=selection)
        except LayoutError as exc:
            raise ConfigError(
                f"{context} layout is not for robot {robot!r}: {exc}",
                path=entry_field,
                got=str(selection.path),
            ) from exc
        result = (layout, payload)
    assert result is not None
    return result


def _validate_task_objects(plan: Mapping[str, Any]) -> None:
    raw_objects = str(plan["resolved"]["task"]["objects"])
    from robosuite_env.tasks.layout_selection import LAYOUT_PARAMETERS

    simulation_parameters = plan["resolved"]["simulation"].get("parameters") or {}
    for layout_task, layout_parameter in LAYOUT_PARAMETERS.items():
        if (
            simulation_parameters.get(layout_parameter) is not None
            and raw_objects != layout_task
        ):
            raise ConfigError(
                f"simulation.parameters.{layout_parameter} selects the "
                f"{layout_task} layout, but task.objects is {raw_objects!r}",
                path=f"simulation.parameters.{layout_parameter}",
            )
    if raw_objects == "box_pack":
        from robosuite_env.tasks.box_pack import (
            BOX_PACK_GOAL_CONDITION,
            BOX_PACK_OBJECT_NAMES,
            BOX_PACK_SCORE_CONDITION,
        )

        parameters = plan["resolved"]["simulation"]["parameters"]
        raw_index = parameters.get("box_pack_task_index")
        if raw_index is None:
            raise ConfigError(
                "task.objects box_pack requires "
                "simulation.parameters.box_pack_task_index",
                path="simulation.parameters.box_pack_task_index",
            )
        task_index = int(raw_index)
        context = f"task.objects box_pack at box_pack_task_index {task_index}"
        _require_task_contract(
            plan,
            {
                "goal_condition": BOX_PACK_GOAL_CONDITION,
                "score_condition": BOX_PACK_SCORE_CONDITION,
            },
            context,
        )
        # The instruction may depart from the task default, because the
        # checkpoints under test were trained on different wordings of the same
        # task, but only towards what its own checkpoint carries. It must still
        # name this index's object and hand.
        instruction = str(plan["resolved"]["task"]["instruction"])
        _require_instruction_names_the_task(instruction, task_index, context=context)
        _require_trained_instruction(plan, instruction, context=context)
        # A checkpoint the task table registers gets the exact check the worker
        # makes once the policy server tells it which checkpoint it serves
        # (rollout/stages/init_policy_render.py). Without this, a wording its
        # prompt.yaml cannot correct -- the pi05_box_ct checkpoint has no decoded
        # tape row, and the lane key ignores the article -- passes here and then
        # stops the run on the node, after the policy server is up.
        from robosuite_env.tasks.box_pack import (
            box_pack_checkpoint_is_registered,
            box_pack_instruction_for_checkpoint,
        )

        served = (plan["resolved"].get("policy") or {}).get("checkpoint")
        if box_pack_checkpoint_is_registered(served):
            expected = box_pack_instruction_for_checkpoint(str(served), task_index)
            if instruction != expected:
                raise ConfigError(
                    f"{context} gives its registered checkpoint "
                    f"{str(served).rstrip('/').rsplit('/', 1)[-1]!r} the wording "
                    f"{instruction!r}, but the worker will require {expected!r} "
                    "for this checkpoint and stop the run once the policy server "
                    "is up. Use the registered wording.",
                    path="task.instruction",
                )
        # The layout this run resolves -- the profile's box_pack_layout_json or
        # the Duo default -- placed for this run's scene.
        _validate_selected_layout(
            plan,
            "box_pack",
            expected_objects=BOX_PACK_OBJECT_NAMES,
            context="task.objects box_pack",
        )
        return
    if raw_objects in {"cup_plate"}:
        return
    if not raw_objects.endswith(".json"):
        raise ConfigError(
            "task.objects must be cup_plate, box_pack "
            "or a positions JSON"
        )

    object_path = Path(raw_objects)
    if not object_path.is_absolute():
        object_path = Path(plan["repo_root"]) / object_path
    object_path = object_path.resolve()
    if not object_path.is_file():
        raise ConfigError(f"task.objects positions JSON not found: {object_path}")
    try:
        positions = json.loads(object_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"task.objects positions JSON is unreadable: {object_path}"
        ) from exc
    if not isinstance(positions, dict):
        raise ConfigError(
            f"task.objects positions JSON must contain an object: {object_path}"
        )
    for field in ("cup_xy", "plate_xy"):
        pair = positions.get(field)
        valid_pair = isinstance(pair, list) and len(pair) == 2
        if valid_pair:
            try:
                valid_pair = all(
                    not isinstance(item, bool)
                    and isinstance(item, (int, float))
                    and math.isfinite(float(item))
                    for item in pair
                )
            except OverflowError:
                valid_pair = False
        if not valid_pair:
            raise ConfigError(
                f"task.objects {field} must be a two-element array of finite numbers: "
                f"{object_path}"
            )

#: A placeholder test artifact: ``access: fixture`` with a uri under this prefix
#: (artifacts/manifest.yaml, e.g. tiny-checkpoint-v1).
_PLACEHOLDER_FIXTURE_URI_PREFIX = "fixture://tests/fixtures/artifacts/"


def _placeholder_fixture_checkpoint_artifact(resolved: Mapping[str, Any]) -> str | None:
    """Return the checkpoint artifact id if it is a placeholder test fixture.

    Keyed on the bound artifact, never on ``reproducibility.mode``: the r10
    profiles run in fixture mode yet bind real generated checkpoints. An unbound
    checkpoint, an unreadable manifest or any non-placeholder entry is not one.
    """

    from real2sim.artifacts import ManifestError, load_manifest

    reproducibility = resolved.get("reproducibility")
    if not isinstance(reproducibility, Mapping):
        return None
    artifacts = reproducibility.get("artifacts")
    artifact_id = artifacts.get("checkpoint") if isinstance(artifacts, Mapping) else None
    manifest_path = reproducibility.get("artifact_manifest")
    if not isinstance(artifact_id, str) or not isinstance(manifest_path, str):
        return None
    try:
        entry = load_manifest(manifest_path)["artifacts"].get(artifact_id)
    except ManifestError:
        return None
    if (
        not isinstance(entry, Mapping)
        or entry.get("access") != "fixture"
        or not str(entry.get("uri", "")).startswith(_PLACEHOLDER_FIXTURE_URI_PREFIX)
    ):
        return None
    return artifact_id


def _placeholder_without_norm_stats(
    resolved: Mapping[str, Any], exc: Exception, what: str
) -> str:
    """Allow a checkpoint with no norm stats only if it is a placeholder fixture."""

    placeholder = _placeholder_fixture_checkpoint_artifact(resolved)
    if placeholder is None:
        raise ConfigError(
            f"simulation.parameters.policy_state_quat_hemisphere: {what} ({exc}); "
            "a checkpoint without readable norm stats is refused at check-runtime "
            "and launch, whatever reproducibility.mode says, unless "
            "reproducibility.artifacts.checkpoint is a placeholder fixture "
            f"artifact (access: fixture, {_PLACEHOLDER_FIXTURE_URI_PREFIX}...)"
        ) from exc
    return placeholder


def _openpi_norm_stats_for_plan(
    plan: Mapping[str, Any],
) -> tuple[Path, str]:
    """Resolve the checkpoint stats selected by the plan's OpenPI config."""

    from robosuite_env.policy import policy_batch_infer as pbi

    resolved = plan["resolved"]
    policy = resolved["policy"]
    config_name = policy.get("config_name")
    config_source = policy.get("config_source")
    if not isinstance(config_name, str) or not isinstance(config_source, str):
        raise ValueError(
            "resolved policy must carry string config_name and config_source"
        )
    _config, path, asset_id = pbi.resolve_openpi_norm_stats(
        policy["checkpoint"],
        config=config_name,
        config_src=config_source,
        action_horizon=int(policy.get("action_horizon") or 0),
        repo_root=plan.get("repo_root", REPO_ROOT),
    )
    return path, asset_id


def _policy_state_quat_hemisphere_for_plan(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve a plan's quaternion sign from its config-selected stats."""

    from robosuite_env.policy import policy_io as pio

    path, asset_id = _openpi_norm_stats_for_plan(plan)
    return {
        "norm_stats_asset_id": asset_id,
        **pio.policy_state_quat_hemisphere_from_norm_stats(path),
    }


def validate_policy_state_quat_hemisphere(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Re-resolve the state quaternion hemisphere against the checkpoint to run.

    Plan resolution recorded the hemisphere and, for ``auto``, the norm stats
    it came from. The same resolver runs again here, so a checkpoint whose norm
    stats changed since resolution, a worker argument that does not carry the
    recorded value, or a profile value the checkpoint contradicts is refused
    before any worker starts. A checkpoint with no norm stats at all is refused
    as well unless its artifact is a placeholder test fixture, which is reported
    unverified; norm stats that exist but are unreadable are always refused.
    """

    from real2sim.config.parameters import (
        POLICY_STATE_QUAT_HEMISPHERE_CHOICES,
        POLICY_STATE_QUAT_HEMISPHERE_FLAG,
    )
    from robosuite_env.policy import policy_io as pio

    resolved = plan["resolved"]
    simulation = resolved["simulation"]
    configured = (simulation.get("parameters") or {}).get(
        "policy_state_quat_hemisphere"
    )
    record = simulation.get("policy_state_quat_hemisphere")
    if (
        configured == "auto"
        and isinstance(record, Mapping)
        and record.get("source") == "unresolved"
    ):
        # Plan resolution found no norm stats. Only a placeholder fixture
        # checkpoint artifact may continue; a real checkpoint fails closed in
        # every reproducibility mode, and stats that appeared after resolution
        # are refused because the sealed worker argv was rendered without them.
        if _extra_argument_occurrences(
            simulation["cli_args"], POLICY_STATE_QUAT_HEMISPHERE_FLAG
        ):
            raise ConfigError(
                f"an unresolved plan must not carry {POLICY_STATE_QUAT_HEMISPHERE_FLAG}"
            )
        try:
            current = _policy_state_quat_hemisphere_for_plan(plan)
        except pio.PolicyStateQuatHemisphereAbsent as exc:
            placeholder = _placeholder_without_norm_stats(
                resolved, exc, "auto is unresolved"
            )
            return {
                "hemisphere": None,
                "verified_against_norm_stats": False,
                "reason": f"placeholder fixture checkpoint artifact {placeholder}: {exc}",
            }
        except pio.PolicyStateQuatHemisphereError as exc:
            raise ConfigError(
                "simulation.parameters.policy_state_quat_hemisphere: auto cannot "
                f"read the checkpoint norm stats: {exc}"
            ) from exc
        except LockedImportError as exc:
            raise ConfigError(
                "simulation.parameters.policy_state_quat_hemisphere: auto cannot "
                f"resolve the OpenPI config's norm stats: locked OpenPI check failed: {exc}"
            ) from exc
        except (ImportError, ValueError) as exc:
            raise ConfigError(
                "simulation.parameters.policy_state_quat_hemisphere: auto cannot "
                f"resolve the OpenPI config's norm stats: {exc}"
            ) from exc
        raise ConfigError(
            f"checkpoint norm stats {current['norm_stats_path']} appeared after the "
            "plan was resolved without them; re-resolve the profile so the worker "
            f"argument carries hemisphere {current['hemisphere']!r}"
        )
    if (
        configured not in POLICY_STATE_QUAT_HEMISPHERE_CHOICES
        or not isinstance(record, Mapping)
        or record.get("configured") != configured
        or record.get("hemisphere") not in pio.POLICY_STATE_QUAT_HEMISPHERES
    ):
        raise ConfigError(
            "simulation.policy_state_quat_hemisphere: the resolved plan carries no "
            f"valid hemisphere resolution (configured={configured!r}, "
            f"record={record!r})"
        )
    hemisphere = record["hemisphere"]
    cli_args = simulation["cli_args"]
    rendered = _extra_argument_value(cli_args, POLICY_STATE_QUAT_HEMISPHERE_FLAG)
    expected = None if hemisphere == "positive" else hemisphere
    if (
        _extra_argument_occurrences(cli_args, POLICY_STATE_QUAT_HEMISPHERE_FLAG) > 1
        or rendered != expected
    ):
        raise ConfigError(
            f"worker argument {POLICY_STATE_QUAT_HEMISPHERE_FLAG} is {rendered!r} "
            f"but the resolved hemisphere is {hemisphere!r} (positive renders no "
            "argument)"
        )
    try:
        current = _policy_state_quat_hemisphere_for_plan(plan)
    except pio.PolicyStateQuatHemisphereAbsent as exc:
        if configured == "auto":
            raise ConfigError(
                "simulation.parameters.policy_state_quat_hemisphere: auto cannot "
                f"re-read the checkpoint norm stats it was resolved from: {exc}"
            ) from exc
        # An explicit value can only be checked against norm stats; without
        # any, only a placeholder fixture checkpoint may run unverified.
        placeholder = _placeholder_without_norm_stats(
            resolved, exc, f"explicit {hemisphere} cannot be checked"
        )
        return {
            "hemisphere": hemisphere,
            "verified_against_norm_stats": False,
            "reason": f"placeholder fixture checkpoint artifact {placeholder}: {exc}",
        }
    except pio.PolicyStateQuatHemisphereError as exc:
        raise ConfigError(
            "simulation.parameters.policy_state_quat_hemisphere: cannot re-read "
            f"the checkpoint norm stats: {exc}"
        ) from exc
    except LockedImportError as exc:
        raise ConfigError(
            "simulation.parameters.policy_state_quat_hemisphere: cannot resolve "
            f"the OpenPI config's norm stats: locked OpenPI check failed: {exc}"
        ) from exc
    except (ImportError, ValueError) as exc:
        raise ConfigError(
            "simulation.parameters.policy_state_quat_hemisphere: cannot resolve "
            f"the OpenPI config's norm stats: {exc}"
        ) from exc
    if current["hemisphere"] != hemisphere:
        raise ConfigError(
            f"policy state quaternion hemisphere {hemisphere!r} "
            f"({record.get('source')}) disagrees with checkpoint norm stats "
            f"{current['norm_stats_path']} (sha256 {current['norm_stats_sha256']}, "
            f"qw q01 {current['qw_q01']}, q99 {current['qw_q99']}), which resolve "
            f"to {current['hemisphere']!r}"
        )
    if configured == "auto" and current["norm_stats_sha256"] != record.get(
        "norm_stats_sha256"
    ):
        raise ConfigError(
            f"checkpoint norm stats {current['norm_stats_path']} changed since the "
            f"plan was resolved (sha256 {record.get('norm_stats_sha256')} -> "
            f"{current['norm_stats_sha256']})"
        )
    return {"hemisphere": hemisphere, "verified_against_norm_stats": True, **current}

def validate_runtime_paths(plan: Mapping[str, Any]) -> dict[str, Any] | None:
    resolved = plan["resolved"]
    reproducibility = resolved.get("reproducibility")
    if reproducibility is not None and reproducibility.get("inputs_root") is not None:
        inputs_root = Path(reproducibility["inputs_root"])
        if inputs_root.is_symlink() or not inputs_root.is_dir():
            raise ConfigError(
                f"{INPUTS_ROOT_ENV} is not an existing directory "
                f"(symlinks are rejected): {inputs_root}"
            )
    run_root = Path(plan["run_root"])
    if run_root.exists() or run_root.is_symlink():
        raise ConfigError(f"output.run_name already exists under output.root: {run_root}")
    for field, executable in (
        ("runner", Path(plan["runner_script"])),
        ("summary tool", SUMMARY_TOOL),
    ):
        if not executable.is_file():
            raise ConfigError(f"{field} is not a file: {executable}")
    python_path = Path(resolved["policy"]["python"])
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise ConfigError(f"policy.python is not executable: {python_path}")
    checkpoint = Path(resolved["policy"]["checkpoint"])
    if not checkpoint.is_dir():
        raise ConfigError(f"policy.checkpoint is not a directory: {checkpoint}")
    validate_policy_state_quat_hemisphere(plan)
    _validate_task_objects(plan)
    contract_report = _validate_schema_v2_contracts(plan)
    cli_args = resolved["simulation"]["cli_args"]
    desk_value = _extra_argument_value(cli_args, "--desk")
    desk_world_aligned = "--desk-world-aligned" in cli_args
    if desk_world_aligned and desk_value is None:
        raise ConfigError("--desk-world-aligned requires --desk")
    if desk_value is not None and desk_value not in {"robosuite", "scene"}:
        desk_mesh = Path(desk_value).expanduser()
        if not desk_mesh.is_absolute():
            desk_mesh = Path(plan["repo_root"]) / desk_mesh
        desk_mesh = Path(os.path.abspath(desk_mesh))
        if not desk_mesh.is_file():
            raise ConfigError(
                f"configured simulation --desk is not a file: {desk_mesh}"
            )
        if desk_world_aligned:
            collision_dir = desk_mesh.parent / "coacd"
            collision_meshes = (
                list(collision_dir.glob("*.obj"))
                if collision_dir.is_dir()
                else []
            )
            if not any(mesh.is_file() for mesh in collision_meshes):
                raise ConfigError(
                    "registered desk collision meshes not found: "
                    f"{collision_dir}/*.obj"
                )
    simulation_parameters = resolved["simulation"].get("parameters") or {}
    initial_arm_joints_json = simulation_parameters.get("initial_arm_joints_json")
    base_placement = simulation_parameters.get("base_placement")
    for field, raw in (
        ("simulation.init_state", resolved["simulation"]["init_state"]),
        (
            "simulation.gripper_calibration",
            resolved["simulation"]["gripper_calibration"],
        ),
        ("task.policy_image_episode", resolved["task"]["policy_image_episode"]),
        (
            "simulation.parameters.initial_arm_joints_json",
            initial_arm_joints_json,
        ),
    ):
        if raw is not None and not Path(raw).is_file():
            raise ConfigError(f"{field} is not a file: {raw}")
    if initial_arm_joints_json is not None:
        # Parse the selected pose here so an invalid schema, an out-of-range
        # gripper angle, or a mis-declared gripper coordinate stops the launch
        # before any GPU work instead of inside the first worker.
        #
        from robosuite_env.robots.fr3.initial_pose import (
            load_initial_arm_joint_pose,
        )

        try:
            load_initial_arm_joint_pose(initial_arm_joints_json)
        except (OSError, ValueError) as exc:
            raise ConfigError(
                "simulation.parameters.initial_arm_joints_json is not a valid "
                f"initial arm joint pose: {exc}",
                path="simulation.parameters.initial_arm_joints_json",
                got=initial_arm_joints_json,
            ) from exc
    _validate_base_placement(resolved["robot"], base_placement)
    desk_texture = resolved["render"]["desk_texture"]
    if desk_texture is not None:
        desk = Path(desk_texture)
        if not desk.is_file():
            raise ConfigError(f"render.desk_texture is not a file: {desk}")
    gaussian_repo = Path(resolved["runtime"]["gaussian_wrapping_repo"])
    if not gaussian_repo.is_dir():
        raise ConfigError(f"runtime.gaussian_wrapping_repo is not a directory: {gaussian_repo}")
    return contract_report

def validate_production_preflight(
    plan: Mapping[str, Any],
    *,
    capture_artifact_verification: bool = False,
) -> dict[str, Any] | None:
    """Apply the support matrix and doctor gates before a production run."""

    reproducibility = plan["resolved"].get("reproducibility")
    if reproducibility is None or not reproducibility["production"]:
        return None
    artifact_ids = list(dict.fromkeys(reproducibility["artifacts"].values()))
    environment = build_runner_environment(plan)
    policy_python_ok, policy_python_detail = _check_policy_python_runtime(
        plan,
        environment,
    )
    if not policy_python_ok:
        raise ConfigError(
            "production preflight failed: "
            f"policy_python: {policy_python_detail}"
        )
    try:
        report = run_doctor(
            repo_root=Path(plan["repo_root"]),
            manifest_path=Path(reproducibility["artifact_manifest"]),
            artifact_ids=artifact_ids,
            support_matrix_path=Path(reproducibility["support_matrix"]),
            profile_name=reproducibility["accelerator_profile"]["name"],
            output_path=Path(plan["run_root"]),
            min_free_gib=0.01,
            production=True,
            lockfiles=reproducibility["lockfiles"],
            environment=environment,
            capture_artifact_verification=capture_artifact_verification,
        )
    except (DoctorError, ProvenanceError) as exc:
        raise ConfigError(f"production preflight failed: {exc}") from exc
    report["checks"].append(
        {
            "name": "policy_python",
            "status": "ok",
            "detail": policy_python_detail,
        }
    )
    failures = [
        f"{item['name']}: {item['detail']}"
        for item in report["checks"]
        if item["status"] != "ok"
    ]
    if not isinstance(report.get("gpu_bindings"), Mapping):
        failures.append("gpu_binding: production doctor returned no sealed GPU binding")
    if failures:
        raise ConfigError("production preflight failed: " + "; ".join(failures))
    return report

def _check_policy_python_runtime(
    plan: Mapping[str, Any],
    environment: Mapping[str, str],
) -> tuple[bool, dict[str, Any]]:
    """Verify that production workers use this exact locked Python environment."""

    resolved = plan["resolved"]
    python_path = Path(resolved["policy"]["python"])
    expected_version = str(
        resolved["reproducibility"]["accelerator_profile"]["python"]
    )
    probe = (
        "import json,platform,sys;"
        "print(json.dumps({"
        "'version':platform.python_version(),"
        "'executable':sys.executable,"
        "'prefix':sys.prefix,"
        "'base_prefix':sys.base_prefix"
        "},sort_keys=True))"
    )
    detail: dict[str, Any] = {
        "path": str(python_path),
        "expected_version": expected_version,
        "expected_prefix": str(python_path.parent.parent.resolve()),
        "launcher_prefix": str(Path(sys.prefix).resolve()),
    }
    try:
        completed = subprocess.run(
            [str(python_path), "-I", "-c", probe],
            cwd=REPO_ROOT,
            env=dict(environment),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        detail["error"] = f"{type(exc).__name__}: {exc}"
        return False, detail
    if completed.returncode != 0:
        detail["returncode"] = completed.returncode
        detail["error"] = (completed.stderr or completed.stdout).strip()
        return False, detail
    try:
        observed = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        detail["error"] = f"invalid Python identity output: {exc}"
        return False, detail
    required = {"version", "executable", "prefix", "base_prefix"}
    if not isinstance(observed, dict) or set(observed) != required or not all(
        isinstance(observed[key], str) and observed[key] for key in required
    ):
        detail["error"] = "Python identity output has an invalid schema"
        return False, detail
    detail["observed"] = observed
    observed_prefix = Path(observed["prefix"]).resolve()
    expected_prefix = python_path.parent.parent.resolve()
    launcher_prefix = Path(sys.prefix).resolve()
    base_prefix = Path(observed["base_prefix"]).resolve()
    checks = {
        "version_matches": observed["version"] == expected_version,
        "is_virtual_environment": observed_prefix != base_prefix,
        "path_matches_prefix": observed_prefix == expected_prefix,
        "matches_launcher_environment": observed_prefix == launcher_prefix,
    }
    detail["checks"] = checks
    return all(checks.values()), detail

def _robot_embodiment(robot: str) -> str:
    """Return the mechanism family a resolved rig is built from."""

    from robosuite_env.robots.profiles import robot_embodiment

    try:
        return robot_embodiment(robot)
    except (KeyError, ValueError) as exc:
        raise ConfigError(
            f"robot {robot!r} is not a supported simulation rig: {exc}",
            path="robot",
            got=robot,
        ) from exc


def _ruckig_runtime_requirement(resolved: Mapping[str, Any]) -> str | None:
    """Name the Ruckig trajectory generator this run executes, if any.

    The rule follows the runtime paths rather than a rig name, so a rig built
    on an existing mechanism inherits its check:

    * every FR3-embodiment rig (``fr3`` and ``fr3_duo``) builds
      ``StatefulJerkLimitedTrajectory`` for joint_ik under dynamic physics
      (``robosuite_env/rollout/stages/init_policy_render.py``); it imports
      ruckig and checks ``ruckig.__version__``.

    An embodiment without a rule is refused rather than silently skipped.
    """

    embodiment = _robot_embodiment(resolved["robot"])
    if embodiment == "fr3":
        simulation = resolved["simulation"]
        parameters = simulation["parameters"]
        physics = (
            parameters["physics"]
            if parameters is not None
            else _extra_argument_value(simulation["cli_args"], "--physics")
            or "dynamic"
        )
        if simulation["fr3_controller"] == "joint_ik" and physics == "dynamic":
            return (
                "the FR3 joint_ik jerk-limited trajectory "
                "(robosuite_env/rollout/control/joint_trajectory.py)"
            )
        return None
    raise ConfigError(
        f"no Ruckig runtime rule is defined for robot embodiment {embodiment!r}",
        path="robot",
        got=resolved["robot"],
    )


_RUCKIG_PROBE = """\
import sys

expected = sys.argv[1]
import ruckig

failures = []
actual = getattr(ruckig, "__version__", None)
if actual != expected:
    failures.append(f"expected ruckig=={expected}, got {actual!r}")
sys.exit("; ".join(failures) if failures else 0)
"""


def validate_joint_trajectory_runtime(plan: Mapping[str, Any]) -> None:
    """Fail before output or servers if a Ruckig path lacks the pinned Ruckig."""

    resolved = plan["resolved"]
    requirement = _ruckig_runtime_requirement(resolved)
    if requirement is None:
        return
    python_path = str(resolved["policy"]["python"])
    try:
        completed = subprocess.run(
            [
                python_path,
                "-c",
                _RUCKIG_PROBE,
                REQUIRED_RUCKIG_VERSION,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConfigError(
            f"Ruckig runtime preflight failed for {requirement}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise ConfigError(
            "policy.python must provide "
            f"ruckig=={REQUIRED_RUCKIG_VERSION} for {requirement}{suffix}"
        )


def validate_locked_imports(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Run the full locked-import check once per launch.

    This is the one place a launch runs ``require_locked_imports`` in
    ``FULL`` mode: every installed file against its ``RECORD`` hash, the
    symlink and unlisted-module walk, and the cheap checks. A production
    launch checks the packages the run manifest records
    (``real2sim.provenance.LOCKED_RUN_IMPORTS``: OpenPI, its client and
    robosuite); the report carries the records, and ``prepare_run_manifest``
    records them rather than hashing again. A fixture launch checks robosuite,
    which every rollout builds its model with, and its manifest records the
    packages unverified. Rollout workers and policy loaders run only the
    ``CHEAP`` mode. Production workers share this process's environment
    (``_check_policy_python_runtime``), so this answers for their files too.
    Returns the verified records.
    """

    from real2sim.locked_imports import FULL, ROBOSUITE_PACKAGES, require_locked_imports
    from real2sim.provenance import LOCKED_RUN_IMPORTS

    reproducibility = (plan.get("resolved") or {}).get("reproducibility")
    production = reproducibility is not None and bool(reproducibility["production"])
    packages = LOCKED_RUN_IMPORTS if production else ROBOSUITE_PACKAGES
    try:
        return require_locked_imports(
            plan.get("repo_root", REPO_ROOT), packages, mode=FULL
        )
    except LockedImportError as exc:
        raise ConfigError(f"locked import check failed: {exc}") from exc


def validate_launch_preflight(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Run the one preflight shared by check-runtime and execution."""

    # Keep the production doctor first so a standalone manifest preparation
    # reports the established "production preflight failed" diagnostic before
    # host-specific path errors. check-runtime and execute call this same order.
    doctor = validate_production_preflight(
        plan,
        capture_artifact_verification=True,
    )
    contracts = validate_runtime_paths(plan)
    locked_imports = validate_locked_imports(plan)
    validate_joint_trajectory_runtime(plan)
    report = {
        "contracts": contracts,
        "doctor": doctor,
        "_locked_imports": locked_imports,
    }
    if doctor is not None and isinstance(doctor.get("gpu_bindings"), Mapping):
        report["gpu_bindings"] = doctor["gpu_bindings"]
    if doctor is not None and "_artifact_verification" in doctor:
        doctor_report = dict(doctor)
        report["_artifact_verification"] = doctor_report.pop(
            "_artifact_verification"
        )
        report["doctor"] = doctor_report
    return report
