# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Which hand-authored layout a box run places its objects from.

Box packing places its objects from a
hand-authored layout (:mod:`robosuite_env.tasks.layouts`). A layout is placed
for one scene, and the rigs do not share one: Franka Duo stands on
``franka_duo``, Franka Flat on ``franka_flat``. A profile therefore
selects its layout, and every consumer -- the launcher preflight, the rollout
worker, the environment that loads the file, the init-state identity and the
run summary -- resolves that selection through :func:`select_task_layout`, in
one order that is the same for every task:

1. the profile parameter (``simulation.parameters.box_pack_layout_json``). Box packing also accepts a
   list with exactly one entry per rollout; ``run_NN`` selects entry
   ``(NN - 1) % len(list)``. The launcher renders the selected entry as the
   worker flag of the same name;
2. the harness-only environment override (``RS_BOX_PACK_LAYOUT_JSON``). The launcher refuses
   these in every profile, so only tests and hand-run diagnostic workers reach
   this step. When the parameter and the override are both set they must name
   the same file: the worker exports the profile's file under the override
   name so the environment can read it,
   and two different files are refused rather than ranked;
3. the task's default: the Duo layout for box.

This module imports no robosuite, so the launcher can use it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .layouts import LayoutError, validate_manual_layout

BOX_PACK_TASK = "box_pack"

#: ``task.objects`` -> the profile parameter (and worker flag) selecting its layout.
LAYOUT_PARAMETERS = {
    BOX_PACK_TASK: "box_pack_layout_json",
}
#: ``task.objects`` -> the harness-only environment override.
LAYOUT_OVERRIDE_ENV = {
    BOX_PACK_TASK: "RS_BOX_PACK_LAYOUT_JSON",
}

SELECTED_BY_PROFILE = "profile"
SELECTED_BY_OVERRIDE = "harness_override"
SELECTED_BY_DEFAULT = "default"

class LayoutSelectionError(LayoutError, RuntimeError):
    """A layout cannot be selected, or does not belong to the running scene.

    It is a :class:`LayoutError` like every layout failure, and a
    ``RuntimeError`` because the layout lookup it replaces raised one.
    """


@dataclass(frozen=True)
class LayoutSelection:
    """The layout file one run resolves, and which step of the order chose it."""

    task: str
    path: Path
    selected_by: str
    layout_index: int = 0

    @property
    def parameter(self) -> str:
        return LAYOUT_PARAMETERS[self.task]

    @property
    def override_env(self) -> str:
        return LAYOUT_OVERRIDE_ENV[self.task]


def _normalized(raw: str | os.PathLike[str]) -> Path:
    # Like the launcher's _resolved_path: absolute, ``..`` collapsed, symlinks
    # kept, so the path a record names is the one the profile wrote.
    return Path(os.path.abspath(os.path.expanduser(os.fspath(raw))))


def resolve_layout_path(
    task: str,
    *,
    configured: str | os.PathLike[str] | None,
    default: str | os.PathLike[str] | None,
    environ: Mapping[str, str] | None = None,
    missing_default: str | None = None,
) -> LayoutSelection:
    """Apply the selection order (see the module docstring) for one task."""

    if task not in LAYOUT_PARAMETERS:
        raise LayoutSelectionError(f"no hand-authored layout is selected for task {task!r}")
    parameter = LAYOUT_PARAMETERS[task]
    override_env = LAYOUT_OVERRIDE_ENV[task]
    environment = os.environ if environ is None else environ
    profile = str(os.fspath(configured) if configured is not None else "").strip()
    override = str(environment.get(override_env, "") or "").strip()
    if profile and override:
        if os.path.realpath(_normalized(profile)) != os.path.realpath(_normalized(override)):
            raise LayoutSelectionError(
                f"the {task} layout is selected twice: {parameter} names "
                f"{_normalized(profile)} and the harness-only override "
                f"{override_env} names {_normalized(override)}. Unset "
                f"{override_env}; a profile selects its layout with "
                f"simulation.parameters.{parameter}"
            )
        return LayoutSelection(task, _normalized(profile), SELECTED_BY_PROFILE)
    if profile:
        return LayoutSelection(task, _normalized(profile), SELECTED_BY_PROFILE)
    if override:
        return LayoutSelection(task, _normalized(override), SELECTED_BY_OVERRIDE)
    if default is None:
        raise LayoutSelectionError(
            missing_default
            or f"the {task} task has no default layout; set simulation.parameters.{parameter}"
        )
    return LayoutSelection(task, _normalized(default), SELECTED_BY_DEFAULT)


def box_pack_layout_path(
    configured: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> LayoutSelection:
    """The box-packing layout: parameter, then override, then the Duo board."""

    from . import box_pack

    return resolve_layout_path(
        BOX_PACK_TASK,
        configured=configured,
        default=box_pack.BOX_PACK_LAYOUT_PATH,
        environ=environ,
    )


def select_task_layout(
    task: str,
    configured: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None,
    *,
    robot: str | None,
    environ: Mapping[str, str] | None = None,
    run_number: int | None = None,
) -> LayoutSelection:
    """Resolve ``task``'s layout from its profile value; the one shared entry point."""

    layout_index = 0
    if isinstance(configured, Sequence) and not isinstance(
        configured, (str, bytes, os.PathLike)
    ):
        if task != BOX_PACK_TASK:
            raise LayoutSelectionError(
                f"only {BOX_PACK_TASK} accepts a list of hand-authored layouts"
            )
        if not configured:
            raise LayoutSelectionError("box_pack layout list must not be empty")
        if run_number is None or isinstance(run_number, bool) or run_number < 1:
            raise LayoutSelectionError(
                "selecting from a box_pack layout list requires a 1-based run_number"
            )
        layout_index = (run_number - 1) % len(configured)
        configured = configured[layout_index]

    if task == BOX_PACK_TASK:
        selection = box_pack_layout_path(configured, environ=environ)
    else:
        raise LayoutSelectionError(f"no hand-authored layout is selected for task {task!r}")
    return LayoutSelection(
        selection.task, selection.path, selection.selected_by, layout_index
    )


def describe_selection(selection: LayoutSelection) -> str:
    """Say which step of the order chose a layout, in a refusal's words.

    Inside the environment a profile's file arrives through the override name
    (the worker exports it there), so that step names both.
    """

    if selection.selected_by == SELECTED_BY_PROFILE:
        return f"selected by simulation.parameters.{selection.parameter}"
    if selection.selected_by == SELECTED_BY_OVERRIDE:
        return (
            f"selected by {selection.override_env}, which the worker sets from "
            f"--{selection.parameter.replace('_', '-')}"
        )
    return f"the {selection.task} default"


def load_selected_layout(
    selection: LayoutSelection, *, expected_objects: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Read a selected layout once; return ``(layout, raw payload, record)``.

    ``record`` names the file and the digest of the exact bytes that were
    validated, and the scene the layout was placed for, so provenance can
    never describe a file other than the one the objects came from.
    """

    real = Path(os.path.realpath(selection.path))
    try:
        data = real.read_bytes()
        payload = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LayoutError(
            f"{selection.task} layout JSON ({describe_selection(selection)}) "
            f"is unreadable: {real}: {exc}"
        ) from exc
    layout = validate_manual_layout(payload, expected_objects=expected_objects)
    record = {
        "path": str(real),
        "sha256": hashlib.sha256(data).hexdigest(),
        "scene": str(layout["source"]["scene"]),
    }
    return layout, payload, record


def layout_sweep_offset_mm(
    payload: Mapping[str, Any], key: str
) -> tuple[dict[str, float] | None, str | None]:
    """Return a normalized sweep offset and a non-raising validation reason."""

    source = payload.get("source")
    sweep = source.get("sweep") if isinstance(source, Mapping) else None
    if not isinstance(sweep, Mapping) or key not in sweep:
        return None, None
    value = sweep[key]
    if isinstance(value, Mapping) and set(value) == {"dx", "dy"}:
        components = (value["dx"], value["dy"])
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        components = (value[0], value[1])
    else:
        return None, f"source.sweep.{key} must be [dx, dy] or {{dx, dy}} in millimetres"
    if any(
        isinstance(component, bool) or not isinstance(component, (int, float))
        for component in components
    ):
        return None, f"source.sweep.{key} dx and dy must be finite numbers"
    try:
        if any(abs(component) > 1000 for component in components):
            return None, f"source.sweep.{key} dx and dy must be within ±1000 mm"
        converted = (float(components[0]), float(components[1]))
    except (OverflowError, TypeError, ValueError) as exc:
        return None, f"source.sweep.{key} dx and dy cannot be converted: {exc}"
    if not all(math.isfinite(component) for component in converted):
        return None, f"source.sweep.{key} dx and dy must be finite numbers"
    return {"dx": converted[0], "dy": converted[1]}, None


def require_layout_scene(
    layout: Mapping[str, Any],
    *,
    running_scene: str | None,
    selection: LayoutSelection,
) -> None:
    """Refuse a layout placed for another scene than the one this run uses.

    ``running_scene`` is the run's scene tag: the launcher's ``scenes[].tag``,
    which it checks against the scene contract's name, or in the worker the
    same tag derived from the validated contract. An environment that cannot
    name its scene is refused too: nothing would show the layout belongs there.
    """

    placed_for = str(layout["source"]["scene"])
    if not running_scene:
        raise LayoutSelectionError(
            f"{selection.task} layout {selection.path} ({describe_selection(selection)}) "
            f"was placed for scene {placed_for!r}, but this environment names no "
            "validated scene; a hand-authored layout is only placed on a schema-v2 "
            "scene contract (--desk scene with its --gs-background)"
        )
    if placed_for != running_scene:
        raise LayoutSelectionError(
            f"{selection.task} layout {selection.path} ({describe_selection(selection)}) "
            f"was placed for scene {placed_for!r}, but this run's scene is "
            f"{running_scene!r}. Select the layout placed for {running_scene!r} "
            f"with simulation.parameters.{selection.parameter}"
        )


def layout_robots(layout: Mapping[str, Any]) -> tuple[str, ...] | None:
    """The rigs a layout was placed for, or ``None`` when it names none.

    ``source.robots`` lists them (validated by ``validate_manual_layout``).
    """

    source = layout.get("source")
    listed = source.get("robots") if isinstance(source, Mapping) else None
    return None if listed is None else tuple(str(name) for name in listed)


def require_layout_rig(
    layout: Mapping[str, Any],
    *,
    robot: str | None,
    selection: LayoutSelection,
) -> None:
    """Refuse a layout placed for other rigs than the one this run builds.

    Rigs do not share a layout: each is placed for the rig that stands on its
    scene, so a layout placed for one rig is not valid for another. A layout that
    names no rig is accepted on any rig of its scene; one that names rigs is
    refused on any other rig, and on an environment that names no rig.
    """

    robots = layout_robots(layout)
    if robots is None:
        return
    placed_for = (
        f"the {robots[0]!r} rig"
        if len(robots) == 1
        else "the rigs " + ", ".join(repr(name) for name in robots)
    )
    if not robot:
        raise LayoutSelectionError(
            f"{selection.task} layout {selection.path} ({describe_selection(selection)}) "
            f"was placed for {placed_for}, but this environment names no rig"
        )
    if robot not in robots:
        raise LayoutSelectionError(
            f"{selection.task} layout {selection.path} ({describe_selection(selection)}) "
            f"was placed for {placed_for}, not {robot!r}. Select the layout placed "
            f"for {robot!r} with simulation.parameters.{selection.parameter}"
        )


__all__ = [
    "BOX_PACK_TASK",
    "LAYOUT_OVERRIDE_ENV",
    "LAYOUT_PARAMETERS",
    "LayoutSelection",
    "LayoutSelectionError",
    "SELECTED_BY_DEFAULT",
    "SELECTED_BY_OVERRIDE",
    "SELECTED_BY_PROFILE",
    "box_pack_layout_path",
    "describe_selection",
    "layout_robots",
    "layout_sweep_offset_mm",
    "load_selected_layout",
    "require_layout_rig",
    "require_layout_scene",
    "resolve_layout_path",
    "select_task_layout",
]
