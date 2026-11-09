# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Architecture checks for the task package boundary."""

from importlib import import_module
from pathlib import Path

from robosuite_env.paths import PACKAGE_ROOT


TASKS_ROOT = PACKAGE_ROOT / "tasks"

# Task implementations live only below ``robosuite_env/tasks``; the flat
# pre-split modules must never come back. This tuple is frozen on purpose: it
# records what used to sit at the package root, so unlike a list of what the
# package holds today it cannot go stale as tasks are added.
OLD_ROOT_MODULES = (
    "objects.py",
    "robot_scene.py",
    "cup_plate_env.py",
    "cup_plate_env_gw.py",
)

# The layers the split gave the package: shared task definitions directly
# under ``tasks/``, robosuite environments under ``tasks/envs/``, success
# criteria under ``tasks/success/``.
TASK_SUBPACKAGES = ("envs", "success")


def _task_modules():
    """Every implementation module below ``tasks/``, as relative posix paths.

    Discovered rather than enumerated, so new task modules are covered. The
    statement checked is: a task implementation lives in one of the split's
    layers and nowhere else, and no module has a twin back at the package
    root. Derived, that statement also covers the tasks nobody has written yet.
    """

    return tuple(
        sorted(
            path.relative_to(TASKS_ROOT).as_posix()
            for path in TASKS_ROOT.rglob("*.py")
            if "__pycache__" not in path.parts and path.name != "__init__.py"
        )
    )


def test_task_implementations_live_only_below_tasks_package():
    for relative in OLD_ROOT_MODULES:
        assert not (PACKAGE_ROOT / relative).exists()
    assert not tuple((PACKAGE_ROOT / "success_criteria").glob("*.py"))

    modules = _task_modules()
    assert modules, "the task package cannot be empty"
    # The general form of the OLD_ROOT_MODULES check: whatever the package
    # holds now, none of it may also sit flat at the package root.
    for relative in modules:
        assert not (PACKAGE_ROOT / Path(relative).name).exists(), relative


def test_the_task_package_keeps_the_layers_the_split_gave_it():
    by_layer = {layer: [] for layer in TASK_SUBPACKAGES}
    shared = []
    for relative in _task_modules():
        parts = Path(relative).parts
        assert len(parts) <= 2, f"{relative}: tasks/ has no third layer"
        if len(parts) == 1:
            shared.append(relative)
            continue
        assert parts[0] in TASK_SUBPACKAGES, f"{relative}: unknown subpackage"
        by_layer[parts[0]].append(relative)

    assert shared, "the shared task definitions vanished from tasks/"
    for layer, modules in by_layer.items():
        assert modules, f"tasks/{layer}/ lost its modules"


def test_every_success_criterion_module_is_wired_into_the_registry():
    """A criterion module on disk that nothing can select is dead weight."""

    from robosuite_env.tasks.success import registry

    on_disk = {
        f"robosuite_env.tasks.success.{Path(relative).stem}"
        for relative in _task_modules()
        if Path(relative).parts[0] == "success"
        # ``registry`` is the table itself; ``_``-prefixed modules are shared
        # helpers, not criteria.
        and Path(relative).name != "registry.py"
        and not Path(relative).name.startswith("_")
    }

    wired = set(registry._CRITERIA.values())
    assert {cls.__module__ for cls in wired} == on_disk
    # The name a criterion answers to is the name it reports in its records.
    for selector, cls in registry._CRITERIA.items():
        assert cls.name == selector


def test_every_task_module_imports(robosuite):
    for relative in _task_modules():
        import_module("robosuite_env.tasks." + relative[: -len(".py")].replace("/", "."))


def test_task_resources_keep_the_central_runtime_paths(robosuite):
    from robosuite_env.paths import ASSETS_ROOT, OBJECT_CALIBRATION_ROOT
    from robosuite_env.robots.base_placement import (
        DEFAULT_BASE_PLACEMENTS,
        default_base_placement_path,
    )
    from robosuite_env.tasks import objects

    assert objects.ASSETS_ROOT == ASSETS_ROOT
    # Every rig's default calibration resolves under the central calibration
    # root. The task layer no longer reads one at import time.
    for robot in DEFAULT_BASE_PLACEMENTS:
        assert default_base_placement_path(robot).parent == OBJECT_CALIBRATION_ROOT


def test_registered_task_variants_extend_their_shared_environments(robosuite):
    from robosuite_env.tasks.envs.cup_plate import CupPlateLift
    from robosuite_env.tasks.envs.cup_plate_registered import CupPlateLiftGW

    assert issubclass(CupPlateLiftGW, CupPlateLift)
