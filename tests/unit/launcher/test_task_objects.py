# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Named tasks and file inputs share one early validation contract."""
from types import SimpleNamespace

import pytest

from real2sim.config.task_objects import SUPPORTED_TASKS, validate_task_objects


@pytest.mark.parametrize("value", SUPPORTED_TASKS)
def test_named_task(value):
    validate_task_objects(value, require_file=True)


def test_position_path_requires_a_file_at_worker_start(tmp_path):
    path = tmp_path / "positions.json"
    validate_task_objects(str(path))
    with pytest.raises(ValueError, match="positions JSON"):
        validate_task_objects(str(path), require_file=True)
    path.write_text('{"cup_xy": [0, 0], "plate_xy": [0.1, 0.2]}')
    validate_task_objects(str(path), require_file=True)


def test_model_rejects_unknown_input_before_startup(monkeypatch):
    from robosuite_env.rollout.build import model

    def unexpected(*args):
        raise AssertionError("unknown input reached GS startup")

    monkeypatch.setattr(model, "setup_gs_server", unexpected)
    with pytest.raises(SystemExit, match="unknown task.objects"):
        model.run(SimpleNamespace(A=SimpleNamespace(objects="unknown_task")))
