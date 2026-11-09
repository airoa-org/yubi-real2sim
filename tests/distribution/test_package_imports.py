# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

SRC = Path(__file__).resolve().parents[2] / "src"


def test_fr3_controller_selection_imports_without_robosuite_env(tmp_path):
    # The wheel ships only real2sim, so these modules must not import robosuite_env.
    code = (
        "import sys; sys.modules['robosuite_env'] = None; "
        "import real2sim.fr3_controllers, real2sim.config.controllers"
    )
    subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env={"PYTHONPATH": str(SRC)}, check=True
    )
