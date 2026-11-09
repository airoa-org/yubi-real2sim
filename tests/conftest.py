# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture()
def clean_env(tmp_path: Path):
    """Isolate tests from operator-specific real2sim environment settings."""
    snapshot = dict(os.environ)
    for name in tuple(os.environ):
        if name.startswith(("REAL2SIM_", "RS_")) or name in {
            "PINHOLE_LEFT",
            "PINHOLE_RIGHT",
        }:
            os.environ.pop(name, None)
    empty = tmp_path / "empty.env"
    empty.write_text("", encoding="utf-8")
    try:
        yield empty
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


@pytest.fixture()
def isolated_environ(clean_env: Path) -> Path:
    """Empty env file for launcher.main calls that must not read repo .env."""
    return clean_env


@pytest.fixture(scope="session")
def mujoco():
    return pytest.importorskip("mujoco")


@pytest.fixture(scope="session")
def robosuite():
    return pytest.importorskip("robosuite")


@pytest.fixture(scope="session")
def ruckig():
    return pytest.importorskip("ruckig")


@pytest.fixture(scope="session")
def msgpack():
    return pytest.importorskip("msgpack")


@pytest.fixture(scope="session")
def matplotlib():
    return pytest.importorskip("matplotlib")


@pytest.fixture(scope="session")
def openpi():
    return pytest.importorskip("openpi")


@pytest.fixture(scope="session")
def jax():
    module = pytest.importorskip("jax")
    pytest.importorskip("flax")
    return module


@pytest.fixture(scope="session")
def real_inputs_root():
    from real2sim.artifacts import resolve_inputs_root

    root = resolve_inputs_root() or REPO_ROOT / "local-inputs"
    if not root.is_dir():
        pytest.skip("real input assets are not installed")
    return root


@pytest.fixture(scope="session")
def gpu_egl(mujoco):
    previous = os.environ.get("MUJOCO_GL")
    os.environ["MUJOCO_GL"] = "egl"
    try:
        renderer = mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"))
    except Exception as error:
        pytest.skip(f"MuJoCo EGL is unavailable: {error}")
    finally:
        if previous is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous
    renderer.close()
    return mujoco


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    heavy_fixtures = {
        "mujoco",
        "robosuite",
        "ruckig",
        "msgpack",
        "matplotlib",
        "openpi",
        "jax",
    }
    for item in items:
        fixtures = set(item.fixturenames)
        if fixtures & heavy_fixtures:
            item.add_marker(pytest.mark.heavy)
        if "real_inputs_root" in fixtures:
            item.add_marker(pytest.mark.assets)
        if "gpu_egl" in fixtures:
            item.add_marker(pytest.mark.gpu)
