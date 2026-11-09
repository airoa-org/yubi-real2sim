# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The cup and plate match the real objects: every cup 85.0 mm tall at its
unchanged 73.1 mm diameter, every plate 180.0 mm across, masses unchanged.

Measured the way a rollout builds the scene: each object as the robosuite
MujocoXMLObject the tasks use, merged into a MujocoWorldBase, compiled by
MuJoCo, and read back from the compiled collision geometry and body masses.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("robosuite")

from robosuite.models.world import MujocoWorldBase  # noqa: E402

from robosuite_env.contracts.assets import runtime_geometry_for_object  # noqa: E402
from robosuite_env.paths import ASSETS_ROOT  # noqa: E402
from robosuite_env.tasks import objects  # noqa: E402

TOL_MM = 0.05
EXPECTED = {
    # name: (class, height mm or None, max diameter mm, mass g)
    "cup": (objects.CupObject, 85.0, None, 28.0),
    "plate": (objects.PlateObject, None, 180.0, 50.0),
}


def _installed() -> bool:
    return all((ASSETS_ROOT / "trellis_cup_plate" / name).is_file() for name in (
        "coll/cup_000.obj", "coll/plate_000.obj", "cup.png", "plate.png",
    ))


@pytest.fixture(scope="module")
def compiled():
    if not _installed():
        pytest.skip("licensed trellis_cup_plate meshes are not installed")
    world = MujocoWorldBase()
    for name, (cls, *_rest) in EXPECTED.items():
        obj = cls(name=name)
        world.worldbody.append(obj.get_obj())
        world.merge_assets(obj)
    model = world.get_model(mode="mujoco")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _measure(model, data, name):
    root = model.body(f"{name}_main").id
    bodies = {b for b in range(model.nbody) if b == root or _is_descendant(model, b, root)}
    lo = np.full(3, np.inf)
    hi = -lo
    radial = 0.0
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in bodies or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        mid = model.geom_dataid[g]
        start = model.mesh_vertadr[mid]
        verts = model.mesh_vert[start:start + model.mesh_vertnum[mid]]
        world = (data.geom_xmat[g].reshape(3, 3) @ verts.T).T + data.geom_xpos[g]
        lo = np.minimum(lo, world.min(0))
        hi = np.maximum(hi, world.max(0))
        centre = data.xpos[root]
        radial = max(radial, float(np.hypot(world[:, 0] - centre[0], world[:, 1] - centre[1]).max()))
    mass = float(model.body_subtreemass[root])
    return (hi - lo) * 1000.0, radial * 1000.0, mass * 1000.0


def _is_descendant(model, body, root):
    while body != 0:
        if body == root:
            return True
        body = model.body_parentid[body]
    return False


@pytest.mark.parametrize("name", list(EXPECTED))
def test_compiled_size_and_mass_match_the_real_objects(compiled, name):
    model, data = compiled
    _cls, height, diameter, mass = EXPECTED[name]
    extent, radial, measured_mass = _measure(model, data, name)
    assert measured_mass == pytest.approx(mass, abs=0.01)
    if height is not None:
        assert extent[2] == pytest.approx(height, abs=TOL_MM)
        # The resize is height-only: the footprint stays the registered cup's.
        assert extent[0] == pytest.approx(73.12, abs=TOL_MM)
        assert extent[1] == pytest.approx(73.17, abs=TOL_MM)
    else:
        assert max(extent[0], extent[1]) == pytest.approx(diameter, abs=TOL_MM)
    geometry = "cup" if name.startswith("cup") else "plate"
    contract = runtime_geometry_for_object(geometry)["compiled_collision_observed"]
    assert radial == pytest.approx(1000.0 * contract["xy_radius_m"], abs=TOL_MM)


def test_success_criterion_plate_uses_the_resized_plate():
    from robosuite_env.tasks.success.cup_on_plate import CupOnPlateSpec

    plate = runtime_geometry_for_object("plate")["semantic_sites_m"]
    spec = CupOnPlateSpec()
    assert spec.plate_support_local == plate["top_site"]
    assert spec.plate_bottom_radius_m == plate["horizontal_radius_site"][0]
