# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The cardboard box carries its mass low and from its own hulls.

The box is scaled to 11/14 of its captured height. Its 0.100 kg is spread over
the 23 collision hulls by per-hull densities solved so the centre of mass lands
near the base; a thin-shell <inertial> (0.2794 kg at 87 mm) would tip the box
off the desk at 35 deg. The 23 hulls are the box every published Duo box run
used. The Flat lineage's base slab box_023 is a 24th collision geom: it closes
the cavity floor and carries no mass.

The builder's thin-shell helper is also exercised here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER = REPO_ROOT / "scripts" / "build_office_object_assets.py"
ASSETS = REPO_ROOT / "robosuite_env" / "assets"
BOX_XML = ASSETS / "box.xml"
BOX_VISUAL = ASSETS / "trellis_office" / "box_vis.obj"

BOX_MASS_KG = 0.100
#: Solved per-hull densities, kg/m^3, ascending.
SOLVED_HULL_DENSITIES = (
    2.48825, 2.93131, 3.67761, 3.96987, 5.71147, 8.86265, 9.97161, 10.1155,
    15.3379, 16.0318, 19.7898, 33.6372, 36.1085, 50.3626, 59.8414, 88.5817,
    93.0678, 126.608, 129.174, 131.641, 166.017, 182.43, 188.436,
)

BASE_SLAB_GEOM = "p_c23"


def _builder():
    pytest.importorskip("trimesh")
    spec = importlib.util.spec_from_file_location("build_office_object_assets", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _object_body():
    return ET.parse(BOX_XML).getroot().find(".//body[@name='object']")


def _compiled_box():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(BOX_XML))
    body = next(
        body_id
        for body_id in range(model.nbody)
        if model.body(body_id).name == "object"
    )
    return model, body


def test_thin_shell_of_a_hollow_cube_matches_the_closed_form():
    builder = _builder()
    side, mass = 0.2, 0.5
    h = side / 2.0
    corners = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)])
    index = {tuple(np.sign(c).astype(int)): i for i, c in enumerate(corners)}
    faces = []
    for axis in range(3):
        for sign in (-1, 1):
            others = [a for a in range(3) if a != axis]
            quad = []
            for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                key = [0, 0, 0]
                key[axis], key[others[0]], key[others[1]] = sign, u, v
                quad.append(index[tuple(key)])
            faces += [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]
    offset = np.array([0.3, -0.1, 0.05])
    centre, full = builder.thin_shell_mass_properties(corners + offset, np.array(faces), mass)
    np.testing.assert_allclose(centre, offset, atol=1e-12)
    # A hollow cube of side a: I = 5 m a^2 / 18 about each axis, no products.
    np.testing.assert_allclose(full[:3], 5.0 * mass * side**2 / 18.0, rtol=1e-12)
    np.testing.assert_allclose(full[3:], 0.0, atol=1e-12)


def test_box_mjcf_takes_its_mass_from_hull_densities_not_an_inertial():
    body = _object_body()
    assert body.find("inertial") is None, (
        "box.xml declares an <inertial>; the box sets its mass through "
        "per-hull densities instead"
    )
    collision = [g for g in body.findall("geom") if g.get("group") == "0"]
    assert len(collision) == 24
    slab = [g for g in collision if g.get("name") == BASE_SLAB_GEOM]
    assert len(slab) == 1
    assert slab[0].get("mass") == "0" and slab[0].get("density") is None
    visual = [g for g in body.findall("geom") if g.get("group") == "1"]
    assert len(visual) == 1 and visual[0].get("mass") == "0"

    densities = sorted(
        float(g.get("density")) for g in collision if g.get("name") != BASE_SLAB_GEOM
    )
    # Exactly the solved densities, so a re-solve that moved the
    # mass around cannot pass.
    assert densities == pytest.approx(SOLVED_HULL_DENSITIES, rel=1e-9)


@pytest.mark.skipif(not BOX_VISUAL.is_file(), reason="the box meshes (gitignored) are not installed")
def test_the_compiled_box_carries_its_mass_near_its_base():
    model, body = _compiled_box()
    assert model.body_mass[body] == pytest.approx(BOX_MASS_KG, rel=1e-6)
    centre = model.body_ipos[body]
    assert abs(centre[0]) < 0.005
    assert -0.045 < centre[1] < -0.025
    # 31 mm up a 232 mm box, against the 87 mm the thin-shell inertial gave.
    assert 0.025 < centre[2] < 0.045


@pytest.mark.skipif(not BOX_VISUAL.is_file(), reason="the box meshes (gitignored) are not installed")
def test_the_box_as_robosuite_compiles_it_keeps_that_mass():
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("robosuite")
    from robosuite.models.world import MujocoWorldBase

    from robosuite_env.tasks.objects import BoxObject

    world = MujocoWorldBase()
    box = BoxObject("box")
    world.worldbody.append(box.get_obj())
    world.merge_assets(box)
    model = world.get_model(mode="mujoco")
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_main")
    standalone, standalone_body = _compiled_box()
    assert model.body_mass[body] == pytest.approx(BOX_MASS_KG, rel=1e-6)
    np.testing.assert_allclose(
        model.body_ipos[body], standalone.body_ipos[standalone_body], atol=1e-9
    )
