# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest

try:  # pragma: no cover - heavy dependencies absent from clean-lock CI
    import mujoco
    from scipy.spatial import ConvexHull

    from robosuite_env.tools.build_yubi_full_collision import (
        GRASP_CORRIDOR_VALIDATION,
        GUARD_CONTACT,
        read_obj,
    )
except ImportError:  # pragma: no cover - heavy dependencies absent from CI
    mujoco = None
    ConvexHull = None
    GRASP_CORRIDOR_VALIDATION = None
    GUARD_CONTACT = None
    read_obj = None


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET_ROOT = REPO_ROOT / "robosuite_env/assets"
BUILDER = REPO_ROOT / "robosuite_env/tools/build_yubi_full_collision.py"
CONTRACT = ASSET_ROOT / "yubi_guard_collision_manifest.json"
SIDE_CASES = (
    ("right", ASSET_ROOT / "yubi_right.xml", ASSET_ROOT / "yubi_right", 178, 240, 11),
    ("left", ASSET_ROOT / "yubi_left.xml", ASSET_ROOT / "yubi_left", 179, 241, 12),
)
requires_guard_generator = pytest.mark.skipif(
    ConvexHull is None or read_obj is None,
    reason="SciPy and the guard generator are not installed by requirements-lock.ci.txt",
)
EXPECTED_BODY_MASS = np.asarray([0.0, 0.3, 0.0, 0.25366304438539744, 0.0, 0.03, 0.03])
EXPECTED_BODY_INERTIA = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [0.001, 0.001, 0.0008],
        [0.0, 0.0, 0.0],
        [0.00024157532941164017, 0.0001834786887866505, 9.484148979499522e-05],
        [0.0, 0.0, 0.0],
        [1e-05, 1e-05, 1e-05],
        [1e-05, 1e-05, 1e-05],
    ]
)


def _is_collision(geom: ET.Element) -> bool:
    return (
        geom.get("type") == "mesh"
        and int(geom.get("contype", "1")) != 0
        and int(geom.get("conaffinity", "1")) != 0
    )


def _compiled_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        for geom_id in range(model.ngeom)
    ]


def _used_face_vertices(path: Path) -> np.ndarray:
    vertices, groups = read_obj(path)
    vertex_ids = np.unique(
        np.asarray(
            [face for group in groups for face in group.faces], dtype=np.int64
        ).reshape(-1)
    )
    return vertices[vertex_ids]


def _outside_proxy_union(
    points: np.ndarray, hull_equations: list[np.ndarray]
) -> np.ndarray:
    """Return each point's least violation across convex proxy hulls.

    Batching bounds temporary memory for the detailed visual meshes.  Einsum
    is intentional: it keeps the result of this highly rectangular matrix
    multiplication independent of the BLAS backend.
    """

    outside = np.full(len(points), np.inf)
    for equations in hull_equations:
        for start in range(0, len(points), 512):
            stop = min(start + 512, len(points))
            violation = np.max(
                np.einsum("ij,kj->ik", points[start:stop], equations[:, :3])
                + equations[:, 3],
                axis=1,
            )
            outside[start:stop] = np.minimum(outside[start:stop], violation)
    return outside


@requires_guard_generator
@pytest.mark.usefixtures("mujoco")
def test_checked_in_guard_outputs_are_generator_clean():
    result = subprocess.run(
        [sys.executable, str(BUILDER), "--check"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_namespaced_guards_are_never_legacy_override_targets():
    # robosuite_env.rollout.contact_overrides belongs to the rollout bucket, so
    # import it inside the test and keep this module collectable without it.
    from robosuite_env.rollout.contact_overrides import (
        is_legacy_finger_contact_override_target,
    )

    for geom_name in (
        "guard_finger_left_geometry0",
        "gripper0_right_guard_finger_left_geometry0",
        "gripper1_right_guard_body_geometry42",
    ):
        assert not is_legacy_finger_contact_override_target(
            body_name="gripper0_right_finger_left",
            geom_name=geom_name,
            geom_group=0,
        )
    assert is_legacy_finger_contact_override_target(
        body_name="gripper0_right_finger_left",
        geom_name="gripper0_right_finger_left_pad_collision_h000",
        geom_group=0,
    )


@requires_guard_generator
@pytest.mark.usefixtures("mujoco")
@pytest.mark.parametrize(
    "side,xml_path,asset_dir,collision_count,ngeom,pad_count", SIDE_CASES
)
def test_all_visual_parts_have_shared_collision_coverage_and_reviewed_contact(
    side: str,
    xml_path: Path,
    asset_dir: Path,
    collision_count: int,
    ngeom: int,
    pad_count: int,
):
    root = ET.parse(xml_path).getroot()
    mesh_files = {
        mesh.get("name"): mesh.get("file") for mesh in root.findall("./asset/mesh")
    }
    visual_geoms = [
        geom
        for geom in root.iter("geom")
        if geom.get("type") == "mesh"
        and geom.get("contype") == "0"
        and geom.get("conaffinity") == "0"
    ]
    collision_geoms = [geom for geom in root.iter("geom") if _is_collision(geom)]
    guard_geoms = [
        geom
        for geom in collision_geoms
        if geom.get("name", "").startswith("guard_")
    ]
    assert len(visual_geoms) == 62
    assert len(collision_geoms) == collision_count
    assert len(guard_geoms) == 102
    assert len({mesh_files[geom.get("mesh")] for geom in guard_geoms}) == 102
    for geom in guard_geoms:
        for key, expected in GUARD_CONTACT.items():
            assert geom.get(key) == expected

    expected_gids = {
        str(part["gid"])
        for part in json.loads((asset_dir / "manifest.json").read_text())["parts"]
    }
    covered_gids: set[str] = set()
    for geom in collision_geoms:
        relative = mesh_files[geom.get("mesh")]
        if relative is None:
            continue
        stem = Path(relative).stem
        if "/guard_collision/" in relative:
            stem = stem.split("_proxy_h", 1)[0].removesuffix("_convex")
            covered_gids.add(stem)
        elif "/coll_mj_hulls/" in relative:
            covered_gids.add(stem.rsplit("_h", 1)[0])
    assert covered_gids == expected_gids

    legacy_finger = [
        geom
        for geom in collision_geoms
        if geom.get("name", "").startswith("finger_")
    ]
    legacy_pads = [geom for geom in legacy_finger if "pad_collision" in geom.get("name", "")]
    legacy_structure = [geom for geom in legacy_finger if geom not in legacy_pads]
    assert len(legacy_structure) == 33
    assert len(legacy_pads) == pad_count
    for geom in legacy_structure:
        assert geom.get("friction") == "0.5 0.05 0.0001"
        assert geom.get("solref") == "0.01 1"
        assert geom.get("solimp") == "0.95 0.99 0.001"
        assert geom.get("priority") == "0"
    for geom in legacy_pads:
        assert geom.get("friction") == "4 0.1 0.0001"
        assert geom.get("solref") == "0.01 1"
        assert geom.get("solimp") == "0.95 0.99 0.001"
        assert geom.get("priority") == "1"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert model.ngeom == ngeom
    names = _compiled_names(model)
    guard_ids = [index for index, name in enumerate(names) if name.startswith("guard_")]
    assert len(guard_ids) == 102
    np.testing.assert_array_equal(model.geom_condim[guard_ids], 3)
    np.testing.assert_allclose(
        model.geom_friction[guard_ids],
        np.tile([0.1, 0.005, 0.0001], (len(guard_ids), 1)),
    )
    np.testing.assert_allclose(
        model.geom_solref[guard_ids], np.tile([0.004, 1.0], (len(guard_ids), 1))
    )
    np.testing.assert_allclose(
        model.geom_solimp[guard_ids, :3],
        np.tile([0.99, 0.9999, 0.0001], (len(guard_ids), 1)),
    )
    np.testing.assert_array_equal(model.geom_priority[guard_ids], 1)
    np.testing.assert_allclose(model.geom_margin[guard_ids], 0.001)
    np.testing.assert_array_equal(model.geom_gap[guard_ids], 0.0)
    np.testing.assert_array_equal(model.geom_contype[guard_ids], 2)
    np.testing.assert_array_equal(model.geom_conaffinity[guard_ids], 1)
    np.testing.assert_array_equal(model.body_mass, EXPECTED_BODY_MASS)
    np.testing.assert_array_equal(model.body_inertia, EXPECTED_BODY_INERTIA)


@requires_guard_generator
@pytest.mark.usefixtures("mujoco")
@pytest.mark.parametrize("_side,xml_path,_asset_dir,_collision,_ngeom,_pads", SIDE_CASES)
def test_guard_is_excluded_from_legacy_runtime_override_and_self_contact(
    _side: str,
    xml_path: Path,
    _asset_dir: Path,
    _collision: int,
    _ngeom: int,
    _pads: int,
):
    from robosuite_env.rollout.contact_overrides import (
        legacy_finger_contact_override_ids,
    )

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    names = _compiled_names(model)
    guard_ids = np.asarray(
        [index for index, name in enumerate(names) if name.startswith("guard_")],
        dtype=np.int64,
    )
    override_ids = np.asarray(
        list(legacy_finger_contact_override_ids(model, mujoco)), dtype=np.int64
    )
    assert len(set(guard_ids.tolist()) & set(override_ids.tolist())) == 0
    assert len(override_ids) in {44, 45}
    before = (
        model.geom_solref[guard_ids].copy(),
        model.geom_friction[guard_ids].copy(),
        model.geom_priority[guard_ids].copy(),
    )
    model.geom_solref[override_ids] = [0.04, 1.0]
    model.geom_friction[override_ids] = [0.25, 0.025, 0.0001]
    model.geom_priority[override_ids] = 1
    np.testing.assert_array_equal(model.geom_solref[guard_ids], before[0])
    np.testing.assert_array_equal(model.geom_friction[guard_ids], before[1])
    np.testing.assert_array_equal(model.geom_priority[guard_ids], before[2])

    for qpos in np.linspace(0.0, 0.7854, 17):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        assert data.ncon == 0


@requires_guard_generator
@pytest.mark.usefixtures("mujoco")
def test_source_collision_surface_is_inside_guard_proxy_union():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["grasp_corridor_validation"] == GRASP_CORRIDOR_VALIDATION
    assert GRASP_CORRIDOR_VALIDATION["geometry0_max_clearance_loss_m"] <= 0.003
    assert GRASP_CORRIDOR_VALIDATION["geometry2_max_clearance_loss_m"] <= 0.005
    assert GRASP_CORRIDOR_VALIDATION["missing_source_grid_points"] == 0
    assert GRASP_CORRIDOR_VALIDATION["center_axis_guard_contacts"] == 0

    for _side, _xml, asset_dir, *_rest in SIDE_CASES:
        side = contract["sides"][_side]
        for part in side["parts"]:
            gid = part["gid"]
            source_surface_vertices = _used_face_vertices(
                asset_dir / "coll_mj" / f"{gid}.obj"
            )
            proxy_paths = sorted((asset_dir / "guard_collision").glob(f"{gid}_proxy_h*.obj"))
            if not proxy_paths:
                proxy_paths = [asset_dir / "guard_collision" / f"{gid}_convex.obj"]
            hull_equations = []
            for proxy_path in proxy_paths:
                proxy_vertices, _ = read_obj(proxy_path)
                hull_equations.append(ConvexHull(proxy_vertices).equations)
            outside = _outside_proxy_union(source_surface_vertices, hull_equations)
            assert float(outside.max()) <= 2e-8, (asset_dir.name, gid, outside.max())


@requires_guard_generator
@pytest.mark.usefixtures("mujoco")
def test_visual_surface_is_inside_guard_proxy_union_within_contact_margin():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for side_name, _xml, asset_dir, *_rest in SIDE_CASES:
        for part in contract["sides"][side_name]["parts"]:
            gid = part["gid"]
            visual_surface_vertices = _used_face_vertices(
                asset_dir / "visual_mj" / f"{gid}.obj"
            )
            proxy_paths = sorted(
                (asset_dir / "guard_collision").glob(f"{gid}_proxy_h*.obj")
            )
            if not proxy_paths:
                proxy_paths = [
                    asset_dir / "guard_collision" / f"{gid}_convex.obj"
                ]
            hull_equations = []
            for proxy_path in proxy_paths:
                proxy_vertices, _ = read_obj(proxy_path)
                hull_equations.append(ConvexHull(proxy_vertices).equations)
            outside = _outside_proxy_union(visual_surface_vertices, hull_equations)
            assert float(outside.max()) <= 0.0001, (
                asset_dir.name,
                gid,
                outside.max(),
            )


@requires_guard_generator
@pytest.mark.usefixtures("mujoco")
@pytest.mark.parametrize("speed_m_s", [0.01, 0.4])
def test_hard_margin_contact_stops_probe_before_surface_penetration(speed_m_s: float):
    xml = """
    <mujoco>
      <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
      <worldbody>
        <geom type="plane" size="1 1 .1" priority="0"/>
        <body pos="0 0 .012">
          <freejoint/>
          <geom type="sphere" size=".01" mass=".01" condim="3"
                friction=".1 .005 .0001" solref=".004 1"
                solimp=".99 .9999 .0001" priority="1"
                margin=".001" gap="0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qvel[2] = -speed_m_s
    distances = []
    first_contact = None
    for _ in range(500):
        mujoco.mj_step(model, data)
        if data.ncon and first_contact is None:
            first_contact = data.contact[0]
        distances.extend(data.contact[index].dist for index in range(data.ncon))
    assert distances and first_contact is not None and min(distances) > 0.0
    assert first_contact.dim == 3
    np.testing.assert_allclose(
        first_contact.friction, [0.1, 0.1, 0.005, 0.0001, 0.0001]
    )
    np.testing.assert_allclose(first_contact.solref, [0.004, 1.0])
    np.testing.assert_allclose(
        first_contact.solimp, [0.99, 0.9999, 0.0001, 0.5, 2.0]
    )
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert all(warning.number == 0 for warning in data.warning)


@requires_guard_generator
@pytest.mark.usefixtures("mujoco")
@pytest.mark.parametrize("_side,xml_path,_asset_dir,_collision,_ngeom,_pads", SIDE_CASES)
def test_standalone_guard_step_cost_is_bounded(
    _side: str,
    xml_path: Path,
    _asset_dir: Path,
    _collision: int,
    _ngeom: int,
    _pads: int,
):
    start = time.perf_counter()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    compile_seconds = time.perf_counter() - start
    data = mujoco.MjData(model)
    start = time.perf_counter()
    for _ in range(1000):
        mujoco.mj_step(model, data)
    step_seconds = time.perf_counter() - start
    assert compile_seconds < 5.0
    assert step_seconds < 0.5
    assert data.ncon == 0
    assert all(warning.number == 0 for warning in data.warning)
