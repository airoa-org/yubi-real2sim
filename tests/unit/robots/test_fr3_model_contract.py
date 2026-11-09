# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The MuJoCo FR3 arm matches the public franka_description FR3 URDF (docs/fr3_impedance.md#arm-model)."""
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
_GROUP = "run `uv sync --frozen --no-build-isolation --group fr3-model-contract`; CI runs it with no skips"
pin = pytest.importorskip("pinocchio", reason=f"needs pinocchio (PyPI: pin) for the URDF reference; {_GROUP}")
xacro = pytest.importorskip("xacro", reason=f"needs xacro to render fr3.urdf.xacro; {_GROUP}")
from xacro import substitution_args

from robosuite_env.paths import ASSETS_ROOT
from robosuite_env.robots.profiles import get_robot_profile

READY_Q = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]
NULL_SPACE_HOME_Q = {
    "L": [1.2119, .4187, -.1350, -2.4298, 1.3875, 3.9066, .4440],
    "R": [-1.0224, .3637, .3511, -2.5326, -1.0257, 3.6844, 1.1113],
}
PLACEHOLDER_BODY = "right_hand"  # robosuite flange inertial placeholder.
REFLECTED_ROTOR_INERTIA = [0.6057, 0.6057, 0.4625, 0.4625, 0.2055, 0.2055, 0.2055]  # kg·m², docs/fr3_impedance.md#arm-model
SIMULATED_ARMATURE = [5 / i for i in range(1, 8)]  # robosuite RobotModel default for joints that declare none.


def _configurations(lower, upper, count=5):
    mid, half = (lower + upper) / 2, (upper - lower) / 2
    rng = np.random.default_rng(0)
    return [np.asarray(READY_Q), *map(np.asarray, NULL_SPACE_HOME_Q.values()),
            *(mid + 0.9 * half * rng.uniform(-1, 1, 7) for _ in range(count))]


@pytest.fixture(scope="module")
def urdf_doc(repo_root: Path):
    description = repo_root / "third_party" / "franka_description"
    xacro_path = description / "robots" / "fr3" / "fr3.urdf.xacro"
    if not xacro_path.is_file():
        pytest.skip(f"{xacro_path} is missing; run scripts/bootstrap.sh")
    with pytest.MonkeyPatch.context() as patch:
        # Resolve $(find franka_description) to the bootstrap checkout without an ament index.
        patch.setattr(substitution_args, "_eval_find", {"franka_description": str(description)}.__getitem__)
        return xacro.process_file(str(xacro_path), mappings={"hand": "false", "ros2_control": "false"})


@pytest.fixture(scope="module")
def urdf_model(urdf_doc):
    return pin.buildModelFromXML(urdf_doc.toxml())


def _compile_without_missing_meshes(spec):
    """Compile without the gitignored collision meshes; bodies that lose a geom must not take inertia from it."""
    assert spec.compiler.inertiafromgeom != mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE
    missing = {m.name for m in spec.meshes if not (ASSETS_ROOT / spec.meshdir / m.file).is_file()}
    for body in spec.bodies:
        dropped = [g for g in body.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname in missing]
        assert not dropped or body.explicitinertial, body.name
        for geom in dropped:
            spec.delete(geom)
    for mesh in [m for m in spec.meshes if m.name in missing]:
        spec.delete(mesh)
    return spec.compile()


@pytest.fixture
def mj_arm():
    model = _compile_without_missing_meshes(mujoco.MjSpec.from_file(str(ASSETS_ROOT / "fr3_robot.xml")))
    return model, mujoco.MjData(model)


def test_joint_order_matches_urdf(urdf_model, mj_arm):
    model, _ = mj_arm
    urdf_names = [urdf_model.names[i] for i in range(1, urdf_model.njoints)]
    assert urdf_names == [f"fr3_joint{i}" for i in range(1, 8)]
    assert [model.joint(i).name for i in range(model.njnt)] == [f"joint{i}" for i in range(1, 8)]
    np.testing.assert_array_equal(model.jnt_qposadr, np.arange(7))
    np.testing.assert_array_equal(model.jnt_dofadr, np.arange(7))
    for side, arm in get_robot_profile("fr3").arms.items():
        prefix = arm.base_body.removesuffix("link0")
        assert arm.joint_names == tuple(f"{prefix}joint{i}" for i in range(1, 8)), side


def test_joint_limits_damping_and_friction_match_urdf(urdf_model, mj_arm):
    model, _ = mj_arm
    np.testing.assert_array_equal(model.jnt_range, np.c_[urdf_model.lowerPositionLimit, urdf_model.upperPositionLimit])
    assert model.jnt_limited.all()
    assert [model.joint(j).name for j in model.actuator_trnid[:, 0]] == [f"joint{i}" for i in range(1, 8)]
    np.testing.assert_array_equal(model.actuator_ctrlrange, np.c_[-urdf_model.effortLimit, urdf_model.effortLimit])
    np.testing.assert_array_equal(model.dof_damping, urdf_model.damping)
    np.testing.assert_array_equal(model.dof_frictionloss, urdf_model.friction)


def test_saturated_actuators_apply_the_urdf_effort_limit(urdf_model, mj_arm):
    pytest.importorskip("robosuite")
    from robosuite_env.robots.fr3.model import FR3

    simulated = _compile_without_missing_meshes(mujoco.MjSpec.from_string(FR3().get_xml()))
    for model in (mj_arm[0], simulated):
        data = mujoco.MjData(model)
        joints = model.actuator_trnid[:, 0]
        names = [model.joint(j).name for j in joints]
        assert [name[-6:] for name in names] == [f"joint{i}" for i in range(1, 8)], names
        for sign in (1, -1):
            data.ctrl[:] = sign * 1e3 * urdf_model.effortLimit
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.qfrc_actuator[model.jnt_dofadr[joints]], sign * urdf_model.effortLimit, rtol=1e-12)


def test_link7_and_flange_poses_match_urdf(urdf_model, mj_arm):
    model, data = mj_arm
    pdata = urdf_model.createData()
    link0 = urdf_model.getFrameId("fr3_link0")
    b0 = model.body("link0").id
    for q in _configurations(urdf_model.lowerPositionLimit, urdf_model.upperPositionLimit):
        pin.framesForwardKinematics(urdf_model, pdata, q)
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        r0 = data.xmat[b0].reshape(3, 3)
        for body in ("link7", "link8"):
            expected = pdata.oMf[link0].actInv(pdata.oMf[urdf_model.getFrameId(f"fr3_{body}")])
            b = model.body(body).id
            position = r0.T @ (data.xpos[b] - data.xpos[b0])
            rotation = r0.T @ data.xmat[b].reshape(3, 3)
            np.testing.assert_allclose(position, expected.translation, atol=1e-9)
            cos_angle = (np.trace(rotation.T @ expected.rotation) - 1) / 2
            assert np.arccos(np.clip(cos_angle, -1, 1)) < 1e-6, body  # MJCF quaternions carry 6 digits.


def test_armature_is_the_documented_difference(urdf_doc, mj_arm):
    pytest.importorskip("robosuite")
    from robosuite_env.robots.fr3.model import FR3

    dynamics = {j.getAttribute("name"): j.getElementsByTagName("dynamics") for j in urdf_doc.getElementsByTagName("joint")}
    reflected = [float(d.getAttribute("gear_ratio")) ** 2 * float(d.getAttribute("motor_inertia"))
                 for d in (dynamics[f"fr3_joint{i}"][0] for i in range(1, 8))]
    np.testing.assert_allclose(reflected, REFLECTED_ROTOR_INERTIA, atol=1e-4)
    np.testing.assert_array_equal(mj_arm[0].dof_armature, 0)
    robot = FR3()
    model = _compile_without_missing_meshes(mujoco.MjSpec.from_string(robot.get_xml()))
    dofs = [model.joint(name).dofadr[0] for name in robot.joints]
    np.testing.assert_allclose(model.dof_armature[dofs], SIMULATED_ARMATURE)


def test_dynamics_match_urdf_without_flange_placeholder(urdf_model, mj_arm):
    model, data = mj_arm
    body = model.body(PLACEHOLDER_BODY).id
    assert model.body_mass[body] == pytest.approx(0.5)
    np.testing.assert_allclose(model.body_inertia[body], 0.001)
    np.testing.assert_allclose(model.opt.gravity, urdf_model.gravity.linear)
    model.body_mass[body], model.body_inertia[body] = 0, 0
    pdata = urdf_model.createData()
    mass = np.zeros((model.nv, model.nv))
    for q in _configurations(urdf_model.lowerPositionLimit, urdf_model.upperPositionLimit):
        data.qpos[:], data.qvel[:] = q, 0
        mujoco.mj_forward(model, data)
        mujoco.mj_fullM(model, mass, data.qM)
        np.testing.assert_allclose(data.qfrc_bias, pin.computeGeneralizedGravity(urdf_model, pdata, q), atol=1e-4)
        expected = pin.crba(urdf_model, pdata, q)
        expected = np.triu(expected) + np.triu(expected, 1).T
        scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
        np.testing.assert_array_less(np.abs(mass - expected), 1e-5 * scale)  # M[6,6] is ~1.2e-4 kg·m².


def test_simulated_mass_matrix_is_urdf_plus_robosuite_armature(urdf_model):
    """The arm inertia the impedance hook reads with mj_fullM includes robosuite's default armature."""
    pytest.importorskip("robosuite")
    from robosuite_env.robots.fr3.model import FR3

    robot = FR3()
    model = _compile_without_missing_meshes(mujoco.MjSpec.from_string(robot.get_xml()))
    data = mujoco.MjData(model)
    joints = [model.joint(name).id for name in robot.joints]
    dofs = model.jnt_dofadr[joints]
    body = model.body(robot.naming_prefix + PLACEHOLDER_BODY).id
    model.body_mass[body], model.body_inertia[body] = 0, 0
    pdata = urdf_model.createData()
    mass = np.zeros((model.nv, model.nv))
    for q in _configurations(urdf_model.lowerPositionLimit, urdf_model.upperPositionLimit):
        data.qpos[model.jnt_qposadr[joints]], data.qvel[:] = q, 0
        mujoco.mj_forward(model, data)
        mujoco.mj_fullM(model, mass, data.qM)
        rigid = pin.crba(urdf_model, pdata, q)
        rigid = np.triu(rigid) + np.triu(rigid, 1).T
        scale = np.sqrt(np.outer(np.diag(rigid), np.diag(rigid)))
        expected = rigid + np.diag(SIMULATED_ARMATURE)
        np.testing.assert_array_less(np.abs(mass[np.ix_(dofs, dofs)] - expected), 1e-5 * scale)
