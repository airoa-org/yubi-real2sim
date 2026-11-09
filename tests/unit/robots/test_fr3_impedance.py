# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Physical and source-contract oracles for the deployed FR3 control port."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
pytest.importorskip("scipy")
from scipy.spatial.transform import Rotation

from robosuite_env.robots.fr3.impedance import (
    CartesianImpedanceHook, ImpedanceConfig, cartesian_torque, orientation_error,
    inertia_aware_damping, spring_rotation_error, joint_gravity,
)


def test_mass_aware_damping_satisfies_modal_critical_damping():
    rng = np.random.default_rng(321)
    j = rng.normal(size=(6, 7))
    a = rng.normal(size=(7, 7))
    mass = a @ a.T + np.eye(7)
    k = np.array([1500.] * 3 + [250.] * 3)
    inverse_inertia = j @ np.linalg.solve(mass, j.T) + 1e-3 * np.eye(6)
    d = inertia_aware_damping(j, mass, k, 1.3)
    np.testing.assert_allclose(d, d.T, atol=1e-10)
    assert min(np.linalg.eigvalsh(d)) > 0
    np.testing.assert_allclose(d @ inverse_inertia @ d, 4 * 1.3**2 * np.diag(k), atol=1e-7)
    # Include rank-deficient configurations, where the deployed regularizer matters.
    assert np.all(np.isfinite(inertia_aware_damping(np.zeros((6, 7)), mass, k, 1)))


def test_wrench_virtual_work_damping_and_deployed_damped_nullspace():
    rng = np.random.default_rng(322)
    j, v = rng.normal(size=(6, 7)), rng.normal(size=7)
    cfg = ImpedanceConfig()
    q, home, bias = np.ones(7), np.zeros(7), np.zeros(7)
    args = (j, np.eye(7), np.array([.01, -.02, .03, .1, .2, -.1]), v, q, home, bias, cfg)
    task, wrench = cartesian_torque(*args, nullspace=False)
    combined, _ = cartesian_torque(*args)
    assert task @ v == pytest.approx(wrench @ (j @ v))
    # Independent SVD oracle; source uses a damped, not orthogonal, projector.
    _, s, vh = np.linalg.svd(j, full_matrices=True)
    n = vh.T @ np.diag(np.r_[.05**2 / (s*s + .05**2), 1.]) @ vh
    np.testing.assert_allclose(combined-task, n @ (-.1*q - 2*np.sqrt(.1)*v), atol=1e-11)
    damping, _ = cartesian_torque(j, np.eye(7), np.zeros(6), v, q, home, bias, cfg, False)
    assert damping @ v < 0


def test_source_rotation_is_half_angle_and_preserves_clamp_discontinuity():
    actual = Rotation.from_euler('xyz', [.2, -.4, .7])
    vector = np.array([.1, -.2, .3])
    angle = np.linalg.norm(vector)
    target = Rotation.from_rotvec(vector) * actual
    a, b = (r.as_quat()[[3, 0, 1, 2]] for r in (target, actual))
    for sa in (-1, 1):
        for sb in (-1, 1):
            np.testing.assert_allclose(orientation_error(sa*a, sb*b), vector, atol=1e-12)
            error, ref, clipped = spring_rotation_error(sa*a, sb*b, 1.)
            np.testing.assert_allclose(error, vector / angle * np.sin(angle/2), atol=1e-12)
            assert not clipped
    error, ref, clipped = spring_rotation_error(a, b, .1)
    assert clipped
    assert np.linalg.norm(error) == pytest.approx(.1)
    assert np.linalg.norm(orientation_error(ref, b)) == pytest.approx(.1)
    # On the following tick, the same stored reference produces sin(.1/2).
    next_error, _, clipped = spring_rotation_error(ref, b, .1)
    assert not clipped
    assert np.linalg.norm(next_error) == pytest.approx(np.sin(.05))


def test_base_to_world_rotation_preserves_commanded_joint_torque():
    # Duo's differently mounted arms must yield the same motor effort when
    # expressing the same Cartesian state in world rather than base axes.
    rng = np.random.default_rng(421)
    jacobian, velocity, error = rng.normal(size=(6, 7)), rng.normal(size=7), rng.normal(size=6)
    a = rng.normal(size=(7, 7))
    mass = a @ a.T + np.eye(7)
    rotation = Rotation.from_euler('xyz', [.4, -.8, 1.2]).as_matrix()
    transform = np.kron(np.eye(2), rotation)
    rest = (velocity, np.ones(7), np.zeros(7), np.ones(7), ImpedanceConfig())
    base, _ = cartesian_torque(jacobian, mass, error, *rest)
    world, _ = cartesian_torque(transform @ jacobian, mass, transform @ error, *rest)
    np.testing.assert_allclose(world, base, atol=1e-8)


@pytest.mark.parametrize('values', [
    {'stiffness_linear': float('nan')}, {'damping_ratio': -1},
    {'tau_limit': [1]*6}, {'tau_limit': [0]*7}, {'max_pose_offset_rot': True},
    {'eef_rotation_wxyz': [1, 1, 0, 0]}, {'null_space_q_left': [False]*7}, {'typo': 1}, [],
])
def test_bad_parameters_fail_before_simulation(values):
    with pytest.raises(ValueError):
        ImpedanceConfig.from_mapping(values)


def test_cmd_twist_controller_without_torque_hook_fails_in_args_run(monkeypatch):
    from real2sim import fr3_controllers as controllers
    from robosuite_env.rollout.setup import args as setup_args
    monkeypatch.setattr(controllers, 'FR3_IMPEDANCE_HOOK_CONTROLLERS', frozenset())
    monkeypatch.setattr('sys.argv', ['worker', '--robot', 'fr3', '--physics', 'dynamic',
                                     '--fr3-controller', 'cartesian_impedance',
                                     '--policy-batch-server', '127.0.0.1:1',
                                     '--no-policy-action-clamp', '--fr3-impedance', '{}'])
    state = SimpleNamespace()
    with pytest.raises(SystemExit, match='has no torque hook'):
        setup_args.run(state)
    assert not hasattr(state, 'FR3_IMPEDANCE_CONFIG')


def _fr3_state(mujoco):
    assets = Path(__file__).resolve().parents[3] / 'robosuite_env/assets'
    if not (assets / 'fr3_collision/link0.stl').exists():
        pytest.skip('FR3 collision payload not installed')
    model = mujoco.MjModel.from_xml_path(str(assets / 'fr3_robot.xml'))
    model.opt.timestep = .001
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)
    data.qpos[:] = [0, -.785398, 0, -2.356194, 0, 1.570796, .785398]
    mujoco.mj_forward(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_hand')
    arm = {'qi': np.arange(7), 'dof': np.arange(7), 'eef': body, 'policy': body,
           'policy_offset': np.array([.02, -.015, .08])}
    def upstream(action, policy_step):
        data.ctrl[:] = 80  # Must be replaced, not added to impedance torque.
    return SimpleNamespace(m=model, d=data, ARM={'R': arm},
        A=SimpleNamespace(fr3_nullspace='off'), Q_ROBOT_TO_CANON=np.array([1.,0,0,0]),
        env=SimpleNamespace(_pre_action=upstream))


def test_gravity_split_matches_inverse_dynamics_at_zero_velocity(mujoco):
    state = _fr3_state(mujoco)
    rng = np.random.default_rng(82)
    rest = mujoco.MjData(state.m)
    for _ in range(15):
        state.d.qpos[:] += rng.normal(scale=.02, size=7)
        state.d.qvel[:] = rng.normal(size=7)
        mujoco.mj_forward(state.m, state.d)
        rest.qpos[:] = state.d.qpos
        mujoco.mj_forward(state.m, rest)
        np.testing.assert_allclose(joint_gravity(state.m, state.d, np.arange(7)), rest.qfrc_bias, atol=1e-10)


def test_twist_integrator_preserves_lag_lever_arm_and_timeout(mujoco):
    state = _fr3_state(mujoco)
    hook = CartesianImpedanceHook(state, ImpedanceConfig())
    p, q = hook.pose('R')
    origin = p - np.array([.02, 0, 0])
    hook.set_twist('R', [.01, 0, 0], [0, 0, .5], origin)
    # Simulate ten controller ticks with the measured pose held still. The
    # equilibrium must accumulate all ten increments, not reanchor each tick.
    expected = p.copy()
    for i in range(10):
        expected += (np.array([.01, 0, 0]) + np.cross([0,0,.5], expected-origin)) * .001
        hook._advance_reference('R', p, q, i*.001)
    np.testing.assert_allclose(hook.reference['R'][0], expected, atol=1e-12)
    assert np.linalg.norm(orientation_error(hook.reference['R'][1], q)) == pytest.approx(.005)
    before = tuple(x.copy() for x in hook.reference['R'])
    *_, stale = hook._advance_reference('R', p, q, .201)
    assert stale
    for a, b in zip(before, hook.reference['R']):
        np.testing.assert_allclose(a,b)


def test_policy_mapping_uses_live_frame_and_raw_quaternion_vector(mujoco):
    state = _fr3_state(mujoco)
    state.ARM['L'] = state.ARM['R']  # Only exercise conversion; never simulate shared motors.
    state.Q_ROBOT_TO_CANON = Rotation.from_euler('x', .7).as_quat()[[3,0,1,2]]
    hook = CartesianImpedanceHook(state, ImpedanceConfig())
    action = np.array([.01,.02,-.03,.1,-.2,.3,-.8]*2+[.7,.7])
    hook.set_policy_actions(action, 10)
    p,q = hook.policy_pose('R')
    rotation = Rotation.from_quat(q[[1,2,3,0]]) * Rotation.from_euler('x', .7)
    linear,angular,origin,_ = hook.commands['R']
    np.testing.assert_allclose(linear, rotation.apply([.1,.2,-.3]))
    np.testing.assert_allclose(angular, rotation.apply([2,-4,6]))
    np.testing.assert_allclose(origin,p)


@pytest.mark.parametrize('eef_offset', [(0., 0., 0.), (.02, -.015, .08)])
def test_real_fr3_dynamics_tracks_small_step_without_state_pinning(mujoco, eef_offset):
    state = _fr3_state(mujoco)
    rotation = tuple(Rotation.from_euler('xyz', [.2, -.1, .3]).as_quat()[[3, 0, 1, 2]])
    hook = CartesianImpedanceHook(state, ImpedanceConfig(eef_offset_m=eef_offset, eef_rotation_wxyz=rotation))
    position, quaternion = hook.pose('R')
    target = position + np.array([.01, -.01, .01])
    # Persistent equilibrium after a finite movement (then zero twist).
    hook.reference['R'] = (target, quaternion)
    for _ in range(2500):
        q_before, v_before = state.d.qpos.copy(), state.d.qvel.copy()
        hook(None)
        np.testing.assert_array_equal(state.d.qpos, q_before)
        np.testing.assert_array_equal(state.d.qvel, v_before)
        assert np.all(np.abs(state.d.ctrl) <= np.array([87]*4+[12]*3))
        command = np.asarray(hook.samples[-1]['arms']['R']['command_torque_without_gravity_nm'])
        assert np.all(np.abs(command) <= hook.config.tau_limit)
        mujoco.mj_step(state.m, state.d)
        mujoco.mj_forward(state.m, state.d)
    assert np.linalg.norm(hook.pose('R')[0] - target) < .001
    assert np.linalg.norm(orientation_error(quaternion, hook.pose('R')[1])) < .01
    assert hook.ticks == 2500
    assert len(hook.drain_samples()) == 2500
    assert hook.drain_samples() == []


@pytest.mark.parametrize('eef_offset', [(0., 0., 0.), (.02, -.015, .08)])
def test_external_force_yields_physical_compliance(mujoco, eef_offset):
    state = _fr3_state(mujoco)
    hook = CartesianImpedanceHook(state, ImpedanceConfig(eef_offset_m=eef_offset))
    position, _ = hook.pose('R')
    body = state.ARM['R']['eef']
    force = np.array([3., 0., 0.])
    for _ in range(3000):
        state.d.qfrc_applied[:] = 0
        mujoco.mj_applyFT(state.m, state.d, force, np.zeros(3), hook.pose('R')[0], body,
                         state.d.qfrc_applied)
        hook(None)
        mujoco.mj_step(state.m, state.d)
        mujoco.mj_forward(state.m, state.d)
    displacement = hook.pose('R')[0] - position
    assert displacement[0] == pytest.approx(3 / hook.config.stiffness_linear, abs=.0005)
    assert np.linalg.norm(displacement[1:]) < .0005


def test_controller_limit_excludes_gravity(mujoco):
    state = _fr3_state(mujoco)
    hook = CartesianImpedanceHook(state, ImpedanceConfig(tau_limit=(.001,)*7))
    p,q = hook.pose('R')
    hook.reference['R']=(p + [.01,0,0], q)
    hook(None)
    rec = hook.samples[-1]['arms']['R']
    assert rec['saturated']
    np.testing.assert_allclose(np.array(rec['command_torque_without_gravity_nm'])+rec['gravity_torque_nm'], state.d.ctrl)
    assert np.max(np.abs(state.d.ctrl)) > 1


def test_hook_rejects_nonfinite_twist_and_duplicate_physics_tick(mujoco):
    state = _fr3_state(mujoco)
    hook = CartesianImpedanceHook(state, ImpedanceConfig())
    with pytest.raises(ValueError, match='invalid.*twist'):
        hook.set_twist('R', [np.nan,0,0], [0,0,0], [0,0,0])
    hook(None)
    with pytest.raises(RuntimeError, match='once per physics'):
        hook(None)


def test_summary_records_no_non_public_source_revisions(mujoco):
    # #85 R2: the summary once carried revisions of non-public reference
    # sources. The port's revision is the run manifest's code commit.
    state = _fr3_state(mujoco)
    hook = CartesianImpedanceHook(state, ImpedanceConfig())
    hook(None)
    summary = json.loads(json.dumps(hook.summary()))
    assert not {"deploy_commit", "controller_source_commit"} & set(summary)
    assert summary["ticks"] == 1
    assert summary["physics_dt_s"] == pytest.approx(.001)
    assert summary["source_default_overrides"] == []


def test_summary_body_omits_non_public_source_revisions_without_mujoco():
    # Runs the real summary() body on the attributes it reads, so the CPU
    # contracts cover #85 R2 without the heavy mujoco fixture.
    hook = SimpleNamespace(
        config=ImpedanceConfig(), state=SimpleNamespace(A=SimpleNamespace(fr3_nullspace='home')),
        dt=.001, ticks=3, saturated_ticks=0, motor_saturated_ticks=0, position_clamped_ticks=0,
        rotation_clamped_ticks=0, stale_ticks=0, max_position_error_m=0., max_rotation_error_rad=0.,
        limits={'R': np.zeros((7, 2))})
    summary = json.loads(json.dumps(CartesianImpedanceHook.summary(hook)))
    assert not {'deploy_commit', 'controller_source_commit'} & set(summary)
    assert summary['ticks'] == 3
    assert summary['nullspace_enabled']
    assert summary['source_default_overrides'] == []
    assert summary['torque_limits_nm'] == {'R': [[0., 0.]]*7}
