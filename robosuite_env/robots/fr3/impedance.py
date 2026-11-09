# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""FR3 cmd_twist torque control; source contracts are in docs/fr3_impedance.md."""
from dataclasses import asdict, dataclass
import math

import numpy as np

from robosuite_env.robots.fr3.controllers import fr3_uses_impedance_hook


@dataclass(frozen=True)
class ImpedanceConfig:
    stiffness_linear: float = 1500.0  # Launch YAML overrides the C++ declaration defaults.
    stiffness_rotational: float = 250.0
    damping_ratio: float = 1.0
    max_pose_offset_lin: float = .05
    max_pose_offset_rot: float = .10
    target_twist_timeout_s: float = .2
    null_space_stiffness: float = .1
    null_space_damping_ratio: float = 1.0
    tau_limit: tuple = (40., 40., 40., 8., 8., 8., 8.)
    null_space_q_left: tuple = (1.2119, .4187, -.1350, -2.4298, 1.3875, 3.9066, .4440)
    null_space_q_right: tuple = (-1.0224, .3637, .3511, -2.5326, -1.0257, 3.6844, 1.1113)
    eef_offset_m: tuple = (0., 0., 0.)  # URDF hand_tcp does not establish hardware F_T_EE.
    eef_rotation_wxyz: tuple = (1., 0., 0., 0.)

    @classmethod
    def from_mapping(cls, values):
        if not isinstance(values, dict):
            raise ValueError("fr3_impedance must be a mapping")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown fr3_impedance fields: {sorted(unknown)}")
        values = dict(values)
        arrays = {"tau_limit": 7, "null_space_q_left": 7, "null_space_q_right": 7,
                  "eef_offset_m": 3, "eef_rotation_wxyz": 4}
        for name, length in arrays.items():
            value = values.get(name, getattr(cls(), name))
            if not isinstance(value, (list, tuple)) or len(value) != length:
                raise ValueError(f"fr3_impedance.{name} requires {length} numbers")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
                raise ValueError(f"invalid fr3_impedance.{name}")
            values[name] = tuple(value)
        config = cls(**values)
        zero_allowed = {"null_space_stiffness", "null_space_damping_ratio", "target_twist_timeout_s"}
        for name, value in asdict(config).items():
            if name in arrays:
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or (value < 0 if name in zero_allowed else value <= 0)):
                raise ValueError(f"invalid fr3_impedance.{name}: {value}")
        if min(config.tau_limit) <= 0 or not np.isclose(np.linalg.norm(config.eef_rotation_wxyz), 1, atol=1e-8):
            raise ValueError("tau_limit must be positive and eef_rotation_wxyz unit length")
        return config


def _rotation(quaternion):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]])


def _wxyz(rotation):
    return rotation.as_quat()[[3, 0, 1, 2]]


def orientation_error(target, actual):
    """Shortest world rotation vector for telemetry only, not the torque law."""
    return (_rotation(target) * _rotation(actual).inv()).as_rotvec()


def spring_rotation_error(target, actual, limit):
    """Preserve the source's quaternion-vector / AngleAxis clamp mismatch."""
    from scipy.spatial.transform import Rotation
    a, t = _rotation(actual), _rotation(target)
    relative = (a.inv() * t).as_quat()
    if relative[3] < 0:
        relative *= -1
    error = a.apply(relative[:3])
    norm = np.linalg.norm(error)
    clipped = norm > limit
    if clipped:
        error *= limit / norm
        target = _wxyz(Rotation.from_rotvec(error) * a)
    return error, np.asarray(target).copy(), bool(clipped)


def inertia_aware_damping(jacobian, mass, stiffness, ratio):
    """Deployed regularized operational inertia / double diagonalization."""
    inertia = np.linalg.inv(jacobian @ np.linalg.solve(mass, jacobian.T) + 1e-3 * np.eye(6))
    eigenvalues, vectors = np.linalg.eigh(inertia)
    root = np.sqrt(np.maximum(eigenvalues, 0))
    square_root = (vectors * root) @ vectors.T
    inverse_root = (vectors / root) @ vectors.T
    omega2, modes = np.linalg.eigh((inverse_root * stiffness) @ inverse_root)
    return (square_root @ (modes * (2 * ratio * np.sqrt(np.maximum(omega2, 0))))
            @ modes.T @ square_root)


def cartesian_torque(jacobian, mass, error, velocity, joint_position, home,
                     coriolis, config, nullspace=True):
    """Return non-gravity torque and wrench, before the controller's limits."""
    stiffness = np.array([config.stiffness_linear] * 3 + [config.stiffness_rotational] * 3)
    damping = inertia_aware_damping(jacobian, mass, stiffness, config.damping_ratio)
    wrench = stiffness * error - damping @ (jacobian @ velocity)
    torque = jacobian.T @ wrench + coriolis
    if nullspace and config.null_space_stiffness > 0:
        # This damped projector intentionally has task-space leakage, like C++.
        projector = np.eye(7) - jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + .05**2 * np.eye(6), jacobian)
        k = config.null_space_stiffness
        torque += projector @ (k * (home - joint_position)
                               - 2 * config.null_space_damping_ratio * math.sqrt(k) * velocity)
    return torque, wrench


def joint_gravity(model, data, dofs):
    """Separate libfranka's implicit gravity effort from limited command torque."""
    import mujoco
    jacobian = np.zeros((3, model.nv))
    gravity = np.empty(len(dofs))
    for i, dof in enumerate(dofs):
        body = model.jnt_bodyid[model.dof_jntid[dof]]
        mujoco.mj_jacSubtreeCom(model, data, jacobian, int(body))
        gravity[i] = -model.body_subtreemass[body] * (model.opt.gravity @ jacobian[:, dof])
    return gravity


class CartesianImpedanceHook:
    """Replace FR3 motor outputs after the ready servo and YUBI pre_action."""

    def __init__(self, state, config):
        import mujoco
        self.state, self.config = state, config
        self.original_pre_action = state.env._pre_action
        self.dt = float(state.m.opt.timestep)
        self.actuators, self.limits, self.reference, self.commands = {}, {}, {}, {}
        self.home, self.ee_to_policy = {}, {}
        self.samples = []
        self.ticks = self.saturated_ticks = self.motor_saturated_ticks = 0
        self.position_clamped_ticks = self.rotation_clamped_ticks = self.stale_ticks = 0
        self.last_time = None
        self.max_position_error_m = self.max_rotation_error_rad = 0.
        self.mass = np.empty((state.m.nv, state.m.nv))
        for key, arm in state.ARM.items():
            indices = []
            for dof in arm["dof"]:
                joint = int(state.m.dof_jntid[dof])
                candidates = np.flatnonzero((state.m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                                           & (state.m.actuator_trnid[:, 0] == joint))
                if len(candidates) != 1:
                    raise ValueError(f"expected exactly one FR3 motor for {key} joint {joint}")
                actuator = int(candidates[0])
                if (state.m.actuator_dyntype[actuator] != mujoco.mjtDyn.mjDYN_NONE
                    or state.m.actuator_gaintype[actuator] != mujoco.mjtGain.mjGAIN_FIXED
                    or state.m.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_NONE
                    or state.m.actuator_gainprm[actuator, 0] != 1
                    or not np.array_equal(state.m.actuator_gear[actuator], [1, 0, 0, 0, 0, 0])):
                    raise ValueError("Cartesian impedance requires unit-gear torque motors")
                if not state.m.actuator_ctrllimited[actuator]:
                    raise ValueError("Cartesian impedance requires explicit motor torque limits")
                indices.append(actuator)
            indices = np.asarray(indices, dtype=int)
            limits = state.m.actuator_ctrlrange[indices].copy()
            for i, actuator in enumerate(indices):
                if state.m.actuator_forcelimited[actuator]:
                    limits[i, 0] = max(limits[i, 0], state.m.actuator_forcerange[actuator, 0])
                    limits[i, 1] = min(limits[i, 1], state.m.actuator_forcerange[actuator, 1])
            self.actuators[key], self.limits[key] = indices, limits
            p, q = self.pose(key)
            self.reference[key] = (p.copy(), q.copy())
            pp, pq = self.policy_pose(key)
            self.ee_to_policy[key] = (_rotation(q).inv().apply(pp - p), _rotation(q).inv() * _rotation(pq))
            self.commands[key] = (np.zeros(3), np.zeros(3), pp.copy(), None)
            self.home[key] = np.asarray(config.null_space_q_left if key == "L" else config.null_space_q_right)

    def pose(self, key):
        body = self.state.ARM[key]["eef"]
        rotation = _rotation(self.state.d.xquat[body])
        return (self.state.d.xpos[body] + rotation.apply(self.config.eef_offset_m),
                _wxyz(rotation * _rotation(self.config.eef_rotation_wxyz)))

    def policy_pose(self, key):
        arm = self.state.ARM[key]
        rotation = self.state.d.xmat[arm["policy"]].reshape(3, 3)
        return (self.state.d.xpos[arm["policy"]] + rotation @ arm["policy_offset"],
                self.state.d.xquat[arm["policy"]].copy())

    def set_twist(self, key, linear_world, angular_world, origin_world=None, stamp=None):
        values = [np.asarray(v, dtype=float).copy() for v in (linear_world, angular_world)]
        origin = None if origin_world is None else np.asarray(origin_world, dtype=float).copy()
        if origin is not None:
            values.append(origin)
        if any(v.shape != (3,) or not np.all(np.isfinite(v)) for v in values):
            raise ValueError("invalid Cartesian impedance twist")
        stamp = float(self.state.d.time if stamp is None else stamp)
        if not math.isfinite(stamp):
            raise ValueError("invalid Cartesian impedance stamp")
        self.commands[key] = (values[0], values[1], origin, stamp)

    def set_policy_actions(self, action, frequency, freeze_left=False):
        # cmd_twist consumes raw quaternion xyz, including its sign and norm.
        action = np.asarray(action, dtype=float)
        if action.ndim != 1 or len(action) < 14 or not np.all(np.isfinite(action)) or frequency <= 0:
            raise ValueError("invalid Cartesian impedance policy action")
        for key, offset in (("L", 0), ("R", 7)):
            p, q = self.policy_pose(key)
            rotation = _rotation(q) * _rotation(self.state.Q_ROBOT_TO_CANON)
            linear = rotation.apply(action[offset:offset+3] * frequency)
            angular = rotation.apply(action[offset+3:offset+6] * (2 * frequency))
            if key == "L" and freeze_left:
                linear[:], angular[:] = 0, 0
            self.set_twist(key, linear, angular, p)

    def _advance_reference(self, key, position, quaternion, now):
        from scipy.spatial.transform import Rotation
        p, q = self.reference[key]
        linear, angular, origin, stamp = self.commands[key]
        stale = (stamp is not None and self.config.target_twist_timeout_s > 0
                 and now - stamp > self.config.target_twist_timeout_s)
        if stale:
            linear, angular = np.zeros(3), np.zeros(3)
        p = p + (linear + (np.cross(angular, p - origin) if origin is not None else 0)) * self.dt
        if np.linalg.norm(angular) > 1e-9:
            q = _wxyz(Rotation.from_rotvec(angular * self.dt) * _rotation(q))
        offset = p - position
        norm = np.linalg.norm(offset)
        pos_clamped = norm > self.config.max_pose_offset_lin
        if pos_clamped:
            p = position + offset * (self.config.max_pose_offset_lin / norm)
        error_r, q, rot_clamped = spring_rotation_error(q, quaternion, self.config.max_pose_offset_rot)
        self.reference[key] = (p, q)
        return np.r_[p - position, error_r], bool(pos_clamped), rot_clamped, bool(stale)

    def __call__(self, action, policy_step=False):
        import mujoco
        self.original_pre_action(action, policy_step)
        state = self.state
        now = float(state.d.time)
        if self.last_time is not None and not math.isclose(now - self.last_time, self.dt, abs_tol=1e-9):
            raise RuntimeError("Cartesian impedance did not run once per physics step")
        self.last_time = now
        # MuJoCo 3.8 uses mj_fullM(model, data, dst); allow older bindings too.
        if not hasattr(self, "_full_mass"):
            try:
                mujoco.mj_fullM(state.m, state.d, self.mass)
                self._full_mass = lambda: mujoco.mj_fullM(state.m, state.d, self.mass)
            except TypeError:
                self._full_mass = lambda: mujoco.mj_fullM(state.m, self.mass, state.d.qM)
        self._full_mass()
        sample = {"sim_time_s": now, "arms": {}}
        for key, arm in state.ARM.items():
            position, quaternion = self.pose(key)
            error, pos_clamp, rot_clamp, stale = self._advance_reference(key, position, quaternion, now)
            jacp, jacr = np.zeros((3, state.m.nv)), np.zeros((3, state.m.nv))
            mujoco.mj_jac(state.m, state.d, jacp, jacr, position, arm["eef"])
            dofs = arm["dof"]
            jacobian = np.vstack([jacp[:, dofs], jacr[:, dofs]])
            gravity = joint_gravity(state.m, state.d, dofs)
            raw, wrench = cartesian_torque(jacobian, self.mass[np.ix_(dofs, dofs)], error,
                state.d.qvel[dofs], state.d.qpos[arm["qi"]], self.home[key],
                state.d.qfrc_bias[dofs] - gravity, self.config, nullspace=state.A.fr3_nullspace == "home")
            if not np.all(np.isfinite(raw)):
                raise RuntimeError(f"non-finite Cartesian impedance torque for {key}")
            limits = self.limits[key]
            commanded = np.clip(raw, -np.asarray(self.config.tau_limit), self.config.tau_limit)
            total = commanded + gravity
            torque = np.clip(total, limits[:, 0], limits[:, 1])
            state.d.ctrl[self.actuators[key]] = torque
            saturated, motor_saturated = bool(np.any(raw != commanded)), bool(np.any(total != torque))
            self.saturated_ticks += int(saturated)
            self.motor_saturated_ticks += int(motor_saturated)
            self.position_clamped_ticks += int(pos_clamp)
            self.rotation_clamped_ticks += int(rot_clamp)
            self.stale_ticks += int(stale)
            ref_p, ref_q = self.reference[key]
            geometric_rotation_error = orientation_error(ref_q, quaternion)
            self.max_position_error_m = max(self.max_position_error_m, float(np.linalg.norm(error[:3])))
            self.max_rotation_error_rad = max(self.max_rotation_error_rad, float(np.linalg.norm(geometric_rotation_error)))
            point, rotation = self.ee_to_policy[key]
            if hasattr(state, "TGT"):
                state.TGT[key] = [ref_p + _rotation(ref_q).apply(point), _wxyz(_rotation(ref_q) * rotation)]
            sample["arms"][key] = {
                "reference_position": ref_p.tolist(), "reference_quaternion_wxyz": ref_q.tolist(),
                "position_error_m": error[:3].tolist(), "rotation_error_rad": geometric_rotation_error.tolist(),
                "spring_rotation_error_quaternion_vector": error[3:].tolist(),
                "q": state.d.qpos[arm["qi"]].tolist(), "dq": state.d.qvel[dofs].tolist(),
                "tcp_position": position.tolist(), "tcp_quaternion_wxyz": quaternion.tolist(),
                "command_linear_world_m_s": self.commands[key][0].tolist(),
                "command_angular_world_rad_s": self.commands[key][1].tolist(),
                "command_origin_world_m": (None if self.commands[key][2] is None else self.commands[key][2].tolist()),
                "torque_raw_nm": raw.tolist(), "command_torque_without_gravity_nm": commanded.tolist(),
                "gravity_torque_nm": gravity.tolist(), "torque_nm": torque.tolist(),
                "wrench": wrench.tolist(), "saturated": saturated, "motor_saturated": motor_saturated,
                "position_clamped": pos_clamp, "rotation_clamped": rot_clamp, "command_stale": stale,
            }
        self.ticks += 1
        self.samples.append(sample)

    def drain_samples(self):
        samples, self.samples = self.samples, []
        return samples

    def summary(self):
        config = asdict(self.config)
        overrides = [key for key, value in config.items()
                     if value != getattr(ImpedanceConfig(), key)]
        return {"config": config, "gain_provenance": "controller_launch_yaml_with_recorded_overrides",
                "source_default_overrides": overrides,
                "nullspace_enabled": self.state.A.fr3_nullspace == "home",
                "command_contract": "deploy_cmd_twist_per_step_delta",
                "control_frame_assumption": "URDF_hand_tcp_plus_configured_offset_not_measured_F_T_EE",
                "physics_dt_s": self.dt, "ticks": self.ticks,
                "saturated_arm_ticks": self.saturated_ticks, "motor_saturated_arm_ticks": self.motor_saturated_ticks,
                "position_clamped_arm_ticks": self.position_clamped_ticks,
                "rotation_clamped_arm_ticks": self.rotation_clamped_ticks, "stale_arm_ticks": self.stale_ticks,
                "max_reference_position_error_mm": 1000 * self.max_position_error_m,
                "max_reference_orientation_error_deg": math.degrees(self.max_rotation_error_rad),
                "torque_limits_nm": {k: v.tolist() for k, v in self.limits.items()},
                "ready_controller": "joint_ik", "gravity_compensation": "added_after_non_gravity_command_limit",
                "torque_rate_limit": "none_in_pinned_controller_hardware_filter_not_emulated"}


def install_fr3_impedance(state):
    previous = getattr(state, "FR3_IMPEDANCE", None)
    if previous is not None:
        state.env._pre_action = previous.original_pre_action
    state.FR3_IMPEDANCE = None
    if (state.EMBODIMENT == "fr3" and state.A.physics == "dynamic"
        and fr3_uses_impedance_hook(state.A.fr3_controller) and not state.A.state_log_replay):
        state.FR3_IMPEDANCE = CartesianImpedanceHook(state, state.FR3_IMPEDANCE_CONFIG)
        state.env._pre_action = state.FR3_IMPEDANCE
