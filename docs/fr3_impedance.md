# FR3 Cartesian impedance

- Set `simulation.fr3_controller: cartesian_impedance`, `physics: dynamic` and `policy_action_clamp: false`.
- Select `fr3` or `fr3_duo`. Physics defaults to 1 ms; `mujoco_timestep` can override it up to 2 ms.
- Ready positioning uses joint IK, then Cartesian torque control starts from the measured pose.
- MuJoCo integrates arm state under motor torque and contact reaction forces.
- Policy deltas become `cmd_twist`: translation × action Hz and quaternion xyz × 2 × action Hz.
- The equilibrium persists across chunks. Commands older than 0.2 s stop advancing it.

## Reference

- Effective parameters are recorded under `arm_controller_detail.cartesian_impedance` in each rollout summary.
- The port's revision is `code.commit`, with `code.dirty`, in the `run_manifest.json` that the launcher writes. Direct diagnostic runs and the wall probe write no run manifest, so they record no revision. Summaries from earlier versions also carry `deploy_commit` and `controller_source_commit`: revisions of the non-public reference sources, which no reader uses.
- `simulation.parameters.fr3_impedance` accepts the source-named gains, offset bounds, timeout, nullspace settings and seven `tau_limit` values.
- Defaults: linear stiffness 1500 N/m, rotational stiffness 250, damping ratio 1, linear offset 0.05 m, rotation error bound 0.10.
- Nullspace stiffness is 0.1; per-arm references follow the bimanual reference configuration. `fr3_nullspace: off` disables that term.
- Non-gravity torque limits are `[40, 40, 40, 8, 8, 8, 8]` Nm. Gravity is added after these limits, matching libfranka.
- Damping uses the regularized Cartesian inertia and modal double diagonalization. The nullspace projector uses the source's damped pseudoinverse.
- The orientation spring uses quaternion xyz, so its error magnitude is `sin(angle / 2)`. The source reconstructs a clamped reference with that magnitude as an angle.
- The port preserves that clamp behavior. Geometric orientation error and spring error are logged separately.

## Contact probe

- FR3 simulation only. The probe attaches a 12 mm sphere at `hand_tcp` and isolates its wall collision.
- It presses at 20 mm/s for 4 s, lets the command expire for 3 s, withdraws at 20 mm/s for 4 s, then settles for 1 s.
- The controller limits equilibrium lead and torque. It has no contact-force-triggered retreat, so an expired twist can leave static pressure.
- Lead and torque bounds do not impose a universal contact-normal force cap; impact speed and posture affect contact force.
- Withdrawal is a commanded movement. A pass requires bounded contact force, at most 5 cm lead, and contact release after withdrawal.

```bash
python -m robosuite_env.tools.probe_fr3_impedance_wall --output outputs/impedance/wall
```

## Evaluation notes

- The controlled frame assumes URDF `hand_tcp`, with optional `eef_offset_m` and `eef_rotation_wxyz`. Hardware `F_T_EE` must be measured separately.
- Hardware filtering, torque-rate limiting, network timing and identified loads are not reproduced.
- Record installed package versions with results; fixture runs can differ from the production lockfile.
- `scripts/analyze_fr3_impedance.py` checks frame and physics-tick counts, finite state, torque limits, gravity decomposition and equilibrium bounds.

## Arm model

- `tests/unit/robots/test_fr3_model_contract.py` checks `robosuite_env/assets/fr3_robot.xml` against the bootstrap-pinned franka_description `fr3.urdf.xacro` with `hand:=false`.
- It compares joint order, position limits, the joint torque of each saturated actuator against the effort limit in `fr3_robot.xml` and in the robosuite-loaded arm, damping, friction, the link0 → link7 and link0 → link8 (flange) poses, gravity torque and the mass matrix, both without armature and as `mj_fullM` returns it for the robosuite-loaded arm, which the impedance hook reads.
- `pin` and `xacro` come only with the `fr3-model-contract` dependency group, which `scripts/bootstrap.sh` does not install: `uv sync --frozen --no-build-isolation --group fr3-model-contract`. Without them, or without `third_party/franka_description`, which `scripts/bootstrap.sh` checks out, the test skips with that reason. The `locked-runtime` CI job syncs the group after the CPU contracts, checks out the pinned franka_description and fails if a test skips or xfails.
- Without the gitignored collision meshes, the test drops their geoms and requires each affected body to declare its inertial.
- The MJCF has no joint velocity limit, so `velocityLimit` is not compared. Gripper and hand frames are out of scope.
- The model keeps two known differences:
  - `right_hand` carries robosuite's flange placeholder: 0.5 kg and inertia 0.001 kg·m². It shifts gravity torque by up to 2.46 Nm and the mass matrix by up to 0.27, so the test zeroes it.
  - Armature: the MJCF declares none, so robosuite's `RobotModel` sets its default 5/i kg·m² on joint i. The simulated arm therefore carries 5, 2.5, 1.667, 1.25, 1.0, 0.833 and 0.714 kg·m² (joints 1–7).
  - franka_description gives `motor_inertia` and `gear_ratio` per joint (`robots/fr3/dynamics.yaml`), so its reflected rotor inertia is 0.6057 (joints 1–2), 0.4625 (3–4) and 0.2055 kg·m² (5–7). pinocchio does not read these attributes. The rigid-body comparison excludes both values; the robosuite-loaded comparison adds the 5/i armature to the URDF mass matrix. The simulated armature is 2.7–8.3× the reflected rotor inertia. Rollouts keep it; the test records the difference rather than changing it.
  - Gripper and hand inertia are not in the compared matrices.
