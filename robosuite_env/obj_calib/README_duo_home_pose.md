# FR3 Duo base pose and home joints

Two files describe where the Duo arms are and how they start. The mount pose lives here;
the start pose lives with the other named poses under `assets/initial_poses/`.

| File | What it holds |
| --- | --- |
| `fr3_base_duo.json` | Where the two arm bases sit on the shared mount |
| `assets/initial_poses/fr3/duo/home.json` | The joint targets both arms hold at the start of a rollout (registry name `fr3/duo/home`) |

## The mount pose

`fr3_base_duo.json` comes from the Duo rig's as-built TF (rmse 3.2 mm) by way
of the 3DGS capture's `gs_to_mujoco_transform.json`, `frames.mujoco_world`. Unlike every
earlier scene it sets `rpy_deg` rather than `yaw_deg`: the Duo mount rolls both arms by
±51.17° and pitches them −10.03°, and a yaw alone cannot express that.

Two frame conventions meet in this file, and both are already applied to the numbers in it:

- The registration world puts `world -> mount_link` at identity; the runtime world puts the
  table top at z = 0. The scene contract bakes `--calibration-world-offset 0 0 -0.1377`, so
  `runtime_z = registration_z + 0.1377`.
- The whole scene is rotated −90° about z, from `frames.world_to_table_origin`.

The angles are quoted the way URDF composes them, `Rz(y) @ Ry(p) @ Rx(r)`. `place_task_robots`
re-solves them for robosuite's opposite order before calling `set_base_ori` — see
`_robosuite_euler` in `robosuite_env/tasks/scene.py`. Do not hand these three numbers to
`set_base_ori` yourself.

## The home joints

`assets/initial_poses/fr3/duo/home.json` is a schema-v2 file registered under the
name `fr3/duo/home`, so a profile selects it the way every other checked-in pose is selected:

    simulation:
      parameters:
        initial_arm_joints: fr3/duo/home

`real2sim-run --list-initial-arm-joints` prints it. It replaces only the ready joint goals.
The two arms hold **different** joint values, because the pose was taught on the rig rather
than configured.

The home pose is the pose taught on the physical arms, not the symmetric textbook pose
`[0, -45, 0, -135, 0, 90, 45]` degrees; the policy was not trained from the textbook pose.

## Verified

Rendered in the Duo scene: both arms hold the commanded joints to within 0.11°.

The large `res_pos_mm` a rollout reports against a home pose is the distance to the
cup-relative ready TCP, not a tracking error — a home pose is not a grasp pose.
