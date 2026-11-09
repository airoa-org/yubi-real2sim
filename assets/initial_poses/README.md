# Initial arm poses

Every start pose a profile can select lives once, at

```
assets/initial_poses/<robot>/<rig>/<name>.json
```

and is selected by that same path without the extension:

```yaml
simulation:
  parameters:
    initial_arm_joints: fr3/flat/grip07
```

`robot`/`rig` is `fr3/duo` (FR3 Duo, profile `robot: fr3_duo`) or `fr3/flat`
(Franka Flat, `robot: fr3`).

| Name | Used by |
| --- | --- |
| `fr3/duo/init_high` | the Duo default layouts; no release profile |
| `fr3/duo/home`, `fr3/duo/home_20260906` | the Duo base placement and the earlier Duo default layouts |
| `fr3/flat/grip07` | both release profiles (FR3 Flat cup) |
| `fr3/flat/registered_cup_calibrated` | the CI fixture `config/release/fixture_v2.yaml` |

## Signature

Each file starts with a `signature` block: the robot and rig of its name, the
scene it was taught or derived for, and every scene it is used with:

```json
"signature": {"robot": "fr3", "rig": "flat", "scene": "franka_flat", "scenes": ["franka_flat"]}
```

The launcher refuses a profile whose `robot:` belongs to another rig than the
signature's. The pose loaders read the joint values exactly as before; a pose
file without a `signature` block is still accepted.

## Older names and paths

The older names (`duo_init_high`, `grip07`, `registered_cup_calibrated`,
and the Flat poses' `fr3/franka_flat/<name>` spelling) still resolve to the
same files. Duo poses are selected only as `fr3/duo/<name>`.
`real2sim-run --list-initial-arm-joints` prints both lists. A profile or
worker command line that spells an old file path is redirected through
`legacy_paths.json`.
