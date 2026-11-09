# Simulation operations

Runtime validation is grouped under `robosuite_env/contracts/`: `assets.py`
owns repository asset bundles, `scene.py` owns the 3DGS and table-collision
bundle, and `worker.py` owns exact official-worker arguments. The former
top-level contract modules remain thin CLI and import compatibility shims;
new code must use the grouped paths.

The official workflow has one public entry point:

```bash
real2sim-run --config config/release/<robot>/<rig>/<room>_<task>_<checkpoint>.yaml --check-runtime
real2sim-run --config config/release/<robot>/<rig>/<room>_<task>_<checkpoint>.yaml
```

Invoke it from a GPU-capable interactive shell after the host environment has
provided the required resources. Real2Sim starts only local child processes; it
does not submit, allocate, cancel, or query workload-scheduler jobs.

All production inputs are folders in two gated Hugging Face repositories
that `real2sim-download-release --profile <profile>` installs at the revisions
and tree pins in `artifacts/release_downloads.json`; users log in with
`.venv/bin/hf auth login` and accept both repositories' terms once. A profile's inputs
are its checkpoint, its schema-v2 scene, and the cup/plate and FR3 collision
meshes. The PaliGemma tokenizer
is intentionally excluded under the Gemma terms; OpenPI fetches it at runtime
from `gs://big_vision/paligemma_tokenizer.model`.

The local policy-batch runner and `robosuite_env/sim.py` are internal workers. A
schema-v2 run writes `run_manifest.json` before starting them; workers reject
an absent manifest or asset/scene bytes that changed after preflight. Direct
`sim.py` runs remain available for development, but their summaries are marked
`execution_class: diagnostic` and cannot complete an official evaluation.
`sim.py` contains no simulation implementation; it dispatches to the
responsibility-separated `robosuite_env.rollout` package. Robot code is under
`robots/`, task code under `tasks/`, policy I/O under `policy/`, and rendering
under `rendering/`. This ownership split is part of the runtime contract and is
checked in CI.

## Three operator-facing contracts

Routine operation changes only three contracts:

1. The rollout YAML selects policy, task, controllers, grasp rules, OBS
   processing, rendering, scene paths, and concurrency.
2. `robosuite_env/assets/manifest.json` groups FR3, left/right YUBI, cup, and
   plate assets and declares their visual/collision roles.
3. Each scene's exact `background.json` binds the 3DGS point cloud, world-to-GS
   transform, and world-aligned table collision.

The launcher automatically pins the remaining reproducibility inputs: code
commits and dependency locks, checkpoint and policy-anchor artifacts,
deterministic seed/init state, GPU assignment and render locks, success/score
definitions, and output-completion checks. These are recorded in
`run_manifest.json` and per-rollout summaries rather than maintained through
additional ad-hoc launch scripts.

The table has no MuJoCo visual mesh. Its appearance is already in the 3DGS;
the scene contract contains only its invisible collision set. `desk: scene` is
therefore mandatory in schema v2. A missing, empty, extra, or modified table
collision set stops before MuJoCo construction.

The repository asset manifest is a logical bundle over the existing asset
directories. No bulk move or duplicate copy is needed. Every MJCF file
reference must exist below `robosuite_env/assets/`; visual and collision mesh
paths must match their declared role. The cup collision set is exactly
`cup_000.obj` through `cup_037.obj`; other cup collision sets are rejected.
Validate the installed repository assets directly with:

```bash
python -m robosuite_env.contracts.assets
```

## Scene contract

A schema-v2 `background.json` has this minimal shape:

```json
{
  "schema_version": 2,
  "id": "unique-scene-content-id",
  "name": "experiment_scene_tag",
  "ply_path": "point_cloud.ply",
  "ply_sha256": "<64 lowercase hex characters>",
  "w2gs": "gs_w2gs.npy",
  "w2gs_sha256": "<64 lowercase hex characters>",
  "calibration_table_z_m": -0.0141,
  "calibration_table_z_method": "ply-plane-fit-v1",
  "table": {
    "appearance": "embedded_in_3dgs",
    "collision": {
      "directory": "table_collision",
      "glob": "*.obj",
      "count": 25,
      "tree_sha256": "<64 lowercase hex characters>",
      "frame": "mujoco_world"
    }
  }
}
```

`name` becomes the rollout directory tag, so it must be unique within a
comparison. `id` must uniquely identify the actual scene content and is also
rejected when duplicated. All scenes in one comparison must use the same table
collision digest so background is the only changed factor.

### Table height check

The hashes prove which bytes a scene contains; `calibration_table_z_m`
proves that those bytes agree on where the table is. It is the z coordinate,
in MuJoCo world metres, of the 3DGS table-top plane once the point cloud is
mapped through `inv(mat_w2gs)` (or `inv(w2gs)`). Schema v2 pins the runtime
reference plane `table_z` to `0.0`, so a correctly baked scene declares a value
close to zero (the public Flat Reading Room declares about `-0.035`). The check does not constrain the bound table collision: it is
compiled as given in `mujoco_world`, its top need not be at z=0 (the Duo slab's
is +34 mm), and objects are seated on it at reset (see
[Where the objects start](configuration.md#where-the-objects-start)).
`calibration_table_z_method` names the measurement;
`ply-plane-fit-v1` is the only supported value and is assumed when the field
is omitted.

`validate_scene_contract()` requires `calibration_table_z_m` to be a finite
number and rejects the scene when:

- `|calibration_table_z_m| > 0.05 m` — the declared table-top is not the
  z=0 runtime convention;
- `|calibration_table_z_m - measured| > 0.05 m` — the value does not match
  what the verified point cloud and transform actually produce, i.e. the PLY
  or the matrix was swapped for one calibrated to another convention.

`ply-plane-fit-v1` (`robosuite_env.contracts.scene.measure_gs_table_plane_z`)
is deterministic and bounded: every 50th PLY vertex is read through a memory
map, mapped into MuJoCo world as a homogeneous point through `inv(w2gs)`
(uniform scale is fine; the result is divided by w), and kept when both
`|x - 0.0|` and `|y - (-0.10)|` are within 0.15 m of the table probe centre.
The kept z values are histogrammed in 5 mm bins over [-1.5, 1.5] m; the
result is the median of the points within 1 cm of the modal bin. The
measurement is re-run on every strict validation (launcher preflight, worker
start, rerender) and reported as `gs_table_plane_z_m` next to the declared
`calibration_table_z_m`; it costs a few seconds for a multi-GB scene once
the file is in the page cache. Fixture point clouds with fewer than 100 vertices cannot carry a table
surface, so the measurement is skipped for them (`gs_table_plane_z_m` is
`null`) and the declared value is accepted as-is; every real 3DGS scene has
hundreds of thousands of points.

`python -m robosuite_env.contracts.scene upgrade` measures and writes both
fields at bake time, using the *effective* matrix when
`--calibration-world-offset` is given. It refuses to write a contract whose
measured table-top is outside the tolerance: for a scene registered with the
table at another height, the error names the measured value and the
`--calibration-world-offset 0 0 <z>` that moves it to zero. A rewritten
`background.json` changes the scene's artifact digest, which must be
re-recorded in `artifacts/manifest.yaml`. `python -m robosuite_env.contracts.scene validate
<scene>` prints both `calibration_table_z_m` and `gs_table_plane_z_m`.

Preview conversion of an existing scene without writing:

```bash
python -m robosuite_env.contracts.scene upgrade \
  /path/to/scene \
  --table-collision-dir /path/to/table/coacd
```

After reviewing the JSON, add `--write`, then validate it:

```bash
python -m robosuite_env.contracts.scene upgrade \
  /path/to/scene \
  --table-collision-dir /path/to/table/coacd \
  --write
python -m robosuite_env.contracts.scene validate /path/to/scene
```

If a profile applied a runtime GS calibration offset, pass that offset
to the upgrade command (`--calibration-world-offset X Y Z`) so it is composed
into the new validated matrix. Without it the upgrade stops when
`calibration_table_z_m` places the 3DGS table-top outside the
`mujoco_world table_z=0` convention (see "Table height check" above). Never
edit an installed archive tree in place. Validate a separately staged scene,
give it a new immutable identity, update the checked-in artifact metadata and
its download pins together. Never modify a live scene while rollout workers
are reading it.

## Fail-closed checks

Before output workers or GPU servers start, official runs verify:

- exact checkpoint, policy environment, and controller runtime. The Ruckig
  check follows the mechanism, not the rig name
  (`validate_joint_trajectory_runtime`): FR3-embodiment rigs (`fr3`,
  `fr3_duo`) running dynamic `joint_ik` need `ruckig.__version__` 0.19.4 in
  `policy.python`; an embodiment without a rule is refused;
- FR3/YUBI/task-object MJCF references and visual/collision roles, and each
  task-object bundle's mesh set against its pin
  ([runtime mesh-set pins](assets.md#runtime-mesh-set-pins));
- the selected box-packing layout, its scene, and its rig
  ([task layout selection](configuration.md#task-layout-selection));
- the policy state quaternion hemisphere against the checkpoint's norm stats
  ([policy state quaternion hemisphere](configuration.md#policy-state-quaternion-hemisphere));
- forbidden asset filename and content hashes;
- exact scene path, point-cloud hash, transform validity/hash, and table
  collision count/hash;
- the OBS preprocessing contract, including contrast before padding so padded
  bands remain RGB `(0, 0, 0)`; and
- the required grid-render template for automatic 10-, 30-, and 100-repeat grids.

The current FR3 profile uses `policy_image_resize: pad`, so the
contrast-before-padding rule directly governs its padded bands.

The asset and table-collision digests are part of deterministic init-state
cache identity. A physics asset change cannot silently reuse an older state.
Neither can a data change: `robosuite_env/rollout/state/init_identity.py`
hashes the content of the files that place the objects (the selected layout,
positions file,
initial arm pose, and base placement) and the task
selectors, independent of where the files live. The batch runner adds that
digest (`python -m robosuite_env.rollout.state.init_identity -- <worker args>`)
to the cache name as `scene_inputs_sha`, and every written state records it
under `init_identity` together with where the environment seated each object.
Loading a state, cached or through `--init-state-in`, refuses any difference,
including a seated height more than 1 µm off and a state written before the
identity existed.
Every worker summary records the execution class and asset/scene contract
digests. Both release profiles use `policy_batch_size: 2`, so a one-scene run
uses five policy inference waves of two repeats. Multi-scene comparisons divide
that configured concurrency equally across scenes.
Every official batch client uses `runner.max_wall_sec` as a finite policy
barrier timeout. A missing client therefore fails the whole exact-batch wave
instead of waiting forever or degrading to a partial inference batch.
Schema v2 and the rollout worker therefore accept only the `single` RF sampler,
so `runner.policy_batch_size` is the exact policy model batch size.

After a complete launch, the entry point writes `rollout_success_summary.csv`,
`.json`, and `.png` under the run root. When the YAML contains exactly two
scenes, the JSON also reports their success-rate difference and a two-sided
Fisher exact p-value at the declared 0.05 threshold. This keeps a two-scene
comparison and its statistical result tied to the same pinned run.
