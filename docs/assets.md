# Assets

For the two release profiles, FR3 Flat cup on plate in the public A Room and
Reading Room, install by profile from the gated Hugging Face repositories
[`airoa-org/yubi-real2sim-policies`](https://huggingface.co/airoa-org/yubi-real2sim-policies)
(checkpoints, Gemma Terms of Use) and
[`airoa-org/yubi-real2sim-assets`](https://huggingface.co/datasets/airoa-org/yubi-real2sim-assets)
(scenes and meshes; the rooms are CC BY 4.0, AIRoA's `background.json` and
`table_collision/` Apache-2.0). Log in with `.venv/bin/hf auth login` and accept the
terms on both pages first:

```bash
.venv/bin/real2sim-download-release \
  --profile config/release/fr3/flat/a_room_cup_cup1trim.yaml
```

See the [profile-to-scene mapping and installer options](../README.md#release-evaluation).
The installer reads both repository ids, their pinned revisions and its
tree pins from `artifacts/release_downloads.json`. To read another revision,
pass `--hf-revision policies=COMMIT` / `assets=COMMIT` (or set
`REAL2SIM_HF_POLICIES_REVISION` / `REAL2SIM_HF_ASSETS_REVISION`), or
`--allow-unpinned` to read the `main` branch; the tree pins are checked either
way. The user workflow needs
no release index or separate restore route.

## Installed layout

`.env` contains the one machine-local asset path, `REAL2SIM_INPUTS_ROOT`. It
must be absolute and must not be a symlink. Profiles and
`artifacts/manifest.yaml` resolve external inputs through `{inputs_root}`:

```text
$REAL2SIM_INPUTS_ROOT/
├── checkpoints/
│   └── <checkpoint>/
├── scenes/
│   └── <scene>/
└── release-assets/
    └── INSTALLED.json
```

The same installer places repository-local runtime bundles at:

```text
robosuite_env/assets/
├── trellis_cup_plate/
│   ├── cup_vis.obj, plate_vis.obj, cup.png, plate.png
│   └── coll/
└── fr3_collision/
    └── link0.stl … link7.stl
```

With the default input root, the canonical patterns are
`local-inputs/checkpoints/<name>/` and `local-inputs/scenes/<scene>/`.

Never move a 3DGS scene below `robosuite_env/assets/`. A profile uses
`{inputs_root}/scenes/<scene>`, and its `scenes[].tag` must equal `name` in the scene's
`background.json`; the directory name need not equal the tag.

## What Git and the downloads own

Git contains the source, YAML profiles, artifact metadata, validation code,
scene contract code, and small checked-in robot/runtime assets. In particular,
`artifacts/manifest.yaml` binds runtime artifacts, while
`robosuite_env/assets/manifest.json` declares the visual/collision role of robot
and task meshes. Checked-in geometry includes `fr3_visual/`, the YUBI hand
assets, and `fr3_yubi_right_registered/`; their manifests retain upstream and
build provenance. The YUBI hands carry a shared bilateral exterior-guard
collision contract: 102 generated guards per hand, 178 right / 179 left
collision geoms including the mount and the existing fingertip/pad set, built
and checked by `robosuite_env/tools/build_yubi_full_collision.py`. See
[Shared YUBI exterior collision contract](yubi_guard_collision.md).
The FR3 visual source of truth is
`robosuite_env/assets/fr3_visual/manifest.json`.

The downloads contain the two 3DGS scenes, the `pi05_cup` checkpoint, the
cup/plate object meshes, and the FR3 collision meshes. Those large bytes are
deliberately not tracked by Git, even for the meshes installed under
`robosuite_env/assets/`.

## Manifest roles

`artifacts/manifest.yaml` is the runtime source of truth for artifact paths,
sizes, counts, and digests. Its repository-local production entries are:

| Role | Artifact id | Installed path |
| --- | --- | --- |
| cup/plate visuals | `public-cup-plate-visual-v1` | four paths below `{repo_root}/robosuite_env/assets/trellis_cup_plate` |
| cup/plate collisions | `public-cup-plate-collision-v1` | `{repo_root}/robosuite_env/assets/trellis_cup_plate/coll` |
| FR3 collisions | `fr3-collision-franka-description-2fd6aaa8` | eight paths below `{repo_root}/robosuite_env/assets/fr3_collision` |

The production environment check
`config/release/fr3/flat/a_room_cup_cup1trim.yaml` binds the
`pi05-cup` checkpoint together with the
public `public-scene-flat-a-room-v2` scene; the profile installer
downloads both at the pinned revisions and checks their tree pins.

Production preflight fails closed unless a resolved `local_path` and its kind,
size, file count, and Real2Sim digest match the selected manifest entry.

## Runtime mesh-set pins

The task-object meshes under `robosuite_env/assets/` (`trellis_cup_plate/`,
`trellis_office/`) are gitignored build outputs. The rest of the repository
asset contract reads tracked text, so a stale or partial mesh copy under a
current MJCF would still load, with the wrong shape and origin. The top-level
`runtime_mesh_set_sha256` map in `robosuite_env/assets/manifest.json` therefore
pins each task-object bundle's mesh set (currently box, cup, lan, pen, plate,
and tape). A bundle that pins runtime geometry
without a mesh-set pin is refused; a derived bundle (`derived_from`) cannot be
pinned itself and is checked against its parent's pin. The contract
(`robosuite_env/contracts/assets.py`) checks the bundles a run loads at
`--check-runtime` and launch, and again at worker start. Check one bundle with
`python -m robosuite_env.contracts.assets --bundle <id>`.

A pin is the SHA-256 of the `sha256sum` listing of the bundle's visual and
collision meshes (the `roles.*.expected_files` of its manifest entry, which are
exactly the meshes its MJCF selects), sorted by asset-root-relative path. A
copied tree can be checked with coreutils alone:

```bash
cd robosuite_env/assets
sha256sum trellis_cup_plate/plate_vis.obj trellis_cup_plate/coll/plate_*.obj \
  | LC_ALL=C sort -k2 | sha256sum
# 07f4ae6964166c432b55a53abf2139ca76b2833e254f48bf4f2ae21698613cf6  -
```

The printed digest must equal `runtime_mesh_set_sha256.plate`.

## Scene contract and provenance

The distributed scene trees are immutable schema-v2 bundles. Installation is
byte-for-byte placement, not a rename or migration.

A production `background.json` binds:

- `schema_version: 2`, a content-derived `id`, and `name`;
- `ply_path` and `ply_sha256`, plus exactly one of inline `mat_w2gs` or a
  `w2gs` file with `w2gs_sha256`;
- `calibration_table_z_m` with method `ply-plane-fit-v1`;
- `table.appearance: embedded_in_3dgs`; and
- collision directory, glob, count, tree digest, and `mujoco_world` frame.

Schema v2 pins the runtime reference plane `table_z` to 0. The collision parts
are compiled exactly as the contract gives them in `mujoco_world` (scale 1, no
offset), so their top need not be at z=0: the Duo slab's is +34 mm. Strict validation
remeasures the 3DGS table plane and rejects an absolute or declared/measured
discrepancy over 0.05 m. Runtime `gs_calibration_world_offset` is pinned to
zero; alignment corrections belong in a new immutable scene transform.

Validate an installed scene without writing it:

```bash
python -m robosuite_env.contracts.scene validate \
  "$REAL2SIM_INPUTS_ROOT/scenes/flat_reading_room"
```

To build a different scene, stage it outside the installed tree, use
`python -m robosuite_env.contracts.scene upgrade` with its table-collision
directory and any required calibration offset, validate it, assign a new
identity, and add a manifest entry. Never rewrite an installed scene; doing
so invalidates its tree digest.

### Public SuperSplat scenes

The release profiles use two public CC BY 4.0 rooms from SuperSplat, each
registered to the FR3 Flat rig. The attribution and how the rooms were
registered are in [`assets/public_scenes/`](../assets/public_scenes/README.md).
The registration itself, including the room's height, is only in each
scene directory's `background.json`. The repository pins each scene's tree in
`artifacts/release_downloads.json` and `artifacts/manifest.yaml`.

| Scene | Room and source | Artifact |
| --- | --- | --- |
| `flat_reading_room` | "Reading Room (No Lidar, Only Photos)" by Ethan, <https://superspl.at/scene/9d370db9> | `public-scene-flat-reading-room-v2` |
| `flat_a_room` | "A Room" by parsa mahdavi, <https://superspl.at/scene/3e754ea2> | `public-scene-flat-a-room-v2` |

Validate an installed public scene the same way. The command prints the
scene id, `calibration_table_z_m` and the measured table plane:

```bash
python -m robosuite_env.contracts.scene validate \
  "$REAL2SIM_INPUTS_ROOT/scenes/flat_reading_room"
```

## Checkpoint and tokenizer

A release checkpoint's
`assets/<asset_id>/norm_stats.json` must remain in place because OpenPI does not
search for alternate statistics.

A box-packing checkpoint may carry a `prompt.yaml` that records the
instructions it was trained on; preflight checks a `box_pack` profile's
instruction against it. The release checkpoint, `pi05_cup`, has none, and
preflight makes no prompt check for a cup profile.

The PaliGemma tokenizer is not redistributed because it is covered by the
Gemma terms. OpenPI anonymously fetches
`gs://big_vision/paligemma_tokenizer.model` on first use and caches it below
`OPENPI_DATA_HOME`. Do not add it to the artifact manifest or archive.

## Installer safety and recovery

The installer downloads only the profile's folders, into
`$REAL2SIM_INPUTS_ROOT/release-assets/downloads/`, where an interrupted
download resumes. It verifies each folder's tree digest (each file's path,
size and SHA-256) before installing it and deletes a folder that differs. It
reuses a verified installation, refuses to replace different existing data,
and records the repositories and revisions in
`$REAL2SIM_INPUTS_ROOT/release-assets/INSTALLED.json`. A 401 or 403 from
Hugging Face means the login or the accepted terms are missing.

Restore a changed or incomplete installation by removing the differing tree
and rerunning the same profile. Do not edit the manifest to bless a local
mutation.

## Licences

Each `license` field in `artifacts/manifest.yaml` is an SPDX licence
expression. For the public entries it matches `THIRD_PARTY_NOTICES` section
C; the test fixtures (`access: fixture`) carry `CC0-1.0`. The FR3 collision
meshes and the cup/plate meshes carry `Apache-2.0` (C.2, C.3). The public 3DGS
rooms carry `CC-BY-4.0 AND Apache-2.0`: the room stays under CC BY 4.0, and
AIRoA's `background.json` and `table_collision/` in each archive are under the
Apache License 2.0 (C.4). The policy checkpoint that `artifacts/release_downloads.json` publishes is a
Gemma Model Derivative, and its manifest entry carries
`LicenseRef-Gemma-Terms-of-Use` (see the License section of the README and
`THIRD_PARTY_NOTICES`).
