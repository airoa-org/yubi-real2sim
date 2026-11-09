# Release condition assets

One directory per rig and task holds every repository file a condition in
`config/release/<robot>/<rig>/<room>_<task>_<checkpoint>.yaml` reads:

```
assets/release/<robot>/<rig>/<task>/
  MANIFEST.json            what each condition reads: files (source path and
                           sha256), checkpoint, scene, initial pose
  base_placement.json      robot base pose in the world
  gripper_calib_rad.json   YUBI jaw calibration
```

The release has one such directory, `fr3/flat/cup/`, which both release
profiles read. `fr3/flat` is the Franka Flat rig (scene tag `franka_flat`).
`<room>` is `reading_room` or `a_room`, the public 3DGS room the condition
runs in (`{inputs_root}/scenes/flat_<room>`, see
[`assets/public_scenes/`](../public_scenes/README.md)); `MANIFEST.json` keys
each condition as `<room>_<checkpoint>`. Both rooms read the same files.

The initial object positions live in [`assets/initpose/`](../initpose/README.md)
as `<robot>/<scene>/<task>.json`. The initial arm pose is not copied here either: each profile names it as
`<robot>/<rig>/<name>` and it lives once under
[`assets/initial_poses/`](../initial_poses/README.md); `MANIFEST.json` records
which one each condition uses.

Every file is a byte-identical copy of the file the published run read, or for
the file the rig's profile read; only the name changed; `MANIFEST.json` records
the original path and sha256. They are copies, not symlinks, because the
launcher seals every runtime file and refuses symlinks; the bundle is frozen
for the release, so a later change to a code default does not move it.

Some copies were later edited only in their provenance text; the positions and
parameters they carry are unchanged. For such a file `MANIFEST.json` pins the
tracked bytes as `sha256` and keeps the digest the published run read as
`recorded_sha256`.

## Large inputs

Install the inputs for a profile, after `.venv/bin/hf auth login` and accepting the
terms of both release repositories on Hugging Face:

```bash
.venv/bin/real2sim-download-release \
  --profile config/release/fr3/flat/a_room_cup_cup1trim.yaml
```

Repeat `--profile` to install both profiles into one root. The installer reuses matching inputs,
checks each downloaded folder's tree digest, installs repository meshes
when needed, and writes `.env`. Its `--dry-run` lists the exact destinations
without downloading.

Checkpoints and 3DGS scenes are not in the repository. Download them from
the gated Hugging Face repositories
[`airoa-org/yubi-real2sim-policies`](https://huggingface.co/airoa-org/yubi-real2sim-policies)
(one folder per checkpoint) and
[`airoa-org/yubi-real2sim-assets`](https://huggingface.co/datasets/airoa-org/yubi-real2sim-assets)
(`scenes/` and `assets/`), and place them in the layout below. A shared host
can lay them out per condition the same way, with hard links, under

```
/ABSOLUTE/PATH/TO/real2sim-assets/release/<robot>/<rig>/<task>/
  checkpoints/<checkpoint>/
  scenes/<scene>/
```

Point `REAL2SIM_INPUTS_ROOT` at `<robot>/<rig>/<task>` and run, for example,

```bash
REAL2SIM_INPUTS_ROOT=/ABSOLUTE/PATH/TO/real2sim-assets/release/fr3/flat/cup \
  real2sim-run --config config/release/fr3/flat/reading_room_cup_cup1trim.yaml --check-runtime
```
