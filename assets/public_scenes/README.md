# Public 3DGS scenes

The release profiles render their background from two public 3D Gaussian
Splatting rooms published on SuperSplat under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

| Room | Creator | Source | Licence |
| --- | --- | --- | --- |
| "Reading Room (No Lidar, Only Photos)" | Ethan (SuperSplat user `ethan3111`) | <https://superspl.at/scene/9d370db9> | CC BY 4.0 |
| "A Room" | parsa mahdavi (SuperSplat user `parsamahdavi48`) | <https://superspl.at/scene/3e754ea2> | CC BY 4.0 |

The creators do not endorse this project. Each room is registered to the FR3
Flat rig, so there are two scene directories:

| Scene directory | Artifact | Rig and scene tag |
| --- | --- | --- |
| `flat_reading_room` | `public-scene-flat-reading-room-v2` | FR3 Flat, `franka_flat` |
| `flat_a_room` | `public-scene-flat-a-room-v2` | FR3 Flat, `franka_flat` |

Each scene directory is the folder `scenes/<scene>/` in the dataset
repository `airoa-org/yubi-real2sim-assets` on Hugging Face. The installer
puts it in `{inputs_root}/scenes/<scene>/`.

## Where the registration lives

The registration is only in the scene directory. This repository has no copy of it.
The installed `background.json` (schema v2) holds:

- `mat_w2gs`, the world-to-GS similarity transform. It includes the room's
  height above or below the simulator table.
- `calibration_table_z_m`, the measured height of the room's table plane in
  the simulator world.
- `id`, the scene id. It is derived from the scene's content, so it changes
  when the registration changes.

The rollout reads this file from the installed directory. The repository
keeps only the integrity pin of each scene directory: the SHA-256 and size of
its tree. These are in
`artifacts/release_downloads.json` and `artifacts/manifest.yaml`. The
installer and the launcher refuse a scene whose tree does not match its pin.

To read the registration of an installed scene:

```bash
python -m robosuite_env.contracts.scene validate \
  "$REAL2SIM_INPUTS_ROOT/scenes/flat_reading_room"
```

To change a registration, for example the room's height:

1. In a copy of the installed scene directory, regenerate `background.json`
   with `python -m robosuite_env.contracts.scene upgrade`. The command
   composes `--calibration-world-offset` into `mat_w2gs`, and it measures
   `calibration_table_z_m` and derives `id` again (see
   [Simulation operations](../../docs/simulation_operations.md)). Then
   validate the copy.
2. Upload the directory to `scenes/<scene>/` in the dataset repository, and
   update the dataset revision and the tree pin in
   `artifacts/release_downloads.json` and the tree pin in
   `artifacts/manifest.yaml`.

## How the rooms are registered

- The Reading Room was registered to the FR3 Flat rig by hand in Blender.
- "A Room" was registered to the FR3 Flat rig by hand in Blender, at life size
  (scale 1). In the floor plane, each room is placed so that no room furniture
  is inside the simulator table box or near the robot bases.

The table the simulation uses is not taken from either room. Each archive's
`table_collision/` is the rig's simulator table box: one invisible box, the
same file in both rooms. It is the rig's existing simulator table, carried
over unchanged, and was not fitted to these rooms. MuJoCo uses it as the
table: the objects are seated on it at reset. The rooms are registered to this
box, not the box to the rooms. The registration sets where a room's visible
surface lies relative to the box top (`calibration_table_z_m`, measured from
the point cloud, is −35 mm in the Reading Room and −32 mm in A Room), but the
box's outline is the rig table's, not the outline of a table in the room.

## Changes from the originals

- The SuperSplat SOG files were converted to a standard 3DGS PLY with
  `@playcanvas/splat-transform` 3.4.2. For "A Room", only level of detail 0 was
  converted, and spherical-harmonic bands above degree 0 were dropped
  (`-H 0`). The result equals the PLY from each scene page's Download button
  in Gaussian count, order and every kept value; the
  "A Room" download has bands 1 to 3, which are the ones dropped.
- The spherical-harmonic fields were zero-padded to degree 3, because the
  GS server loads degree-3 models. Zero coefficients leave the rendered
  colour unchanged.
- The registration only places the room in the simulator world
  (`mat_w2gs`, a similarity transform with uniform scale). The Gaussians are
  not edited.
- `table_collision/` is the rig's simulator table box. Its geometry is the rig's
  existing simulator table, unchanged, and it is not part of the original capture.

Each archive also ships these changes, the CC BY 4.0 notice and the
Apache-2.0 statement below in its `ATTRIBUTION.txt`.

This project adds `background.json` (the registration) and `table_collision/`
(the simulator table box) to each scene directory. Both are AIRoA's work and
are licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). This licence
applies to these files in the archives on their own; it is not the licence of
the repository's code, which is set separately. The room
itself, the point cloud, stays under CC BY 4.0; the `license` field inside
`background.json` names the room's licence. Each scene directory carries
the full licence text as `LICENSE-APACHE-2.0`.

Other rights, such as publicity, privacy or moral rights, and rights in
works visible in a capture, are not licensed by CC BY 4.0. Third-party works
visible in a room, such as the framed photographs on the Reading Room wall,
are not covered by CC BY 4.0 or by Apache-2.0. Rights holders who want such a
work removed can contact <ext-oss@airoa.org>.
