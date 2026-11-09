# 3DGS scene contract

3DGS payloads are external runtime inputs. A scene directory must contain a
`background.json` object with:

- a stable `name` (used as the YAML scene tag);
- a PLY path under `ply`, `gs`, `gs_ply`, `point_cloud_ply`, `ply_path`, or
  `path`; and
- a world-to-GS transform, either inline as `mat_w2gs` or referenced by
  `w2gs`, `gs_w2gs`, `mat_w2gs_path`, `world_to_gs`, or
  `world_to_gs_path`.

Relative asset paths resolve from the scene directory. Large PLY files are
ignored by Git.

If the transform was calibrated in a world with a different origin from the
active MuJoCo scene, pass `--gs-calibration-world-offset X Y Z`. The vector is
added before the world-to-GS matrix (`M_effective = M_w2gs @ Translation(T)`).
For example, a transform calibrated with the table at `z=0.8 m` needs
`0 0 0.8` when the runtime desk is at `z=0`. The release configurations
use `0 0 0`.
