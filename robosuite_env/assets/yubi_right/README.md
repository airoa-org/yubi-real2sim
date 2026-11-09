# YUBI gripper meshes (right)

The visual meshes (`visual/`, `visual_mj/`), the collision meshes (`coll/`,
`coll_mj/`, `coll_mj_hulls/`, `guard_collision/`) and `manifest.json` in this
directory are derived from the YUBI Gripper (DYNAMIXEL) design. YUBI (Yielding
Universal Bidigital Interface) is developed by Frontier Research Center, Toyota
Motor Corporation.

- Source Location: https://github.com/Toyota/yubi-hw
- Copyright 2026 Toyota Motor Corporation
- Licence: CERN Open Hardware Licence Version 2 - Weakly Reciprocal
  (CERN-OHL-W-2.0), not the Apache License 2.0 of this repository's code.

Modification notice (CERN-OHL-W-2.0 section 3.3(b)). Modified by AIRoA:

- Made on 2026-07-05: the assembled gripper COLLADA scene
  (`YUBI_Gripper_DYNAMIXEL_right_use_assembled.dae`, recorded as `source_dae`
  in `manifest.json`; not part of this repository) was split into 60 per-part
  OBJ meshes, re-exported for MuJoCo, and given CoACD convex collision parts.
- `guard_collision/` (added on 2026-08-25): 102 convex exterior collision
  guards, generated from those meshes by
  `robosuite_env/tools/build_yubi_full_collision.py`
  (`docs/yubi_guard_collision.md`).

`THIRD_PARTY_NOTICES` A.4 gives the full notice and lists the files. Section F
of `THIRD_PARTY_NOTICES` reproduces the CERN-OHL-W-2.0 text from the `LICENSE`
file of https://github.com/Toyota/yubi-hw at commit
`dd8bd13d2fd8e5003057243576f88be333d95fc5`.
