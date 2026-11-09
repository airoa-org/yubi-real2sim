# YUBI attachment and flange meshes (FR3 mount)

The YUBI attachment and flange meshes that mount the gripper on the FR3
(`visual/` and `collision/`) and `manifest.json` in this directory are derived
from the Franka mounting parts of the YUBI Gripper (DYNAMIXEL) design. YUBI
(Yielding Universal Bidigital Interface) is developed by Frontier Research
Center, Toyota Motor Corporation.

- Source Location: https://github.com/Toyota/yubi-hw
- Copyright 2026 Toyota Motor Corporation
- Licence: CERN Open Hardware Licence Version 2 - Weakly Reciprocal
  (CERN-OHL-W-2.0), not the Apache License 2.0 of this repository's code.

Modification notice (CERN-OHL-W-2.0 section 3.3(b)). Modified by AIRoA:

- Added on 2026-08-01: the attachment and flange meshes were placed in the
  YUBI physical-root frame of the FR3 mount chain recorded in `manifest.json`,
  and given CoACD convex collision parts.

`THIRD_PARTY_NOTICES` A.4 gives the full notice and lists the files. Section F
of `THIRD_PARTY_NOTICES` reproduces the CERN-OHL-W-2.0 text from the `LICENSE`
file of https://github.com/Toyota/yubi-hw at commit
`dd8bd13d2fd8e5003057243576f88be333d95fc5`.
