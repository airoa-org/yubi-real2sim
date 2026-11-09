# Provenance

This repository grew out of robosuite's environment layer. It is published
without its development history, so the lineage facts that normally live in
Git history are recorded here instead.

## Lineage

- The object/robot layer is built on
  [robosuite](https://github.com/ARISE-Initiative/robosuite)'s `Environment`
  stack. Everything under `robosuite_env/` descends from that base.
- Upstream pin: ARISE-Initiative/robosuite commit
  `85abee228d1c43ab1939bce33028099945d453b4` (v1.5.2), declared once as the
  `pyproject.toml` source and locked in `uv.lock` (what actually installs and
  runs). There is no `third_party/robosuite` checkout; the launcher and the
  rollout worker refuse a robosuite that does not import from that installed
  distribution, and a test keeps this document on the locked revision.

## What is inherited from robosuite (and still executes)

No upstream source file is vendored or copied into this tree; robosuite is a
library dependency. Six modules subclass it, and the inherited machinery runs
on every step:

| This repository | Upstream base | Overridden |
| --- | --- | --- |
| `robosuite_env/tasks/envs/cup_plate.py` | `environments.manipulation.lift.Lift` | `_load_model` |
| `robosuite_env/tasks/envs/cup_plate_registered.py` | `CupPlateLift` (above) | `_load_model`, `_configure_task_objects` |
| `robosuite_env/tasks/objects.py` | `models.objects.MujocoXMLObject` | constructor |
| `robosuite_env/robots/fr3/model.py` | `models.robots.ManipulatorModel` | defaults, `init_qpos` |
| `robosuite_env/robots/yubi/model.py` | `models.grippers.GripperModel` | `format_action`, `dof`, geoms |
| `robosuite_env/tasks/scene.py` | (uses `base_xpos_offset` API) | — |

At runtime, `reset`/`step`, observation assembly, and the `JOINT_POSITION`
servo of the `joint_ik` controller are upstream robosuite code; this
repository owns the joint targets (null-space IK + Ruckig trajectories),
grasping, success criteria, 3DGS compositing, and the entire rollout,
rendering, and reproducibility pipeline (~98% of the Python in the tree).

One upstream table is reused: the `base_xpos_offset` values of `robosuite_env/robots/fr3/model.py` are those of
robosuite's Panda model (MIT License; see `THIRD_PARTY_NOTICES`).

## Code adapted from openpi

Two policy modules adapt [openpi](https://github.com/Physical-Intelligence/openpi)
code at the `uv.lock` pin `e6b0441` (Apache License 2.0). Each file's header
names its source and states the changes, and `THIRD_PARTY_NOTICES` lists them:

- `robosuite_env/policy/yubi_openpi/yubi_policy.py`: `_parse_image` and the
  Inputs/Outputs transform pattern of `src/openpi/policies/libero_policy.py`.
- `robosuite_env/policy/rf_action_inpainting.py`: the Euler sampler of
  `Pi0.sample_actions` in `src/openpi/models/pi0.py`.

## Asset attribution

- `robosuite_env/assets/fr3_visual/`, `fr3_collision/`: derived from
  [franka_description](https://github.com/frankarobotics/franka_description)
  (Apache-2.0; licence and pinned commit recorded in
  `robosuite_env/assets/fr3_visual/README.md`). The FR3 MJCF is generated,
  not copied from robosuite; only Franka's DH constants coincide.
- `third_party/GaussianWrapping`: a submodule pinned to the
  [airoa-org/GaussianWrapping](https://github.com/airoa-org/GaussianWrapping)
  fork of [GaussianWrapping](https://github.com/diego1401/GaussianWrapping)
  `11e3b6f` with the forward KB4 fisheye rasterizer
  ([diego1401/GaussianWrapping#13](https://github.com/diego1401/GaussianWrapping/pull/13))
  and a per-call KB4 near cull distance (`kb4_near`), a fork commit made for
  this repository. Every 3DGS render runs through code under the
  Gaussian-Splatting License (Inria and MPII, non-commercial research and
  evaluation use only): the root `LICENSE.md` covers the rasterizers and
  `simple-knn`, and 18 files of `gaussian_wrapping/`, among them the
  `gaussian_renderer`, `scene` and `utils` modules the GS server imports, carry
  Inria headers under that licence. `gaussian_wrapping/LICENSE` is Apache-2.0
  but does not relicense those files, and `gaussian_wrapping/utils/sh_utils.py`
  is BSD-2-Clause (PlenOctree authors). `THIRD_PARTY_NOTICES` lists the files.
- The two fork commits (`cd8cb18`, the KB4 rasterizer, and `1470252`, the
  `kb4_near` commit the gitlink pins) were written by AIRoA and change
  Gaussian-Splatting-License files in
  `submodules/diff-gaussian-rasterization_ours`. They are distributed under
  the Gaussian-Splatting License, as its section 4 requires; this
  repository's Apache License 2.0 does not apply to them.
- The YUBI gripper meshes (`robosuite_env/assets/yubi_left/`, `yubi_right/`,
  `fr3_yubi_right_registered/`) and `robosuite_env/assets/yubi_guard_collision_manifest.json`
  are derived from the YUBI hardware design,
  [Toyota/yubi-hw](https://github.com/Toyota/yubi-hw) (Copyright 2026 Toyota
  Motor Corporation, CERN-OHL-W-2.0), and stay under CERN-OHL-W-2.0;
  `THIRD_PARTY_NOTICES` A.4 lists the files and the modifications. The YUBI
  MJCF files and `robosuite_env/robots/yubi/` are written by AIRoA.
- The cup/plate meshes (`trellis_cup_plate` download) were generated by AIRoA with TRELLIS.2
  from AIRoA's own photographs, after background removal with BRIA RMBG-2.0; TRELLIS.2 uses
  Meta DINOv3 and NVIDIA nvdiffrast. `THIRD_PARTY_NOTICES` C.3 lists these tools and their
  licences, which do not cover the meshes.
- The public 3DGS rooms (download archives, described in
  `assets/public_scenes/`) are adaptations of "Reading Room (No Lidar, Only
  Photos)" by Ethan
  (<https://superspl.at/scene/9d370db9>) and "A Room" by parsa mahdavi
  (<https://superspl.at/scene/3e754ea2>), both licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The changes
  (format conversion, spherical-harmonic padding, registration to the FR3 Flat rig,
  the added simulator table box) are listed in
  `assets/public_scenes/README.md` and in each archive's `ATTRIBUTION.txt`.
  What this project adds to each room, `background.json` (the
  registration) and `table_collision/` (the simulator table box), is AIRoA's
  work under Apache-2.0. That licence covers these archive files on their own,
  not the repository's code.
