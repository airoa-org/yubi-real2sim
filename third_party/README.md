# Third-party layout

`scripts/bootstrap.sh` creates pinned checkouts. `GaussianWrapping` is a tracked
root submodule; bootstrap initializes it automatically after an ordinary
`git clone`, without its nested submodules, which nothing here builds or imports.
The other source checkouts are
ignored:

```text
third_party/franka_description/
```

OpenPI and robosuite are not checked out here. `uv sync` installs the
`openpi`, `openpi-client` and `robosuite` distributions from the commits
`uv.lock` pins, and the launcher preflight and the rollout worker refuse a copy
that imports from anywhere else (including a leftover `third_party/openpi/` or
`third_party/robosuite/` checkout).

`GaussianWrapping` pins the `main` branch of the
[airoa-org/GaussianWrapping](https://github.com/airoa-org/GaussianWrapping) fork:
upstream [diego1401/GaussianWrapping](https://github.com/diego1401/GaussianWrapping)
`11e3b6f` plus the forward KB4 fisheye rasterizer
([diego1401/GaussianWrapping#13](https://github.com/diego1401/GaussianWrapping/pull/13))
and a per-call KB4 near cull distance (`kb4_near`, GS units, default 0.2; see
[docs/direct_kb4_development.md](../docs/direct_kb4_development.md)).
Its root, rasterizers and `simple-knn` are under the Gaussian-Splatting License
(Inria and MPII, non-commercial research and evaluation use only). So are 18
files of `gaussian_wrapping/` that carry Inria headers, among them the
`gaussian_renderer`, `scene` and `utils` modules the GS server imports:
`gaussian_wrapping/LICENSE` is Apache-2.0 but does not relicense them, and
`utils/sh_utils.py` is BSD-2-Clause. The fork commits (the KB4 rasterizer and
`kb4_near`) were written by AIRoA and change Gaussian-Splatting-License files, so
they are distributed under that licence, not under this repository's Apache
License 2.0. `THIRD_PARTY_NOTICES` lists the files and licences.

The YUBI policy transform and OpenPI deploy configuration used at runtime live
in `robosuite_env/policy/yubi_openpi/`, a first-party package installed with
the project rather than added to `sys.path`.
`tests/fixtures/franka_yubi_sim/build_scene.py` is a reference copy of the
wrist-camera calibration. Nothing imports or ships it:
`robosuite_env/rendering/fisheye_projection.py` copies its constants, and
`tests/unit/rendering/test_wrist_render.py` checks them against it.
