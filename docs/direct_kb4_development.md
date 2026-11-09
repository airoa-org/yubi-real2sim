# Direct KB4 wrist rendering

- `direct_kb4` is the only backend for simulated wrists, and the default of `simulation.parameters.policy_wrist_backend` and `--policy-wrist-backend`. Any other value of the key or flag is rejected like any unknown value.
- The launcher always passes `--policy-wrist-backend`, so the recorded worker command names the backend.
- It applies to every profile and command, including those that omit the key or flag: `policy_image_source: sim` profiles under `config/`, direct `python -m robosuite_env.sim` runs, policy observations and wrist video (also in runs with `policy_image_source: lerobot` or without a policy).
- Rerendering takes the backend from the recorded command, else from `sim_summary.json` or the first logged policy observation. Every run records the backend. A run that records no backend, a null backend, or a backend other than `direct_kb4` is refused ([Runs without a supported backend](#runs-without-a-supported-backend)).
- The same wrist factory serves rollout policy observations and wrist video.
- A direct `python -m robosuite_env.sim --state-log-replay` resolves the backend like rerendering, with the same function (`robosuite_env/rollout/replay/recorded_wrist.py`) and the `sim_summary.json` beside the state log: without the flag it takes the recorded backend, it refuses a run without a supported backend, and it refuses a `--policy-wrist-backend` that contradicts the recorded one.
- Select the camera profile independently with `simulation.parameters.policy_yubi_render_profile` or `--policy-yubi-render-profile`. The direct backend supports both named profiles and rejects custom FOV/crop overrides.
- The default ELP-mean profile uses one 640 × 480 source image per wrist. The selectable `franka-pinhole-wrist-rgb` profile retains its wider source and side-dependent crop.

## Development dependencies

- Use a declared editable installation of this checkout so `real2sim`, `robosuite_env` and the first-party `scene` package resolve from the same source. Do not insert the checkout into `sys.path` or `PYTHONPATH`.
- Only an editable install is supported: a non-editable wheel packages `robosuite_env` without its assets.
- The runner starts `scene/gs_background_server.py` as a file, and KB4 request validation imports `robosuite_env`. So the server's interpreter needs this checkout installed. Without it, the server does not advertise `kb4-radial-depth-v1`, and a `direct_kb4` worker stops before its first capture.
- The normal runtime dependencies remain in `pyproject.toml`/`uv.lock`. A focused renderer environment can install its dependencies separately and then run `uv pip install --no-deps --editable <checkout>`; this does not install a complete simulator or policy environment.
- Direct GS requires GaussianWrapping's `render_ours(camera_settings=...)` adapter and rebuilt `diff_gaussian_rasterization_ours` native extension. `third_party/GaussianWrapping` is a submodule pinned to the [airoa-org/GaussianWrapping](https://github.com/airoa-org/GaussianWrapping) fork: upstream GaussianWrapping `11e3b6f` plus the KB4 rasterizer, which is open upstream as [diego1401/GaussianWrapping#13](https://github.com/diego1401/GaussianWrapping/pull/13), and the per-call KB4 near cull `kb4_near` (below).
- The fork keeps the upstream licences: the Gaussian-Splatting License (Inria and MPII, non-commercial research and evaluation use only) for the root, the rasterizers, `simple-knn` and the 18 Inria-headed files of `gaussian_wrapping/`, which include the modules the GS server imports. `gaussian_wrapping/LICENSE` (Apache-2.0) does not relicense those files. The fork commits change Gaussian-Splatting-License files and are under that licence too; see [`THIRD_PARTY_NOTICES`](../THIRD_PARTY_NOTICES).
- An existing checkout needs `scripts/bootstrap.sh` in the supported CUDA environment, which checks out the gitlink and rebuilds extensions when their recorded build inputs change. Bootstrap rejects a GaussianWrapping checkout with tracked changes; restore it with `git -C third_party/GaussianWrapping checkout -- .` and rerun.
- Point the development GS server at that explicitly prepared source/runtime. The server checks the adapter/settings/native KB4 interface and records loaded module/native paths and hashes. A checkout revision alone does not identify a previously built extension.
- Unsupported renderer features raise an error. Direct mode does not substitute another projection when dependencies, server capabilities or scene features are incompatible.

## Image and depth contract

- KB4 near cull: the native KB4 projection drops every Gaussian whose mean lies within `kb4_near` GS units of the camera centre. It is a per-call setting (`GaussianRasterizationSettings.kb4_near`, default 0.2). A GS unit is 1/s metres for a scene whose world-to-GS scale is s, so a fixed 0.2 GS-unit cut is 0.2/s m: 0.2 m only in a life-size scene, and larger in a scene registered with s < 1. In the wrist cameras such a cut removes the scene within that radius, including the table and objects next to the gripper, and leaves black pixels.
- `--gs-kb4-near-cull` selects it. `znear` (default) requests the GS client's `znear`, 0.01 m converted to GS units with the scene's scale: the same limit the depth encoding uses, equal to the profile's 10 mm camera setback behind the optical centre, and far above float32 resolution at the scene's coordinate range. `legacy` requests 0.2 GS units whatever the scale.
- Configuration profiles do not set the option; the launcher always passes `--gs-kb4-near-cull znear`, so the recorded worker command names the mode even when a run stops before it records anything else. A run records `gs_kb4_near_cull` in `sim_summary.json` and in each policy observation, and the effective distance in metres as `gs_kb4_near_m`. Both describe the `direct_kb4` wrist render only. Rerendering takes the mode from the recorded command, else from `sim_summary.json` or the first logged policy observation; a direct `python -m robosuite_env.sim --state-log-replay` takes it from the same function (`robosuite_env/rollout/replay/recorded_wrist.py`). A run that records none replays with `legacy`, and a flag that contradicts the recorded mode is refused.
- An official rerender requires the renderer's GaussianWrapping checkout at the commit the run recorded. `legacy` covers only replays outside the official rerender, such as `python -m robosuite_env.sim --state-log-replay`: it applies the fixed 0.2 GS-unit cull, but its output is not an official rerender of the run.
- The server advertises `kb4-near-v1` only when the loaded rasterizer has the `kb4_near` setting and the ten-value native KB4 signature. A KB4 request must carry `kb4_near`. A `direct_kb4` worker therefore stops before its first capture against a build without `kb4_near` instead of silently using the fixed cull.
- The pinhole path keeps GaussianWrapping's fixed 0.2 GS-unit frustum cull (`auxiliary.h` `in_frustum`). The only pinhole render of a rollout or rerender is the overview background.

- One MuJoCo optical view and one GS KB4 camera produce each source image. MuJoCo uses separate color and segmentation/depth subpasses.
- Both layers use the same source intrinsics, corrected front-camera pose, 200° validity limit and world-to-GS alignment.
- Direct MuJoCo depth is Euclidean meters. GS returns native radial distance; the client converts it to meters once. Depth composition uses a 2 mm radial tolerance; foreground composition keeps each MuJoCo occupied pixel. Direct GS always requests RGBD.
- Missing reconstructed GS geometry has infinite depth. Invalid image-circle pixels have `nan` depth and black RGB before calibration. Foreground occupancy comes from segmentation.
- Composition, profile crop, RGB calibration, policy contrast and policy resize run in that order. Observation records include `policy_wrist_backend`.
- The lifecycle has three stages: capture both MuJoCo wrists, complete GS requests after releasing the render lock, then finalize on CPU. Direct GL resources are shared by the wrist pair and close before the owner context.
- The wrist renderer refreshes MuJoCo's camera/light cache after applying the profile's 10 mm setback and after restoring the shared model, so `Renderer.update_scene` reads the configured camera pose.

## Runs without a supported backend

- A rerender or `python -m robosuite_env.sim --state-log-replay` refuses a run that records no wrist backend, or a backend other than `direct_kb4`. The replay is not switched to `direct_kb4`, because a new `direct_kb4` render is a different camera, not a replay.

## Validation scope

- Focused tests cover the real camera-cache transaction and restoration, source radial depth composition, typed configuration propagation and actual small-scene direct capture/resource sharing.
- GPU tests require a working EGL runtime. A skipped GPU/asset test is not validation of that behavior.
- A frozen-model integration harness can exercise production `WristFisheye` and the native GS server without constructing a full rollout or applying actions. Record whole-frame wall time, loaded dependency identities, camera/state hashes and final policy input images.
- Component benchmarks, offline policy calls and frozen-state rendering do not establish closed-loop task success or real-robot operation.
- `test_kb4_native.py` runs the native path with k1..k4 = 0 and with the ELP-mean k1..k4, and checks that a Gaussian 0.03 m from the camera, inside 0.2 GS units for a GS unit of 1 m and of 0.25 m, renders with `znear` and is culled with `legacy`. It skips without `GS_KB4_NATIVE_REPO` and fails against a build without `kb4_near`.
- Direct-backend timings have only been measured at one pose on one GPU with persistent renderers. Launcher workers pass `--render-lock`, so they build and close the wrist pair for every observation: each observation compiles the GL programs and uploads meshes and textures. That mode is not measured.
- `mujoco_kb4.py` raises `NotImplementedError` for scene features it does not reproduce (shadows, reflections, more than 8 lights, …). No test checks the maintained scenes against that list, so a scene that uses one stops at its first direct wrist render.
- The 30-array pinhole byte comparison is reproducible with the GaussianWrapping submodule's `submodules/diff-gaussian-rasterization_ours/tests/README.md` ("Pinhole byte comparison"): it generates the synthetic fixture and compares the upstream and KB4 builds with `pinhole_regression.py`. Its arrays (108 MB each) are not committed.
