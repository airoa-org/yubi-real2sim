<div align="center">

# YUBI-Real2Sim policy evaluation

🌐 [**Project Page**](https://research.airoa.io/publications/yubi-real2sim/) &nbsp;|&nbsp; 📄 [**arXiv**](https://arxiv.org/abs/2610.17521) &nbsp;|&nbsp; 🎬 **Video (Coming Soon)**

</div>

This repository evaluates an OpenPI pi-0.5 policy on a Franka FR3 with bimanual YUBI in MuJoCo.
A 3D Gaussian-Splat (3DGS) room capture supplies the background. The included task asks the robot
to put a cup on a plate. A run produces scored rollouts, logs, videos, and an aggregate summary.

The repository code is Apache-2.0, but the 3DGS renderer (GaussianWrapping) is non-commercial, so
the full pipeline cannot be used commercially; see [License](#license).

## Code, inputs, and results

| Part | Contents | Location |
| --- | --- | --- |
| Checkout | Launcher, simulator, tools, and YAML profiles | This Git repository |
| Inputs | 3DGS scenes, checkpoint, table collision, and runtime meshes | `$REAL2SIM_INPUTS_ROOT` (default `<repo>/local-inputs`) |
| Results | Per-rollout logs and videos, summaries, and a grid video | `outputs/<run_name>/` |

- On a fresh install, the asset installer creates `.env` from `.env.example`.
- `REAL2SIM_INPUTS_ROOT` is the only machine-local input path and must be absolute and symlink-free.
- Keep 3DGS scenes out of `robosuite_env/assets/`.

Run the workflow from a GPU-capable interactive shell whose resources have already been provided
by the host environment.

- This repository never submits or cancels allocations and never discovers job IDs; optional
  [rollout shards](docs/rollout_shards.md) can attach `srun` steps to existing jobs.
- The release profiles select visible GPU slot `0` for the policy, 3DGS, and MuJoCo:
  - When `CUDA_VISIBLE_DEVICES` is set, slot `0` means its first entry.
  - When it is unset, slot `0` means host GPU ordinal `0`.

## Host prerequisites

Hardware and OS:

- x86-64 Linux with glibc 2.31 or newer
- NVIDIA driver 570.26 or newer
- One CUDA GPU with enough memory for OpenPI, the 3DGS workers and MuJoCo
- About 30 GB of free disk for the environment check on a host. The measured parts come to about
  26 GB: `.venv` about 11 GB, its inputs about 14 GB, and the checkout with one run's outputs about
  1 GB. About 10.5 GB of `.venv` is hard-linked from uv's cache, so the cache adds little while it
  is on the same filesystem as the checkout; on another filesystem uv copies, and the cache needs
  about 10 GB more. The [container](docs/container.md#host-requirements) setup needs about 55 GB.

Software:

- System Python 3.9+
- Git and curl
- `uv` 0.12.3 (exact version): `curl -LsSf https://astral.sh/uv/0.12.3/install.sh | sh`, or
  `uv self update 0.12.3` if a standalone uv is already installed
- GCC/G++ 13
- CUDA 12.8 toolkit with `nvcc` (exact version)

Bootstrap installs CPython 3.11.15 and FFmpeg 7.0.2 itself.

The two release profiles run the policy, the 3DGS workers, and MuJoCo on that one GPU (slot `0`).
The environment check, `config/release/fr3/flat/a_room_cup_cup1trim.yaml`, and
`config/release/fr3/flat/reading_room_cup_cup1trim.yaml` run two rollout workers at a time. Both
set:

- `runner.policy_batch_size: 2`
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`
- `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`

These settings target one 16 GB GPU such as an RTX 5070 Ti. See
[VRAM and reporting](docs/rtx_environment.md#vram-and-reporting) and
[Known limitations](#known-limitations).

A two-GPU host also runs one 3DGS server and all MuJoCo on slot `0` by default. To use both GPUs,
set `runtime.gs_cuda_devices` and `runtime.sim_cuda_devices` to `"0,1"` in a local copy of the
profile under `config/`, such as the git-ignored `config/local_run_two_gpu.yaml`. The launcher
only accepts profiles under `config/`.

The toolchain is defined in [`config/support-matrix.yaml`](config/support-matrix.yaml).

- The single locked runtime uses PyTorch 2.7.1 / torchvision 0.22.1 from the explicit cu128
  index for Hopper and RTX 50 / Blackwell GPUs.
- GPU execution still needs validation on the target machine; see
  [RTX setup and verification](docs/rtx_environment.md), including VRAM limits and a
  batch-size-one smoke run.
- An RTX 5070 with 12 GB is experimental.

Select GCC/G++ 13 and CUDA 12.8 using the host's package or environment manager before the host
gate. The commands must resolve to the selected toolchain in the current shell:

```bash
gcc -dumpfullversion
g++ -dumpfullversion
nvcc --version
```

A host without these tools can use the image that [`docker/Dockerfile`](docker/Dockerfile) builds.

- It provides CUDA 12.8, GCC/G++ 13, uv 0.12.3, Git, curl and system Python.
- The host then needs the NVIDIA driver, Docker and the NVIDIA Container Toolkit.

[Container environment](docs/container.md) gives the build and run commands; run steps 2 to 10
below inside that container.

## From clone to a finished rollout

Run this first on a new host. The FR3 Flat cup-on-plate profile
`config/release/fr3/flat/a_room_cup_cup1trim.yaml` is the environment check: it is the only release
profile in `reproducibility.mode: production`, so it runs every production gate before its ten
rollouts. Use it for installation, resolve, runtime check, and execution. Run the other release
profile only after it passes.

Production mode requires:

- a clean Git tree: `git status --porcelain --untracked-files=all` prints nothing (`.env`,
  `.venv/`, `outputs/` and installed assets are git-ignored), and every submodule is at its
  recorded commit;
- the runtime from `scripts/bootstrap.sh`, matching `uv.lock` exactly, with `uv 0.12.3` on `PATH`
  and the Gaussian CUDA extensions built by bootstrap;
- the profile's inputs installed by step 5, with digests that match `artifacts/manifest.yaml`;
- a CUDA GPU and driver that `nvidia-smi` reports, and FFmpeg;
- if an `origin` remote is set, an HTTPS or GitHub SSH URL for it (an `origin` that is a local
  path or a bundle fails the git check; a checkout without `origin` passes).

To experiment, use the fixture-mode profile `config/release/fr3/flat/reading_room_cup_cup1trim.yaml`
or a git-ignored copy such as `config/local_run_check.yaml` with `reproducibility.mode: fixture`.

### 1. Clone and enter the repository

```bash
git clone https://github.com/airoa-org/yubi-real2sim.git real2sim && cd real2sim
```

Success means the checkout contains `scripts/check_host.sh` and `config/release/fr3/flat/a_room_cup_cup1trim.yaml`.

### 2. Run the read-only host gate

```bash
./scripts/check_host.sh
```

Continue only on all of:

- exit `0`;
- `DECISION: PROCEED`;
- `NEXT_COMMAND: ./scripts/bootstrap.sh`.

On `DECISION: STOP`, do not bootstrap. Instead:

1. Return everything between `COPY_REPORT_BEGIN` and `COPY_REPORT_END` to the host administrator.
2. Correct every blocker.
3. Rerun this step.

### 3. Bootstrap the locked runtime

```bash
scripts/bootstrap.sh
```

Success ends with `[bootstrap] complete`. The script:

1. repeats the host gate;
2. creates `.venv`;
3. syncs frozen `uv.lock`, which builds the three Gaussian CUDA extensions;
4. checks the installed torch/torchvision CUDA versions.

Bootstrap initializes the root GaussianWrapping submodule but not its nested submodules, which
nothing here builds or imports. `git clone --recursive` is not required, and it would fetch them
too. OpenPI and robosuite are installed from the commits `uv.lock` pins, not from checkouts.

For CPU-only development, or a clone without GaussianWrapping:

- use `uv sync --locked --no-group cuda-extensions`;
- add `--no-group cuda-extensions` to `uv run` too.

A default sync includes the three CUDA builds and requires the
GaussianWrapping checkout and supported compiler/toolkit. See
[build inputs and stale-wheel checks](docs/rtx_environment.md#build-and-check-the-environment).

### 4. Preview the profile's installation

The [release profile installer](#release-evaluation) installs what one profile needs.

```bash
.venv/bin/real2sim-download-release \
  --profile config/release/fr3/flat/a_room_cup_cup1trim.yaml --dry-run
```

This writes nothing. Success lists each Hugging Face folder with its tree SHA-256, size and
destination:

- the `flat_a_room` scene;
- the `pi05_cup` checkpoint;
- the cup/plate and FR3 collision meshes.

It also prints no `NOT DOWNLOADABLE` line.

### 5. Download and install the inputs

```bash
.venv/bin/real2sim-download-release \
  --profile config/release/fr3/flat/a_room_cup_cup1trim.yaml
```

Both release repositories are gated. Before the first download:

1. log in with `.venv/bin/hf auth login` (or set `HF_TOKEN`);
2. accept the terms on each repository page (see [Release evaluation](#release-evaluation)).

The installer:

- verifies each downloaded folder against its pinned tree hash;
- installs real files;
- writes `REAL2SIM_INPUTS_ROOT` to `.env`.

The PaliGemma tokenizer is not redistributed. On first use, OpenPI fetches it anonymously from
`gs://big_vision/paligemma_tokenizer.model` and caches it below `OPENPI_DATA_HOME`. The runner
sets `OPENPI_DATA_HOME` to `<repo>/runtime-cache/openpi`, which is git-ignored. If the GPU run
has no network access, fetch the tokenizer there first:

```bash
OPENPI_DATA_HOME="$PWD/runtime-cache/openpi" .venv/bin/python -c \
  'from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer()'
```

Success creates `runtime-cache/openpi/big_vision/paligemma_tokenizer.model`.

### 6. Activate the environment

```bash
source .venv/bin/activate
```

Success makes the installed `real2sim-*` entry points available in this shell.

### 7. Resolve the profile without touching runtime files

```bash
real2sim-run --config config/release/fr3/flat/a_room_cup_cup1trim.yaml --dry-run --format table
```

Success prints:

- the `flat_a_room` scene path;
- the checkpoint/config pair;
- ten expected rollouts;
- GPUs;
- ports;
- the output root.

JSON is the default when `--format table` is omitted; it reports
`"reproducibility_mode": "production"` for this profile.

### 8. Verify every runtime input

```bash
real2sim-run --config config/release/fr3/flat/a_room_cup_cup1trim.yaml --check-runtime
```

Continue only when the JSON result contains `"status": "ok"`. It verifies:

- the checkpoint;
- Python;
- scene bytes;
- repository assets;
- dependency paths;
- artifact digests.

It also runs the production preflight and doctor:

- clean Git trees;
- the locked environment;
- the Gaussian CUDA extensions;
- the GPU binding;
- FFmpeg.

A failure names the check that failed.

### 9. Run ten rollouts

```bash
real2sim-run --config config/release/fr3/flat/a_room_cup_cup1trim.yaml
```

Success means exit `0` after:

- all ten rollouts are summarized;
- videos are finalized;
- the grid is built.

Add `--timestamp YYYYMMDD_HHMMSS` to all three launcher commands to fix the run name.

### 10. Confirm the environment check passed

The check passes when:

- steps 8 and 9 exit `0`, so the production preflight and doctor passed;
- `run_manifest.json` has `"mode": "production"` and its `preflight` field holds the doctor
  report, with every check `"status": "ok"`;
- the run root holds the outputs listed under [Read the result](#read-the-result), with ten
  rollout directories, ten videos, and `rollout_grid_10.mp4`;
- `rollout_success_summary.json` reports `new_final_success_count` of 2 or more. In this room
  the profile scored 5 of 10 rollouts in one simulator run at 300 frames (Wilson 95 % interval
  24–76 %; conditions under [Release evaluation](#release-evaluation)).

The success-count condition separates a working installation from one that almost never
succeeds. It does not detect a partial loss, such as an installation that keeps only half of its
successes. For ten rollouts and a pass at 2 or more successes (binomial):

| True success rate | Passes | Fails |
| --- | ---: | ---: |
| 50 % (the recorded rate) | 98.9 % | 1.1 % |
| 40 % | 95.4 % | 4.6 % |
| 30 % | 85.1 % | 14.9 % |
| 23.7 % (the interval's lower end) | 72.4 % | 27.6 % |
| 20 % | 62.4 % | 37.6 % |
| 10 % | 26.4 % | 73.6 % |

A correct installation therefore fails in about 1 % of runs at the recorded rate, and fewer than 2
successes points to a setup problem. A pass does not show that the installation reproduces the
recorded rate: one whose true rate has fallen to 30 %, 20 % or 10 % still passes in 85 %, 62 % or
26 % of runs.

Whether two GPUs of different models give bit-identical rollouts has not been established, so
compare success counts, not output bytes.

## Read the result

The run root is `outputs/release/release-flat-a-room-cupplate-cup1trim_<timestamp>/`. Its main contents are:

```text
franka_flat/run_01 ... run_10/  one directory per rollout
  logs/sim_log.jsonl            every saved simulation frame
  RUN_COMMAND.md                exact worker invocation
  franka_flat_run_NN.mp4        canonical third-person video
rollout_success_summary.json    aggregate result (+ .csv and .png)
rollout_grid_10.mp4             ten-rollout grid (+ poster and JSON)
run_manifest.json               pinned inputs, seeds, and GPU assignment
resolved_rollout_config.json    fully resolved plan
```

Wrist camera model:

- Simulated wrist cameras default to `elp-mean-wrist-rgb`, the mean board calibration of 9 ELP wrist
  units.
- An alternative pinhole camera model, `franka-pinhole-wrist-rgb`, can be selected with
  `simulation.parameters.policy_yubi_render_profile`
  ([Wrist camera model](docs/configuration.md#wrist-camera-model)).

Wrist rendering backend:

- Simulated wrist cameras render with `simulation.parameters.policy_wrist_backend: direct_kb4` by default:
  one KB4 fisheye view per wrist from MuJoCo and from the 3DGS server.
- The server needs the KB4 rasterizer of the [GaussianWrapping](https://github.com/airoa-org/GaussianWrapping)
  submodule, so rerun `scripts/bootstrap.sh` on an existing checkout.
- `direct_kb4` is the only backend.
- A rerender refuses a run that records no wrist backend
  ([Runs without a supported backend](docs/direct_kb4_development.md#runs-without-a-supported-backend)).

In `rollout_success_summary.json`:

- `official_success_field` names the authoritative per-rollout field.
- `new_final_success_count` and `new_final_success_rate` are the run-wide result.

The launcher exits non-zero if it summarizes fewer rollouts than `runner.expected_rollouts`.

## Post-hoc rendering

Profiles with `runner.policy_batch_size` of 10 or more omit live third-person rendering during
policy inference; wrist observations still render. Before success, the runner then:

1. replays every `logs/sim_log.jsonl`;
2. validates videos;
3. builds the grid with `config/render_templates/rollout_grid_policy_eval_4x3.json`.

Set `render.third_person: false` to omit videos and the grid.

The release profiles run two workers at a time, so each:

1. renders third-person video during the rollouts;
2. replaces each video's wrist insets with the saved policy observations;
3. builds the grid with `config/render_templates/rollout_grid_clear_headers_4x3.json`.

To retry interrupted post-hoc finalization, replace `RUN_NAME` and run:

```bash
python -m robosuite_env.tools.rerender_third_person outputs/RUN_NAME --gpu 0 --grid-template config/render_templates/rollout_grid_policy_eval_4x3.json
```

Success restores each MP4 and the grid. Replay logs go under `rerender/` and never overwrite policy
artifacts. Useful flags include:

- `--runs run_01,run_02`
- `--skip-grid`
- camera overrides
- `--wrist-insets saved|off`
- `--saved-wrist-display raw224|aspect-fit`

To rebuild only the `franka_flat` grid of the environment check:

```bash
python -m robosuite_env.tools.make_rollout_grid_video outputs/RUN_NAME --output outputs/RUN_NAME/rollout_grid_10.mp4 --scene franka_flat --template config/render_templates/rollout_grid_clear_headers_4x3.json --expected-frames 300
```

Success writes the MP4, poster, and grid JSON at the requested output path.

## Swap the checkpoint

Change the checkpoint, its matching OpenPI config, and its artifact binding together:

```yaml
policy:
  checkpoint: "{inputs_root}/checkpoints/CHECKPOINT_NAME"
  config_name: MATCHING_OPENPI_CONFIG
  config_source: deploy
reproducibility:
  artifacts:
    checkpoint: MATCHING_ARTIFACT_ID
```

The last token of a release profile name selects its checkpoint; both release profiles end in
`cup1trim`:

| Profile token | Checkpoint | OpenPI config | Artifact id |
|---|---|---|---|
| `cup1trim` | `checkpoints/pi05_cup` | `pi05_cup` | `pi05-cup` |

For a replacement:

- It needs `assets/<asset_id>/norm_stats.json` for its OpenPI config.
- Repeat steps 7 through 9; mismatched paths, bytes, configs, or stats fail early.

## Swap the scene

Each profile binds one scene. The FR3 Flat cup profile `config/release/fr3/flat/reading_room_cup_cup1trim.yaml`,
for example, binds:

```yaml
scenes:
  - tag: franka_flat
    path: "{inputs_root}/scenes/flat_reading_room"
reproducibility:
  artifacts:
    scene.franka_flat: public-scene-flat-reading-room-v2
```

- The tag names the rig's scene slot and equals `background.json`'s `name`.
- The path selects the room.

The environment check, `a_room_cup_cup1trim.yaml`, binds the path `{inputs_root}/scenes/flat_a_room`
and the artifact `public-scene-flat-a-room-v2` instead.

Install and run that profile the same way, in steps 4 through 9. Validate an installed scene (or
replace its path):

```bash
python -m robosuite_env.contracts.scene validate local-inputs/scenes/flat_a_room
```

Success prints `id`, `contract_digest`, `calibration_table_z_m`, and `gs_table_plane_z_m`. The tag
must equal `background.json`'s `name`. Do not modify installed scene trees.

## Troubleshooting

| Message | Action |
| --- | --- |
| `[host-check] DECISION: STOP` | Do not bootstrap. Return the `COPY_REPORT_BEGIN..COPY_REPORT_END` block, fix every blocker, and rerun the host gate. |
| `expected uv 0.12.3, found ...` | Install exact `uv 0.12.3`; the requirement is not a minimum. |
| `CUDA_HOME environment variable is not set` | Select CUDA 12.8 with `CUDA_HOME` or the host's environment manager. `--skip-cuda-extensions` is only for CPU contract tests. |
| `third_party/GaussianWrapping is at ..., but the build expects ...` | Rerun `scripts/bootstrap.sh`, because it checks out the pinned GaussianWrapping before the extension build. |
| `REAL2SIM_GPU_IDS is retired` | The shell exports `REAL2SIM_GPU_IDS` (even empty), or the profile names or references it; the launcher does not read it. Unset it and remove it from the profile. Use `CUDA_VISIBLE_DEVICES` to limit visible host GPUs and YAML `runtime.*_cuda_devices` fields to select zero-based visible slots. |
| `PORTABLE_POLICY_SERVER is retired` | The shell sets `PORTABLE_POLICY_SERVER` to a non-empty value, or the profile names or references it. This repository has no remote policy bridge, so the run would evaluate the local checkpoint instead of the requested remote policy. Unset it and remove it from the profile. |
| `required environment variables are unset: REAL2SIM_INPUTS_ROOT` | Rerun the installer, or set an absolute, non-symlink input root in `.env`. |
| `scene contract failed for '<tag>': ...` | Validate the scene directory. Do not bypass the runtime gate. |
| `artifact <id> SHA-256 mismatch: ...` | Restore the installed bytes, or deliberately update the artifact record for rebuilt content. |
| `artifact role <role> resolves to X, but the config uses Y` | Make the profile path and `reproducibility.artifacts` binding refer to the same input. |
| `norm stats required by OpenPI config '<name>' were not found: ...` | Use the matching checkpoint/config pair and preserve the checkpoint's `assets/` subtree. |
| `renderer checkout commit changed after source validation` | Stop editing the checkout and rerun the post-hoc renderer. |
| `output.run_name already exists under output.root: ...` | Choose a different `--timestamp`, or change `output.run_name`. |

For all lower-level worker options, run `.venv/bin/python -m robosuite_env.sim --help-all`.

## Release evaluation

The two release profiles live in `config/release/<robot>/<rig>/<room>_<task>_<checkpoint>.yaml`:

| Profile | Room | Mode |
| --- | --- | --- |
| `config/release/fr3/flat/a_room_cup_cup1trim.yaml` | A Room | `production` (the environment check) |
| `config/release/fr3/flat/reading_room_cup_cup1trim.yaml` | Reading Room | `fixture` |

Both put a cup on a plate with the FR3 Flat rig and the `pi05_cup` checkpoint. Each runs in one of
two public CC BY 4.0 3DGS rooms from SuperSplat, registered to the FR3 Flat rig
(see [`assets/public_scenes/`](assets/public_scenes/README.md)).

Only the environment check records a success count, in
[step 10](#10-confirm-the-environment-check-passed):

- one ten-rollout simulator run of that profile with its settings
- on one 16 GB GPU with `runner.policy_batch_size: 2` and `reproducibility.mode: fixture`
- success is the run summary's `new_final_success`

Step 10 of the environment check derives its pass rule from that count. The Reading Room profile
records no success count.

Both release profiles run ten rollouts of 300 frames:

- `runner.frames: 300` (30 s at `frame_hz: 10`)
- `runner.max_wall_sec: 7200`

Their files in this repository:

- Small per-condition files: [`assets/release/`](assets/release/README.md)
- Initial object positions: [`assets/initpose/`](assets/initpose/README.md) (`<robot>/<scene>/<task>.json`)
- Start poses: [`assets/initial_poses/`](assets/initial_poses/README.md)

The large inputs are extracted folders in two gated Hugging Face repositories:

| Repository | Folders the release profiles read | Licence |
| --- | --- | --- |
| [`airoa-org/yubi-real2sim-policies`](https://huggingface.co/airoa-org/yubi-real2sim-policies) (model) | `pi05_cup/`: the policy checkpoint | Gemma Terms of Use |
| [`airoa-org/yubi-real2sim-assets`](https://huggingface.co/datasets/airoa-org/yubi-real2sim-assets) (dataset) | `scenes/flat_a_room/`, `scenes/flat_reading_room/` (the 3DGS rooms), `assets/trellis_cup_plate/`, `assets/fr3_collision/` (cup/plate and FR3 meshes) | rooms CC BY 4.0; AIRoA's `background.json` and `table_collision/` Apache-2.0 |

1. Log in with `.venv/bin/hf auth login` (or set `HF_TOKEN`).
2. Accept the terms on both pages; access is approved automatically.

The cup/plate meshes and the FR3 collision meshes are not tracked in this repository. The
installer below reads each repository at the revision pinned in
`artifacts/release_downloads.json` and installs the folders a profile needs.
Its `runtime_bundles` mapping lists the archives for each runtime bundle;
an empty list means the bundle's payloads are tracked in Git. Bundle selection
uses the same robot and `task.objects` selector as runtime preflight. Profiles
requiring unpublished bundles (currently `box_pack`) fail before any download.

Select the same profile for installation and execution:

```bash
.venv/bin/real2sim-download-release \
  --profile config/release/fr3/flat/reading_room_cup_cup1trim.yaml --dry-run
.venv/bin/real2sim-download-release \
  --profile config/release/fr3/flat/reading_room_cup_cup1trim.yaml
real2sim-run --config config/release/fr3/flat/reading_room_cup_cup1trim.yaml --check-runtime
```

This downloads the profile's checkpoint, scene and required meshes at the
pinned revisions. It:

- verifies each folder against its pinned tree hash
- installs real files
- writes `REAL2SIM_INPUTS_ROOT` to `.env`

Options:

- Repeat `--profile` to install both profiles into one root; shared inputs are verified and reused.
- `--inputs-root` selects a different root for checkpoints and scenes (meshes always install under
  `robosuite_env/assets/`).

Existing differing files are refused.

The scene tag is the rig's; the installed directory is the room's:

| Profile | Scene tag | Installed scene under `REAL2SIM_INPUTS_ROOT/scenes/` |
| --- | --- | --- |
| `fr3/flat/reading_room_cup_cup1trim.yaml` | `franka_flat` | `flat_reading_room` |
| `fr3/flat/a_room_cup_cup1trim.yaml` | `franka_flat` | `flat_a_room` |

The installer installs the scene named by the profile's scene
path, from the pins in
[`artifacts/release_downloads.json`](artifacts/release_downloads.json).
Manual downloads remain supported as described in the dataset repository's
`README.md` and [`assets/release/README.md`](assets/release/README.md).

## Known limitations

Known constraints of this release, each with the place that documents it:

- **The renderer is non-commercial.** The code is Apache-2.0, but every profile renders through
  GaussianWrapping code under the Inria/MPII Gaussian-Splatting License, which allows
  non-commercial research and evaluation use only ([License](#license),
  [`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES) B.1).
- **Two runnable profiles.** Only the two FR3 Flat cup-on-plate profiles are released
  ([Release evaluation](#release-evaluation)). The code also accepts `robot: fr3_duo` and
  `task.objects: box_pack`, but the release downloads include no Duo scene and no `box_pack`
  checkpoint or object meshes ([`artifacts/release_downloads.json`](artifacts/release_downloads.json),
  [Adding a task](docs/configuration.md#adding-a-task)), so neither has public inputs to run with.
- **One recorded success count.** Only the environment check has one: 5 of 10 rollouts in one
  simulator run at 300 frames ([step 10](#10-confirm-the-environment-check-passed)). The Reading
  Room profile has no recorded count to compare a run with.
- **Hardware.** The GPU runtime needs x86-64 Linux and an NVIDIA CUDA GPU
  ([Host prerequisites](#host-prerequisites)). The release profiles use one GPU (slot `0`) unless
  a local copy of the profile sets more. Their settings, including `runner.policy_batch_size: 2`,
  target one 16 GB GPU, and the recorded count was measured on one.
  An RTX 5070 with 12 GB is experimental, and the host gate makes no VRAM capacity promise
  ([VRAM and reporting](docs/rtx_environment.md#vram-and-reporting)). CPU checks do not show that
  the CUDA kernels, EGL or a rollout work on a particular GPU
  ([RTX environment verification](docs/rtx_environment.md)).
- **Bit-identical output is not established.** The profiles fix `policy.noise_seed` and
  `simulation.seed` and set `policy.deterministic_xla` and `simulation.deterministic`, but whether
  two GPUs of different models give bit-identical rollouts has not been established; compare
  success counts, not output bytes ([step 10](#10-confirm-the-environment-check-passed)).
- **Downloads.** Both Hugging Face repositories are gated: log in and accept the terms on each
  repository page before the first download ([step 5](#5-download-and-install-the-inputs)). The
  PaliGemma tokenizer is not in them: OpenPI fetches it anonymously on first use, so a GPU host
  without network access needs it fetched beforehand (also step 5).
- **CI runs no rollout.** CI runs on GitHub-hosted runners without a GPU and deselects tests
  marked `gpu`, `heavy` or `assets`, apart from the MuJoCo CPU wrist regressions
  ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). No job downloads the release inputs,
  renders 3DGS or runs a rollout, so a change can break the rollout path while CI passes.
- **Some settings still come from environment variables.** The worker reads some argument
  defaults from its environment, for example `POLICY_POSE_REPR`
  (`robosuite_env/rollout/cli.py`). `real2sim-run` passes the runner only the names in the
  profile's `reproducibility.environment.allowlist` and the values it derives from the profile,
  so a shell value of such a name reaches a launcher run only if the profile allowlists that
  name. The launcher also refuses to start while the shell exports one of a fixed list of
  behaviour-changing names (`_BEHAVIOR_ENV_NAMES` in `src/real2sim/config/keys.py`) or any `RS_`
  name that the profile does not allowlist. A worker or `scripts/run_policy_batch.sh` started
  directly has no such filter and takes these defaults from the shell.
- **GaussianWrapping is imported from its checkout.** It is not an installable package, and its
  top-level `scene` package clashes with this repository's. The runner therefore starts
  `scene/gs_background_server.py` as a file, and the server adds the profile's
  `runtime.gaussian_wrapping_repo` checkout (`third_party/GaussianWrapping` in the release
  profiles) to `sys.path` before it imports the renderer
  ([`scene/gs_background_server.py`](scene/gs_background_server.py),
  [Direct KB4 development](docs/direct_kb4_development.md)).

## Lint checks

The CI `lint` job runs both checks with the settings in `pyproject.toml`.
Use the versions that CI pins, because another version can report different results:

```bash
uvx ruff@0.17.0 check --no-fix . robosuite_env/rollout/build
uvx vulture@2.16  # unused code in src/, robosuite_env/, scene/, scripts/
```

- vulture does not scan `tests/`, so code that only tests call is reported as unused.
- When something outside the scanned code reads a definition (a registry, a third-party
  object, a shell caller), mark only that line with vulture's code and name the reader:
  - Example: `def do_GET(self):  # noqa: V105  # BaseHTTPRequestHandler hook`.
  - Codes: V101 attribute, V102 class, V103 function, V105 method, V106 property, V107 variable.
  - On a decorated definition, the mark goes on the first decorator line.
- Delete the code instead when nothing reads it.
- CI also fails:
  - when a mark hides no finding, for example after the definition moves or scanned
    code starts to read it;
  - when vulture does not apply a V code.

  The check cannot see the outside reader, so review the mark when that reader changes.
  Run it with `uv run --no-project --with vulture==2.16 python scripts/check_vulture_marks.py`.
- CI fails on an unmarked finding that `scripts/vulture_baseline.txt` does not list.
  - The baseline lists accepted existing findings, by file and message without line numbers.
  - The check prints entries that no longer occur without failing; remove them when you can.
  - Never add an entry for a new finding.

  Run it with
  `uv run --no-project --with vulture==2.16 python scripts/check_vulture_baseline.py`.

## Citation

Paper: [YUBI-Real2Sim: 3DGS Scene Verification for Reliable Policy Evaluation](https://arxiv.org/abs/2610.17521).
If you use this repository, please cite it (also in [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{nishizawa2026yubi,
  title={YUBI-Real2Sim: 3DGS Scene Verification for Reliable Policy Evaluation},
  author={Nishizawa, Hiroki and Ohkawa, Takehiko and Scianca, Nicola and Matsushima, Tatsuya and Morishima, Shigeo and Ota, Kei},
  journal={arXiv preprint arXiv:2610.17521},
  year={2026}
}
```

The simulation runs on [robosuite](https://robosuite.ai); please also cite it as its authors ask:

```bibtex
@inproceedings{robosuite2020,
  title={robosuite: A Modular Simulation Framework and Benchmark for Robot Learning},
  author={Yuke Zhu and Josiah Wong and Ajay Mandlekar and Roberto Mart\'{i}n-Mart\'{i}n and Abhishek Joshi and Soroush Nasiriany and Yifeng Zhu and Kevin Lin},
  booktitle={arXiv preprint arXiv:2009.12293},
  year={2020}
}
```

## License

The code in this repository is licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
(see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)).

**The 3DGS renderer is non-commercial, so the full pipeline cannot be used commercially.**
Every profile renders its background through `third_party/GaussianWrapping`. Its rasterizers,
`simple-knn` and the `gaussian_wrapping` modules the GS server imports are under the
Inria/MPII Gaussian-Splatting License, which allows non-commercial research and evaluation
use only. The pinned fork commits change those rasterizer files and are under the same
licence. The Apache License 2.0 of this repository's code does not change that
([`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES) B.1).

Third-party components keep their own licences, listed in
[`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES) with the copied files and their changes.
In particular:

- **Copied code keeps its licence.** Parts of `robosuite_env/policy/yubi_openpi/yubi_policy.py`
  and `robosuite_env/policy/rf_action_inpainting.py` are adapted from
  [openpi](https://github.com/Physical-Intelligence/openpi) (Apache License 2.0), and
  `robosuite_env/assets/fr3_visual/` is converted from
  [franka_description](https://github.com/frankarobotics/franka_description) (Apache License 2.0).
- **The YUBI gripper meshes are open hardware.** `robosuite_env/assets/yubi_left/`,
  `yubi_right/`, `fr3_yubi_right_registered/` and `yubi_guard_collision_manifest.json` are
  derived from the YUBI hardware design,
  [Toyota/yubi-hw](https://github.com/Toyota/yubi-hw) (Copyright 2026 Toyota Motor Corporation),
  and stay under the [CERN Open Hardware Licence Version 2 - Weakly Reciprocal](https://ohwr.org/cern_ohl_w_v2.txt)
  (CERN-OHL-W-2.0), not under the Apache License 2.0. `THIRD_PARTY_NOTICES` A.4 lists the files
  and the changes.
- **Copyleft packages are installed, not redistributed.** The locked environment installs some
  GPL/LGPL packages, among them `plyfile`, which the scene contract imports, and a GPL build of
  FFmpeg that runs as a separate process. Whoever redistributes a built environment or container
  must comply with their licences.

Downloaded inputs are not covered by `LICENSE`; each keeps its own licence:

| Download | Contents | Licence |
| --- | --- | --- |
| `pi05_cup` | Policy checkpoint fine-tuned from openpi pi0.5 (PaliGemma) | [Gemma Terms of Use](https://ai.google.dev/gemma/terms); use is restricted by the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) |
| `fr3_collision` | Unmodified franka_description FR3 collision meshes | Apache License 2.0 (Franka Robotics GmbH) |
| `trellis_cup_plate` | Cup and plate meshes made by AIRoA with TRELLIS.2 (which uses DINOv3 and nvdiffrast), CoACD and trimesh from AIRoA's photographs, background-removed with BRIA RMBG-2.0 (model licence CC BY-NC 4.0; see `THIRD_PARTY_NOTICES` C.3) | Apache License 2.0 (AIRoA) |

The public 3DGS rooms adapted from SuperSplat captures (the `flat_reading_room` and `flat_a_room`
scenes) carry two licences:

- **The rooms** stay under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); each
  archive's `ATTRIBUTION.txt` names the creator and the changes.
- **What AIRoA adds to a room**, its `background.json` registration and the `table_collision/`
  simulator table box, is AIRoA's work under the Apache License 2.0.

The PaliGemma tokenizer is not redistributed; openpi downloads it on first use under the
Gemma Terms of Use.

## Acknowledgements

The simulated gripper is the YUBI Gripper, from the open-source YUBI hardware design by
Frontier Research Center, Toyota Motor Corporation
([Toyota/yubi-hw](https://github.com/Toyota/yubi-hw), CERN-OHL-W-2.0). If you use the YUBI
gripper model, please also cite:

T. Ohkawa et al., "YUBI: Yielding Universal Bidigital Interface for Bimanual Dexterous
Manipulation at Scale", arXiv:2606.10244, 2026. <https://arxiv.org/abs/2606.10244>

This project is based on the YUBI open-source hardware project. Toyota Motor Corporation does not
endorse or certify this project.

## Where to go next

| Intent | Who it is for | Read |
| --- | --- | --- |
| Run without installing the toolchain on the host | Users whose host lacks the CUDA 12.8 toolkit, GCC/G++ 13 or uv 0.12.3 | [Container environment](docs/container.md) |
| Change what an experiment does | Profile authors changing checkpoints, scenes, rollout counts, timing, GPUs, rendering, or controllers | [Configuration](docs/configuration.md) |
| Understand the code layout | Contributors finding component boundaries and file ownership | [Architecture](docs/architecture.md) |
| Operate or reproduce a strict run | Operators validating, running, rerendering, or comparing production evaluations | [Simulation operations](docs/simulation_operations.md) |
| Audit assets and distribution | Maintainers installing, verifying, or packaging runtime inputs | [Assets](docs/assets.md) |
| Contribute or report a vulnerability | Contributors and security researchers | [Contributing](CONTRIBUTING.md), [Security policy](SECURITY.md) |
