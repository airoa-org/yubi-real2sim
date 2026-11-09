# Container environment

[`docker/Dockerfile`](../docker/Dockerfile) builds the
[host prerequisites](../README.md#host-prerequisites) as an image, for hosts that
have an NVIDIA driver but not the CUDA 12.8 toolkit, GCC/G++ 13 or uv 0.12.3.
The image provides only that toolchain:

| Component | Version in the image |
| --- | --- |
| Base | `nvidia/cuda:12.8.1-devel-ubuntu24.04`, pinned by digest |
| CUDA toolkit | 12.8 (`nvcc` V12.8.93) at `/usr/local/cuda-12.8` |
| GCC/G++ | 13 (Ubuntu 24.04 packages) |
| uv | exactly 0.12.3, copied from `ghcr.io/astral-sh/uv:0.12.3`, pinned by digest |
| Other tools | Git, curl, system Python 3 (for `check_host.sh`), glibc 2.39 |
| Rendering | GLVND `libEGL`/`libGL` for `MUJOCO_GL=egl`; the NVIDIA runtime adds the driver's EGL library |

The image contains no repository content. You mount a checkout into the
container and run the README steps there. Bootstrap still installs CPython,
the locked packages, the three Gaussian CUDA extensions and FFmpeg into the
checkout's `.venv`, exactly as on a host.

## Host requirements

- x86-64 Linux with an NVIDIA driver 570.26 or newer and a CUDA GPU; see
  [Host prerequisites](../README.md#host-prerequisites) for GPU memory.
- Docker with the NVIDIA Container Toolkit, so that `docker run --gpus` works.
- About 55 GB of free disk for the environment check. The image is about 15 GB, `.venv` about
  11 GB, the container `HOME` (`$STATE` below) about 13 GB, the inputs about 14 GB, and the
  checkout with one run's outputs about 1 GB. The image sets `UV_LINK_MODE=copy`, so `.venv`
  is a copy and does not share space with uv's cache in `HOME`.
- Network access from the container: bootstrap downloads Python packages and
  clones the pinned GaussianWrapping and franka_description sources, and the
  installer downloads the release inputs from Hugging Face.

## Build the image

From the checkout root:

```bash
docker build -t real2sim-env:cuda12.8 docker/
```

The build context is `docker/`, which holds only the Dockerfile, so nothing
else from the checkout enters the image.
Success ends with the image tagged `real2sim-env:cuda12.8`.

## Start a shell in the container

From the checkout root, start the container and run every README step from
step 2 on in it:

```bash
STATE="$HOME/.cache/real2sim-container"
mkdir -p "$STATE"
docker run --rm -it --init \
  --gpus device=0 \
  --user "$(id -u):$(id -g)" \
  -e HOME="$STATE" -v "$STATE:$STATE" \
  -v "$PWD:$PWD" -w "$PWD" \
  --memory 22g --memory-swap 22g --shm-size 8g \
  real2sim-env:cuda12.8 bash
```

- `--gpus device=0` passes one host GPU. Inside the container it is visible
  GPU slot `0`, which the release profiles use. For the two-GPU setup in the
  README, pass `--gpus '"device=0,1"'`.
- `--user` runs as your own user, so files in the checkout stay yours.
  Do not run the container as root: the checkout would collect root-owned
  files, and Git refuses to work in a checkout that another user owns.
- `HOME` is a directory on the host that persists across containers. It holds
  uv's cache, the CPython that bootstrap installs and `.venv` links to, and
  the Hugging Face login from `.venv/bin/hf auth login`.
- The checkout is mounted at the same absolute path as on the host, so paths
  in `.env`, `run_manifest.json` and logs name the same location on both
  sides. Keep `REAL2SIM_INPUTS_ROOT` inside a mounted directory; the default
  `<repo>/local-inputs` is. To use another inputs root, mount it at the same
  path with another `-v DIR:DIR`.
- `.venv` links to the CPython under the container's `HOME`, so use it only
  inside this container setup, not directly on the host.

## Run the README steps

Inside the container, follow [From clone to a finished rollout](../README.md#from-clone-to-a-finished-rollout)
from step 2 without changes:

```bash
./scripts/check_host.sh
scripts/bootstrap.sh
.venv/bin/real2sim-download-release \
  --profile config/release/fr3/flat/a_room_cup_cup1trim.yaml
source .venv/bin/activate
real2sim-run --config config/release/fr3/flat/a_room_cup_cup1trim.yaml --check-runtime
real2sim-run --config config/release/fr3/flat/a_room_cup_cup1trim.yaml
```

Each command has the success condition the README gives for its step; for example,
`check_host.sh` must print `DECISION: PROCEED`. Run `.venv/bin/hf auth login` (or pass
`-e HF_TOKEN`) inside the container before the download.

## Memory

`--memory` and `--memory-swap` cap the container, so a job that runs out of
memory is stopped inside the container instead of exhausting the host.
Docker counts the container's file cache against the cap, so a run can reach the
cap and continue.

| Step | Cap that completed | Notes |
| --- | --- | --- |
| `scripts/bootstrap.sh` (CUDA extension build) | 16 GiB | default `MAX_JOBS=4`; lower `MAX_JOBS` on a smaller cap |
| Environment check, ten rollouts | 20 to 22 GiB | two rollout workers on one 16 GB RTX 5070 Ti; memory use reached the cap (file cache included) without an out-of-memory kill |

These figures come from the hosts the release was checked on, not from a
lower bound; see also [VRAM and reporting](rtx_environment.md#vram-and-reporting).
Leave the host enough memory outside the cap for the desktop and other users.
The validated runs used `--shm-size 8g`; Docker's default of 64 MB has not
been tested.
