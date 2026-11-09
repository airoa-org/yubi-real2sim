# RTX 50 environment verification

Use a fresh clone/worktree and its own `.venv`. The runtime is pinned to CUDA
12.8, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, GCC/G++ 13, and driver
570.26 or newer. The values live in `config/support-matrix.yaml`.
The [PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/)
and [CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
document this wheel combination and the toolkit's driver requirement.

CPU checks verify the locked packages; they do not establish that native
CUDA kernels, EGL, or a full rollout work on a particular GPU. Report those
results separately using the steps below.

## Build and check the environment

Select CUDA 12.8 and GCC/G++ 13 in the current shell. `CUDA_HOME` takes
precedence over `nvcc` on `PATH`; `/usr/local/cuda-12.8` is the final fallback.
Bootstrap exports that toolkit as `CUDA_HOME` for the extension build. Before
a direct `uv sync` or `uv run`, export `CC`, `CXX`, and `CUDA_HOME` for the same
toolchain. The wheel hook rejects a compiler or toolkit that disagrees with
the support matrix, so a later sync cannot replace a valid wheel with a system
GCC 11 build. For example, on a host with versioned compiler commands:

```bash
export CC=gcc-13 CXX=g++-13 CUDA_HOME=/usr/local/cuda-12.8
```
Build the declared environment without bootstrap:

```bash
git submodule update --init third_party/GaussianWrapping
uv sync --locked
```
System Python 3.9+ reads the JSON-compatible support-matrix YAML without
installing anything. The runtime interpreter remains pinned by `.python-version`.

```bash
set -o pipefail
mkdir -p outputs/rtx-environment
git rev-parse HEAD | tee outputs/rtx-environment/commit.txt
./scripts/check_host.sh 2>&1 | tee outputs/rtx-environment/host.log
```

Proceed only on `DECISION: PROCEED`. Keep the printed GPU memory and process
inventory: it distinguishes toolchain failures from GPU contention.

```bash
scripts/bootstrap.sh 2>&1 | tee outputs/rtx-environment/bootstrap.log
uv sync --frozen --check --inexact
uv pip check --python .venv/bin/python
.venv/bin/python src/real2sim/support_matrix.py check-torch
git submodule status --recursive
git -C third_party/GaussianWrapping submodule status --recursive
```

Bootstrap initializes the GaussianWrapping submodule automatically. It does
not fetch GaussianWrapping's own nested submodules, because nothing builds or
imports them, so the second status command lists them with a leading `-`. A
production run records them but does not require them. OpenPI
and robosuite are not cloned: `uv sync` installs them from the commits
`uv.lock` pins, and the launcher and worker refuse any other copy. Bootstrap installs
source dependencies without optional LFS media; runtime assets are installed
separately by `real2sim-download-release` from the pinned Hugging Face folders.
It validates both CUDA wheel variants. `uv sync` builds the three Gaussian
extensions of the `cuda-extensions` dependency group for the architectures in
`TORCH_CUDA_ARCH_LIST`, or for every visible GPU when it is unset. Set it
(`12.0` on an RTX 5070 Ti) when no GPU is visible, for example in an image
build. The project's `concurrent-builds = 1` applies to direct uv commands too.
The wheel hook defaults `MAX_JOBS` to 4 and accepts an explicit override.
`third_party/GaussianWrapping` must be a real directory, because the build
finds `scripts/` relative to its resolved path.

For CPU-only development, or a clone without GaussianWrapping, omit the CUDA
group explicitly. This installs the project's CUDA torch wheel but does not
compile the three extensions or require a local CUDA toolkit:

```bash
uv sync --locked --no-group cuda-extensions
uv run --locked --no-group cuda-extensions python -m pytest tests/distribution
```

The `third_party/GaussianWrapping` gitlink is the GaussianWrapping pin. Bootstrap
checks out that commit, and the wheel hook refuses any other checkout. The
`REAL2SIM_BUILD_INPUTS` digest covers the gitlink, so a pin change must also
update that literal, which makes uv rebuild the extensions.

Each wheel contains `_real2sim_build.json` beside its native module. It records
the source revision and tracked source bytes, the adapter/header/profile,
locked and installed torch, compiler/toolkit paths and versions, architectures,
build variables, and native binary hashes. Doctor verifies these records before
launching kernels. That check compares the recorded compiler and `nvcc` versions
with the support profile instead of the run shell's tools, so the run host needs
no compiler or `nvcc`. A changed checkout or stale binary therefore stops preflight,
even when its CUDA kernels would run. Bootstrap checks these same records and
reinstalls mismatched extensions.

Inherited `NVCC_APPEND_FLAGS` are recorded and checked at build time, including an
empty value when unset. Changing these flags requires rebuilding the extensions.
Doctor checks the recorded flags, not the run shell's: release profiles do not pass
them and they do not change an installed binary.
Wheels without this recorded input must also be rebuilt.
`PYTORCH_NVCC` overrides are unsupported during builds because they bypass the selected
toolkit; unset the variable and select the toolkit with `CUDA_HOME` instead. Doctor
ignores this variable at runtime and validates the recorded toolkit.

uv's path-source cache does not track changing compiler binaries, architectures,
or edited extension sources. `REAL2SIM_BUILD_INPUTS` covers checked-in adapter
inputs; the distribution test derives its required value. For other input
changes, rebuild explicitly, then run doctor:

```bash
uv sync --locked --reinstall-package diff-gaussian-rasterization \
  --reinstall-package diff-gaussian-rasterization-ours --reinstall-package simple-knn
```

The wheel records and doctor's verification do not depend on bootstrap, so they
would still hold if bootstrap only ran `git submodule update` and `uv sync --locked`.
Removing bootstrap's repair step would not make uv detect these dynamic inputs:
the explicit rebuild command remains required. Doctor ignores
`TORCH_CUDA_ARCH_LIST` at runtime and checks kernel compatibility on the selected
GPUs instead of demanding the build's exact GPU visibility.

Check actual GPU execution, not just imports:

```bash
.venv/bin/python - <<'PY'
import torch
import diff_gaussian_rasterization
import diff_gaussian_rasterization_ours
import simple_knn._C

assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("compiled architectures:", torch.cuda.get_arch_list())
if torch.cuda.get_device_capability(0) == (12, 0):
    assert "sm_120" in torch.cuda.get_arch_list()
x = torch.ones((64, 64), device="cuda")
assert torch.all(x @ x == 64).item()
torch.cuda.synchronize()
print("torch GPU kernel and Gaussian imports: ok")
PY

JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python - <<'PY'
import jax
import jax.numpy as jnp

print(jax.devices())
assert jax.default_backend() == "gpu"
x = jnp.ones((64, 64))
assert bool(jnp.all((x @ x).block_until_ready() == 64))
print("JAX GPU kernel: ok")
PY

MUJOCO_GL=egl .venv/bin/python - <<'PY'
import mujoco

model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with mujoco.Renderer(model, height=64, width=64) as renderer:
    renderer.update_scene(data)
    assert renderer.render().shape == (64, 64, 3)
print("MuJoCo EGL: ok")
PY
```

## Inputs and a single-rollout smoke test

Follow the README's public-asset installation steps in this checkout. A local
`.env` and the repository asset files are needed even if another checkout has
already downloaded the release inputs. Do not copy a virtualenv or built extensions
from another worktree.

First check an unchanged release profile:

```bash
.venv/bin/real2sim-run --config config/release/fr3/flat/a_room_cup_cup1trim.yaml --check-runtime
```

`status: ok` establishes preflight success, not rollout completion. Create an
ignored local profile for a batch-size-one smoke run:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import yaml

source = Path("config/release/fr3/flat/a_room_cup_cup1trim.yaml")
profile = yaml.safe_load(source.read_text())
profile["name"] = "fr3-rtx-environment-smoke"
profile["runner"].update(repeats=1, policy_batch_size=1, expected_rollouts=1)
profile["render"]["enabled"] = False
profile["output"]["run_name"] = "fr3_rtx_environment_smoke_{timestamp}"
Path("config/local_run_rtx_smoke.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
PY
.venv/bin/real2sim-run --config config/local_run_rtx_smoke.yaml --dry-run --format table
.venv/bin/real2sim-run --config config/local_run_rtx_smoke.yaml --check-runtime
.venv/bin/real2sim-run --config config/local_run_rtx_smoke.yaml
```

This disables third-person videos because the checked-in grid templates start
at ten repeats; policy wrist observations still render with 3DGS. Record whether
initialization, policy inference, 3DGS wrist rendering, and finalization succeed.
The README's unchanged ten-rollout environment check additionally checks
third-person videos and the grid; report that run separately.

## VRAM and reporting

RTX 5070 12 GB is experimental. The environment check,
`config/release/fr3/flat/a_room_cup_cup1trim.yaml`, targets one dedicated
16 GB GPU such as an RTX 5070 Ti: it sets `runner.policy_batch_size: 2`,
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.45` and
`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`.
`config/release/fr3/flat/reading_room_cup_cup1trim.yaml`, the other release
profile, uses the same three settings. If a run runs out of GPU memory, lower
`runner.policy_batch_size` in a local profile under `config/` and report the
value.

A two-GPU host also runs one 3DGS server and all MuJoCo EGL contexts on
slot `0`. Set `runtime.gs_cuda_devices` and `runtime.sim_cuda_devices` to
`"0,1"` in a local profile to use both GPUs. Other processes on the GPU reduce
the memory available to the run.

Both release profiles use GPU slot `0` for the policy, 3DGS, and MuJoCo EGL;
all three contribute to memory use. The smoke profile above uses one
worker/inference batch with the same scene, checkpoint, and GPU slot.
Run on an available GPU and record `nvidia-smi` before and during execution.
Do not terminate other users' processes to obtain memory.

The launcher already sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` for workers.
Both release profiles set `XLA_PYTHON_CLIENT_MEM_FRACTION` and
`XLA_PYTHON_CLIENT_ALLOCATOR` in the profile. Lowering the fraction is not
evidence that the checkpoint and renderers will fit. If experimenting with it,
declare it in `reproducibility.environment.allowlist` and `values` in a local
profile and report the value, batch size, peak memory, and exact failure stage.
The host gate prints free memory/processes but deliberately makes no universal
VRAM capacity promise.

When you report a result in an issue, include the commit,
GPU/driver/toolkit/compiler, bootstrap log, torch/JAX/EGL results,
`--check-runtime` result, and any rollout result. For a failure, include the
first error and the relevant log; keep environment-build failures separate from
inference OOM and task scores.
