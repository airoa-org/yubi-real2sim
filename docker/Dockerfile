# Host prerequisites for Real2Sim as a container image: the toolchain that
# README.md ("Host prerequisites") and config/support-matrix.yaml require,
# and nothing else. The image holds no repository content. Bind-mount a
# checkout and run scripts/check_host.sh and scripts/bootstrap.sh inside the
# container; bootstrap installs CPython, the locked packages, the three
# Gaussian CUDA extensions and FFmpeg into the checkout's .venv.
# Build, run and memory guidance: docs/container.md.
#
#   docker build -t real2sim-env:cuda12.8 docker/

# Exact uv 0.12.3, the version check_host.sh and bootstrap.sh require.
FROM ghcr.io/astral-sh/uv:0.12.3@sha256:2d890623d310b57771ce840f0da5eed5fc6d657da05ffaa45d82797b53fa3abc AS uv

# CUDA toolkit 12.8 (nvcc 12.8.93) on Ubuntu 24.04 (glibc 2.39, GCC/G++ 13).
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Ubuntu 24.04 publishes one version of each package per pocket, so the
# release pins the major versions (GCC 13) and the base digest instead.
# - git, curl, ca-certificates: check_host.sh, and bootstrap's source clones.
# - python3: the system Python that check_host.sh reads the support matrix with.
# - gcc/g++ 13: the compilers check_host.sh accepts for the CUDA extensions.
# - libegl1, libgl1: the GLVND EGL/GL loaders for MUJOCO_GL=egl; the NVIDIA
#   container runtime adds the driver's EGL vendor library ("graphics" below).
# - libglib2.0-0t64: GLib, which the locked opencv-python wheel loads.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git python3 \
        gcc-13 g++-13 gcc g++ \
        libegl1 libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/* \
    && test "$(gcc -dumpversion | cut -d. -f1)" = 13 \
    && test "$(g++ -dumpversion | cut -d. -f1)" = 13

COPY --from=uv /uv /uvx /usr/local/bin/

# "graphics" makes the NVIDIA container runtime mount the EGL vendor library.
# uv's cache (under HOME) and the checkout's .venv are separate bind mounts,
# so hard links fail; copy instead of printing a fallback warning per package.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    UV_LINK_MODE=copy \
    LANG=C.UTF-8

CMD ["bash"]
