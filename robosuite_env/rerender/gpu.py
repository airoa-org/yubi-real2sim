# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""GPU allocation-slot resolution, EGL binding, and post-hoc GPU locks."""


from __future__ import annotations

import fcntl
import os
from pathlib import Path

from robosuite_env.rerender.common import GpuBinding, RerenderError
from real2sim.gpu_binding import (
    GPU_BINDINGS_ENV,
    GpuBindingError,
    _nvidia_maps,
    contract_from_environment,
    resolve_cuda_token as _resolve_cuda_token,
    resolve_gpu_binding as _resolve_gpu_binding,
)


def resolve_cuda_device(gpu_index: int) -> str:
    """Map an allocation-local slot without widening current CUDA visibility."""

    try:
        return _resolve_cuda_token(gpu_index, environment=os.environ)
    except GpuBindingError as exc:
        raise RerenderError(str(exc)) from exc


def _nvidia_gpu_uuid_to_index() -> dict[str, int]:
    """Return an exact host GPU UUID to physical EGL-index mapping."""
    try:
        mapping, _ = _nvidia_maps(None)
        return mapping
    except GpuBindingError as exc:
        raise RerenderError(str(exc)) from exc


def resolve_gpu_binding(gpu_index: int) -> GpuBinding:
    """Resolve a slot into distinct exact-CUDA and physical-EGL identities."""

    try:
        if (
            isinstance(gpu_index, bool)
            or not isinstance(gpu_index, int)
            or gpu_index < 0
        ):
            raise GpuBindingError(
                f"GPU selector must be a non-negative integer, got {gpu_index!r}"
            )
        if os.environ.get(GPU_BINDINGS_ENV):
            contract = contract_from_environment(os.environ)
            matches = {
                (
                    str(entry["cuda_token"]),
                    int(entry["egl_physical_index"]),
                )
                for entries in contract["roles"].values()
                for entry in entries
                if entry["allocation_slot"] == gpu_index
            }
            if len(matches) != 1:
                raise GpuBindingError(
                    f"GPU selector {gpu_index} is not uniquely bound by the sealed contract"
                )
            cuda_token, physical_index = matches.pop()
            return GpuBinding(
                allocation_slot=gpu_index,
                cuda_token=cuda_token,
                egl_physical_index=physical_index,
            )
        binding = _resolve_gpu_binding(gpu_index, environment=os.environ)
    except GpuBindingError as exc:
        raise RerenderError(str(exc)) from exc
    return GpuBinding(
        allocation_slot=binding.allocation_slot,
        cuda_token=binding.cuda_token,
        egl_physical_index=binding.egl_physical_index,
    )


def _coerce_gpu_binding(gpu: int | GpuBinding) -> GpuBinding:
    return gpu if isinstance(gpu, GpuBinding) else resolve_gpu_binding(gpu)


def canonical_render_lock(gpu: int | GpuBinding) -> str:
    binding = _coerce_gpu_binding(gpu)
    root = Path(f"/tmp/real2sim-render-locks-{os.getuid()}")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return str(
        root / f"mujoco_render_faces_gpu_{binding.egl_physical_index}.lock"
    )


def _acquire_posthoc_gpu_lock(
    gpu: int | GpuBinding, *, lock_root: Path | None = None
):
    """Hold one GS-server / rerender lifecycle per visible GPU and user."""

    binding = _coerce_gpu_binding(gpu)
    root = lock_root or Path(f"/tmp/real2sim-render-locks-{os.getuid()}")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = root / f"posthoc_gpu_{binding.egl_physical_index}.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RerenderError(
            "another post-hoc renderer already owns physical GPU "
            f"{binding.egl_physical_index}; lock={path}"
        ) from exc
    return handle
