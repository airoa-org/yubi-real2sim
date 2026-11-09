# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Reproducible MuJoCo integration-state snapshots."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


INTEGRATION_STATE_KEY = "mujoco_integration_state"
INTEGRATION_STATE_FORMAT_VERSION = 1


def _integration_spec(mujoco):
    return mujoco.mjtState.mjSTATE_INTEGRATION


def capture_integration_state(mujoco, model, data):
    """Return the complete MuJoCo state required for deterministic stepping."""
    spec = _integration_spec(mujoco)
    size = int(mujoco.mj_stateSize(model, spec))
    values = np.empty(size, dtype=np.float64)
    mujoco.mj_getState(model, data, values, spec)
    return {
        "format_version": INTEGRATION_STATE_FORMAT_VERSION,
        "spec": int(spec),
        "size": size,
        "values": [float(value) for value in values],
    }


def has_integration_state(payload):
    return isinstance(payload.get(INTEGRATION_STATE_KEY), Mapping)


def restore_integration_or_legacy_state(mujoco, model, data, payload):
    """Restore a v2 integration state or canonicalize missing v1 solver state.

    The legacy JSON format did not contain ``qacc_warmstart``. Leaving the
    destination value untouched makes a restore depend on the simulation that
    happened to run in that process before loading the file. Legacy snapshots
    therefore use an explicit zero warm-start unless they contain a separately
    saved value.
    """
    spec = _integration_spec(mujoco)
    expected_spec = int(spec)
    expected_size = int(mujoco.mj_stateSize(model, spec))
    block = payload.get(INTEGRATION_STATE_KEY)

    if block is not None:
        if not isinstance(block, Mapping):
            raise ValueError(f"{INTEGRATION_STATE_KEY} must be an object")
        format_version = int(block.get("format_version", -1))
        if format_version != INTEGRATION_STATE_FORMAT_VERSION:
            raise ValueError(
                f"unsupported {INTEGRATION_STATE_KEY} format_version {format_version}"
            )
        stored_spec = int(block.get("spec", -1))
        stored_size = int(block.get("size", -1))
        if stored_spec != expected_spec:
            raise ValueError(
                f"{INTEGRATION_STATE_KEY} spec mismatch: "
                f"got {stored_spec}, expected {expected_spec}"
            )
        if stored_size != expected_size:
            raise ValueError(
                f"{INTEGRATION_STATE_KEY} size mismatch: "
                f"got {stored_size}, expected {expected_size}"
            )
        values = np.asarray(block.get("values", []), dtype=np.float64)
        if values.shape != (expected_size,):
            raise ValueError(
                f"{INTEGRATION_STATE_KEY} values shape mismatch: "
                f"got {values.shape}, expected {(expected_size,)}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{INTEGRATION_STATE_KEY} contains non-finite values")
        mujoco.mj_setState(model, data, values, spec)
        warmstart = np.asarray(data.qacc_warmstart, dtype=np.float64).copy()
        source = "mjSTATE_INTEGRATION"
    else:
        raw_warmstart = payload.get("qacc_warmstart")
        if raw_warmstart is None:
            warmstart = np.zeros(model.nv, dtype=np.float64)
            source = "legacy-zero-warmstart"
        else:
            warmstart = np.asarray(raw_warmstart, dtype=np.float64)
            if warmstart.shape != (model.nv,):
                raise ValueError(
                    "qacc_warmstart shape mismatch: "
                    f"got {warmstart.shape}, expected {(model.nv,)}"
                )
            if not np.all(np.isfinite(warmstart)):
                raise ValueError("qacc_warmstart contains non-finite values")
            warmstart = warmstart.copy()
            source = "legacy-saved-warmstart"
        data.qacc_warmstart[:] = warmstart

        # These inputs are part of mjSTATE_INTEGRATION but were absent from v1.
        # Canonical zero values prevent the local pre-load rollout from leaking
        # applied forces into the restored episode.
        data.qfrc_applied[:] = np.asarray(
            payload.get("qfrc_applied", np.zeros(model.nv)), dtype=np.float64
        ).reshape(model.nv)
        data.xfrc_applied[:] = np.asarray(
            payload.get("xfrc_applied", np.zeros((model.nbody, 6))), dtype=np.float64
        ).reshape(model.nbody, 6)

    return {
        "source": source,
        "spec": expected_spec,
        "size": expected_size,
        "warmstart": warmstart,
    }


def forward_preserving_warmstart(mujoco, model, data, warmstart):
    """Refresh derived fields without consuming the restored next-step seed."""
    warmstart = np.asarray(warmstart, dtype=np.float64).reshape(model.nv)
    data.qacc_warmstart[:] = warmstart
    mujoco.mj_forward(model, data)
    data.qacc_warmstart[:] = warmstart
