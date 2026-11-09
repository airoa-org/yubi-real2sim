# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Batch inference helpers for OpenPI policy rollouts.

The simulator feeds unbatched observations to ``policy.infer``.  OpenPI then
applies input transforms, adds a batch dimension, runs the model, removes the
batch dimension, and applies output transforms.  For multi-sim rollout we need
the same transforms, but with one model call over many environments.

This module keeps the OpenPI/JAX/Torch dependency behind runtime imports so the
lightweight policy I/O tests can import it without a policy environment.
"""

from __future__ import annotations

import dataclasses as _dc
import hashlib
import importlib
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from . import policy_io as pio


_ACTION_CONDITIONING_INPUT_KEYS = (
    "observation.pose.left_hand_root.relative",
    "observation.pose.right_hand_root.relative",
    "action.joint_states",
)
_POLICY_LOAD_METADATA_ATTR = "_real2sim_policy_load_metadata"


def ensure_locked_openpi(
    repo_root: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Point OpenPI at the repository cache and refuse an unlocked OpenPI.

    OpenPI and ``robosuite_env.policy.yubi_openpi`` import from the installed
    environment; nothing is added to ``sys.path``. ``require_locked_imports``
    in ``CHEAP`` mode fails closed when ``openpi`` or ``openpi_client`` would
    import from a file the distribution's ``RECORD`` does not list at its
    installed path (a checkout, an editable install, a symlinked top-level
    path) or from a distribution whose ``direct_url.json`` does not name the
    commit ``uv.lock`` pins.

    The launcher's norm-stats resolution, the policy loader (in-process or
    batch server) and the rollout worker all call this one function before
    they import OpenPI. It does not hash the installed files: a production
    launch runs the ``FULL`` check once (``real2sim.preflight
    .validate_locked_imports``). It returns the verified records (module,
    distribution, version, origin, direct_url, ``check: "cheap"``, checks), so
    the process that imports OpenPI records what it actually ran.
    """

    from real2sim.locked_imports import CHEAP, OPENPI_PACKAGES, require_locked_imports

    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    openpi_cache = root / "runtime-cache/openpi"
    if openpi_cache.is_dir():
        os.environ.setdefault("OPENPI_DATA_HOME", str(openpi_cache))
    return require_locked_imports(root, OPENPI_PACKAGES, mode=CHEAP)


def load_openpi_policy(
    checkpoint: str | os.PathLike[str],
    *,
    config: str,
    config_src: str = "deploy",
    action_horizon: int = 0,
    instruction: str = "Pick up the cup with your right hand and set it on the plate",
    repo_root: str | os.PathLike[str] | None = None,
) -> Any:
    """Load the YUBI OpenPI policy exactly like ``robosuite_env/sim.py``."""

    # This process imports OpenPI to run the model, so it records the locked
    # distributions it verified itself rather than relying on the launcher's.
    cfg, norm_stats_path, norm_stats_asset_id, locked_imports = _resolve_locked_norm_stats(
        checkpoint,
        config=config,
        config_src=config_src,
        action_horizon=action_horizon,
        repo_root=repo_root,
    )
    from openpi.policies import policy_config as _polcfg

    ckpt = Path(checkpoint).expanduser().resolve()
    # Resolve the relative-hand state quaternion sign from the exact norm stats
    # this policy normalises with, before loading weights. A batch server
    # publishes it (hello packet and policy_batch_server_ready.json) so every
    # worker can refuse a hemisphere that disagrees with it.
    state_quat_hemisphere = pio.policy_state_quat_hemisphere_from_norm_stats(
        norm_stats_path
    )
    policy = _polcfg.create_trained_policy(
        cfg,
        ckpt,
        default_prompt=instruction,
    )
    load_metadata = {
        "norm_stats_asset_id": norm_stats_asset_id,
        "norm_stats_path": str(norm_stats_path),
        "norm_stats_sha256": _file_sha256(norm_stats_path),
        "policy_state_quat_hemisphere": state_quat_hemisphere["hemisphere"],
        "policy_state_quat_hemisphere_resolution": state_quat_hemisphere,
        "locked_imports": locked_imports,
    }
    if state_quat_hemisphere["norm_stats_sha256"] != load_metadata["norm_stats_sha256"]:
        raise RuntimeError(
            f"norm stats {norm_stats_path} changed while the policy was loading"
        )
    setattr(policy, _POLICY_LOAD_METADATA_ATTR, load_metadata)
    print(
        "[policy] loaded norm stats "
        f"asset_id={norm_stats_asset_id!r} path={norm_stats_path} "
        f"sha256={load_metadata['norm_stats_sha256']} "
        f"state_quat_hemisphere={state_quat_hemisphere['hemisphere']} "
        f"({state_quat_hemisphere['classification']}, qw q01="
        f"{state_quat_hemisphere['qw_q01']:.4f} q99="
        f"{state_quat_hemisphere['qw_q99']:.4f})",
        flush=True,
    )
    return policy


def resolve_openpi_norm_stats(
    checkpoint: str | os.PathLike[str],
    *,
    config: str,
    config_src: str = "deploy",
    action_horizon: int = 0,
    repo_root: str | os.PathLike[str] | None = None,
) -> tuple[Any, Path, str]:
    """Load an OpenPI config and resolve the exact norm-stats asset it selects.

    The launcher preflight and parameter resolution call this; the policy
    loader uses the same resolver and also keeps the guard's records.
    """

    train_config, norm_stats_path, asset_id, _records = _resolve_locked_norm_stats(
        checkpoint,
        config=config,
        config_src=config_src,
        action_horizon=action_horizon,
        repo_root=repo_root,
    )
    return train_config, norm_stats_path, asset_id


def _resolve_locked_norm_stats(
    checkpoint: str | os.PathLike[str],
    *,
    config: str,
    config_src: str,
    action_horizon: int,
    repo_root: str | os.PathLike[str] | None,
) -> tuple[Any, Path, str, list[dict[str, Any]]]:
    """Load an OpenPI config and resolve the exact norm-stats asset it selects.

    This is the common resolver for launcher checks and policy loading.  An
    entirely absent assets tree remains distinguishable from a checkpoint that
    contains stats for another config: the former can represent a placeholder
    fixture, while the latter is a mismatched checkpoint and must fail closed.
    """

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    assets = checkpoint_path / "assets"
    if not assets.is_dir():
        raise pio.PolicyStateQuatHemisphereAbsent(
            f"policy checkpoint has no assets directory: {assets}"
        )

    locked_imports = ensure_locked_openpi(repo_root)
    if config_src == "deploy":
        config_module = importlib.import_module(
            "robosuite_env.policy.yubi_openpi.config_yubi"
        )

        train_config = config_module.get_config(config)
    elif config_src == "training":
        from openpi.training import config as config_module

        train_config = config_module.get_config(config)
        train_config = _dc.replace(
            train_config,
            model=_dc.replace(train_config.model, action_horizon=32),
        )
    else:
        raise ValueError(
            f"unknown config_src {config_src!r}; expected 'deploy' or 'training'"
        )

    if (
        action_horizon > 0
        and getattr(train_config.model, "action_horizon", None) != action_horizon
    ):
        train_config = _dc.replace(
            train_config,
            model=_dc.replace(
                train_config.model,
                action_horizon=int(action_horizon),
            ),
        )
    try:
        norm_stats_path, asset_id = resolve_config_norm_stats_path(
            train_config,
            checkpoint_path,
        )
    except FileNotFoundError as exc:
        found = sorted(
            Path(directory) / "norm_stats.json"
            for directory, _subdirectories, files in os.walk(
                assets, followlinks=True
            )
            if "norm_stats.json" in files
        )
        if not found:
            raise pio.PolicyStateQuatHemisphereAbsent(
                f"policy checkpoint has no assets/**/norm_stats.json: {assets}"
            ) from exc
        raise pio.PolicyStateQuatHemisphereError(
            f"{exc}; checkpoint contains other norm stats: "
            + ", ".join(str(path) for path in found)
        ) from exc
    return train_config, norm_stats_path, asset_id, locked_imports


def resolve_config_norm_stats_path(
    train_config: Any,
    checkpoint: str | os.PathLike[str],
) -> tuple[Path, str]:
    """Resolve only the norm-stats asset declared by the OpenPI data config."""

    data_factory = train_config.data
    lightweight_resolver = getattr(
        data_factory, "resolve_norm_stats_asset_id", None
    )
    if callable(lightweight_resolver):
        asset_id = lightweight_resolver()
    else:
        # Custom and upstream training factories can override the asset id in
        # create(). Execute them rather than inferring from similarly named
        # fields. The checked-in YUBI factory exposes the lightweight method
        # above because its create() also constructs tokenizer transforms.
        data_config = data_factory.create(
            train_config.assets_dirs,
            train_config.model,
        )
        asset_id = getattr(data_config, "asset_id", None)
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise ValueError(
            f"OpenPI config {train_config.name!r} must declare a non-empty norm-stats asset_id"
        )
    asset_path = Path(asset_id)
    if asset_path.is_absolute() or ".." in asset_path.parts:
        raise ValueError(
            f"OpenPI config {train_config.name!r} has unsafe norm-stats asset_id {asset_id!r}"
        )

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    norm_stats_path = checkpoint_path / "assets" / asset_path / "norm_stats.json"
    if not norm_stats_path.is_file():
        raise FileNotFoundError(
            f"norm stats required by OpenPI config {train_config.name!r} were not found: "
            f"{norm_stats_path} (asset_id={asset_id!r})"
        )
    return norm_stats_path, asset_id


def policy_load_metadata(policy: Any) -> dict[str, Any]:
    """Return config-derived load provenance recorded on a loaded policy."""

    metadata = getattr(policy, _POLICY_LOAD_METADATA_ATTR, {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _copy_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_tree(v) for v in value)
    return value


def stack_pytrees(items: Sequence[Any]) -> Any:
    """Stack matching pytrees along a new leading batch axis."""

    if not items:
        raise ValueError("cannot stack an empty sequence")
    first = items[0]
    if isinstance(first, dict):
        keys = tuple(first.keys())
        for item in items[1:]:
            if not isinstance(item, dict) or tuple(item.keys()) != keys:
                raise ValueError("cannot stack pytrees with different dict keys")
        return {k: stack_pytrees([item[k] for item in items]) for k in keys}
    if isinstance(first, tuple):
        for item in items[1:]:
            if not isinstance(item, tuple) or len(item) != len(first):
                raise ValueError("cannot stack pytrees with different tuple arity")
        return tuple(stack_pytrees([item[i] for item in items]) for i in range(len(first)))
    if isinstance(first, list):
        for item in items[1:]:
            if not isinstance(item, list) or len(item) != len(first):
                raise ValueError("cannot stack pytrees with different list arity")
        return [stack_pytrees([item[i] for item in items]) for i in range(len(first))]
    return np.stack([np.asarray(item) for item in items], axis=0)


def unstack_pytree(tree: Any, batch_size: int) -> list[Any]:
    """Split a batched pytree into ``batch_size`` unbatched pytrees."""

    if isinstance(tree, dict):
        parts = {k: unstack_pytree(v, batch_size) for k, v in tree.items()}
        return [{k: parts[k][i] for k in parts} for i in range(batch_size)]
    if isinstance(tree, tuple):
        parts = [unstack_pytree(v, batch_size) for v in tree]
        return [tuple(part[i] for part in parts) for i in range(batch_size)]
    if isinstance(tree, list):
        parts = [unstack_pytree(v, batch_size) for v in tree]
        return [[part[i] for part in parts] for i in range(batch_size)]
    arr = np.asarray(tree)
    if arr.shape[0] != batch_size:
        raise ValueError(f"batched leaf has leading dim {arr.shape[0]}, expected {batch_size}")
    return [arr[i] for i in range(batch_size)]


def _has_openpi_policy_internals(policy: Any) -> bool:
    return all(
        hasattr(policy, name)
        for name in (
            "_input_transform",
            "_output_transform",
            "_sample_actions",
            "_sample_kwargs",
            "_is_pytorch_model",
        )
    )


def policy_sample_num_steps(policy: Any, *, default: int = 10) -> int:
    """Return the policy's configured RF ODE step count."""

    sample_kwargs = getattr(policy, "_sample_kwargs", {})
    value = dict(sample_kwargs or {}).get("num_steps", default)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"policy num_steps must be a positive integer, got {value!r}")
    steps = int(value)
    if steps <= 0:
        raise ValueError(f"policy num_steps must be positive, got {steps}")
    return steps


def _prepare_noise(noise: np.ndarray | None, batch_size: int) -> np.ndarray | None:
    if noise is None:
        return None
    arr = np.asarray(noise)
    if arr.ndim == 2:
        arr = np.broadcast_to(arr[None, ...], (batch_size, *arr.shape))
    if arr.shape[0] != batch_size:
        raise ValueError(f"noise leading dim {arr.shape[0]} does not match batch size {batch_size}")
    return arr


def prepare_action_conditioning(
    prefixes: Sequence[np.ndarray | None],
    *,
    action_horizon: int,
    action_dim: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Pad per-item normalized prefixes into full RF conditioning tensors."""

    items = list(prefixes)
    if not any(prefix is not None for prefix in items):
        return None, None
    horizon = int(action_horizon)
    dim = int(action_dim)
    conditioned = np.zeros((len(items), horizon, dim), dtype=np.float32)
    mask = np.zeros((len(items), horizon, 1), dtype=bool)
    for batch_index, prefix in enumerate(items):
        if prefix is None:
            continue
        arr = np.asarray(prefix, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != dim:
            raise ValueError(
                f"normalized action prefix must have shape [steps, {dim}], got {arr.shape}"
            )
        if arr.shape[0] <= 0 or arr.shape[0] > horizon:
            raise ValueError(
                f"action prefix length must be in [1, {horizon}], got {arr.shape[0]}"
            )
        conditioned[batch_index, : arr.shape[0]] = arr
        mask[batch_index, : arr.shape[0], 0] = True
    return conditioned, mask


def infer_many(
    policy: Any,
    obs_list: Sequence[dict],
    *,
    noise: np.ndarray | None = None,
) -> list[dict]:
    """Run one policy inference call for many unbatched observations.

    For OpenPI ``Policy`` objects this uses the policy's transform/model/output
    pipeline directly with a real leading batch dimension.  Other policies may
    expose ``infer_many(obs_list)``; otherwise we fall back to sequential
    ``infer`` calls so callers can use the same interface in tests.
    """

    obs_seq = list(obs_list)
    if not obs_seq:
        return []
    has_action_conditioning = any(
        any(key in obs for key in _ACTION_CONDITIONING_INPUT_KEYS) for obs in obs_seq
    )
    if (
        not has_action_conditioning
        and hasattr(policy, "infer_many")
        and callable(policy.infer_many)
    ):
        if noise is None:
            return list(policy.infer_many(obs_seq))
        return list(
            policy.infer_many(
                obs_seq,
                noise=_prepare_noise(noise, len(obs_seq)),
            )
        )
    if not _has_openpi_policy_internals(policy):
        if has_action_conditioning:
            raise TypeError(
                "RF action conditioning requires OpenPI Policy internals; "
                "custom infer_many/infer wrappers cannot apply ODE prefix projection"
            )
        if noise is not None:
            raise TypeError(
                "fixed RF noise requires OpenPI Policy internals or a custom "
                "infer_many(obs_list, noise=...) implementation"
            )
        return [policy.infer(obs) for obs in obs_seq]

    batch_size = len(obs_seq)
    transformed = []
    normalized_prefixes: list[np.ndarray | None] = []
    for obs in obs_seq:
        item = policy._input_transform(_copy_tree(obs))  # noqa: SLF001
        normalized_prefixes.append(item.pop("actions", None))
        transformed.append(item)
    batched_inputs = stack_pytrees(transformed)

    sample_kwargs = dict(policy._sample_kwargs)  # noqa: SLF001
    batched_noise = _prepare_noise(noise, batch_size)
    is_pytorch = bool(policy._is_pytorch_model)  # noqa: SLF001
    model = policy._model  # noqa: SLF001
    model_config = getattr(model, "config", None)
    action_horizon = int(getattr(model, "action_horizon", getattr(model_config, "action_horizon", 0)))
    action_dim = int(getattr(model, "action_dim", getattr(model_config, "action_dim", 0)))
    conditioned_actions, condition_mask = prepare_action_conditioning(
        normalized_prefixes,
        action_horizon=action_horizon,
        action_dim=action_dim,
    )

    start = time.monotonic()
    if is_pytorch:
        import torch

        device = policy._pytorch_device  # noqa: SLF001
        model_inputs = _torch_tree(batched_inputs, device)
        if batched_noise is not None:
            sample_kwargs["noise"] = torch.from_numpy(np.asarray(batched_noise)).to(device)
        if conditioned_actions is not None:
            sample_kwargs["conditioned_actions"] = torch.from_numpy(conditioned_actions).to(device)
            sample_kwargs["condition_mask"] = torch.from_numpy(condition_mask).to(device)
        rng_or_device = device
    else:
        import jax
        import jax.numpy as jnp

        model_inputs = jax.tree.map(lambda x: jnp.asarray(x), batched_inputs)
        policy._rng, rng_or_device = jax.random.split(policy._rng)  # noqa: SLF001
        if batched_noise is not None:
            sample_kwargs["noise"] = jnp.asarray(batched_noise)
        if conditioned_actions is not None:
            sample_kwargs["conditioned_actions"] = jnp.asarray(conditioned_actions)
            sample_kwargs["condition_mask"] = jnp.asarray(condition_mask)

    from openpi.models import model as _model

    observation = _model.Observation.from_dict(model_inputs)
    sample_actions = policy._sample_actions  # noqa: SLF001
    if conditioned_actions is not None:
        from .rf_action_inpainting import get_conditioned_sample_actions

        sample_actions = get_conditioned_sample_actions(policy)
    outputs = {
        "state": model_inputs["state"],
        "actions": sample_actions(rng_or_device, observation, **sample_kwargs),
    }
    model_sec = time.monotonic() - start

    if is_pytorch:
        import jax

        outputs_np = jax.tree.map(
            lambda x: np.asarray(x.detach().cpu()) if hasattr(x, "detach") else np.asarray(x),
            outputs,
        )
    else:
        import jax

        outputs_np = jax.tree.map(lambda x: np.asarray(x), outputs)

    results = []
    for batch_index, item in enumerate(unstack_pytree(outputs_np, batch_size)):
        out = policy._output_transform(item)  # noqa: SLF001
        out["policy_timing"] = {
            "infer_ms": model_sec * 1000.0,
            "batch_size": batch_size,
        }
        results.append(out)
    return results


def _torch_tree(tree: Any, device: str) -> Any:
    import torch

    if isinstance(tree, dict):
        return {k: _torch_tree(v, device) for k, v in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_torch_tree(v, device) for v in tree)
    if isinstance(tree, list):
        return [_torch_tree(v, device) for v in tree]
    return torch.from_numpy(np.asarray(tree)).to(device)
