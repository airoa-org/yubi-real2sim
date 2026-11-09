# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

# Portions of this file are adapted from openpi
# (https://github.com/Physical-Intelligence/openpi, commit e6b0441,
# src/openpi/models/pi0.py), Copyright the openpi authors, licensed under the
# Apache License, Version 2.0. See THIRD_PARTY_NOTICES.
# _sample_actions_conditioned is adapted from Pi0.sample_actions and changed by
# AIRoA to project the known action prefix onto the rectified-flow path at
# every Euler step.

"""Rectified-flow action-prefix inpainting for OpenPI Pi0 policies.

The active OpenPI checkout is an ignored external dependency, so this module
builds a conditioned version of ``Pi0.sample_actions`` at runtime.  The
conditioned sampler is bound to the loaded NNX model and JIT-compiled with the
same helper used by OpenPI's ``Policy`` class.
"""

from __future__ import annotations

import types
from typing import Any


_CACHE_ATTR = "_rf_conditioned_sample_actions"


def _sample_actions_conditioned(
    self: Any,
    rng: Any,
    observation: Any,
    *,
    num_steps: int = 10,
    noise: Any | None = None,
    conditioned_actions: Any,
    condition_mask: Any,
) -> Any:
    """Pi0 Euler sampler with hard RF-path conditioning.

    This intentionally follows the OpenPI Pi0 implementation closely.  The
    model's initial Gaussian noise is also the fixed noise used to re-noise the
    known actions at every ODE time.
    """

    import einops
    import jax
    import jax.numpy as jnp

    from openpi.models import model as openpi_model
    from openpi.models.pi0 import make_attn_mask

    observation = openpi_model.preprocess_observation(None, observation, train=False)
    batch_size = observation.state.shape[0]
    expected_action_shape = (batch_size, self.action_horizon, self.action_dim)
    if noise is None:
        noise = jax.random.normal(rng, expected_action_shape)

    initial_noise = jnp.asarray(noise)
    conditioned_actions = jnp.asarray(conditioned_actions, dtype=initial_noise.dtype)
    condition_mask = jnp.asarray(condition_mask, dtype=jnp.bool_)
    if initial_noise.shape != expected_action_shape:
        raise ValueError(
            f"noise shape {initial_noise.shape} must match model action shape {expected_action_shape}"
        )
    if conditioned_actions.shape != expected_action_shape:
        raise ValueError(
            "conditioned_actions shape "
            f"{conditioned_actions.shape} must match model action shape {expected_action_shape}"
        )
    expected_mask_shape = (batch_size, self.action_horizon, 1)
    if condition_mask.shape != expected_mask_shape:
        raise ValueError(
            f"condition_mask shape {condition_mask.shape} must be {expected_mask_shape}"
        )

    dt = -1.0 / num_steps

    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )

    def project(x_t: Any, time: Any) -> Any:
        clipped_time = jnp.clip(jnp.asarray(time, dtype=x_t.dtype), 0.0, 1.0)
        known_xt = clipped_time * initial_noise + (1.0 - clipped_time) * conditioned_actions
        return jnp.where(condition_mask, known_xt, x_t)

    def step(carry: tuple[Any, Any]) -> tuple[Any, Any]:
        x_t, time = carry
        x_t = project(x_t, time)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        suffix_prefix_mask = einops.repeat(
            prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
        )
        full_attn_mask = jnp.concatenate([suffix_prefix_mask, suffix_attn_mask], axis=-1)
        expected_attn_shape = (
            batch_size,
            suffix_tokens.shape[1],
            prefix_tokens.shape[1] + suffix_tokens.shape[1],
        )
        assert full_attn_mask.shape == expected_attn_shape
        positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None]
            + jnp.cumsum(suffix_mask, axis=-1)
            - 1
        )

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        next_time = time + dt
        next_x = project(x_t + dt * v_t, next_time)
        return next_x, next_time

    def cond(carry: tuple[Any, Any]) -> Any:
        _, time = carry
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (initial_noise, 1.0))
    # Avoid any residual floating-point time error in the final conditioned slots.
    return project(x_0, 0.0)


def get_conditioned_sample_actions(policy: Any) -> Any:
    """Return a cached JIT-compiled conditioned sampler for an OpenPI policy."""

    if bool(getattr(policy, "_is_pytorch_model", False)):
        raise NotImplementedError(
            "RF action inpainting currently supports JAX Pi0/Pi0.5 policies only; "
            "PyTorch policy sampling is not implemented"
        )

    cached = getattr(policy, _CACHE_ATTR, None)
    if cached is not None:
        return cached

    model = getattr(policy, "_model", None)
    if model is None:
        raise TypeError("expected an OpenPI Policy with a loaded `_model`")
    required_model_attrs = (
        "action_dim",
        "action_horizon",
        "embed_prefix",
        "embed_suffix",
        "action_out_proj",
        "PaliGemma",
    )
    missing = [name for name in required_model_attrs if not hasattr(model, name)]
    if missing:
        raise TypeError(
            "RF action inpainting requires a Pi0/Pi0.5 model; "
            f"missing attributes: {', '.join(missing)}"
        )

    from openpi.shared import nnx_utils

    bound_sampler = types.MethodType(_sample_actions_conditioned, model)
    sampler = nnx_utils.module_jit(bound_sampler)
    setattr(policy, _CACHE_ATTR, sampler)
    return sampler
