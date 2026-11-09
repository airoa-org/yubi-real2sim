# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from scene.gs_background_server import _cache_static_sh, _static_sh_model


def test_static_sh_values_and_storage_are_shared():
    dc = torch.arange(24.0).reshape(8, 1, 3).requires_grad_()
    rest = torch.arange(360.0).reshape(8, 15, 3).requires_grad_()
    model = SimpleNamespace(_features_dc=dc, _features_rest=rest)
    expected = torch.cat((dc, rest), dim=1).detach().clone()
    _cache_static_sh(model)
    assert torch.equal(model._render_features, expected)
    assert model._render_features.is_contiguous()
    assert model._render_features.grad_fn is None
    assert model._features_dc.untyped_storage().data_ptr() == model._render_features.untyped_storage().data_ptr()
    assert model._features_rest.untyped_storage().data_ptr() == model._render_features.untyped_storage().data_ptr()


def test_static_model_renders_from_the_cached_buffer():
    class GaussianModel:
        @property
        def get_features(self):
            return torch.cat((self._features_dc, self._features_rest), dim=1)

    model = _static_sh_model(GaussianModel)()
    model._features_dc = torch.zeros(4, 1, 3)
    model._features_rest = torch.ones(4, 15, 3)
    _cache_static_sh(model)
    assert model.get_features is model._render_features
