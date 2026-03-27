"""Smoke tests for the V-JEPA2 vision transformer encoder."""

import pytest
import torch
from functools import partial
import torch.nn as nn
from vjepa2.app.vjepa_2_1.models.vision_transformer import VisionTransformer


B, C, T, H, W = 1, 3, 4, 32, 32


@pytest.fixture(scope="module")
def encoder():
    m = VisionTransformer(
        patch_size=16,
        img_size=(H, W),
        num_frames=T,
        tubelet_size=2,
        embed_dim=4,
        depth=12,
        num_heads=1,
        mlp_ratio=2,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_sdpa=False,
        use_SiLU=False,
        wide_SiLU=False,
        uniform_power=True,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    m.eval()
    return m


def test_output_shape(encoder):
    x = torch.randn(B, C, T, H, W)
    with torch.no_grad():
        out = encoder(x)
    num_patches = (H // 16) * (W // 16) * (T // 2)
    assert out.ndim == 3
    assert out.shape[0] == B
    assert out.shape[1] == num_patches


def test_no_nan_in_output(encoder):
    x = torch.randn(B, C, T, H, W)
    with torch.no_grad():
        out = encoder(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_different_inputs_give_different_outputs(encoder):
    x1 = torch.randn(B, C, T, H, W)
    x2 = torch.randn(B, C, T, H, W)
    with torch.no_grad():
        out1 = encoder(x1)
        out2 = encoder(x2)
    assert not torch.allclose(out1, out2)
