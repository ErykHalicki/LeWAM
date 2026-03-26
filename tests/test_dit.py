"""Smoke tests for the DiT model — checks shapes and forward pass only."""

import pytest
import torch
from wam.models.DiT import DiT, DiT_S


B = 2
K = 4   # future frames
T = 2   # past frames
H = 7   # patch grid height
W = 7   # patch grid width
S = 16  # language sequence length
IN_DIM = 768
LANG_DIM = 768
STATE_DIM = 64


@pytest.fixture(scope="module")
def model():
    m = DiT_S(
        num_frames=K,
        num_past_frames=T,
        patch_h=H,
        patch_w=W,
        in_dim=IN_DIM,
        lang_dim=LANG_DIM,
        state_dim=STATE_DIM,
    )
    m.eval()
    return m


def make_inputs(device="cpu", with_mask=False):
    x           = torch.randn(B, K * H * W, IN_DIM, device=device)
    t           = torch.rand(B, device=device)
    past_frames = torch.randn(B, T * H * W, IN_DIM, device=device)
    l           = torch.randn(B, S, LANG_DIM, device=device)
    state       = torch.randn(B, STATE_DIM, device=device)
    l_mask      = torch.zeros(B, S, dtype=torch.bool, device=device) if with_mask else None
    return x, t, past_frames, l, state, l_mask


def test_output_shape(model):
    x, t, past_frames, l, state, _ = make_inputs()
    with torch.no_grad():
        out = model(x, t, past_frames, l, state)
    assert out.shape == (B, K * H * W, IN_DIM)


def test_output_shape_with_mask(model):
    x, t, past_frames, l, state, l_mask = make_inputs(with_mask=True)
    with torch.no_grad():
        out = model(x, t, past_frames, l, state, l_mask)
    assert out.shape == (B, K * H * W, IN_DIM)


def test_language_dropout():
    m = DiT_S(
        num_frames=K, num_past_frames=T, patch_h=H, patch_w=W,
        in_dim=IN_DIM, lang_dim=LANG_DIM, state_dim=STATE_DIM,
        language_dropout_prob=1.0,
    )
    m.train()
    x, t, past_frames, l, state, _ = make_inputs()
    out = m(x, t, past_frames, l, state)
    assert out.shape == (B, K * H * W, IN_DIM)


def test_state_dropout():
    m = DiT_S(
        num_frames=K, num_past_frames=T, patch_h=H, patch_w=W,
        in_dim=IN_DIM, lang_dim=LANG_DIM, state_dim=STATE_DIM,
        state_dropout_prob=1.0,
    )
    m.train()
    x, t, past_frames, l, state, _ = make_inputs()
    out = m(x, t, past_frames, l, state)
    assert out.shape == (B, K * H * W, IN_DIM)


def test_no_nan_in_output(model):
    x, t, past_frames, l, state, _ = make_inputs()
    with torch.no_grad():
        out = model(x, t, past_frames, l, state)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_timestep_embedding_varies():
    """t_embedder must produce different conditioning vectors for different timesteps."""
    from wam.models.DiT import TimestepEmbedder
    embedder = TimestepEmbedder(hidden_size=384)
    embedder.eval()
    with torch.no_grad():
        e0 = embedder(torch.zeros(B))
        e1 = embedder(torch.ones(B))
    assert not torch.allclose(e0, e1)
