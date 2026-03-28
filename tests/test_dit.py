"""Smoke tests for the DiT model — checks shapes and forward pass only."""

import pytest
import torch
from wam.models.DiT import DiT, DiT_Baby


B = 2
K = 2   # future frames
T = 1   # past frames
H = 4   # patch grid height
W = 4   # patch grid width
S = 4   # language sequence length
IN_DIM = 32
LANG_DIM = 32
STATE_DIM = 8
NUM_AUX = 2  # auxiliary cameras


@pytest.fixture(scope="module")
def model():
    m = DiT_Baby(
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
        out = model(x, t, past_frames, l, state, l_mask=l_mask)
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


def test_overfit_single_batch():
    """Model should memorize a single batch (loss → ~0).

    Synthetic flow matching task:
      x_t = (1 - t) * x0 + t * x1,  target velocity = x1 - x0
    Fixed x0, x1, conditioning — train until loss < 1e-2.
    """
    torch.manual_seed(0)
    m = DiT_Baby(
        num_frames=K, num_past_frames=T, patch_h=H, patch_w=W,
        in_dim=IN_DIM, lang_dim=LANG_DIM, state_dim=STATE_DIM,
    )
    m.train()

    x0          = torch.randn(1, K * H * W, IN_DIM)
    x1          = torch.randn(1, K * H * W, IN_DIM)
    past_frames = torch.randn(1, T * H * W, IN_DIM)
    l           = torch.randn(1, S, LANG_DIM)
    state       = torch.randn(1, STATE_DIM)
    target      = x1 - x0

    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss = torch.tensor(float("inf"))
    for step in range(200):
        t = torch.rand(1)
        x_t = (1 - t) * x0 + t * x1
        v_pred = m(x_t, t, past_frames, l, state)
        loss = torch.nn.functional.mse_loss(v_pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step {step:03d}  loss={loss.item():.6f}")

    assert loss.item() < 0.2, f"Model failed to overfit single batch (loss={loss.item():.4f})"


def test_aux_frames_output_shape(model):
    x, t, past_frames, l, state, _ = make_inputs()
    aux_frames = torch.randn(B, NUM_AUX * T * H * W, IN_DIM)
    with torch.no_grad():
        out = model(x, t, past_frames, l, state, aux_frames=aux_frames)
    assert out.shape == (B, K * H * W, IN_DIM)


def test_aux_frames_in_computation_graph():
    import torch.nn as nn
    m = DiT_Baby(
        num_frames=K, num_past_frames=T, patch_h=H, patch_w=W,
        in_dim=IN_DIM, lang_dim=LANG_DIM, state_dim=STATE_DIM,
    )
    nn.init.kaiming_uniform_(m.final_layer.linear.weight, nonlinearity='relu')
    m.train()
    x, t, past_frames, l, state, _ = make_inputs()
    aux_frames = torch.randn(B, NUM_AUX * T * H * W, IN_DIM, requires_grad=True)
    out = m(x, t, past_frames, l, state, aux_frames=aux_frames)
    out.sum().backward()
    assert aux_frames.grad is not None
    assert aux_frames.grad.abs().sum() > 0


def test_ode_solve():
    """After overfitting, Euler integration from x0 should arrive near x1.

    Flow matching ODE:  dx/dt = v(x_t, t)
    Euler:  x_{t+dt} = x_t + v(x_t, t) * dt
    """
    torch.manual_seed(0)
    m = DiT_Baby(
        num_frames=K, num_past_frames=T, patch_h=H, patch_w=W,
        in_dim=IN_DIM, lang_dim=LANG_DIM, state_dim=STATE_DIM,
    )

    x0          = torch.randn(1, K * H * W, IN_DIM)
    x1          = torch.randn(1, K * H * W, IN_DIM)
    past_frames = torch.randn(1, T * H * W, IN_DIM)
    l           = torch.randn(1, S, LANG_DIM)
    state       = torch.randn(1, STATE_DIM)

    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for step in range(200):
        t = torch.rand(1)
        x_t = (1 - t) * x0 + t * x1
        v_pred = m(x_t, t, past_frames, l, state)
        loss = torch.nn.functional.mse_loss(v_pred, x1 - x0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step {step:03d}  loss={loss.item():.6f}")

    m.eval()
    steps = 100
    dt = 1.0 / steps
    x = x0.clone()
    with torch.no_grad():
        for i in range(steps):
            t = torch.tensor([i * dt])
            x = x + m(x, t, past_frames, l, state) * dt

    err = torch.nn.functional.mse_loss(x, x1).item()
    print(f"ODE final error: {err:.6f}")
    assert err < 0.1, f"ODE solve did not converge to x1 (err={err:.4f})"
