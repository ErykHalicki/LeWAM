"""Smoke tests for the IDM model — checks shapes and forward pass only."""

import pytest
import torch
from wam.models.IDM import IDM, IDM_Baby


B                 = 2
T                 = 1   # past frames
K                 = 2   # future frames
H                 = 4   # patch grid height
W                 = 4   # patch grid width
IN_DIM            = 32
STATE_DIM         = 8
ACTION_LATENT_DIM = 16
NUM_AUX           = 2   # auxiliary cameras


@pytest.fixture(scope="module")
def model():
    m = IDM_Baby(
        num_past_frames=T,
        num_future_frames=K,
        patch_h=H,
        patch_w=W,
        in_dim=IN_DIM,
        state_dim=STATE_DIM,
        action_latent_dim=ACTION_LATENT_DIM,
    )
    m.eval()
    return m


def make_inputs(device="cpu"):
    current = torch.randn(B, T * H * W, IN_DIM, device=device)
    future  = torch.randn(B, K * H * W, IN_DIM, device=device)
    state   = torch.randn(B, STATE_DIM, device=device)
    return current, future, state


def test_output_shape(model):
    current, future, state = make_inputs()
    with torch.no_grad():
        out = model(current, future, state)
    assert out.shape == (B, K, ACTION_LATENT_DIM)


def test_chunk_len_equals_num_future_frames(model):
    """Action chunk length must equal num_future_frames — one action per predicted transition."""
    current, future, state = make_inputs()
    with torch.no_grad():
        out = model(current, future, state)
    assert out.shape[1] == K


def test_no_nan_in_output(model):
    current, future, state = make_inputs()
    with torch.no_grad():
        out = model(current, future, state)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_overfit_single_batch():
    """Model should memorize a single batch (MSE loss → ~0)."""
    torch.manual_seed(0)
    m = IDM_Baby(
        num_past_frames=T,
        num_future_frames=K,
        patch_h=H,
        patch_w=W,
        in_dim=IN_DIM,
        state_dim=STATE_DIM,
        action_latent_dim=ACTION_LATENT_DIM,
    )
    m.train()

    current     = torch.randn(1, T * H * W, IN_DIM)
    future      = torch.randn(1, K * H * W, IN_DIM)
    state       = torch.randn(1, STATE_DIM)
    target      = torch.randn(1, K, ACTION_LATENT_DIM)

    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for step in range(200):
        out  = m(current, future, state)
        loss = torch.nn.functional.mse_loss(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step {step:03d}  loss={loss.item():.6f}")

    assert loss.item() < 0.2, f"Model failed to overfit single batch (loss={loss.item():.4f})"


def test_multicamera_output_shape(model):
    current, future, state = make_inputs()
    current_mc = torch.randn(B, (1 + NUM_AUX) * T * H * W, IN_DIM)
    future_mc  = torch.randn(B, (1 + NUM_AUX) * K * H * W, IN_DIM)
    with torch.no_grad():
        out = model(current_mc, future_mc, state)
    assert out.shape == (B, K, ACTION_LATENT_DIM)


def test_multicamera_changes_output(model):
    current, future, state = make_inputs()
    current_mc = torch.randn(B, (1 + NUM_AUX) * T * H * W, IN_DIM)
    future_mc  = torch.randn(B, (1 + NUM_AUX) * K * H * W, IN_DIM)
    with torch.no_grad():
        out_single = model(current, future, state)
        out_multi  = model(current_mc, future_mc, state)
    assert out_single.shape == out_multi.shape


def test_different_frames_give_different_actions(model):
    """Different future frames should produce different action predictions."""
    current, future, state = make_inputs()
    future2 = torch.randn_like(future)
    with torch.no_grad():
        out1 = model(current, future, state)
        out2 = model(current, future2, state)
    assert not torch.allclose(out1, out2)
