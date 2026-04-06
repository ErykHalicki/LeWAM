import torch
import pytest
from lewam.training.losses import SIGReg


@pytest.fixture
def sigreg():
    return SIGReg(knots=17, num_proj=1024,dtype=torch.float32)


def test_gaussian_loss_is_low(sigreg):
    proj = torch.randn(256, 4, 768)
    loss = sigreg(proj)
    assert loss.item() < 5.0, f"Expected low loss for N(0,1) input, got {loss.item():.4f}"


def test_uniform_loss_is_high(sigreg):
    proj = torch.rand(256, 4, 768) * 6 - 3
    loss = sigreg(proj)
    assert loss.item() > 20.0, f"Expected high loss for uniform input, got {loss.item():.4f}"


def test_wide_gaussian_loss_is_high(sigreg):
    proj = torch.randn(256, 4, 768) * 5.0
    loss = sigreg(proj)
    assert loss.item() > 100.0, f"Expected high loss for N(0,25) input, got {loss.item():.4f}"
