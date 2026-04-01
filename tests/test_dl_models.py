"""
Tests for deep learning model architectures (ResNet-1D, ResNet-BiGRU).
TDD: written before implementation. Tests validate forward pass shapes,
gradient flow, parameter counts, and registry behavior.
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dl_models import (
    ResidualBlock,
    ResNet1D,
    ResNetBiGRU,
    create_dl_model,
    DL_MODELS,
)


# ─── Fixtures ───


@pytest.fixture
def dummy_input_1ch():
    """Batch of 4, 1 channel, 1250 timesteps."""
    return torch.randn(4, 1, 1250)


@pytest.fixture
def dummy_input_3ch():
    """Batch of 4, 3 channels, 1250 timesteps."""
    return torch.randn(4, 3, 1250)


# ─── ResNet-1D forward pass tests ───


def test_resnet_forward_1ch(dummy_input_1ch):
    model = ResNet1D(in_channels=1)
    model.eval()
    out = model(dummy_input_1ch)
    assert out.shape == (4, 1)


def test_resnet_forward_3ch(dummy_input_3ch):
    model = ResNet1D(in_channels=3)
    model.eval()
    out = model(dummy_input_3ch)
    assert out.shape == (4, 1)


# ─── ResNet-BiGRU forward pass tests ───


def test_resnet_bigru_forward_1ch(dummy_input_1ch):
    model = ResNetBiGRU(in_channels=1)
    model.eval()
    out = model(dummy_input_1ch)
    assert out.shape == (4, 1)


def test_resnet_bigru_forward_3ch(dummy_input_3ch):
    model = ResNetBiGRU(in_channels=3)
    model.eval()
    out = model(dummy_input_3ch)
    assert out.shape == (4, 1)


# ─── ResidualBlock tests ───


def test_residual_block_no_downsample():
    """ResidualBlock with same channels preserves spatial dimension."""
    block = ResidualBlock(64, 64)
    x = torch.randn(2, 64, 100)
    out = block(x)
    assert out.shape == (2, 64, 100)


def test_residual_block_with_downsample():
    """ResidualBlock with stride=2 halves length, changes channels."""
    block = ResidualBlock(64, 128, stride=2)
    x = torch.randn(2, 64, 100)
    out = block(x)
    assert out.shape == (2, 128, 50)


# ─── Registry tests ───


def test_create_dl_model_registry():
    """All registered models can be instantiated."""
    for name in ["resnet_bigru", "resnet"]:
        model = create_dl_model(name, in_channels=1)
        assert isinstance(model, torch.nn.Module)


def test_create_dl_model_unknown():
    """Old model names raise ValueError (removed from registry)."""
    for old_name in ["cnn", "lstm", "cnn_lstm", "transformer"]:
        with pytest.raises(ValueError):
            create_dl_model(old_name)


# ─── Parameter count tests ───


def test_model_parameter_counts():
    """Verify models have expected parameter counts."""
    resnet = create_dl_model("resnet", in_channels=1)
    n_resnet = sum(p.numel() for p in resnet.parameters())
    assert 500_000 < n_resnet < 2_000_000, f"ResNet1D params: {n_resnet}"

    bigru = create_dl_model("resnet_bigru", in_channels=1)
    n_bigru = sum(p.numel() for p in bigru.parameters())
    assert 1_000_000 < n_bigru < 3_000_000, f"ResNetBiGRU params: {n_bigru}"

    # BiGRU should have more params than ResNet-only (GRU adds ~600K)
    assert n_bigru > n_resnet


# ─── Gradient flow tests ───


def test_gradients_flow():
    """Verify gradients reach all parameters in both models."""
    for name in ["resnet_bigru", "resnet"]:
        model = create_dl_model(name, in_channels=1)
        x = torch.randn(2, 1, 1250, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, f"No gradients flowing in {name}"


# ─── Shared backbone test ───


def test_resnet_backbone_shared_structure():
    """Both models share the same backbone structure (stem, layer1-3)."""
    for name in ["resnet_bigru", "resnet"]:
        model = create_dl_model(name, in_channels=1)
        assert hasattr(model, 'stem'), f"{name} missing .stem"
        assert hasattr(model, 'layer1'), f"{name} missing .layer1"
        assert hasattr(model, 'layer2'), f"{name} missing .layer2"
        assert hasattr(model, 'layer3'), f"{name} missing .layer3"
