"""Tests for DL model definitions and forward pass."""
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dl_models import CNN1D, LSTMModel, CNNLSTM, TransformerModel, create_dl_model


@pytest.fixture
def dummy_input_1ch():
    """Batch of 4, 1 channel, 1250 timesteps."""
    return torch.randn(4, 1, 1250)


@pytest.fixture
def dummy_input_3ch():
    """Batch of 4, 3 channels, 1250 timesteps."""
    return torch.randn(4, 3, 1250)


def test_cnn_forward_1ch(dummy_input_1ch):
    model = CNN1D(in_channels=1)
    out = model(dummy_input_1ch)
    assert out.shape == (4, 1)


def test_cnn_forward_3ch(dummy_input_3ch):
    model = CNN1D(in_channels=3)
    out = model(dummy_input_3ch)
    assert out.shape == (4, 1)


def test_lstm_forward_1ch(dummy_input_1ch):
    model = LSTMModel(in_channels=1)
    out = model(dummy_input_1ch)
    assert out.shape == (4, 1)


def test_lstm_forward_3ch(dummy_input_3ch):
    model = LSTMModel(in_channels=3)
    out = model(dummy_input_3ch)
    assert out.shape == (4, 1)


def test_cnn_lstm_forward(dummy_input_1ch):
    model = CNNLSTM(in_channels=1)
    out = model(dummy_input_1ch)
    assert out.shape == (4, 1)


def test_transformer_forward(dummy_input_1ch):
    model = TransformerModel(in_channels=1)
    out = model(dummy_input_1ch)
    assert out.shape == (4, 1)


def test_transformer_forward_3ch(dummy_input_3ch):
    model = TransformerModel(in_channels=3)
    out = model(dummy_input_3ch)
    assert out.shape == (4, 1)


def test_create_dl_model_registry():
    for name in ["cnn", "lstm", "cnn_lstm", "transformer"]:
        model = create_dl_model(name, in_channels=2)
        assert isinstance(model, torch.nn.Module)


def test_create_dl_model_unknown():
    with pytest.raises(ValueError):
        create_dl_model("unknown_model")


def test_model_parameter_counts():
    """Verify models have reasonable parameter counts."""
    for name in ["cnn", "lstm", "cnn_lstm", "transformer"]:
        model = create_dl_model(name, in_channels=1)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 1000, f"{name} has too few params: {n_params}"
        assert n_params < 10_000_000, f"{name} has too many params: {n_params}"


def test_gradients_flow():
    """Verify gradients flow through all models."""
    for name in ["cnn", "lstm", "cnn_lstm", "transformer"]:
        model = create_dl_model(name, in_channels=1)
        x = torch.randn(2, 1, 1250, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check at least some parameters have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad, f"{name}: no gradients flowing"
