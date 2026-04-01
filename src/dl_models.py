"""
Deep learning model definitions for raw PPG -> BP estimation.
2 architectures: ResNet-1D (baseline), ResNet-BiGRU (primary).
All models: input=(batch, channels, 1250), output=(batch, 1).

ResNet-BiGRU: 1D residual blocks capture morphological features (waveform shape),
bidirectional GRU captures temporal dependencies across the full 10-second window.
Based on Fan 2026 (MAE 4.78/2.98 on VitalDB) and Moulaeifard 2025 (PulseDB benchmark).
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """1D residual block with optional downsampling."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


def _make_layer(in_channels: int, out_channels: int, n_blocks: int, stride: int = 1):
    """Build a sequence of residual blocks."""
    layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
    for _ in range(1, n_blocks):
        layers.append(ResidualBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


class ResNet1D(nn.Module):
    """1D ResNet for time-series regression (baseline architecture)."""

    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = _make_layer(64, 64, n_blocks=2)
        self.layer2 = _make_layer(64, 128, n_blocks=2, stride=2)
        self.layer3 = _make_layer(128, 256, n_blocks=2, stride=2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


class ResNetBiGRU(nn.Module):
    """ResNet backbone + Bidirectional GRU for temporal modeling.

    ResNet extracts local morphological features from the PPG waveform.
    BiGRU reads the resulting feature sequence to capture long-range
    temporal dependencies (e.g., beat-to-beat evolution across the 10s window).
    """

    def __init__(self, in_channels: int = 1, gru_hidden: int = 128,
                 gru_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = _make_layer(64, 64, n_blocks=2)
        self.layer2 = _make_layer(64, 128, n_blocks=2, stride=2)
        self.layer3 = _make_layer(128, 256, n_blocks=2, stride=2)

        self.gru = nn.GRU(
            input_size=256,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # (batch, 256, seq_len) -> (batch, seq_len, 256) for GRU
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru(x)
        # Last timestep captures full bidirectional context
        x = gru_out[:, -1, :]
        return self.head(x)


# Model registry
DL_MODELS = {
    "resnet_bigru": ResNetBiGRU,
    "resnet": ResNet1D,
}


def create_dl_model(model_name: str, in_channels: int = 1, **kwargs) -> nn.Module:
    """Create a DL model by name."""
    if model_name not in DL_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(DL_MODELS.keys())}")
    return DL_MODELS[model_name](in_channels=in_channels, **kwargs)
