"""
Deep learning model definitions for raw signal BP estimation.
4 architectures: 1D CNN, LSTM, CNN-LSTM, Transformer.
All models: input=(batch, channels, timesteps), output=(batch, 1).
"""

import math
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for time-series regression."""

    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)


class LSTMModel(nn.Module):
    """Bidirectional LSTM for time-series regression."""

    def __init__(self, in_channels: int = 1, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: (batch, channels, timesteps) -> (batch, timesteps, channels)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        # Use last timestep output
        last_hidden = lstm_out[:, -1, :]
        return self.regressor(last_hidden)


class CNNLSTM(nn.Module):
    """Hybrid CNN-LSTM: CNN extracts local features, LSTM captures temporal."""

    def __init__(self, in_channels: int = 1, hidden_size: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # After CNN: (batch, 128, timesteps/4)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # CNN feature extraction
        x = self.cnn(x)  # (batch, 128, T/4)
        # Permute for LSTM: (batch, T/4, 128)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.regressor(last_hidden)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer encoder for time-series regression."""

    def __init__(self, in_channels: int = 1, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(in_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, channels, timesteps) -> (batch, timesteps, channels)
        x = x.permute(0, 2, 1)
        x = self.input_projection(x)  # (batch, timesteps, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # Mean pooling over time
        x = x.mean(dim=1)  # (batch, d_model)
        return self.regressor(x)


# Model registry
DL_MODELS = {
    "cnn": CNN1D,
    "lstm": LSTMModel,
    "cnn_lstm": CNNLSTM,
    "transformer": TransformerModel,
}


def create_dl_model(model_name: str, in_channels: int = 1, **kwargs) -> nn.Module:
    """Create a DL model by name."""
    if model_name not in DL_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(DL_MODELS.keys())}")
    return DL_MODELS[model_name](in_channels=in_channels, **kwargs)
