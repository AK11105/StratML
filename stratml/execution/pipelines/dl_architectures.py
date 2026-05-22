"""
dl_architectures.py
-------------------
PyTorch nn.Module definitions for all supported DL architectures.

Adding a new architecture:
  1. Define a new nn.Module class here.
  2. Register it in _build_model().
  3. Add its name to _DL_MODELS in experiment_config_builder.py.
  No other files need to change.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Fully-connected network for tabular data.

    Per hidden layer: Linear → [BatchNorm1d] → ReLU → [Dropout]
    Final layer:      Linear(output_dim)  — no activation, no BN
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_units: int,
                 layers: int, dropout: float, batch_norm: bool):
        super().__init__()
        dims = [input_dim] + [hidden_units] * layers
        blocks: list[nn.Module] = []
        for i in range(len(dims) - 1):
            blocks.append(nn.Linear(dims[i], dims[i + 1]))
            if batch_norm:
                blocks.append(nn.BatchNorm1d(dims[i + 1]))
            blocks.append(nn.ReLU())
            if dropout > 0.0:
                blocks.append(nn.Dropout(dropout))
        blocks.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1D(nn.Module):
    """
    1-D convolutional network for tabular data with local feature structure.

    Treats the feature vector as a 1-channel sequence of length F.
    Two conv layers extract local patterns; a linear head maps to output.

    (batch, F) → unsqueeze(1) → (batch, 1, F)
    Conv1d(1→H, k=3) → [BN] → ReLU → Dropout
    Conv1d(H→H//2, k=3) → [BN] → ReLU
    flatten → Linear(H//2 * F, output_dim)
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_units: int,
                 dropout: float, batch_norm: bool):
        super().__init__()
        h2 = max(1, hidden_units // 2)
        layers: list[nn.Module] = [nn.Conv1d(1, hidden_units, kernel_size=3, padding=1)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_units))
        layers += [nn.ReLU(), nn.Dropout(dropout),
                   nn.Conv1d(hidden_units, h2, kernel_size=3, padding=1)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(h2))
        layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(h2 * input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return self.head(self.conv(x).flatten(1))


class RNN(nn.Module):
    """
    LSTM-based network for ordered/sequential feature sets.

    Each feature is treated as a time step of a single-channel sequence.
    Gradient clipping (applied in the training loop) is critical for stability.

    (batch, F) → unsqueeze(2) → (batch, F, 1)
    LSTM(input=1, hidden=H, num_layers=L)
    last hidden state → Linear(H, output_dim)
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_units: int,
                 layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_units,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_units, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def build_model(arch: str, input_dim: int, output_dim: int, hp: dict) -> nn.Module:
    """
    Instantiate the correct architecture from the arch string and hyperparameters.
    Unknown arch strings fall back to MLP.
    """
    hidden     = int(hp.get("hidden_units", 64))
    layers     = int(hp.get("layers", 2))
    dropout    = float(hp.get("dropout", 0.0))
    batch_norm = bool(hp.get("batch_norm", False))

    if arch == "CNN1D":
        return CNN1D(input_dim, output_dim, hidden, dropout, batch_norm)
    if arch == "RNN":
        return RNN(input_dim, output_dim, hidden, layers, dropout)
    return MLP(input_dim, output_dim, hidden, layers, dropout, batch_norm)
