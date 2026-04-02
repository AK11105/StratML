"""
dl_pipeline.py
--------------
Phase 5 (DL) — PyTorch training pipeline.

Supported architectures (set via config.hyperparameters["architecture"]):
    "MLP"   — fully connected, tabular (default)
    "CNN1D" — 1-D convolutional, tabular treated as sequence
    "RNN"   — LSTM-based, tabular treated as sequence

Supported tasks (inferred from config.hyperparameters["task"]):
    "classification" (default)
    "regression"
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stratml.execution.schemas import ExperimentConfig, DataSplit


@dataclass
class DLPipelineResult:
    model: nn.Module
    y_val_pred: np.ndarray
    train_curve: list[float]
    val_curve: list[float]
    runtime: float


# ── Architectures ─────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_units: int, layers: int, dropout: float):
        super().__init__()
        dims = [input_dim] + [hidden_units] * layers
        blocks = []
        for i in range(len(dims) - 1):
            blocks += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        blocks.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CNN1D(nn.Module):
    """Treats each feature as a channel of length 1 — uses 1-D conv over feature space."""
    def __init__(self, input_dim: int, output_dim: int, hidden_units: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_units, hidden_units // 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_out = (hidden_units // 2) * input_dim
        self.head = nn.Linear(conv_out, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features) → (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(1)
        return self.head(x)


class _RNN(nn.Module):
    """Treats each feature as a time step of a single-channel sequence."""
    def __init__(self, input_dim: int, output_dim: int, hidden_units: int, layers: int, dropout: float):
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
        # x: (batch, features) → (batch, features, 1)
        x = x.unsqueeze(2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # last time step


def _build_model(arch: str, input_dim: int, output_dim: int, hp: dict) -> nn.Module:
    hidden = int(hp.get("hidden_units", 64))
    layers = int(hp.get("layers", 2))
    dropout = float(hp.get("dropout", 0.0))

    if arch == "CNN1D":
        return _CNN1D(input_dim, output_dim, hidden, dropout)
    if arch == "RNN":
        return _RNN(input_dim, output_dim, hidden, layers, dropout)
    return _MLP(input_dim, output_dim, hidden, layers, dropout)  # default: MLP


# ── Main entry point ──────────────────────────────────────────────────────────

def run_dl_pipeline(config: ExperimentConfig, data_split: DataSplit) -> DLPipelineResult:
    hp         = config.hyperparameters
    arch       = str(hp.get("architecture", "MLP")).upper()
    task       = str(hp.get("task", "classification")).lower()
    lr         = float(hp.get("learning_rate", 1e-3))
    batch_size = int(hp.get("batch_size", 32))
    epochs     = int(hp.get("epochs", 20))

    X_train_np = data_split.X_train.values.astype(np.float32)
    X_val_np   = data_split.X_val.values.astype(np.float32)
    X_train_t  = torch.tensor(X_train_np)
    X_val_t    = torch.tensor(X_val_np)

    input_dim = X_train_t.shape[1]

    if task == "regression":
        y_train_t = torch.tensor(data_split.y_train.values.astype(np.float32)).unsqueeze(1)
        y_val_t   = torch.tensor(data_split.y_val.values.astype(np.float32)).unsqueeze(1)
        output_dim = 1
        criterion  = nn.MSELoss()
        label_map  = None
    else:
        # classification — encode labels to 0-based int
        classes   = sorted(data_split.y_train.unique())
        label_map = {c: i for i, c in enumerate(classes)}
        y_train_t = torch.tensor(data_split.y_train.map(label_map).values, dtype=torch.long)
        y_val_t   = torch.tensor(data_split.y_val.map(label_map).fillna(-1).values, dtype=torch.long)
        output_dim = len(classes)
        criterion  = nn.CrossEntropyLoss()

    model     = _build_model(arch, input_dim, output_dim, hp)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader    = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    train_curve: list[float] = []
    val_curve:   list[float] = []
    best_val_loss    = float("inf")
    patience_counter = 0

    t0 = time.perf_counter()

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_curve.append(round(epoch_loss / len(X_train_t), 6))

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        val_curve.append(round(val_loss, 6))

        if config.early_stopping:
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    break

    runtime = round(time.perf_counter() - t0, 4)

    # ── Predictions ───────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        out = model(X_val_t)

    if task == "regression":
        y_val_pred = out.squeeze(1).numpy()
    else:
        preds = out.argmax(dim=1).numpy()
        inv_map = {i: c for c, i in label_map.items()}
        y_val_pred = np.array([inv_map[i] for i in preds])

    return DLPipelineResult(
        model=model,
        y_val_pred=y_val_pred,
        train_curve=train_curve,
        val_curve=val_curve,
        runtime=runtime,
    )
