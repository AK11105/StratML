"""
dl_pipeline.py
--------------
Phase 5 (DL) — PyTorch MLP training pipeline with early stopping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

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


def run_dl_pipeline(config: ExperimentConfig, data_split: DataSplit) -> DLPipelineResult:
    hp = config.hyperparameters
    hidden_units = int(hp.get("hidden_units", 64))
    layers       = int(hp.get("layers", 2))
    dropout      = float(hp.get("dropout", 0.0))
    lr           = float(hp.get("learning_rate", 1e-3))
    batch_size   = int(hp.get("batch_size", 32))
    epochs       = int(hp.get("epochs", 20))

    X_train = torch.tensor(data_split.X_train.values, dtype=torch.float32)
    y_train_raw = data_split.y_train
    X_val   = torch.tensor(data_split.X_val.values,   dtype=torch.float32)

    # Encode labels to 0-based integers
    classes = sorted(y_train_raw.unique())
    label_map = {c: i for i, c in enumerate(classes)}
    y_train_enc = torch.tensor(y_train_raw.map(label_map).values, dtype=torch.long)
    y_val_enc   = torch.tensor(data_split.y_val.map(label_map).fillna(-1).values, dtype=torch.long)

    input_dim  = X_train.shape[1]
    output_dim = len(classes)

    model = _MLP(input_dim, output_dim, hidden_units, layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(TensorDataset(X_train, y_train_enc), batch_size=batch_size, shuffle=True)

    train_curve: list[float] = []
    val_curve:   list[float] = []
    best_val_loss = float("inf")
    patience_counter = 0

    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_curve.append(round(epoch_loss / len(X_train), 6))

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val_enc).item()
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

    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        y_val_pred = logits.argmax(dim=1).numpy()

    # Decode back to original labels
    inv_map = {i: c for c, i in label_map.items()}
    y_val_pred_decoded = np.array([inv_map[i] for i in y_val_pred])

    return DLPipelineResult(
        model=model,
        y_val_pred=y_val_pred_decoded,
        train_curve=train_curve,
        val_curve=val_curve,
        runtime=runtime,
    )
