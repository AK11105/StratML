"""
dl_pipeline.py
--------------
Phase 5 (DL) — PyTorch training pipeline.

Supported architectures (set via config.hyperparameters["architecture"]):
    "MLP"   — fully connected, tabular (default)
    "CNN1D" — 1-D convolutional over feature space
    "RNN"   — LSTM-based, features treated as time steps

Supported tasks (inferred from config.hyperparameters["task"]):
    "classification" (default)
    "regression"

GPU is used automatically when available (cuda → mps → cpu).
TensorBoard curves are written to tensorboard_log_dir when provided.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stratml.execution.schemas import ExperimentConfig, DataSplit


# ── Device selection ──────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class DLPipelineResult:
    model: nn.Module
    y_val_pred: np.ndarray
    train_curve: list[float]
    val_curve: list[float]
    runtime: float
    device_used: str = "cpu"
    epochs_run: int = 0
    early_stopped: bool = False
    model_state: dict = field(default_factory=dict)   # state_dict for .pth saving


# ── Architectures ─────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    """
    Fully-connected network for tabular data.

    Architecture:
        [Linear → BatchNorm → ReLU → Dropout] × layers  →  Linear(output)

    BatchNorm is applied before activation to stabilise training on varied
    feature scales. It is skipped on the final output layer.
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


class _CNN1D(nn.Module):
    """
    1-D convolutional network for tabular data with local feature structure.

    The feature vector is treated as a 1-channel sequence of length F.
    Two conv layers extract local patterns; a linear head maps to output.

    Architecture:
        (batch, F) → unsqueeze → (batch, 1, F)
        Conv1d(1 → H, k=3) → BN → ReLU → Dropout
        Conv1d(H → H//2, k=3) → BN → ReLU
        flatten → Linear(output)
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_units: int,
                 dropout: float, batch_norm: bool):
        super().__init__()
        h2 = max(1, hidden_units // 2)
        conv_layers: list[nn.Module] = [
            nn.Conv1d(1, hidden_units, kernel_size=3, padding=1),
        ]
        if batch_norm:
            conv_layers.append(nn.BatchNorm1d(hidden_units))
        conv_layers += [nn.ReLU(), nn.Dropout(dropout),
                        nn.Conv1d(hidden_units, h2, kernel_size=3, padding=1)]
        if batch_norm:
            conv_layers.append(nn.BatchNorm1d(h2))
        conv_layers.append(nn.ReLU())
        self.conv = nn.Sequential(*conv_layers)
        self.head = nn.Linear(h2 * input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)          # (batch, 1, features)
        x = self.conv(x)
        x = x.flatten(1)
        return self.head(x)


class _RNN(nn.Module):
    """
    LSTM-based network. Each feature is treated as a time step of a
    single-channel sequence — suitable when features represent an ordered
    progression (e.g. windowed sensor readings, ordered embeddings).

    Architecture:
        (batch, F) → unsqueeze → (batch, F, 1)
        LSTM(input=1, hidden=H, layers=L)
        last hidden state → Linear(output)
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
        x = x.unsqueeze(2)              # (batch, features, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]) # last time step


def _build_model(arch: str, input_dim: int, output_dim: int, hp: dict) -> nn.Module:
    hidden     = int(hp.get("hidden_units", 64))
    layers     = int(hp.get("layers", 2))
    dropout    = float(hp.get("dropout", 0.0))
    batch_norm = bool(hp.get("batch_norm", False))

    if arch == "CNN1D":
        return _CNN1D(input_dim, output_dim, hidden, dropout, batch_norm)
    if arch == "RNN":
        return _RNN(input_dim, output_dim, hidden, layers, dropout)
    return _MLP(input_dim, output_dim, hidden, layers, dropout, batch_norm)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_dl_pipeline(
    config: ExperimentConfig,
    data_split: DataSplit,
    tensorboard_log_dir: Optional[str] = None,
) -> DLPipelineResult:
    """
    Train a DL model and return predictions + training curves.

    Args:
        config:               ExperimentConfig with model_type="dl".
        data_split:           Pre-processed DataSplit.
        tensorboard_log_dir:  If provided, training/val loss curves are written
                              here as TensorBoard scalar events.
    """
    hp         = config.hyperparameters
    arch       = str(hp.get("architecture", "MLP")).upper()
    task       = str(hp.get("task", "classification")).lower()
    lr         = float(hp.get("learning_rate", 1e-3))
    batch_size = int(hp.get("batch_size", 32))
    epochs     = int(hp.get("epochs", 20))
    scheduler_type = str(hp.get("scheduler", "plateau")).lower()  # plateau | cosine | none

    device = _get_device()

    # ── Tensors ───────────────────────────────────────────────────────────────
    X_train_t = torch.tensor(data_split.X_train.values.astype(np.float32), device=device)
    X_val_t   = torch.tensor(data_split.X_val.values.astype(np.float32),   device=device)
    input_dim = X_train_t.shape[1]

    if task == "regression":
        y_train_t  = torch.tensor(data_split.y_train.values.astype(np.float32), device=device).unsqueeze(1)
        y_val_t    = torch.tensor(data_split.y_val.values.astype(np.float32),   device=device).unsqueeze(1)
        output_dim = 1
        criterion  = nn.MSELoss()
        label_map  = None
    else:
        classes   = sorted(data_split.y_train.unique())
        label_map = {c: i for i, c in enumerate(classes)}
        y_train_t = torch.tensor(
            data_split.y_train.map(label_map).values, dtype=torch.long, device=device
        )
        y_val_t = torch.tensor(
            data_split.y_val.map(label_map).fillna(-1).values, dtype=torch.long, device=device
        )
        output_dim = len(classes)
        criterion  = nn.CrossEntropyLoss()

    # ── Model + optimiser + scheduler ────────────────────────────────────────
    model     = _build_model(arch, input_dim, output_dim, hp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler: Optional[object] = None
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    # ── TensorBoard writer ────────────────────────────────────────────────────
    writer = None
    if tensorboard_log_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=tensorboard_log_dir)
        except Exception:
            pass  # TensorBoard optional — never crash the pipeline

    # ── Training loop ─────────────────────────────────────────────────────────
    train_curve: list[float] = []
    val_curve:   list[float] = []
    best_val_loss    = float("inf")
    best_state_dict  = None
    patience_counter = 0
    early_stopped    = False

    # Always enable early stopping for DL with a sensible default
    patience = config.early_stopping_patience if config.early_stopping else max(5, epochs // 4)

    t0 = time.perf_counter()

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        train_loss = round(epoch_loss / len(X_train_t), 6)
        train_curve.append(train_loss)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_loss = round(criterion(model(X_val_t), y_val_t).item(), 6)
        val_curve.append(val_loss)

        # ── Scheduler step ────────────────────────────────────────────────────
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # ── TensorBoard ───────────────────────────────────────────────────────
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val",   val_loss,   epoch)

        # ── Early stopping — always active for DL ─────────────────────────────
        if val_loss < best_val_loss - 1e-4:
            best_val_loss   = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                early_stopped = True
                break

    runtime = round(time.perf_counter() - t0, 4)

    if writer is not None:
        writer.close()

    # Restore best weights
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    # ── Predictions ───────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        out = model(X_val_t).cpu()

    if task == "regression":
        y_val_pred = out.squeeze(1).numpy()
    else:
        preds  = out.argmax(dim=1).numpy()
        inv_map = {i: c for c, i in label_map.items()}
        y_val_pred = np.array([inv_map[i] for i in preds])

    # state_dict on CPU for serialisation
    final_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return DLPipelineResult(
        model=model.cpu(),
        y_val_pred=y_val_pred,
        train_curve=train_curve,
        val_curve=val_curve,
        runtime=runtime,
        device_used=str(device),
        epochs_run=len(train_curve),
        early_stopped=early_stopped,
        model_state=final_state,
    )
