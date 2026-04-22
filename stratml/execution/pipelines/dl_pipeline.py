"""
dl_pipeline.py
--------------
Phase 5 (DL) — PyTorch training pipeline.

Supported architectures (config.hyperparameters["architecture"]):
    "MLP"   — fully connected, tabular (default)
    "CNN1D" — 1-D convolutional over feature space
    "RNN"   — LSTM-based, features treated as time steps

Supported tasks (config.hyperparameters["task"]):
    "classification" (default)
    "regression"

Production features:
    - GPU auto-selection  (cuda → mps → cpu)
    - Mixed precision     (AMP / FP16, CUDA only, opt-in via mixed_precision=True)
    - Gradient clipping   (grad_clip, default 1.0 — critical for RNN stability)
    - Weight decay        (weight_decay in Adam, default 0.0)
    - LR schedulers       (plateau | cosine | none)
    - BatchNorm           (batch_norm=True)
    - Early stopping      (always active, best-weight restore)
    - TensorBoard         (when tensorboard_log_dir is provided)
    - Lazy DataLoader     (TabularDataset — no full-tensor pre-allocation)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from stratml.execution.schemas import ExperimentConfig, DataSplit


# ── Device ────────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Lazy Dataset (no full-tensor pre-allocation) ──────────────────────────────

class _TabularDataset(Dataset):
    """
    Wraps numpy arrays as a PyTorch Dataset.

    Slices are converted to tensors on-demand per batch, avoiding
    pre-allocating the entire dataset on the GPU. This is safe for
    datasets of any size — the DataLoader handles batching.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X  # float32 numpy array
        self.y = y  # float32 or int64 numpy array

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx : idx + 1]).squeeze(0)


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
    model_state: dict = field(default_factory=dict)  # CPU state_dict for .pth saving


# ── Architectures ─────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    """
    Fully-connected network for tabular data.

    Architecture per hidden layer:
        Linear(in, H) → [BatchNorm1d(H)] → ReLU → [Dropout(p)]
    Final layer:
        Linear(H, output_dim)  — no activation, no BN

    BatchNorm before activation stabilises training across varied feature
    scales and reduces sensitivity to learning rate choice.
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

    Treats the feature vector as a 1-channel sequence of length F.
    Two conv layers extract local patterns; a linear head maps to output.

    Architecture:
        (batch, F) → unsqueeze(1) → (batch, 1, F)
        Conv1d(1→H, k=3, pad=1) → [BN] → ReLU → Dropout
        Conv1d(H→H//2, k=3, pad=1) → [BN] → ReLU
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
        x = x.unsqueeze(1)   # (batch, 1, F)
        x = self.conv(x)
        return self.head(x.flatten(1))


class _RNN(nn.Module):
    """
    LSTM-based network for ordered/sequential feature sets.

    Each feature is treated as a time step of a single-channel sequence.
    Gradient clipping (applied externally) is critical for LSTM stability.

    Architecture:
        (batch, F) → unsqueeze(2) → (batch, F, 1)
        LSTM(input=1, hidden=H, num_layers=L, dropout between layers)
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
        x = x.unsqueeze(2)               # (batch, F, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # last time step


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
        data_split:           Pre-processed DataSplit (all numeric, no nulls).
        tensorboard_log_dir:  Optional path for TensorBoard SummaryWriter.

    Returns:
        DLPipelineResult with model (CPU), predictions, curves, and metadata.
    """
    hp             = config.hyperparameters
    arch           = str(hp.get("architecture", "MLP")).upper()
    task           = str(hp.get("task", "classification")).lower()
    lr             = float(hp.get("learning_rate", 1e-3))
    weight_decay   = float(hp.get("weight_decay", 0.0))
    batch_size     = int(hp.get("batch_size", 32))
    epochs         = int(hp.get("epochs", 20))
    scheduler_type = str(hp.get("scheduler", "plateau")).lower()
    grad_clip      = float(hp.get("grad_clip", 1.0))   # 0.0 = disabled
    mixed_prec     = bool(hp.get("mixed_precision", False))

    device     = _get_device()
    use_amp    = mixed_prec and device.type == "cuda"  # AMP only on CUDA
    scaler     = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Prepare numpy arrays ──────────────────────────────────────────────────
    X_train_np = data_split.X_train.values.astype(np.float32)
    X_val_np   = data_split.X_val.values.astype(np.float32)
    input_dim  = X_train_np.shape[1]

    if task == "regression":
        y_train_np = data_split.y_train.values.astype(np.float32).reshape(-1, 1)
        y_val_np   = data_split.y_val.values.astype(np.float32).reshape(-1, 1)
        output_dim = 1
        criterion  = nn.MSELoss()
        label_map  = None
    else:
        classes    = sorted(data_split.y_train.unique())
        label_map  = {c: i for i, c in enumerate(classes)}
        y_train_np = data_split.y_train.map(label_map).values.astype(np.int64)
        y_val_np   = data_split.y_val.map(label_map).fillna(-1).values.astype(np.int64)
        output_dim = len(classes)
        criterion  = nn.CrossEntropyLoss()

    # ── DataLoaders (lazy — no full GPU pre-allocation) ───────────────────────
    train_loader = DataLoader(
        _TabularDataset(X_train_np, y_train_np),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0,  # 0 = main process; safe for all platforms
    )

    # Val tensors are small enough to keep on device for fast eval
    X_val_t = torch.tensor(X_val_np, device=device)
    y_val_t = torch.tensor(y_val_np, device=device)
    if task == "regression":
        y_val_t = y_val_t.reshape(-1, 1)

    # ── Model + optimiser ─────────────────────────────────────────────────────
    model     = _build_model(arch, input_dim, output_dim, hp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ── LR scheduler ─────────────────────────────────────────────────────────
    scheduler: Optional[object] = None
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = None
    if tensorboard_log_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=tensorboard_log_dir)
        except Exception:
            pass

    # ── Training loop ─────────────────────────────────────────────────────────
    train_curve: list[float] = []
    val_curve:   list[float] = []
    best_val_loss   = float("inf")
    best_state_dict = None
    patience_counter = 0
    early_stopped    = False
    patience = config.early_stopping_patience if config.early_stopping else max(5, epochs // 4)

    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(xb), yb)
                scaler.scale(loss).backward()
                if grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(xb), yb)
                loss.backward()
                if grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            epoch_loss += loss.item() * len(xb)

        train_loss = round(epoch_loss / len(X_train_np), 6)
        train_curve.append(train_loss)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_loss = round(criterion(model(X_val_t), y_val_t).item(), 6)
        val_curve.append(val_loss)

        # ── Scheduler ─────────────────────────────────────────────────────────
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # ── TensorBoard ───────────────────────────────────────────────────────
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val",   val_loss,   epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # ── Early stopping + best-weight tracking ─────────────────────────────
        if val_loss < best_val_loss - 1e-4:
            best_val_loss    = val_loss
            best_state_dict  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
        preds   = out.argmax(dim=1).numpy()
        inv_map = {i: c for c, i in label_map.items()}
        y_val_pred = np.array([inv_map[i] for i in preds])

    return DLPipelineResult(
        model=model.cpu(),
        y_val_pred=y_val_pred,
        train_curve=train_curve,
        val_curve=val_curve,
        runtime=runtime,
        device_used=str(device),
        epochs_run=len(train_curve),
        early_stopped=early_stopped,
        model_state={k: v.cpu() for k, v in model.state_dict().items()},
    )
