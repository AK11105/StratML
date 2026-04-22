# DL Pipeline — Architectures, Tasks & Configuration

## Overview

`stratml/execution/pipelines/dl_pipeline.py` is the single entry point (`run_dl_pipeline`)
for all PyTorch training. Everything is controlled through `config.hyperparameters` —
no code changes needed to switch architectures, tasks, or training behaviour.

---

## Production Features

| Feature | Hyperparameter | Default | Notes |
|---|---|---|---|
| GPU auto-selection | — | auto | cuda → mps → cpu |
| Mixed precision (FP16) | `mixed_precision` | `False` | CUDA only; silently off on CPU/MPS |
| Gradient clipping | `grad_clip` | `1.0` | Set `0.0` to disable; critical for RNN |
| Weight decay (L2) | `weight_decay` | `0.0` | Applied in Adam optimizer |
| LR scheduler | `scheduler` | `"plateau"` | `plateau` \| `cosine` \| `none` |
| BatchNorm | `batch_norm` | `False` | MLP and CNN1D only |
| Early stopping | always on | patience=5 | Best-weight restore on trigger |
| TensorBoard | `tensorboard_log_dir` arg | `None` | Writes Loss/train, Loss/val, LR |
| Lazy DataLoader | — | always | No full-tensor GPU pre-allocation |

---

## Device Selection

```
CUDA (NVIDIA GPU)  →  MPS (Apple Silicon)  →  CPU
```

`DLPipelineResult.device_used` reports which device was used.
`ResourceUsage.gpu_used` is `True` when CUDA or MPS is active.

---

## Supported Tasks

Set via `config.hyperparameters["task"]`:

| Value | Loss | Output | Prediction |
|---|---|---|---|
| `"classification"` (default) | CrossEntropyLoss | Linear → N classes | argmax → decoded original label |
| `"regression"` | MSELoss | Linear → 1 neuron | raw float |

Labels are encoded to 0-based integers internally and decoded back before returning.
The metrics engine and decision engine never see encoded integers.

---

## Supported Architectures

Set via `config.hyperparameters["architecture"]`:

### MLP (default)

Fully-connected network for standard tabular data.

```
Input (F features)
  ↓
[Linear(H) → BatchNorm(H) → ReLU → Dropout(p)] × L
  ↓
Linear(output_dim)
```

BatchNorm is applied per layer when `batch_norm=True`.

---

### CNN1D

1-D convolutional network. Treats the feature vector as a 1-channel sequence of
length F. Useful when features have local or spatial structure.

```
(batch, F) → unsqueeze(1) → (batch, 1, F)
Conv1d(1→H, k=3, pad=1) → [BN] → ReLU → Dropout
Conv1d(H→H//2, k=3, pad=1) → [BN] → ReLU
flatten → Linear(H//2 * F, output_dim)
```

---

### RNN (LSTM)

LSTM-based network. Each feature is treated as a time step of a single-channel
sequence. Gradient clipping (`grad_clip`) is critical for LSTM stability.

```
(batch, F) → unsqueeze(2) → (batch, F, 1)
LSTM(input=1, hidden=H, num_layers=L, dropout between layers)
last hidden state → Linear(H, output_dim)
```

---

## Full Hyperparameter Reference

| Key | Type | Default | Description |
|---|---|---|---|
| `architecture` | str | `"MLP"` | `"MLP"`, `"CNN1D"`, or `"RNN"` |
| `task` | str | `"classification"` | `"classification"` or `"regression"` |
| `hidden_units` | int | `64` | Width of hidden layers |
| `layers` | int | `2` | Depth (MLP: FC layers, RNN: LSTM layers) |
| `dropout` | float | `0.0` | Dropout probability (0.0 = disabled) |
| `batch_norm` | bool | `False` | BatchNorm after each hidden layer (MLP, CNN1D) |
| `learning_rate` | float | `1e-3` | Adam initial learning rate |
| `weight_decay` | float | `0.0` | L2 regularization in Adam (AdamW-style) |
| `batch_size` | int | `32` | Mini-batch size |
| `epochs` | int | `20` | Maximum training epochs |
| `scheduler` | str | `"plateau"` | LR scheduler: `"plateau"`, `"cosine"`, `"none"` |
| `grad_clip` | float | `1.0` | Max gradient norm; `0.0` = disabled |
| `mixed_precision` | bool | `False` | FP16 AMP training (CUDA only) |

---

## Gradient Clipping

Controlled via `grad_clip` (default `1.0`).

Applied after `loss.backward()` and before `optimizer.step()` using
`torch.nn.utils.clip_grad_norm_`. This prevents gradient explosion, which is
especially common in RNN/LSTM training on tabular data.

```python
# Enabled (default)
{"grad_clip": 1.0}

# Disabled
{"grad_clip": 0.0}

# Stricter clipping for unstable training
{"grad_clip": 0.5}
```

**Always leave enabled for RNN.** For MLP/CNN1D it's a safety net with negligible cost.

---

## Weight Decay

Controlled via `weight_decay` (default `0.0`).

Passed directly to `torch.optim.Adam` as the `weight_decay` parameter. This applies
L2 regularization to all parameters during the optimizer step — equivalent to AdamW
when non-zero.

```python
# Light regularization
{"weight_decay": 1e-4}

# Stronger regularization (high-dimensional features)
{"weight_decay": 1e-3}
```

The `change_optimizer` action automatically bumps `weight_decay` by `1e-4` when the
LR scale factor is ≤ 0.1 (aggressive reduction implies more regularization is needed).

---

## Mixed Precision Training (FP16)

Controlled via `mixed_precision` (default `False`).

Uses `torch.amp.autocast("cuda")` + `GradScaler` to run forward/backward passes in
FP16 where safe, falling back to FP32 for numerically sensitive operations.

**Benefits on CUDA:**
- ~1.5–2× training speedup
- ~50% GPU memory reduction
- Enables larger batch sizes

**Behaviour on CPU/MPS:** silently disabled — no error, no performance change.

```python
{"mixed_precision": True}  # only effective on CUDA
```

Gradient clipping is applied correctly with AMP via `scaler.unscale_()` before
`clip_grad_norm_`, ensuring clipping operates on the true gradient scale.

---

## LR Schedulers

| Value | Behaviour |
|---|---|
| `"plateau"` (default) | `ReduceLROnPlateau` — halves LR when val loss stalls for 3 epochs |
| `"cosine"` | `CosineAnnealingLR` — smoothly decays LR to 0 over `epochs` |
| `"none"` | Fixed LR throughout |

Current LR is logged to TensorBoard as `LR` scalar each epoch when a writer is active.

The `change_optimizer` action sets `scheduler="cosine"` automatically when `lr_scale ≤ 0.1`.

---

## Early Stopping

Always active for DL. Patience is `ExperimentConfig.early_stopping_patience` (default 5).

- Monitors validation loss each epoch
- Stops if val loss doesn't improve by > `1e-4` for `patience` consecutive epochs
- **Best weights are restored** — the returned model is from the epoch with lowest val loss
- `DLPipelineResult.early_stopped` is `True` when triggered
- `DLPipelineResult.epochs_run` reports actual epochs completed

---

## Lazy DataLoader

The pipeline uses a custom `_TabularDataset` (a `torch.utils.data.Dataset` subclass)
instead of pre-allocating full tensors on the GPU.

**Why this matters:** Pre-allocating `torch.tensor(X_train.values, device="cuda")` for
a 500k-row dataset will OOM on most GPUs. The lazy loader converts numpy slices to
tensors per batch, keeping GPU memory proportional to `batch_size`, not dataset size.

`pin_memory=True` is set automatically when CUDA is available, enabling faster
host→device transfers.

---

## TensorBoard Integration

Pass `tensorboard_log_dir` to `run_dl_pipeline`:

```python
result = run_dl_pipeline(config, data_split, tensorboard_log_dir="outputs/runs/exp_001")
```

Writes per epoch:
- `Loss/train` — training loss
- `Loss/val` — validation loss
- `LR` — current learning rate (useful for debugging scheduler behaviour)

View with:
```bash
tensorboard --logdir outputs/runs/
```

The orchestrator sets this to `outputs/<run_id>/tensorboard/<experiment_id>/` automatically.

---

## Artifact Saving

For DL models, two files are saved:

| File | Format | Use |
|---|---|---|
| `model.pkl` | joblib | Used by `model.py` script and report generator |
| `model.pth` | `torch.save` of state_dict + metadata | Proper PyTorch format for resuming |

The `.pth` checkpoint contains:
```python
{
    "state_dict":      model.state_dict(),   # weights (CPU tensors)
    "architecture":    "MLP",
    "hyperparameters": {...},                # full hp dict to rebuild the model
    "experiment_id":   "exp_001",
}
```

To load and resume:
```python
import torch
checkpoint = torch.load("model.pth", weights_only=True)
# Rebuild model with same arch + hp, then:
model.load_state_dict(checkpoint["state_dict"])
```

---

## Capacity Mutations (Decision Engine Actions)

| Action | DL behaviour | ML behaviour |
|---|---|---|
| `increase_model_capacity` | `hidden_units × scale`; +1 layer if `scale ≥ 1.5` (max 6) | `n_estimators × scale` |
| `decrease_model_capacity` | `hidden_units × scale` (min 16); -1 layer if `scale ≤ 0.75` (min 1) | `n_estimators × scale` (min 10) |
| `modify_regularization` | `dropout ± 0.1` (clamped 0.0–0.5) | Adjusts `C` / `alpha` / `max_depth` |
| `change_optimizer` | Scales `lr`; sets `scheduler="cosine"` + bumps `weight_decay` if `scale ≤ 0.1` | No-op |

---

## DLPipelineResult

```python
@dataclass
class DLPipelineResult:
    model: nn.Module          # trained model (moved to CPU)
    y_val_pred: np.ndarray    # predictions on val set (decoded labels or floats)
    train_curve: list[float]  # per-epoch training loss
    val_curve: list[float]    # per-epoch validation loss
    runtime: float            # wall-clock seconds
    device_used: str          # "cuda", "mps", or "cpu"
    epochs_run: int           # actual epochs completed (< epochs if early stopped)
    early_stopped: bool       # True if early stopping triggered
    best_epoch: int           # epoch index with lowest val loss (used as convergence_epoch)
    model_state: dict         # CPU state_dict — used for .pth saving
```

`train_curve` and `val_curve` are always the same length (`epochs_run`).

`best_epoch` and `early_stopped` are forwarded into `ExperimentResult` and then into
`StateModel.convergence_epoch` / `StateModel.early_stopped` so the decision engine
can use them for convergence and stability signal assessment.

---

## Example ActionDecisions

**MLP with full production config:**
```json
{
  "action_type": "switch_model",
  "parameters": {
    "model_name": "MLP",
    "hidden_units": 128,
    "layers": 3,
    "dropout": 0.2,
    "batch_norm": true,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "grad_clip": 1.0,
    "scheduler": "plateau",
    "epochs": 50
  }
}
```

**RNN with mixed precision (GPU server):**
```json
{
  "action_type": "switch_model",
  "parameters": {
    "model_name": "RNN",
    "task": "classification",
    "hidden_units": 64,
    "layers": 2,
    "dropout": 0.1,
    "grad_clip": 0.5,
    "mixed_precision": true,
    "scheduler": "cosine",
    "epochs": 100
  }
}
```

**CNN1D regression, fast baseline:**
```json
{
  "action_type": "switch_model",
  "parameters": {
    "model_name": "CNN1D",
    "task": "regression",
    "hidden_units": 32,
    "dropout": 0.0,
    "epochs": 20
  }
}
```

---

## Architecture Selection Guide

| Scenario | Recommended |
|---|---|
| Standard tabular, no feature ordering | MLP |
| Features have local / spatial structure | CNN1D |
| Features represent a time sequence | RNN |
| Regression on tabular data | MLP with `task: regression` |
| Fast DL baseline | MLP, `hidden_units: 32`, `layers: 1`, `epochs: 10` |
| Regularisation needed | `dropout: 0.3` + `weight_decay: 1e-4` |
| GPU server, large dataset | `mixed_precision: true`, `batch_size: 256` |
| Unstable RNN training | Reduce `grad_clip` to `0.5` |

---

## Extending with a New Architecture

Add a new `nn.Module` class and register it in `_build_model`:

```python
class _Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, layers, dropout):
        ...

def _build_model(arch, input_dim, output_dim, hp):
    if arch == "TRANSFORMER":
        return _Transformer(input_dim, output_dim, ...)
    ...
```

Add `"TRANSFORMER"` to `_DL_MODELS` in `experiment_config_builder.py`.
No other changes needed in the orchestrator, metrics engine, or decision engine.
