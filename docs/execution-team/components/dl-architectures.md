# DL Pipeline — Architectures, Tasks & Configuration

## Overview

`stratml/execution/pipelines/dl_pipeline.py` is the single entry point (`run_dl_pipeline`)
for all PyTorch training. Architecture, task, and training behaviour are controlled
entirely through `config.hyperparameters` — no code changes needed to switch between them.

---

## Device Selection

The pipeline automatically selects the best available device:

```
CUDA (NVIDIA GPU)  →  MPS (Apple Silicon)  →  CPU
```

`DLPipelineResult.device_used` reports which device was used. `ResourceUsage.gpu_used`
is set to `True` when CUDA or MPS is active.

---

## Supported Tasks

Set via `config.hyperparameters["task"]`:

| Value | Loss | Output | Prediction |
|---|---|---|---|
| `"classification"` (default) | CrossEntropyLoss | Linear → N classes | argmax → decoded original label |
| `"regression"` | MSELoss | Linear → 1 neuron | raw float |

Labels are encoded to 0-based integers internally and decoded back before returning
predictions. The metrics engine and decision engine never see encoded integers.

---

## Supported Architectures

Set via `config.hyperparameters["architecture"]`:

### MLP (default)

Fully-connected network for standard tabular data.

```
Input (F features)
  ↓
[Linear(H) → BatchNorm(H) → ReLU → Dropout] × L layers
  ↓
Linear(output_dim)
```

BatchNorm is applied per layer when `batch_norm=True` (default: off).
Hyperparameters: `hidden_units`, `layers`, `dropout`, `batch_norm`

---

### CNN1D

1-D convolutional network. Treats the feature vector as a 1-channel sequence of
length F. Useful when features have local or spatial structure (e.g. ordered sensor
readings, windowed time-series features).

```
Input (batch, F)
  ↓  unsqueeze  →  (batch, 1, F)
Conv1d(1 → H, kernel=3, pad=1)  →  [BatchNorm]  →  ReLU  →  Dropout
Conv1d(H → H//2, kernel=3, pad=1)  →  [BatchNorm]  →  ReLU
  ↓  flatten  →  (batch, H//2 * F)
Linear(output_dim)
```

Hyperparameters: `hidden_units`, `dropout`, `batch_norm`

---

### RNN (LSTM)

LSTM-based network. Each feature is treated as a time step of a single-channel
sequence. Suitable when features represent a temporal or ordered progression.

```
Input (batch, F)
  ↓  unsqueeze  →  (batch, F, 1)
LSTM(input=1, hidden=H, num_layers=L, dropout between layers)
  ↓  last hidden state
Linear(output_dim)
```

Hyperparameters: `hidden_units`, `layers`, `dropout`

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
| `batch_size` | int | `32` | Mini-batch size |
| `epochs` | int | `20` | Maximum training epochs |
| `scheduler` | str | `"plateau"` | LR scheduler: `"plateau"`, `"cosine"`, or `"none"` |

---

## Learning Rate Schedulers

Controlled via `config.hyperparameters["scheduler"]`:

| Value | Behaviour |
|---|---|
| `"plateau"` (default) | `ReduceLROnPlateau` — halves LR when val loss stops improving for 3 epochs |
| `"cosine"` | `CosineAnnealingLR` — smoothly decays LR to 0 over `epochs` |
| `"none"` | Fixed LR throughout training |

The `change_optimizer` action automatically switches to `"cosine"` when the LR scale
factor is ≤ 0.1 (aggressive reduction).

---

## Early Stopping

Early stopping is **always active** for DL models. The patience is set via
`ExperimentConfig.early_stopping_patience` (default: 5).

- Monitors validation loss each epoch
- Stops if val loss doesn't improve by more than `1e-4` for `patience` consecutive epochs
- **Best weights are restored** after stopping — the model returned is the one from the
  epoch with the lowest validation loss, not the final epoch
- `DLPipelineResult.early_stopped` is `True` when triggered
- `DLPipelineResult.epochs_run` reports how many epochs actually ran

---

## TensorBoard Integration

Pass `tensorboard_log_dir` to `run_dl_pipeline` to write training curves:

```python
result = run_dl_pipeline(config, data_split, tensorboard_log_dir="outputs/runs/exp_001")
```

This writes `Loss/train` and `Loss/val` scalars per epoch. View with:

```bash
tensorboard --logdir outputs/runs/
```

The orchestrator automatically sets this to `outputs/<run_id>/tensorboard/<experiment_id>/`
for every DL run.

---

## Artifact Saving

For DL models, two model files are saved:

| File | Format | Use |
|---|---|---|
| `model.pkl` | joblib | Used by `model.py` script and report generator |
| `model.pth` | `torch.save` of state_dict + metadata | Proper PyTorch format for resuming training |

The `.pth` file contains:
```python
{
    "state_dict":      model.state_dict(),   # weights
    "architecture":    "MLP",                # arch string
    "hyperparameters": {...},                # full hp dict
    "experiment_id":   "exp_001",
}
```

To load and resume:
```python
import torch
checkpoint = torch.load("model.pth")
# rebuild model with same arch + hp, then:
model.load_state_dict(checkpoint["state_dict"])
```

---

## Capacity Mutations (Decision Engine Actions)

The config builder handles DL capacity actions differently from ML:

| Action | DL behaviour | ML behaviour |
|---|---|---|
| `increase_model_capacity` | `hidden_units × scale`; adds a layer if `scale ≥ 1.5` | `n_estimators × scale` |
| `decrease_model_capacity` | `hidden_units × scale` (min 16); removes a layer if `scale ≤ 0.75` | `n_estimators × scale` (min 10) |
| `modify_regularization` | Adjusts `dropout` by ±0.1 (clamped 0.0–0.5) | Adjusts `C` / `alpha` / `max_depth` |
| `change_optimizer` | Scales `learning_rate`; sets `scheduler="cosine"` if scale ≤ 0.1 | No-op (sklearn has no LR) |

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
    model_state: dict         # state_dict on CPU — used for .pth saving
```

`train_curve` and `val_curve` are always the same length (`epochs_run`).

---

## Architecture Selection Guide

| Scenario | Recommended |
|---|---|
| Standard tabular, no feature ordering | MLP |
| Features have local / spatial structure | CNN1D |
| Features represent a time sequence | RNN |
| Regression on tabular data | MLP with `task: regression` |
| Fast DL baseline | MLP, `hidden_units: 32`, `layers: 1`, `epochs: 10` |
| Regularisation needed | MLP with `dropout: 0.3` or `batch_norm: true` |

---

## Example ActionDecisions

**MLP classification with BatchNorm:**
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
    "epochs": 50,
    "scheduler": "plateau"
  }
}
```

**CNN1D regression:**
```json
{
  "action_type": "switch_model",
  "parameters": {
    "model_name": "CNN1D",
    "task": "regression",
    "hidden_units": 64,
    "dropout": 0.1,
    "epochs": 30,
    "scheduler": "cosine"
  }
}
```

**RNN classification:**
```json
{
  "action_type": "switch_model",
  "parameters": {
    "model_name": "RNN",
    "task": "classification",
    "hidden_units": 32,
    "layers": 2,
    "epochs": 100
  }
}
```

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
    if arch == "CNN1D":
        ...
```

Add `"TRANSFORMER"` to `_DL_MODELS` in `experiment_config_builder.py`. No other
changes needed in the orchestrator, metrics engine, or decision engine.
