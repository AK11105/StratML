# DL Pipeline — Architectures, Tasks & Configuration

## Overview

`stratml/execution/pipelines/dl_pipeline.py` is a single entry point (`run_dl_pipeline`) that handles all PyTorch training. The architecture and task are controlled entirely through `config.hyperparameters` — no code changes needed to switch between them.

---

## Supported Tasks

Set via `config.hyperparameters["task"]`:

| Value | Loss function | Output layer | Prediction |
|---|---|---|---|
| `"classification"` (default) | CrossEntropyLoss | Linear → N classes | argmax → decoded label |
| `"regression"` | MSELoss | Linear → 1 neuron | raw float |

For classification, labels are encoded to 0-based integers internally and decoded back to original values before returning predictions. Team B and the metrics engine never see the encoded integers.

---

## Supported Architectures

Set via `config.hyperparameters["architecture"]`:

### MLP (default)

Fully connected network. Best for standard tabular data.

```
Input (features)
    ↓
[Linear → ReLU → Dropout] × layers
    ↓
Linear → output
```

Hyperparameters: `hidden_units`, `layers`, `dropout`

### CNN1D

1-D convolutional network. Treats the feature vector as a sequence — useful when features have local structure (e.g. ordered sensor readings, time-windowed features).

```
Input (batch, features)
    ↓  unsqueeze → (batch, 1, features)
Conv1d(1 → hidden_units, kernel=3)  → ReLU → Dropout
Conv1d(hidden_units → hidden_units//2, kernel=3)  → ReLU
    ↓  flatten
Linear → output
```

Hyperparameters: `hidden_units`, `dropout`

### RNN (LSTM)

LSTM-based network. Treats each feature as a time step of a single-channel sequence. Useful when features represent a temporal or ordered progression.

```
Input (batch, features)
    ↓  unsqueeze → (batch, features, 1)
LSTM(input=1, hidden=hidden_units, num_layers=layers)
    ↓  last time step hidden state
Linear → output
```

Hyperparameters: `hidden_units`, `layers`, `dropout` (applied between LSTM layers when layers > 1)

---

## Full Hyperparameter Reference

| Key | Type | Default | Description |
|---|---|---|---|
| `architecture` | str | `"MLP"` | `"MLP"`, `"CNN1D"`, or `"RNN"` |
| `task` | str | `"classification"` | `"classification"` or `"regression"` |
| `hidden_units` | int | `64` | Width of hidden layers |
| `layers` | int | `2` | Depth (MLP: FC layers, RNN: LSTM layers) |
| `dropout` | float | `0.0` | Dropout probability |
| `learning_rate` | float | `1e-3` | Adam optimizer LR |
| `batch_size` | int | `32` | Mini-batch size |
| `epochs` | int | `20` | Max training epochs |

---

## Early Stopping

Controlled by `ExperimentConfig.early_stopping` and `early_stopping_patience`.

- Monitors validation loss each epoch
- Stops if val loss doesn't improve by more than `1e-4` for `patience` consecutive epochs
- `train_curve` and `val_curve` will be shorter than `epochs` when triggered

```json
{
  "action_type": "early_stop",
  "parameters": {
    "model_name": "MLP",
    "early_stopping_patience": 5
  }
}
```

---

## Example ActionDecisions

**MLP classification (default):**
```json
{
  "action_type": "switch_model",
  "parameters": {
    "model_name": "MLP",
    "hidden_units": 128,
    "layers": 3,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "epochs": 50
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
    "epochs": 30
  }
}
```

**RNN classification with early stopping:**
```json
{
  "action_type": "early_stop",
  "parameters": {
    "model_name": "RNN",
    "task": "classification",
    "hidden_units": 32,
    "layers": 2,
    "epochs": 100,
    "early_stopping_patience": 7
  }
}
```

---

## Output

All architectures and tasks return the same `DLPipelineResult`:

```python
@dataclass
class DLPipelineResult:
    model: nn.Module          # trained PyTorch model
    y_val_pred: np.ndarray    # predictions on val set (decoded labels or floats)
    train_curve: list[float]  # per-epoch training loss
    val_curve: list[float]    # per-epoch validation loss
    runtime: float            # wall-clock seconds
```

`train_curve` and `val_curve` are always the same length (number of epochs actually run).

---

## Architecture Selection Guide

| Scenario | Recommended |
|---|---|
| Standard tabular, no feature ordering | MLP |
| Features have local/spatial structure | CNN1D |
| Features represent a time sequence | RNN |
| Regression on tabular data | MLP with `task: regression` |
| Fast baseline DL run | MLP, `hidden_units: 32`, `layers: 1`, `epochs: 10` |

---

## Extending with a New Architecture

Add a new `nn.Module` class and register it in `_build_model`:

```python
class _Transformer(nn.Module):
    ...

def _build_model(arch, input_dim, output_dim, hp):
    if arch == "TRANSFORMER":
        return _Transformer(input_dim, output_dim, hp)
    if arch == "CNN1D":
        ...
```

No changes needed in the orchestrator, config builder, or metrics engine.
