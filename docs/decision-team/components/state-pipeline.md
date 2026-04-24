# Dev B — State Pipeline: Full Baseline Implementation

## Overview

This document covers the complete **rule-based baseline** for Dev B's responsibilities.
All five files are implemented and wired together.

---

## File Responsibilities

| File | Phase | Input | Output |
|---|---|---|---|
| `core/schemas.py` | 1 | — | All shared contracts (frozen) |
| `decision/state_builder.py` | 1 | ExperimentResult / DataProfile | StateObject |
| `decision/state_history.py` | 2 | ExperimentResult buffer | TrajectoryFeatures |
| `decision/meta_features.py` | 3 | DataProfile | DatasetMetaFeatures |
| `decision/signals.py` | 2 | StateObject | StateSignals |

---

## Full Pipeline

```
Iteration 0 (no experiment yet):
    DataProfile
        → build_state_from_profile()
        → meta_features.extract()       (num_features, entropy, imbalance)
        → StateObject  [iteration=0, all metrics=0, all signals=False]

Iteration 1+ (after each training run):
    ExperimentResult
        → state_history.push() + compute_trajectory()
              slope, volatility, best_score, mean_score,
              steps_since_improvement, runtime_trend
        → meta_features.extract()       (if DataProfile provided)
        → StateObject assembled
        → signals.compute_signals()     (trajectory-aware)
        → StateObject  [fully populated]
```

---

## state_history.py

Maintains a rolling buffer of the last 3 `ExperimentResult` objects.

### Computed features

| Feature | Formula |
|---|---|
| `improvement_rate` | `score_t − score_t-1` |
| `slope` | `(score_t − score_0) / (n − 1)` over window |
| `loss_slope` | same formula applied to validation_loss |
| `volatility` | `stdev(scores)` over window |
| `best_score` | `max(scores)` in buffer |
| `mean_score` | `mean(scores)` in buffer |
| `steps_since_improvement` | iterations without a new best score |
| `runtime_trend` | `runtime_t − runtime_t-1` |
| `model_switch_frequency` | count of model name changes in window |
| `trend` | `improving` / `stagnating` / `degrading` based on slope |

### Usage

```python
from decision.state_history import ExperimentHistory

history = ExperimentHistory()   # create once, share across iterations
history.push(result)
features = history.compute_trajectory(primary_metric="accuracy")
```

---

## meta_features.py

Computes dataset-level characteristics from a `DataProfile`. Called once per dataset (or per iteration if profile is passed to `build_state`).

### Computed features

| Feature | Formula |
|---|---|
| `num_samples` | `DataProfile.rows` |
| `num_features` | `DataProfile.columns − 1` (exclude target) |
| `feature_sample_ratio` | `num_features / num_samples` |
| `class_entropy` | `−Σ p·log₂(p)` over class distribution |
| `missing_value_ratio` | `DataProfile.missing_value_ratio` |
| `feature_variance_mean` | mean unique-value count across features (proxy) |
| `imbalance_ratio` | `max_class_count / min_class_count` |

### Usage

```python
from decision.meta_features import extract

features = extract(profile)
# features.class_entropy, features.imbalance_ratio, etc.
```

---

## signals.py

Converts a `StateObject` (with trajectory context) into a fully populated `StateSignals` block. Uses slope and `steps_since_improvement` rather than single-step improvement rate.

### Signal rules

| Signal | Rule |
|---|---|
| `underfitting` | `primary < 0.60` |
| `overfitting` | `val_loss − train_loss > 0.10` |
| `well_fitted` | not under/over AND `primary ≥ 0.75` |
| `converged` | `\|slope\| < 0.001` AND `primary ≥ 0.75` |
| `stagnating` | `steps_since_improvement ≥ 2` AND not converged |
| `diverging` | `slope < −0.02` |
| `unstable_training` | `val_loss > train_loss × 2.0` |
| `high_variance` | `volatility > 0.05` |
| `too_slow` | `runtime > 300s` |
| `plateau_detected` | `steps_since_improvement ≥ 3` |
| `diminishing_returns` | `0 < improvement_rate < 0.005` |

### Usage

```python
from decision.signals import compute_signals

signals = compute_signals(state)   # called automatically inside build_state()
```

---

## state_builder.py — Updated API

### Iteration 0

```python
from decision.state_builder import build_state_from_profile

state = build_state_from_profile(
    profile,
    primary_metric="accuracy",
    optimization_goal="maximize",
    allowed_models=["RandomForest", "XGBoost"],
    max_iterations=20,
)
# state.meta.iteration == 0
# state.meta.experiment_id == "bootstrap"
# state.dataset populated from DataProfile
# all signals False, all metrics 0.0
```

### Iteration 1+

```python
from decision.state_builder import build_state
from decision.state_history import ExperimentHistory

history = ExperimentHistory()   # shared across all iterations

state = build_state(
    result,
    history=history,
    profile=profile,            # optional, populates dataset fields
    primary_metric="accuracy",
    optimization_goal="maximize",
    allowed_models=["RandomForest", "XGBoost"],
    max_iterations=20,
    previous_action="switch_model",
    previous_action_success=True,
    models_tried=["LogisticRegression", "RandomForest"],
    remaining_budget=18.0,
)
```

---

## Integration Point with Dev A

```
Dev B produces:   StateObject   (via build_state or build_state_from_profile)
Dev A consumes:   StateObject   → CandidateAction list → ActionDecision
```

Dev A reads:
- `state.signals` — to generate candidate actions
- `state.trajectory` — to assess momentum
- `state.constraints` — to filter allowed actions
- `state.resources` — to check budget

---

## What Comes Next (Later Phases)

| Phase | File | Replaces |
|---|---|---|
| 4 | `learning/value_model.py` | rule thresholds in `signals.py` |
| 4 | `learning/uncertainty.py` | `state.uncertainty` defaults (None) |
| 4 | `agents/` | binary signals → scored evaluations |
