# Dev B — State Pipeline: Phase 1 Implementation

## Overview

This document covers the **Phase 1** deliverables for **Dev B (State Pipeline)** on the Decision Team.

Phase 1 establishes the working rule-based loop:

```
ExperimentResult  →  StateObject  →  ActionDecision  (Dev A consumes)
```

---

## Deliverables

| File | Responsibility |
|---|---|
| `core/schemas.py` | All shared contracts (ExperimentResult, StateObject, ActionDecision, …) |
| `decision/state_builder.py` | Converts ExperimentResult → StateObject |
| `decision/run_state_builder.py` | CLI runner for manual testing |

---

## Folder Structure After Phase 1

```
StratML/
│
├── core/
│   ├── __init__.py
│   └── schemas.py              ← ALL shared contracts (frozen after Phase 1)
│
├── decision/
│   ├── __init__.py
│   ├── state_builder.py        ← Phase 1: ExperimentResult → StateObject
│   ├── run_state_builder.py    ← Phase 1: CLI runner
│   │
│   ├── state_history.py        ← Phase 2 (trajectory features)
│   ├── meta_features.py        ← Phase 3 (dataset meta-features)
│   ├── signals.py              ← Phase 2 (rich signal extraction)
│   ├── action_generator.py     ← Dev A Phase 1
│   ├── policy_selector.py      ← Dev A Phase 1
│   ├── logger.py               ← Dev A later
│   │
│   ├── agents/
│   │   ├── performance_agent.py
│   │   ├── efficiency_agent.py
│   │   ├── stability_agent.py
│   │   └── coordinator_agent.py
│   │
│   ├── learning/
│   │   ├── dataset_builder.py
│   │   ├── value_model.py
│   │   ├── uncertainty.py
│   │   └── calibration.py
│   │
│   └── validation/
│       └── counterfactual.py
│
└── outputs/
    └── <dataset_name>/
        ├── data_profile.json   ← Team A output
        └── state_object.json   ← Team B Phase 1 output
```

---

## Phase 1 Scope

Phase 1 produces a **fully valid StateObject** from a single ExperimentResult.

Fields populated in Phase 1:

| Section | Fields | Source |
|---|---|---|
| `meta` | experiment_id, iteration, timestamp | ExperimentResult |
| `objective` | primary_metric, optimization_goal | CLI config |
| `metrics` | primary, secondary, train_val_gap | ExperimentResult.metrics |
| `trajectory` | improvement_rate, trend, history_length | ExperimentResult + previous |
| `model` | model_name, model_type, hyperparameters, complexity_hint, runtime | ExperimentResult |
| `generalization` | train_loss, validation_loss, gap | ExperimentResult.metrics |
| `resources` | runtime, gpu_used, cpu_time, budget_exhausted | ExperimentResult.resource_usage |
| `search` | models_tried, unique_models_count, repeated_configs | Caller-maintained list |
| `signals` | underfitting, overfitting, well_fitted, converged, stagnating, … | Rule-based (Phase 1) |

Fields left as safe defaults for later phases:

| Field | Populated by |
|---|---|
| `trajectory.slope`, `volatility`, `best_score`, `mean_score` | `state_history.py` (Phase 2) |
| `dataset.num_features`, `feature_to_sample_ratio`, `missing_ratio` | `meta_features.py` (Phase 3) |
| `uncertainty.prediction_variance`, `confidence` | `learning/uncertainty.py` (Phase 4) |

---

## Signal Rules (Phase 1)

All signals are derived deterministically from the current ExperimentResult.

| Signal | Rule |
|---|---|
| `underfitting` | `primary_metric < 0.60` |
| `overfitting` | `val_loss − train_loss > 0.10` |
| `well_fitted` | not underfitting AND not overfitting AND primary ≥ 0.75 |
| `converged` | `\|improvement_rate\| < 0.001` AND primary ≥ 0.75 |
| `stagnating` | `\|improvement_rate\| < 0.001` AND not converged |
| `diverging` | `improvement_rate < −0.05` |
| `unstable_training` | `val_loss > train_loss × 2.0` |
| `high_variance` | `gap > 0.20` |
| `too_slow` | `runtime > 300s` |
| `plateau_detected` | same as stagnating |
| `diminishing_returns` | `0 < improvement_rate < 0.005` |

These thresholds are intentionally conservative. Phase 2 (`signals.py`) will refine them using trajectory context.

---

## Improvement Rate

```
improvement_rate = current_primary − previous_primary   (maximize)
improvement_rate = previous_primary − current_primary   (minimize)
```

On iteration 0 (no previous result), `improvement_rate = 0.0`.

---

## Usage

### Run the state builder manually

```bash
# First iteration
python3 decision/run_state_builder.py outputs/iris/experiment_result.json

# Subsequent iteration (pass previous result for improvement_rate)
python3 decision/run_state_builder.py outputs/iris/experiment_result_2.json \
                                      outputs/iris/experiment_result_1.json
```

### Use in code

```python
from core.schemas import ExperimentResult
from decision.state_builder import build_state

state = build_state(
    result,
    previous_result=previous,
    primary_metric="accuracy",
    optimization_goal="maximize",
    allowed_models=["RandomForest", "XGBoost"],
    max_iterations=20,
    models_tried=["LogisticRegression", "RandomForest"],
)
# state is a fully validated StateObject — pass to Dev A's action_generator
```

---

## Integration Point with Dev A

```
Dev B produces:   StateObject
Dev A consumes:   StateObject  →  [CandidateAction, ...]  →  ActionDecision
```

Dev A's `action_generator.py` receives the `StateObject` and reads `state.signals` to generate candidate actions.

---

## What Comes Next

| Phase | File | Owner |
|---|---|---|
| Phase 2 | `decision/state_history.py` | Dev B |
| Phase 2 | `decision/signals.py` | Dev B |
| Phase 3 | `decision/meta_features.py` | Dev B |
| Phase 1 | `decision/action_generator.py` | Dev A |
| Phase 1 | `decision/policy_selector.py` | Dev A |
