# Decision Pipeline — Full Flow

This document describes the current implemented pipeline from experiment result to action
decision. It reflects the actual code, not a plan.

---

## High-Level Flow

```
User / Orchestrator
        │
        ├─ receive_profile(DataProfile)     ← iteration 0 only
        │
        └─ receive_result(ExperimentResult) ← iteration 1+
                │
                ▼
        DecisionEngine (engine.py)
                │
                ▼
        StateObject construction
        (state_builder.py)
                │
                ├── ExperimentHistory → TrajectoryFeatures
                ├── DatasetMetaFeatures (meta_features.py)
                └── SignalExtraction (signals.py)
                │
                ▼
        ActionGenerator (action_generator.py)
        → list[CandidateAction]
                │
                ▼
        ValueModel (value_model.py)
        → list[ValuePrediction]
                │
                ▼
        Calibration (calibration.py)
        → list[ValuePrediction] (calibrated)
                │
                ▼
        Uncertainty (uncertainty.py)
        → list[UncertaintyEstimate]
                │
                ▼
        Decision Council
        ├── PerformanceAgent → dict[action_type, score]
        ├── EfficiencyAgent  → dict[action_type, score]
        └── StabilityAgent   → dict[action_type, score]
                │
                ▼
        CoordinatorAgent (coordinator_agent.py)
        → list[RankedAction]
                │
                ▼
        ActionSelector (action_selector.py)
        → ActionDecision
                │
                ├── DecisionLogger  (decision_logger.py)
                ├── DatasetBuilder  (dataset_builder.py)
                └── Counterfactual  (counterfactual.py)  ← stub
                │
                ▼
        ActionDecision returned to Orchestrator
```

---

## Entry Points

### Iteration 0 — `receive_profile(profile: DataProfile)`

Called once before any experiment runs. Builds a bootstrap `StateObject` from the data
profile alone (no metrics yet). The action generator always uses the rule-based bootstrap
path at iteration 0 — no LLM call.

### Iteration 1+ — `receive_result(result: ExperimentResult)`

Called after every experiment. Builds a full `StateObject` from the result, history, and
profile. Runs the complete pipeline.

---

## State Construction

`state_builder.py` has two public functions:

- `build_state_from_profile()` — iteration 0, minimal state, no metrics
- `build_state()` — iteration 1+, full state including trajectory, signals, generalization

The `StateObject` is the single shared contract between state construction and all
downstream decision components. Nothing downstream reads `ExperimentResult` directly.

### StateObject top-level fields

| Field | Source |
|---|---|
| `meta` | iteration, experiment_id, timestamp |
| `objective` | primary_metric, optimization_goal |
| `metrics` | primary score, secondary metrics, train/val gap |
| `trajectory` | slope, trend, volatility, best_score, steps_since_improvement |
| `dataset` | num_samples, num_features, imbalance_ratio, missing_ratio |
| `model` | model_name, model_type, hyperparameters, complexity_hint |
| `generalization` | train_loss, val_loss, gap |
| `resources` | runtime, remaining_budget, budget_exhausted |
| `search` | models_tried, repeated_configs |
| `signals` | all diagnostic flags (see below) |
| `constraints` | allowed_models, max_iterations, time_budget |
| `action_context` | previous_action, previous_action_success |

---

## Signal Extraction

`signals.py` uses a LangGraph ReAct agent (llama-3.3-70b-versatile) with five tools,
falling back to direct rule evaluation when `GROQ_API_KEY` is absent.

### Signal groups

| Tool | Signals produced |
|---|---|
| `assess_fitting` | underfitting, overfitting, well_fitted |
| `assess_convergence` | converged, stagnating, diverging |
| `assess_stability` | unstable_training, high_variance |
| `assess_efficiency` | too_slow |
| `assess_optimization` | plateau_detected, diminishing_returns |

Each signal is `"none" | "weak" | "strong"` with an associated confidence float.

---

## Action Generation

`action_generator.py` produces `list[CandidateAction]`. Three paths:

1. **Bootstrap** (iteration 0) — always rule-based, returns two `switch_model` candidates
2. **LLM** (iteration 1+, GROQ_API_KEY present) — proposes 2–4 context-sensitive candidates
3. **Rule fallback** — if/else tree over signals, always includes `terminate`

Valid action types: `switch_model`, `increase_model_capacity`, `decrease_model_capacity`,
`modify_regularization`, `change_optimizer`, `add_preprocessing`, `terminate`.

---

## Value Model & Learning Pipeline

| Component | File | Active when |
|---|---|---|
| Value model | `value_model.py` | ≥ 50 rows with `observed_gain` in decision dataset |
| Calibration | `calibration.py` | ≥ 50 rows with `(predicted, actual)` pairs |
| Uncertainty | `uncertainty.py` | Always (variance = 0 in stub mode) |

Below 50 rows all three return neutral stub values. The dataset accumulates across runs
at `runs/decision_logs/decision_dataset.csv`.

---

## Decision Council

Three specialist agents score each candidate independently:

| Agent | Concern | LLM prompt focus |
|---|---|---|
| `performance_agent` | Accuracy gain | Fitting state + trajectory |
| `efficiency_agent` | Compute cost | Runtime + budget remaining |
| `stability_agent` | Training risk | Divergence + variance signals |

Each agent has a rule-based lookup table fallback. The `coordinator_agent` then resolves
disagreements and produces a ranked list. Fallback weights: performance=0.50,
efficiency=0.25, stability=0.25.

---

## Action Selection

`action_selector.py` picks `ranked[0]` (currently greedy — see improvements roadmap).
Builds the final `ActionDecision` with trigger, evidence dict, and source
(`"learned"` if LLM coordinator path was taken, `"rule"` otherwise).

---

## Outputs per Run

All outputs go to `outputs/<run_id>/`:

```
outputs/<run_id>/
├── decision_logs/
│   ├── decision_<iteration>.json     ← per-iteration decision log
│   ├── decision_dataset.csv          ← learning dataset (run-scoped copy)
│   └── counterfactual_log.jsonl      ← stub log
└── ...
```

Unified learning dataset (accumulates across all runs):
```
runs/decision_logs/decision_dataset.csv
```
