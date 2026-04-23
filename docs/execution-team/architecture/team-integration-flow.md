# Decision Integration Flow

This document explains how **Team B's `ActionDecision` is integrated into Team A's execution pipeline** and how the full feedback loop operates.

---

# Roles

| Team   | Authority            | Does NOT                              |
|--------|----------------------|---------------------------------------|
| Team A | Execution            | Define rules, select actions          |
| Team B | Decision             | Implement models, touch training code |

---

# Full Integration Sequence

```
1. Team A generates DataProfile
2. Team B analyzes dataset → decides preprocessing + produces first ActionDecision
3. Team A translates ActionDecision → ExperimentConfig (model + preprocessing)
4. Team A executes preprocessing (cleaning, encoding, scaling, imbalance correction)
5. Team A executes training on cleaned data
6. Team A generates ExperimentResult (includes preprocessing_applied)
7. Team B builds StateObject from ExperimentResult
8. Team B runs decision pipeline → produces next ActionDecision
9. Repeat from step 3 until terminate
```

---

# Step 1 — DataProfile Handoff

Team A sends `DataProfile` to Team B once, before any training.

```
DataProfiler
     ↓
DataProfile
     ↓
Team B (meta_features.py + state_builder.py)
```

Team B uses this to:
- Compute `dataset_meta_features` for `StateObject`
- Inform the first `ActionDecision`

---

# Step 2 — Receive ActionDecision

Team A receives an `ActionDecision` from Team B.

Example:

```json
{
  "experiment_id": "exp_001",
  "action_type": "switch_model",
  "parameters": {
    "model_name": "RandomForestClassifier",
    "n_estimators": 100
  },
  "reason": "baseline experiment on dataset profile",
  "expected_gain": 0.0,
  "expected_cost": 3.0,
  "confidence": 1.0
}
```

---

# Step 3 — Translate to ExperimentConfig

The `ActionDecision` is mapped to an `ExperimentConfig` by Team A's orchestrator. This includes both the model configuration and the preprocessing instructions.

```
ActionDecision
      ↓
ExperimentConfig  (model config + PreprocessingConfig)
```

Example result:

```
ExperimentConfig
  model_name: RandomForestClassifier
  model_type: ml
  n_estimators: 100
  early_stopping: false
  preprocessing:
    missing_value_strategy: median
    scaling: standard
    encoding: onehot
    imbalance_strategy: none
    feature_selection: none
```

---

# Step 3b — Data Cleaning and Preprocessing

Before any model is initialized, Team A applies the preprocessing steps specified by the agent.

```
ExperimentConfig.preprocessing
      ↓
missing value imputation
      ↓
categorical encoding
      ↓
numerical scaling
      ↓
imbalance correction (if needed)
      ↓
feature selection (if needed)
      ↓
clean dataset ready for training
```

The agent decides *what* to apply based on `DataProfile` signals:

| DataProfile signal                    | Agent decision                          |
|---------------------------------------|-----------------------------------------|
| `missing_value_ratio > 0`             | set `missing_value_strategy`            |
| `categorical_columns` non-empty       | set `encoding`                          |
| `feature_summary.distribution=skewed` | set `scaling: minmax`                   |
| class imbalance in `class_distribution` | set `imbalance_strategy: oversample`  |

The exact preprocessing applied is recorded in `ExperimentResult.preprocessing_applied`.

---

# Step 4 — Pipeline Selection

Based on `model_type` in `ExperimentConfig`:

```
model_type: ml  → scikit-learn pipeline
model_type: dl  → PyTorch MLP pipeline
```

---

# Step 5 — Execute Training

```
Initialize model
      ↓
Load dataset splits
      ↓
Train model
      ↓
Validate model
      ↓
Collect metrics + curves
```

Logs pushed to:
- **MLflow** — hyperparameters, metrics, run metadata
- **TensorBoard** — loss curves (DL runs only)

---

# Step 6 — Generate ExperimentResult

After training completes:

```
raw training outputs
      ↓
MetricsEngine
      ↓
ArtifactManager
      ↓
ExperimentResult
      ↓
Team B (state_builder.py)
```

---

# Step 7 — Team B Decision Pipeline

Team B's pipeline on receiving `ExperimentResult`:

```
ExperimentResult
      ↓
state_builder.py       → StateObject
      ↓
state_history.py       → trajectory features injected
      ↓
meta_features.py       → dataset features injected
      ↓
signals.py             → underfitting / overfitting / instability scores
      ↓
action_generator.py    → CandidateAction list
      ↓
value_model.py         → predicted gain + cost per action
      ↓
calibration.py         → calibrated gain
      ↓
uncertainty.py         → confidence per action
      ↓
Decision Council       → performance / efficiency / stability agent scores
      ↓
coordinator_agent.py   → aggregated score
      ↓
policy_selector.py     → ActionDecision
      ↓
logger.py              → decision record saved
```

---

# Step 8 — Continuous Loop

```
ExperimentResult
      ↓
Team B Decision Pipeline
      ↓
ActionDecision
      ↓
Team A Execution
      ↓
ExperimentResult
```

Loop terminates when:
- Team B sends `action_type: terminate`
- Budget is exhausted (enforced by Team A's orchestrator)
- Performance goal is met

After the loop, Team A runs **test set evaluation** (ML models only):
- Loads best `model.pkl` from `outputs/<run_id>/artifacts/`
- Applies last-iteration preprocessing to `X_test`
- Computes metrics and saves `test_metrics.json`
- PDF report includes a "Test Set Metrics" section

---

# Architecture Diagram

```mermaid
flowchart TD

Dataset --> DataProfiler
DataProfiler -->|DataProfile| TeamB

TeamB -->|ActionDecision\n(model + preprocessing)| Orchestrator
Orchestrator --> ExperimentConfig
ExperimentConfig --> Preprocessor

Preprocessor --> PipelineSelector
PipelineSelector -->|ml| SklearnPipeline
PipelineSelector -->|dl| PyTorchPipeline

SklearnPipeline --> MetricsEngine
PyTorchPipeline --> MetricsEngine

MetricsEngine --> ArtifactManager
ArtifactManager -->|ExperimentResult\n(incl. preprocessing_applied)| TeamB

TeamB -->|next ActionDecision| Orchestrator
```

---

<!-- # Boundary Rules

Team A must not:
- Inspect `StateObject` internals
- Modify rule logic or action scoring
- Make decisions about which model to try next

Team B must not:
- Access raw training loops
- Modify `ExperimentConfig` directly
- Read artifact files from disk

The only objects that cross the boundary are `DataProfile`, `ExperimentResult`, and `ActionDecision`. -->
