# Team A — Execution Pipeline Implementation Phases

## Overview

Team A owns the **Execution Layer** of the StratML system.

Responsibilities:

- Dataset ingestion and profiling
- Translating `ActionDecision` → runnable experiment configuration
- Training ML/DL models
- Logging metrics and artifacts
- Returning structured `ExperimentResult` to Team B

Team A does **not** define decision rules, select actions, or modify agent logic.

---

# Phase 1 — Dataset Ingestion

**Objective:** Load the dataset and prepare it for profiling.

Tasks:
- Load dataset from path (CSV supported; extensible to other formats)
- Validate schema and detect target column
- Determine dataset type (`tabular` | `text` | `vision`)

Output: internal `Dataset` object.

```
Dataset
├── dataset_name
├── rows
├── columns
├── target_column
└── raw_dataframe
```

---

# Phase 2 — Data Profiling

**Objective:** Generate a `DataProfile` and send it to Team B before any training begins.

This is the **only time Team A initiates communication**. All subsequent communication is response-driven.

Tasks:
- Compute feature types (numerical / categorical)
- Compute missing value ratios per feature and globally
- Compute class distribution (classification) or target statistics (regression)
- Infer `problem_type`
- Recommend evaluation metrics

Output: `DataProfile` (see `schemas_and_interface.md`).

```
Team A → DataProfile → Team B
```

Team B uses this to populate `StateObject.dataset_meta_features` and plan the first experiment.

---

# Phase 3 — Await ActionDecision from Team B

After sending `DataProfile`, Team A waits for Team B's first `ActionDecision`.

Team B's agent analyzes the dataset profile and returns an `ActionDecision` specifying:
- which model to run
- what hyperparameters to use
- what action type is being requested

```
Team B → ActionDecision → Team A
```

---

# Phase 4 — Experiment Configuration

**Objective:** Translate `ActionDecision` into a runnable `ExperimentConfig`.

The mapping depends on `action_type`:

| action_type               | Config change                                      |
|---------------------------|----------------------------------------------------|
| `switch_model`            | Set `model_name` and reset hyperparameters         |
| `increase_model_capacity` | Increase `n_estimators`, `hidden_units`, `layers`  |
| `decrease_model_capacity` | Reduce model complexity parameters                 |
| `modify_regularization`   | Adjust `dropout`, `l2_lambda`, `weight_decay`      |
| `change_optimizer`        | Update `optimizer` and/or `learning_rate`          |
| `apply_preprocessing`     | Update preprocessing config, keep model unchanged  |
| `early_stop`              | Set `early_stopping=True` with patience parameter  |
| `terminate`               | Exit the experiment loop                           |

The `ActionDecision.preprocessing` block is always extracted and stored as part of `ExperimentConfig` regardless of `action_type`.

Example:

```bash
ActionDecision:
  action_type: switch_model
  parameters:
    model_name: GradientBoostingClassifier
    n_estimators: 200
  preprocessing:
    missing_value_strategy: median
    scaling: standard
    encoding: onehot
    imbalance_strategy: oversample
    feature_selection: none
```

```bash
ExperimentConfig:
  model_name: GradientBoostingClassifier
  model_type: ml
  n_estimators: 200
  preprocessing:
    missing_value_strategy: median
    scaling: standard
    encoding: onehot
    imbalance_strategy: oversample
    feature_selection: none
```

---

# Phase 4b — Data Cleaning and Preprocessing Execution

**Objective:** Apply the preprocessing steps specified by the agent before training begins.

This phase runs after `ExperimentConfig` is built and before the model is initialized.

The agent decides *what* to apply based on `DataProfile`. Team A executes it.

Steps executed in order:

1. **Missing value handling** — apply `missing_value_strategy` (mean/median/mode imputation or row drop)
2. **Encoding** — apply `encoding` to `categorical_columns` from `DataProfile` (one-hot or label encoding). Label encoding closure bug fixed.
3. **Scaling** — apply `scaling` to numerical columns (StandardScaler, MinMaxScaler, RobustScaler, or none). `active_num_cols` recomputed after encoding to avoid KeyError on dropped columns.
4. **Imbalance correction** — apply `imbalance_strategy` if classification and class imbalance detected (SMOTE oversample or random undersample). Emits `UserWarning` if `imbalanced-learn` is not installed.
5. **Feature selection** — apply `feature_selection` if specified (variance threshold filter)

The exact `PreprocessingConfig` that was applied is recorded in `ExperimentResult.preprocessing_applied` so Team B can track what was done across iterations.

---

# Phase 5 — Pipeline Selection and Training

**Objective:** Select the appropriate pipeline and execute training.

Pipeline selection:
- `model_type: ml` → scikit-learn pipeline
- `model_type: dl` → PyTorch MLP pipeline (tabular)

Training responsibilities:
- Initialize model from config
- For ML: if `config.tune=True` and model has a param grid, run `RandomizedSearchCV` (10 iterations, 3-fold CV); otherwise single-shot fit
- Train on training split
- Validate on validation split
- Capture per-epoch loss curves (DL) or single-pass metrics (ML)
- Apply early stopping if configured

Logging during training:
- MLflow: hyperparameters, metrics, run metadata
- TensorBoard: loss curves (DL runs)

---

# Phase 6 — Metrics Processing

**Objective:** Convert raw training outputs into structured `ExperimentMetrics`.

Classification metrics: `accuracy`, `f1_score`, `precision`, `recall`, `train_loss`, `validation_loss`

Regression metrics: `mse`, `rmse`, `r2`, `train_loss`, `validation_loss`

Training curves captured as lists:
- `train_curve`: per-epoch training loss
- `validation_curve`: per-epoch validation loss

---

# Phase 7 — Artifact Management

**Objective:** Persist all experiment artifacts in a structured layout.

```
artifacts/
  {experiment_id}/
      model.pkl          # serialized trained model
      metrics.json       # final metrics snapshot
      config.json        # exact ExperimentConfig used
runs/
  {experiment_id}/       # TensorBoard event files
```

Artifact paths are recorded in `ArtifactRefs` inside `ExperimentResult`.

---

# Phase 8 — ExperimentResult Generation

**Objective:** Assemble and return the `ExperimentResult` to Team B.

This is the primary output of each execution cycle.

The result includes:
- experiment identity (`experiment_id`, `iteration`, `model_name`)
- full `ExperimentMetrics`
- `train_curve` and `validation_curve`
- `runtime` in seconds
- `resource_usage`
- `artifacts` references

```
Team A → ExperimentResult → Team B
```

Team B's `state_builder.py` consumes this to construct the next `StateObject`.

---

# Phase 9 — Iterative Execution Loop

After returning `ExperimentResult`, Team A waits for the next `ActionDecision`.

```
ActionDecision (from Team B)
        ↓
ExperimentConfig (Team A translates)
        ↓
Training Execution
        ↓
ExperimentResult (Team A returns)
        ↓
Team B Decision Loop
```

Loop terminates when Team B sends `action_type: terminate` or budget is exhausted.

---

# Full Interaction Flow

```
Dataset
   ↓
Data Profiling (Team A)
   ↓
DataProfile → Team B
   ↓
ActionDecision ← Team B  (includes preprocessing decisions)
   ↓
ExperimentConfig
   ↓
Data Cleaning & Preprocessing (Team A executes agent's instructions)
   ↓
Training Execution
   ↓
ExperimentResult → Team B  (includes preprocessing_applied)
   ↓
[repeat from ActionDecision]
```
