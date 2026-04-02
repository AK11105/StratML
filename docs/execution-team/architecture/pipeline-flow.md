# Execution Pipeline — Components, Flow & Architecture

## Overview

The execution layer is a chain of independent components. Each component has one job, one input, one output. The orchestrator is the only thing that calls them in sequence.

```
Dataset (CSV)
      ↓
loader.py          → (DataFrame, dataset_name)
      ↓
validator.py       → Dataset
      ↓
profiler.py        → DataProfile  ──────────────────→ Team B (once)
      ↓
splitter.py        → DataSplit    (reused every iteration)
      ↓
◄─────────────────────────────────────────────────────────────────────
                    ActionDecision ← Team B (each iteration)
──────────────────────────────────────────────────────────────────────►
      ↓
experiment_config_builder.py  → ExperimentConfig
      ↓
preprocessor.py               → clean DataSplit
      ↓
ml_pipeline.py / dl_pipeline.py  → PipelineResult
      ↓
metrics_engine.py             → ExperimentMetrics
      ↓
artifact_manager.py           → ArtifactRefs
      ↓
result_builder.py             → ExperimentResult ──→ Team B (each iteration)
      ↓
◄─────────────────────────────────────────────────────────────────────
                    ActionDecision ← Team B (next iteration)
──────────────────────────────────────────────────────────────────────►
      ↓
[loop until action_type == "terminate" or budget exhausted]
```

---

## Component Reference

### Phase 1 — `execution/data/loader.py`

**Input:** file path (str)
**Output:** `(pd.DataFrame, dataset_name: str)`

Loads a CSV into a DataFrame. Returns the file stem as the dataset name.
Only `.csv` is supported currently.

```python
df, name = load_dataframe("data/iris.csv")
# name → "iris"
```

---

### Phase 1 — `execution/data/validator.py`

**Input:** `(DataFrame, dataset_name, target_column)`
**Output:** `Dataset`

Validates that the target column exists. Builds the internal `Dataset` object.
`dataset_type` is always `"tabular"` for now.

```python
dataset = build_dataset(df, "iris", "species")
# dataset.rows → 150
# dataset.columns → 6
```

---

### Phase 2 — `execution/data/profiler.py`

**Input:** `Dataset`
**Output:** `DataProfile`

Computes the full dataset profile. This is the only object sent to Team B before any training.

What it computes:
- numerical vs categorical column split
- global missing value ratio
- problem type: `classification` if target has ≤20 unique values or is object dtype, else `regression`
- class distribution (classification only)
- per-feature: dtype, unique count, missing %, distribution shape (Shapiro-Wilk + skewness → normal / skewed / uniform / unknown)
- recommended metrics based on problem type

```python
profile = build_profile(dataset)
# profile.problem_type → "classification"
# profile.recommended_metrics → ["accuracy", "f1_score"]
```

---

### Phase 3 — `execution/preprocessing/splitter.py`

**Input:** `Dataset`, `SplitConfig`, `problem_type`
**Output:** `DataSplit`

Splits the raw DataFrame into train / val / test. Called once per run, the same `DataSplit` is reused across all iterations.

- Classification → stratified split (preserves class ratios in all three sets)
- Regression → random split

Default split ratios: 70% train / 10% val / 20% test

```python
split = split_dataset(dataset, SplitConfig(method="stratified"), "classification")
# split.X_train.shape → (105, 5)
# split.X_val.shape   → (15, 5)
# split.X_test.shape  → (30, 5)
```

`DataSplit` is internal — it never crosses the Team A/B boundary.

---

### Phase 4 — `execution/config/experiment_config_builder.py`

**Input:** `ActionDecision` (from Team B)
**Output:** `ExperimentConfig`

Translates the decision into a runnable config. The `action_type` field drives what changes:

| action_type | What changes |
|---|---|
| `switch_model` | New model name, new hyperparameters |
| `increase_model_capacity` | Bump n_estimators / hidden_units / layers |
| `decrease_model_capacity` | Reduce complexity params |
| `modify_regularization` | Adjust dropout / l2_lambda / weight_decay |
| `change_optimizer` | Update optimizer / learning_rate |
| `apply_preprocessing` | Only preprocessing changes, model unchanged |
| `early_stop` | Sets early_stopping=True with patience |
| `terminate` | Caller must check — do not pass to builder |

`ActionDecision.preprocessing` is always extracted into `ExperimentConfig.preprocessing` regardless of action_type.

`model_type` is inferred from model name: `"MLP"` or `"PyTorchMLP"` → `"dl"`, everything else → `"ml"`.

```python
config = build_experiment_config(action)
# config.model_name → "RandomForestClassifier"
# config.model_type → "ml"
```

---

### Phase 4b — `execution/preprocessing/preprocessor.py`

**Input:** `DataSplit`, `PreprocessingConfig`, `DataProfile`
**Output:** `(clean DataSplit, PreprocessingConfig applied)`

Applies preprocessing in a fixed order. All transformers are fit on `X_train` only.

Order (must not change — each step depends on the previous):

1. **Missing value imputation** — mean / median / mode on numerical; most_frequent on categorical; or drop rows
2. **Categorical encoding** — one-hot (OneHotEncoder) or label (LabelEncoder per column)
3. **Numerical scaling** — StandardScaler or MinMaxScaler on numerical columns
4. **Imbalance correction** — SMOTE oversample or RandomUnderSampler, applied to train only
5. **Feature selection** — VarianceThreshold, removes zero-variance features

Returns the same `PreprocessingConfig` that was applied — this gets recorded in `ExperimentResult.preprocessing_applied`.

```python
clean_split, applied = apply_preprocessing(split, config.preprocessing, profile)
```

---

### Phase 5a — `execution/pipelines/ml_pipeline.py`

**Input:** `ExperimentConfig`, `DataSplit`
**Output:** `MLPipelineResult`

Instantiates a scikit-learn model from the registry, trains on `X_train`, evaluates on `X_val`.

Model registry:
```
LogisticRegression
RandomForestClassifier / Regressor
GradientBoostingClassifier / Regressor
SVC / SVR
```

ML has no epoch loop — `train_curve` and `val_curve` are single-element lists (log-loss if predict_proba is available, else 0.0).

```python
result = run_ml_pipeline(config, clean_split)
# result.y_val_pred → np.ndarray
# result.runtime    → 0.10 (seconds)
# result.train_curve → [0.035]
```

---

### Phase 5b — `execution/pipelines/dl_pipeline.py`

**Input:** `ExperimentConfig`, `DataSplit`
**Output:** `DLPipelineResult`

Builds a PyTorch MLP from hyperparameters, runs a full training loop with optional early stopping.

Hyperparameters read from `config.hyperparameters`:
- `hidden_units` (default 64)
- `layers` (default 2)
- `dropout` (default 0.0)
- `learning_rate` (default 1e-3)
- `batch_size` (default 32)
- `epochs` (default 20)

`train_curve` and `val_curve` are per-epoch loss lists. Early stopping triggers when val loss doesn't improve for `early_stopping_patience` epochs.

Labels are encoded to 0-based integers internally and decoded back before returning predictions.

```python
result = run_dl_pipeline(config, clean_split)
# result.train_curve → [0.82, 0.61, 0.45, ...]
# result.val_curve   → [0.85, 0.66, 0.50, ...]
```

---

### Phase 6 — `execution/metrics/metrics_engine.py`

**Input:** `y_true`, `y_pred`, `train_curve`, `val_curve`, `problem_type`
**Output:** `ExperimentMetrics`

Classification: accuracy, f1 (weighted), precision (weighted), recall (weighted), train_loss, validation_loss
Regression: mse, rmse, r2, train_loss, validation_loss

`train_loss` and `validation_loss` are always the last value in the respective curve.

```python
metrics = compute_metrics(y_val, y_pred, train_curve, val_curve, "classification")
# metrics.accuracy → 0.966
# metrics.f1_score → 0.966
```

---

### Phase 7 — `execution/artifacts/artifact_manager.py`

**Input:** `experiment_id`, `model`, `ExperimentMetrics`, `ExperimentConfig`
**Output:** `ArtifactRefs`

Saves to disk:
```
outputs/artifacts/{experiment_id}/
    model.pkl       ← joblib serialized model (sklearn) or state_dict (PyTorch)
    metrics.json    ← ExperimentMetrics as JSON
    config.json     ← ExperimentConfig as JSON
outputs/runs/{experiment_id}/
    (TensorBoard event files — DL only)
```

```python
refs = save_artifacts("exp_001", model, metrics, config)
# refs.model_path   → "outputs/artifacts/exp_001/model.pkl"
# refs.metrics_file → "outputs/artifacts/exp_001/metrics.json"
```

---

### Phase 8 — `execution/result_builder.py`

**Input:** all phase outputs
**Output:** `ExperimentResult`

Thin assembler. Combines config, metrics, curves, runtime, resource usage, artifacts, and preprocessing into a single Pydantic-validated `ExperimentResult`. This is the object sent back to Team B.

```python
result = build_experiment_result(config, metrics, train_curve, val_curve,
                                  runtime, resource_usage, artifacts,
                                  applied_preprocessing, iteration, dataset_name)
```

---

### Phase 9 — `orchestration/orchestrator.py`

**The main loop.** The only file that calls all other components.

Responsibilities:
- Runs phases 1–2 once at startup
- Splits the dataset once (phase 3), reuses across all iterations
- Sends `DataProfile` to Team B, receives first `ActionDecision`
- Each iteration: phases 4 → 4b → 5 → 6 → 7 → 8 → send result → receive next action
- Enforces time budget
- Terminates when `action_type == "terminate"` or budget exhausted

Team B is injected as two callables — the orchestrator has no knowledge of Team B internals:

```python
orchestrator = ExecutionOrchestrator(
    send_profile=team_b.receive_profile,   # DataProfile → ActionDecision
    send_result=team_b.receive_result,     # ExperimentResult → ActionDecision
    time_budget=600.0,                     # seconds
)
orchestrator.run("data/iris.csv", "species")
```

---

## Cross-Team Boundary

Only three objects cross the Team A / Team B boundary:

| Object | Direction | When |
|---|---|---|
| `DataProfile` | A → B | Once, before any training |
| `ActionDecision` | B → A | Each iteration (including iteration 0) |
| `ExperimentResult` | A → B | Each iteration after training |

Everything else (`Dataset`, `DataSplit`, `ExperimentConfig`, `MLPipelineResult`, etc.) is internal to Team A and never shared.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        TEAM A (Execution)                   │
│                                                             │
│  CSV ──► loader ──► validator ──► profiler                  │
│                                      │                      │
│                                  DataProfile ───────────────┼──► TEAM B
│                                      │                      │
│                                  splitter                   │
│                                  (once)                     │
│                                      │                      │
│                                  DataSplit ◄── reused ──┐   │
│                                      │                  │   │
│  ActionDecision ◄────────────────────┼──────────────────┼───┼──── TEAM B
│       │                              │                  │   │
│  config_builder                      │                  │   │
│       │                              │                  │   │
│  ExperimentConfig                    │                  │   │
│       │                              │                  │   │
│  preprocessor ◄──────────────────────┘                  │   │
│       │                                                  │   │
│  clean DataSplit                                         │   │
│       │                                                  │   │
│  ml_pipeline / dl_pipeline                               │   │
│       │                                                  │   │
│  metrics_engine                                          │   │
│       │                                                  │   │
│  artifact_manager                                        │   │
│       │                                                  │   │
│  result_builder                                          │   │
│       │                                                  │   │
│  ExperimentResult ───────────────────────────────────────┼───┼──► TEAM B
│                                                          │   │
│  [loop] ─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Rules

- All sklearn/torch transformers are fit on `X_train` only — never on val or test
- `DataSplit` is created once and reused — preprocessing creates a clean copy each iteration
- The orchestrator enforces budget — individual pipeline files do not
- Every output object is Pydantic-validated before being returned or saved
- `action_type == "terminate"` must be checked before calling `build_experiment_config`
- MLflow logging goes inside pipeline files; TensorBoard only in the DL pipeline (not yet wired — see next steps)
