# Execution Team — Implementation & Testing Steps

## Current Status

| Phase | File | Status |
|---|---|---|
| 1 | `execution/data/loader.py` | ✅ Done |
| 1 | `execution/data/validator.py` | ✅ Done |
| 2 | `execution/data/profiler.py` | ✅ Done |
| 2 | `execution/schemas.py` | ✅ Done (extended with SplitConfig, ExperimentConfig, DataSplit) |
| 3 | `execution/preprocessing/splitter.py` | ✅ Done |
| 4 | `execution/config/experiment_config_builder.py` | ✅ Done |
| 4b | `execution/preprocessing/preprocessor.py` | ✅ Done |
| 5a | `execution/pipelines/ml_pipeline.py` | ✅ Done |
| 5b | `execution/pipelines/dl_pipeline.py` | ✅ Done |
| 6 | `execution/metrics/metrics_engine.py` | ✅ Done |
| 7 | `execution/artifacts/artifact_manager.py` | ✅ Done |
| 8 | `execution/result_builder.py` | ✅ Done |
| 9 | `orchestration/orchestrator.py` | ✅ Done |

---

## What Remains to Implement

### 1. Observability (not yet wired)

These files exist as empty `__init__.py` stubs. Need to be implemented before production use.

**`stratml/observability/mlflow_logger.py`**
- Log hyperparameters, metrics, run metadata to MLflow
- Called inside `ml_pipeline.py` and `dl_pipeline.py` after training
- Requires `enable_mlflow: true` in config

**`stratml/observability/tensorboard.py`**
- Log per-epoch loss curves to TensorBoard event files
- Called inside `dl_pipeline.py` during the training loop
- Writes to `outputs/runs/{experiment_id}/`

**`stratml/observability/langsmith.py`**
- Log decision traces (Team B's responsibility, but Team A needs to pass experiment_id)

### 2. CLI `run` command — connect to orchestrator

`stratml/cli/main.py` → `run_pipeline()` currently prints a placeholder:
```
[Placeholder] Orchestrator not yet connected.
```

Needs to:
1. Instantiate `ExecutionOrchestrator`
2. Wire Team B's interface (once Team B's `agent.py` / `bootstrap.py` is ready)
3. Call `orchestrator.run(dataset_path, target_column)`

### 3. Team B integration

The orchestrator expects two callables:
```python
send_profile: DataProfile → ActionDecision
send_result:  ExperimentResult → ActionDecision
```

Once Team B implements `stratml/decision/bootstrap.py` (iteration 0) and `stratml/decision/agent.py` (iteration 1+), these get wired into the orchestrator.

### 4. Extend ML model registry

Current registry has 7 models. Add before integration testing:
- `DecisionTreeClassifier` / `DecisionTreeRegressor`
- `KNeighborsClassifier` / `KNeighborsRegressor`
- `LinearRegression` / `Ridge` / `Lasso`
- `XGBClassifier` / `XGBRegressor` (if xgboost is added to dependencies)

### 5. DL pipeline — regression support

Current `dl_pipeline.py` only handles classification (CrossEntropyLoss + label encoding).
Needs a regression branch: MSELoss, no label encoding, single output neuron.

### 6. `SplitConfig` schema discrepancy

`execution/schemas.py` has `SplitConfig` but it is not yet exported from `execution/__init__.py`.
Add to `stratml/execution/__init__.py`:
```python
from stratml.execution.schemas import SplitConfig, ExperimentConfig, DataSplit
```

---

## Testing Plan

### Unit Tests

Each component should be tested in isolation with a small synthetic DataFrame.

**File:** `tests/unit/test_splitter.py`
```
- classification split preserves class ratios in all three sets
- regression split uses random (no stratify)
- val + test sizes sum correctly
- reset_index produces clean 0-based index
```

**File:** `tests/unit/test_experiment_config_builder.py`
```
- switch_model sets correct model_name and hyperparameters
- increase_model_capacity passes params through
- apply_preprocessing keeps model_name unchanged
- early_stop sets early_stopping=True
- terminate raises ValueError
- model_type inferred correctly (MLP → dl, RandomForest → ml)
```

**File:** `tests/unit/test_preprocessor.py`
```
- imputation: mean fills NaN in numerical columns
- imputation: drop removes rows with any NaN
- onehot encoding expands categorical columns
- label encoding maps categories to integers
- standard scaling: X_train mean ≈ 0, std ≈ 1
- minmax scaling: X_train values in [0, 1]
- variance_threshold removes zero-variance columns
- all transformers fit on X_train only (val/test not used for fitting)
```

**File:** `tests/unit/test_ml_pipeline.py`
```
- unknown model_name raises ValueError
- returns MLPipelineResult with correct fields
- runtime > 0
- y_val_pred length matches X_val length
- train_curve and val_curve are single-element lists
```

**File:** `tests/unit/test_metrics_engine.py`
```
- classification: accuracy, f1, precision, recall all populated; mse/rmse/r2 are None
- regression: mse, rmse, r2 all populated; accuracy/f1 are None
- train_loss = last value of train_curve
- validation_loss = last value of val_curve
```

**File:** `tests/unit/test_result_builder.py`
```
- output is a valid ExperimentResult (Pydantic validates)
- experiment_id matches config.experiment_id
- iteration matches input
- preprocessing_applied matches input
```

---

### Integration Tests

Test the full pipeline end-to-end without Team B (use a stub ActionDecision).

**File:** `tests/integration/test_full_pipeline_iris.py`

```python
# Stub: simulates Team B returning a fixed ActionDecision
def stub_send_profile(profile):
    return ActionDecision(
        experiment_id="test_001",
        action_type="switch_model",
        parameters={"model_name": "RandomForestClassifier", "n_estimators": 10},
        preprocessing=PreprocessingConfig(
            missing_value_strategy="mean", scaling="standard",
            encoding="none", imbalance_strategy="none", feature_selection="none"
        ),
        reason="test", expected_gain=0.0, expected_cost=1.0, confidence=1.0
    )

def stub_send_result(result):
    return ActionDecision(..., action_type="terminate", ...)

orchestrator = ExecutionOrchestrator(
    send_profile=stub_send_profile,
    send_result=stub_send_result,
)
orchestrator.run("data/iris.csv", "species")
# Assert: outputs/artifacts/test_001/model.pkl exists
# Assert: outputs/artifacts/test_001/metrics.json is valid JSON
```

**File:** `tests/integration/test_full_pipeline_housing.py`

Same pattern but with housing dataset (regression), `RandomForestRegressor`, and checking mse/rmse/r2 instead of accuracy.

**File:** `tests/integration/test_multi_iteration.py`

Simulate 3 iterations with different action_types:
1. `switch_model` → RandomForestClassifier
2. `increase_model_capacity` → n_estimators=200
3. `terminate`

Assert that 2 artifact directories are created and metrics improve or stay stable.

---

### Pipeline-Specific Tests

**File:** `tests/unit/test_dl_pipeline.py`
```
- MLP builds with correct layer count
- train_curve length == epochs (or less if early stopped)
- early stopping triggers when val loss stagnates
- y_val_pred decoded back to original label types
- runtime > 0
```

---

## How to Run Tests

```bash
# Activate environment
source .venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run only unit tests
python -m pytest tests/unit/ -v

# Run only integration tests
python -m pytest tests/integration/ -v

# Run a specific test file
python -m pytest tests/unit/test_splitter.py -v
```

---

## Quick Manual Smoke Test (no pytest needed)

Run this to verify the full pipeline works end-to-end:

```bash
cd /path/to/StratML
source .venv/bin/activate

python -c "
from stratml.execution.data.loader import load_dataframe
from stratml.execution.data.validator import build_dataset
from stratml.execution.data.profiler import build_profile
from stratml.execution.preprocessing.splitter import split_dataset
from stratml.execution.preprocessing.preprocessor import apply_preprocessing
from stratml.execution.config.experiment_config_builder import build_experiment_config
from stratml.execution.pipelines.ml_pipeline import run_ml_pipeline
from stratml.execution.metrics.metrics_engine import compute_metrics
from stratml.execution.schemas import SplitConfig, ActionDecision, PreprocessingConfig, ResourceUsage, ArtifactRefs
from stratml.execution.result_builder import build_experiment_result

df, name = load_dataframe('data/iris.csv')
dataset  = build_dataset(df, name, 'species')
profile  = build_profile(dataset)
split    = split_dataset(dataset, SplitConfig(method='stratified'), profile.problem_type)

action = ActionDecision(
    experiment_id='smoke_001',
    action_type='switch_model',
    parameters={'model_name': 'RandomForestClassifier', 'n_estimators': 50},
    preprocessing=PreprocessingConfig(
        missing_value_strategy='mean', scaling='standard',
        encoding='none', imbalance_strategy='none', feature_selection='none'
    ),
    reason='smoke test', expected_gain=0.0, expected_cost=1.0, confidence=1.0
)

config       = build_experiment_config(action)
clean, prep  = apply_preprocessing(split, config.preprocessing, profile)
pipe_result  = run_ml_pipeline(config, clean)
metrics      = compute_metrics(clean.y_val, pipe_result.y_val_pred,
                               pipe_result.train_curve, pipe_result.val_curve,
                               profile.problem_type)
print('accuracy:', metrics.accuracy)
print('PASSED')
"
```

---

## Folder Structure (Current State)

```
stratml/
├── execution/
│   ├── data/
│   │   ├── loader.py              ✅
│   │   ├── validator.py           ✅
│   │   └── profiler.py            ✅
│   ├── preprocessing/
│   │   ├── splitter.py            ✅
│   │   └── preprocessor.py        ✅
│   ├── config/
│   │   └── experiment_config_builder.py  ✅
│   ├── pipelines/
│   │   ├── ml_pipeline.py         ✅
│   │   └── dl_pipeline.py         ✅
│   ├── metrics/
│   │   └── metrics_engine.py      ✅
│   ├── artifacts/
│   │   └── artifact_manager.py    ✅
│   ├── result_builder.py          ✅
│   └── schemas.py                 ✅
├── orchestration/
│   └── orchestrator.py            ✅
├── observability/
│   ├── mlflow_logger.py           ⬜ not implemented
│   ├── tensorboard.py             ⬜ not implemented
│   └── langsmith.py               ⬜ not implemented
├── decision/                      ⬜ Team B (in progress)
└── cli/
    └── main.py                    ⚠️  run command not wired to orchestrator yet

tests/
├── unit/                          ⬜ not written yet
└── integration/                   ⬜ not written yet
```

---

## Dependency Notes

All current dependencies are in `pyproject.toml`. For imbalanced-learn (SMOTE):

```bash
uv add imbalanced-learn
# or
pip install imbalanced-learn
```

If not installed, the preprocessor silently skips imbalance correction (no crash).
