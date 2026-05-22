# Execution Critique

## Artifacts

- ✅ ML artifacts (model.pkl, metrics.json, config.json) — done
- ✅ MLflow logging wired — `enable_mlflow=True` in orchestrator/CLI triggers `mlflow.start_run` + metric/artifact logging inside `artifact_manager.py`
- ⚠️ DL artifacts: model saved as `.pkl` (full `nn.Module` via joblib). Explicit `.pth` state-dict saving and optimizer state not yet implemented
- ⚠️ TensorBoard: log dir path is set but `SummaryWriter` calls are not wired inside `dl_pipeline.py`

## Config

- ✅ `_CAPACITY_PARAMS` dict added — covers RF, ExtraTrees, GBM, AdaBoost, DecisionTree, LR, SVC, KNN, SGD with per-model parameter lists
- ✅ `_REG_PARAM` covers LogisticRegression, SVC, Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, ExtraTrees, DecisionTree
- ✅ `change_optimizer` adjusts `learning_rate` via `learning_rate_scale`
- ⚠️ DL-specific capacity actions (e.g. `hidden_units`, `layers`, `weight_decay`) not yet covered by `_CAPACITY_PARAMS`
- ⚠️ `tune=True` flag exists in `ExperimentConfig` and is respected by `ml_pipeline.py`, but is not yet exposed via CLI or set by the config builder

## Data

- ✅ Loader supports CSV, TSV, JSON, Parquet, Excel
- ✅ Validator hardened: zero-row check, duplicate column names, all-null column drop, target null/uniqueness checks
- ✅ Problem type inference improved: float target with many unique values relative to dataset size → regression (avoids misclassifying rounded-price targets)
- ✅ `imbalance_ratio`, `feature_variance_mean`, `class_entropy` added to `DataProfile` and computed in profiler
- ⚠️ DL-specific data handling (explicit `torch.utils.data.Dataset` / `DataLoader` wrapping) not yet separated from the pipeline

## Validation

- ✅ Zero rows, duplicate columns, all-null columns, target entirely null, target with < 2 unique values — all raise or warn correctly
- ⚠️ `dataset_type` is hardcoded `"tabular"` — multi-modal handling not yet implemented

## Pipelines

- ✅ 24 sklearn models in registry
- ✅ `_PARAM_GRIDS` defined for 15 model families; `tune=True` triggers `RandomizedSearchCV` (10 iterations, 3-fold CV)
- ✅ DL pipeline supports MLP, CNN1D, RNN; classification and regression; early stopping with best-weight restore
- ⚠️ `tune` flag not yet wired through CLI/config builder — implemented but unreachable from user-facing commands

## Preprocessing

- ✅ Fit-on-train-only discipline enforced throughout
- ✅ `RobustScaler` added as `scaling: "robust"` option (for skewed features)
- ✅ `active_num_cols` recomputed after one-hot encoding step (avoids KeyError when cat cols are dropped)
- ✅ Label encoding closure bug fixed — `known` set captured before lambda
- ✅ Missing `imbalanced-learn` now emits a `UserWarning` instead of silently doing nothing

## Orchestrator

- ✅ Test set evaluation implemented for ML models: loads best `model.pkl`, applies last-iteration preprocessing to `X_test`, computes and saves `test_metrics.json`
- ✅ `enable_mlflow` flag threaded from orchestrator constructor through to `artifact_manager`
- ⚠️ Test set evaluation not yet implemented for DL models

## Report

- ✅ PDF includes "Test Set Metrics" section when `test_metrics.json` exists
- ⚠️ Training curve plots (matplotlib embeds) not yet in PDF — iteration table only

## Schema

- Must reference all schemas from `core/schemas.py` — that must remain the single source of truth
- `execution/schemas.py` still duplicates `ExperimentMetrics`, `ResourceUsage`, `ArtifactRefs`, `ExperimentResult`, `ActionDecision` — these should be imports, not redefinitions
