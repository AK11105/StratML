# ML Model Registry — Reference

## Overview

`stratml/execution/pipelines/ml_pipeline.py` maintains a flat registry dict mapping model name strings to sklearn classes. The orchestrator passes a model name via `ActionDecision.parameters.model_name` → `ExperimentConfig.model_name`, and the pipeline looks it up at runtime.

Adding a new model = one line in `MODEL_REGISTRY`.

---

## Full Registry (23 models)

### Linear

| Name | Task | Key hyperparameters |
|---|---|---|
| `LogisticRegression` | classification | `C`, `max_iter`, `solver` |
| `LinearRegression` | regression | — |
| `Ridge` | regression | `alpha` |
| `Lasso` | regression | `alpha` |
| `ElasticNet` | regression | `alpha`, `l1_ratio` |
| `SGDClassifier` | classification | `loss`, `alpha`, `learning_rate` |
| `SGDRegressor` | regression | `loss`, `alpha`, `learning_rate` |

### Tree

| Name | Task | Key hyperparameters |
|---|---|---|
| `DecisionTreeClassifier` | classification | `max_depth`, `min_samples_split` |
| `DecisionTreeRegressor` | regression | `max_depth`, `min_samples_split` |

### Ensemble

| Name | Task | Key hyperparameters |
|---|---|---|
| `RandomForestClassifier` | classification | `n_estimators`, `max_depth` |
| `RandomForestRegressor` | regression | `n_estimators`, `max_depth` |
| `ExtraTreesClassifier` | classification | `n_estimators`, `max_depth` |
| `ExtraTreesRegressor` | regression | `n_estimators`, `max_depth` |
| `GradientBoostingClassifier` | classification | `n_estimators`, `learning_rate`, `max_depth` |
| `GradientBoostingRegressor` | regression | `n_estimators`, `learning_rate`, `max_depth` |
| `AdaBoostClassifier` | classification | `n_estimators`, `learning_rate` |
| `AdaBoostRegressor` | regression | `n_estimators`, `learning_rate` |

### SVM

| Name | Task | Key hyperparameters |
|---|---|---|
| `SVC` | classification | `C`, `kernel`, `gamma` |
| `SVR` | regression | `C`, `kernel`, `epsilon` |

### Neighbors

| Name | Task | Key hyperparameters |
|---|---|---|
| `KNeighborsClassifier` | classification | `n_neighbors`, `weights`, `metric` |
| `KNeighborsRegressor` | regression | `n_neighbors`, `weights`, `metric` |

### Probabilistic

| Name | Task | Key hyperparameters |
|---|---|---|
| `GaussianNB` | classification | `var_smoothing` |
| `LinearDiscriminantAnalysis` | classification | `solver`, `shrinkage` |

---

## How Hyperparameters Are Applied

The pipeline filters `config.hyperparameters` to only the keys accepted by the model's `__init__`. Unknown keys are silently dropped — no crash.

```python
valid_params = inspect.signature(cls.__init__).parameters
hp = {k: v for k, v in config.hyperparameters.items() if k in valid_params}
model = cls(**hp)
```

This means Team B can safely pass extra keys (e.g. `architecture`, `task`) without breaking ML runs.

---

## Hyperparameter Tuning

When `ExperimentConfig.tune=True` and the model name has an entry in `_PARAM_GRIDS`, the pipeline runs `RandomizedSearchCV` (10 iterations, 3-fold CV, `n_jobs=-1`) and returns the best estimator.

`_PARAM_GRIDS` covers: RandomForest, ExtraTrees, GradientBoosting, LogisticRegression, SVC, SVR, Ridge, Lasso, KNeighbors, DecisionTree (classifier and regressor variants).

Models without a grid entry fall back to single-shot training with the provided hyperparameters.

> Note: `tune=True` is not yet exposed via CLI or set by the config builder — it is implemented but must be set programmatically.

---

## Loss Curves

ML models have no epoch loop. `train_curve` and `val_curve` are always single-element lists.

- If the model has `predict_proba` → uses `log_loss` as the loss value
- Otherwise → `[0.0]`

This keeps the `ExperimentResult` schema consistent between ML and DL runs.

---

## How Team B Selects a Model

Team B sends an `ActionDecision` with `action_type` and `parameters`:

```json
{
  "action_type": "switch_model",
  "parameters": {
    "model_name": "GradientBoostingClassifier",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4
  }
}
```

The `experiment_config_builder` extracts `model_name`, passes the rest as `hyperparameters`. The pipeline filters to valid params and trains.

---

## Adding a New Model

```python
# In ml_pipeline.py — MODEL_REGISTRY
from xgboost import XGBClassifier, XGBRegressor

MODEL_REGISTRY["XGBClassifier"] = XGBClassifier
MODEL_REGISTRY["XGBRegressor"]  = XGBRegressor
```

No other changes needed. The pipeline, config builder, and orchestrator all work automatically.
