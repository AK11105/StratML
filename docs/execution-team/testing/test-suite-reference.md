# Execution Team — Test Suite Reference

## Structure

```
tests/
├── conftest.py              ← shared fixtures (datasets, splits, configs)
├── smoke/
│   └── test_smoke.py        ← fast sanity checks after every commit
├── unit/
│   ├── test_splitter.py     ← split sizes, stratification, reproducibility
│   ├── test_config_builder.py ← action_type mapping, model_type inference
│   ├── test_preprocessor.py ← imputation, scaling, encoding, feature selection
│   ├── test_metrics_engine.py ← classification vs regression outputs
│   ├── test_ml_pipeline.py  ← registry, output shapes, regression models
│   └── test_result_builder.py ← field preservation, Pydantic validation
└── integration/
    └── test_full_pipeline.py ← end-to-end loop with stub Team B
```

**Total: 100 tests** — 27 smoke, 62 unit, 11 integration.

---

## How to Run

```bash
# Activate environment
source .venv/bin/activate

# Run everything
python -m pytest tests/ -v

# Run only smoke tests (fastest — ~3s)
python -m pytest tests/smoke/ -v

# Run only unit tests
python -m pytest tests/unit/ -v

# Run only integration tests
python -m pytest tests/integration/ -v

# Run a single file
python -m pytest tests/unit/test_splitter.py -v

# Run a single test
python -m pytest tests/unit/test_splitter.py::TestStratification::test_class_ratios_preserved_in_train -v

# Stop on first failure
python -m pytest tests/ -x

# Show print output
python -m pytest tests/ -s
```

---

## Smoke Tests (`tests/smoke/test_smoke.py`)

Purpose: confirm nothing crashes after a commit. No detailed value assertions.

| Class | What it checks |
|---|---|
| `TestImports` | All execution modules + schemas + core schemas import cleanly |
| `TestLoaderSmoke` | CSV loads, missing file raises, unsupported format raises |
| `TestProfilerSmoke` | Iris (classification) and housing (regression) profiles run |
| `TestSplitterSmoke` | Iris split produces non-empty train/val/test |
| `TestMLPipelineSmoke` | RandomForest on iris runs; all 23 registry models instantiate |
| `TestDLPipelineSmoke` | MLP cls, MLP reg, CNN1D, RNN all run without error |

Run these after every commit — they complete in ~3 seconds.

---

## Unit Tests

### `test_splitter.py` — 12 tests

Tests `execution/preprocessing/splitter.py`.

| Class | Tests |
|---|---|
| `TestSplitSizes` | Total rows preserved, test/val sizes within 5% of target, X/y lengths match |
| `TestStratification` | Class ratios preserved in train and val, regression uses random split |
| `TestIndexReset` | train/val/test all start at index 0 after reset |
| `TestReproducibility` | Same seed → same split; different seed → different split |

Key invariant tested: stratified split preserves class distribution across all three sets.

---

### `test_config_builder.py` — 15 tests

Tests `execution/config/experiment_config_builder.py`.

| Class | Tests |
|---|---|
| `TestActionTypeMapping` | All 7 action types map correctly; `terminate` raises `ValueError` |
| `TestModelTypeInference` | MLP/PyTorchMLP → `"dl"`, all sklearn models → `"ml"` |
| `TestPreprocessingPassthrough` | Preprocessing config copied verbatim; experiment_id preserved |

Key invariant tested: `terminate` must raise — the orchestrator checks this before calling the builder.

---

### `test_preprocessor.py` — 14 tests

Tests `execution/preprocessing/preprocessor.py`.

| Class | Tests |
|---|---|
| `TestImputation` | Mean fills NaN; drop removes rows; imputer fit on train only (val NaN filled with train mean) |
| `TestScaling` | Standard: train mean≈0, std≈1; MinMax: train in [0,1]; scaler fit on train only (val can exceed [0,1]) |
| `TestEncoding` | OneHot expands columns; Label produces integers; None leaves columns unchanged |
| `TestFeatureSelection` | VarianceThreshold removes zero-variance column from train, val, and test |
| `TestReturnValue` | Returns same PreprocessingConfig; original split not mutated |

Key invariant tested throughout: **all transformers fit on X_train only**. The `test_imputer_not_fit_on_val` and `test_scaler_fit_on_train_only` tests specifically verify this by constructing splits where train and val have different value ranges.

---

### `test_metrics_engine.py` — 12 tests

Tests `execution/metrics/metrics_engine.py`.

| Class | Tests |
|---|---|
| `TestClassificationMetrics` | accuracy/f1/precision/recall populated; mse/rmse/r2 are None; train_loss = last curve value; perfect predictions → accuracy=1.0 |
| `TestRegressionMetrics` | mse/rmse/r2 populated; rmse = sqrt(mse); accuracy/f1 are None; perfect predictions → mse≈0 |

Key invariant tested: classification and regression outputs are mutually exclusive — no cross-contamination.

---

### `test_ml_pipeline.py` — 11 tests

Tests `execution/pipelines/ml_pipeline.py`.

| Class | Tests |
|---|---|
| `TestRegistry` | Expected model families present; regression models present; unknown model raises |
| `TestPipelineOutput` | Pred length matches val set; runtime > 0; train/val curves are single-element; model object returned; invalid hyperparams silently dropped |
| `TestRegressionModel` | RandomForestRegressor and Ridge both run on regression split |

Key invariant tested: invalid hyperparameter keys are silently dropped (not a crash) — Team B can safely pass extra keys.

---

### `test_result_builder.py` — 9 tests

Tests `execution/result_builder.py`.

| Class | Tests |
|---|---|
| `TestResultBuilder` | experiment_id, model_name, model_type, iteration, dataset_name, metrics, curves, runtime all preserved; Pydantic validation passes |

This is a thin assembler — tests confirm field pass-through and that the output is a valid Pydantic model.

---

## Integration Tests (`tests/integration/test_full_pipeline.py`) — 11 tests

Tests the full phases 1–8 pipeline end-to-end using real datasets and a stub `ActionDecision` (no real Team B).

| Class | Tests |
|---|---|
| `TestIrisPipeline` | Valid ExperimentResult; classification metrics populated; accuracy > 0.7; train curve non-empty; runtime > 0; model_type = "ml" |
| `TestHousingPipeline` | Valid ExperimentResult; regression metrics populated; R² > 0.5 |
| `TestMultiIteration` | 3 iterations with 3 different models all produce valid results; `increase_model_capacity` action type works end-to-end |

These tests use real data from `data/iris.csv` and `data/housing.csv`. They take ~10 seconds.

---

## Shared Fixtures (`tests/conftest.py`)

Available to all test files automatically via pytest.

| Fixture | Type | Description |
|---|---|---|
| `default_prep` | `PreprocessingConfig` | mean/standard/none/none/none |
| `clf_df` | `pd.DataFrame` | 120 rows, 4 features, 3 balanced classes |
| `clf_dataset` | `Dataset` | Dataset wrapping clf_df |
| `clf_profile` | `DataProfile` | Classification profile for clf_df |
| `reg_df` | `pd.DataFrame` | 100 rows, 4 features, float target |
| `reg_dataset` | `Dataset` | Dataset wrapping reg_df |
| `reg_profile` | `DataProfile` | Regression profile for reg_df |
| `clf_split` | `DataSplit` | Pre-split classification data (70/15/15) |
| `reg_split` | `DataSplit` | Pre-split regression data (70/15/15) |
| `base_action` | `ActionDecision` | switch_model → RandomForestClassifier, n_estimators=10 |
| `base_config` | `ExperimentConfig` | RandomForestClassifier, ml, n_estimators=10 |

---

## What Is Not Tested Yet

| Component | Reason |
|---|---|
| `artifact_manager.py` | Requires disk I/O — add when testing with tmp_path fixture |
| `dl_pipeline.py` unit tests | Covered by smoke tests; detailed unit tests (early stopping, curve length) to be added |
| `orchestrator.py` | Requires Team B stub — add as integration test once Team B bootstrap is ready |
| `loader.py` format tests | JSON/Parquet/Excel require sample files — add with fixture files |
