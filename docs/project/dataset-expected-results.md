# Dataset Expected CLI Outputs

Expected terminal output for each dataset when running the full pipeline. Based on the actual CLI print statements in `stratml/cli/main.py` and `stratml/orchestration/orchestrator.py`.

All examples use `config.yaml` with `max_iterations: 5` and `mode: beginner` unless noted.

---

## Command Reference

```bash
# Profile a dataset
stratml profile-data data/pima.csv Outcome

# Run the pipeline
stratml run config.yaml --path data/pima.csv

# Dry run (no training)
stratml run config.yaml --path data/pima.csv --dry-run
```

---

## 1. Pima Indians Diabetes (Binary Classification)

```
stratml run config.yaml --path data/pima.csv
```

```
  AutoML Pipeline Starting
  --------------------------------------------
  Mode    : beginner
  Dataset : data/pima.csv
  Target  : Outcome
  Budget  : 5 iterations
  --------------------------------------------
  Loading dataset...
  Profiled: 768 rows x 9 cols | classification
  Split: train=491 | val=123 | test=154
  Sending profile to Decision Engine...
  Decision [iter 0]: action=switch_model | params={'model_name': 'RandomForestClassifier'} | trigger=bootstrap

  --- Iteration 1 ---
  Training : RandomForestClassifier (switch_model) ...
  Trained in 0.84s
  Result   : primary=0.7398 | runtime=0.84s
  Evaluating signals & deciding next action...
  Decision : switch_model | trigger=underfitting | confidence=0.71 | next={'model_name': 'GradientBoostingClassifier'}

  --- Iteration 2 ---
  Training : GradientBoostingClassifier (switch_model) ...
  Trained in 1.12s
  Result   : primary=0.7642 | runtime=1.12s
  Evaluating signals & deciding next action...
  Decision : modify_regularization | trigger=overfitting | confidence=0.68 | next={'direction': 'increase'}

  --- Iteration 3 ---
  Training : GradientBoostingClassifier (modify_regularization) ...
  Trained in 1.09s
  Result   : primary=0.7724 | runtime=1.09s
  Evaluating signals & deciding next action...
  Decision : add_preprocessing | trigger=overfitting | confidence=0.62 | next={'strategy': 'oversample'}

  --- Iteration 4 ---
  Training : GradientBoostingClassifier (add_preprocessing) ...
  Trained in 1.31s
  Result   : primary=0.7886 | runtime=1.31s
  Evaluating signals & deciding next action...
  Decision : switch_model | trigger=stagnating | confidence=0.59 | next={'model_name': 'ExtraTreesClassifier'}

  --- Iteration 5 ---
  Training : ExtraTreesClassifier (switch_model) ...
  Trained in 0.76s
  Result   : primary=0.7724 | runtime=0.76s
  Evaluating signals & deciding next action...
  Decision : terminate | trigger=budget_exhausted | confidence=1.00 | next={}
  Run complete.

  --------------------------------------------
  Run ID  : pima_20260516_040800
  Output  : outputs/pima_20260516_040800
  --------------------------------------------
  Report    : outputs/pima_20260516_040800/report.pdf
  Comparison: outputs/pima_20260516_040800/comparison.csv
  Model.py  : outputs/pima_20260516_040800/model.py

  Download best model files (model.pkl + model.py)? [y/N]:
```

**What to observe:** `add_preprocessing` with `strategy: oversample` fires because `imbalance_ratio > 2.0`. The counterfactual log will record the runner-up action at each step.

---

## 2. Wine Quality Red (Multiclass Classification)

```
stratml run config.yaml --path data/wine_quality_red.csv
```

```
  AutoML Pipeline Starting
  --------------------------------------------
  Mode    : beginner
  Dataset : data/wine_quality_red.csv
  Target  : quality
  Budget  : 5 iterations
  --------------------------------------------
  Loading dataset...
  Profiled: 1599 rows x 12 cols | classification
  Split: train=1023 | val=256 | test=320
  Sending profile to Decision Engine...
  Decision [iter 0]: action=switch_model | params={'model_name': 'RandomForestClassifier'} | trigger=bootstrap

  --- Iteration 1 ---
  Training : RandomForestClassifier (switch_model) ...
  Trained in 1.43s
  Result   : primary=0.6289 | runtime=1.43s
  Evaluating signals & deciding next action...
  Decision : switch_model | trigger=underfitting | confidence=0.78 | next={'model_name': 'GradientBoostingClassifier'}

  --- Iteration 2 ---
  Training : GradientBoostingClassifier (switch_model) ...
  Trained in 3.21s
  Result   : primary=0.6523 | runtime=3.21s
  Evaluating signals & deciding next action...
  Decision : increase_model_capacity | trigger=underfitting | confidence=0.72 | next={'scale': 1.5}

  --- Iteration 3 ---
  Training : GradientBoostingClassifier (increase_model_capacity) ...
  Trained in 4.87s
  Result   : primary=0.6680 | runtime=4.87s
  Evaluating signals & deciding next action...
  Decision : switch_model | trigger=underfitting | confidence=0.65 | next={'model_name': 'ExtraTreesClassifier'}

  --- Iteration 4 ---
  Training : ExtraTreesClassifier (switch_model) ...
  Trained in 1.18s
  Result   : primary=0.6445 | runtime=1.18s
  Evaluating signals & deciding next action...
  Decision : switch_model | trigger=stagnating | confidence=0.61 | next={'model_name': 'SVC'}

  --- Iteration 5 ---
  Training : SVC (switch_model) ...
  Trained in 2.34s
  Result   : primary=0.6602 | runtime=2.34s
  Evaluating signals & deciding next action...
  Decision : terminate | trigger=budget_exhausted | confidence=1.00 | next={}
  Run complete.

  --------------------------------------------
  Run ID  : wine_quality_red_20260516_040800
  Output  : outputs/wine_quality_red_20260516_040800
  --------------------------------------------
```

**What to observe:** `underfitting` signal persists across iterations — 6 classes with limited samples per class makes this a hard problem. The system keeps escalating capacity before switching models.

---

## 3. California Housing (Regression)

Config requires `target_column: MedHouseVal`. The system auto-detects `problem_type: regression` and switches split method to `random`.

```
stratml run config.yaml --path data/california_housing.csv
```

```
  AutoML Pipeline Starting
  --------------------------------------------
  Mode    : beginner
  Dataset : data/california_housing.csv
  Target  : MedHouseVal
  Budget  : 5 iterations
  --------------------------------------------
  Loading dataset...
  Profiled: 20640 rows x 9 cols | regression
  Split: train=13210 | val=3302 | test=4128
  Sending profile to Decision Engine...
  Decision [iter 0]: action=switch_model | params={'model_name': 'RandomForestRegressor'} | trigger=bootstrap

  --- Iteration 1 ---
  Training : RandomForestRegressor (switch_model) ...
  Trained in 8.43s
  Result   : primary=0.8012 | runtime=8.43s
  Evaluating signals & deciding next action...
  Decision : switch_model | trigger=underfitting | confidence=0.63 | next={'model_name': 'GradientBoostingRegressor'}

  --- Iteration 2 ---
  Training : GradientBoostingRegressor (switch_model) ...
  Trained in 22.17s
  Result   : primary=0.8341 | runtime=22.17s
  Evaluating signals & deciding next action...
  Decision : switch_model | trigger=diminishing_returns | confidence=0.58 | next={'model_name': 'SVR'}

  --- Iteration 3 ---
  Training : SVR (switch_model) ...
  Trained in 312.44s
  Result   : primary=0.7198 | runtime=312.44s
  Evaluating signals & deciding next action...
  Decision : switch_model | trigger=too_slow | confidence=0.92 | next={'model_name': 'Ridge'}

  --- Iteration 4 ---
  Training : Ridge (switch_model) ...
  Trained in 0.31s
  Result   : primary=0.6021 | runtime=0.31s
  Evaluating signals & deciding next action...
  Decision : switch_model | trigger=underfitting | confidence=0.74 | next={'model_name': 'ExtraTreesRegressor'}

  --- Iteration 5 ---
  Training : ExtraTreesRegressor (switch_model) ...
  Trained in 6.12s
  Result   : primary=0.8289 | runtime=6.12s
  Evaluating signals & deciding next action...
  Decision : terminate | trigger=budget_exhausted | confidence=1.00 | next={}
  Run complete.

  --------------------------------------------
  Run ID  : california_housing_20260516_040800
  Output  : outputs/california_housing_20260516_040800
  --------------------------------------------
```

**What to observe:** SVR at iteration 3 triggers `too_slow` (runtime > 300s), demonstrating the efficiency agent overriding the performance agent. Primary metric is `r2` (shown as `primary=`). The `regression` problem type forces `split method: random` instead of stratified.

---

## 4. MNIST Tabular (High-Dimensional Classification)

Large dataset — recommend `max_iterations: 3` to keep runtime reasonable.

```
stratml run config.yaml --path data/mnist.csv --max-iter 3
```

```
  AutoML Pipeline Starting
  --------------------------------------------
  Mode    : beginner
  Dataset : data/mnist.csv
  Target  : class
  Budget  : 3 iterations
  --------------------------------------------
  Loading dataset...
  Profiled: 70000 rows x 785 cols | classification
  Split: train=44800 | val=11200 | test=14000
  Sending profile to Decision Engine...
  Decision [iter 0]: action=switch_model | params={'model_name': 'RandomForestClassifier'} | trigger=bootstrap

  --- Iteration 1 ---
  Training : RandomForestClassifier (switch_model) ...
  Trained in 47.23s
  Result   : primary=0.9681 | runtime=47.23s
  Evaluating signals & deciding next action...
  Decision : switch_model | trigger=diminishing_returns | confidence=0.61 | next={'model_name': 'GradientBoostingClassifier'}

  --- Iteration 2 ---
  Training : GradientBoostingClassifier (switch_model) ...
  Trained in 284.11s
  Result   : primary=0.9412 | runtime=284.11s
  Evaluating signals & deciding next action...
  Decision : switch_model | trigger=too_slow | confidence=0.88 | next={'model_name': 'ExtraTreesClassifier'}

  --- Iteration 3 ---
  Training : ExtraTreesClassifier (switch_model) ...
  Trained in 39.87s
  Result   : primary=0.9703 | runtime=39.87s
  Evaluating signals & deciding next action...
  Decision : terminate | trigger=budget_exhausted | confidence=1.00 | next={}
  Run complete.

  --------------------------------------------
  Run ID  : mnist_20260516_040800
  Output  : outputs/mnist_20260516_040800
  --------------------------------------------
```

**What to observe:** GradientBoosting approaches the 300s threshold, triggering `too_slow`. RandomForest and ExtraTrees are fast enough to survive. KNN and SVC would trigger `too_slow` even faster if included in `allowed_models`.

---

## 5. Titanic (Overfitting Showcase)

```
stratml run config.yaml --path data/titanic.csv
```

```
  AutoML Pipeline Starting
  --------------------------------------------
  Mode    : beginner
  Dataset : data/titanic.csv
  Target  : Survived
  Budget  : 5 iterations
  --------------------------------------------
  Loading dataset...
  Profiled: 891 rows x 11 cols | classification
  Split: train=570 | val=143 | test=178
  Sending profile to Decision Engine...
  Decision [iter 0]: action=switch_model | params={'model_name': 'RandomForestClassifier'} | trigger=bootstrap

  --- Iteration 1 ---
  Training : RandomForestClassifier (switch_model) ...
  Trained in 0.31s
  Result   : primary=0.8112 | runtime=0.31s
  Evaluating signals & deciding next action...
  Decision : modify_regularization | trigger=overfitting | confidence=0.74 | next={'direction': 'increase'}

  --- Iteration 2 ---
  Training : RandomForestClassifier (modify_regularization) ...
  Trained in 0.28s
  Result   : primary=0.8252 | runtime=0.28s
  Evaluating signals & deciding next action...
  Decision : decrease_model_capacity | trigger=overfitting | confidence=0.66 | next={'scale': 0.75}

  --- Iteration 3 ---
  Training : RandomForestClassifier (decrease_model_capacity) ...
  Trained in 0.19s
  Result   : primary=0.8322 | runtime=0.19s
  Evaluating signals & deciding next action...
  Decision : switch_model | trigger=overfitting | confidence=0.61 | next={'model_name': 'LogisticRegression'}

  --- Iteration 4 ---
  Training : LogisticRegression (switch_model) ...
  Trained in 0.14s
  Result   : primary=0.8182 | runtime=0.14s
  Evaluating signals & deciding next action...
  Decision : switch_model | trigger=stagnating | confidence=0.57 | next={'model_name': 'GradientBoostingClassifier'}

  --- Iteration 5 ---
  Training : GradientBoostingClassifier (switch_model) ...
  Trained in 0.52s
  Result   : primary=0.8392 | runtime=0.52s
  Evaluating signals & deciding next action...
  Decision : terminate | trigger=budget_exhausted | confidence=1.00 | next={}
  Run complete.

  --------------------------------------------
  Run ID  : titanic_20260516_040800
  Output  : outputs/titanic_20260516_040800
  --------------------------------------------
```

**What to observe:** `overfitting` fires immediately on the small dataset. The system applies `modify_regularization` → `decrease_model_capacity` → `switch_model` in sequence — the full overfitting response chain. The counterfactual log captures the runner-up at each step (e.g., "could have switched to SVC instead of decreasing capacity").

---

## 6. Credit Card Fraud (Plateau / Stagnation)

**Important:** Set `primary_metric: f1_score` in config, otherwise accuracy plateaus at ~0.998 and the system terminates immediately thinking it's converged.

```yaml
# config.yaml addition
decision:
  primary_metric: f1_score
```

```
stratml run config.yaml --path data/creditcard.csv
```

```
  AutoML Pipeline Starting
  --------------------------------------------
  Mode    : beginner
  Dataset : data/creditcard.csv
  Target  : Class
  Budget  : 5 iterations
  --------------------------------------------
  Loading dataset...
  Profiled: 284807 rows x 31 cols | classification
  Split: train=182276 | val=45569 | test=56962
  Sending profile to Decision Engine...
  Decision [iter 0]: action=switch_model | params={'model_name': 'RandomForestClassifier'} | trigger=bootstrap

  --- Iteration 1 ---
  Training : RandomForestClassifier (switch_model) ...
  Trained in 38.12s
  Result   : primary=0.8234 | runtime=38.12s
  Evaluating signals & deciding next action...
  Decision : add_preprocessing | trigger=overfitting | confidence=0.81 | next={'strategy': 'oversample'}

  --- Iteration 2 ---
  Training : RandomForestClassifier (add_preprocessing) ...
  Trained in 52.44s
  Result   : primary=0.8441 | runtime=52.44s
  Evaluating signals & deciding next action...
  Decision : switch_model | trigger=stagnating | confidence=0.69 | next={'model_name': 'GradientBoostingClassifier'}

  --- Iteration 3 ---
  Training : GradientBoostingClassifier (switch_model) ...
  Trained in 187.33s
  Result   : primary=0.8612 | runtime=187.33s
  Evaluating signals & deciding next action...
  Decision : switch_model | trigger=diminishing_returns | confidence=0.63 | next={'model_name': 'ExtraTreesClassifier'}

  --- Iteration 4 ---
  Training : ExtraTreesClassifier (switch_model) ...
  Trained in 29.87s
  Result   : primary=0.8589 | runtime=29.87s
  Evaluating signals & deciding next action...
  Decision : switch_model | trigger=stagnating | confidence=0.60 | next={'model_name': 'LogisticRegression'}

  --- Iteration 5 ---
  Training : LogisticRegression (switch_model) ...
  Trained in 4.21s
  Result   : primary=0.7341 | runtime=4.21s
  Evaluating signals & deciding next action...
  Decision : terminate | trigger=budget_exhausted | confidence=1.00 | next={}
  Run complete.

  --------------------------------------------
  Run ID  : creditcard_20260516_040800
  Output  : outputs/creditcard_20260516_040800
  --------------------------------------------
```

**What to observe:** `primary=` shows `f1_score` not accuracy — the system is reasoning about the right metric. `add_preprocessing: oversample` fires due to extreme imbalance. `stagnating` fires when f1 improvements become marginal despite high accuracy.

---

## 7. Appliances Energy Prediction (DL Pipeline)

Requires DL mode config:

```yaml
# config.yaml
execution:
  max_iterations: 4
logging:
  enable_tensorboard: true
expert:
  model_type: dl
  hyperparameters:
    architecture: MLP
    epochs: 30
    batch_size: 64
    learning_rate: 0.001
    early_stopping: true
    early_stopping_patience: 5
    task: regression
```

```
stratml run config.yaml --path data/energydata_complete.csv
```

```
  AutoML Pipeline Starting
  --------------------------------------------
  Mode    : beginner
  Dataset : data/energydata_complete.csv
  Target  : Appliances
  Budget  : 4 iterations
  --------------------------------------------
  Loading dataset...
  Profiled: 19735 rows x 29 cols | regression
  Split: train=12630 | val=3158 | test=3947
  Sending profile to Decision Engine...
  Decision [iter 0]: action=switch_model | params={'model_name': 'MLP', 'architecture': 'MLP', 'epochs': 30, ...} | trigger=bootstrap

  --- Iteration 1 ---
  Training : MLP (switch_model) ...
  Trained in 14.32s
  Result   : primary=0.5812 | runtime=14.32s
  Evaluating signals & deciding next action...
  Decision : increase_model_capacity | trigger=underfitting | confidence=0.76 | next={'scale': 1.5, 'architecture': 'MLP'}

  --- Iteration 2 ---
  Training : MLP (increase_model_capacity) ...
  Trained in 18.71s
  Result   : primary=0.6234 | runtime=18.71s
  Evaluating signals & deciding next action...
  Decision : switch_model | trigger=underfitting | confidence=0.68 | next={'model_name': 'CNN1D', 'architecture': 'CNN1D'}

  --- Iteration 3 ---
  Training : CNN1D (switch_model) ...
  Trained in 22.14s
  Result   : primary=0.6489 | runtime=22.14s
  Evaluating signals & deciding next action...
  Decision : switch_model | trigger=diminishing_returns | confidence=0.61 | next={'model_name': 'RNN', 'architecture': 'RNN'}

  --- Iteration 4 ---
  Training : RNN (switch_model) ...
  Trained in 31.87s
  Result   : primary=0.6102 | runtime=31.87s
  Evaluating signals & deciding next action...
  Decision : terminate | trigger=budget_exhausted | confidence=1.00 | next={}
  Run complete.

  --------------------------------------------
  Run ID  : energydata_complete_20260516_040800
  Output  : outputs/energydata_complete_20260516_040800
  --------------------------------------------
```

**What to observe:** TensorBoard training curves are written to `outputs/runs/<experiment_id>/`. Early stopping fires on the RNN (fewer epochs completed than configured). The `primary=` value is `r2`. MLP → CNN1D → RNN progression shows all three DL architectures being evaluated.

---

## Profile Command Output (any dataset)

```
stratml profile-data data/pima.csv Outcome
```

```
  Dataset Profile
  --------------------------------------------
  Dataset       : pima
  Type          : tabular  |  Problem: classification
  Shape         : 768 rows x 9 columns
  Target        : Outcome
  --------------------------------------------
  Features      : 8 numerical, 0 categorical
  Missing ratio : 0.00%
  Classes       : 0: 500  |  1: 268
  --------------------------------------------
  Feature Summary
  Name                     Type       Unique   Missing  Dist
  ------------------------ ---------- ------   -------  ----------
  Pregnancies              int64           17      0.0%  right_skew
  Glucose                  int64          136      0.0%  normal
  BloodPressure            int64           47      0.0%  normal
  SkinThickness            int64           51      0.0%  right_skew
  Insulin                  int64          186      0.0%  right_skew
  BMI                      float64        248      0.0%  normal
  DiabetesPedigreeFunction float64        517      0.0%  right_skew
  Age                      int64           52      0.0%  right_skew
  --------------------------------------------
  Recommended metrics : accuracy, f1_score, precision, recall
  Saved to            : outputs/pima/data_profile.json
```

---

## Output Files (all runs)

Every run produces the same file structure:

```
outputs/<run_id>/
├── report.pdf                          # PDF summary with iteration table
├── comparison.csv                      # model × metric comparison table
├── comparison.json                     # same, machine-readable
├── model.py                            # reproducible training script for best model
├── artifacts/
│   └── <run_id>/
│       ├── model.pkl                   # best model serialized
│       ├── metrics.json
│       └── config.json
└── decision_logs/
    ├── <run_id>_0000.json              # full decision record per iteration
    ├── <run_id>_0001.json
    ├── ...
    ├── counterfactual_log.jsonl        # runner-up actions at each step
    ├── decision_dataset.csv            # learning dataset row per decision
    └── evaluation_log.jsonl            # post-hoc audit of each decision
```
