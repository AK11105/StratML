# ⚙️ CLI Configuration System

This document defines the configuration structure for running the AutoML system via CLI.

---

# 🧩 Config Philosophy

The system supports **multiple user control modes**:

- **Beginner** → Fully automatic
- **Intermediate** → Partial control
- **Expert** → Full manual control

---

# 📄 Recommended Format: YAML

---

# 🧾 Full Config Schema

```yaml
mode: beginner  # beginner | intermediate | expert

dataset:
  path: "data/train.csv"
  target_column: "label"

execution:
  max_iterations: 20
  timeout_per_run: 300
  random_seed: 42

split:
  method: stratified  # stratified | random
  test_size: 0.2

logging:
  enable_mlflow: true
  enable_tensorboard: true
  log_level: info

constraints:
  max_memory: null
  max_cpu: null

# -----------------------
# MODE-SPECIFIC SETTINGS
# -----------------------

beginner:
  fully_automatic: true

intermediate:
  allowed_models:
    - random_forest
    - xgboost

  allow_preprocessing_control: true

expert:
  model:
    type: xgboost
    hyperparameters:
      n_estimators: 200
      max_depth: 6

  preprocessing:
    missing_value_strategy: mean
    encoding: onehot
    scaling: standard

  training:
    epochs: 50
    batch_size: 32
```

---

# 🧠 Mode Behavior Rules

## Beginner Mode
- Ignores all manual overrides
- Team B has full decision control

## Intermediate Mode
- Restricts model/search space
- Allows partial overrides

## Expert Mode
- User-defined pipeline
- Team B decisions are overridden or constrained

---

# 💻 CLI Commands

## Core Commands

```bash
automl run --config config.yaml
automl validate-config config.yaml
automl profile-data --input data.csv
```

---

## Advanced Usage

```bash
automl run \
  --config config.yaml \
  --mode expert \
  --max-iter 10 \
  --dry-run
```

---

## Utility Commands

```bash
automl init          # create default config
automl doctor        # check environment
automl explain-run   # inspect past run
```

---

# ⚠️ Override Priority

1. CLI flags
2. Config file
3. Mode defaults

---

# 🧱 Design Notes

- YAML preferred for readability
- Must support future extensions (new models, pipelines)
- Mode abstraction must be enforced in orchestrator
