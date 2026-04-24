# Config Schema Reference

Full YAML configuration schema for `stratml run`.

---

## Minimal Config

```yaml
dataset:
  path: "data/train.csv"
  target_column: "label"
```

Everything else has defaults.

---

## Full Schema

```yaml
# Control mode — determines how much the user overrides the decision engine
mode: beginner  # beginner | intermediate | expert

dataset:
  path: "data/train.csv"
  target_column: "label"

execution:
  max_iterations: 20
  timeout_per_run: 300   # seconds per experiment
  random_seed: 42

split:
  method: stratified     # stratified | random
  test_size: 0.2

logging:
  enable_mlflow: true
  enable_tensorboard: true
  log_level: info        # debug | info | warning

# intermediate mode — restrict the model search space
intermediate:
  allowed_models:
    - random_forest
    - gradient_boosting
  allow_preprocessing_control: true

# expert mode — full manual control, decision engine is constrained
expert:
  model:
    type: gradient_boosting
    hyperparameters:
      n_estimators: 200
      max_depth: 6

  preprocessing:
    missing_value_strategy: mean   # mean | median | drop
    encoding: onehot               # onehot | label | none
    scaling: standard              # standard | minmax | none

  training:
    epochs: 50
    batch_size: 32
```

---

## Mode Behaviour

| Mode | Decision engine | Model choice | Preprocessing |
|---|---|---|---|
| `beginner` | Full control | Automatic | Automatic |
| `intermediate` | Constrained to `allowed_models` | From allowed list | Automatic |
| `expert` | Overridden | User-defined | User-defined |

---

## Override Priority

CLI flags > config file > mode defaults.

Example — override max iterations at runtime:

```bash
stratml run config.yaml --max-iter 10
```
