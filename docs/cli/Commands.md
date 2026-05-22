# StratML CLI Commands

---

## init

Create a default `config.yaml` in the current directory.

```bash
stratml init
```

Edit `dataset.path` and `dataset.target_column` before running anything else.

---

## validate-config

Check that a config file is valid before running.

```bash
stratml validate-config <config>
```

Example:

```bash
stratml validate-config config.yaml
```

Checks:
- `dataset.path` is set
- `dataset.target_column` is set
- `mode` is one of `beginner`, `intermediate`, `expert`

---

## profile-data

Load a dataset, compute a DataProfile, and print a summary to the terminal.
Saves full JSON output to `outputs/<dataset_name>/data_profile.json`.

```bash
stratml profile-data <dataset> <target>
```

Example:

```bash
stratml profile-data data/iris.csv species
```

Output includes: shape, problem type, feature types, missing ratio, class distribution, per-feature distribution.

---

## run

Run the full AutoML pipeline.

```bash
stratml run <config> [options]
```

Example:

```bash
stratml run config.yaml
```

### Options

```
--path <dataset>                 Override dataset.path from config
--mode <beginner|intermediate|expert>
--max-iter <n>                   Override execution.max_iterations
--dry-run                        Print resolved config without executing
--dl                             Enable deep learning mode (PyTorch)
--architecture <MLP|CNN1D|RNN>   DL architecture (default: MLP)
--epochs <n>                     DL training epochs (default: 20)
--lr <float>                     DL learning rate (default: 0.001)
--batch-size <n>                 DL batch size (default: 32)
```

### Dry run

```bash
stratml run config.yaml --path data/iris.csv --dry-run
```

Prints the fully resolved config (after YAML + CLI overrides) without running anything.

### ML pipeline (default)

```bash
stratml run config.yaml
```

### DL pipeline

```bash
# Default MLP
stratml run config.yaml --dl

# CNN1D with custom epochs and LR
stratml run config.yaml --dl --architecture CNN1D --epochs 50 --lr 0.0005

# RNN
stratml run config.yaml --dl --architecture RNN --epochs 30 --batch-size 64
```

DL mode can also be set permanently in `config.yaml`:

```yaml
deep_learning:
  enabled: true
  architecture: MLP   # MLP | CNN1D | RNN
  epochs: 20
  learning_rate: 0.001
  batch_size: 32
```

### Outputs

After a run, outputs are written to `outputs/<run_id>/`:

```
outputs/<run_id>/
  report.pdf              ← PDF execution report
  comparison.csv          ← per-iteration metrics table
  comparison.json         ← same, as JSON
  artifacts/
    model.pkl             ← best model (joblib)
    model.pth             ← best DL model state_dict (PyTorch only)
    model.py              ← auto-generated inference script
    metrics.json          ← final metrics
    config.json           ← experiment config
  decision_logs/          ← per-iteration decision records (JSON)
  tensorboard/            ← TensorBoard event files (DL only)
```

At the end of a run you are prompted to download `model.pkl` + `model.py` to the current directory.

---

## doctor

Check that all required packages are installed and show their versions.

```bash
stratml doctor
```

Checks: `pandas`, `numpy`, `sklearn`, `torch`, `pydantic`, `mlflow`, `yaml`

---

## Argument Style

All commands use positional arguments — no `--input` or `--config` flags needed.

| Command | Positional args |
|---|---|
| `run` | `config` |
| `validate-config` | `config` |
| `profile-data` | `dataset`, `target` |
| `init` | none |
| `doctor` | none |
