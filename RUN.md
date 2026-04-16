# RUN.md — Running StratML (What's Implemented)

Install the CLI first (one-time, run from PowerShell):

```powershell
.\stratml\cli\install.ps1
```

Then restart your terminal. All commands below use `stratml` directly.

---

## Step 1 — Check environment

```bash
stratml doctor
```

---

## Step 2 — Create a config

```bash
stratml init
```

Edit the generated `config.yaml` — set `dataset.path` and `dataset.target_column`.

---

## Step 3 — Validate config

```bash
stratml validate-config config.yaml
```

---

## Step 4 — Profile a dataset

```bash
stratml profile-data data/iris.csv species
```

Prints a full dataset profile to the terminal.  
Saves JSON to `outputs/iris/data_profile.json`.

---

## Step 5 — Dry-run the pipeline

```bash
stratml run config.yaml --path data/iris.csv --dry-run
```

Resolves and prints the full config without executing.

---

## Step 6 — Run the full pipeline (ML)

```bash
stratml run config.yaml
```

Runs the full AutoML loop: profile → decision engine → ML pipeline → report.

---

## Step 7 — Run with deep learning

Pass `--dl` to switch the execution engine to PyTorch:

```bash
# Default MLP
stratml run config.yaml --dl

# 1-D CNN
stratml run config.yaml --dl --architecture CNN1D --epochs 50 --lr 0.0005

# LSTM/RNN
stratml run config.yaml --dl --architecture RNN --epochs 30 --batch-size 64
```

Or set `deep_learning.enabled: true` in `config.yaml` to make DL the default for a project.

---

## What's Implemented

| Component | Status |
|---|---|
| `stratml doctor` | ✅ |
| `stratml init` | ✅ |
| `stratml validate-config` | ✅ |
| `stratml profile-data` | ✅ |
| `stratml run --dry-run` | ✅ |
| `stratml run` (full ML pipeline) | ✅ |
| `stratml run --dl` (DL pipeline) | ✅ |
| ML pipelines (scikit-learn) | ✅ |
| DL pipelines (PyTorch MLP/CNN1D/RNN) | ✅ |
| Decision agent / rule engine | ✅ |
| MLflow / TensorBoard / LangSmith | 🔲 |
