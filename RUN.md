# RUN.md — Running StratML

Install the CLI first (one-time):

```bash
# Linux / macOS
bash stratml/cli/install.sh
source ~/.bashrc

# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File stratml\cli\install.ps1
```

Then restart your terminal.

---

## Environment check

```bash
stratml doctor
```

---

## Profile a dataset

```bash
stratml profile-data data/raw/titanic.csv Survived
stratml profile-data data/raw/pima.csv Outcome
stratml profile-data data/raw/wine_quality_red.csv quality
stratml profile-data data/raw/california_housing.csv MedHouseVal
stratml profile-data data/external/creditcard.csv Class
stratml profile-data data/raw/mnist.csv label
stratml profile-data data/raw/energydata_complete.csv Appliances
```

---

## Run a demo

Each command below runs the full hardcoded demo for that dataset.  
Edit `config.yaml` to set `dataset.path` before running, or pass `--path` inline.

```bash
# Classification — overfitting chain
stratml run config.yaml --path data/external/titanic.csv 

# Classification — imbalance + regularization
stratml run config.yaml --path data/raw/pima.csv

# Classification — multiclass, underfitting chain
stratml run config.yaml --path data/raw/wine_quality_red.csv

# Regression — smooth convergence
stratml run config.yaml --path data/raw/california_housing.csv

# Classification — extreme imbalance (fraud)
stratml run config.yaml --path data/external/creditcard.csv

# Multiclass image classification (DL)
stratml run config.yaml --path data/raw/mnist.csv

# Regression — multi-output energy forecasting
stratml run config.yaml --path data/raw/energydata_complete.csv
```

---

## Dry-run (resolve config without executing)

```bash
stratml run config.yaml --path data/external/titanic.csv --dry-run
```

---

## Validate config

```bash
stratml validate-config config.yaml
```

---

## Generate a fresh config

```bash
stratml init
```

---

## Status

| Component | Status |
|---|---|
| `stratml doctor` | ✅ |
| `stratml init` | ✅ |
| `stratml validate-config` | ✅ |
| `stratml profile-data` | ✅ |
| `stratml run --dry-run` | ✅ |
| `stratml run` (demo datasets) | ✅ hardcoded |
| `stratml run` (real pipeline) | 🔲 |
