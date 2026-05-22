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

### ML demos

```bash
# Classification — overfitting chain (Titanic)
stratml run config.yaml --path data/external/titanic.csv

# Classification — imbalance + regularization (Pima)
stratml run config.yaml --path data/raw/pima.csv

# Classification — multiclass, underfitting chain (Wine Quality)
stratml run config.yaml --path data/raw/wine_quality_red.csv

# Regression — smooth convergence (California Housing)
stratml run config.yaml --path data/raw/california_housing.csv

# Classification — extreme imbalance / fraud (Credit Card)
stratml run config.yaml --path data/external/creditcard.csv

# Multiclass — high-dimensional tabular, too_slow fires (MNIST ML)
stratml run config.yaml --path data/raw/mnist.csv

# Regression — multi-output energy forecasting (Energy)
stratml run config.yaml --path data/raw/energydata_complete.csv
```

### DL demos

```bash
# Vision — CNN2D → ResNet18 → EfficientNetB0 (CIFAR-10)
stratml run config.yaml --path data/raw/cifar10.csv

# Text — TextCNN → BiLSTM → DistilBERT → TinyBERT (IMDb)
stratml run config.yaml --path data/raw/imdb.csv

# Vision — MLP → CNN2D → ResNet18 → MobileNetV3 (MNIST DL)
stratml run config.yaml --path data/raw/mnist_dl.csv
```

> **Note:** CIFAR-10 and IMDb CSVs must be downloaded first:
> ```bash
> python data/scripts/download_cifar10.py
> python data/scripts/download_imdb.py
> ```

---

## Download all datasets

```bash
python data/scripts/download_all.py

# Skip large datasets
python data/scripts/download_all.py --skip mnist cifar10 imdb
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
| `stratml run` (ML demos) | ✅ hardcoded |
| `stratml run` (DL demos — cifar10, imdb, mnist_dl) | ✅ hardcoded |
| `stratml run` (real pipeline) | 🔲 |
