# RUN.md — Running StratML

Install the CLI first (one-time):

```bash
# Linux / macOS
bash stratml/cli/install.sh
source ~/.bashrc

# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File stratml\cli\install.ps1
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

Prints a full dataset profile. Saves JSON to `outputs/iris/data_profile.json`.

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

Pass `--dl` to switch to PyTorch:

```bash
# Default MLP
stratml run config.yaml --dl

# CNN1D
stratml run config.yaml --dl --architecture CNN1D --epochs 50 --lr 0.0005

# RNN
stratml run config.yaml --dl --architecture RNN --epochs 30 --batch-size 64
```

Or set `deep_learning.enabled: true` in `config.yaml` to make DL the default.

### DL features active by default
- GPU auto-selected (cuda → mps → cpu)
- Early stopping with best-weight restore
- Gradient clipping (`grad_clip: 1.0`)
- ReduceLROnPlateau scheduler

### Advanced DL hyperparameters (via config.yaml)

```yaml
deep_learning:
  enabled: true
  architecture: MLP
  epochs: 50
  learning_rate: 0.001
  batch_size: 32
  weight_decay: 0.0001     # L2 regularization in Adam
  dropout: 0.2             # dropout probability
  batch_norm: true         # BatchNorm per hidden layer
  scheduler: plateau       # plateau | cosine | none
  grad_clip: 1.0           # max gradient norm (0.0 = disabled)
  mixed_precision: false   # FP16 AMP — CUDA only
```

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
| ML pipelines (scikit-learn, 24 models) | ✅ |
| DL pipelines (MLP / CNN1D / RNN) | ✅ |
| GPU support (CUDA / MPS) | ✅ |
| Mixed precision (FP16 AMP) | ✅ |
| Gradient clipping | ✅ |
| TensorBoard training curves | ✅ |
| Decision agent / rule engine | ✅ |
| PDF report + comparison.csv | ✅ |
| model.pkl + model.pth artifacts | ✅ |
| MLflow logging | 🔲 |
| LangSmith tracing | 🔲 |
