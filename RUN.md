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

Resolves and prints the full config. Orchestrator not yet connected — `run` without `--dry-run` prints a placeholder.

---

## What's Implemented

| Component | Status |
|---|---|
| `stratml doctor` | ✅ |
| `stratml init` | ✅ |
| `stratml validate-config` | ✅ |
| `stratml profile-data` | ✅ |
| `stratml run --dry-run` | ✅ |
| `stratml run` (full pipeline) | 🔲 orchestrator not connected |
| ML/DL pipelines | 🔲 |
| Decision agent / rule engine | 🔲 |
| MLflow / TensorBoard / LangSmith | 🔲 |
