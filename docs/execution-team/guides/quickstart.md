# Running Phase 1 & Phase 2

## Create python virtual environment (optional but recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Prerequisites

From the project root, install dependencies if not already done:

```bash
pip3 install scikit-learn pandas pydantic scipy
```

---

## Run

```bash
# from project root
python3 execution/run_ingestion_profiling.py <dataset_path> <target_column>
```

**Examples:**

```bash
python3 execution/run_ingestion_profiling.py data/iris.csv species
python3 execution/run_ingestion_profiling.py data/housing.csv MedHouseVal
```

---

## What happens

1. Dataset is loaded and validated
2. `DataProfile` is computed and printed to stdout
3. Profile is saved to `outputs/<dataset_name>/data_profile.json`

---

## Output location

```
outputs/
  <dataset_name>/
      data_profile.json    ← DataProfile sent to Team B
```

---

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError` | Dataset path wrong | Check path relative to project root |
| `ValueError: Target column '...' not found` | Wrong target column name | Check column names with `head data/file.csv` |
| `ValueError: Unsupported format` | Non-CSV file | Only `.csv` supported currently |
