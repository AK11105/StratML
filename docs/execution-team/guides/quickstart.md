# Quickstart — Profile a Dataset

## Setup

```bash
# from project root
source .venv/bin/activate   # or activate your environment
```

---

## Profile a dataset (CLI)

```bash
stratml profile-data data/iris.csv species
stratml profile-data data/housing.csv MedHouseVal
```

Prints a full dataset profile to the terminal and saves JSON to `outputs/<dataset_name>/data_profile.json`.

---

## Profile programmatically

```python
from stratml.execution.data.loader import load_dataframe
from stratml.execution.data.validator import build_dataset
from stratml.execution.data.profiler import build_profile

df, name = load_dataframe("data/iris.csv")
dataset  = build_dataset(df, name, "species")
profile  = build_profile(dataset)

print(profile.problem_type)       # "classification"
print(profile.imbalance_ratio)    # 1.0 (balanced)
print(profile.class_entropy)      # 1.585
print(profile.feature_variance_mean)
```

---

## Output location

```
outputs/
  <dataset_name>/
      data_profile.json    ← DataProfile (also sent to Team B at run time)
```

---

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError` | Dataset path wrong | Check path relative to project root |
| `ValueError: Target column '...' not found` | Wrong target column name | Check column names in the file |
| `ValueError: Unsupported format` | Unsupported extension | Supported: `.csv`, `.tsv`, `.json`, `.parquet`, `.xlsx`, `.xls` |
| `ValueError: Dataset has 0 rows` | Empty file | Check the file |
| `ValueError: Duplicate column names` | Repeated column headers | Fix the source data |
| `ValueError: fewer than 2 unique non-null values` | Target is constant | Not a valid ML problem |
