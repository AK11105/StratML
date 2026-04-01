# Dataset Loader — Supported Formats

## Overview

`stratml/execution/data/loader.py` loads any supported file into a `(DataFrame, dataset_name)` tuple. The format is detected from the file extension. `dataset_name` is always the file stem (filename without extension).

---

## Supported Formats

| Extension | Reader | Notes |
|---|---|---|
| `.csv` | `pd.read_csv` | Standard comma-separated |
| `.tsv` | `pd.read_csv(sep="\t")` | Tab-separated |
| `.json` | `pd.read_json` | Records or columns orientation |
| `.parquet` | `pd.read_parquet` | Requires `pyarrow` or `fastparquet` |
| `.xlsx` | `pd.read_excel` | Requires `openpyxl` |
| `.xls` | `pd.read_excel` | Legacy Excel format |

---

## Usage

```python
from stratml.execution.data.loader import load_dataframe

df, name = load_dataframe("data/housing.csv")
# name → "housing"

df, name = load_dataframe("data/sensors.parquet")
# name → "sensors"
```

The returned `df` is a plain `pd.DataFrame`. The `name` is passed directly into `build_dataset` and flows through to `DataProfile.dataset_name` and all artifact paths.

---

## Error Handling

```python
# File not found
load_dataframe("data/missing.csv")
# → FileNotFoundError: Dataset not found: data/missing.csv

# Unsupported format
load_dataframe("data/model.pkl")
# → ValueError: Unsupported format '.pkl'. Supported: ['.csv', '.tsv', '.json', '.parquet', '.xlsx', '.xls']
```

---

## Adding a New Format

The loader uses a dispatch dict — adding a format is one line:

```python
_LOADERS = {
    ".csv":     lambda p: pd.read_csv(p),
    ".tsv":     lambda p: pd.read_csv(p, sep="\t"),
    ".json":    lambda p: pd.read_json(p),
    ".parquet": lambda p: pd.read_parquet(p),
    ".xlsx":    lambda p: pd.read_excel(p),
    ".xls":     lambda p: pd.read_excel(p),
    # Add here:
    ".feather": lambda p: pd.read_feather(p),
}
```

---

## Optional Dependencies

Some formats require extra packages not in the base install:

| Format | Package | Install |
|---|---|---|
| `.parquet` | `pyarrow` or `fastparquet` | `pip install pyarrow` |
| `.xlsx` | `openpyxl` | `pip install openpyxl` |
| `.xls` | `xlrd` | `pip install xlrd` |

If the dependency is missing, pandas will raise an `ImportError` with a clear message when that format is used. Other formats are unaffected.

---

## CLI Usage

The `profile-data` command accepts any supported format:

```bash
stratml profile-data data/housing.csv MedHouseVal
stratml profile-data data/sensors.parquet target
stratml profile-data data/records.json label
```
