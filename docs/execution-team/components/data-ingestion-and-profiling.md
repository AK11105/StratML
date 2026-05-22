# Phase 1 & Phase 2 — Ingestion and Profiling

## Overview

These two phases are the entry point of the entire pipeline. They run once per dataset, before any training begins. The output (`DataProfile`) is the only thing Team A sends to Team B unprompted — everything after this is response-driven.

---

## Phase 1 — Dataset Ingestion

**Goal:** Load the dataset from disk and produce an internal `Dataset` object.

**Files:**
- `execution/data/loader.py` — reads the file, returns `(DataFrame, dataset_name)`
- `execution/data/validator.py` — validates the DataFrame, builds `Dataset`

**What it does:**
1. Reads the file into a pandas DataFrame. Supported formats: `.csv`, `.tsv`, `.json`, `.parquet`, `.xlsx`, `.xls`
2. Validates structural integrity before any processing:
   - Raises `ValueError` if the DataFrame has 0 rows
   - Raises `ValueError` if duplicate column names are detected
   - Raises `ValueError` if `target_column` is not found
   - Warns and drops any all-null columns
   - Raises `ValueError` if the target is entirely null or has fewer than 2 unique non-null values
3. Returns a `Dataset` object holding the DataFrame in memory (`dataset_type` is always `"tabular"`)

**Output — internal `Dataset`:**

```
dataset_name     str       stem of the filename (e.g. "iris")
rows             int       number of rows (after null-column drops)
columns          int       total columns including target
target_column    str       prediction target
dataset_type     str       "tabular"
raw_dataframe    object    in-memory only, never serialized
```

---

## Phase 2 — Data Profiling

**Goal:** Compute a `DataProfile` from the `Dataset` and send it to Team B.

**File:** `execution/data/profiler.py`

**What it does:**
1. Splits features into `numerical_columns` and `categorical_columns` (excludes target)
2. Infers `problem_type` with improved logic to avoid misclassifying rounded-float regression targets:
   - `"classification"` if target dtype is `object`
   - `"regression"` if target dtype is float AND `nunique > 10` AND `nunique / n > 0.05`
   - `"classification"` if `nunique ≤ 20`
   - `"regression"` otherwise
3. Computes `class_distribution` (classification only) — label → count
4. Computes `missing_value_ratio` — global fraction of missing cells across all columns
5. For each feature, computes a `FeatureInfo`:
   - `dtype`, `unique_values`, `missing_percentage`
   - `distribution` — heuristic via Shapiro-Wilk + skewness: `normal | skewed | uniform | unknown`
6. Sets `recommended_metrics` based on problem type:
   - classification → `["accuracy", "f1_score"]`
   - regression → `["mse", "rmse", "r2"]`
7. Computes three additional fields used by the Decision Engine at iteration 0:
   - `imbalance_ratio` — `max_class_count / min_class_count` (classification only)
   - `feature_variance_mean` — mean variance across all numerical features
   - `class_entropy` — Shannon entropy of the class distribution (classification only)

**Output — `DataProfile` (sent to Team B):**

```
dataset_name            str
dataset_type            str          "tabular" | "text" | "vision"
rows                    int
columns                 int
target_column           str
problem_type            str          "classification" | "regression"
numerical_columns       list[str]
categorical_columns     list[str]
missing_value_ratio     float        global ratio across all features
class_distribution      dict         label → count (empty for regression)
feature_summary         list[FeatureInfo]
recommended_metrics     list[str]
imbalance_ratio         float|None   max_class / min_class (classification only)
feature_variance_mean   float|None   mean variance across numerical features
class_entropy           float|None   Shannon entropy of class distribution
```

---

## Data flow

```
file (CSV/TSV/JSON/Parquet/Excel)
   ↓  loader.py
DataFrame
   ↓  validator.py  (zero-row, duplicate-col, all-null, target validity checks)
Dataset  (internal, in-memory)
   ↓  profiler.py
DataProfile  →  outputs/<dataset_name>/data_profile.json  →  Team B
```

---

## Schema enforcement

All objects are Pydantic models defined in `execution/schemas.py`.
Invalid field values raise a `ValidationError` at construction time — nothing invalid can be passed to Team B.

---

## What Team B does with DataProfile

Team B uses it to:
- Populate `StateObject.dataset` (samples, features, imbalance, missing ratio)
- Use `imbalance_ratio`, `feature_variance_mean`, and `class_entropy` to inform the first `ActionDecision` at iteration 0
- Decide what preprocessing to apply (missing values, encoding, scaling, imbalance)
