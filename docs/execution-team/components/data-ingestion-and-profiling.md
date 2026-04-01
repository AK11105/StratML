# Phase 1 & Phase 2 — Ingestion and Profiling

## Overview

These two phases are the entry point of the entire pipeline. They run once per dataset, before any training begins. The output (`DataProfile`) is the only thing Team A sends to Team B unprompted — everything after this is response-driven.

---

## Phase 1 — Dataset Ingestion

**Goal:** Load the dataset from disk and produce an internal `Dataset` object.

**Files:**
- `execution/ingestion/loader.py` — reads the file, returns `(DataFrame, dataset_name)`
- `execution/ingestion/validator.py` — validates target column exists, builds `Dataset`

**What it does:**
1. Reads the CSV file into a pandas DataFrame
2. Confirms the `target_column` exists in the data
3. Infers `dataset_type` — currently always `"tabular"` for CSV (extensible to text/vision)
4. Returns a `Dataset` object holding the DataFrame in memory

**Output — internal `Dataset`:**

```
dataset_name     str       stem of the filename (e.g. "iris")
rows             int       number of rows
columns          int       total columns including target
target_column    str       prediction target
dataset_type     str       "tabular" | "text" | "vision"
raw_dataframe    object    in-memory only, never serialized
```

---

## Phase 2 — Data Profiling

**Goal:** Compute a `DataProfile` from the `Dataset` and send it to Team B.

**File:** `execution/profiling/profiler.py`

**What it does:**
1. Splits features into `numerical_columns` and `categorical_columns` (excludes target)
2. Infers `problem_type`:
   - `"classification"` if target is non-numeric, or numeric with ≤ 20 unique values
   - `"regression"` otherwise
3. Computes `class_distribution` (classification only) — label → count
4. Computes `missing_value_ratio` — global fraction of missing cells across all columns
5. For each feature, computes a `FeatureInfo`:
   - `dtype`, `unique_values`, `missing_percentage`
   - `distribution` — heuristic via Shapiro-Wilk + skewness: `normal | skewed | uniform | unknown`
6. Sets `recommended_metrics` based on problem type:
   - classification → `["accuracy", "f1_score"]`
   - regression → `["mse", "rmse", "r2"]`

**Output — `DataProfile` (sent to Team B):**

```
dataset_name          str
dataset_type          str          "tabular" | "text" | "vision"
rows                  int
columns               int
target_column         str
problem_type          str          "classification" | "regression"
numerical_columns     list[str]
categorical_columns   list[str]
missing_value_ratio   float        global ratio across all features
class_distribution    dict         label → count (empty for regression)
feature_summary       list[FeatureInfo]
recommended_metrics   list[str]
```

---

## Data flow

```
CSV file
   ↓  loader.py
DataFrame
   ↓  validator.py
Dataset  (internal, in-memory)
   ↓  profiler.py
DataProfile  →  outputs/<dataset_name>/data_profile.json  →  Team B
```

---

## Schema enforcement

All objects are Pydantic models defined in `execution/schemas.py`.  
Invalid field values (e.g. unknown `problem_type`, out-of-range `confidence`) raise a `ValidationError` at construction time — nothing invalid can be passed to Team B.

---

## What Team B does with DataProfile

Team B uses it to:
- Populate `StateObject.dataset_meta_features`
- Decide what preprocessing to apply (missing values, encoding, scaling, imbalance)
- Plan the first experiment and return the first `ActionDecision`
