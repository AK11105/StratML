"""
validator.py
------------
Phase 1 — Dataset Ingestion: validate the loaded DataFrame and build a Dataset object.
"""

import warnings

import pandas as pd
from stratml.execution.schemas import Dataset


def build_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    target_column: str,
) -> Dataset:
    """
    Validate the DataFrame and construct a Dataset object.

    Raises:
        ValueError: on structural problems that would break downstream steps.
    """
    # 1. Zero rows
    if len(df) == 0:
        raise ValueError("Dataset has 0 rows.")

    # 2. Duplicate column names
    seen, dupes = set(), set()
    for c in df.columns:
        (dupes if c in seen else seen).add(c)
    if dupes:
        raise ValueError(f"Duplicate column names detected: {sorted(dupes)}")

    # 3. Target column exists
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # 4. All-null columns — warn and drop
    all_null = [c for c in df.columns if df[c].isnull().all()]
    if all_null:
        warnings.warn(f"Dropping all-null columns: {all_null}", UserWarning, stacklevel=2)
        df = df.drop(columns=all_null)

    # 5. Target validity
    target = df[target_column]
    if target.isnull().all():
        raise ValueError(f"Target column '{target_column}' is entirely null.")
    if target.dropna().nunique() < 2:
        raise ValueError(
            f"Target column '{target_column}' has fewer than 2 unique non-null values — "
            "not a valid ML problem."
        )

    return Dataset(
        dataset_name=dataset_name,
        rows=len(df),
        columns=len(df.columns),
        target_column=target_column,
        dataset_type="tabular",
        raw_dataframe=df,
    )
