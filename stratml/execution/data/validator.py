"""
validator.py
------------
Phase 1 — Dataset Ingestion: validate the loaded DataFrame and build a Dataset object.
"""

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
        ValueError: target_column not found in df.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    return Dataset(
        dataset_name=dataset_name,
        rows=len(df),
        columns=len(df.columns),
        target_column=target_column,
        dataset_type=_infer_dataset_type(df),
        raw_dataframe=df,
    )


def _infer_dataset_type(df: pd.DataFrame) -> str:
    return "tabular"
