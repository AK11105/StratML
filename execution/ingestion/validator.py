"""
validator.py
------------
Phase 1 — Dataset Ingestion: validate the loaded DataFrame and build a Dataset object.

Responsibilities:
- Confirm target column exists in the DataFrame
- Infer dataset_type (currently always "tabular" for CSV; extensible)
- Construct and return the internal Dataset object

Raises:
    ValueError: if target_column is missing from the DataFrame
"""

import pandas as pd
from execution.schemas import Dataset


def build_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    target_column: str,
) -> Dataset:
    """
    Validate the DataFrame and construct a Dataset object.

    Args:
        df: Raw DataFrame from loader.
        dataset_name: Stem of the source file.
        target_column: Column to use as the prediction target.

    Returns:
        Dataset instance.

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
    """
    Infer dataset type from column dtypes.

    Currently returns "tabular" for all structured CSV data.
    Extend here to detect text/vision datasets when needed.
    """
    return "tabular"
