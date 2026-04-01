"""
splitter.py
-----------
Phase 3 — Split Dataset into train / val / test sets.
Must run before any preprocessing to prevent data leakage.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from stratml.execution.schemas import Dataset, SplitConfig, DataSplit


def split_dataset(dataset: Dataset, config: SplitConfig, problem_type: str) -> DataSplit:
    """
    Split dataset.raw_dataframe into train / val / test.

    - classification → stratified split
    - regression     → random split
    """
    df: pd.DataFrame = dataset.raw_dataframe
    X = df.drop(columns=[dataset.target_column])
    y = df[dataset.target_column]

    stratify = y if problem_type == "classification" else None

    # First cut: split off test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=stratify,
    )

    # Second cut: split val from train
    # val_size is relative to the full dataset, so adjust for the remaining fraction
    relative_val = config.val_size / (1.0 - config.test_size)
    stratify_val = y_trainval if problem_type == "classification" else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=relative_val,
        random_state=config.random_seed,
        stratify=stratify_val,
    )

    return DataSplit(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
