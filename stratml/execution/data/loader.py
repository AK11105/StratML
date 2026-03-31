"""
loader.py
---------
Phase 1 — Dataset Ingestion: load a CSV file into a pandas DataFrame.
"""

from pathlib import Path
import pandas as pd

_SUPPORTED_FORMATS = {".csv"}


def load_dataframe(path: str | Path) -> tuple[pd.DataFrame, str]:
    """
    Load a dataset file into a DataFrame.

    Returns:
        (dataframe, dataset_name) tuple where dataset_name is the file stem.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    ext = p.suffix.lower()
    if ext not in _SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {_SUPPORTED_FORMATS}")

    df = pd.read_csv(p)
    return df, p.stem
