"""
loader.py
---------
Phase 1 — Dataset Ingestion: load a dataset file into a pandas DataFrame.

Supported formats:
    .csv   — comma-separated
    .tsv   — tab-separated
    .json  — records or columns orientation
    .parquet
    .xlsx / .xls — Excel (requires openpyxl)
"""

from pathlib import Path
import pandas as pd

_LOADERS = {
    ".csv":     lambda p: pd.read_csv(p),
    ".tsv":     lambda p: pd.read_csv(p, sep="\t"),
    ".json":    lambda p: pd.read_json(p),
    ".parquet": lambda p: pd.read_parquet(p),
    ".xlsx":    lambda p: pd.read_excel(p),
    ".xls":     lambda p: pd.read_excel(p),
}


def load_dataframe(path: str | Path) -> tuple[pd.DataFrame, str]:
    """
    Load a dataset file into a DataFrame.

    Returns:
        (dataframe, dataset_name) where dataset_name is the file stem.

    Raises:
        FileNotFoundError: file does not exist.
        ValueError: unsupported file format.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    ext = p.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {list(_LOADERS)}")

    df = loader(p)
    return df, p.stem
