"""
loader.py
---------
Phase 1 — Dataset Ingestion: load a CSV file into a pandas DataFrame.

Responsibilities:
- Read the file from disk
- Detect file format from extension (CSV supported; extensible)
- Return a raw DataFrame for downstream validation

Raises:
    FileNotFoundError: if the path does not exist
    ValueError: if the file format is unsupported
"""

from pathlib import Path
import pandas as pd


_SUPPORTED_FORMATS = {".csv"}


def load_dataframe(path: str | Path) -> tuple[pd.DataFrame, str]:
    """
    Load a dataset file into a DataFrame.

    Args:
        path: Absolute or relative path to the dataset file.

    Returns:
        (dataframe, dataset_name) tuple where dataset_name is the file stem.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: unsupported file extension.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    ext = p.suffix.lower()
    if ext not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{ext}'. Supported: {_SUPPORTED_FORMATS}"
        )

    df = pd.read_csv(p)
    return df, p.stem
