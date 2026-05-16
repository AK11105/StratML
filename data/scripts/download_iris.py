"""Iris — sklearn builtin → data/raw/iris.csv"""
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

out = Path(__file__).parents[1] / "raw" / "iris.csv"
ds = load_iris(as_frame=True)
df = ds.data.copy()
df["species"] = ds.target
df.to_csv(out, index=False)
print(f"Saved {out}  ({len(df)} rows)")
