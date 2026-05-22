"""Wine Quality (sklearn) — sklearn builtin → data/raw/wine.csv"""
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_wine

out = Path(__file__).parents[1] / "raw" / "wine.csv"
ds = load_wine(as_frame=True)
df = ds.data.copy()
df["target"] = ds.target
df.to_csv(out, index=False)
print(f"Saved {out}  ({len(df)} rows, {len(df.columns)-1} features, {df['target'].nunique()} classes)")
