"""
MNIST Tabular — OpenML dataset #554 → data/raw/mnist.csv

Downloads ~55MB. Requires scikit-learn (uses fetch_openml).
"""
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml

out = Path(__file__).parents[1] / "raw" / "mnist.csv"
print("Downloading MNIST from OpenML (this may take a minute)...")
ds = fetch_openml("mnist_784", version=1, as_frame=True, parser="auto")
df = ds.frame.copy()
df = df.rename(columns={"class": "label"})
df.to_csv(out, index=False)
print(f"Saved {out}  ({len(df)} rows, {len(df.columns)-1} features, {df['label'].nunique()} classes)")
