"""Digits (noisy) — sklearn builtin + Gaussian noise → data/raw/digits_noisy.csv"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

out = Path(__file__).parents[1] / "raw" / "digits_noisy.csv"
rng = np.random.default_rng(42)
digits = load_digits()
X = digits.data.astype(float) + rng.normal(0, 4.0, digits.data.shape)
df = pd.DataFrame(X, columns=[f"pixel_{i}" for i in range(X.shape[1])])
df["target"] = digits.target
df.to_csv(out, index=False)
print(f"Saved {out}  ({len(df)} rows, {X.shape[1]} features, {df['target'].nunique()} classes)")
