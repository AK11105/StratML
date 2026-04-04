"""
Download a harder dataset for testing multi-iteration reasoning.
Uses sklearn digits (64 features, 10 classes) with added noise to prevent
any single model from converging immediately.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

rng = np.random.default_rng(42)
digits = load_digits()

X = digits.data.copy().astype(float)
# Add Gaussian noise to make it harder
X += rng.normal(0, 4.0, X.shape)

df = pd.DataFrame(X, columns=[f"pixel_{i}" for i in range(X.shape[1])])
df["target"] = digits.target
df.to_csv("digits_noisy.csv", index=False)
print(f"Saved digits_noisy.csv — {len(df)} rows, {X.shape[1]} features, {df['target'].nunique()} classes")
