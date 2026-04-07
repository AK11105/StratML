"""Download wine quality dataset — harder than iris, no single model dominates."""
from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine(as_frame=True)
df = wine.data.copy()
df["target"] = wine.target
df.to_csv("wine.csv", index=False)
print(f"Saved wine.csv — {len(df)} rows, {len(df.columns)-1} features, {df['target'].nunique()} classes")
