"""California Housing — sklearn builtin → data/raw/california_housing.csv"""
from pathlib import Path
from sklearn.datasets import fetch_california_housing

out = Path(__file__).parents[1] / "raw" / "california_housing.csv"
ds = fetch_california_housing(as_frame=True)
df = ds.frame
df.to_csv(out, index=False)
print(f"Saved {out}  ({len(df)} rows, {len(df.columns)-1} features)")
