from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing(as_frame=True)
df = data.frame
df.to_csv("housing.csv", index=False)
print(f"Saved housing.csv — {len(df)} rows, {len(df.columns)-1} features")
