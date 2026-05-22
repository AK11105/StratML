"""Appliances Energy Prediction — UCI via direct URL → data/raw/energydata_complete.csv"""
from pathlib import Path
import urllib.request
import pandas as pd

out = Path(__file__).parents[1] / "raw" / "energydata_complete.csv"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
urllib.request.urlretrieve(url, out)
df = pd.read_csv(out)
# Drop date column — not useful for tabular regression
df = df.drop(columns=["date"], errors="ignore")
df.to_csv(out, index=False)
print(f"Saved {out}  ({len(df)} rows, {len(df.columns)-1} features)")
