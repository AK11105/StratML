"""Wine Quality Red — UCI via direct URL → data/raw/wine_quality_red.csv"""
from pathlib import Path
import urllib.request
import pandas as pd

out = Path(__file__).parents[1] / "raw" / "wine_quality_red.csv"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
urllib.request.urlretrieve(url, out)
df = pd.read_csv(out, sep=";")
df.to_csv(out, index=False)
print(f"Saved {out}  ({len(df)} rows, {len(df.columns)-1} features, {df['quality'].nunique()} classes)")
