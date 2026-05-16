"""Pima Indians Diabetes — UCI via ucimlrepo → data/raw/pima.csv"""
from pathlib import Path

out = Path(__file__).parents[1] / "raw" / "pima.csv"

try:
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=34)
    import pandas as pd
    df = ds.data.features.copy()
    df["Outcome"] = ds.data.targets.values
    df.to_csv(out, index=False)
    print(f"Saved {out}  ({len(df)} rows)")
except ImportError:
    # Fallback: direct URL download
    import urllib.request, pandas as pd
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI",
            "DiabetesPedigreeFunction","Age","Outcome"]
    urllib.request.urlretrieve(url, out)
    df = pd.read_csv(out, header=None, names=cols)
    df.to_csv(out, index=False)
    print(f"Saved {out}  ({len(df)} rows)")
