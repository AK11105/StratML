"""
download_all.py
---------------
Downloads all auto-downloadable datasets to data/raw/.

Datasets requiring manual Kaggle download are skipped with instructions.
Run from the project root:

    python data/scripts/download_all.py
    python data/scripts/download_all.py --skip mnist   # skip large datasets
"""
from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path

SCRIPTS = [
    ("iris",               "download_iris.py"),
    ("digits_noisy",       "download_digits_noisy.py"),
    ("wine_sklearn",       "download_wine_sklearn.py"),
    ("wine_quality",       "download_wine_quality.py"),
    ("pima",               "download_pima.py"),
    ("california_housing", "download_california_housing.py"),
    ("energy",             "download_energy.py"),
    ("mnist",              "download_mnist.py"),
    ("cifar10",            "download_cifar10.py"),
    ("imdb",               "download_imdb.py"),
]

MANUAL = [
    ("titanic",      "data/external/titanic.csv",     "https://www.kaggle.com/c/titanic/data"),
    ("creditcard",   "data/external/creditcard.csv",  "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"),
]


def run_script(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("_script", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", default=[], help="Dataset keys to skip")
    args = parser.parse_args()
    skip = set(args.skip)

    scripts_dir = Path(__file__).parent

    print("\n=== Downloading datasets to data/raw/ ===\n")
    for key, filename in SCRIPTS:
        if key in skip:
            print(f"  [skip]  {key}")
            continue
        script = scripts_dir / filename
        print(f"  [{key}]")
        try:
            run_script(script)
        except Exception as e:
            print(f"  [ERROR] {key}: {e}")

    print("\n=== Manual downloads required ===\n")
    for key, dest, url in MANUAL:
        if Path(dest).exists():
            print(f"  [ok]    {dest} already exists")
        else:
            print(f"  [missing] {dest}")
            print(f"            Download from: {url}")
            print(f"            See data/external/README.md for instructions")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
