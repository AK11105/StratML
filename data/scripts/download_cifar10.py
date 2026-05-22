"""
download_cifar10.py
-------------------
Download CIFAR-10 via torchvision and save as a flat CSV.

Columns: pixel_0 ... pixel_3071, label
Shape  : 60000 rows x 3073 cols (3072 pixels + 1 label)
Output : data/raw/cifar10.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
OUT  = ROOT / "data" / "raw" / "cifar10.csv"


def main() -> None:
    try:
        from torchvision.datasets import CIFAR10
        import torchvision.transforms as T
    except ImportError:
        raise SystemExit("torchvision is required: pip install torchvision")

    cache = ROOT / "data" / "raw" / "_cifar10_cache"
    transform = T.ToTensor()

    print("Downloading CIFAR-10 (train + test)...")
    train_ds = CIFAR10(root=str(cache), train=True,  download=True, transform=transform)
    test_ds  = CIFAR10(root=str(cache), train=False, download=True, transform=transform)

    def to_array(ds):
        X = np.stack([img.numpy().flatten() for img, _ in ds]).astype(np.float32)
        y = np.array([lbl for _, lbl in ds], dtype=np.int64)
        return X, y

    print("Converting to flat arrays...")
    X_tr, y_tr = to_array(train_ds)
    X_te, y_te = to_array(test_ds)

    X = np.vstack([X_tr, X_te])
    y = np.concatenate([y_tr, y_te])

    cols = [f"pixel_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved: {OUT}  ({len(df):,} rows x {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
