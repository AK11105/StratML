"""
download_imdb.py
----------------
Download IMDb sentiment dataset via HuggingFace `datasets` and save as CSV.

Tokenizes with a simple whitespace tokenizer (max 512 tokens, padded).
For BERT models, use the HuggingFace tokenizer directly in your pipeline.

Columns: token_0 ... token_511, label  (0=neg, 1=pos)
Shape  : 50000 rows x 513 cols
Output : data/raw/imdb.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
OUT  = ROOT / "data" / "raw" / "imdb.csv"
MAX_LEN = 512


def _simple_tokenize(text: str, vocab: dict, max_len: int) -> list[int]:
    """Whitespace tokenizer with a simple vocab. Returns padded int list."""
    tokens = text.lower().split()[:max_len]
    ids = [vocab.get(t, 1) for t in tokens]   # 1 = <UNK>
    ids += [0] * (max_len - len(ids))          # 0 = <PAD>
    return ids


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("datasets is required: pip install datasets")

    print("Downloading IMDb dataset...")
    ds = load_dataset("imdb")

    all_texts  = [ex["text"]  for split in ["train", "test"] for ex in ds[split]]
    all_labels = [ex["label"] for split in ["train", "test"] for ex in ds[split]]

    print("Building vocabulary...")
    from collections import Counter
    word_counts = Counter(w for t in all_texts for w in t.lower().split())
    # 0=PAD, 1=UNK, 2+ = actual tokens (top 30000)
    vocab = {w: i + 2 for i, (w, _) in enumerate(word_counts.most_common(30000))}

    print(f"Tokenizing {len(all_texts):,} reviews (max_len={MAX_LEN})...")
    rows = [_simple_tokenize(t, vocab, MAX_LEN) for t in all_texts]

    cols = [f"token_{i}" for i in range(MAX_LEN)]
    df = pd.DataFrame(rows, columns=cols, dtype=np.int32)
    df["label"] = all_labels

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved: {OUT}  ({len(df):,} rows x {len(df.columns)} cols)")
    print(f"Vocab size: {len(vocab) + 2} (including PAD and UNK)")


if __name__ == "__main__":
    main()
