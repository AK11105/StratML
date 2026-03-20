"""
run_ingestion_profiling.py
--------------------------
Phase 1 + Phase 2 runner: ingest a dataset and produce a DataProfile.

Usage:
    python3 execution/run_ingestion_profiling.py <dataset_path> <target_column>

Example:
    python3 execution/run_ingestion_profiling.py data/iris.csv species

Output:
    outputs/<dataset_name>/data_profile.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.ingestion.loader import load_dataframe
from execution.ingestion.validator import build_dataset
from execution.profiling.profiler import build_profile

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


def main(dataset_path: str, target_column: str) -> None:
    print(f"\n[Phase 1] Loading dataset: {dataset_path}")
    df, dataset_name = load_dataframe(dataset_path)
    print(f"  → {len(df)} rows × {len(df.columns)} columns")

    print(f"[Phase 1] Validating schema, target='{target_column}'")
    dataset = build_dataset(df, dataset_name, target_column)
    print(f"  → dataset_type: {dataset.dataset_type}")

    print("[Phase 2] Computing DataProfile ...")
    profile = build_profile(dataset)

    # Persist output
    out_dir = OUTPUTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "data_profile.json"
    out_file.write_text(json.dumps(profile.model_dump(), indent=2))

    print(f"\n[Phase 2] DataProfile saved → {out_file}")
    print(json.dumps(profile.model_dump(), indent=2))

    # Schema round-trip validation
    from execution.schemas import DataProfile
    DataProfile.model_validate(profile.model_dump())
    print("\n✅ DataProfile schema validation passed.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 execution/run_ingestion_profiling.py <dataset_path> <target_column>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
