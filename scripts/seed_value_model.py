"""
seed_value_model.py
-------------------
Runs the full pipeline on each available dataset to populate
runs/decision_logs/decision_dataset.csv with real observed_gain rows,
activating the RandomForest value model.

Usage:
    uv run python scripts/seed_value_model.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stratml.decision.engine import DecisionEngine
from stratml.orchestration.orchestrator import ExecutionOrchestrator
from stratml.execution.schemas import SplitConfig

DATASETS = [
    ("data/external/titanic.csv",        "Survived"),
    ("data/raw/pima.csv",                "Outcome"),
    ("data/raw/iris.csv",                "species"),
    ("data/raw/wine.csv",                "class"),
    ("data/raw/wine_quality_red.csv",    "quality"),
    ("data/raw/digits_noisy.csv",        "label"),
    ("data/raw/california_housing.csv",  "median_house_value"),
    ("data/raw/energydata_complete.csv", "Appliances"),
]

MAX_ITERATIONS = 8  # enough decisions per dataset without being slow


def run_dataset(dataset_path: str, target: str) -> int:
    name = Path(dataset_path).stem
    run_id = f"seed_{name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    print(f"\n[{name}] Starting (target={target}, run_id={run_id})")

    engine = DecisionEngine(
        primary_metric="accuracy",
        max_iterations=MAX_ITERATIONS,
        run_id=run_id,
    )
    orchestrator = ExecutionOrchestrator(
        send_profile=engine.receive_profile,
        send_result=engine.receive_result,
        split_config=SplitConfig(method="stratified"),
        run_id=run_id,
        log=lambda msg: print(f"  {msg}"),
    )
    try:
        orchestrator.run(dataset_path, target)
        print(f"[{name}] Done")
    except Exception as exc:
        print(f"[{name}] Failed: {exc}")

    # Count filled rows contributed to unified dataset
    unified = Path("runs/decision_logs/decision_dataset.csv")
    if unified.exists():
        import csv
        with open(unified, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        filled = sum(1 for r in rows if r.get("observed_gain", "").strip() not in ("", "None"))
        return filled
    return 0


def main():
    print("=== Seeding value model dataset ===")
    unified = Path("runs/decision_logs/decision_dataset.csv")

    for dataset_path, target in DATASETS:
        if not Path(dataset_path).exists():
            print(f"[SKIP] {dataset_path} not found")
            continue
        filled = run_dataset(dataset_path, target)
        print(f"  → Unified dataset: {filled} filled rows so far")

    # Final report
    if unified.exists():
        import csv
        with open(unified, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        total = len(rows)
        filled = sum(1 for r in rows if r.get("observed_gain", "").strip() not in ("", "None"))
        print(f"\n=== Done: {filled}/{total} rows have observed_gain ===")
        if filled >= 50:
            print("✓ RandomForest value model will activate on next run.")
        else:
            print(f"✗ Still need {50 - filled} more rows to activate RF.")
    else:
        print("\n=== No unified dataset found ===")


if __name__ == "__main__":
    main()
