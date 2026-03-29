"""
run_state_builder.py
--------------------
Phase 1 (Dev B) runner: load a saved ExperimentResult JSON and produce a
StateObject, printing it to stdout and saving it to outputs/.

Usage:
    python3 decision/run_state_builder.py <experiment_result_json> [<previous_result_json>]

Examples:
    # First iteration — no previous result
    python3 decision/run_state_builder.py outputs/iris/experiment_result.json

    # Subsequent iteration — pass previous result for improvement_rate
    python3 decision/run_state_builder.py outputs/iris/experiment_result_2.json \\
                                          outputs/iris/experiment_result_1.json

Output:
    outputs/<dataset_name>/state_object.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.schemas import ExperimentResult
from decision.state_builder import build_state

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


def main(result_path: str, previous_path: str | None = None) -> None:
    print(f"\n[Phase 1 — Dev B] Loading ExperimentResult: {result_path}")
    result = ExperimentResult.model_validate(
        json.loads(Path(result_path).read_text())
    )
    print(f"  → experiment_id : {result.experiment_id}")
    print(f"  → iteration     : {result.iteration}")
    print(f"  → model         : {result.model_name} ({result.model_type})")

    previous: ExperimentResult | None = None
    if previous_path:
        print(f"[Phase 1 — Dev B] Loading previous ExperimentResult: {previous_path}")
        previous = ExperimentResult.model_validate(
            json.loads(Path(previous_path).read_text())
        )

    print("[Phase 1 — Dev B] Building StateObject ...")
    state = build_state(result, previous_result=previous)

    # Persist
    out_dir = OUTPUTS_DIR / result.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "state_object.json"
    out_file.write_text(json.dumps(state.model_dump(), indent=2))

    print(f"\n[Phase 1 — Dev B] StateObject saved → {out_file}")
    print(json.dumps(state.model_dump(), indent=2))

    # Schema round-trip validation
    from core.schemas import StateObject
    StateObject.model_validate(state.model_dump())
    print("\n✅ StateObject schema validation passed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python3 decision/run_state_builder.py "
            "<experiment_result_json> [<previous_result_json>]"
        )
        sys.exit(1)

    main(sys.argv[1], sys.argv[2] if len(sys.argv) >= 3 else None)
