"""
counterfactual.py
-----------------
Decision/Validation — Counterfactual Validator.  [STUB]

Records the action taken each cycle for future A/B comparison.
Full implementation: execute an alternative action in parallel and compare outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path

from stratml.core.schemas import ActionDecision

_CF_LOG = Path("runs/decision_logs/counterfactual_log.jsonl")


def record(decision: ActionDecision) -> None:
    """Append the selected action to the counterfactual log."""
    _CF_LOG.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "experiment_id": decision.experiment_id,
        "iteration": decision.iteration,
        "action_type": decision.action_type,
        "parameters": decision.parameters,
        "confidence": decision.confidence,
    }

    with open(_CF_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
