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

from typing import Optional

from stratml.core.schemas import ActionDecision
from stratml.decision.agents.coordinator_agent import RankedAction

_CF_LOG = Path("runs/decision_logs/counterfactual_log.jsonl")


def record(decision: ActionDecision, runner_up: Optional[RankedAction] = None) -> None:
    """Append the selected action and runner-up to the counterfactual log."""
    _CF_LOG.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "experiment_id": decision.experiment_id,
        "iteration": decision.iteration,
        "action_type": decision.action_type,
        "parameters": decision.parameters,
        "confidence": decision.confidence,
        "expected_gain": decision.expected_gain,
        "runner_up_action": runner_up.action_type if runner_up else None,
        "runner_up_predicted_gain": runner_up.predicted_gain if runner_up else None,
    }

    with open(_CF_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
