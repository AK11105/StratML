"""
decision_logger.py
------------------
Decision/Logging — Decision Logger.

Writes a DecisionRecord (state snapshot + candidates + selected action)
to runs/decision_logs/{experiment_id}_{iteration}.json after every cycle.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from stratml.core.schemas import (
    ActionDecision,
    CandidateAction,
    DecisionRecord,
    StateObject,
)

_LOG_DIR = Path("runs/decision_logs")


def log(
    state: StateObject,
    candidates: list[CandidateAction],
    decision: ActionDecision,
) -> Path:
    """Persist DecisionRecord to disk. Returns the written file path."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    record = DecisionRecord(
        experiment_id=state.meta.experiment_id,
        iteration=state.meta.iteration,
        timestamp=datetime.now(timezone.utc).isoformat(),
        state_snapshot=state,
        candidate_actions=candidates,
        selected_action=decision,
    )

    filename = f"{state.meta.experiment_id}_{state.meta.iteration:04d}.json"
    path = _LOG_DIR / filename

    with open(path, "w", encoding="utf-8") as f:
        f.write(record.model_dump_json(indent=2))

    return path
