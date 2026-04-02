"""
efficiency_agent.py
-------------------
Decision Council — Efficiency Agent.

Scores each candidate action on compute cost and runtime efficiency.
Prefers cheap actions when budget is tight or runtime is high.
"""

from __future__ import annotations

from stratml.core.schemas import StateObject
from stratml.decision.learning.uncertainty import UncertaintyEstimate

# Base cost weight per action type (lower = cheaper)
_ACTION_COST: dict[str, float] = {
    "terminate": 0.0,
    "modify_regularization": 0.2,
    "decrease_model_capacity": 0.3,
    "change_optimizer": 0.4,
    "increase_model_capacity": 0.6,
    "switch_model": 0.7,
}


def score(state: StateObject, estimates: list[UncertaintyEstimate]) -> dict[str, float]:
    """Return efficiency_score per action_type. Higher = more efficient."""
    r = state.resources

    budget_pressure = 0.0
    if r.remaining_budget is not None and state.constraints.max_iterations > 0:
        used_ratio = 1.0 - (r.remaining_budget / state.constraints.max_iterations)
        budget_pressure = max(0.0, min(used_ratio, 1.0))

    runtime_pressure = min(r.runtime / 300.0, 1.0)  # normalise to 300s ceiling

    scores: dict[str, float] = {}
    for e in estimates:
        base_cost = _ACTION_COST.get(e.action_type, 0.5)
        # Penalise expensive actions more when budget/runtime pressure is high
        penalty = base_cost * (0.5 * budget_pressure + 0.5 * runtime_pressure)
        raw = 1.0 - base_cost - penalty * 0.3 - e.predicted_cost * 0.2
        scores[e.action_type] = round(max(0.0, min(raw, 1.0)), 4)

    return scores
