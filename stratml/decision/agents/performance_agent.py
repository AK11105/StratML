"""
performance_agent.py
--------------------
Decision Council — Performance Agent.

Scores each candidate action on how well it is expected to improve
model accuracy, given the current fitting/trajectory signals.
"""

from __future__ import annotations

from stratml.core.schemas import StateObject
from stratml.decision.learning.uncertainty import UncertaintyEstimate


# Score boost/penalty per action type based on fitting state
_FITTING_PRIORITY: dict[str, dict[str, float]] = {
    "underfitting": {
        "switch_model": 0.85,
        "increase_model_capacity": 0.80,
        "modify_regularization": 0.40,
        "decrease_model_capacity": 0.10,
        "change_optimizer": 0.50,
        "terminate": 0.05,
    },
    "overfitting": {
        "switch_model": 0.60,
        "increase_model_capacity": 0.10,
        "modify_regularization": 0.85,
        "decrease_model_capacity": 0.75,
        "change_optimizer": 0.40,
        "terminate": 0.20,
    },
    "well_fitted": {
        "switch_model": 0.30,
        "increase_model_capacity": 0.20,
        "modify_regularization": 0.20,
        "decrease_model_capacity": 0.20,
        "change_optimizer": 0.20,
        "terminate": 0.90,
    },
    "default": {
        "switch_model": 0.60,
        "increase_model_capacity": 0.50,
        "modify_regularization": 0.50,
        "decrease_model_capacity": 0.40,
        "change_optimizer": 0.45,
        "terminate": 0.30,
    },
}


def score(state: StateObject, estimates: list[UncertaintyEstimate]) -> dict[str, float]:
    """Return performance_score per action_type."""
    sig = state.signals

    if sig.underfitting:
        table = _FITTING_PRIORITY["underfitting"]
    elif sig.overfitting:
        table = _FITTING_PRIORITY["overfitting"]
    elif sig.well_fitted:
        table = _FITTING_PRIORITY["well_fitted"]
    else:
        table = _FITTING_PRIORITY["default"]

    return {
        e.action_type: round(table.get(e.action_type, 0.5) + e.predicted_gain * 0.3, 4)
        for e in estimates
    }
