"""
stability_agent.py
------------------
Decision Council — Stability Agent.

Scores each candidate action on training stability.
Prefers actions that reduce instability, divergence, and high variance.
"""

from __future__ import annotations

from stratml.core.schemas import StateObject
from stratml.decision.learning.uncertainty import UncertaintyEstimate

# How each action affects stability (base score)
_STABILITY_BASE: dict[str, float] = {
    "terminate": 0.95,
    "modify_regularization": 0.85,
    "decrease_model_capacity": 0.80,
    "change_optimizer": 0.70,
    "switch_model": 0.65,
    "increase_model_capacity": 0.35,
}


def score(state: StateObject, estimates: list[UncertaintyEstimate]) -> dict[str, float]:
    """Return stability_score per action_type. Higher = more stable choice."""
    sig = state.signals

    instability_penalty = 0.0
    if sig.diverging:
        instability_penalty += 0.3
    if sig.unstable_training:
        instability_penalty += 0.2
    if sig.high_variance:
        instability_penalty += 0.1

    scores: dict[str, float] = {}
    for e in estimates:
        base = _STABILITY_BASE.get(e.action_type, 0.5)
        # Risky actions get penalised more when training is already unstable
        risk = (1.0 - base) * instability_penalty
        raw = base - risk - e.variance * 0.2
        scores[e.action_type] = round(max(0.0, min(raw, 1.0)), 4)

    return scores
