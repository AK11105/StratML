"""
action_selector.py
------------------
Decision/Policy — Action Selection Policy.

Picks the top-ranked action from the coordinator's output and
builds the final ActionDecision returned to Team A.
"""

from __future__ import annotations

from datetime import datetime, timezone

from stratml.core.schemas import (
    ActionDecision,
    DecisionReason,
    PreprocessingConfig,
    StateObject,
)
from stratml.decision.agents.coordinator_agent import RankedAction

_DEFAULT_PREPROCESSING = PreprocessingConfig(
    missing_value_strategy="mean",
    scaling="standard",
    encoding="onehot",
    imbalance_strategy="none",
    feature_selection="none",
)


def select(state: StateObject, ranked: list[RankedAction]) -> ActionDecision:
    """Pick the highest-scored action and return an ActionDecision."""
    best = ranked[0]

    trigger = _infer_trigger(state)
    evidence = _build_evidence(state)

    return ActionDecision(
        experiment_id=state.meta.experiment_id,
        iteration=state.meta.iteration,
        action_type=best.action_type,
        parameters=best.parameters,
        preprocessing=_DEFAULT_PREPROCESSING,
        expected_gain=best.predicted_gain,
        expected_cost=best.predicted_cost,
        confidence=best.confidence,
        agent_scores=best.agent_scores,
        reason=DecisionReason(
            trigger=trigger,
            evidence=evidence,
            source="rule",
        ),
    )


def _infer_trigger(state: StateObject) -> str:
    sig = state.signals
    if state.meta.iteration == 0:
        return "bootstrap"
    if sig.converged and sig.well_fitted:
        return "convergence"
    if sig.underfitting:
        return "underfitting"
    if sig.overfitting:
        return "overfitting"
    if sig.stagnating or sig.plateau_detected:
        return "stagnation"
    if sig.diverging:
        return "divergence"
    if sig.diminishing_returns:
        return "diminishing_returns"
    return "exploration"


def _build_evidence(state: StateObject) -> dict:
    sig = state.signals
    t = state.trajectory
    return {
        "primary_metric": state.metrics.primary,
        "best_score": t.best_score,
        "slope": t.slope,
        "steps_since_improvement": t.steps_since_improvement,
        "underfitting": sig.underfitting,
        "overfitting": sig.overfitting,
        "converged": sig.converged,
        "stagnating": sig.stagnating,
        "remaining_budget": state.resources.remaining_budget,
    }
