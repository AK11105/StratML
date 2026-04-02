"""
uncertainty.py
--------------
Decision/Learning — Uncertainty Estimator.  [STUB]

Runs an ensemble of value models and computes variance across predictions.
Stub returns confidence=0.5, variance=0.0 until ensemble is trained.
"""

from __future__ import annotations

from dataclasses import dataclass

from stratml.decision.learning.value_model import ValuePrediction


@dataclass
class UncertaintyEstimate:
    action_type: str
    parameters: dict
    predicted_gain: float
    predicted_cost: float
    confidence: float
    variance: float


def estimate(predictions: list[ValuePrediction]) -> list[UncertaintyEstimate]:
    """Wrap each prediction with neutral confidence/variance."""
    return [
        UncertaintyEstimate(
            action_type=p.action_type,
            parameters=p.parameters,
            predicted_gain=p.predicted_gain,
            predicted_cost=p.predicted_cost,
            confidence=0.5,
            variance=0.0,
        )
        for p in predictions
    ]
