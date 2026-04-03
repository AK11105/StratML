"""
value_model.py
--------------
Decision/Learning — Decision Value Model.  [STUB]

Predicts (predicted_gain, predicted_cost) per candidate action.
Stub returns neutral values until decision_dataset.csv has enough rows to train.
"""

from __future__ import annotations

from dataclasses import dataclass

from stratml.core.schemas import CandidateAction, StateObject


@dataclass
class ValuePrediction:
    action_type: str
    parameters: dict
    predicted_gain: float
    predicted_cost: float


def predict(state: StateObject, candidates: list[CandidateAction]) -> list[ValuePrediction]:
    """Return neutral predictions for each candidate action."""
    return [
        ValuePrediction(
            action_type=c.action_type,
            parameters=c.parameters,
            predicted_gain=0.05,
            predicted_cost=0.5,
        )
        for c in candidates
    ]
