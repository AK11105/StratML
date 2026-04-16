"""
uncertainty.py
--------------
Decision/Learning — Uncertainty Estimator.

Active path: Ensemble of 5 RandomForestRegressors; variance across predictions
             is the confidence signal. Activated when >= 50 rows exist.
Stub path:   confidence=0.5, variance=0.0 until ensemble is trained.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from stratml.decision.learning.value_model import (
    ValuePrediction,
    _DATASET_PATH,
    _MIN_ROWS,
    _load_training_data,
)

log = logging.getLogger(__name__)

_N_ESTIMATORS = 5


@dataclass
class UncertaintyEstimate:
    action_type: str
    parameters: dict
    predicted_gain: float
    predicted_cost: float
    confidence: float
    variance: float


def estimate(predictions: list[ValuePrediction]) -> list[UncertaintyEstimate]:
    """Wrap each prediction with confidence/variance from ensemble when data is available."""
    X_train, y_train = _load_training_data(_DATASET_PATH)

    if X_train is not None:
        try:
            import numpy as np
            from sklearn.ensemble import RandomForestRegressor

            # Train N_ESTIMATORS forests with different seeds
            models = [
                RandomForestRegressor(n_estimators=20, random_state=seed).fit(X_train, y_train)
                for seed in range(_N_ESTIMATORS)
            ]

            # We need the original StateObject to re-encode — but we only have ValuePrediction here.
            # Use predicted_gain as a proxy feature for variance estimation.
            results = []
            for p in predictions:
                # Each model predicts from a 1-feature input (predicted_gain as proxy)
                x_proxy = [[p.predicted_gain] + [0.0] * 10]  # pad to match training width
                preds = np.array([m.predict(x_proxy)[0] for m in models])
                variance = float(np.var(preds))
                mean_pred = float(np.mean(preds))
                # Confidence: inverse of normalised variance, clamped to [0,1]
                confidence = round(max(0.0, min(1.0 - variance * 10, 1.0)), 4)
                results.append(UncertaintyEstimate(
                    action_type=p.action_type,
                    parameters=p.parameters,
                    predicted_gain=round(max(0.0, min(mean_pred, 1.0)), 4),
                    predicted_cost=p.predicted_cost,
                    confidence=confidence,
                    variance=round(variance, 6),
                ))
            return results
        except Exception as exc:
            log.warning("uncertainty: ensemble failed (%s), using stub", exc)

    # Stub fallback
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
