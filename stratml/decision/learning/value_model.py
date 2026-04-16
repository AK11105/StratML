"""
value_model.py
--------------
Decision/Learning — Decision Value Model.

Active path: RandomForestRegressor trained on decision_dataset.csv
             when >= 50 rows with observed_gain are available.
Stub path:   Returns neutral values (predicted_gain=0.05, predicted_cost=0.5)
             until enough data exists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from stratml.core.schemas import CandidateAction, StateObject

log = logging.getLogger(__name__)

_MIN_ROWS = 50

# Injected by DecisionEngine to point at the run-specific CSV
_DATASET_PATH = Path("runs/decision_logs/decision_dataset.csv")

_ACTION_COST_PROXY: dict[str, float] = {
    "terminate": 0.0,
    "modify_regularization": 0.2,
    "decrease_model_capacity": 0.3,
    "change_optimizer": 0.4,
    "increase_model_capacity": 0.6,
    "switch_model": 0.7,
}


@dataclass
class ValuePrediction:
    action_type: str
    parameters: dict
    predicted_gain: float
    predicted_cost: float


def _load_training_data(csv_path: Path):
    """Return (X, y) arrays from the CSV, or (None, None) if insufficient data."""
    if not csv_path.exists():
        return None, None
    try:
        import pandas as pd
        import numpy as np

        df = pd.read_csv(csv_path)
        df = df[df["observed_gain"].notna() & (df["observed_gain"] != "")]
        if len(df) < _MIN_ROWS:
            return None, None

        feature_cols = [
            "best_score", "improvement_rate", "slope", "volatility",
            "steps_since_improvement", "num_samples", "num_features",
            "missing_ratio", "runtime", "remaining_budget",
        ]
        # Encode action_type as integer
        df["action_type_enc"] = df["action_type"].astype("category").cat.codes
        X = df[feature_cols + ["action_type_enc"]].fillna(0).values
        y = df["observed_gain"].astype(float).values
        return X, y
    except Exception as exc:
        log.warning("value_model: failed to load training data (%s)", exc)
        return None, None


def _encode_state_action(state: StateObject, action_type: str) -> list[float]:
    t = state.trajectory
    d = state.dataset
    r = state.resources
    # action_type_enc: use a stable hash mod 100 as a proxy (consistent within a run)
    action_enc = abs(hash(action_type)) % 100
    return [
        t.best_score, t.improvement_rate, t.slope, t.volatility,
        float(t.steps_since_improvement), float(d.num_samples), float(d.num_features),
        d.missing_ratio, r.runtime, r.remaining_budget or 0.0,
        float(action_enc),
    ]


def predict(state: StateObject, candidates: list[CandidateAction]) -> list[ValuePrediction]:
    """Predict (gain, cost) per candidate. Uses RF model when data is sufficient."""
    X_train, y_train = _load_training_data(_DATASET_PATH)

    if X_train is not None:
        try:
            import numpy as np
            from sklearn.ensemble import RandomForestRegressor

            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_train, y_train)

            results = []
            for c in candidates:
                x = _encode_state_action(state, c.action_type)
                gain = float(rf.predict([x])[0])
                gain = max(0.0, min(gain, 1.0))
                cost = _ACTION_COST_PROXY.get(c.action_type, 0.5)
                results.append(ValuePrediction(
                    action_type=c.action_type,
                    parameters=c.parameters,
                    predicted_gain=round(gain, 4),
                    predicted_cost=cost,
                ))
            return results
        except Exception as exc:
            log.warning("value_model: RF prediction failed (%s), using stub", exc)

    # Stub fallback — neutral values matching original contract
    return [
        ValuePrediction(
            action_type=c.action_type,
            parameters=c.parameters,
            predicted_gain=0.05,
            predicted_cost=0.5,
        )
        for c in candidates
    ]
