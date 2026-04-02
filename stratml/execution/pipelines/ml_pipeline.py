"""
ml_pipeline.py
--------------
Phase 5 — Train a scikit-learn model and return raw predictions + timing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR

from stratml.execution.schemas import ExperimentConfig, DataSplit

MODEL_REGISTRY: dict = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "SVC": SVC,
    "SVR": SVR,
}


@dataclass
class MLPipelineResult:
    model: object
    y_val_pred: np.ndarray
    train_curve: list[float]   # single value — no epochs in ML
    val_curve: list[float]
    runtime: float


def run_ml_pipeline(config: ExperimentConfig, data_split: DataSplit) -> MLPipelineResult:
    """Instantiate, train, and evaluate a sklearn model."""
    cls = MODEL_REGISTRY.get(config.model_name)
    if cls is None:
        raise ValueError(f"Unknown model '{config.model_name}'. Available: {list(MODEL_REGISTRY)}")

    # Filter hyperparameters to only those accepted by the model
    import inspect
    valid_params = inspect.signature(cls.__init__).parameters
    hp = {k: v for k, v in config.hyperparameters.items() if k in valid_params}

    model = cls(**hp)

    t0 = time.perf_counter()
    model.fit(data_split.X_train, data_split.y_train)
    runtime = round(time.perf_counter() - t0, 4)

    y_val_pred = model.predict(data_split.X_val)

    # ML has no epoch loop — represent as single-step curves using loss proxy
    try:
        from sklearn.metrics import log_loss
        train_loss = round(float(log_loss(data_split.y_train, model.predict_proba(data_split.X_train))), 6)
        val_loss   = round(float(log_loss(data_split.y_val,   model.predict_proba(data_split.X_val))),   6)
    except Exception:
        train_loss = 0.0
        val_loss   = 0.0

    return MLPipelineResult(
        model=model,
        y_val_pred=y_val_pred,
        train_curve=[train_loss],
        val_curve=[val_loss],
        runtime=runtime,
    )
