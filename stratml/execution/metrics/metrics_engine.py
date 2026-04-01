"""
metrics_engine.py
-----------------
Phase 6 — Convert raw predictions into a validated ExperimentMetrics object.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score,
)

from stratml.execution.schemas import ExperimentMetrics


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    train_curve: list[float],
    val_curve: list[float],
    problem_type: str,
) -> ExperimentMetrics:
    """Compute ExperimentMetrics from predictions and loss curves."""
    train_loss = round(float(train_curve[-1]), 6) if train_curve else None
    val_loss   = round(float(val_curve[-1]),   6) if val_curve   else None

    if problem_type == "classification":
        avg = "weighted"
        return ExperimentMetrics(
            accuracy=round(float(accuracy_score(y_true, y_pred)), 6),
            f1_score=round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 6),
            precision=round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 6),
            recall=round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 6),
            train_loss=train_loss,
            validation_loss=val_loss,
        )
    else:
        mse  = round(float(mean_squared_error(y_true, y_pred)), 6)
        rmse = round(float(np.sqrt(mse)), 6)
        r2   = round(float(r2_score(y_true, y_pred)), 6)
        return ExperimentMetrics(
            mse=mse,
            rmse=rmse,
            r2=r2,
            train_loss=train_loss,
            validation_loss=val_loss,
        )
