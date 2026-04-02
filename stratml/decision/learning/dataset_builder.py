"""
dataset_builder.py
------------------
Decision/Learning — Decision Dataset Builder.  [STUB]

Appends one row per decision cycle to decision_dataset.csv.
Columns: state features + action_type + observed_gain (filled retroactively).

Full implementation: populate observed_gain after next ExperimentResult arrives.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from stratml.core.schemas import CandidateAction, StateObject

_DATASET_PATH = Path("runs/decision_logs/decision_dataset.csv")

_COLUMNS = [
    "experiment_id", "iteration",
    "primary_metric", "best_score", "improvement_rate", "slope", "volatility",
    "steps_since_improvement", "trend",
    "underfitting", "overfitting", "well_fitted", "converged", "stagnating",
    "num_samples", "num_features", "missing_ratio",
    "runtime", "remaining_budget",
    "action_type", "action_params",
    "observed_gain",  # filled retroactively
]


def record(state: StateObject, action: CandidateAction) -> None:
    """Append a row for the chosen action. observed_gain left blank until next result."""
    _DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not _DATASET_PATH.exists()

    row = {
        "experiment_id": state.meta.experiment_id,
        "iteration": state.meta.iteration,
        "primary_metric": state.objective.primary_metric,
        "best_score": state.trajectory.best_score,
        "improvement_rate": state.trajectory.improvement_rate,
        "slope": state.trajectory.slope,
        "volatility": state.trajectory.volatility,
        "steps_since_improvement": state.trajectory.steps_since_improvement,
        "trend": state.trajectory.trend,
        "underfitting": state.signals.underfitting,
        "overfitting": state.signals.overfitting,
        "well_fitted": state.signals.well_fitted,
        "converged": state.signals.converged,
        "stagnating": state.signals.stagnating,
        "num_samples": state.dataset.num_samples,
        "num_features": state.dataset.num_features,
        "missing_ratio": state.dataset.missing_ratio,
        "runtime": state.resources.runtime,
        "remaining_budget": state.resources.remaining_budget,
        "action_type": action.action_type,
        "action_params": str(action.parameters),
        "observed_gain": "",
    }

    with open(_DATASET_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
