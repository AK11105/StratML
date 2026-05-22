"""
ml_mutations.py
---------------
Hyperparameter mutation logic for ML (sklearn) models.
Called by experiment_config_builder.py for ML action types.
"""

from __future__ import annotations

# (param_name, default_value, delta_when_increasing)
_REG_PARAM: dict[str, tuple[str, float, float]] = {
    "LogisticRegression": ("C",         1.0,  0.1),
    "SVC":                ("C",         1.0,  0.1),
    "Ridge":              ("alpha",     1.0,  10.0),
    "Lasso":              ("alpha",     1.0,  10.0),
    "ElasticNet":         ("alpha",     1.0,  10.0),
    "RandomForest":       ("max_depth", 10,   -2),
    "GradientBoosting":   ("max_depth",  3,   -1),
    "ExtraTrees":         ("max_depth", 10,   -2),
    "DecisionTree":       ("max_depth", 10,   -2),
}


def mutate_regularization(model_name: str, hp: dict, direction: str) -> dict:
    """Return updated hyperparams with regularization adjusted for the given model."""
    for prefix, (param, default, delta) in _REG_PARAM.items():
        if model_name.startswith(prefix):
            current = hp.get(param, default)
            if direction == "increase":
                if param == "C":
                    new_val = round(float(current) * 0.1, 6)
                elif delta < 0:
                    new_val = max(1, int(current) + int(delta))
                else:
                    new_val = round(float(current) * 10.0, 6)
            else:
                if param == "C":
                    new_val = round(float(current) * 10.0, 6)
                elif delta < 0:
                    new_val = int(current) + 2
                else:
                    new_val = round(float(current) * 0.1, 6)
            return {**hp, param: new_val}
    return hp


def increase_capacity(hp: dict, scale: float) -> dict:
    n = hp.get("n_estimators", 100)
    return {**hp, "n_estimators": int(n * scale)}


def decrease_capacity(hp: dict, scale: float) -> dict:
    n = hp.get("n_estimators", 100)
    return {**hp, "n_estimators": max(10, int(n * scale))}
