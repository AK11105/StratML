"""
experiment_config_builder.py
-----------------------------
Phase 4 — Translate ActionDecision → ExperimentConfig.

Mutation logic lives in ml_mutations.py and dl_mutations.py.
This file is a pure dispatcher — one function, no mutation logic.
"""

from __future__ import annotations

from stratml.execution.schemas import ActionDecision, ExperimentConfig
from stratml.execution.config import ml_mutations, dl_mutations

_DL_MODELS = {"MLP", "CNN1D", "RNN", "PyTorchMLP"}

# Regularization param per model family
_REG_PARAM: dict[str, tuple[str, float, float]] = {
    # model_name_prefix -> (param, default, scale_factor when increasing)
    "LogisticRegression": ("C",     1.0,  0.1),   # lower C = more regularization
    "SVC":                ("C",     1.0,  0.1),
    "Ridge":              ("alpha", 1.0,  10.0),
    "Lasso":              ("alpha", 1.0,  10.0),
    "ElasticNet":         ("alpha", 1.0,  10.0),
    "RandomForest":       ("max_depth", 10, -2),   # reduce depth
    "GradientBoosting":   ("max_depth",  3, -1),
    "ExtraTrees":         ("max_depth", 10, -2),
    "DecisionTree":       ("max_depth", 10, -2),
}

# Capacity params per model family: list of (param, default, scale_direction)
# scale_direction: +1 = multiply by scale to increase, -1 = divide by scale to increase
_CAPACITY_PARAMS: dict[str, list[tuple[str, object, int]]] = {
    "RandomForest":       [("n_estimators", 100, 1), ("max_depth", 10, 1), ("min_samples_split", 2, -1), ("max_features", "sqrt", 0)],
    "ExtraTrees":         [("n_estimators", 100, 1), ("max_depth", 10, 1)],
    "GradientBoosting":   [("n_estimators", 100, 1), ("learning_rate", 0.1, -1), ("subsample", 1.0, 1)],
    "AdaBoost":           [("n_estimators", 50,  1)],
    "DecisionTree":       [("max_depth", 5, 1)],
    "LogisticRegression": [("max_iter", 100, 1)],
    "SVC":                [("max_iter", 1000, 1)],
    "KNeighbors":         [("n_neighbors", 5, -1)],  # fewer neighbors = more capacity
    "SGD":                [("max_iter", 1000, 1)],
}


def _get_reg_mutation(model_name: str, current_hp: dict, direction: str) -> dict:
    """Return updated hyperparams with regularization adjusted."""
    for prefix, (param, default, delta) in _REG_PARAM.items():
        if model_name.startswith(prefix):
            current = current_hp.get(param, default)
            if direction == "increase":
                # More regularization
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
            return {**current_hp, param: new_val}
    return current_hp


def _apply_capacity(model_name: str, hp: dict, scale: float, increase: bool) -> dict:
    """Scale capacity-related hyperparameters for the given model family."""
    hp = dict(hp)
    for prefix, params in _CAPACITY_PARAMS.items():
        if model_name.startswith(prefix):
            for param, default, direction in params:
                if direction == 0:
                    continue  # categorical param, skip
                current = hp.get(param, default)
                if not isinstance(current, (int, float)):
                    continue
                if direction == 1:
                    new_val = current * scale if increase else current * (1.0 / scale)
                else:
                    new_val = current * (1.0 / scale) if increase else current * scale
                hp[param] = max(1, int(round(new_val))) if isinstance(default, int) else round(new_val, 6)
            return hp
    # Fallback: just scale n_estimators if present
    if "n_estimators" in hp:
        n = hp["n_estimators"]
        hp["n_estimators"] = max(10, int(n * scale if increase else n * (1.0 / scale)))
    return hp


def build_experiment_config(action: ActionDecision, tune: bool = False) -> ExperimentConfig:
    params      = dict(action.parameters)
    action_type = action.action_type
    model_name  = params.pop("model_name", "LogisticRegression")
    hp          = dict(params)
    is_dl       = model_name in _DL_MODELS

    if action_type == "switch_model":
        hp = {}

    elif action_type == "modify_regularization":
        direction = hp.pop("direction", "increase")
        hp = dl_mutations.mutate_regularization(hp, direction) if is_dl \
            else ml_mutations.mutate_regularization(model_name, hp, direction)

    elif action_type == "increase_model_capacity":
        scale = float(hp.pop("scale", 1.5))
        hp = dl_mutations.increase_capacity(hp, scale) if is_dl \
            else ml_mutations.increase_capacity(hp, scale)

    elif action_type == "decrease_model_capacity":
        scale = float(hp.pop("scale", 0.75))
        hp = dl_mutations.decrease_capacity(hp, scale) if is_dl \
            else ml_mutations.decrease_capacity(hp, scale)

    elif action_type == "change_optimizer":
        lr_scale = float(hp.pop("learning_rate_scale", 0.1))
        hp = dl_mutations.mutate_optimizer(hp, lr_scale)

    elif action_type in ("apply_preprocessing", "early_stop", "terminate"):
        pass

    elif action_type != "terminate":
        import logging
        logging.getLogger(__name__).warning(
            "Unknown action_type='%s' — treating as apply_preprocessing (model unchanged).",
            action_type,
        )

    else:
        raise ValueError(f"Unknown action_type: '{action_type}'")

    model_type = "dl" if is_dl else "ml"
    early_stopping = True if is_dl else (action_type == "early_stop")
    patience = int(params.get("early_stopping_patience", 5))

    return ExperimentConfig(
        experiment_id=action.experiment_id,
        model_name=model_name,
        model_type=model_type,
        hyperparameters=hp,
        preprocessing=action.preprocessing,
        early_stopping=early_stopping,
        early_stopping_patience=patience,
        tune=tune and not is_dl,
    )
