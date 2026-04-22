"""
experiment_config_builder.py
-----------------------------
Phase 4 — Translate ActionDecision → ExperimentConfig.
"""

from __future__ import annotations

from stratml.execution.schemas import ActionDecision, ExperimentConfig

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


def build_experiment_config(action: ActionDecision) -> ExperimentConfig:
    params      = dict(action.parameters)
    action_type = action.action_type

    # All non-switch actions carry model_name injected by orchestrator
    model_name = params.pop("model_name", "LogisticRegression")
    hyperparameters = dict(params)

    if action_type == "switch_model":
        hyperparameters = {}  # fresh start for new model

    elif action_type == "modify_regularization":
        direction = hyperparameters.pop("direction", "increase")
        hyperparameters = _get_reg_mutation(model_name, hyperparameters, direction)

    elif action_type == "increase_model_capacity":
        scale = float(hyperparameters.pop("scale", 1.5))
        hyperparameters = _apply_capacity(model_name, hyperparameters, scale, increase=True)

    elif action_type == "decrease_model_capacity":
        scale = float(hyperparameters.pop("scale", 0.75))
        hyperparameters = _apply_capacity(model_name, hyperparameters, scale, increase=False)

    elif action_type == "change_optimizer":
        lr_scale = float(hyperparameters.pop("learning_rate_scale", 0.1))
        lr = hyperparameters.get("learning_rate", 0.1)
        hyperparameters["learning_rate"] = round(float(lr) * lr_scale, 6)

    elif action_type in ("apply_preprocessing", "early_stop"):
        pass

    elif action_type != "terminate":
        import logging
        logging.getLogger(__name__).warning(
            "Unknown action_type='%s' — treating as apply_preprocessing (model unchanged).",
            action_type,
        )

    model_type     = "dl" if model_name in _DL_MODELS else "ml"
    early_stopping = action_type == "early_stop"
    patience       = int(params.get("early_stopping_patience", 5))

    return ExperimentConfig(
        experiment_id=action.experiment_id,
        model_name=model_name,
        model_type=model_type,
        hyperparameters=hyperparameters,
        preprocessing=action.preprocessing,
        early_stopping=bool(early_stopping),
        early_stopping_patience=patience,
    )
