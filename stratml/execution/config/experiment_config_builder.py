"""
experiment_config_builder.py
-----------------------------
Phase 4 — Translate ActionDecision → ExperimentConfig.

Capacity mutations are model-type aware:
  - ML  : mutates n_estimators (tree ensembles) or max_depth
  - DL  : mutates hidden_units and/or layers

change_optimizer is DL-only (scales learning_rate).
modify_regularization is ML-only (adjusts C / alpha / max_depth).
"""

from __future__ import annotations

from stratml.execution.schemas import ActionDecision, ExperimentConfig

_DL_MODELS = {"MLP", "CNN1D", "RNN", "PyTorchMLP"}

# ML regularization param per model family
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


def _get_reg_mutation(model_name: str, current_hp: dict, direction: str) -> dict:
    for prefix, (param, default, delta) in _REG_PARAM.items():
        if model_name.startswith(prefix):
            current = current_hp.get(param, default)
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
            return {**current_hp, param: new_val}
    return current_hp


def _increase_dl_capacity(hp: dict, scale: float) -> dict:
    """Scale up hidden_units and optionally add a layer."""
    hp = dict(hp)
    hp["hidden_units"] = int(hp.get("hidden_units", 64) * scale)
    # Add a layer every other capacity increase (scale >= 1.5 threshold)
    if scale >= 1.5:
        hp["layers"] = min(int(hp.get("layers", 2)) + 1, 6)
    return hp


def _decrease_dl_capacity(hp: dict, scale: float) -> dict:
    """Scale down hidden_units and optionally remove a layer."""
    hp = dict(hp)
    hp["hidden_units"] = max(16, int(hp.get("hidden_units", 64) * scale))
    if scale <= 0.75:
        hp["layers"] = max(1, int(hp.get("layers", 2)) - 1)
    return hp


def build_experiment_config(action: ActionDecision) -> ExperimentConfig:
    params      = dict(action.parameters)
    action_type = action.action_type

    model_name      = params.pop("model_name", "LogisticRegression")
    hyperparameters = dict(params)
    model_type      = "dl" if model_name in _DL_MODELS else "ml"

    if action_type == "switch_model":
        hyperparameters = {}  # fresh start for new model

    elif action_type == "modify_regularization":
        direction = hyperparameters.pop("direction", "increase")
        if model_type == "ml":
            hyperparameters = _get_reg_mutation(model_name, hyperparameters, direction)
        else:
            # DL regularization: adjust dropout
            current_dropout = float(hyperparameters.get("dropout", 0.0))
            if direction == "increase":
                hyperparameters["dropout"] = round(min(current_dropout + 0.1, 0.5), 2)
            else:
                hyperparameters["dropout"] = round(max(current_dropout - 0.1, 0.0), 2)

    elif action_type == "increase_model_capacity":
        scale = float(hyperparameters.pop("scale", 1.5))
        if model_type == "dl":
            hyperparameters = _increase_dl_capacity(hyperparameters, scale)
        else:
            n = hyperparameters.get("n_estimators", 100)
            hyperparameters["n_estimators"] = int(n * scale)

    elif action_type == "decrease_model_capacity":
        scale = float(hyperparameters.pop("scale", 0.75))
        if model_type == "dl":
            hyperparameters = _decrease_dl_capacity(hyperparameters, scale)
        else:
            n = hyperparameters.get("n_estimators", 100)
            hyperparameters["n_estimators"] = max(10, int(n * scale))

    elif action_type == "change_optimizer":
        lr_scale = float(hyperparameters.pop("learning_rate_scale", 0.1))
        lr = float(hyperparameters.get("learning_rate", 1e-3))
        hyperparameters["learning_rate"] = round(lr * lr_scale, 8)
        # Also switch scheduler to cosine when reducing LR aggressively
        if lr_scale <= 0.1:
            hyperparameters.setdefault("scheduler", "cosine")

    elif action_type in ("apply_preprocessing", "early_stop"):
        pass

    elif action_type != "terminate":
        raise ValueError(f"Cannot build config for action_type='{action_type}'")

    # DL always uses early stopping — patience from params or sensible default
    if model_type == "dl":
        early_stopping = True
        patience = int(params.get("early_stopping_patience", 5))
    else:
        early_stopping = action_type == "early_stop"
        patience = int(params.get("early_stopping_patience", 5))

    return ExperimentConfig(
        experiment_id=action.experiment_id,
        model_name=model_name,
        model_type=model_type,
        hyperparameters=hyperparameters,
        preprocessing=action.preprocessing,
        early_stopping=early_stopping,
        early_stopping_patience=patience,
    )
