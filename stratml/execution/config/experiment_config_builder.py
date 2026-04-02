"""
experiment_config_builder.py
-----------------------------
Phase 4 — Translate ActionDecision → ExperimentConfig.
"""

from __future__ import annotations

from stratml.execution.schemas import ActionDecision, ExperimentConfig

# Maps model names to their type (ml or dl)
_DL_MODELS = {"MLP", "PyTorchMLP"}

_ML_CAPACITY_PARAMS = ("n_estimators", "max_depth", "hidden_units", "layers")
_DL_CAPACITY_PARAMS = ("hidden_units", "layers", "epochs")


def build_experiment_config(action: ActionDecision) -> ExperimentConfig:
    """Translate an ActionDecision into a runnable ExperimentConfig."""
    params = dict(action.parameters)
    action_type = action.action_type

    if action_type == "switch_model":
        model_name = params.pop("model_name")
        hyperparameters = params  # remaining params are hyperparameters

    elif action_type in ("increase_model_capacity", "decrease_model_capacity"):
        # model_name must be in parameters
        model_name = params.pop("model_name")
        hyperparameters = params

    elif action_type in ("modify_regularization", "change_optimizer"):
        model_name = params.pop("model_name")
        hyperparameters = params

    elif action_type == "apply_preprocessing":
        # Keep model unchanged — model_name must still be provided
        model_name = params.pop("model_name")
        hyperparameters = params

    elif action_type == "early_stop":
        model_name = params.pop("model_name")
        hyperparameters = params

    else:
        # terminate or unknown — caller should check action_type before calling this
        raise ValueError(f"Cannot build config for action_type='{action_type}'")

    model_type = "dl" if model_name in _DL_MODELS else "ml"
    early_stopping = action_type == "early_stop" or params.get("early_stopping", False)
    patience = int(params.get("early_stopping_patience", 5))

    return ExperimentConfig(
        experiment_id=action.experiment_id,
        model_name=model_name,
        model_type=model_type,
        hyperparameters=hyperparameters,
        preprocessing=action.preprocessing,
        early_stopping=bool(early_stopping),
        early_stopping_patience=patience,
    )
