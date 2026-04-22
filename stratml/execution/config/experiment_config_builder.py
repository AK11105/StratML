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


def build_experiment_config(action: ActionDecision) -> ExperimentConfig:
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
    )
