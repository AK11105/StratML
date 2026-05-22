"""
dl_mutations.py
---------------
Hyperparameter mutation logic for DL (PyTorch) models.
Called by experiment_config_builder.py for DL action types.
"""

from __future__ import annotations


def increase_capacity(hp: dict, scale: float) -> dict:
    """Scale up hidden_units; add a layer when scale >= 1.5 (max 6 layers)."""
    hp = dict(hp)
    hp["hidden_units"] = int(hp.get("hidden_units", 64) * scale)
    if scale >= 1.5:
        hp["layers"] = min(int(hp.get("layers", 2)) + 1, 6)
    return hp


def decrease_capacity(hp: dict, scale: float) -> dict:
    """Scale down hidden_units (min 16); remove a layer when scale <= 0.75 (min 1)."""
    hp = dict(hp)
    hp["hidden_units"] = max(16, int(hp.get("hidden_units", 64) * scale))
    if scale <= 0.75:
        hp["layers"] = max(1, int(hp.get("layers", 2)) - 1)
    return hp


def mutate_regularization(hp: dict, direction: str) -> dict:
    """Adjust dropout by ±0.1, clamped to [0.0, 0.5]."""
    hp = dict(hp)
    current = float(hp.get("dropout", 0.0))
    if direction == "increase":
        hp["dropout"] = round(min(current + 0.1, 0.5), 2)
    else:
        hp["dropout"] = round(max(current - 0.1, 0.0), 2)
    return hp


def mutate_optimizer(hp: dict, lr_scale: float) -> dict:
    """Scale learning_rate; switch to cosine scheduler + bump weight_decay on aggressive reduction."""
    hp = dict(hp)
    lr = float(hp.get("learning_rate", 1e-3))
    hp["learning_rate"] = round(lr * lr_scale, 8)
    if lr_scale <= 0.1:
        hp.setdefault("scheduler", "cosine")
        wd = float(hp.get("weight_decay", 0.0))
        hp["weight_decay"] = round(min(wd + 1e-4, 1e-2), 6)
    return hp
