"""
calibration.py
--------------
Decision/Learning — Calibration Layer.  [STUB]

Corrects bias in predicted gains using a fitted isotonic regression.
Stub is a pass-through until enough (predicted, actual) pairs are collected.
"""

from __future__ import annotations

from stratml.decision.learning.value_model import ValuePrediction


def calibrate(predictions: list[ValuePrediction]) -> list[ValuePrediction]:
    """Pass-through. Returns predictions unchanged."""
    return predictions
