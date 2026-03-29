"""
signals.py
----------
Phase 2 (Dev B) — Signal Extraction.

Converts a StateObject (with trajectory context) into a fully populated
StateSignals block.  Replaces the inline rule-based signals in state_builder.py
with trajectory-aware logic.

Signal definitions (rule-based baseline):

    underfitting        primary_metric < 0.60
    overfitting         val_loss - train_loss > 0.10
    well_fitted         not under/over AND primary >= 0.75

    converged           |slope| < 0.001 AND primary >= 0.75
    stagnating          steps_since_improvement >= 2 AND not converged
    diverging           slope < -0.02

    unstable_training   val_loss > train_loss * 2.0
    high_variance       volatility > 0.05

    too_slow            runtime > 300s

    plateau_detected    steps_since_improvement >= 3
    diminishing_returns 0 < improvement_rate < 0.005

Usage:
    from decision.signals import compute_signals
    signals = compute_signals(state)
"""

from __future__ import annotations

from core.schemas import StateObject, StateSignals


def compute_signals(state: StateObject) -> StateSignals:
    """
    Derive all signals from a StateObject with trajectory context.

    Args:
        state: A StateObject with trajectory fields populated
               (by state_history.py or state_builder.py defaults).

    Returns:
        A fully populated StateSignals instance.
    """
    m = state.metrics
    t = state.trajectory
    g = state.generalization
    r = state.resources

    primary = m.primary
    train_loss = g.train_loss
    val_loss = g.validation_loss
    gap = g.gap

    slope = t.slope
    improvement_rate = t.improvement_rate
    volatility = t.volatility
    steps_since = t.steps_since_improvement

    underfitting = primary < 0.60
    overfitting = gap > 0.10
    well_fitted = not underfitting and not overfitting and primary >= 0.75

    converged = abs(slope) < 0.001 and primary >= 0.75
    stagnating = steps_since >= 2 and not converged
    diverging = slope < -0.02

    unstable_training = val_loss > train_loss * 2.0 and train_loss > 0
    high_variance = volatility > 0.05

    too_slow = r.runtime > 300.0

    plateau_detected = steps_since >= 3
    diminishing_returns = 0.0 < improvement_rate < 0.005

    return StateSignals(
        underfitting=underfitting,
        overfitting=overfitting,
        well_fitted=well_fitted,
        converged=converged,
        stagnating=stagnating,
        diverging=diverging,
        unstable_training=unstable_training,
        high_variance=high_variance,
        too_slow=too_slow,
        plateau_detected=plateau_detected,
        diminishing_returns=diminishing_returns,
    )
