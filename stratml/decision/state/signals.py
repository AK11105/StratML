"""
signals.py
----------
Phase 2 (Dev B) — Signal Extraction.
Converts a StateObject into a fully populated StateSignals block.

Each signal is now a strength string ("none" | "weak" | "strong") paired
with a confidence score in [0.0, 1.0].
"""

from __future__ import annotations

from stratml.core.schemas import StateObject, StateSignals


def _strength(condition_weak: bool, condition_strong: bool) -> str:
    if condition_strong:
        return "strong"
    if condition_weak:
        return "weak"
    return "none"


def compute_signals(state: StateObject) -> StateSignals:
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

    # --- underfitting ---
    uf_weak   = 0.60 <= primary < 0.70
    uf_strong = primary < 0.60
    uf_conf   = round(max(0.0, min((0.70 - primary) / 0.70, 1.0)), 4)

    # --- overfitting ---
    of_weak   = 0.05 < gap <= 0.10
    of_strong = gap > 0.10
    of_conf   = round(min(gap / 0.20, 1.0), 4)

    # --- well_fitted ---
    not_uf = not (uf_weak or uf_strong)
    not_of = not (of_weak or of_strong)
    wf_weak   = not_uf and not_of and 0.70 <= primary < 0.75
    wf_strong = not_uf and not_of and primary >= 0.75
    wf_conf   = round(min(primary / 0.90, 1.0), 4) if (wf_weak or wf_strong) else 0.0

    # --- converged ---
    cv_weak   = abs(slope) < 0.005 and primary >= 0.70
    cv_strong = abs(slope) < 0.001 and primary >= 0.75
    cv_conf   = round(max(0.0, 1.0 - abs(slope) / 0.005), 4) if cv_weak or cv_strong else 0.0

    # --- stagnating ---
    converged_any = cv_weak or cv_strong
    sg_weak   = steps_since >= 2 and not converged_any
    sg_strong = steps_since >= 4 and not converged_any
    sg_conf   = round(min(steps_since / 5.0, 1.0), 4) if sg_weak or sg_strong else 0.0

    # --- diverging ---
    dv_weak   = -0.02 <= slope < -0.01
    dv_strong = slope < -0.02
    dv_conf   = round(min(abs(slope) / 0.05, 1.0), 4) if dv_weak or dv_strong else 0.0

    # --- unstable_training ---
    ratio = val_loss / train_loss if train_loss > 0 else 0.0
    ut_weak   = 1.5 < ratio <= 2.0
    ut_strong = ratio > 2.0
    ut_conf   = round(min((ratio - 1.0) / 2.0, 1.0), 4) if ut_weak or ut_strong else 0.0

    # --- high_variance ---
    hv_weak   = 0.03 < volatility <= 0.05
    hv_strong = volatility > 0.05
    hv_conf   = round(min(volatility / 0.10, 1.0), 4) if hv_weak or hv_strong else 0.0

    # --- too_slow ---
    ts_weak   = 200.0 < r.runtime <= 300.0
    ts_strong = r.runtime > 300.0
    ts_conf   = round(min(r.runtime / 600.0, 1.0), 4) if ts_weak or ts_strong else 0.0

    # --- plateau_detected ---
    pl_weak   = steps_since == 3
    pl_strong = steps_since >= 4
    pl_conf   = round(min(steps_since / 5.0, 1.0), 4) if pl_weak or pl_strong else 0.0

    # --- diminishing_returns ---
    dr_weak   = 0.005 <= improvement_rate < 0.01
    dr_strong = 0.0 < improvement_rate < 0.005
    dr_conf   = round(max(0.0, 1.0 - improvement_rate / 0.01), 4) if dr_weak or dr_strong else 0.0

    return StateSignals(
        underfitting=_strength(uf_weak, uf_strong),
        underfitting_confidence=uf_conf,
        overfitting=_strength(of_weak, of_strong),
        overfitting_confidence=of_conf,
        well_fitted=_strength(wf_weak, wf_strong),
        well_fitted_confidence=wf_conf,
        converged=_strength(cv_weak, cv_strong),
        converged_confidence=cv_conf,
        stagnating=_strength(sg_weak, sg_strong),
        stagnating_confidence=sg_conf,
        diverging=_strength(dv_weak, dv_strong),
        diverging_confidence=dv_conf,
        unstable_training=_strength(ut_weak, ut_strong),
        unstable_training_confidence=ut_conf,
        high_variance=_strength(hv_weak, hv_strong),
        high_variance_confidence=hv_conf,
        too_slow=_strength(ts_weak, ts_strong),
        too_slow_confidence=ts_conf,
        plateau_detected=_strength(pl_weak, pl_strong),
        plateau_detected_confidence=pl_conf,
        diminishing_returns=_strength(dr_weak, dr_strong),
        diminishing_returns_confidence=dr_conf,
    )
