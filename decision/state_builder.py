"""
state_builder.py
----------------
Phase 1 (Dev B) — State Pipeline: convert ExperimentResult → StateObject.

Responsibilities:
- Extract base metrics from ExperimentResult
- Compute improvement_rate against the previous experiment (if any)
- Derive train/validation gap (generalization proxy)
- Populate dataset state from ExperimentResult.dataset_snapshot fields
- Populate model context
- Populate resource state
- Maintain a minimal search history (models tried, repeated configs)
- Compute rule-based signals (underfitting, overfitting, convergence, etc.)
- Return a fully populated StateObject

What this module does NOT do (Phase 2+):
- Trajectory slope / volatility over N experiments  → state_history.py
- Dataset meta-features (entropy, variance)         → meta_features.py
- Rich signal extraction                            → signals.py

Those fields are pre-populated with safe defaults so the StateObject
contract is always valid and Dev A can consume it immediately.

Input:
    ExperimentResult

Output:
    StateObject
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from core.schemas import (
    ExperimentResult,
    StateObject,
    StateMeta,
    StateObjective,
    StateMetrics,
    SecondaryMetrics,
    StateTrajectory,
    StateDataset,
    StateModel,
    StateGeneralization,
    StateResources,
    StateSearch,
    StateSignals,
    StateUncertainty,
    StateActionContext,
    StateConstraints,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_state(
    result: ExperimentResult,
    *,
    previous_result: Optional[ExperimentResult] = None,
    primary_metric: str = "accuracy",
    optimization_goal: str = "maximize",
    allowed_models: Optional[list[str]] = None,
    max_iterations: int = 20,
    time_budget: Optional[float] = None,
    previous_action: Optional[str] = None,
    previous_action_success: Optional[bool] = None,
    models_tried: Optional[list[str]] = None,
    repeated_configs: int = 0,
    remaining_budget: Optional[float] = None,
) -> StateObject:
    """
    Convert an ExperimentResult into a StateObject.

    Args:
        result:                  The latest ExperimentResult from Team A.
        previous_result:         The immediately preceding ExperimentResult,
                                 used to compute improvement_rate.
        primary_metric:          Which metric to treat as the optimisation
                                 target (e.g. "accuracy", "f1_score", "r2").
        optimization_goal:       "maximize" or "minimize".
        allowed_models:          Model whitelist from CLI config.
        max_iterations:          Hard cap from CLI config.
        time_budget:             Wall-clock budget in seconds (optional).
        previous_action:         action_type from the last ActionDecision.
        previous_action_success: Whether the previous action improved the
                                 primary metric.
        models_tried:            Cumulative list of model names run so far
                                 (including this result).
        repeated_configs:        Count of iterations that reused an identical
                                 (model, hyperparameters) pair.
        remaining_budget:        Remaining iteration budget (optional).

    Returns:
        A fully populated StateObject ready for Dev A's decision engine.
    """
    primary_value = _extract_primary(result, primary_metric)
    prev_primary = _extract_primary(previous_result, primary_metric) if previous_result else None

    improvement_rate = _compute_improvement(primary_value, prev_primary, optimization_goal)

    train_loss = result.metrics.train_loss or 0.0
    val_loss = result.metrics.validation_loss or 0.0
    gap = round(val_loss - train_loss, 6)

    models_tried = list(models_tried or [result.model_name])
    if result.model_name not in models_tried:
        models_tried.append(result.model_name)

    signals = _compute_signals(
        primary_value=primary_value,
        train_loss=train_loss,
        val_loss=val_loss,
        improvement_rate=improvement_rate,
        runtime=result.runtime,
    )

    return StateObject(
        meta=StateMeta(
            experiment_id=result.experiment_id,
            iteration=result.iteration,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
        objective=StateObjective(
            primary_metric=primary_metric,
            optimization_goal=optimization_goal,
        ),
        metrics=StateMetrics(
            primary=primary_value,
            secondary=SecondaryMetrics(
                accuracy=result.metrics.accuracy,
                precision=result.metrics.precision,
                recall=result.metrics.recall,
                f1_score=result.metrics.f1_score,
                mse=result.metrics.mse,
                rmse=result.metrics.rmse,
                r2=result.metrics.r2,
            ),
            train_val_gap=gap,
        ),
        trajectory=StateTrajectory(
            history_length=1 if previous_result is None else 2,
            improvement_rate=improvement_rate,
            slope=improvement_rate,          # refined by state_history.py (Phase 2)
            volatility=0.0,                  # refined by state_history.py (Phase 2)
            best_score=primary_value,        # refined by state_history.py (Phase 2)
            mean_score=primary_value,        # refined by state_history.py (Phase 2)
            steps_since_improvement=0 if improvement_rate > 0 else 1,
            trend=_infer_trend(improvement_rate),
        ),
        dataset=StateDataset(
            num_samples=result.metrics.accuracy and _safe_int(
                getattr(result, "dataset_snapshot", {})
            ) or 0,
            num_features=0,                  # injected by meta_features.py (Phase 3)
            feature_to_sample_ratio=0.0,     # injected by meta_features.py (Phase 3)
            missing_ratio=0.0,               # injected by meta_features.py (Phase 3)
            class_distribution=None,
            imbalance_ratio=None,
        ),
        model=StateModel(
            model_name=result.model_name,
            model_type=result.model_type,
            hyperparameters=result.hyperparameters,
            complexity_hint=_infer_complexity(result.hyperparameters),
            runtime=result.runtime,
            convergence_epoch=len(result.train_curve),
            early_stopped=None,
        ),
        generalization=StateGeneralization(
            train_loss=train_loss,
            validation_loss=val_loss,
            gap=gap,
        ),
        resources=StateResources(
            runtime=result.runtime,
            gpu_used=result.resource_usage.gpu_used,
            cpu_time=result.resource_usage.cpu_time_sec,
            remaining_budget=remaining_budget,
            budget_exhausted=(remaining_budget is not None and remaining_budget <= 0),
        ),
        search=StateSearch(
            models_tried=models_tried,
            unique_models_count=len(set(models_tried)),
            repeated_configs=repeated_configs,
        ),
        signals=signals,
        uncertainty=StateUncertainty(),
        action_context=StateActionContext(
            previous_action=previous_action,
            previous_action_success=previous_action_success,
            action_effect_magnitude=abs(improvement_rate) if improvement_rate else None,
        ),
        constraints=StateConstraints(
            allowed_models=allowed_models or [],
            max_iterations=max_iterations,
            time_budget=time_budget,
        ),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_primary(result: Optional[ExperimentResult], metric: str) -> float:
    """Pull the primary metric value from an ExperimentResult. Returns 0.0 if absent."""
    if result is None:
        return 0.0
    return float(getattr(result.metrics, metric, None) or 0.0)


def _compute_improvement(current: float, previous: Optional[float], goal: str) -> float:
    """
    Signed improvement rate.

    For "maximize": positive means better.
    For "minimize":  negative delta means better, so we negate.
    """
    if previous is None:
        return 0.0
    delta = current - previous
    return round(delta if goal == "maximize" else -delta, 6)


def _infer_trend(improvement_rate: float) -> str:
    if improvement_rate > 0.001:
        return "improving"
    if improvement_rate < -0.001:
        return "degrading"
    return "stagnating"


def _infer_complexity(hyperparameters: dict) -> Optional[str]:
    """
    Heuristic complexity hint derived from common hyperparameter names.
    Returns None when no recognisable parameters are present.
    """
    n = hyperparameters.get("n_estimators") or hyperparameters.get("hidden_units") or 0
    layers = hyperparameters.get("num_layers") or hyperparameters.get("layers") or 1
    depth = hyperparameters.get("max_depth") or 0

    score = int(n) // 100 + int(layers) + int(depth) // 5
    if score == 0:
        return None
    if score <= 2:
        return "low"
    if score <= 5:
        return "medium"
    return "high"


def _compute_signals(
    *,
    primary_value: float,
    train_loss: float,
    val_loss: float,
    improvement_rate: float,
    runtime: float,
) -> StateSignals:
    """
    Rule-based signal extraction for Phase 1.

    Thresholds are intentionally conservative — Phase 2 (signals.py) will
    refine these with trajectory context.
    """
    gap = val_loss - train_loss

    underfitting = primary_value < 0.60
    overfitting = gap > 0.10
    well_fitted = not underfitting and not overfitting and primary_value >= 0.75

    converged = abs(improvement_rate) < 0.001 and primary_value >= 0.75
    stagnating = abs(improvement_rate) < 0.001 and not converged
    diverging = improvement_rate < -0.05

    unstable_training = val_loss > train_loss * 2.0 and train_loss > 0
    high_variance = gap > 0.20

    too_slow = runtime > 300.0

    plateau_detected = stagnating
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


def _safe_int(obj: object) -> int:
    """Safely extract an integer from an object that may be a dict or missing."""
    if isinstance(obj, dict):
        return int(obj.get("num_samples", 0))
    return 0
