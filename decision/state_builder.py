"""
state_builder.py
----------------
Dev B — State Pipeline orchestrator.

Two public entry points:

    build_state_from_profile(profile, history)
        Iteration 0.  No ExperimentResult yet.  Builds a bootstrap
        StateObject from DataProfile + meta_features.

    build_state(result, history, profile)
        Iteration 1+.  Full pipeline:
            ExperimentResult
                → state_history  (trajectory features)
                → meta_features  (dataset features, if profile provided)
                → signals        (trajectory-aware signal extraction)
                → StateObject
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from execution.schemas import DataProfile, ExperimentResult
from core.schemas import (
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
from decision.state_history import ExperimentHistory
from decision.meta_features import extract as extract_meta
from decision.signals import compute_signals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_state_from_profile(
    profile: DataProfile,
    *,
    primary_metric: str = "accuracy",
    optimization_goal: str = "maximize",
    allowed_models: Optional[list[str]] = None,
    max_iterations: int = 20,
    time_budget: Optional[float] = None,
) -> StateObject:
    """
    Iteration 0 entry point.

    Called when Team A sends a DataProfile and no ExperimentResult exists yet.
    Produces a bootstrap StateObject so Dev A can generate the very first
    ActionDecision (model selection + preprocessing plan).

    Metrics, trajectory, generalization, and signals are all zeroed/defaulted
    because no experiment has run yet.  Dev A must check
    ``state.meta.iteration == 0`` to know this is a bootstrap state.

    Args:
        profile:           DataProfile received from Team A.
        primary_metric:    Optimisation target (e.g. "accuracy", "r2").
        optimization_goal: "maximize" or "minimize".
        allowed_models:    Model whitelist from CLI config.
        max_iterations:    Hard cap from CLI config.
        time_budget:       Wall-clock budget in seconds (optional).

    Returns:
        A StateObject with iteration=0 and dataset fields populated from
        the DataProfile.  All metric/trajectory fields are zero.
    """
    num_samples = profile.rows
    num_features = profile.columns - 1  # exclude target column

    class_dist = dict(profile.class_distribution) if profile.class_distribution else None
    imbalance_ratio: Optional[float] = None
    if class_dist and len(class_dist) >= 2:
        counts = list(class_dist.values())
        imbalance_ratio = round(max(counts) / max(min(counts), 1), 4)

    return StateObject(
        meta=StateMeta(
            experiment_id="bootstrap",
            iteration=0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
        objective=StateObjective(
            primary_metric=primary_metric,
            optimization_goal=optimization_goal,
        ),
        metrics=StateMetrics(
            primary=0.0,
            secondary=SecondaryMetrics(),
            train_val_gap=0.0,
        ),
        trajectory=StateTrajectory(
            history_length=0,
            improvement_rate=0.0,
            slope=0.0,
            volatility=0.0,
            best_score=0.0,
            mean_score=0.0,
            steps_since_improvement=0,
            trend="stagnating",
        ),
        dataset=StateDataset(
            num_samples=num_samples,
            num_features=num_features,
            feature_to_sample_ratio=round(num_features / max(num_samples, 1), 6),
            missing_ratio=profile.missing_value_ratio,
            class_distribution=class_dist,
            imbalance_ratio=imbalance_ratio,
        ),
        model=StateModel(
            model_name="none",
            model_type="ml",
            hyperparameters={},
            complexity_hint=None,
            runtime=0.0,
            convergence_epoch=0,
            early_stopped=None,
        ),
        generalization=StateGeneralization(
            train_loss=0.0,
            validation_loss=0.0,
            gap=0.0,
        ),
        resources=StateResources(
            runtime=0.0,
            gpu_used=False,
            cpu_time=0.0,
            remaining_budget=float(max_iterations),
            budget_exhausted=False,
        ),
        search=StateSearch(
            models_tried=[],
            unique_models_count=0,
            repeated_configs=0,
        ),
        signals=StateSignals(),   # all False — no experiment has run yet
        uncertainty=StateUncertainty(),
        action_context=StateActionContext(),
        constraints=StateConstraints(
            allowed_models=allowed_models or [],
            max_iterations=max_iterations,
            time_budget=time_budget,
        ),
    )


def build_state(
    result: ExperimentResult,
    *,
    history: Optional[ExperimentHistory] = None,
    profile: Optional[DataProfile] = None,
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
    Convert an ExperimentResult into a fully populated StateObject.

    Pipeline:
        1. Push result into history buffer (state_history)
        2. Compute trajectory features from history
        3. Extract dataset meta-features from DataProfile (if provided)
        4. Assemble StateObject
        5. Compute trajectory-aware signals (signals.py)

    Args:
        result:          Latest ExperimentResult from Team A.
        history:         ExperimentHistory instance (shared across iterations).
                         If None, a single-experiment history is used.
        profile:         DataProfile from Team A (used to populate dataset
                         meta-features).  Optional — defaults to zeros.
        primary_metric:  Optimisation target metric name.
        optimization_goal: "maximize" or "minimize".
        allowed_models:  Model whitelist from CLI config.
        max_iterations:  Hard cap from CLI config.
        time_budget:     Wall-clock budget in seconds (optional).
        previous_action: action_type from the last ActionDecision.
        previous_action_success: Whether the previous action improved the metric.
        models_tried:    Cumulative list of model names run so far.
        repeated_configs: Count of identical (model, hyperparameters) reuses.
        remaining_budget: Remaining iteration budget (optional).

    Returns:
        A fully populated StateObject ready for Dev A's decision engine.
    """
    # --- 1. trajectory ---
    if history is None:
        history = ExperimentHistory()
    history.push(result)
    traj = history.compute_trajectory(primary_metric)

    # adjust improvement_rate sign for "minimize" goals
    improvement_rate = traj.improvement_rate
    if optimization_goal == "minimize":
        improvement_rate = -improvement_rate

    # --- 2. dataset meta-features ---
    if profile is not None:
        meta = extract_meta(profile)
        num_samples = meta.num_samples
        num_features = meta.num_features
        fsr = meta.feature_sample_ratio
        missing_ratio = meta.missing_value_ratio
        class_dist = dict(profile.class_distribution) if profile.class_distribution else None
        imbalance_ratio = meta.imbalance_ratio
    else:
        num_samples = 0
        num_features = 0
        fsr = 0.0
        missing_ratio = 0.0
        class_dist = None
        imbalance_ratio = None

    train_loss = result.metrics.train_loss or 0.0
    val_loss = result.metrics.validation_loss or 0.0
    gap = round(val_loss - train_loss, 6)

    models_tried = list(models_tried or [result.model_name])
    if result.model_name not in models_tried:
        models_tried.append(result.model_name)

    # --- 3. assemble StateObject (signals computed after) ---
    state = StateObject(
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
            primary=traj.best_score if optimization_goal == "maximize" else traj.mean_score,
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
            history_length=traj.history_length,
            improvement_rate=improvement_rate,
            slope=traj.slope,
            volatility=traj.volatility,
            best_score=traj.best_score,
            mean_score=traj.mean_score,
            steps_since_improvement=traj.steps_since_improvement,
            trend=traj.trend,
        ),
        dataset=StateDataset(
            num_samples=num_samples,
            num_features=num_features,
            feature_to_sample_ratio=fsr,
            missing_ratio=missing_ratio,
            class_distribution=class_dist,
            imbalance_ratio=imbalance_ratio,
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
        signals=StateSignals(),   # placeholder — overwritten below
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

    # --- 4. trajectory-aware signals ---
    state.signals = compute_signals(state)
    return state


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _infer_complexity(hyperparameters: dict) -> Optional[str]:
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
