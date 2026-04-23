"""
test_full_pipeline.py
---------------------
Integration tests — full execution loop with a stub Team B.

Tests the complete path:
  loader → validator → profiler → splitter → config_builder →
  preprocessor → ml_pipeline → metrics_engine → result_builder

No real Team B involved — ActionDecisions are returned by stub functions.
"""

import pytest

from stratml.execution.data.loader import load_dataframe
from stratml.execution.data.validator import build_dataset
from stratml.execution.data.profiler import build_profile
from stratml.execution.preprocessing.splitter import split_dataset
from stratml.execution.preprocessing.preprocessor import apply_preprocessing
from stratml.execution.config.experiment_config_builder import build_experiment_config
from stratml.execution.pipelines.ml_pipeline import run_ml_pipeline
from stratml.execution.metrics.metrics_engine import compute_metrics
from stratml.execution.result_builder import build_experiment_result
from stratml.execution.schemas import (
    ActionDecision, ArtifactRefs, PreprocessingConfig,
    ResourceUsage, SplitConfig,
)


def _prep(strategy="mean", scaling="standard", encoding="none"):
    return PreprocessingConfig(
        missing_value_strategy=strategy, scaling=scaling,
        encoding=encoding, imbalance_strategy="none", feature_selection="none",
    )


def _action(exp_id, model_name, action_type="switch_model", **hp):
    return ActionDecision(
        experiment_id=exp_id,
        action_type=action_type,
        parameters={"model_name": model_name, **hp},
        preprocessing=_prep(),
        reason="integration test",
        expected_gain=0.0, expected_cost=1.0, confidence=1.0,
    )


def _run_one_iteration(dataset_path, target, model_name, exp_id="int_test"):
    """Helper: run phases 1-8 for one iteration, return ExperimentResult."""
    df, name = load_dataframe(dataset_path)
    dataset  = build_dataset(df, name, target)
    profile  = build_profile(dataset)
    split    = split_dataset(dataset, SplitConfig(method="stratified" if profile.problem_type == "classification" else "random"), profile.problem_type)

    action      = _action(exp_id, model_name, n_estimators=10)
    config      = build_experiment_config(action)
    clean, prep = apply_preprocessing(split, config.preprocessing, profile)
    pipe        = run_ml_pipeline(config, clean)
    metrics     = compute_metrics(clean.y_val, pipe.y_val_pred, pipe.train_curve, pipe.val_curve, profile.problem_type)

    return build_experiment_result(
        config=config, metrics=metrics,
        train_curve=pipe.train_curve, validation_curve=pipe.val_curve,
        runtime=pipe.runtime, resource_usage=ResourceUsage(cpu_time_sec=pipe.runtime),
        artifacts=ArtifactRefs(model_path="x", metrics_file="x", tensorboard_logs="x"),
        preprocessing_applied=prep, iteration=1, dataset_name=profile.dataset_name,
    )


# ── Iris (classification) ─────────────────────────────────────────────────────

class TestIrisPipeline:
    def test_experiment_result_valid(self):
        result = _run_one_iteration("data/iris.csv", "species", "RandomForestClassifier")
        assert result.experiment_id == "int_test"
        assert result.iteration == 1
        assert result.dataset_name == "iris"

    def test_classification_metrics_populated(self):
        result = _run_one_iteration("data/iris.csv", "species", "RandomForestClassifier")
        assert result.metrics.accuracy is not None
        assert result.metrics.f1_score is not None
        assert result.metrics.mse is None  # not a regression run

    def test_accuracy_reasonable(self):
        result = _run_one_iteration("data/iris.csv", "species", "RandomForestClassifier")
        assert result.metrics.accuracy > 0.7  # iris is easy — RF should do well

    def test_train_curve_non_empty(self):
        result = _run_one_iteration("data/iris.csv", "species", "RandomForestClassifier")
        assert len(result.train_curve) >= 1

    def test_runtime_positive(self):
        result = _run_one_iteration("data/iris.csv", "species", "RandomForestClassifier")
        assert result.runtime > 0

    def test_model_type_is_ml(self):
        result = _run_one_iteration("data/iris.csv", "species", "RandomForestClassifier")
        assert result.model_type == "ml"


# ── Housing (regression) ──────────────────────────────────────────────────────

class TestHousingPipeline:
    def test_experiment_result_valid(self):
        result = _run_one_iteration("data/housing.csv", "MedHouseVal", "RandomForestRegressor")
        assert result.dataset_name == "housing"

    def test_regression_metrics_populated(self):
        result = _run_one_iteration("data/housing.csv", "MedHouseVal", "RandomForestRegressor")
        assert result.metrics.mse is not None
        assert result.metrics.rmse is not None
        assert result.metrics.r2 is not None
        assert result.metrics.accuracy is None  # not a classification run

    def test_r2_reasonable(self):
        result = _run_one_iteration("data/housing.csv", "MedHouseVal", "RandomForestRegressor")
        assert result.metrics.r2 > 0.5  # RF on housing should get decent R2


# ── Multi-iteration loop ──────────────────────────────────────────────────────

class TestMultiIteration:
    def test_three_iterations_different_models(self):
        """Simulate 3 iterations with different models, assert all produce valid results."""
        models = ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"]
        results = []
        for i, model in enumerate(models, start=1):
            r = _run_one_iteration("data/iris.csv", "species", model, exp_id=f"iter_{i}")
            results.append(r)

        assert len(results) == 3
        for i, r in enumerate(results, start=1):
            assert r.iteration == 1  # each is a fresh call
            assert r.metrics.accuracy is not None

    def test_action_type_increase_capacity(self):
        """increase_model_capacity action type should produce a valid result."""
        df, name = load_dataframe("data/iris.csv")
        dataset  = build_dataset(df, name, "species")
        profile  = build_profile(dataset)
        split    = split_dataset(dataset, SplitConfig(method="stratified"), "classification")

        action = ActionDecision(
            experiment_id="cap_test",
            action_type="increase_model_capacity",
            parameters={"model_name": "RandomForestClassifier", "n_estimators": 50},
            preprocessing=_prep(),
            reason="test", expected_gain=0.05, expected_cost=2.0, confidence=0.8,
        )
        config      = build_experiment_config(action)
        clean, prep = apply_preprocessing(split, config.preprocessing, profile)
        pipe        = run_ml_pipeline(config, clean)
        metrics     = compute_metrics(clean.y_val, pipe.y_val_pred, pipe.train_curve, pipe.val_curve, "classification")

        assert metrics.accuracy is not None
        # n_estimators=50 scaled by default scale=1.5 → 75
        assert config.hyperparameters.get("n_estimators") == 75
