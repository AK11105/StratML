"""
test_ml_pipeline.py
-------------------
Unit tests for execution/pipelines/ml_pipeline.py
"""

import numpy as np
import pytest

from stratml.execution.pipelines.ml_pipeline import run_ml_pipeline, MODEL_REGISTRY
from stratml.execution.schemas import ExperimentConfig, PreprocessingConfig


def _config(model_name, **hp):
    return ExperimentConfig(
        experiment_id="t",
        model_name=model_name,
        model_type="ml",
        hyperparameters=hp,
        preprocessing=PreprocessingConfig(
            missing_value_strategy="mean", scaling="none",
            encoding="none", imbalance_strategy="none", feature_selection="none",
        ),
    )


class TestRegistry:
    def test_registry_has_expected_families(self):
        names = list(MODEL_REGISTRY.keys())
        assert "LogisticRegression" in names
        assert "RandomForestClassifier" in names
        assert "GradientBoostingClassifier" in names
        assert "KNeighborsClassifier" in names
        assert "GaussianNB" in names
        assert "SVC" in names

    def test_registry_has_regression_models(self):
        names = list(MODEL_REGISTRY.keys())
        assert "Ridge" in names
        assert "RandomForestRegressor" in names
        assert "SVR" in names

    def test_unknown_model_raises(self, clf_split):
        with pytest.raises(ValueError, match="Unknown model"):
            run_ml_pipeline(_config("NonExistentModel"), clf_split)


class TestPipelineOutput:
    def test_returns_correct_pred_length(self, clf_split):
        result = run_ml_pipeline(_config("RandomForestClassifier", n_estimators=5), clf_split)
        assert len(result.y_val_pred) == len(clf_split.y_val)

    def test_runtime_positive(self, clf_split):
        result = run_ml_pipeline(_config("RandomForestClassifier", n_estimators=5), clf_split)
        assert result.runtime > 0

    def test_train_curve_single_element(self, clf_split):
        result = run_ml_pipeline(_config("RandomForestClassifier", n_estimators=5), clf_split)
        assert len(result.train_curve) == 1

    def test_val_curve_single_element(self, clf_split):
        result = run_ml_pipeline(_config("RandomForestClassifier", n_estimators=5), clf_split)
        assert len(result.val_curve) == 1

    def test_model_object_returned(self, clf_split):
        result = run_ml_pipeline(_config("RandomForestClassifier", n_estimators=5), clf_split)
        assert result.model is not None

    def test_invalid_hyperparams_silently_dropped(self, clf_split):
        # "fake_param" should be ignored, not crash
        result = run_ml_pipeline(_config("RandomForestClassifier", n_estimators=5, fake_param=999), clf_split)
        assert result.y_val_pred is not None


class TestRegressionModel:
    def test_regression_model_runs(self, reg_split):
        result = run_ml_pipeline(_config("RandomForestRegressor", n_estimators=5), reg_split)
        assert len(result.y_val_pred) == len(reg_split.y_val)

    def test_ridge_regression_runs(self, reg_split):
        result = run_ml_pipeline(_config("Ridge", alpha=1.0), reg_split)
        assert result.runtime > 0
