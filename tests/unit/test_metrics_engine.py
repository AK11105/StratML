"""
test_metrics_engine.py
----------------------
Unit tests for execution/metrics/metrics_engine.py
"""

import numpy as np
import pytest

from stratml.execution.metrics.metrics_engine import compute_metrics


class TestClassificationMetrics:
    def setup_method(self):
        self.y_true = np.array(["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"])
        self.y_pred = np.array(["a", "b", "c", "a", "b", "c", "a", "b", "a", "a"])  # 1 wrong
        self.train_curve = [0.5, 0.3, 0.2]
        self.val_curve   = [0.6, 0.4, 0.25]

    def test_accuracy_populated(self):
        m = compute_metrics(self.y_true, self.y_pred, self.train_curve, self.val_curve, "classification")
        assert m.accuracy is not None
        assert 0.0 <= m.accuracy <= 1.0

    def test_f1_populated(self):
        m = compute_metrics(self.y_true, self.y_pred, self.train_curve, self.val_curve, "classification")
        assert m.f1_score is not None

    def test_precision_recall_populated(self):
        m = compute_metrics(self.y_true, self.y_pred, self.train_curve, self.val_curve, "classification")
        assert m.precision is not None
        assert m.recall is not None

    def test_regression_fields_are_none(self):
        m = compute_metrics(self.y_true, self.y_pred, self.train_curve, self.val_curve, "classification")
        assert m.mse is None
        assert m.rmse is None
        assert m.r2 is None

    def test_train_loss_is_last_train_curve_value(self):
        m = compute_metrics(self.y_true, self.y_pred, self.train_curve, self.val_curve, "classification")
        assert m.train_loss == self.train_curve[-1]

    def test_val_loss_is_last_val_curve_value(self):
        m = compute_metrics(self.y_true, self.y_pred, self.train_curve, self.val_curve, "classification")
        assert m.validation_loss == self.val_curve[-1]

    def test_perfect_predictions_accuracy_one(self):
        y = np.array(["a", "b", "c"])
        m = compute_metrics(y, y, [0.0], [0.0], "classification")
        assert m.accuracy == 1.0


class TestRegressionMetrics:
    def setup_method(self):
        np.random.seed(0)
        self.y_true = np.random.randn(50).astype(np.float32)
        self.y_pred = self.y_true + np.random.randn(50) * 0.1
        self.train_curve = [1.0, 0.5]
        self.val_curve   = [1.1, 0.6]

    def test_mse_populated(self):
        m = compute_metrics(self.y_true, self.y_pred, self.train_curve, self.val_curve, "regression")
        assert m.mse is not None
        assert m.mse >= 0.0

    def test_rmse_is_sqrt_mse(self):
        m = compute_metrics(self.y_true, self.y_pred, self.train_curve, self.val_curve, "regression")
        assert abs(m.rmse - m.mse ** 0.5) < 1e-4

    def test_r2_populated(self):
        m = compute_metrics(self.y_true, self.y_pred, self.train_curve, self.val_curve, "regression")
        assert m.r2 is not None

    def test_classification_fields_are_none(self):
        m = compute_metrics(self.y_true, self.y_pred, self.train_curve, self.val_curve, "regression")
        assert m.accuracy is None
        assert m.f1_score is None
        assert m.precision is None
        assert m.recall is None

    def test_perfect_predictions_mse_near_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(y, y, [0.0], [0.0], "regression")
        assert m.mse < 1e-9
