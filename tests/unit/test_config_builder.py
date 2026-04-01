"""
test_config_builder.py
----------------------
Unit tests for execution/config/experiment_config_builder.py
"""

import pytest

from stratml.execution.config.experiment_config_builder import build_experiment_config
from stratml.execution.schemas import ActionDecision, PreprocessingConfig


def _action(action_type, params, prep=None):
    return ActionDecision(
        experiment_id="exp_test",
        action_type=action_type,
        parameters=params,
        preprocessing=prep or PreprocessingConfig(
            missing_value_strategy="mean", scaling="none",
            encoding="none", imbalance_strategy="none", feature_selection="none",
        ),
        reason="test",
        expected_gain=0.0,
        expected_cost=1.0,
        confidence=1.0,
    )


class TestActionTypeMapping:
    def test_switch_model_sets_model_name(self):
        config = build_experiment_config(_action("switch_model", {"model_name": "Ridge"}))
        assert config.model_name == "Ridge"

    def test_switch_model_passes_hyperparams(self):
        config = build_experiment_config(_action("switch_model", {"model_name": "Ridge", "alpha": 0.5}))
        assert config.hyperparameters.get("alpha") == 0.5

    def test_increase_capacity(self):
        config = build_experiment_config(_action("increase_model_capacity", {"model_name": "RandomForestClassifier", "n_estimators": 200}))
        assert config.model_name == "RandomForestClassifier"
        assert config.hyperparameters.get("n_estimators") == 200

    def test_decrease_capacity(self):
        config = build_experiment_config(_action("decrease_model_capacity", {"model_name": "RandomForestClassifier", "n_estimators": 10}))
        assert config.model_name == "RandomForestClassifier"

    def test_modify_regularization(self):
        config = build_experiment_config(_action("modify_regularization", {"model_name": "LogisticRegression", "C": 0.1}))
        assert config.hyperparameters.get("C") == 0.1

    def test_change_optimizer(self):
        config = build_experiment_config(_action("change_optimizer", {"model_name": "MLP", "learning_rate": 0.01}))
        assert config.hyperparameters.get("learning_rate") == 0.01

    def test_apply_preprocessing_keeps_model(self):
        config = build_experiment_config(_action("apply_preprocessing", {"model_name": "SVC"}))
        assert config.model_name == "SVC"

    def test_early_stop_sets_flag(self):
        config = build_experiment_config(_action("early_stop", {"model_name": "MLP", "early_stopping_patience": 7}))
        assert config.early_stopping is True
        assert config.early_stopping_patience == 7

    def test_terminate_raises(self):
        with pytest.raises(ValueError, match="terminate"):
            build_experiment_config(_action("terminate", {}))


class TestModelTypeInference:
    def test_mlp_inferred_as_dl(self):
        config = build_experiment_config(_action("switch_model", {"model_name": "MLP"}))
        assert config.model_type == "dl"

    def test_pytorch_mlp_inferred_as_dl(self):
        config = build_experiment_config(_action("switch_model", {"model_name": "PyTorchMLP"}))
        assert config.model_type == "dl"

    def test_sklearn_model_inferred_as_ml(self):
        config = build_experiment_config(_action("switch_model", {"model_name": "RandomForestClassifier"}))
        assert config.model_type == "ml"

    def test_ridge_inferred_as_ml(self):
        config = build_experiment_config(_action("switch_model", {"model_name": "Ridge"}))
        assert config.model_type == "ml"


class TestPreprocessingPassthrough:
    def test_preprocessing_copied_to_config(self):
        prep = PreprocessingConfig(
            missing_value_strategy="median", scaling="minmax",
            encoding="onehot", imbalance_strategy="oversample", feature_selection="variance_threshold",
        )
        config = build_experiment_config(_action("switch_model", {"model_name": "SVC"}, prep=prep))
        assert config.preprocessing.scaling == "minmax"
        assert config.preprocessing.encoding == "onehot"

    def test_experiment_id_preserved(self):
        config = build_experiment_config(_action("switch_model", {"model_name": "SVC"}))
        assert config.experiment_id == "exp_test"
