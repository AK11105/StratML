"""
test_result_builder.py
----------------------
Unit tests for execution/result_builder.py
"""

import pytest

from stratml.execution.result_builder import build_experiment_result
from stratml.execution.schemas import (
    ArtifactRefs, ExperimentMetrics, ResourceUsage,
)


def _build(config, iteration=1, dataset_name="test_ds"):
    return build_experiment_result(
        config=config,
        metrics=ExperimentMetrics(accuracy=0.9, f1_score=0.89, train_loss=0.2, validation_loss=0.25),
        train_curve=[0.5, 0.3, 0.2],
        validation_curve=[0.6, 0.4, 0.25],
        runtime=1.23,
        resource_usage=ResourceUsage(cpu_time_sec=1.23),
        artifacts=ArtifactRefs(model_path="m.pkl", metrics_file="m.json", tensorboard_logs="tb/"),
        preprocessing_applied=config.preprocessing,
        iteration=iteration,
        dataset_name=dataset_name,
    )


class TestResultBuilder:
    def test_experiment_id_matches_config(self, base_config):
        result = _build(base_config)
        assert result.experiment_id == base_config.experiment_id

    def test_model_name_matches_config(self, base_config):
        result = _build(base_config)
        assert result.model_name == base_config.model_name

    def test_model_type_matches_config(self, base_config):
        result = _build(base_config)
        assert result.model_type == base_config.model_type

    def test_iteration_set_correctly(self, base_config):
        result = _build(base_config, iteration=3)
        assert result.iteration == 3

    def test_dataset_name_set_correctly(self, base_config):
        result = _build(base_config, dataset_name="iris")
        assert result.dataset_name == "iris"

    def test_metrics_preserved(self, base_config):
        result = _build(base_config)
        assert result.metrics.accuracy == 0.9
        assert result.metrics.validation_loss == 0.25

    def test_curves_preserved(self, base_config):
        result = _build(base_config)
        assert result.train_curve == [0.5, 0.3, 0.2]
        assert result.validation_curve == [0.6, 0.4, 0.25]

    def test_runtime_preserved(self, base_config):
        result = _build(base_config)
        assert result.runtime == 1.23

    def test_pydantic_validation_passes(self, base_config):
        # If Pydantic validation fails it raises — this just confirms it doesn't
        result = _build(base_config)
        assert result.model_dump() is not None
