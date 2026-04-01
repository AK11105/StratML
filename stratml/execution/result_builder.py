"""
result_builder.py
-----------------
Phase 8 — Assemble ExperimentResult from all phase outputs.
"""

from __future__ import annotations

from stratml.execution.schemas import (
    ExperimentConfig, ExperimentMetrics, ResourceUsage,
    ArtifactRefs, PreprocessingConfig, ExperimentResult,
)


def build_experiment_result(
    config: ExperimentConfig,
    metrics: ExperimentMetrics,
    train_curve: list[float],
    validation_curve: list[float],
    runtime: float,
    resource_usage: ResourceUsage,
    artifacts: ArtifactRefs,
    preprocessing_applied: PreprocessingConfig,
    iteration: int,
    dataset_name: str,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=config.experiment_id,
        iteration=iteration,
        dataset_name=dataset_name,
        model_name=config.model_name,
        model_type=config.model_type,
        hyperparameters=config.hyperparameters,
        preprocessing_applied=preprocessing_applied,
        metrics=metrics,
        train_curve=train_curve,
        validation_curve=validation_curve,
        runtime=runtime,
        resource_usage=resource_usage,
        artifacts=artifacts,
    )
