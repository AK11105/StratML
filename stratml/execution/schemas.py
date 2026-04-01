"""
schemas.py
----------
Pydantic models for all data structures used by the Execution team.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd
from pydantic import BaseModel, Field


class Dataset(BaseModel):
    """Internal representation of a loaded dataset. Not sent to Team B."""

    dataset_name: str
    rows: int
    columns: int
    target_column: str
    dataset_type: str = Field(..., pattern="^(tabular|text|vision)$")
    raw_dataframe: object = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}


class FeatureInfo(BaseModel):
    name: str
    dtype: str
    unique_values: int
    missing_percentage: float
    distribution: str = Field(..., pattern="^(normal|skewed|uniform|unknown)$")


class DataProfile(BaseModel):
    dataset_name: str
    dataset_type: str = Field(..., pattern="^(tabular|text|vision)$")
    rows: int
    columns: int
    target_column: str
    problem_type: str = Field(..., pattern="^(classification|regression)$")
    numerical_columns: list[str]
    categorical_columns: list[str]
    missing_value_ratio: float
    class_distribution: dict[str, int] = Field(default_factory=dict)
    feature_summary: list[FeatureInfo]
    recommended_metrics: list[str]


class PreprocessingConfig(BaseModel):
    missing_value_strategy: str = Field(..., pattern="^(mean|median|mode|drop)$")
    scaling: str = Field(..., pattern="^(standard|minmax|none)$")
    encoding: str = Field(..., pattern="^(onehot|label|none)$")
    imbalance_strategy: str = Field(..., pattern="^(oversample|undersample|none)$")
    feature_selection: str = Field(..., pattern="^(variance_threshold|none)$")


class ExperimentMetrics(BaseModel):
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    train_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None


class ResourceUsage(BaseModel):
    gpu_used: bool = False
    gpu_memory_gb: float = 0.0
    cpu_time_sec: float = 0.0


class ArtifactRefs(BaseModel):
    model_path: str
    metrics_file: str
    tensorboard_logs: str


class ExperimentResult(BaseModel):
    experiment_id: str
    iteration: int
    dataset_name: str
    model_name: str
    model_type: str = Field(..., pattern="^(ml|dl)$")
    hyperparameters: dict
    preprocessing_applied: PreprocessingConfig
    metrics: ExperimentMetrics
    train_curve: list[float]
    validation_curve: list[float]
    runtime: float
    resource_usage: ResourceUsage
    artifacts: ArtifactRefs


class ActionDecision(BaseModel):
    experiment_id: str
    action_type: str
    parameters: dict
    preprocessing: PreprocessingConfig
    reason: str
    expected_gain: float
    expected_cost: float
    confidence: float = Field(..., ge=0.0, le=1.0)


class SplitConfig(BaseModel):
    method: str = Field(..., pattern="^(stratified|random)$")
    test_size: float = 0.2
    val_size: float = 0.1
    random_seed: int = 42


class ExperimentConfig(BaseModel):
    experiment_id: str
    model_name: str
    model_type: str = Field(..., pattern="^(ml|dl)$")
    hyperparameters: dict
    preprocessing: PreprocessingConfig
    early_stopping: bool = False
    early_stopping_patience: int = 5


@dataclass
class DataSplit:
    """Internal train/val/test split. Never crosses the Team A/B boundary."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
