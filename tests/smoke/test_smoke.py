"""
test_smoke.py
-------------
Smoke tests — "does it run without crashing?"

These are fast sanity checks run after every commit.
No detailed assertions — just confirm nothing raises.
"""

import pytest


class TestImports:
    def test_all_execution_modules_importable(self):
        from stratml.execution.data.loader import load_dataframe
        from stratml.execution.data.validator import build_dataset
        from stratml.execution.data.profiler import build_profile
        from stratml.execution.preprocessing.splitter import split_dataset
        from stratml.execution.preprocessing.preprocessor import apply_preprocessing
        from stratml.execution.config.experiment_config_builder import build_experiment_config
        from stratml.execution.pipelines.ml_pipeline import run_ml_pipeline, MODEL_REGISTRY
        from stratml.execution.pipelines.dl_pipeline import run_dl_pipeline
        from stratml.execution.metrics.metrics_engine import compute_metrics
        from stratml.execution.artifacts.artifact_manager import save_artifacts
        from stratml.execution.result_builder import build_experiment_result
        from stratml.orchestration.orchestrator import ExecutionOrchestrator

    def test_schemas_importable(self):
        from stratml.execution.schemas import (
            Dataset, DataProfile, DataSplit, SplitConfig,
            ExperimentConfig, ExperimentResult, ExperimentMetrics,
            PreprocessingConfig, ActionDecision, ResourceUsage, ArtifactRefs,
        )

    def test_core_schemas_importable(self):
        from stratml.core.schemas import StateObject, ActionDecision, ExperimentResult


class TestLoaderSmoke:
    def test_csv_loads(self):
        from stratml.execution.data.loader import load_dataframe
        df, name = load_dataframe("data/iris.csv")
        assert len(df) > 0
        assert name == "iris"

    def test_housing_csv_loads(self):
        from stratml.execution.data.loader import load_dataframe
        df, name = load_dataframe("data/housing.csv")
        assert len(df) > 0

    def test_missing_file_raises(self):
        from stratml.execution.data.loader import load_dataframe
        with pytest.raises(FileNotFoundError):
            load_dataframe("data/does_not_exist.csv")

    def test_unsupported_format_raises(self, tmp_path):
        from stratml.execution.data.loader import load_dataframe
        f = tmp_path / "model.pkl"
        f.write_bytes(b"fake")
        with pytest.raises(ValueError):
            load_dataframe(str(f))


class TestProfilerSmoke:
    def test_iris_profile_runs(self):
        from stratml.execution.data.loader import load_dataframe
        from stratml.execution.data.validator import build_dataset
        from stratml.execution.data.profiler import build_profile
        df, name = load_dataframe("data/iris.csv")
        dataset = build_dataset(df, name, "species")
        profile = build_profile(dataset)
        assert profile.problem_type == "classification"
        assert profile.rows == 150

    def test_housing_profile_runs(self):
        from stratml.execution.data.loader import load_dataframe
        from stratml.execution.data.validator import build_dataset
        from stratml.execution.data.profiler import build_profile
        df, name = load_dataframe("data/housing.csv")
        dataset = build_dataset(df, name, "MedHouseVal")
        profile = build_profile(dataset)
        assert profile.problem_type == "regression"


class TestSplitterSmoke:
    def test_iris_split_runs(self):
        from stratml.execution.data.loader import load_dataframe
        from stratml.execution.data.validator import build_dataset
        from stratml.execution.data.profiler import build_profile
        from stratml.execution.preprocessing.splitter import split_dataset
        from stratml.execution.schemas import SplitConfig
        df, name = load_dataframe("data/iris.csv")
        dataset = build_dataset(df, name, "species")
        profile = build_profile(dataset)
        split = split_dataset(dataset, SplitConfig(method="stratified"), profile.problem_type)
        assert len(split.X_train) > 0
        assert len(split.X_val) > 0
        assert len(split.X_test) > 0


class TestMLPipelineSmoke:
    def test_random_forest_on_iris(self):
        from stratml.execution.data.loader import load_dataframe
        from stratml.execution.data.validator import build_dataset
        from stratml.execution.data.profiler import build_profile
        from stratml.execution.preprocessing.splitter import split_dataset
        from stratml.execution.preprocessing.preprocessor import apply_preprocessing
        from stratml.execution.pipelines.ml_pipeline import run_ml_pipeline
        from stratml.execution.schemas import ExperimentConfig, PreprocessingConfig, SplitConfig

        df, name = load_dataframe("data/iris.csv")
        dataset  = build_dataset(df, name, "species")
        profile  = build_profile(dataset)
        split    = split_dataset(dataset, SplitConfig(method="stratified"), "classification")
        prep     = PreprocessingConfig(missing_value_strategy="mean", scaling="standard",
                                       encoding="none", imbalance_strategy="none", feature_selection="none")
        config   = ExperimentConfig(experiment_id="smoke", model_name="RandomForestClassifier",
                                    model_type="ml", hyperparameters={"n_estimators": 5}, preprocessing=prep)
        clean, _ = apply_preprocessing(split, prep, profile)
        result   = run_ml_pipeline(config, clean)
        assert result.y_val_pred is not None

    def test_all_registry_models_instantiate(self):
        from stratml.execution.pipelines.ml_pipeline import MODEL_REGISTRY
        for name, cls in MODEL_REGISTRY.items():
            instance = cls()
            assert instance is not None


class TestDLPipelineSmoke:
    def test_mlp_classification_runs(self, clf_split):
        from stratml.execution.pipelines.dl_pipeline import run_dl_pipeline
        from stratml.execution.schemas import ExperimentConfig, PreprocessingConfig
        prep   = PreprocessingConfig(missing_value_strategy="mean", scaling="none",
                                     encoding="none", imbalance_strategy="none", feature_selection="none")
        config = ExperimentConfig(
            experiment_id="smoke_dl", model_name="MLP", model_type="dl",
            hyperparameters={"architecture": "MLP", "task": "classification",
                             "hidden_units": 16, "layers": 1, "epochs": 2},
            preprocessing=prep,
        )
        result = run_dl_pipeline(config, clf_split)
        assert result.y_val_pred is not None
        assert len(result.train_curve) == 2

    def test_mlp_regression_runs(self, reg_split):
        from stratml.execution.pipelines.dl_pipeline import run_dl_pipeline
        from stratml.execution.schemas import ExperimentConfig, PreprocessingConfig
        prep   = PreprocessingConfig(missing_value_strategy="mean", scaling="none",
                                     encoding="none", imbalance_strategy="none", feature_selection="none")
        config = ExperimentConfig(
            experiment_id="smoke_dl_reg", model_name="MLP", model_type="dl",
            hyperparameters={"architecture": "MLP", "task": "regression",
                             "hidden_units": 16, "layers": 1, "epochs": 2},
            preprocessing=prep,
        )
        result = run_dl_pipeline(config, reg_split)
        assert result.y_val_pred is not None

    def test_cnn1d_runs(self, clf_split):
        from stratml.execution.pipelines.dl_pipeline import run_dl_pipeline
        from stratml.execution.schemas import ExperimentConfig, PreprocessingConfig
        prep   = PreprocessingConfig(missing_value_strategy="mean", scaling="none",
                                     encoding="none", imbalance_strategy="none", feature_selection="none")
        config = ExperimentConfig(
            experiment_id="smoke_cnn", model_name="CNN1D", model_type="dl",
            hyperparameters={"architecture": "CNN1D", "task": "classification",
                             "hidden_units": 16, "epochs": 2},
            preprocessing=prep,
        )
        result = run_dl_pipeline(config, clf_split)
        assert result.y_val_pred is not None

    def test_rnn_runs(self, clf_split):
        from stratml.execution.pipelines.dl_pipeline import run_dl_pipeline
        from stratml.execution.schemas import ExperimentConfig, PreprocessingConfig
        prep   = PreprocessingConfig(missing_value_strategy="mean", scaling="none",
                                     encoding="none", imbalance_strategy="none", feature_selection="none")
        config = ExperimentConfig(
            experiment_id="smoke_rnn", model_name="RNN", model_type="dl",
            hyperparameters={"architecture": "RNN", "task": "classification",
                             "hidden_units": 16, "layers": 1, "epochs": 2},
            preprocessing=prep,
        )
        result = run_dl_pipeline(config, clf_split)
        assert result.y_val_pred is not None
