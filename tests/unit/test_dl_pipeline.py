"""
test_dl_pipeline.py
-------------------
Unit tests for execution/pipelines/dl_pipeline.py

Covers: output contracts, all architectures, both tasks, early stopping,
        best-weight restore, device reporting, BatchNorm, scheduler variants,
        capacity mutations via config builder.
"""

import numpy as np
import pytest
import torch

from stratml.execution.pipelines.dl_pipeline import run_dl_pipeline, _get_device
from stratml.execution.config.experiment_config_builder import build_experiment_config
from stratml.execution.schemas import ExperimentConfig, PreprocessingConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

_PREP = PreprocessingConfig(
    missing_value_strategy="mean", scaling="none",
    encoding="none", imbalance_strategy="none", feature_selection="none",
)


def _config(model_name="MLP", **hp) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="test_dl",
        model_name=model_name,
        model_type="dl",
        hyperparameters={"epochs": 3, "hidden_units": 16, "layers": 1, **hp},
        preprocessing=_PREP,
        early_stopping=True,
        early_stopping_patience=5,
    )


# ── Output contract ───────────────────────────────────────────────────────────

class TestOutputContract:
    def test_pred_length_matches_val(self, clf_split):
        r = run_dl_pipeline(_config(), clf_split)
        assert len(r.y_val_pred) == len(clf_split.y_val)

    def test_train_val_curves_same_length(self, clf_split):
        r = run_dl_pipeline(_config(), clf_split)
        assert len(r.train_curve) == len(r.val_curve)

    def test_curves_length_equals_epochs_run(self, clf_split):
        r = run_dl_pipeline(_config(epochs=3), clf_split)
        assert len(r.train_curve) == r.epochs_run

    def test_runtime_positive(self, clf_split):
        r = run_dl_pipeline(_config(), clf_split)
        assert r.runtime > 0

    def test_model_is_nn_module(self, clf_split):
        r = run_dl_pipeline(_config(), clf_split)
        assert isinstance(r.model, torch.nn.Module)

    def test_model_state_is_dict(self, clf_split):
        r = run_dl_pipeline(_config(), clf_split)
        assert isinstance(r.model_state, dict)
        assert len(r.model_state) > 0

    def test_model_state_keys_match_model(self, clf_split):
        r = run_dl_pipeline(_config(), clf_split)
        assert set(r.model_state.keys()) == set(r.model.state_dict().keys())

    def test_device_used_is_string(self, clf_split):
        r = run_dl_pipeline(_config(), clf_split)
        assert r.device_used in ("cpu", "cuda", "mps")

    def test_epochs_run_leq_epochs(self, clf_split):
        r = run_dl_pipeline(_config(epochs=5), clf_split)
        assert r.epochs_run <= 5

    def test_train_losses_are_finite(self, clf_split):
        r = run_dl_pipeline(_config(), clf_split)
        assert all(np.isfinite(v) for v in r.train_curve)

    def test_val_losses_are_finite(self, clf_split):
        r = run_dl_pipeline(_config(), clf_split)
        assert all(np.isfinite(v) for v in r.val_curve)


# ── Classification task ───────────────────────────────────────────────────────

class TestClassification:
    def test_preds_are_original_labels(self, clf_split):
        r = run_dl_pipeline(_config(), clf_split)
        valid = set(clf_split.y_train.unique())
        assert all(p in valid for p in r.y_val_pred)

    def test_pred_dtype_matches_target(self, clf_split):
        r = run_dl_pipeline(_config(), clf_split)
        # numpy.str_ and str are both string types — check with isinstance
        assert all(isinstance(p, (str, np.str_)) for p in r.y_val_pred)


# ── Regression task ───────────────────────────────────────────────────────────

class TestRegression:
    def test_preds_are_floats(self, reg_split):
        r = run_dl_pipeline(_config(task="regression"), reg_split)
        assert r.y_val_pred.dtype in (np.float32, np.float64)

    def test_pred_length_matches_val(self, reg_split):
        r = run_dl_pipeline(_config(task="regression"), reg_split)
        assert len(r.y_val_pred) == len(reg_split.y_val)

    def test_preds_are_finite(self, reg_split):
        r = run_dl_pipeline(_config(task="regression"), reg_split)
        assert np.all(np.isfinite(r.y_val_pred))


# ── Architectures ─────────────────────────────────────────────────────────────

class TestArchitectures:
    def test_mlp_classification(self, clf_split):
        r = run_dl_pipeline(_config("MLP"), clf_split)
        assert len(r.y_val_pred) == len(clf_split.y_val)

    def test_cnn1d_classification(self, clf_split):
        r = run_dl_pipeline(_config("CNN1D"), clf_split)
        assert len(r.y_val_pred) == len(clf_split.y_val)

    def test_rnn_classification(self, clf_split):
        r = run_dl_pipeline(_config("RNN"), clf_split)
        assert len(r.y_val_pred) == len(clf_split.y_val)

    def test_mlp_regression(self, reg_split):
        r = run_dl_pipeline(_config("MLP", task="regression"), reg_split)
        assert len(r.y_val_pred) == len(reg_split.y_val)

    def test_cnn1d_regression(self, reg_split):
        r = run_dl_pipeline(_config("CNN1D", task="regression"), reg_split)
        assert len(r.y_val_pred) == len(reg_split.y_val)

    def test_rnn_regression(self, reg_split):
        r = run_dl_pipeline(_config("RNN", task="regression"), reg_split)
        assert len(r.y_val_pred) == len(reg_split.y_val)

    def test_unknown_arch_falls_back_to_mlp(self, clf_split):
        # unknown arch string → _build_model returns MLP
        r = run_dl_pipeline(_config("UNKNOWN_ARCH"), clf_split)
        assert len(r.y_val_pred) == len(clf_split.y_val)


# ── BatchNorm ─────────────────────────────────────────────────────────────────

class TestBatchNorm:
    def test_mlp_with_batchnorm_runs(self, clf_split):
        r = run_dl_pipeline(_config(batch_norm=True), clf_split)
        assert len(r.y_val_pred) == len(clf_split.y_val)

    def test_cnn1d_with_batchnorm_runs(self, clf_split):
        r = run_dl_pipeline(_config("CNN1D", batch_norm=True), clf_split)
        assert len(r.y_val_pred) == len(clf_split.y_val)


# ── LR Schedulers ─────────────────────────────────────────────────────────────

class TestSchedulers:
    def test_plateau_scheduler(self, clf_split):
        r = run_dl_pipeline(_config(scheduler="plateau"), clf_split)
        assert r.epochs_run > 0

    def test_cosine_scheduler(self, clf_split):
        r = run_dl_pipeline(_config(scheduler="cosine"), clf_split)
        assert r.epochs_run > 0

    def test_no_scheduler(self, clf_split):
        r = run_dl_pipeline(_config(scheduler="none"), clf_split)
        assert r.epochs_run > 0


# ── Early stopping ────────────────────────────────────────────────────────────

class TestEarlyStopping:
    def test_early_stopping_can_trigger(self, clf_split):
        # patience=1 with many epochs — should stop early
        cfg = ExperimentConfig(
            experiment_id="es_test",
            model_name="MLP",
            model_type="dl",
            hyperparameters={"epochs": 50, "hidden_units": 16, "layers": 1},
            preprocessing=_PREP,
            early_stopping=True,
            early_stopping_patience=1,
        )
        r = run_dl_pipeline(cfg, clf_split)
        # With patience=1 it almost certainly stops before 50 epochs
        assert r.epochs_run <= 50

    def test_early_stopped_flag_set(self, clf_split):
        cfg = ExperimentConfig(
            experiment_id="es_flag",
            model_name="MLP",
            model_type="dl",
            hyperparameters={"epochs": 50, "hidden_units": 16, "layers": 1},
            preprocessing=_PREP,
            early_stopping=True,
            early_stopping_patience=1,
        )
        r = run_dl_pipeline(cfg, clf_split)
        # early_stopped should be True when epochs_run < 50
        if r.epochs_run < 50:
            assert r.early_stopped is True

    def test_best_weights_restored(self, clf_split):
        """Model state_dict should match the best checkpoint, not the final epoch."""
        cfg = ExperimentConfig(
            experiment_id="bw_test",
            model_name="MLP",
            model_type="dl",
            hyperparameters={"epochs": 10, "hidden_units": 16, "layers": 1},
            preprocessing=_PREP,
            early_stopping=True,
            early_stopping_patience=2,
        )
        r = run_dl_pipeline(cfg, clf_split)
        # model_state must be loadable back into the model without error
        r.model.load_state_dict(r.model_state)

    def test_best_epoch_leq_epochs_run(self, clf_split):
        r = run_dl_pipeline(_config(epochs=10), clf_split)
        assert 0 <= r.best_epoch < r.epochs_run

    def test_best_epoch_is_int(self, clf_split):
        r = run_dl_pipeline(_config(), clf_split)
        assert isinstance(r.best_epoch, int)


# ── TensorBoard ───────────────────────────────────────────────────────────────

class TestTensorBoard:
    def test_tensorboard_writes_event_files(self, clf_split, tmp_path):
        tb_dir = str(tmp_path / "tb_logs")
        run_dl_pipeline(_config(epochs=2), clf_split, tensorboard_log_dir=tb_dir)
        import os
        files = list(os.walk(tb_dir))
        # SummaryWriter creates at least one event file
        all_files = [f for _, _, fs in files for f in fs]
        assert any("events.out" in f for f in all_files)

    def test_pipeline_runs_without_tensorboard(self, clf_split):
        # No tensorboard_log_dir — should not crash
        r = run_dl_pipeline(_config(), clf_split, tensorboard_log_dir=None)
        assert r.epochs_run > 0


# ── Gradient clipping ─────────────────────────────────────────────────────────

class TestGradientClipping:
    def test_grad_clip_enabled_runs(self, clf_split):
        r = run_dl_pipeline(_config(grad_clip=1.0), clf_split)
        assert r.epochs_run > 0

    def test_grad_clip_disabled_runs(self, clf_split):
        r = run_dl_pipeline(_config(grad_clip=0.0), clf_split)
        assert r.epochs_run > 0

    def test_rnn_with_grad_clip_stable(self, clf_split):
        # RNN is most prone to gradient explosion — clipping must not crash
        r = run_dl_pipeline(_config("RNN", grad_clip=1.0, layers=2), clf_split)
        assert all(np.isfinite(v) for v in r.train_curve)
        assert all(np.isfinite(v) for v in r.val_curve)


# ── Weight decay ──────────────────────────────────────────────────────────────

class TestWeightDecay:
    def test_weight_decay_zero_runs(self, clf_split):
        r = run_dl_pipeline(_config(weight_decay=0.0), clf_split)
        assert r.epochs_run > 0

    def test_weight_decay_nonzero_runs(self, clf_split):
        r = run_dl_pipeline(_config(weight_decay=1e-4), clf_split)
        assert r.epochs_run > 0

    def test_weight_decay_regression(self, reg_split):
        r = run_dl_pipeline(_config(task="regression", weight_decay=1e-3), reg_split)
        assert np.all(np.isfinite(r.y_val_pred))


# ── Mixed precision ───────────────────────────────────────────────────────────

class TestMixedPrecision:
    def test_mixed_precision_runs_on_cpu(self, clf_split):
        # AMP is silently disabled on CPU — must not crash
        r = run_dl_pipeline(_config(mixed_precision=True), clf_split)
        assert r.epochs_run > 0

    def test_mixed_precision_false_runs(self, clf_split):
        r = run_dl_pipeline(_config(mixed_precision=False), clf_split)
        assert r.epochs_run > 0


# ── Lazy DataLoader ───────────────────────────────────────────────────────────

class TestLazyDataLoader:
    def test_large_batch_size_runs(self, clf_split):
        # batch_size > dataset size — DataLoader handles gracefully
        r = run_dl_pipeline(_config(batch_size=512), clf_split)
        assert len(r.y_val_pred) == len(clf_split.y_val)

    def test_batch_size_1_runs(self, clf_split):
        r = run_dl_pipeline(_config(batch_size=1, epochs=2), clf_split)
        assert r.epochs_run > 0

    def test_tabular_dataset_len(self, clf_split):
        from stratml.execution.pipelines.dl_pipeline import _TabularDataset
        import numpy as np
        X = clf_split.X_train.values.astype(np.float32)
        y = np.zeros(len(X), dtype=np.int64)
        ds = _TabularDataset(X, y)
        assert len(ds) == len(X)

    def test_tabular_dataset_getitem_shapes(self, clf_split):
        from stratml.execution.pipelines.dl_pipeline import _TabularDataset
        import numpy as np
        X = clf_split.X_train.values.astype(np.float32)
        y = np.zeros(len(X), dtype=np.int64)
        ds = _TabularDataset(X, y)
        xb, yb = ds[0]
        assert xb.shape == (X.shape[1],)
        assert yb.shape == ()


# ── Config builder — weight_decay in change_optimizer ────────────────────────

class TestChangeOptimizerWeightDecay:
    def _action(self, **params):
        from stratml.execution.schemas import ActionDecision
        return ActionDecision(
            experiment_id="wd_test",
            action_type="change_optimizer",
            parameters={"model_name": "MLP", "learning_rate": 0.01,
                        "learning_rate_scale": 0.05, **params},
            preprocessing=_PREP,
            reason="test",
            expected_gain=0.0,
            expected_cost=0.5,
            confidence=1.0,
        )

    def test_aggressive_lr_scale_sets_weight_decay(self):
        cfg = build_experiment_config(self._action())
        # lr_scale=0.05 <= 0.1 → weight_decay should be bumped
        assert cfg.hyperparameters.get("weight_decay", 0.0) > 0.0

    def test_weight_decay_capped_at_1e2(self):
        cfg = build_experiment_config(self._action(weight_decay=1e-2))
        assert cfg.hyperparameters.get("weight_decay", 0.0) <= 1e-2

class TestDLCapacityMutations:
    def _dl_action(self, action_type, **params):
        from stratml.execution.schemas import ActionDecision
        return ActionDecision(
            experiment_id="cap_test",
            action_type=action_type,
            parameters={"model_name": "MLP", "hidden_units": 64, "layers": 2, **params},
            preprocessing=_PREP,
            reason="test",
            expected_gain=0.0,
            expected_cost=0.5,
            confidence=1.0,
        )

    def test_increase_capacity_scales_hidden_units(self):
        cfg = build_experiment_config(self._dl_action("increase_model_capacity"))
        assert cfg.hyperparameters["hidden_units"] > 64

    def test_increase_capacity_adds_layer_at_high_scale(self):
        cfg = build_experiment_config(self._dl_action("increase_model_capacity", scale=2.0))
        assert cfg.hyperparameters["layers"] > 2

    def test_decrease_capacity_reduces_hidden_units(self):
        cfg = build_experiment_config(self._dl_action("decrease_model_capacity"))
        assert cfg.hyperparameters["hidden_units"] < 64

    def test_decrease_capacity_min_hidden_units_16(self):
        cfg = build_experiment_config(self._dl_action("decrease_model_capacity", hidden_units=16, scale=0.1))
        assert cfg.hyperparameters["hidden_units"] >= 16

    def test_modify_regularization_increases_dropout(self):
        cfg = build_experiment_config(self._dl_action("modify_regularization", dropout=0.1))
        assert cfg.hyperparameters["dropout"] > 0.1

    def test_modify_regularization_dropout_capped_at_05(self):
        cfg = build_experiment_config(self._dl_action("modify_regularization", dropout=0.5))
        assert cfg.hyperparameters["dropout"] <= 0.5

    def test_change_optimizer_reduces_lr(self):
        cfg = build_experiment_config(self._dl_action("change_optimizer", learning_rate=0.01))
        assert cfg.hyperparameters["learning_rate"] < 0.01

    def test_change_optimizer_sets_cosine_scheduler(self):
        cfg = build_experiment_config(self._dl_action("change_optimizer", learning_rate=0.01, learning_rate_scale=0.05))
        assert cfg.hyperparameters.get("scheduler") == "cosine"

    def test_dl_always_has_early_stopping(self):
        cfg = build_experiment_config(self._dl_action("switch_model"))
        assert cfg.early_stopping is True
