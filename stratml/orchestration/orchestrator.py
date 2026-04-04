"""
orchestrator.py
---------------
Phase 9 — Main execution loop. Wires all components together.
Budget enforcement lives here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from stratml.execution.data.loader import load_dataframe
from stratml.execution.data.validator import build_dataset
from stratml.execution.data.profiler import build_profile
from stratml.execution.preprocessing.splitter import split_dataset
from stratml.execution.preprocessing.preprocessor import apply_preprocessing
from stratml.execution.config.experiment_config_builder import build_experiment_config
from stratml.execution.metrics.metrics_engine import compute_metrics
from stratml.execution.artifacts.artifact_manager import save_artifacts
from stratml.execution.result_builder import build_experiment_result
from stratml.execution.schemas import (
    ActionDecision, ExperimentResult, DataProfile,
    SplitConfig, ResourceUsage,
)


# Type alias for the two Team B interface callables
SendProfileFn = Callable[[DataProfile], ActionDecision]
SendResultFn  = Callable[[ExperimentResult], ActionDecision]


class ExecutionOrchestrator:
    """
    Drives the full experiment loop.

    Usage:
        orchestrator = ExecutionOrchestrator(
            send_profile=team_b.receive_profile,
            send_result=team_b.receive_result,
        )
        orchestrator.run("data/iris.csv", "species")
    """

    def __init__(
        self,
        send_profile: SendProfileFn,
        send_result: SendResultFn,
        split_config: SplitConfig | None = None,
        time_budget: float | None = None,
        run_id: str = "run",
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.send_profile = send_profile
        self.send_result  = send_result
        self.split_config = split_config or SplitConfig(method="stratified")
        self.time_budget  = time_budget
        self.run_id       = run_id
        self.log          = log or (lambda msg: None)

    def run(self, dataset_path: str, target_column: str) -> None:
        # ── Phase 1+2: Ingest and profile ────────────────────────────────────
        self.log("  Loading dataset...")
        df, name = load_dataframe(dataset_path)
        dataset  = build_dataset(df, name, target_column)
        profile  = build_profile(dataset)
        self.log(f"  Profiled: {profile.rows} rows x {profile.columns} cols | {profile.problem_type}")

        # ── Phase 3: Split once, reuse across all iterations ─────────────────
        split_cfg = SplitConfig(
            method=self.split_config.method
            if profile.problem_type == "classification"
            else "random",
            test_size=self.split_config.test_size,
            val_size=self.split_config.val_size,
            random_seed=self.split_config.random_seed,
        )
        base_split = split_dataset(dataset, split_cfg, profile.problem_type)
        self.log(f"  Split: train={len(base_split.X_train)} | val={len(base_split.X_val)} | test={len(base_split.X_test)}")

        # ── Send DataProfile to Team B, receive first ActionDecision ─────────
        self.log("  Sending profile to Decision Engine...")
        action: ActionDecision = self.send_profile(profile)
        self.log(f"  Decision [iter 0]: action={action.action_type} | params={action.parameters} | trigger={action.reason.trigger}")

        iteration     = 0
        total_runtime = 0.0
        current_model = action.parameters.get("model_name", "LogisticRegression")

        while action.action_type != "terminate":
            iteration += 1
            self.log(f"\n  --- Iteration {iteration} ---")
            if "model_name" not in action.parameters:
                action.parameters["model_name"] = current_model
            current_model = action.parameters.get("model_name", current_model)
            self.log(f"  Training : {current_model} ({action.action_type}) ...")

            # ── Phase 4: Translate ActionDecision → ExperimentConfig ─────────
            config = build_experiment_config(action)

            # ── Phase 4b: Apply preprocessing ────────────────────────────────
            clean_split, applied_preprocessing = apply_preprocessing(
                base_split, config.preprocessing, profile
            )

            # ── Phase 5: Train ────────────────────────────────────────────────
            t_start = time.perf_counter()
            if config.model_type == "ml":
                from stratml.execution.pipelines.ml_pipeline import run_ml_pipeline
                pipeline_result = run_ml_pipeline(config, clean_split)
            else:
                from stratml.execution.pipelines.dl_pipeline import run_dl_pipeline
                pipeline_result = run_dl_pipeline(config, clean_split)
            run_time = round(time.perf_counter() - t_start, 4)
            total_runtime += run_time
            self.log(f"  Trained in {run_time:.2f}s")

            # ── Phase 6: Metrics ──────────────────────────────────────────────
            metrics = compute_metrics(
                y_true=clean_split.y_val,
                y_pred=pipeline_result.y_val_pred,
                train_curve=pipeline_result.train_curve,
                val_curve=pipeline_result.val_curve,
                problem_type=profile.problem_type,
            )

            # ── Phase 7: Artifacts ────────────────────────────────────────────
            tb_dir = None
            if config.model_type == "dl":
                tb_dir = f"outputs/runs/{config.experiment_id}"

            artifacts = save_artifacts(
                experiment_id=config.experiment_id,
                model=pipeline_result.model,
                metrics=metrics,
                config=config,
                tensorboard_log_dir=tb_dir,
                artifacts_root=Path("outputs") / self.run_id / "artifacts",
            )

            # ── Phase 8: Assemble ExperimentResult ───────────────────────────
            result = build_experiment_result(
                config=config,
                metrics=metrics,
                train_curve=pipeline_result.train_curve,
                validation_curve=pipeline_result.val_curve,
                runtime=run_time,
                resource_usage=ResourceUsage(cpu_time_sec=run_time),
                artifacts=artifacts,
                preprocessing_applied=applied_preprocessing,
                iteration=iteration,
                dataset_name=profile.dataset_name,
            )

            # ── Budget check ──────────────────────────────────────────────────
            if self.time_budget and total_runtime >= self.time_budget:
                break

            # ── Send result to Team B, receive next ActionDecision ────────────
            primary = metrics.accuracy if metrics.accuracy is not None else (metrics.r2 or 0.0)
            self.log(f"  Result   : primary={primary:.4f} | runtime={run_time:.2f}s")
            self.log("  Evaluating signals & deciding next action...")
            action = self.send_result(result)
            self.log(f"  Decision : {action.action_type} | trigger={action.reason.trigger} | confidence={action.confidence:.2f} | next={action.parameters}")
