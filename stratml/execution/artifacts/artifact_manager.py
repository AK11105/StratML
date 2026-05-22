"""
artifact_manager.py
-------------------
Phase 7 — Persist model, metrics, and config. Return ArtifactRefs.

DL models are saved as both:
  - model.pkl  (joblib, for compatibility with report_generator / model.py script)
  - model.pth  (torch.save of state_dict — proper PyTorch format for resuming)

ML models are saved as model.pkl only.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from stratml.execution.schemas import ExperimentMetrics, ExperimentConfig, ArtifactRefs

_ARTIFACTS_ROOT = Path("outputs/artifacts")
_RUNS_ROOT      = Path("outputs/runs")


def save_artifacts(
    experiment_id: str,
    model: object,
    metrics: ExperimentMetrics,
    config: ExperimentConfig,
    tensorboard_log_dir: str | None = None,
    artifacts_root: Path | None = None,
    dl_result=None,  # DLPipelineResult | None — passed for .pth saving
    enable_mlflow: bool = False,
) -> ArtifactRefs:
    """Save model + metrics + config to disk. Return ArtifactRefs."""
    root    = artifacts_root or (_ARTIFACTS_ROOT / experiment_id)
    out_dir = root
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_path = str(out_dir / "model.pkl")
    joblib.dump(model, model_path)

    # DL: also save proper PyTorch state_dict as .pth
    if config.model_type == "dl" and dl_result is not None:
        try:
            import torch
            pth_path = out_dir / "model.pth"
            torch.save(
                {
                    "state_dict":   dl_result.model_state,
                    "architecture": config.hyperparameters.get("architecture", "MLP"),
                    "hyperparameters": config.hyperparameters,
                    "experiment_id": experiment_id,
                },
                str(pth_path),
            )
        except Exception:
            pass  # never crash artifact saving

    # ── Metrics + config ──────────────────────────────────────────────────────
    metrics_path = str(out_dir / "metrics.json")
    Path(metrics_path).write_text(json.dumps(metrics.model_dump(), indent=2))

    config_path = str(out_dir / "config.json")
    Path(config_path).write_text(json.dumps(config.model_dump(), indent=2))

    # ── TensorBoard log dir ───────────────────────────────────────────────────
    tb_dir = tensorboard_log_dir or str(_RUNS_ROOT / experiment_id)

    if enable_mlflow:
        try:
            import mlflow
            with mlflow.start_run(run_name=experiment_id):
                mlflow.log_metrics({k: v for k, v in metrics.model_dump().items() if v is not None})
                mlflow.log_artifact(model_path)
                mlflow.log_artifact(metrics_path)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("MLflow logging failed: %s", exc)

    return ArtifactRefs(
        model_path=model_path,
        metrics_file=metrics_path,
        tensorboard_logs=tb_dir,
    )
