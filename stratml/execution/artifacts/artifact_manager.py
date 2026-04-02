"""
artifact_manager.py
-------------------
Phase 7 — Persist model, metrics, and config. Return ArtifactRefs.
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
) -> ArtifactRefs:
    """Save model + metrics + config to disk. Return ArtifactRefs."""
    out_dir = _ARTIFACTS_ROOT / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = str(out_dir / "model.pkl")
    joblib.dump(model, model_path)

    metrics_path = str(out_dir / "metrics.json")
    Path(metrics_path).write_text(json.dumps(metrics.model_dump(), indent=2))

    config_path = str(out_dir / "config.json")
    Path(config_path).write_text(json.dumps(config.model_dump(), indent=2))

    tb_dir = tensorboard_log_dir or str(_RUNS_ROOT / experiment_id)

    return ArtifactRefs(
        model_path=model_path,
        metrics_file=metrics_path,
        tensorboard_logs=tb_dir,
    )
