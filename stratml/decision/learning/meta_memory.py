"""
meta_memory.py
--------------
Decision/Learning — Cross-Run Meta-Memory.

Indexes completed runs by DatasetMetaFeatures vector.
At the start of a new run, retrieves top-k similar past runs by cosine
similarity and returns their best-performing action sequences for use
as bootstrap candidates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_MEMORY_FILE = Path("runs/decision_logs/meta_memory.jsonl")
_TOP_K = 3


def _to_vector(meta: dict) -> list[float]:
    """Convert a meta-features dict to a numeric vector."""
    return [
        float(meta.get("num_samples", 0)),
        float(meta.get("num_features", 0)),
        float(meta.get("feature_sample_ratio", 0)),
        float(meta.get("class_entropy", 0)),
        float(meta.get("missing_value_ratio", 0)),
        float(meta.get("imbalance_ratio") or 1.0),
    ]


def _cosine(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve_similar_actions(meta_features, top_k: int = _TOP_K) -> list[str]:
    """
    Given a DatasetMetaFeatures object, return a list of model names that
    worked well on the most similar past runs.
    """
    if not _MEMORY_FILE.exists():
        return []
    try:
        records = []
        with open(_MEMORY_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            return []

        query_vec = _to_vector({
            "num_samples": meta_features.num_samples,
            "num_features": meta_features.num_features,
            "feature_sample_ratio": meta_features.feature_sample_ratio,
            "class_entropy": meta_features.class_entropy,
            "missing_value_ratio": meta_features.missing_value_ratio,
            "imbalance_ratio": meta_features.imbalance_ratio,
        })

        scored = []
        for r in records:
            vec = _to_vector(r.get("meta_features", {}))
            sim = _cosine(query_vec, vec)
            scored.append((sim, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        # Collect best model names from similar runs
        models: list[str] = []
        seen: set[str] = set()
        for _, r in top:
            m = r.get("best_model")
            if m and m not in seen:
                models.append(m)
                seen.add(m)
        return models
    except Exception as exc:
        log.warning("meta_memory: retrieval failed (%s)", exc)
        return []


def record_run(meta_features, best_model: str, best_score: float, run_id: str) -> None:
    """Persist a completed run's meta-features and best outcome."""
    _MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "run_id": run_id,
        "best_model": best_model,
        "best_score": best_score,
        "meta_features": {
            "num_samples": meta_features.num_samples,
            "num_features": meta_features.num_features,
            "feature_sample_ratio": meta_features.feature_sample_ratio,
            "class_entropy": meta_features.class_entropy,
            "missing_value_ratio": meta_features.missing_value_ratio,
            "imbalance_ratio": meta_features.imbalance_ratio,
        },
    }
    with open(_MEMORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
