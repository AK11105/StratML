"""
meta_features.py
----------------
Phase 3 (Dev B) — Dataset Meta-Feature Extraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from stratml.execution.schemas import DataProfile


@dataclass
class DatasetMetaFeatures:
    num_samples: int
    num_features: int
    feature_sample_ratio: float
    class_entropy: float
    missing_value_ratio: float
    feature_variance_mean: float
    imbalance_ratio: Optional[float]


def extract(profile: DataProfile) -> DatasetMetaFeatures:
    num_samples = profile.rows
    num_features = profile.columns - 1

    feature_sample_ratio = round(num_features / max(num_samples, 1), 6)
    class_entropy = _compute_entropy(profile.class_distribution, num_samples)
    feature_variance_mean = _mean_unique_values(profile)

    imbalance_ratio: Optional[float] = None
    if profile.class_distribution and len(profile.class_distribution) >= 2:
        counts = list(profile.class_distribution.values())
        imbalance_ratio = round(max(counts) / max(min(counts), 1), 4)

    return DatasetMetaFeatures(
        num_samples=num_samples,
        num_features=num_features,
        feature_sample_ratio=feature_sample_ratio,
        class_entropy=class_entropy,
        missing_value_ratio=profile.missing_value_ratio,
        feature_variance_mean=feature_variance_mean,
        imbalance_ratio=imbalance_ratio,
    )


def _compute_entropy(class_distribution: dict, num_samples: int) -> float:
    if not class_distribution or num_samples == 0:
        return 0.0
    entropy = 0.0
    for count in class_distribution.values():
        p = count / num_samples
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 6)


def _mean_unique_values(profile: DataProfile) -> float:
    if not profile.feature_summary:
        return 0.0
    return round(
        sum(f.unique_values for f in profile.feature_summary) / len(profile.feature_summary),
        4,
    )
