# ✅ 1. FINAL `ExperimentResult`

```python
ExperimentResult = {
    # =========================
    # IDENTIFICATION
    # =========================
    "experiment_id": str,
    "iteration": int,
    "dataset_name": str,

    # =========================
    # MODEL INFO
    # =========================
    "model_name": str,
    "model_type": str,   # "ml" | "dl"
    "hyperparameters": dict,

    # =========================
    # PREPROCESSING (ACTUAL)
    # =========================
    "preprocessing_applied": {
        "missing_value_strategy": str,
        "scaling": str,
        "encoding": str,
        "imbalance_strategy": str,
        "feature_selection": str
    },

    # =========================
    # CORE METRICS
    # =========================
    "metrics": {
        "accuracy": float,
        "f1_score": float,
        "precision": float,
        "recall": float,
        "train_loss": float,
        "validation_loss": float,
        "mse": float,
        "rmse": float,
        "r2": float
    },

    # =========================
    # TRAINING DYNAMICS
    # =========================
    "train_curve": list[float],
    "validation_curve": list[float],
    "convergence_epoch": int,

    # =========================
    # DERIVED (STANDARDIZED)
    # =========================
    "derived_metrics": {
        "train_val_gap": float
    },

    # =========================
    # DATA SNAPSHOT (POST-PREPROCESSING)
    # =========================
    "dataset_snapshot": {
        "num_samples": int,
        "num_features": int,
        "class_distribution": dict,
        "missing_value_ratio": float
    },

    # =========================
    # SYSTEM METRICS
    # =========================
    "runtime": float,
    "resource_usage": {
        "gpu_used": bool,
        "gpu_memory_gb": float,
        "cpu_time_sec": float
    },

    # =========================
    # ARTIFACTS
    # =========================
    "artifacts": {
        "model_path": str,
        "metrics_file": str,
        "tensorboard_logs": str
    }
}
```

---

# ✅ 2. FINAL `ActionDecision`

```python
ActionDecision = {
    # =========================
    # IDENTIFICATION
    # =========================
    "experiment_id": str,

    # =========================
    # ACTION
    # =========================
    "action_type": str,

    "parameters": dict,

    # =========================
    # PREPROCESSING PLAN
    # =========================
    "preprocessing": {
        "missing_value_strategy": str,
        "scaling": str,
        "encoding": str,
        "imbalance_strategy": str,
        "feature_selection": str
    },

    # =========================
    # DECISION INTELLIGENCE
    # =========================
    "expected_gain": float,
    "expected_cost": float,
    "confidence": float,

    # =========================
    # MULTI-AGENT SCORES
    # =========================
    "agent_scores": {
        "performance": float,
        "efficiency": float,
        "stability": float
    },

    # =========================
    # STRUCTURED REASON
    # =========================
    "reason": {
        "trigger": str,
        "evidence": dict
    }
}
```

---

# ✅ FINAL `DataProfile`

```python
DataProfile = {
    # =========================
    # DATASET IDENTITY
    # =========================
    "dataset_name": str,
    "dataset_type": str,        # "tabular" | "text" | "vision"
    "problem_type": str,        # "classification" | "regression"

    # =========================
    # SIZE & SHAPE
    # =========================
    "num_samples": int,
    "num_features": int,
    "target_column": str,

    # =========================
    # FEATURE TYPES
    # =========================
    "numerical_columns": list[str],
    "categorical_columns": list[str],

    "feature_type_distribution": {
        "numerical_ratio": float,
        "categorical_ratio": float
    },

    # =========================
    # DATA QUALITY
    # =========================
    "missing_value_ratio": float,

    "feature_summary": [
        {
            "name": str,
            "dtype": str,
            "unique_values": int,
            "missing_percentage": float,
            "distribution": str   # "normal" | "skewed" | "uniform" | "unknown"
        }
    ],

    # =========================
    # TARGET DISTRIBUTION
    # =========================
    "class_distribution": dict,      # classification only

    "imbalance_ratio": float,        # max_class / min_class (if classification)

    # =========================
    # STATISTICAL SIGNALS
    # =========================
    "feature_variance_mean": float,
    "class_entropy": float,          # classification only

    # =========================
    # RECOMMENDATIONS (LIGHT GUIDANCE)
    # =========================
    "recommended_metrics": list[str]
}
```

---