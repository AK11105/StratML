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
    "iteration": int,   # ✅ NEW (important for distinguishing iteration 0)

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
    # BOOTSTRAP CONTEXT (ONLY ITERATION 0)
    # =========================
    "bootstrap_context": {   # ✅ NEW (OPTIONAL)
        "source": str,       # "dataprofile"
        "strategy": str      # e.g. "tree_baseline", "linear_baseline"
    } | None,

    # =========================
    # DECISION INTELLIGENCE
    # =========================
    "expected_gain": float | None,   # ⚠️ now optional for iteration 0
    "expected_cost": float,
    "confidence": float | None,      # ⚠️ optional for iteration 0

    # =========================
    # MULTI-AGENT SCORES
    # =========================
    "agent_scores": {
        "performance": float | None,
        "efficiency": float | None,
        "stability": float | None
    },

    # =========================
    # STRUCTURED REASON
    # =========================
    "reason": {
        "trigger": str,
        "evidence": dict,
        "source": str   # ✅ NEW → "bootstrap" | "learned"
    }
}
```

---

# ✅ 3. FINAL `DataProfile`

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

# ✅ 4. FINAL `StateObject`

```python
StateObject = {

    # =========================
    # META (TRACEABILITY)
    # =========================
    "meta": {
        "experiment_id": str,
        "iteration": int,
        "timestamp": str
    },

    # =========================
    # OBJECTIVE CONTEXT
    # =========================
    "objective": {
        "primary_metric": str,        # e.g. "accuracy"
        "optimization_goal": str,     # "maximize" | "minimize"
    },

    # =========================
    # CURRENT PERFORMANCE
    # =========================
    "metrics": {
        "primary": float,             # extracted from metrics

        "secondary": {
            "accuracy": float | None,
            "precision": float | None,
            "recall": float | None,
            "f1_score": float | None,
            "mse": float | None,
            "rmse": float | None,
            "r2": float | None
        },

        "train_val_gap": float        # from derived_metrics
    },

    # =========================
    # TRAJECTORY (HISTORY-BASED)
    # =========================
    "trajectory": {
        "history_length": int,

        "improvement_rate": float,        # Δ metric
        "slope": float,                  # trend over last N
        "volatility": float,             # std dev of recent scores

        "best_score": float,
        "mean_score": float,

        "steps_since_improvement": int,

        "trend": str                     # "improving" | "stagnating" | "degrading"
    },

    # =========================
    # DATASET STATE (DYNAMIC)
    # =========================
    "dataset": {
        "num_samples": int,
        "num_features": int,

        "feature_to_sample_ratio": float,

        "missing_ratio": float,

        "class_distribution": dict | None,
        "imbalance_ratio": float | None
    },

    # =========================
    # MODEL CONTEXT
    # =========================
    "model": {
        "model_name": str,
        "model_type": str,          # "ml" | "dl"

        "hyperparameters": dict,

        "complexity_hint": str | None,   # "low" | "medium" | "high" (derived)

        "training": {
            "runtime": float,
            "convergence_epoch": int,
            "early_stopped": bool | None
        }
    },

    # =========================
    # GENERALIZATION
    # =========================
    "generalization": {
        "train_loss": float,
        "validation_loss": float,
        "gap": float
    },

    # =========================
    # RESOURCES / BUDGET
    # =========================
    "resources": {
        "runtime": float,
        "gpu_used": bool,
        "cpu_time": float,

        "remaining_budget": float | None,
        "budget_exhausted": bool
    },

    # =========================
    # SEARCH CONTEXT
    # =========================
    "search": {
        "models_tried": list[str],
        "unique_models_count": int,

        "repeated_configs": int
    },

    # =========================
    # SIGNALS (DERIVED)
    # Each signal: str ("none" | "weak" | "strong") + float confidence [0.0, 1.0]
    # =========================
    "signals": {

        # fitting
        "underfitting": str,                  # "none" | "weak" | "strong"
        "underfitting_confidence": float,
        "overfitting": str,
        "overfitting_confidence": float,
        "well_fitted": str,
        "well_fitted_confidence": float,

        # convergence
        "converged": str,
        "converged_confidence": float,
        "stagnating": str,
        "stagnating_confidence": float,
        "diverging": str,
        "diverging_confidence": float,

        # stability
        "unstable_training": str,
        "unstable_training_confidence": float,
        "high_variance": str,
        "high_variance_confidence": float,

        # efficiency
        "too_slow": str,
        "too_slow_confidence": float,

        # optimization
        "plateau_detected": str,
        "plateau_detected_confidence": float,
        "diminishing_returns": str,
        "diminishing_returns_confidence": float
    },

    # =========================
    # UNCERTAINTY (MODEL-BASED)
    # =========================
    "uncertainty": {
        "prediction_variance": float | None,
        "confidence": float | None
    },

    # =========================
    # ACTION CONTEXT
    # =========================
    "action_context": {
        "previous_action": str | None,
        "previous_action_success": bool | None,
        "action_effect_magnitude": float | None
    },

    # =========================
    # CONSTRAINTS
    # =========================
    "constraints": {
        "allowed_models": list[str],
        "max_iterations": int,
        "time_budget": float | None
    }
}
```

# ⚙️ CLI Configuration System

This document defines the configuration structure for running the AutoML system via CLI.

---

# 🧩 Config Philosophy

The system supports **multiple user control modes**:

- **Beginner** → Fully automatic
- **Intermediate** → Partial control
- **Expert** → Full manual control

---

# 📄 Recommended Format: YAML

---

# 🧾 Full Config Schema

```yaml
mode: beginner  # beginner | intermediate | expert

dataset:
  path: "data/train.csv"
  target_column: "label"

execution:
  max_iterations: 20
  timeout_per_run: 300
  random_seed: 42

split:
  method: stratified  # stratified | random
  test_size: 0.2

logging:
  enable_mlflow: true
  enable_tensorboard: true
  log_level: info

constraints:
  max_memory: null
  max_cpu: null

# -----------------------
# MODE-SPECIFIC SETTINGS
# -----------------------

beginner:
  fully_automatic: true

intermediate:
  allowed_models:
    - random_forest
    - xgboost

  allow_preprocessing_control: true

expert:
  model:
    type: xgboost
    hyperparameters:
      n_estimators: 200
      max_depth: 6

  preprocessing:
    missing_value_strategy: mean
    encoding: onehot
    scaling: standard

  training:
    epochs: 50
    batch_size: 32
```

---

# 🧠 Mode Behavior Rules

## Beginner Mode
- Ignores all manual overrides
- Team B has full decision control

## Intermediate Mode
- Restricts model/search space
- Allows partial overrides

## Expert Mode
- User-defined pipeline
- Team B decisions are overridden or constrained

---

# 💻 CLI Commands

## Core Commands

```bash
automl run --config config.yaml
automl validate-config config.yaml
automl profile-data --input data.csv
```

---

## Advanced Usage

```bash
automl run \
  --config config.yaml \
  --mode expert \
  --max-iter 10 \
  --dry-run
```

---

## Utility Commands

```bash
automl init          # create default config
automl doctor        # check environment
automl explain-run   # inspect past run
```

---

# ⚠️ Override Priority

1. CLI flags
2. Config file
3. Mode defaults

---

# 🧱 Design Notes

- YAML preferred for readability
- Must support future extensions (new models, pipelines)
- Mode abstraction must be enforced in orchestrator
