DecisionStateObject = {

    # =========================
    # META (trace + reproducibility)
    # =========================
    "meta": {
        "run_id": str,
        "experiment_id": str,
        "iteration": int,
        "total_iterations": int,
        "timestamp": str,
        "random_seed": int,
        "mode": str,  # beginner | intermediate | expert
    },

    # =========================
    # OBJECTIVE CONTEXT
    # =========================
    "objective": {
        "primary_metric": str,   # accuracy / f1 / auc etc.
        "optimization_goal": str,  # maximize / minimize
        "acceptable_threshold": float | None,
    },

    # =========================
    # CURRENT PERFORMANCE
    # =========================
    "metrics": {
        "primary": float,
        "secondary": {
            "accuracy": float | None,
            "precision": float | None,
            "recall": float | None,
            "f1_score": float | None,
            "roc_auc": float | None,
            "log_loss": float | None,
        },
        "train_vs_val_gap": float | None,
    },

    # =========================
    # PERFORMANCE TRAJECTORY
    # =========================
    "trajectory": {
        "history_length": int,

        "improvement_rate": float,
        "relative_improvement": float,
        "slope": float,
        "second_derivative": float,

        "best_score": float,
        "worst_score": float,
        "mean_score": float,

        "steps_since_improvement": int,
        "steps_since_best": int,

        "volatility": float,  # std deviation
        "stability_index": float,

        "trend": str,  # improving | stagnating | degrading
    },

    # =========================
    # DATASET META FEATURES
    # =========================
    "dataset": {
        "num_samples": int,
        "num_features": int,

        "num_numerical": int,
        "num_categorical": int,
        "num_datetime": int,
        "num_text": int,

        "feature_to_sample_ratio": float,

        "missing_ratio": float,
        "missing_per_feature_max": float,

        "class_entropy": float | None,
        "imbalance_ratio": float | None,
        "num_classes": int | None,

        "skewness_mean": float | None,
        "kurtosis_mean": float | None,

        "correlation_mean": float | None,
        "correlation_max": float | None,

        "outlier_ratio": float | None,
    },

    # =========================
    # MODEL CONTEXT
    # =========================
    "model": {
        "type": str,
        "family": str,  # tree / linear / neural / etc.

        "hyperparameters": dict,

        "complexity": {
            "num_parameters": int | None,
            "depth": int | None,
            "regularization_strength": float | None,
        },

        "training": {
            "training_time": float,
            "epochs": int | None,
            "batch_size": int | None,
            "early_stopped": bool,
        },

        "inference_time": float | None,
    },

    # =========================
    # GENERALIZATION CONTEXT
    # =========================
    "generalization": {
        "train_score": float | None,
        "validation_score": float | None,
        "gap": float | None,
        "cross_val_variance": float | None,
    },

    # =========================
    # RESOURCE / BUDGET STATE
    # =========================
    "resources": {
        "elapsed_time": float,
        "remaining_time": float | None,

        "used_iterations": int,
        "remaining_iterations": int,

        "memory_usage": float | None,
        "cpu_usage": float | None,

        "budget_exhausted": bool,
    },

    # =========================
    # SEARCH HISTORY CONTEXT
    # =========================
    "search": {
        "models_tried": list[str],
        "unique_models_count": int,

        "hyperparameter_coverage": float,
        "search_diversity": float,

        "repeated_configs": int,
        "exploration_vs_exploitation_ratio": float,
    },

    # =========================
    # SIGNALS (DERIVED INTELLIGENCE)
    # =========================
    "signals": {

        # learning quality
        "underfitting": bool,
        "overfitting": bool,
        "well_fitted": bool,

        # convergence
        "converged": bool,
        "stagnating": bool,
        "diverging": bool,

        # stability
        "unstable_training": bool,
        "high_variance": bool,
        "noisy_metric": bool,

        # efficiency
        "too_slow": bool,
        "resource_heavy": bool,

        # search quality
        "search_exhausted": bool,
        "low_diversity": bool,
        "repeating_configs": bool,

        # dataset difficulty
        "high_dimensional": bool,
        "imbalanced": bool,
        "noisy_data": bool,

        # optimization state
        "plateau_detected": bool,
        "diminishing_returns": bool,
    },

    # =========================
    # UNCERTAINTY ESTIMATION
    # =========================
    "uncertainty": {
        "prediction_variance": float | None,
        "confidence": float | None,
        "epistemic_uncertainty": float | None,
        "aleatoric_uncertainty": float | None,
    },

    # =========================
    # ACTION CONTEXT (INPUT TO POLICY)
    # =========================
    "action_context": {
        "previous_action": str | None,
        "previous_action_success": bool | None,
        "action_effect_magnitude": float | None,
    },

    # =========================
    # SEARCH SPACE / CONSTRAINTS
    # =========================
    "constraints": {
        "allowed_models": list[str],
        "blocked_models": list[str] | None,

        "hyperparameter_space": dict,

        "max_iterations": int,
        "time_budget": float | None,
    }
}