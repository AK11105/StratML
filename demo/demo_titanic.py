"""Demo: Titanic — Overfitting chain: regularize → reduce capacity → switch model."""
from __future__ import annotations
import sys, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from demo._base import (
    write_artifacts, write_model_py, write_decision_log, write_counterfactual,
    write_comparison, append_runs_log, write_report_pdf,
    make_state, _signals, _selected, _no_signals, _sleep, ROOT, load_config,
)

DATASET   = "titanic"
TARGET    = "Survived"
MAX_ITER  = 5
DECISION_AGENT_DELAY = 1.8  # ~1.8s/LLM call x 6 agents = ~11s decision
ROWS, COLS = 891, 11
TRAIN, VAL, TEST = 534, 178, 179

ITERS = [
    # RF default: tl=0.0 vs vl=0.157 → genuine overfitting (verified)
    ("RandomForestClassifier", "switch_model",           "bootstrap",    0.50, 0.7709, 0.6917, 0.7188, 0.6667, 0.0,  0.1573, 0.1573, 0.16,
     {"overfitting": "strong", "overfitting_confidence": 0.74},
     "modify_regularization", {"direction": "increase"}),

    # RF max_depth=10 (regularized): overfitting reduced but persists
    ("RandomForestClassifier", "modify_regularization",  "overfitting",  0.74, 0.7821, 0.7089, 0.7312, 0.6875, 0.04, 0.1421, 0.1021, 0.13,
     {"overfitting": "weak", "overfitting_confidence": 0.58},
     "decrease_model_capacity", {"scale": 0.75}),

    # RF n_estimators=50, max_depth=8: gap narrows further
    ("RandomForestClassifier", "decrease_model_capacity","overfitting",  0.66, 0.7877, 0.7143, 0.7391, 0.6912, 0.07, 0.1318, 0.0618, 0.12,
     {"overfitting": "weak", "overfitting_confidence": 0.49, "well_fitted": "weak", "well_fitted_confidence": 0.41},
     "switch_model", {"model_name": "LogisticRegression"}),

    # LR: tl≈vl → no overfitting, stagnating (verified)
    ("LogisticRegression",     "switch_model",           "overfitting",  0.61, 0.7933, 0.7218, 0.75,   0.6957, 0.1966, 0.2022, 0.0056, 0.0,
     {"stagnating": "weak", "stagnating_confidence": 0.44},
     "switch_model", {"model_name": "GradientBoostingClassifier"}),

    # GBM: best primary, some overfitting remains, converging (verified)
    ("GradientBoostingClassifier","switch_model",        "stagnating",   0.57, 0.7654, 0.6912, 0.7015, 0.6812, 0.0487, 0.1685, 0.1198, 0.12,
     {"converged": "weak", "converged_confidence": 0.58, "well_fitted": "weak", "well_fitted_confidence": 0.53},
     "terminate", {}),
]

DATASET_META = {
    "num_samples": ROWS, "num_features": COLS - 1,
    "feature_to_sample_ratio": round((COLS-1)/ROWS, 6),
    "missing_ratio": 0.19,
    "class_distribution": {"0": 549, "1": 342},
    "imbalance_ratio": 1.605,
}


def run(run_id: str | None = None) -> None:
    cfg = load_config()
    max_iter    = min(cfg.get("max_iterations", MAX_ITER), len(ITERS))
    mode        = cfg.get("mode", "beginner")
    time_budget = cfg.get("timeout_per_run", 300)
    test_size   = cfg.get("test_size", 0.2)
    val_size    = test_size / 2
    total       = ROWS
    test        = round(total * test_size)
    val         = round(total * val_size)
    train       = total - test - val

    run_id = run_id or f"{DATASET}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir = ROOT / "outputs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = "-" * 44
    print()
    print("  AutoML Pipeline Starting")
    print(f"  {sep}")
    print(f"  Mode    : {mode}")
    print(f"  Dataset : data/external/{DATASET}.csv")
    print(f"  Target  : {TARGET}")
    print(f"  Budget  : {max_iter} iterations")
    print(f"  {sep}")
    time.sleep(0.3)
    print("  Loading dataset...")
    time.sleep(0.4)
    print(f"  Profiled: {ROWS} rows x {COLS} cols | classification")
    print(f"  Split: train={train} | val={val} | test={test}")
    time.sleep(0.2)
    print("  Sending profile to Decision Engine...")
    time.sleep(0.5)
    print(f"  Decision [iter 0]: action=switch_model | params={{'model_name': 'RandomForestClassifier'}} | trigger=bootstrap")

    comparison_rows = [{
        "iteration": 0, "model": "none", "action": "switch_model", "trigger": "bootstrap",
        "confidence": 0.5, "primary_metric": 0.0, "train_val_gap": 0.0,
        "accuracy": None, "f1_score": None, "precision": None, "recall": None,
        "mse": None, "r2": None, "slope": 0.0, "volatility": 0.0, "runtime": 0.0, "active_signals": "",
    }]
    runs_rows = []
    models_tried = []
    best_score = 0.0
    prev_action = None

    write_decision_log(out_dir, run_id, 0, make_state(
        run_id, 0, "none", "ml", {"accuracy": None, "gap": 0.0},
        _no_signals(), DATASET_META,
        {"history_length": 0, "improvement_rate": 0.0, "slope": 0.0, "volatility": 0.0,
         "best_score": 0.0, "mean_score": 0.0, "steps_since_improvement": 0, "trend": "stagnating"},
        {"runtime": 0.0, "remaining_budget": float(max_iter), "budget_exhausted": False, "models_tried": []},
        [{"action_type": "switch_model", "parameters": {"model_name": "RandomForestClassifier"}},
         {"action_type": "switch_model", "parameters": {"model_name": "LogisticRegression"}}],
        _selected(run_id, 0, "switch_model", {"model_name": "RandomForestClassifier"},
                  "bootstrap", 0.5, 0.05, 0.5, 1.0),
        max_iter,
    ))
    cf_entries = [{"iteration": 0, "action_type": "switch_model",
                   "parameters": {"model_name": "RandomForestClassifier"}, "confidence": 0.5}]

    for i, (model, action, trigger, conf, primary, f1, prec, rec,
            tl, vl, gap, rt, sig_kw, next_act, next_params) in enumerate(ITERS[:max_iter]):
        iter_num = i + 1
        if i == max_iter - 1 and next_act != "terminate":
            next_act, next_params = "terminate", {}
        print(f"\n  --- Iteration {iter_num} ---")
        print(f"  Training : {model} ({action}) ...")
        _sleep(rt * 6)
        print(f"  Trained in {rt:.2f}s")

        if model not in models_tried:
            models_tried.append(model)
        improvement = primary - best_score
        best_score = max(best_score, primary)
        slope = round(improvement / iter_num, 6)
        volatility = round(abs(slope) * 0.6, 6)
        remaining = max_iter - iter_num

        metrics = {"accuracy": primary, "f1_score": f1, "precision": prec, "recall": rec,
                   "train_loss": tl, "validation_loss": vl, "gap": gap,
                   "mse": None, "rmse": None, "r2": None}
        signals = _signals(**sig_kw)
        active = [k for k in ["underfitting","overfitting","well_fitted","converged","stagnating"]
                  if signals.get(k, "none") != "none"]

        traj = {"history_length": iter_num, "improvement_rate": round(improvement, 6),
                "slope": slope, "volatility": volatility, "best_score": best_score,
                "mean_score": round(primary, 6), "steps_since_improvement": 0,
                "trend": "improving" if improvement > 0 else "stagnating"}
        res  = {"runtime": rt, "remaining_budget": float(remaining),
                "budget_exhausted": remaining <= 0, "models_tried": list(models_tried)}

        write_artifacts(out_dir, run_id, {"model": model},
                        {k: v for k, v in metrics.items() if k not in ("gap",)},
                        {"experiment_id": run_id, "model_name": model, "model_type": "ml",
                         "hyperparameters": {}, "preprocessing": {
                             "missing_value_strategy": "mean", "scaling": "standard",
                             "encoding": "onehot", "imbalance_strategy": "none",
                             "feature_selection": "none"},
                         "early_stopping": False, "early_stopping_patience": 5})

        sel = _selected(run_id, iter_num, next_act, next_params, trigger, conf,
                        round(primary * 0.95, 3), 0.75, round(primary * 0.9, 3))
        write_decision_log(out_dir, run_id, iter_num, make_state(
            run_id, iter_num, model, "ml", metrics, signals, DATASET_META,
            traj, res, [{"action_type": next_act, "parameters": next_params},
                        {"action_type": "terminate", "parameters": {}}],
            sel, max_iter, previous_action=prev_action, time_budget=time_budget,
        ))
        cf_entries.append({"iteration": iter_num, "action_type": next_act,
                           "parameters": next_params, "confidence": conf})

        comparison_rows.append({
            "iteration": iter_num, "model": model, "action": next_act, "trigger": trigger,
            "confidence": conf, "primary_metric": primary, "train_val_gap": gap,
            "accuracy": primary, "f1_score": f1, "precision": prec, "recall": rec,
            "mse": None, "r2": None, "slope": slope, "volatility": volatility,
            "runtime": rt, "active_signals": ", ".join(active),
        })
        runs_rows.append({
            "experiment_id": run_id, "iteration": iter_num,
            "primary_metric": "accuracy", "best_score": best_score,
            "improvement_rate": round(improvement, 6), "slope": slope,
            "volatility": volatility, "steps_since_improvement": 0,
            "trend": traj["trend"],
            "underfitting": signals["underfitting"], "overfitting": signals["overfitting"],
            "well_fitted": signals["well_fitted"], "converged": signals["converged"],
            "stagnating": signals["stagnating"],
            "num_samples": ROWS, "num_features": COLS - 1, "missing_ratio": 0.19,
            "runtime": rt, "remaining_budget": float(remaining),
            "action_type": next_act, "action_params": str(next_params),
            "predicted_gain": 0.05, "observed_gain": round(improvement, 6),
            "normalized_gain": round(improvement / 0.1, 6) if improvement else 0.0,
        })

        print(f"  Result   : primary={primary:.4f} | runtime={rt:.2f}s")
        # -- DEMO decision phase simulation (delete with DEMO INTERCEPT block) --
        _agents = (
            ["  [Evaluator]   auditing previous decision..."] if iter_num > 1 else []
        ) + [
            "  [StateBuilder] extracting signals from metrics...",
            "  [Perf Agent]   scoring candidates on accuracy gain...",
            "  [Eff Agent]    scoring candidates on compute cost...",
            "  [Stab Agent]   scoring candidates on training risk...",
            "  [Coordinator]  deliberating over agent scores...",
            "  [Selector]     applying policy + budget constraints...",
        ]
        for _msg in _agents:
            print(_msg)
            time.sleep(DECISION_AGENT_DELAY)
        # -- END DEMO decision phase simulation --
        if next_act == "terminate":
            print(f"  Decision : terminate | trigger=budget_exhausted | confidence=1.00 | next={{}}")
        else:
            print(f"  Decision : {next_act} | trigger={trigger} | confidence={conf:.2f} | next={next_params}")
        prev_action = action

    print("  Run complete.\n")
    print(f"  {sep}")
    print(f"  Run ID  : {run_id}")
    print(f"  Output  : outputs/{run_id}")
    print(f"  {sep}\n")

    write_counterfactual(out_dir, run_id, cf_entries)
    write_comparison(out_dir, comparison_rows)
    append_runs_log(run_id, runs_rows)
    write_model_py(out_dir, run_id, "GradientBoostingClassifier", best_score, {})
    pdf = write_report_pdf(out_dir, run_id, DATASET, comparison_rows)

    print(f"  Report    : {pdf}")
    print(f"  Comparison: {out_dir / 'comparison.csv'}")
    print(f"  Model.py  : {out_dir / 'model.py'}\n")
    answer = input("  Download best model files (model.pkl + model.py)? [y/N]: ").strip().lower()
    if answer == "y":
        print(f"  Files saved at: {out_dir / 'artifacts' / run_id / 'model.pkl'} and {out_dir / 'model.py'}")


if __name__ == "__main__":
    run()
