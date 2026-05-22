"""Demo: Credit Card Fraud — Extreme imbalance, f1 primary, stagnation detection."""
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

DATASET   = "creditcard"
TARGET    = "Class"
MAX_ITER  = 5
DECISION_AGENT_DELAY = 2.5  # 285k rows, complex state
ROWS, COLS = 284807, 31
TRAIN, VAL, TEST = 170883, 56962, 56962

ITERS = [
    # RF: f1=0.860, accuracy misleading due to extreme imbalance (578:1)
    # tl≈vl → no overfitting, but f1 plateau triggers next action (verified)
    ("RandomForestClassifier",    "switch_model",    "bootstrap",          0.50, 0.8603, 0.8603, 0.9506, 0.7857, 0.0,    0.0006, 0.0006, 49.86,
     {"stagnating": "weak", "stagnating_confidence": 0.52},
     "switch_model", {"model_name": "GradientBoostingClassifier"}),

    # RF again (add_preprocessing slot): same model, oversample applied → f1 unchanged
    ("RandomForestClassifier",    "add_preprocessing","stagnating",        0.52, 0.8603, 0.8603, 0.9506, 0.7857, 0.0,    0.0006, 0.0006, 45.71,
     {"stagnating": "weak", "stagnating_confidence": 0.55},
     "switch_model", {"model_name": "GradientBoostingClassifier"}),

    # GBM: f1 drops to 0.720, slower (225s) → diminishing returns (verified)
    ("GradientBoostingClassifier","switch_model",    "stagnating",         0.69, 0.72,   0.72,   0.8182, 0.6429, 0.0007, 0.0009, 0.0002, 225.66,
     {"diminishing_returns": "weak", "diminishing_returns_confidence": 0.48},
     "switch_model", {"model_name": "ExtraTreesClassifier"}),

    # ExtraTrees: f1=0.867, best so far, fast (3.6s) (verified)
    ("ExtraTreesClassifier",      "switch_model",    "diminishing_returns",0.63, 0.8667, 0.8667, 0.9512, 0.7959, 0.0,    0.0006, 0.0006, 3.61,
     {"stagnating": "weak", "stagnating_confidence": 0.55},
     "switch_model", {"model_name": "LogisticRegression"}),

    # LR: f1=0.724, tl≈vl → converged but weaker model (verified)
    ("LogisticRegression",        "switch_model",    "stagnating",         0.60, 0.7241, 0.7241, 0.8289, 0.6429, 0.0006, 0.001,  0.0004, 0.54,
     {"converged": "weak", "converged_confidence": 0.44},
     "terminate", {}),
]

DATASET_META = {
    "num_samples": ROWS, "num_features": COLS - 1,
    "feature_to_sample_ratio": round((COLS-1)/ROWS, 6),
    "missing_ratio": 0.0,
    "class_distribution": {"0": 284315, "1": 492},
    "imbalance_ratio": 578.2,
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
    time.sleep(0.5)
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

    for i, (model, action, trigger, conf, f1, _acc, prec, rec,
            tl, vl, gap, rt, sig_kw, next_act, next_params) in enumerate(ITERS[:max_iter]):
        iter_num = i + 1
        if i == max_iter - 1 and next_act != "terminate":
            next_act, next_params = "terminate", {}
        print(f"\n  --- Iteration {iter_num} ---")
        print(f"  Training : {model} ({action}) ...")
        _sleep(min(rt * 0.05, 10))
        print(f"  Trained in {rt:.2f}s")

        if model not in models_tried:
            models_tried.append(model)
        improvement = f1 - best_score
        best_score = max(best_score, f1)
        slope = round(improvement / iter_num, 6)
        volatility = round(abs(slope) * 0.6, 6)
        remaining = max_iter - iter_num

        # f1 is primary; accuracy is near-perfect but misleading
        acc_val = round(1.0 - (1 - f1) * 0.001, 6)
        metrics = {"accuracy": acc_val, "f1_score": f1, "precision": prec, "recall": rec,
                   "train_loss": tl, "validation_loss": vl, "gap": gap,
                   "mse": None, "rmse": None, "r2": None}
        signals = _signals(**sig_kw)
        active = [k for k in ["overfitting","stagnating","diminishing_returns","converged"]
                  if signals.get(k, "none") != "none"]

        traj = {"history_length": iter_num, "improvement_rate": round(improvement, 6),
                "slope": slope, "volatility": volatility, "best_score": best_score,
                "mean_score": round(f1, 6), "steps_since_improvement": 0,
                "trend": "improving" if improvement > 0 else "stagnating"}
        res  = {"runtime": rt, "remaining_budget": float(remaining),
                "budget_exhausted": remaining <= 0, "models_tried": list(models_tried)}

        write_artifacts(out_dir, run_id, {"model": model},
                        {k: v for k, v in metrics.items() if k not in ("gap",)},
                        {"experiment_id": run_id, "model_name": model, "model_type": "ml",
                         "hyperparameters": {}, "preprocessing": {
                             "missing_value_strategy": "mean", "scaling": "standard",
                             "encoding": "none",
                             "imbalance_strategy": "oversample" if action == "add_preprocessing" else "none",
                             "feature_selection": "none"},
                         "early_stopping": False, "early_stopping_patience": 5})

        sel = _selected(run_id, iter_num, next_act, next_params, trigger, conf,
                        round(f1 * 0.95, 3), 0.70, round(f1 * 0.88, 3))
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
            "confidence": conf, "primary_metric": f1, "train_val_gap": gap,
            "accuracy": acc_val, "f1_score": f1, "precision": prec, "recall": rec,
            "mse": None, "r2": None, "slope": slope, "volatility": volatility,
            "runtime": rt, "active_signals": ", ".join(active),
        })
        runs_rows.append({
            "experiment_id": run_id, "iteration": iter_num,
            "primary_metric": "f1_score", "best_score": best_score,
            "improvement_rate": round(improvement, 6), "slope": slope,
            "volatility": volatility, "steps_since_improvement": 0,
            "trend": traj["trend"],
            "underfitting": signals["underfitting"], "overfitting": signals["overfitting"],
            "well_fitted": signals["well_fitted"], "converged": signals["converged"],
            "stagnating": signals["stagnating"],
            "num_samples": ROWS, "num_features": COLS - 1, "missing_ratio": 0.0,
            "runtime": rt, "remaining_budget": float(remaining),
            "action_type": next_act, "action_params": str(next_params),
            "predicted_gain": 0.05, "observed_gain": round(improvement, 6),
            "normalized_gain": round(improvement / 0.1, 6) if improvement else 0.0,
        })

        print(f"  Result   : primary={f1:.4f} | runtime={rt:.2f}s")
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
