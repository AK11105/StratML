"""Demo: California Housing — Regression, SVR triggers too_slow, r2 metric."""
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

DATASET   = "california_housing"
TARGET    = "MedHouseVal"
MAX_ITER  = 5
DECISION_AGENT_DELAY = 2.2  # larger dataset, more context
ROWS, COLS = 20640, 9
TRAIN, VAL, TEST = 12384, 4128, 4128

ITERS = [
    # RF: tl=0.036 vs vl=0.260 → large gap, overfitting not underfitting (verified)
    ("RandomForestRegressor",    "switch_model",    "bootstrap",          0.50, 0.8019, None, None, None, 0.0364, 0.2602, 0.2238, 1.54,
     {"overfitting": "strong", "overfitting_confidence": 0.71},
     "switch_model", {"model_name": "GradientBoostingRegressor"}),

    # GBM: tl≈vl, diminishing returns over RF (verified)
    ("GradientBoostingRegressor","switch_model",    "overfitting",        0.63, 0.7789, None, None, None, 0.2513, 0.2827, 0.0314, 2.07,
     {"diminishing_returns": "weak", "diminishing_returns_confidence": 0.48},
     "switch_model", {"model_name": "SVR"}),

    # SVR: lower r2, slow — too_slow fires (verified: 3.76s on 12k rows)
    ("SVR",                      "switch_model",    "diminishing_returns",0.58, 0.7327, None, None, None, 0.3266, 0.3431, 0.0165, 3.76,
     {"too_slow": "strong", "too_slow_confidence": 0.92},
     "switch_model", {"model_name": "Ridge"}),

    # Ridge: r2=0.57, tl≈vl → underfitting (linear model on nonlinear data, verified)
    ("Ridge",                    "switch_model",    "too_slow",           0.92, 0.5713, None, None, None, 0.5144, 0.5297, 0.0153, 0.01,
     {"underfitting": "strong", "underfitting_confidence": 0.74},
     "switch_model", {"model_name": "ExtraTreesRegressor"}),

    # ExtraTrees: r2≈RF, gap=0.257 → overfitting, but best score so far → terminate
    ("ExtraTreesRegressor",      "switch_model",    "underfitting",       0.68, 0.7988, None, None, None, 0.0, 0.2572, 0.2572, 0.73,
     {"overfitting": "weak", "overfitting_confidence": 0.55, "well_fitted": "weak", "well_fitted_confidence": 0.41},
     "terminate", {}),
]

DATASET_META = {
    "num_samples": ROWS, "num_features": COLS - 1,
    "feature_to_sample_ratio": round((COLS-1)/ROWS, 6),
    "missing_ratio": 0.0,
    "class_distribution": None,
    "imbalance_ratio": None,
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
    print(f"  Dataset : data/raw/{DATASET}.csv")
    print(f"  Target  : {TARGET}")
    print(f"  Budget  : {max_iter} iterations")
    print(f"  {sep}")
    time.sleep(0.3)
    print("  Loading dataset...")
    time.sleep(0.4)
    print(f"  Profiled: {ROWS} rows x {COLS} cols | regression")
    print(f"  Split: train={train} | val={val} | test={test}")
    time.sleep(0.2)
    print("  Sending profile to Decision Engine...")
    time.sleep(0.5)
    print(f"  Decision [iter 0]: action=switch_model | params={{'model_name': 'RandomForestRegressor'}} | trigger=bootstrap")

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
        run_id, 0, "none", "ml", {"r2": None, "gap": 0.0},
        _no_signals(), DATASET_META,
        {"history_length": 0, "improvement_rate": 0.0, "slope": 0.0, "volatility": 0.0,
         "best_score": 0.0, "mean_score": 0.0, "steps_since_improvement": 0, "trend": "stagnating"},
        {"runtime": 0.0, "remaining_budget": float(max_iter), "budget_exhausted": False, "models_tried": []},
        [{"action_type": "switch_model", "parameters": {"model_name": "RandomForestRegressor"}},
         {"action_type": "switch_model", "parameters": {"model_name": "Ridge"}}],
        _selected(run_id, 0, "switch_model", {"model_name": "RandomForestRegressor"},
                  "bootstrap", 0.5, 0.05, 0.5, 1.0),
        max_iter,
    ))
    cf_entries = [{"iteration": 0, "action_type": "switch_model",
                   "parameters": {"model_name": "RandomForestRegressor"}, "confidence": 0.5}]

    for i, (model, action, trigger, conf, r2, _f1, _prec, _rec,
            tl, vl, gap, rt, sig_kw, next_act, next_params) in enumerate(ITERS[:max_iter]):
        iter_num = i + 1
        if i == max_iter - 1 and next_act != "terminate":
            next_act, next_params = "terminate", {}
        print(f"\n  --- Iteration {iter_num} ---")
        print(f"  Training : {model} ({action}) ...")
        # SVR takes long — simulate it
        sleep_t = min(rt * 0.3, 6.0)
        _sleep(sleep_t)
        print(f"  Trained in {rt:.2f}s")

        if model not in models_tried:
            models_tried.append(model)
        improvement = r2 - best_score
        best_score = max(best_score, r2)
        slope = round(improvement / iter_num, 6)
        volatility = round(abs(slope) * 0.7, 6)
        remaining = max_iter - iter_num

        mse_val = round((1 - r2) * 2.1, 4)
        metrics = {"accuracy": None, "f1_score": None, "precision": None, "recall": None,
                   "train_loss": tl, "validation_loss": vl, "gap": gap,
                   "mse": mse_val, "rmse": round(mse_val**0.5, 4), "r2": r2}
        signals = _signals(**sig_kw)
        active = [k for k in ["underfitting","too_slow","diminishing_returns","converged"]
                  if signals.get(k, "none") != "none"]

        traj = {"history_length": iter_num, "improvement_rate": round(improvement, 6),
                "slope": slope, "volatility": volatility, "best_score": best_score,
                "mean_score": round(r2, 6), "steps_since_improvement": 0,
                "trend": "improving" if improvement > 0 else "stagnating"}
        res  = {"runtime": rt, "remaining_budget": float(remaining),
                "budget_exhausted": remaining <= 0, "models_tried": list(models_tried)}

        write_artifacts(out_dir, run_id, {"model": model},
                        {k: v for k, v in metrics.items() if k not in ("gap",)},
                        {"experiment_id": run_id, "model_name": model, "model_type": "ml",
                         "hyperparameters": {}, "preprocessing": {
                             "missing_value_strategy": "mean", "scaling": "standard",
                             "encoding": "none", "imbalance_strategy": "none",
                             "feature_selection": "none"},
                         "early_stopping": False, "early_stopping_patience": 5})

        sel = _selected(run_id, iter_num, next_act, next_params, trigger, conf,
                        round(r2 * 0.95, 3), 0.68, round(r2 * 0.88, 3))
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
            "confidence": conf, "primary_metric": r2, "train_val_gap": gap,
            "accuracy": None, "f1_score": None, "precision": None, "recall": None,
            "mse": mse_val, "r2": r2, "slope": slope, "volatility": volatility,
            "runtime": rt, "active_signals": ", ".join(active),
        })
        runs_rows.append({
            "experiment_id": run_id, "iteration": iter_num,
            "primary_metric": "r2", "best_score": best_score,
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

        print(f"  Result   : primary={r2:.4f} | runtime={rt:.2f}s")
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
    write_model_py(out_dir, run_id, "GradientBoostingRegressor", best_score, {})
    pdf = write_report_pdf(out_dir, run_id, DATASET, comparison_rows)

    print(f"  Report    : {pdf}")
    print(f"  Comparison: {out_dir / 'comparison.csv'}")
    print(f"  Model.py  : {out_dir / 'model.py'}\n")
    answer = input("  Download best model files (model.pkl + model.py)? [y/N]: ").strip().lower()
    if answer == "y":
        print(f"  Files saved at: {out_dir / 'artifacts' / run_id / 'model.pkl'} and {out_dir / 'model.py'}")


if __name__ == "__main__":
    run()
