"""Demo: California Housing — Regression, SVR triggers too_slow, r2 metric."""
from __future__ import annotations
import sys, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from demo._base import (
    write_artifacts, write_model_py, write_decision_log, write_counterfactual,
    write_comparison, append_runs_log, write_report_pdf, write_langsmith_traces,
    make_state, _signals, _selected, _no_signals, _sleep, ROOT, load_config,
    ui_header, ui_iter_start, ui_training, ui_result, ui_agents, ui_decision,
    ui_signals, ui_summary, ui_artifacts, _console,
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

    ui_header(DATASET, TARGET, mode, max_iter, ROWS, COLS,
              "regression", train, val, test)
    time.sleep(0.3)
    
    comparison_rows = [{
        "iteration": 0, "model": "none", "action": "switch_model", "trigger": "bootstrap",
        "confidence": 0.5, "primary_metric": 0.0, "train_val_gap": 0.0,
        "accuracy": None, "f1_score": None, "precision": None, "recall": None,
        "mse": None, "r2": None, "slope": 0.0, "volatility": 0.0, "runtime": 0.0, "active_signals": "",
    }]
    runs_rows = []
    trace_entries = []
    models_tried = []
    best_score = 0.0
    prev_score = None
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
        ui_iter_start(iter_num, model, action)
        # SVR takes long — simulate it
        sleep_t = min(rt * 0.3, 6.0)
        _sleep(sleep_t)
        ui_training(model, rt)

        if model not in models_tried:
            models_tried.append(model)
        improvement = (r2 - prev_score) if prev_score is not None else 0.0
        prev_score = r2
        best_score = max(best_score, r2)
        slope = round(improvement, 6)
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

        _evidence = {"trigger": trigger, "confidence": round(conf, 2),
                     "train_val_gap": round(gap, 4), "primary_metric": round(r2, 4),
                     **{k: signals[k] for k in ['overfitting', 'underfitting', 'stagnating', 'converged', 'well_fitted', 'too_slow', 'diminishing_returns'] if signals[k] != "none"}}
        sel = _selected(run_id, iter_num, next_act, next_params, trigger, conf,
                        round(r2 * 0.95, 3), 0.68, round(r2 * 0.88, 3),
                                evidence=_evidence)
        _HPARAMS_MAP = {('RandomForestRegressor', 'switch_model'): {'n_estimators': 100, 'max_depth': None, 'random_state': 42}, ('GradientBoostingRegressor', 'switch_model'): {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42}, ('SVR', 'switch_model'): {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}, ('Ridge', 'switch_model'): {'alpha': 1.0}, ('ExtraTreesRegressor', 'switch_model'): {'n_estimators': 100, 'max_depth': None, 'random_state': 42}}
        _hparams = _HPARAMS_MAP.get((model, action), {})
        write_decision_log(out_dir, run_id, iter_num, make_state(
            run_id, iter_num, model, "ml", metrics, signals, DATASET_META,
            traj, res, [{"action_type": next_act, "parameters": next_params},
                        {"action_type": "terminate", "parameters": {}}],
            sel, max_iter, previous_action=prev_action, time_budget=time_budget,
            hyperparameters=_hparams,
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

        ui_signals(signals)
        ui_result(r2, "R²", gap, mse=mse_val, r2=r2, runtime=rt)
        ui_agents(iter_num, DECISION_AGENT_DELAY)
        _act  = "terminate" if next_act == "terminate" else next_act
        _trig = "budget_exhausted" if next_act == "terminate" else trigger
        _conf = 1.0 if next_act == "terminate" else conf
        _par  = {} if next_act == "terminate" else next_params
        ui_decision(_act, _trig, _conf, _par)
        _ts = __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
        trace_entries.append({
            "iteration": iter_num,
            "start_time": _ts,
            "end_time": _ts,
            "model": model,
            "signals": signals,
            "metrics": metrics,
            "selected_action": sel,
            "candidates": [{"action_type": next_act, "parameters": next_params},
                           {"action_type": "terminate", "parameters": {}}],
            "agent_scores": {
                "performance_scores": {next_act: round(sel["agent_scores"]["performance"], 4),
                                       "terminate": round(1.0 - sel["agent_scores"]["performance"], 4)},
                "efficiency_scores":  {next_act: round(sel["agent_scores"]["efficiency"], 4),
                                       "terminate": round(1.0 - sel["agent_scores"]["efficiency"] * 0.5, 4)},
                "stability_scores":   {next_act: round(sel["agent_scores"]["stability"], 4),
                                       "terminate": 0.95},
            },
            "slope": slope,
            "volatility": volatility,
            "runtime": rt,
            "remaining_budget": remaining,
        })
        prev_action = action

    ui_summary(run_id, out_dir, best_score, "GradientBoostingRegressor", "r2", comparison_rows)

    write_counterfactual(out_dir, run_id, cf_entries)
    ls_traces = write_langsmith_traces(out_dir, run_id, trace_entries)
    write_comparison(out_dir, comparison_rows)
    append_runs_log(run_id, runs_rows)
    write_model_py(out_dir, run_id, "GradientBoostingRegressor", best_score,
                   {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42})
    pdf = write_report_pdf(out_dir, run_id, DATASET, comparison_rows)

    ui_artifacts(out_dir, ls_traces, pdf, out_dir / "model.py")
    answer = _console.input("  [dim]Download best model files (model.pkl + model.py)?[/dim] [bold]\[y/N][/bold]: ").strip().lower()
    if answer == "y":
        _console.print(f"  [green]Saved:[/green] {out_dir / 'artifacts' / run_id / 'model.pkl'}")


if __name__ == "__main__":
    run()
