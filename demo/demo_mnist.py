"""Demo: MNIST Tabular — High-dimensional, too_slow fires on GBM."""
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

DATASET   = "mnist"
TARGET    = "label"
MAX_ITER  = 3
DECISION_AGENT_DELAY = 2.5  # DL, 70k samples
ROWS, COLS = 70000, 785
TRAIN, VAL, TEST = 42000, 14000, 14000

ITERS = [
    # RF: tl=0.0 vs vl=0.033 → tight gap, diminishing returns (verified)
    ("RandomForestClassifier",    "switch_model",    "bootstrap",          0.50, 0.9649, 0.9649, 0.965,  0.9649, 0.0,    0.0334, 0.0334, 5.24,
     {"diminishing_returns": "weak", "diminishing_returns_confidence": 0.52},
     "switch_model", {"model_name": "GradientBoostingClassifier"}),

    # GBM: lower accuracy AND 1245s → too_slow fires (verified: real runtime)
    ("GradientBoostingClassifier","switch_model",    "diminishing_returns",0.61, 0.9396, 0.9395, 0.9396, 0.9396, 0.0346, 0.0534, 0.0188, 1245.43,
     {"too_slow": "strong", "too_slow_confidence": 0.88},
     "switch_model", {"model_name": "ExtraTreesClassifier"}),

    # ExtraTrees: best accuracy, tight gap → converged (verified)
    ("ExtraTreesClassifier",      "switch_model",    "too_slow",           0.88, 0.9672, 0.9672, 0.9672, 0.9672, 0.0,    0.0301, 0.0301, 8.12,
     {"converged": "weak", "converged_confidence": 0.62, "well_fitted": "strong", "well_fitted_confidence": 0.71},
     "terminate", {}),
]

DATASET_META = {
    "num_samples": ROWS, "num_features": COLS - 1,
    "feature_to_sample_ratio": round((COLS-1)/ROWS, 6),
    "missing_ratio": 0.0,
    "class_distribution": {str(i): 7000 for i in range(10)},
    "imbalance_ratio": 1.0,
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
              "classification", train, val, test)
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
        run_id, 0, "none", "ml", {"accuracy": None, "gap": 0.0},
        _no_signals(), DATASET_META,
        {"history_length": 0, "improvement_rate": 0.0, "slope": 0.0, "volatility": 0.0,
         "best_score": 0.0, "mean_score": 0.0, "steps_since_improvement": 0, "trend": "stagnating"},
        {"runtime": 0.0, "remaining_budget": float(max_iter), "budget_exhausted": False, "models_tried": []},
        [{"action_type": "switch_model", "parameters": {"model_name": "RandomForestClassifier"}},
         {"action_type": "switch_model", "parameters": {"model_name": "ExtraTreesClassifier"}}],
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
        ui_iter_start(iter_num, model, action)
        _sleep(min(rt * 0.1, 15))
        ui_training(model, rt)

        if model not in models_tried:
            models_tried.append(model)
        improvement = (primary - prev_score) if prev_score is not None else 0.0
        prev_score = primary
        best_score = max(best_score, primary)
        slope = round(improvement, 6)
        volatility = round(abs(slope) * 0.5, 6)
        remaining = max_iter - iter_num

        metrics = {"accuracy": primary, "f1_score": f1, "precision": prec, "recall": rec,
                   "train_loss": tl, "validation_loss": vl, "gap": gap,
                   "mse": None, "rmse": None, "r2": None}
        signals = _signals(**sig_kw)
        active = [k for k in ["too_slow","diminishing_returns","converged","well_fitted"]
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
                             "encoding": "none", "imbalance_strategy": "none",
                             "feature_selection": "none"},
                         "early_stopping": False, "early_stopping_patience": 5})

        _evidence = {"trigger": trigger, "confidence": round(conf, 2),
                     "train_val_gap": round(gap, 4), "primary_metric": round(primary, 4),
                     **{k: signals[k] for k in ['overfitting', 'underfitting', 'stagnating', 'converged', 'well_fitted', 'too_slow', 'diminishing_returns'] if signals[k] != "none"}}
        sel = _selected(run_id, iter_num, next_act, next_params, trigger, conf,
                        round(primary * 0.95, 3), 0.65, round(primary * 0.9, 3),
                                evidence=_evidence)
        _HPARAMS_MAP = {('RandomForestClassifier', 'switch_model'): {'n_estimators': 100, 'max_depth': None, 'random_state': 42}, ('GradientBoostingClassifier', 'switch_model'): {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42}, ('ExtraTreesClassifier', 'switch_model'): {'n_estimators': 100, 'max_depth': None, 'random_state': 42}}
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
            "num_samples": ROWS, "num_features": COLS - 1, "missing_ratio": 0.0,
            "runtime": rt, "remaining_budget": float(remaining),
            "action_type": next_act, "action_params": str(next_params),
            "predicted_gain": 0.05, "observed_gain": round(improvement, 6),
            "normalized_gain": round(improvement / 0.1, 6) if improvement else 0.0,
        })

        ui_signals(signals)
        ui_result(primary, "Accuracy", gap, f1=f1, runtime=rt)
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

    ui_summary(run_id, out_dir, best_score, "ExtraTreesClassifier", "accuracy", comparison_rows)

    write_counterfactual(out_dir, run_id, cf_entries)
    ls_traces = write_langsmith_traces(out_dir, run_id, trace_entries)
    write_comparison(out_dir, comparison_rows)
    append_runs_log(run_id, runs_rows)
    write_model_py(out_dir, run_id, "ExtraTreesClassifier", best_score,
                   {"n_estimators": 100, "max_depth": None, "random_state": 42})
    pdf = write_report_pdf(out_dir, run_id, DATASET, comparison_rows)

    ui_artifacts(out_dir, ls_traces, pdf, out_dir / "model.py")
    answer = _console.input("  [dim]Download best model files (model.pkl + model.py)?[/dim] [bold]\[y/N][/bold]: ").strip().lower()
    if answer == "y":
        _console.print(f"  [green]Saved:[/green] {out_dir / 'artifacts' / run_id / 'model.pkl'}")


if __name__ == "__main__":
    run()
