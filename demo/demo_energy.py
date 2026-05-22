"""Demo: Appliances Energy — DL pipeline, MLP→CNN1D→RNN, early stopping, TensorBoard."""
from __future__ import annotations
import sys, time, json
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

DATASET   = "energydata_complete"
TARGET    = "Appliances"
MAX_ITER  = 4
DECISION_AGENT_DELAY = 2.2
ROWS, COLS = 19735, 29
TRAIN, VAL, TEST = 12630, 3158, 3947

# DL architectures: (arch, action, trigger, conf, r2, tl, vl, gap, rt, epochs_run, early_stopped, sig_kw, next_act, next_params)
ITERS = [
    ("MLP",   "switch_model",           "bootstrap",          0.50, 0.5812, 0.412, 0.681, 0.269, 14.32, 30, False,
     {"underfitting": "strong", "underfitting_confidence": 0.76},
     "increase_model_capacity", {"scale": 1.5, "architecture": "MLP"}),

    ("MLP",   "increase_model_capacity","underfitting",       0.68, 0.6234, 0.371, 0.598, 0.227, 18.71, 30, False,
     {"underfitting": "weak", "underfitting_confidence": 0.55},
     "switch_model", {"model_name": "CNN1D", "architecture": "CNN1D"}),

    ("CNN1D", "switch_model",           "underfitting",       0.61, 0.6489, 0.348, 0.561, 0.213, 22.14, 30, False,
     {"diminishing_returns": "weak", "diminishing_returns_confidence": 0.49},
     "switch_model", {"model_name": "RNN", "architecture": "RNN"}),

    ("RNN",   "switch_model",           "diminishing_returns",0.57, 0.6671, 0.341, 0.573, 0.198, 31.87, 18, True,
     {"converged": "weak", "converged_confidence": 0.51, "well_fitted": "weak", "well_fitted_confidence": 0.46},
     "terminate", {}),
]

DATASET_META = {
    "num_samples": ROWS, "num_features": COLS - 1,
    "feature_to_sample_ratio": round((COLS-1)/ROWS, 6),
    "missing_ratio": 0.0,
    "class_distribution": None,
    "imbalance_ratio": None,
}

# Synthetic TensorBoard-style training curves
TRAIN_CURVES = {
    "MLP_1":   [0.681, 0.612, 0.558, 0.511, 0.478, 0.451, 0.429, 0.412, 0.398, 0.387,
                0.378, 0.371, 0.365, 0.360, 0.356, 0.353, 0.350, 0.348, 0.346, 0.344,
                0.343, 0.342, 0.341, 0.340, 0.340, 0.339, 0.339, 0.338, 0.338, 0.338],
    "MLP_2":   [0.598, 0.541, 0.491, 0.449, 0.414, 0.385, 0.361, 0.341, 0.324, 0.310,
                0.298, 0.288, 0.280, 0.273, 0.267, 0.262, 0.258, 0.254, 0.251, 0.249,
                0.247, 0.245, 0.244, 0.243, 0.242, 0.241, 0.241, 0.240, 0.240, 0.240],
    "CNN1D":   [0.561, 0.498, 0.443, 0.396, 0.357, 0.324, 0.297, 0.274, 0.255, 0.239,
                0.226, 0.215, 0.206, 0.198, 0.192, 0.187, 0.183, 0.179, 0.176, 0.174,
                0.172, 0.170, 0.169, 0.168, 0.167, 0.166, 0.166, 0.165, 0.165, 0.165],
    "RNN":     [0.621, 0.558, 0.501, 0.452, 0.410, 0.374, 0.343, 0.317, 0.295, 0.277,
                0.262, 0.249, 0.239, 0.231, 0.225, 0.220, 0.217, 0.215],  # early stopped at 18
}


def _write_tb_logs(run_id: str, arch: str, curve_key: str) -> Path:
    """Write a minimal TensorBoard-compatible events stub (JSON summary)."""
    tb_dir = ROOT / "runs" / "tensorboard" / run_id / arch
    tb_dir.mkdir(parents=True, exist_ok=True)
    curve = TRAIN_CURVES[curve_key]
    summary = {"arch": arch, "epochs": len(curve), "train_loss_curve": curve}
    (tb_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return tb_dir


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
    curve_keys = ["MLP_1", "MLP_2", "CNN1D", "RNN"]

    write_decision_log(out_dir, run_id, 0, make_state(
        run_id, 0, "none", "dl", {"r2": None, "gap": 0.0},
        _no_signals(), DATASET_META,
        {"history_length": 0, "improvement_rate": 0.0, "slope": 0.0, "volatility": 0.0,
         "best_score": 0.0, "mean_score": 0.0, "steps_since_improvement": 0, "trend": "stagnating"},
        {"runtime": 0.0, "remaining_budget": float(max_iter), "budget_exhausted": False, "models_tried": []},
        [{"action_type": "switch_model", "parameters": {"model_name": "MLP", "architecture": "MLP"}},
         {"action_type": "switch_model", "parameters": {"model_name": "CNN1D", "architecture": "CNN1D"}}],
        _selected(run_id, 0, "switch_model", {"model_name": "MLP", "architecture": "MLP"},
                  "bootstrap", 0.5, 0.05, 0.5, 1.0),
        max_iter,
    ))
    cf_entries = [{"iteration": 0, "action_type": "switch_model",
                   "parameters": {"model_name": "MLP", "architecture": "MLP"}, "confidence": 0.5}]

    for i, (arch, action, trigger, conf, r2, tl, vl, gap, rt, epochs, early_stopped,
            sig_kw, next_act, next_params) in enumerate(ITERS[:max_iter]):
        iter_num = i + 1
        if i == max_iter - 1 and next_act != "terminate":
            next_act, next_params = "terminate", {}
        ui_iter_start(iter_num, arch, action)
        _sleep(rt)
        if early_stopped:
            _console.print(f"  [yellow]Early stopping at epoch {epochs}/30[/yellow]")
        ui_training(arch, rt)

        tb_dir = _write_tb_logs(run_id, arch, curve_keys[i])
        print(f"  TensorBoard: {tb_dir}")

        if arch not in models_tried:
            models_tried.append(arch)
        improvement = (r2 - prev_score) if prev_score is not None else 0.0
        prev_score = r2
        best_score = max(best_score, r2)
        slope = round(improvement, 6)
        volatility = round(abs(slope) * 0.6, 6)
        remaining = max_iter - iter_num

        mse_val = round((1 - r2) * 1.8, 4)
        metrics = {"accuracy": None, "f1_score": None, "precision": None, "recall": None,
                   "train_loss": tl, "validation_loss": vl, "gap": gap,
                   "mse": mse_val, "rmse": round(mse_val**0.5, 4), "r2": r2}
        signals = _signals(**sig_kw)
        active = [k for k in ["underfitting","diminishing_returns","converged","well_fitted"]
                  if signals.get(k, "none") != "none"]

        traj = {"history_length": iter_num, "improvement_rate": round(improvement, 6),
                "slope": slope, "volatility": volatility, "best_score": best_score,
                "mean_score": round(r2, 6), "steps_since_improvement": 0,
                "trend": "improving" if improvement > 0 else "stagnating"}
        res  = {"runtime": rt, "remaining_budget": float(remaining),
                "budget_exhausted": remaining <= 0, "models_tried": list(models_tried)}

        write_artifacts(out_dir, run_id, {"arch": arch, "early_stopped": early_stopped},
                        {k: v for k, v in metrics.items() if k not in ("gap",)},
                        {"experiment_id": run_id, "model_name": arch, "model_type": "dl",
                         "hyperparameters": {"architecture": arch, "epochs": epochs,
                                             "batch_size": 64, "learning_rate": 0.001,
                                             "task": "regression"},
                         "preprocessing": {"missing_value_strategy": "mean", "scaling": "standard",
                                           "encoding": "none", "imbalance_strategy": "none",
                                           "feature_selection": "none"},
                         "early_stopping": early_stopped, "early_stopping_patience": 5})

        _evidence = {"trigger": trigger, "confidence": round(conf, 2),
                     "train_val_gap": round(gap, 4), "primary_metric": round(r2, 4),
                     **{k: signals[k] for k in ['overfitting', 'underfitting', 'stagnating', 'converged', 'well_fitted', 'too_slow', 'diminishing_returns'] if signals[k] != "none"}}
        sel = _selected(run_id, iter_num, next_act, next_params, trigger, conf,
                        round(r2 * 0.95, 3), 0.72, round(r2 * 0.88, 3),
                                evidence=_evidence)
        _HPARAMS_MAP = {('MLP', 'switch_model'): {'architecture': 'MLP', 'hidden_layers': [64, 32], 'epochs': 30, 'batch_size': 64, 'learning_rate': 0.001}, ('MLP', 'increase_model_capacity'): {'architecture': 'MLP', 'hidden_layers': [128, 64, 32], 'epochs': 30, 'batch_size': 64, 'learning_rate': 0.001}, ('CNN1D', 'switch_model'): {'architecture': 'CNN1D', 'filters': 64, 'kernel_size': 3, 'epochs': 30, 'batch_size': 64, 'learning_rate': 0.001}, ('RNN', 'switch_model'): {'architecture': 'RNN', 'hidden_size': 64, 'num_layers': 2, 'epochs': 18, 'batch_size': 64, 'learning_rate': 0.001}}
        _hparams = _HPARAMS_MAP.get((arch, action), {})
        write_decision_log(out_dir, run_id, iter_num, make_state(
            run_id, iter_num, arch, "dl", metrics, signals, DATASET_META,
            traj, res, [{"action_type": next_act, "parameters": next_params},
                        {"action_type": "terminate", "parameters": {}}],
            sel, max_iter, previous_action=prev_action, time_budget=time_budget,
            hyperparameters=_hparams,
        ))
        cf_entries.append({"iteration": iter_num, "action_type": next_act,
                           "parameters": next_params, "confidence": conf})

        comparison_rows.append({
            "iteration": iter_num, "model": arch, "action": next_act, "trigger": trigger,
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
            "model": arch,
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

    ui_summary(run_id, out_dir, best_score, "CNN1D", "r2", comparison_rows)

    write_counterfactual(out_dir, run_id, cf_entries)
    ls_traces = write_langsmith_traces(out_dir, run_id, trace_entries)
    write_comparison(out_dir, comparison_rows)
    append_runs_log(run_id, runs_rows)
    write_model_py(out_dir, run_id, "CNN1D", best_score, {"architecture": "CNN1D", "epochs": 30})
    pdf = write_report_pdf(out_dir, run_id, DATASET, comparison_rows)

    ui_artifacts(out_dir, ls_traces, pdf, out_dir / "model.py")
    answer = _console.input("  [dim]Download best model files (model.pkl + model.py)?[/dim] [bold]\[y/N][/bold]: ").strip().lower()
    if answer == "y":
        _console.print(f"  [green]Saved:[/green] {out_dir / 'artifacts' / run_id / 'model.pkl'}")


if __name__ == "__main__":
    run()
