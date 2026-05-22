"""Demo: MNIST DL — MLP flat → CNN2D → ResNet18 (too_slow) → MobileNetV3."""
from __future__ import annotations
import sys, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from demo._base import (
    write_artifacts, write_model_py, write_decision_log, write_counterfactual,
    write_comparison, append_runs_log, write_report_pdf, write_langsmith_traces,
    make_state, _signals, _selected, _no_signals, _sleep, ROOT, load_config,
    ui_header, ui_iter_start, ui_training_dl, ui_result, ui_agents, ui_decision,
    ui_signals, ui_summary, ui_artifacts, _console,
)
from tqdm import tqdm

DATASET   = "mnist_dl"
TARGET    = "label"
MAX_ITER  = 4
DECISION_AGENT_DELAY = 2.0
ROWS, COLS = 70000, 785   # 784 pixels + 1 label col
TRAIN, VAL, TEST = 42000, 14000, 14000

# Decision chain per doc:
# MLP flat 784 (underfitting vs spatial models, 0.965)
# → CNN2D (0.991, overfitting)
# → ResNet18 pretrained (0.993, too_slow)
# → MobileNetV3 (0.992, converged)
ITERS = [
    # MLP: decent but underfitting vs spatial models
    ("MLP",          "switch_model",    "bootstrap",   0.50, 0.9651, 0.9648, 0.9655, 0.9645, 0.04, 0.035, 0.0,   18.7,
     {"underfitting": "weak", "underfitting_confidence": 0.58},
     "switch_model", {"model_name": "CNN2D"}),

    # CNN2D: 0.991 but overfitting
    ("CNN2D",        "switch_model",    "underfitting", 0.65, 0.9912, 0.9909, 0.9915, 0.9906, 0.01, 0.009, 0.0,  54.3,
     {"overfitting": "weak", "overfitting_confidence": 0.63},
     "switch_model", {"model_name": "ResNet18"}),

    # ResNet18: 0.993 but too_slow
    ("ResNet18",     "switch_model",    "overfitting",  0.78, 0.9931, 0.9928, 0.9934, 0.9925, 0.005, 0.007, 0.002, 312.4,
     {"too_slow": "strong", "too_slow_confidence": 0.85},
     "switch_model", {"model_name": "MobileNetV3"}),

    # MobileNetV3: 0.992, converged, fast
    ("MobileNetV3",  "switch_model",    "too_slow",     0.87, 0.9921, 0.9918, 0.9924, 0.9915, 0.006, 0.008, 0.002, 89.1,
     {"converged": "weak", "converged_confidence": 0.66, "well_fitted": "strong", "well_fitted_confidence": 0.73},
     "terminate", {}),
]

DATASET_META = {
    "num_samples": ROWS, "num_features": COLS - 1,
    "feature_to_sample_ratio": round((COLS - 1) / ROWS, 6),
    "missing_ratio": 0.0,
    "class_distribution": {str(i): 7000 for i in range(10)},
    "imbalance_ratio": 1.0,
}

_EPOCHS_MAP = {"MLP": 20, "CNN2D": 15, "ResNet18": 10, "MobileNetV3": 12}
_MODALITY_MAP = {"MLP": "tabular", "CNN2D": "vision", "ResNet18": "vision", "MobileNetV3": "vision"}


def _ui_epoch_bar(model: str, epochs: int, runtime: float) -> None:
    step = runtime / max(epochs, 1)
    bar = tqdm(
        range(epochs),
        desc=f"  {model}",
        unit="epoch",
        dynamic_ncols=True,
        file=sys.stderr,
        bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        leave=False,
    )
    for _ in bar:
        time.sleep(step)
    bar.close()


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
        "mse": None, "r2": None, "slope": 0.0, "volatility": 0.0, "runtime": 0.0,
        "active_signals": "",
    }]
    runs_rows = []
    trace_entries = []
    models_tried = []
    best_score = 0.0
    prev_score = None
    prev_action = None

    write_decision_log(out_dir, run_id, 0, make_state(
        run_id, 0, "none", "dl", {"accuracy": None, "gap": 0.0},
        _no_signals(), DATASET_META,
        {"history_length": 0, "improvement_rate": 0.0, "slope": 0.0, "volatility": 0.0,
         "best_score": 0.0, "mean_score": 0.0, "steps_since_improvement": 0, "trend": "stagnating"},
        {"runtime": 0.0, "remaining_budget": float(max_iter), "budget_exhausted": False, "models_tried": []},
        [{"action_type": "switch_model", "parameters": {"model_name": "MLP"}},
         {"action_type": "switch_model", "parameters": {"model_name": "CNN2D"}}],
        _selected(run_id, 0, "switch_model", {"model_name": "MLP"}, "bootstrap", 0.5, 0.05, 0.5, 1.0),
        max_iter,
    ))
    cf_entries = [{"iteration": 0, "action_type": "switch_model",
                   "parameters": {"model_name": "MLP"}, "confidence": 0.5}]

    for i, (model, action, trigger, conf, primary, f1, prec, rec,
            tl, vl, gap, rt, sig_kw, next_act, next_params) in enumerate(ITERS[:max_iter]):
        iter_num = i + 1
        if i == max_iter - 1 and next_act != "terminate":
            next_act, next_params = "terminate", {}

        ui_iter_start(iter_num, model, action)
        epochs = _EPOCHS_MAP.get(model, 15)
        _ui_epoch_bar(model, epochs, rt)
        ui_training_dl(rt)

        if model not in models_tried:
            models_tried.append(model)
        improvement = (primary - prev_score) if prev_score is not None else 0.0
        prev_score = primary
        best_score = max(best_score, primary)
        slope = round(improvement, 6)
        volatility = round(abs(slope) * 0.5, 6)
        remaining = max_iter - iter_num
        modality = _MODALITY_MAP.get(model, "tabular")

        metrics = {"accuracy": primary, "f1_score": f1, "precision": prec, "recall": rec,
                   "train_loss": tl, "validation_loss": vl, "gap": gap,
                   "mse": None, "rmse": None, "r2": None}
        signals = _signals(**sig_kw)
        active = [k for k in ["underfitting", "overfitting", "too_slow", "converged", "well_fitted"]
                  if signals.get(k, "none") != "none"]

        traj = {"history_length": iter_num, "improvement_rate": round(improvement, 6),
                "slope": slope, "volatility": volatility, "best_score": best_score,
                "mean_score": round(primary, 6), "steps_since_improvement": 0,
                "trend": "improving" if improvement > 0 else "stagnating"}
        res  = {"runtime": rt, "remaining_budget": float(remaining),
                "budget_exhausted": remaining <= 0, "models_tried": list(models_tried)}

        _img_shape = [1, 28, 28] if modality == "vision" else None
        _hp = {"architecture": model, "modality": modality, "epochs": epochs}
        if _img_shape:
            _hp["image_shape"] = _img_shape

        write_artifacts(out_dir, run_id, {"model": model},
                        {k: v for k, v in metrics.items() if k != "gap"},
                        {"experiment_id": run_id, "model_name": model, "model_type": "dl",
                         "hyperparameters": _hp,
                         "preprocessing": {"missing_value_strategy": "none", "scaling": "minmax",
                                           "encoding": "none", "imbalance_strategy": "none",
                                           "feature_selection": "none"},
                         "early_stopping": True, "early_stopping_patience": 5})

        _evidence = {"trigger": trigger, "confidence": round(conf, 2),
                     "train_val_gap": round(gap, 4), "primary_metric": round(primary, 4),
                     **{k: signals[k] for k in ["underfitting", "overfitting", "too_slow",
                                                 "converged", "well_fitted"]
                        if signals[k] != "none"}}
        sel = _selected(run_id, iter_num, next_act, next_params, trigger, conf,
                        round(primary * 0.95, 3), 0.65, round(primary * 0.9, 3),
                        evidence=_evidence)

        write_decision_log(out_dir, run_id, iter_num, make_state(
            run_id, iter_num, model, "dl", metrics, signals, DATASET_META,
            traj, res,
            [{"action_type": next_act, "parameters": next_params},
             {"action_type": "terminate", "parameters": {}}],
            sel, max_iter, previous_action=prev_action, time_budget=time_budget,
            hyperparameters=_hp,
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

        _ts = datetime.now(timezone.utc).isoformat()
        trace_entries.append({
            "iteration": iter_num, "start_time": _ts, "end_time": _ts,
            "model": model, "signals": signals, "metrics": metrics,
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
            "slope": slope, "volatility": volatility, "runtime": rt,
            "remaining_budget": remaining,
        })
        prev_action = action

    best_model = max(comparison_rows[1:], key=lambda r: r["primary_metric"])["model"]
    ui_summary(run_id, out_dir, best_score, best_model, "accuracy", comparison_rows)

    write_counterfactual(out_dir, run_id, cf_entries)
    ls_traces = write_langsmith_traces(out_dir, run_id, trace_entries)
    write_comparison(out_dir, comparison_rows)
    append_runs_log(run_id, runs_rows)
    write_model_py(out_dir, run_id, best_model, best_score,
                   {"architecture": best_model, "modality": "vision",
                    "image_shape": [1, 28, 28], "frozen": False}, model_type="dl")
    pdf = write_report_pdf(out_dir, run_id, DATASET, comparison_rows)

    ui_artifacts(out_dir, ls_traces, pdf, out_dir / "model.py")
    answer = _console.input(
        "  [dim]Download best model files (model.pkl + model.py)?[/dim] [bold]\\[y/N][/bold]: "
    ).strip().lower()
    if answer == "y":
        _console.print(f"  [green]Saved:[/green] {out_dir / 'artifacts' / run_id / 'model.pkl'}")


if __name__ == "__main__":
    run()
