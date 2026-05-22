"""Demo: Appliances Energy — DL pipeline, MLP→CNN1D→RNN, early stopping, TensorBoard."""
from __future__ import annotations
import sys, time, json
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from demo._base import (
    write_artifacts, write_model_py, write_decision_log, write_counterfactual,
    write_comparison, append_runs_log, write_report_pdf,
    make_state, _signals, _selected, _no_signals, _sleep, ROOT, load_config,
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

    ("RNN",   "switch_model",           "diminishing_returns",0.57, 0.6102, 0.389, 0.621, 0.232, 31.87, 18, True,
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
    print(f"  Decision [iter 0]: action=switch_model | params={{'model_name': 'MLP', 'architecture': 'MLP', 'epochs': 30, 'task': 'regression'}} | trigger=bootstrap")

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
        print(f"\n  --- Iteration {iter_num} ---")
        print(f"  Training : {arch} ({action}) ...")
        _sleep(rt * 0.4)
        if early_stopped:
            print(f"  Early stopping at epoch {epochs}/{30}")
        print(f"  Trained in {rt:.2f}s")

        tb_dir = _write_tb_logs(run_id, arch, curve_keys[i])
        print(f"  TensorBoard: {tb_dir}")

        if arch not in models_tried:
            models_tried.append(arch)
        improvement = r2 - best_score
        best_score = max(best_score, r2)
        slope = round(improvement / iter_num, 6)
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

        sel = _selected(run_id, iter_num, next_act, next_params, trigger, conf,
                        round(r2 * 0.95, 3), 0.72, round(r2 * 0.88, 3))
        write_decision_log(out_dir, run_id, iter_num, make_state(
            run_id, iter_num, arch, "dl", metrics, signals, DATASET_META,
            traj, res, [{"action_type": next_act, "parameters": next_params},
                        {"action_type": "terminate", "parameters": {}}],
            sel, max_iter, previous_action=prev_action, time_budget=time_budget,
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
    write_model_py(out_dir, run_id, "CNN1D", best_score, {"architecture": "CNN1D", "epochs": 30})
    pdf = write_report_pdf(out_dir, run_id, DATASET, comparison_rows)

    print(f"  Report    : {pdf}")
    print(f"  Comparison: {out_dir / 'comparison.csv'}")
    print(f"  Model.py  : {out_dir / 'model.py'}\n")
    answer = input("  Download best model files (model.pkl + model.py)? [y/N]: ").strip().lower()
    if answer == "y":
        print(f"  Files saved at: {out_dir / 'artifacts' / run_id / 'model.pkl'} and {out_dir / 'model.py'}")


if __name__ == "__main__":
    run()
