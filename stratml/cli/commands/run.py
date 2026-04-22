"""cli/commands/run.py — `stratml run` command."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_pipeline(args) -> None:
    from stratml.cli.config import resolve, validate_config

    config = resolve(args.config, args)
    try:
        validate_config(config)
    except ValueError as e:
        print(f"\n  [Invalid config] {e}\n")
        sys.exit(1)

    d   = config["dataset"]
    e   = config["execution"]
    sep = "-" * 44

    if args.dry_run:
        _print_dry_run(config, d, e, sep)
        return

    print(f"\n  AutoML Pipeline Starting\n  {sep}")
    print(f"  Mode    : {config['mode']}")
    print(f"  Dataset : {d['path']}")
    print(f"  Target  : {d['target_column']}")
    print(f"  Budget  : {e['max_iterations']} iterations\n  {sep}")

    from stratml.decision.engine import DecisionEngine
    from stratml.orchestration.orchestrator import ExecutionOrchestrator
    from stratml.execution.schemas import SplitConfig

    dl_cfg     = config.get("deep_learning", {})
    dl_enabled = dl_cfg.get("enabled", False)

    if dl_enabled:
        arch           = dl_cfg.get("architecture", "MLP").upper()
        allowed_models = [arch if arch in ("MLP", "CNN1D", "RNN") else "MLP"]
        dl_hyperparams = {
            "architecture":  arch,
            "epochs":        dl_cfg.get("epochs", 20),
            "learning_rate": dl_cfg.get("learning_rate", 0.001),
            "batch_size":    dl_cfg.get("batch_size", 32),
        }
    else:
        allowed_models = (
            config.get("intermediate", {}).get("allowed_models")
            or config.get("expert", {}).get("allowed_models")
            or None
        )
        dl_hyperparams = None

    dataset_name = Path(d["path"]).stem
    run_id       = f"{dataset_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir      = Path("outputs") / run_id

    engine = DecisionEngine(
        max_iterations=e["max_iterations"],
        time_budget=e.get("timeout_per_run"),
        allowed_models=allowed_models,
        run_id=run_id,
        dl_hyperparams=dl_hyperparams,
    )

    ExecutionOrchestrator(
        send_profile=engine.receive_profile,
        send_result=engine.receive_result,
        split_config=SplitConfig(
            method=config["split"]["method"],
            test_size=config["split"]["test_size"],
        ),
        time_budget=e.get("timeout_per_run"),
        run_id=run_id,
        log=print,
    ).run(d["path"], d["target_column"])

    print(f"\n  Run complete.\n  {sep}")
    print(f"  Run ID  : {run_id}")
    print(f"  Output  : {out_dir}\n  {sep}\n")

    _generate_report(run_id, dataset_name, out_dir)
    _prompt_download(out_dir)


def _print_dry_run(config: dict, d: dict, e: dict, sep: str) -> None:
    print(f"\n  Dry Run - Resolved Config\n  {sep}")
    print(f"  Mode          : {config['mode']}")
    print(f"  Dataset       : {d['path']}")
    print(f"  Target        : {d['target_column']}")
    print(f"  Max iterations: {e['max_iterations']}")
    print(f"  Timeout/run   : {e['timeout_per_run']}s")
    print(f"  MLflow        : {config['logging']['enable_mlflow']}")
    dl = config.get("deep_learning", {})
    if dl.get("enabled"):
        print(f"  DL Mode       : enabled  |  Arch: {dl.get('architecture')}  |  Epochs: {dl.get('epochs')}")
    print(f"  {sep}\n")


def _generate_report(run_id: str, dataset_name: str, out_dir: Path) -> None:
    from stratml.reporting.report_generator import generate_report, generate_model_script
    try:
        pdf     = generate_report(run_id=run_id, dataset_name=dataset_name, output_dir=out_dir)
        log_dir = out_dir / "decision_logs"
        records = [
            json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(log_dir.glob(f"{run_id}_*.json"))
        ]
        script = generate_model_script(run_id=run_id, output_dir=out_dir, records=records)
        print(f"  Report    : {pdf}")
        print(f"  Comparison: {out_dir / 'comparison.csv'}")
        print(f"  Model.py  : {script}\n")
    except Exception as ex:
        print(f"  [Warning] Report generation failed: {ex}\n")


def _prompt_download(out_dir: Path) -> None:
    model_pkl    = out_dir / "artifacts" / "model.pkl"
    model_script = out_dir / "artifacts" / "model.py"
    if not model_pkl.exists():
        return
    if input("  Download best model files? [y/N]: ").strip().lower() == "y":
        shutil.copy2(model_pkl, Path.cwd() / "best_model.pkl")
        if model_script.exists():
            shutil.copy2(model_script, Path.cwd() / "model.py")
        print("  Saved: best_model.pkl + model.py\n")
