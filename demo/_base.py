"""
demo/_base.py
Shared helpers for all demo scripts: artifact writing, timing simulation,
decision log writing, comparison files, model.py, runs/ population.
"""
from __future__ import annotations
import csv
import json
import time
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich import box
from rich.rule import Rule

_console = Console(highlight=False)

_SIG_STYLE = {"strong": "bold red", "weak": "yellow", "none": "dim white"}
_ACTION_STYLE = {
    "switch_model":            "bold cyan",
    "modify_regularization":   "bold yellow",
    "increase_model_capacity": "bold green",
    "decrease_model_capacity": "bold yellow",
    "add_preprocessing":       "bold magenta",
    "change_optimizer":        "bold blue",
    "terminate":               "bold red",
}

def ui_header(dataset, target, mode, budget, rows, cols, problem, train, val, test):
    _console.print()
    _console.print(Panel.fit(
        f"[bold white]StratML AutoML[/bold white]  [dim]-[/dim]  "
        f"[cyan]{dataset}[/cyan]  [dim]->[/dim]  [green]{target}[/green]",
        border_style="bright_blue", padding=(0, 2),
    ))
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="dim"); grid.add_column()
    grid.add_row("Mode",    f"[bold]{mode}[/bold]")
    grid.add_row("Problem", f"[bold]{problem}[/bold]")
    grid.add_row("Shape",   f"[white]{rows:,} rows x {cols} cols[/white]")
    grid.add_row("Split",   f"train [green]{train:,}[/green]  val [yellow]{val:,}[/yellow]  test [red]{test:,}[/red]")
    grid.add_row("Budget",  f"[bold]{budget}[/bold] iterations")
    _console.print(grid)
    _console.print()

def ui_iter_start(iter_num, model, action):
    _console.rule(
        f"[bold]Iteration {iter_num}[/bold]  [dim]-[/dim]  "
        f"[cyan]{model}[/cyan]  [dim]via[/dim]  [" + _ACTION_STYLE.get(action, "white") + f"]{action}[/" + _ACTION_STYLE.get(action, "white") + "]",
        style="bright_blue",
    )

def ui_training(model, runtime):
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[dim]Training[/dim] [cyan]{task.description}[/cyan]"),
        BarColumn(bar_width=28, style="cyan", complete_style="green"),
        TimeElapsedColumn(),
        console=_console, transient=True,
    ) as p:
        t = p.add_task(model, total=100)
        step = max(runtime * 6 / 20, 0.01)
        for _ in range(20):
            time.sleep(step)
            p.advance(t, 5)
    _console.print(f"  [dim]Trained in[/dim] [green]{runtime:.2f}s[/green]")

def ui_training_dl(runtime):
    """Used after a tqdm epoch bar -- just prints the elapsed time line."""
    _console.print(f"  [dim]Trained in[/dim] [green]{runtime:.2f}s[/green]")

def ui_result(primary, metric_name, gap, f1=None, mse=None, r2=None, runtime=0.0):
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="dim", width=18); grid.add_column()
    grid.add_row(metric_name, f"[bold green]{primary:.4f}[/bold green]")
    if f1  is not None: grid.add_row("F1",      f"[white]{f1:.4f}[/white]")
    if r2  is not None: grid.add_row("R2",     f"[white]{r2:.4f}[/white]")
    if mse is not None: grid.add_row("MSE",     f"[white]{mse:.4f}[/white]")
    gap_style = "red" if gap > 0.15 else "yellow" if gap > 0.05 else "green"
    grid.add_row("Train/Val gap", f"[{gap_style}]{gap:.4f}[/{gap_style}]")
    grid.add_row("Runtime",       f"[dim]{runtime:.2f}s[/dim]")
    _console.print(grid)

def ui_agents(iter_num, delay):
    agents = (
        [("Evaluator",   "auditing previous decision")] if iter_num > 1 else []
    ) + [
        ("StateBuilder", "extracting signals from metrics"),
        ("Perf Agent",   "scoring candidates on accuracy gain"),
        ("Eff Agent",    "scoring candidates on compute cost"),
        ("Stab Agent",   "scoring candidates on training risk"),
        ("Coordinator",  "deliberating over agent scores"),
        ("Selector",     "applying policy + budget constraints"),
    ]
    for name, task in agents:
        _console.print(f"  [bold cyan]{name:<14}[/bold cyan] [dim]{task}[/dim]")
        time.sleep(delay)

def ui_decision(action, trigger, confidence, params):
    style = _ACTION_STYLE.get(action, "bold white")
    conf_style = "green" if confidence >= 0.7 else "yellow" if confidence >= 0.5 else "red"
    _console.print(
        f"  [dim]Decision[/dim]  [{style}]{action}[/{style}]"
        f"  [dim]trigger=[/dim][yellow]{trigger}[/yellow]"
        f"  [dim]conf=[/dim][{conf_style}]{confidence:.2f}[/{conf_style}]"
    )
    if params:
        _console.print(f"  [dim]params  [/dim]  [white]{params}[/white]")
    _console.print()

def ui_signals(signals):
    active = {k: v for k, v in signals.items() if not k.endswith("_confidence") and v != "none"}
    if not active: return
    parts = [f"[{_SIG_STYLE.get(v, 'white')}]{k}={v}[/{_SIG_STYLE.get(v, 'white')}]" for k, v in active.items()]
    _console.print("  [dim]Signals[/dim]  " + "  ".join(parts))

def ui_summary(run_id, out_dir, best_score, best_model, metric_name, comparison_rows):
    _console.print()
    _console.rule("[bold green]Run Complete[/bold green]", style="green")
    _console.print()
    t = Table(title="Experiment History", box=box.ROUNDED, border_style="bright_blue",
              header_style="bold white on bright_blue", show_lines=False)
    t.add_column("Iter",  justify="right", style="dim",        width=5)
    t.add_column("Model", style="cyan",                        width=32)
    t.add_column("Action", style="yellow",                     width=26)
    t.add_column(metric_name.upper(), justify="right", style="bold green", width=8)
    t.add_column("Gap",   justify="right",                     width=7)
    t.add_column("Runtime", justify="right", style="dim",      width=9)
    for r in comparison_rows[1:]:
        primary = r.get("primary_metric") or 0.0
        gap     = r.get("train_val_gap") or 0.0
        gs = "red" if gap > 0.15 else "yellow" if gap > 0.05 else "green"
        t.add_row(str(r["iteration"]), r["model"], r["action"],
                  f"{primary:.4f}", f"[{gs}]{gap:.4f}[/{gs}]", f"{r.get('runtime', 0):.2f}s")
    _console.print(t)
    _console.print()
    _console.print(Panel.fit(
        f"[bold green]Best Model[/bold green]  [cyan]{best_model}[/cyan]\n"
        f"[dim]{metric_name}[/dim]  [bold white]{best_score:.4f}[/bold white]",
        border_style="green", padding=(0, 2),
    ))
    _console.print()
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=14); grid.add_column(style="white")
    grid.add_row("Run ID", f"[bold]{run_id}[/bold]")
    grid.add_row("Output", str(out_dir))
    _console.print(grid)
    _console.print()


def ui_artifacts(out_dir, ls_traces, pdf, model_py):
    """Print a tidy artifacts panel after the run summary."""
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="dim", width=12)
    grid.add_column(style="dim")
    grid.add_row("LangSmith", str(ls_traces))
    grid.add_row("Report",    str(pdf))
    grid.add_row("Model.py",  str(model_py))
    _console.print(Panel(grid, title="[bold white]Artifacts[/bold white]",
                         border_style="bright_blue", padding=(0, 1)))
    _console.print()


ROOT = Path(__file__).parents[1]
RUNS_LOG = ROOT / "runs" / "decision_logs" / "decision_dataset.csv"


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(path: Path | None = None) -> dict:
    """Load config.yaml; return a flat-ish dict of demo-relevant values."""
    try:
        import yaml
    except ImportError:
        return {}
    cfg_path = path or ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        raw = yaml.safe_load(f) or {}
    return {
        "mode":             raw.get("mode", "beginner"),
        "max_iterations":   raw.get("execution", {}).get("max_iterations", 5),
        "timeout_per_run":  raw.get("execution", {}).get("timeout_per_run", 300),
        "random_seed":      raw.get("execution", {}).get("random_seed", 42),
        "test_size":        raw.get("split", {}).get("test_size", 0.2),
        "split_method":     raw.get("split", {}).get("method", "stratified"),
        "enable_mlflow":    raw.get("logging", {}).get("enable_mlflow", False),
        "enable_tensorboard": raw.get("logging", {}).get("enable_tensorboard", False),
        "log_level":        raw.get("logging", {}).get("log_level", "info"),
    }


# ── Timing ────────────────────────────────────────────────────────────────────

def _sleep(s: float) -> float:
    time.sleep(s)
    return s


# ── Artifact helpers ──────────────────────────────────────────────────────────

def write_artifacts(out_dir: Path, run_id: str, model_obj, metrics: dict, config: dict) -> None:
    art = out_dir / "artifacts" / run_id
    art.mkdir(parents=True, exist_ok=True)
    (art / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (art / "config.json").write_text(json.dumps(config, indent=2))
    ext = ".pt" if config.get("model_type") == "dl" else ".pkl"
    with open(art / f"model{ext}", "wb") as f:
        pickle.dump(model_obj, f)


def write_model_py(out_dir: Path, run_id: str, model_name: str, best_metric: float, hyperparams: dict, model_type: str = "ml") -> Path:
    p = out_dir / "model.py"
    ext = ".pt" if model_type == "dl" else ".pkl"
    loader = "torch.load" if model_type == "dl" else "joblib.load"
    loader_import = "import torch" if model_type == "dl" else "import joblib"
    p.write_text(f'''"""
model.py — Auto-generated by StratML
Run ID  : {run_id}
Model   : {model_name}
Metric  : {best_metric}
"""

{loader_import}
import numpy as np
from pathlib import Path

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(r"outputs/{run_id}/artifacts/{run_id}/model{ext}")
model = {loader}(MODEL_PATH)

# ── Hyperparameters used ──────────────────────────────────────────────────────
hyperparameters = {hyperparams!r}

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(X):
    return model.predict(np.array(X))

def predict_proba(X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(np.array(X))
    raise NotImplementedError(f"{{type(model).__name__}} does not support predict_proba")

if __name__ == "__main__":
    print(f"Model : {{type(model).__name__}}")
    print(f"Params: {{model.get_params()}}")
    print("Ready. Call predict(X) or predict_proba(X).")
''', encoding="utf-8")
    return p


def write_decision_log(out_dir: Path, run_id: str, iteration: int, record: dict) -> None:
    log_dir = out_dir / "decision_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fname = log_dir / f"{run_id}_{iteration:04d}.json"
    fname.write_text(json.dumps(record, indent=2))


def write_counterfactual(out_dir: Path, run_id: str, entries: list[dict]) -> None:
    """Build counterfactual log: for each iteration, show rejected actions and why."""
    log_dir = out_dir / "decision_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for f in sorted(log_dir.glob(f"{run_id}_*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        it = rec.get("iteration", 0)
        if it == 0:
            continue
        sel = rec["selected_action"]
        chosen = sel["action_type"]
        scores = sel.get("agent_scores", {})
        signals = {k: v for k, v in rec["state_snapshot"]["signals"].items()
                   if not k.endswith("_confidence") and v != "none"}
        primary = rec["state_snapshot"]["metrics"].get("primary", 0)
        gap = rec["state_snapshot"]["metrics"].get("train_val_gap", 0)
        # Build richer rejected set: signal-driven alternatives not chosen
        is_dl = rec["state_snapshot"]["model"].get("model_type", "ml") == "dl"
        _ML_ALTS = {
            "overfitting":         ["modify_regularization", "decrease_model_capacity", "switch_model"],
            "underfitting":        ["increase_model_capacity", "switch_model", "modify_regularization"],
            "stagnating":          ["switch_model", "modify_regularization", "terminate"],
            "converged":           ["terminate", "switch_model"],
            "well_fitted":         ["terminate", "switch_model"],
            "too_slow":            ["switch_model", "decrease_model_capacity"],
            "diminishing_returns": ["terminate", "switch_model"],
        }
        _DL_ALTS = {
            "overfitting":         ["switch_model", "tune_hyperparameters"],
            "underfitting":        ["switch_model", "tune_hyperparameters"],
            "stagnating":          ["switch_model", "terminate"],
            "converged":           ["terminate", "switch_model"],
            "well_fitted":         ["terminate", "switch_model"],
            "too_slow":            ["switch_model"],
            "diminishing_returns": ["terminate", "switch_model"],
        }
        _ALTS = _DL_ALTS if is_dl else _ML_ALTS
        alts = []
        for sig in signals:
            alts += _ALTS.get(sig, [])
        rejected = [a for a in dict.fromkeys(alts) if a != chosen]
        entry = {
            "experiment_id": run_id,
            "iteration": it,
            "chosen_action": chosen,
            "chosen_params": sel.get("parameters", {}),
            "confidence": sel.get("confidence", 0),
            "trigger": sel.get("reason", {}).get("trigger", ""),
            "agent_scores": scores,
            "active_signals": signals,
            "primary_metric": primary,
            "train_val_gap": gap,
            "rejected_actions": rejected,
            "rejection_reason": (
                f"Rejected {rejected} "
                f"because signals={list(signals.keys())}, "
                f"gap={gap:.4f}, metric={primary:.4f}, "
                f"agent scores perf={scores.get('performance',0):.2f} "
                f"eff={scores.get('efficiency',0):.2f} "
                f"stab={scores.get('stability',0):.2f} "
                f"favoured '{chosen}'"
            ),
        }
        lines.append(json.dumps(entry))
    (log_dir / "counterfactual_log.jsonl").write_text("\n".join(lines) + "\n")


def write_comparison(out_dir: Path, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(out_dir / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "comparison.json").write_text(json.dumps(rows, indent=2))


def append_runs_log(run_id: str, rows: list[dict]) -> None:
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RUNS_LOG.exists()
    with open(RUNS_LOG, "a", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(rows)


def write_report_pdf(out_dir: Path, run_id: str, dataset_name: str, rows: list[dict]) -> Path:
    """Generate a professional PDF report with charts via pdf_builder."""
    pdf_path = out_dir / "report.pdf"
    try:
        import json as _json
        from stratml.reporting.pdf_builder import build_pdf
        log_dir = out_dir / "decision_logs"
        records = [
            _json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(log_dir.glob(f"{run_id}_*.json"))
            if f.stat().st_size > 0
        ]
        records = [r for r in records if r.get("iteration", 0) > 0]
        build_pdf(run_id, dataset_name, records, out_dir, pdf_path)
        return pdf_path
    except Exception:
        pdf_path.write_bytes(b"")
        return pdf_path


def _write_report_pdf_legacy(out_dir: Path, run_id: str, dataset_name: str, rows: list[dict]) -> Path:
    """Legacy fallback — tables only, no charts."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
        )
        from reportlab.lib import colors

        NAVY   = colors.HexColor("#1a2744")
        BLUE   = colors.HexColor("#2563eb")
        LIGHT  = colors.HexColor("#f0f4ff")
        ALT    = colors.HexColor("#f8fafc")
        BORDER = colors.HexColor("#cbd5e1")
        GREEN  = colors.HexColor("#16a34a")
        RED    = colors.HexColor("#dc2626")
        AMBER  = colors.HexColor("#d97706")
        SLATE  = colors.HexColor("#64748b")
        PAD    = 7

        def base_ts():
            return TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
                ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
                ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, ALT]),
                ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
                ("TOPPADDING",    (0, 0), (-1, -1), PAD),
                ("BOTTOMPADDING", (0, 0), (-1, -1), PAD),
                ("LEFTPADDING",   (0, 0), (-1, -1), PAD),
                ("RIGHTPADDING",  (0, 0), (-1, -1), PAD),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ])

        pdf_path = out_dir / "report.pdf"
        doc = SimpleDocTemplate(
            str(pdf_path), pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2.5*cm, bottomMargin=2*cm,
        )
        s = getSampleStyleSheet()
        title_s = ParagraphStyle("_t",  parent=s["Title"],   fontSize=22, textColor=NAVY, spaceAfter=2, leading=28)
        sub_s   = ParagraphStyle("_s",  parent=s["Normal"],  fontSize=14, textColor=BLUE, spaceAfter=6)
        h2_s    = ParagraphStyle("_h2", parent=s["Heading2"],fontSize=11, textColor=NAVY, spaceBefore=18, spaceAfter=6)
        body_s  = ParagraphStyle("_b",  parent=s["Normal"],  fontSize=9,  textColor=colors.HexColor("#334155"), leading=14, spaceAfter=3)
        cell_s  = ParagraphStyle("_c",  parent=s["Normal"],  fontSize=8,  textColor=colors.HexColor("#1e293b"), leading=11)
        mono_s  = ParagraphStyle("_m",  parent=s["Normal"],  fontName="Courier", fontSize=7.5, textColor=NAVY, leading=11)

        def section(title):
            story.append(Paragraph(title, h2_s))
            story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=6))

        data_rows = [r for r in rows if r.get("iteration", 0) > 0]
        story = []

        # ── Cover ─────────────────────────────────────────────────────────────
        story.append(Paragraph("StratML", title_s))
        story.append(Paragraph("Experiment Report", sub_s))
        story.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=8))
        meta_t = Table(
            [["Run ID", run_id, "Dataset", dataset_name],
             ["Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
              "Iterations", str(len(data_rows))]],
            colWidths=[2.5*cm, 7*cm, 2.5*cm, 3.5*cm],
        )
        meta_t.setStyle(TableStyle([
            ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME",      (2,0), (2,-1), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8.5),
            ("TEXTCOLOR",     (0,0), (0,-1), SLATE),
            ("TEXTCOLOR",     (2,0), (2,-1), SLATE),
            ("TEXTCOLOR",     (1,0), (1,-1), NAVY),
            ("TEXTCOLOR",     (3,0), (3,-1), NAVY),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(meta_t)
        story.append(Spacer(1, 0.5*cm))

        # ── KPI cards ─────────────────────────────────────────────────────────
        if data_rows:
            best = max(data_rows, key=lambda r: r.get("primary_metric") or 0)
            gap  = best.get("train_val_gap", 0) or 0
            gc   = "#16a34a" if gap < 0.05 else ("#d97706" if gap < 0.15 else "#dc2626")
            f1v  = best.get("f1_score")
            kpi_t = Table([[
                Paragraph('<font color="#64748b" size="7">BEST MODEL</font><br/>'
                          f'<font color="#1a2744" size="10"><b>{best.get("model","—")}</b></font>', cell_s),
                Paragraph('<font color="#64748b" size="7">PRIMARY METRIC</font><br/>'
                          f'<font color="#16a34a" size="14"><b>{best.get("primary_metric",0):.4f}</b></font>', cell_s),
                Paragraph('<font color="#64748b" size="7">TRAIN / VAL GAP</font><br/>'
                          f'<font color="{gc}" size="12"><b>{gap:.4f}</b></font>', cell_s),
                Paragraph('<font color="#64748b" size="7">F1 SCORE</font><br/>'
                          f'<font color="#1a2744" size="10"><b>{f"{f1v:.4f}" if f1v else "—"}</b></font>', cell_s),
            ]], colWidths=[4.2*cm, 4.2*cm, 4.2*cm, 4.2*cm])
            kpi_t.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), LIGHT),
                ("BOX",           (0,0), (0,-1),  0.5, BORDER),
                ("BOX",           (1,0), (1,-1),  0.5, BORDER),
                ("BOX",           (2,0), (2,-1),  0.5, BORDER),
                ("BOX",           (3,0), (3,-1),  0.5, BORDER),
                ("TOPPADDING",    (0,0), (-1,-1), 10),
                ("BOTTOMPADDING", (0,0), (-1,-1), 10),
                ("LEFTPADDING",   (0,0), (-1,-1), 10),
                ("RIGHTPADDING",  (0,0), (-1,-1), 10),
            ]))
            story.append(kpi_t)
            story.append(Spacer(1, 0.4*cm))

        # ── Full metrics ───────────────────────────────────────────────────────
        section("Metrics per Iteration")
        mhdr = ["#", "Model", "Primary", "Accuracy", "Precision", "Recall", "F1", "MSE / R²", "Gap"]
        mdata = [mhdr]
        for r in data_rows:
            mse = r.get("mse")
            r2  = r.get("r2")
            mse_r2 = f"{mse:.4f}" if mse is not None else (f"{r2:.4f}" if r2 is not None else "—")
            mdata.append([
                str(r.get("iteration", "")),
                str(r.get("model", ""))[:22],
                f'{r.get("primary_metric", 0) or 0:.4f}',
                f'{r.get("accuracy", 0) or 0:.4f}'  if r.get("accuracy")  is not None else "—",
                f'{r.get("precision", 0) or 0:.4f}' if r.get("precision") is not None else "—",
                f'{r.get("recall", 0) or 0:.4f}'    if r.get("recall")    is not None else "—",
                f'{r.get("f1_score", 0) or 0:.4f}'  if r.get("f1_score")  is not None else "—",
                mse_r2,
                f'{r.get("train_val_gap", 0) or 0:.4f}',
            ])
        mts = base_ts()
        mts.add("ALIGN", (0,0), (0,-1), "CENTER")
        mts.add("ALIGN", (2,0), (-1,-1), "RIGHT")
        for i, r in enumerate(data_rows, start=1):
            gap = r.get("train_val_gap", 0) or 0
            gc = GREEN if gap < 0.05 else (AMBER if gap < 0.15 else RED)
            mts.add("TEXTCOLOR", (8, i), (8, i), gc)
            mts.add("FONTNAME",  (8, i), (8, i), "Helvetica-Bold")
        mt = Table(mdata, colWidths=[0.7*cm, 3.8*cm, 1.9*cm, 2*cm, 2*cm, 1.9*cm, 1.9*cm, 2*cm, 1.8*cm], repeatRows=1)
        mt.setStyle(mts)
        story.append(mt)
        story.append(Spacer(1, 0.3*cm))

        # ── Decision trace (narrative) ─────────────────────────────────────────
        section("Decision Trace")
        for r in data_rows:
            conf = r.get("confidence", 0) or 0
            cc   = "#16a34a" if conf >= 0.7 else ("#d97706" if conf >= 0.5 else "#dc2626")
            sig  = r.get("active_signals", "") or ""
            line = (
                f'<font color="#64748b">Iter {r.get("iteration","?")}  </font>'
                f'<b>{r.get("model","")}</b>'
                f'<font color="#64748b">  →  </font>'
                f'<font color="#2563eb"><b>{r.get("action","")}</b></font>'
                f'<font color="#64748b">  trigger=</font>{r.get("trigger","")}'
                f'<font color="#64748b">  conf=</font><font color="{cc}"><b>{conf:.2f}</b></font>'
            )
            if sig:
                line += f'<font color="#64748b">  signals=</font><i>{sig}</i>'
            story.append(Paragraph(line, body_s))

        doc.build(story)
        return pdf_path
    except Exception:
        pdf_path = out_dir / "report.pdf"
        pdf_path.write_bytes(b"")
        return pdf_path

def make_state(run_id, iteration, model_name, model_type, metrics, signals,
               dataset, trajectory, resources, candidates, selected,
               max_iterations, previous_action=None, time_budget=300.0, hyperparameters=None):
    ts = datetime.now(timezone.utc).isoformat()
    primary = metrics.get("accuracy") or metrics.get("r2") or 0.0
    return {
        "experiment_id": run_id,
        "iteration": iteration,
        "timestamp": ts,
        "state_snapshot": {
            "meta": {"experiment_id": run_id, "iteration": iteration, "timestamp": ts},
            "objective": {"primary_metric": "accuracy" if metrics.get("accuracy") is not None else "r2",
                          "optimization_goal": "maximize"},
            "metrics": {
                "primary": primary,
                "secondary": {
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1_score": metrics.get("f1_score"),
                    "mse": metrics.get("mse"),
                    "rmse": metrics.get("rmse"),
                    "r2": metrics.get("r2"),
                },
                "train_val_gap": metrics.get("gap", 0.0),
            },
            "trajectory": trajectory,
            "dataset": dataset,
            "model": {
                "model_name": model_name,
                "model_type": model_type,
                "hyperparameters": hyperparameters or {},
                "complexity_hint": "low",
                "runtime": resources.get("runtime", 0.0),
                "convergence_epoch": iteration,
                "early_stopped": None,
            },
            "generalization": {
                "train_loss": metrics.get("train_loss", 0.0),
                "validation_loss": metrics.get("validation_loss", 0.0),
                "gap": metrics.get("gap", 0.0),
            },
            "resources": {
                "runtime": resources.get("runtime", 0.0),
                "gpu_used": False,
                "cpu_time": resources.get("runtime", 0.0),
                "remaining_budget": resources.get("remaining_budget", 0.0),
                "budget_exhausted": resources.get("budget_exhausted", False),
            },
            "search": {
                "models_tried": resources.get("models_tried", []),
                "unique_models_count": len(resources.get("models_tried", [])),
                "repeated_configs": 0,
            },
            "signals": signals,
            "uncertainty": {"prediction_variance": None, "confidence": None},
            "action_context": {
                "previous_action": previous_action,
                "previous_action_success": None,
                "action_effect_magnitude": None,
            },
            "constraints": {
                "allowed_models": [],
                "max_iterations": max_iterations,
                "time_budget": time_budget,
            },
        },
        "candidate_actions": candidates,
        "selected_action": selected,
    }


def _no_signals():
    keys = ["underfitting","overfitting","well_fitted","converged","stagnating",
            "diverging","unstable_training","high_variance","too_slow",
            "plateau_detected","diminishing_returns"]
    return {k: "none" for k in keys} | {k+"_confidence": 0.0 for k in keys}


def _signals(**overrides):
    s = _no_signals()
    s.update(overrides)
    return s


def _selected(run_id, iteration, action_type, params, trigger, confidence,
              perf, eff, stab, evidence=None, preprocessing=None):
    return {
        "experiment_id": run_id,
        "iteration": iteration,
        "action_type": action_type,
        "parameters": params,
        "preprocessing": preprocessing or {
            "missing_value_strategy": "mean",
            "scaling": "standard",
            "encoding": "onehot",
            "imbalance_strategy": "none",
            "feature_selection": "none",
        },
        "bootstrap_context": None,
        "expected_gain": 0.05,
        "expected_cost": 0.5,
        "confidence": confidence,
        "agent_scores": {"performance": perf, "efficiency": eff, "stability": stab},
        "reason": {
            "trigger": trigger,
            "evidence": evidence or {"trigger": trigger, "confidence": round(confidence, 2)},
            "source": "rule",
        },
    }


# ── LangSmith trace writer ────────────────────────────────────────────────────

def write_langsmith_traces(out_dir: Path, run_id: str, trace_entries: list[dict]) -> Path:
    """
    Write a LangSmith-compatible traces file to decision_logs/langsmith_traces.jsonl.
    Each entry is one decision cycle: one parent 'chain' run with 6 child 'tool' runs
    (one per agent), matching the real pipeline's agent names and I/O structure.
    """
    import uuid as _uuid

    log_dir = out_dir / "decision_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = log_dir / "langsmith_traces.jsonl"

    lines = []
    for entry in trace_entries:
        iter_num   = entry["iteration"]
        ts_start   = entry["start_time"]
        ts_end     = entry["end_time"]
        signals    = entry["signals"]
        metrics    = entry["metrics"]
        action     = entry["selected_action"]
        candidates = entry["candidates"]
        agent_scores = entry["agent_scores"]
        gap        = metrics.get("gap", 0.0)
        primary    = metrics.get("accuracy") or metrics.get("r2") or 0.0
        model      = entry["model"]

        parent_id = str(_uuid.uuid4())

        # ── Parent: DecisionEngine._decide ───────────────────────────────────
        parent = {
            "id": parent_id,
            "name": "DecisionEngine._decide",
            "run_type": "chain",
            "session_name": "stratml",
            "start_time": ts_start,
            "end_time": ts_end,
            "inputs": {
                "experiment_id": run_id,
                "iteration": iter_num,
                "model": model,
                "primary_metric": round(primary, 4),
                "train_val_gap": round(gap, 4),
                "signals": {k: v for k, v in signals.items() if not k.endswith("_confidence") and v != "none"},
                "candidates": [c["action_type"] for c in candidates],
            },
            "outputs": {
                "selected_action": action["action_type"],
                "parameters": action["parameters"],
                "confidence": action["confidence"],
                "trigger": action["reason"]["trigger"],
            },
            "error": None,
            "extra": {"metadata": {"run_id": run_id, "iteration": iter_num}},
        }
        lines.append(json.dumps(parent))

        # ── Child agents ──────────────────────────────────────────────────────
        agents = [
            ("StateBuilder.extract_signals", "tool",
             {"metrics": {"primary": round(primary, 4), "gap": round(gap, 4)},
              "model": model},
             {"signals": {k: v for k, v in signals.items() if not k.endswith("_confidence") and v != "none"}}),

            ("PerformanceAgent.score", "tool",
             {"fitting_state": {k: signals[k] for k in ["underfitting","overfitting","well_fitted"] if signals[k] != "none"},
              "trajectory_slope": round(entry.get("slope", 0.0), 4),
              "candidates": [c["action_type"] for c in candidates]},
             {"scores": agent_scores.get("performance_scores", {})}),

            ("EfficiencyAgent.score", "tool",
             {"runtime": round(entry.get("runtime", 0.0), 2),
              "remaining_budget": entry.get("remaining_budget", 0),
              "candidates": [c["action_type"] for c in candidates]},
             {"scores": agent_scores.get("efficiency_scores", {})}),

            ("StabilityAgent.score", "tool",
             {"volatility": round(entry.get("volatility", 0.0), 4),
              "gap": round(gap, 4),
              "candidates": [c["action_type"] for c in candidates]},
             {"scores": agent_scores.get("stability_scores", {})}),

            ("CoordinatorAgent.rank", "tool",
             {"perf_scores": agent_scores.get("performance_scores", {}),
              "eff_scores":  agent_scores.get("efficiency_scores", {}),
              "stab_scores": agent_scores.get("stability_scores", {})},
             {"ranked": [{"action": c["action_type"],
                          "composite_score": round(
                              agent_scores.get("performance_scores", {}).get(c["action_type"], 0.5) * 0.5 +
                              agent_scores.get("efficiency_scores",  {}).get(c["action_type"], 0.5) * 0.25 +
                              agent_scores.get("stability_scores",   {}).get(c["action_type"], 0.5) * 0.25, 4)}
                         for c in candidates]}),

            ("ActionSelector.select", "tool",
             {"ranked_top": action["action_type"],
              "budget_remaining": entry.get("remaining_budget", 0),
              "confidence": action["confidence"]},
             {"action": action["action_type"],
              "parameters": action["parameters"],
              "source": "rule"}),
        ]

        for name, run_type, inputs, outputs in agents:
            child = {
                "id": str(_uuid.uuid4()),
                "parent_run_id": parent_id,
                "name": name,
                "run_type": run_type,
                "session_name": "stratml",
                "start_time": ts_start,
                "end_time": ts_end,
                "inputs": inputs,
                "outputs": outputs,
                "error": None,
                "extra": {"metadata": {"run_id": run_id, "iteration": iter_num}},
            }
            lines.append(json.dumps(child))

    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_file
