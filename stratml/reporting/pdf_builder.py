"""
reporting/pdf_builder.py
------------------------
PDF generation using ReportLab.
Reads decision log records and produces a formatted A4 report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


def build_pdf(
    run_id: str,
    dataset_name: str,
    records: list[dict],
    output_dir: Path,
    pdf_path: Path,
) -> None:
    doc    = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    h1  = ParagraphStyle("h1",  parent=styles["Title"],    fontSize=18, spaceAfter=4)
    h2  = ParagraphStyle("h2",  parent=styles["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=4)
    bod = styles["BodyText"]
    mon = ParagraphStyle("mon", parent=bod, fontName="Courier", fontSize=8)

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story += [
        Paragraph("StratML — Execution Report", h1),
        Paragraph(f"Run ID: <b>{run_id}</b>  |  Dataset: <b>{dataset_name}</b>", bod),
        Paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", bod),
        HRFlowable(width="100%", thickness=1, color=colors.grey, spaceAfter=8),
        Paragraph("Summary", h2),
        Paragraph(f"Iterations logged: <b>{len(records)}</b>", bod),
        Paragraph(f"Output directory: <font name='Courier'>{output_dir}</font>", bod),
    ]

    # ── Iteration table ───────────────────────────────────────────────────────
    story.append(Paragraph("Iteration Comparison", h2))
    tdata       = [["Iter", "Model", "Action", "Trigger", "Conf", "Primary", "Gap", "Signals"]]
    best_metric = 0.0
    best_model  = "—"
    best_iter   = 0

    for rec in records:
        it      = rec.get("iteration", "?")
        m       = rec["state_snapshot"]["metrics"]
        sig     = rec["state_snapshot"]["signals"]
        active  = [k for k, v in sig.items() if not k.endswith("_confidence") and v != "none"]
        primary = m.get("primary", 0.0) or 0.0
        tdata.append([
            str(it),
            rec["state_snapshot"]["model"]["model_name"],
            rec["selected_action"]["action_type"],
            rec["selected_action"]["reason"]["trigger"],
            f"{rec['selected_action'].get('confidence') or 0:.2f}",
            f"{primary:.4f}",
            f"{m.get('train_val_gap') or 0:.4f}",
            ", ".join(active) or "—",
        ])
        if primary > best_metric:
            best_metric, best_model, best_iter = primary, rec["state_snapshot"]["model"]["model_name"], it

    _table_style = [
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1,-1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1,-1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("GRID",           (0, 0), (-1,-1), 0.3, colors.grey),
        ("VALIGN",         (0, 0), (-1,-1), "TOP"),
    ]
    story.append(Table(tdata, colWidths=[1*cm,4*cm,3.5*cm,3*cm,1.5*cm,2.2*cm,2*cm,4*cm],
                       repeatRows=1, style=TableStyle(_table_style)))

    # ── Model comparison ──────────────────────────────────────────────────────
    story.append(Paragraph("Model Comparison", h2))
    model_map: dict[str, dict] = {}
    for rec in records:
        mn  = rec["state_snapshot"]["model"]["model_name"]
        pri = rec["state_snapshot"]["metrics"].get("primary") or 0.0
        sec = rec["state_snapshot"]["metrics"].get("secondary", {})
        if mn not in model_map or pri > model_map[mn]["primary"]:
            model_map[mn] = {"primary": pri, "f1": sec.get("f1_score"), "acc": sec.get("accuracy")}

    mdata = [["Model", "Best Primary", "Accuracy", "F1 Score"]]
    for mn, v in model_map.items():
        mdata.append([mn, f"{v['primary']:.4f}",
                      f"{v['acc']:.4f}" if v["acc"] is not None else "—",
                      f"{v['f1']:.4f}"  if v["f1"]  is not None else "—"])
    story.append(Table(mdata, colWidths=[5*cm,3.5*cm,3.5*cm,3.5*cm], repeatRows=1,
                       style=TableStyle(_table_style)))

    # ── Best model ────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 0.4*cm),
        Paragraph("Best Model", h2),
        Paragraph(f"Model: <b>{best_model}</b>  |  Iteration: <b>{best_iter}</b>  |  "
                  f"Primary metric: <b>{best_metric:.4f}</b>", bod),
    ]
    model_file = output_dir / "artifacts" / "model.pkl"
    if model_file.exists():
        story.append(Paragraph(f"Saved at: <font name='Courier'>{model_file}</font>", mon))

    metrics_file = output_dir / "artifacts" / "metrics.json"
    if metrics_file.exists():
        story.append(Paragraph("Final Metrics", h2))
        for k, v in json.loads(metrics_file.read_text()).items():
            if v is not None:
                story.append(Paragraph(f"<font name='Courier'>{k}: {v}</font>", mon))

    doc.build(story)
