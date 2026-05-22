"""
reporting/pdf_builder.py
------------------------
PDF generation using ReportLab — professional A4 report.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1a2744")
BLUE   = colors.HexColor("#2563eb")
LIGHT  = colors.HexColor("#f0f4ff")
ALT    = colors.HexColor("#f8fafc")
BORDER = colors.HexColor("#cbd5e1")
GREEN  = colors.HexColor("#16a34a")
RED    = colors.HexColor("#dc2626")
AMBER  = colors.HexColor("#d97706")
SLATE  = colors.HexColor("#64748b")

# Generous cell padding so text never touches borders
_PAD = 7


def _ts():
    s = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("_t",  parent=s["Title"],   fontSize=22, textColor=NAVY, spaceAfter=2, leading=28),
        "sub":   ParagraphStyle("_s",  parent=s["Normal"],  fontSize=14, textColor=BLUE, spaceAfter=6),
        "h2":    ParagraphStyle("_h2", parent=s["Heading2"],fontSize=11, textColor=NAVY, spaceBefore=18, spaceAfter=6),
        "body":  ParagraphStyle("_b",  parent=s["Normal"],  fontSize=9,  textColor=colors.HexColor("#334155"), leading=14, spaceAfter=3),
        "cell":  ParagraphStyle("_c",  parent=s["Normal"],  fontSize=8,  textColor=colors.HexColor("#1e293b"), leading=11),
        "mono":  ParagraphStyle("_m",  parent=s["Normal"],  fontName="Courier", fontSize=7.5, textColor=NAVY, leading=11),
    }


def _base_style(header_cols=None):
    """Base TableStyle applied to every table."""
    cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, ALT]),
        ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), _PAD),
        ("BOTTOMPADDING", (0, 0), (-1, -1), _PAD),
        ("LEFTPADDING",   (0, 0), (-1, -1), _PAD),
        ("RIGHTPADDING",  (0, 0), (-1, -1), _PAD),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    return TableStyle(cmds)


def _section(story, title, st):
    story.append(Paragraph(title, st["h2"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=6))


# ── Section builders ──────────────────────────────────────────────────────────

def _build_header(run_id, dataset_name, n_iters):
    t = Table(
        [["Run ID",    run_id,    "Dataset",    dataset_name],
         ["Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
          "Iterations", str(n_iters)]],
        colWidths=[2.5*cm, 7*cm, 2.5*cm, 3.5*cm],
    )
    t.setStyle(TableStyle([
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR",     (0, 0), (0, -1), SLATE),
        ("TEXTCOLOR",     (2, 0), (2, -1), SLATE),
        ("TEXTCOLOR",     (1, 0), (1, -1), NAVY),
        ("TEXTCOLOR",     (3, 0), (3, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _build_kpi(best, cell_s):
    sec   = best["state_snapshot"]["metrics"].get("secondary", {})
    pri   = best["state_snapshot"]["metrics"].get("primary", 0) or 0
    gap   = best["state_snapshot"]["metrics"].get("train_val_gap", 0) or 0
    f1v   = sec.get("f1_score")
    model = best["state_snapshot"]["model"]["model_name"]
    gap_color = "#16a34a" if gap < 0.05 else ("#d97706" if gap < 0.15 else "#dc2626")

    t = Table([[
        Paragraph('<font color="#64748b" size="7">BEST MODEL</font><br/>'
                  f'<font color="#1a2744" size="10"><b>{model}</b></font>', cell_s),
        Paragraph('<font color="#64748b" size="7">PRIMARY METRIC</font><br/>'
                  f'<font color="#16a34a" size="14"><b>{pri:.4f}</b></font>', cell_s),
        Paragraph('<font color="#64748b" size="7">TRAIN / VAL GAP</font><br/>'
                  f'<font color="{gap_color}" size="12"><b>{gap:.4f}</b></font>', cell_s),
        Paragraph('<font color="#64748b" size="7">F1 SCORE</font><br/>'
                  f'<font color="#1a2744" size="10"><b>{f"{f1v:.4f}" if f1v else "—"}</b></font>', cell_s),
    ]], colWidths=[4.2*cm, 4.2*cm, 4.2*cm, 4.2*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT),
        ("BOX",           (0, 0), (0, -1),  0.5, BORDER),
        ("BOX",           (1, 0), (1, -1),  0.5, BORDER),
        ("BOX",           (2, 0), (2, -1),  0.5, BORDER),
        ("BOX",           (3, 0), (3, -1),  0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    return t


def _build_dataset(ds):
    """Dataset profile table from state_snapshot.dataset."""
    cd = ds.get("class_distribution", {})
    dist_str = "  |  ".join(f"{k}: {v}" for k, v in cd.items()) if cd else "—"
    rows = [
        ["Samples",          f'{ds.get("num_samples", "—"):,}' if isinstance(ds.get("num_samples"), int) else "—",
         "Features",         str(ds.get("num_features", "—"))],
        ["Missing ratio",    f'{ds.get("missing_ratio", 0):.1%}',
         "Imbalance ratio",  str(ds.get("imbalance_ratio", "—"))],
        ["Class distribution", dist_str, "", ""],
    ]
    t = Table(rows, colWidths=[3.5*cm, 4*cm, 3.5*cm, 4.8*cm])
    ts = TableStyle([
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR",     (0, 0), (0, -1), SLATE),
        ("TEXTCOLOR",     (2, 0), (2, -1), SLATE),
        ("TEXTCOLOR",     (1, 0), (1, -1), NAVY),
        ("TEXTCOLOR",     (3, 0), (3, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("SPAN",          (1, 2), (3, 2)),
        ("BACKGROUND",    (0, 0), (-1, -1), ALT),
        ("BOX",           (0, 0), (-1, -1), 0.4, BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, BORDER),
    ])
    t.setStyle(ts)
    return t


def _build_metrics_table(records):
    """Full metrics per iteration: Primary, Accuracy, Precision, Recall, F1, MSE/R2, Gap."""
    hdr = ["#", "Model", "Primary", "Accuracy", "Precision", "Recall", "F1", "MSE / R²", "Gap"]
    rows = [hdr]
    for rec in records:
        m   = rec["state_snapshot"]["metrics"]
        sec = m.get("secondary", {})
        pri = m.get("primary", 0) or 0
        gap = m.get("train_val_gap", 0) or 0
        acc = sec.get("accuracy")
        pre = sec.get("precision")
        rec_ = sec.get("recall")
        f1  = sec.get("f1_score")
        mse = sec.get("mse")
        r2  = sec.get("r2")
        mse_r2 = f"{mse:.4f}" if mse is not None else (f"{r2:.4f}" if r2 is not None else "—")
        rows.append([
            str(rec.get("iteration", "?")),
            rec["state_snapshot"]["model"]["model_name"][:22],
            f"{pri:.4f}",
            f"{acc:.4f}" if acc is not None else "—",
            f"{pre:.4f}" if pre is not None else "—",
            f"{rec_:.4f}" if rec_ is not None else "—",
            f"{f1:.4f}"  if f1  is not None else "—",
            mse_r2,
            f"{gap:.4f}",
        ])

    ts = _base_style()
    ts.add("ALIGN", (0, 0), (0, -1), "CENTER")
    ts.add("ALIGN", (2, 0), (-1, -1), "RIGHT")
    for i, rec in enumerate(records, start=1):
        gap = rec["state_snapshot"]["metrics"].get("train_val_gap", 0) or 0
        gc = GREEN if gap < 0.05 else (AMBER if gap < 0.15 else RED)
        ts.add("TEXTCOLOR", (8, i), (8, i), gc)
        ts.add("FONTNAME",  (8, i), (8, i), "Helvetica-Bold")

    t = Table(rows,
              colWidths=[0.7*cm, 3.8*cm, 1.9*cm, 2*cm, 2*cm, 1.9*cm, 1.9*cm, 2*cm, 1.8*cm],
              repeatRows=1)
    t.setStyle(ts)
    return t


def _build_hyperparams_table(records, mono_s):
    """One row per iteration showing the hyperparameters used."""
    hdr = ["#", "Model", "Hyperparameters"]
    rows = [hdr]
    for rec in records:
        hp = rec["state_snapshot"]["model"].get("hyperparameters") or {}
        hp_str = "  |  ".join(f"{k}={v}" for k, v in hp.items()) if hp else "defaults"
        rows.append([
            str(rec.get("iteration", "?")),
            rec["state_snapshot"]["model"]["model_name"][:22],
            Paragraph(hp_str, mono_s),
        ])

    ts = _base_style()
    ts.add("ALIGN", (0, 0), (0, -1), "CENTER")
    ts.add("VALIGN", (2, 1), (2, -1), "TOP")

    t = Table(rows, colWidths=[0.7*cm, 3.8*cm, 12.3*cm], repeatRows=1)
    t.setStyle(ts)
    return t


def _build_agent_table(records):
    """Agent scores + candidates considered per iteration."""
    hdr = ["#", "Action Taken", "Perf", "Eff", "Stab", "Candidates Considered", "Trigger", "Conf"]
    rows = [hdr]
    for rec in records:
        sel   = rec["selected_action"]
        ag    = sel.get("agent_scores", {})
        cands = rec.get("candidate_actions", [])
        cand_str = ", ".join(c["action_type"] for c in cands) if cands else "—"
        conf  = sel.get("confidence", 0) or 0
        rows.append([
            str(rec.get("iteration", "?")),
            sel["action_type"],
            f'{ag.get("performance", 0):.2f}' if ag.get("performance") is not None else "—",
            f'{ag.get("efficiency",  0):.2f}' if ag.get("efficiency")  is not None else "—",
            f'{ag.get("stability",   0):.2f}' if ag.get("stability")   is not None else "—",
            cand_str,
            sel["reason"]["trigger"],
            f"{conf:.2f}",
        ])

    ts = _base_style()
    ts.add("ALIGN", (0, 0), (0, -1), "CENTER")
    ts.add("ALIGN", (2, 0), (4, -1), "CENTER")
    ts.add("ALIGN", (7, 0), (7, -1), "CENTER")
    # Colour confidence
    for i, rec in enumerate(records, start=1):
        conf = rec["selected_action"].get("confidence", 0) or 0
        cc = GREEN if conf >= 0.7 else (AMBER if conf >= 0.5 else RED)
        ts.add("TEXTCOLOR", (7, i), (7, i), cc)
        ts.add("FONTNAME",  (7, i), (7, i), "Helvetica-Bold")

    t = Table(rows,
              colWidths=[0.7*cm, 3.5*cm, 1.5*cm, 1.5*cm, 1.5*cm, 4.5*cm, 2.5*cm, 1.6*cm],
              repeatRows=1)
    t.setStyle(ts)
    return t


def _build_trace(records, body_s):
    """Narrative decision trace paragraphs."""
    items = []
    for rec in records:
        sel  = rec["selected_action"]
        conf = sel.get("confidence", 0) or 0
        cc   = "#16a34a" if conf >= 0.7 else ("#d97706" if conf >= 0.5 else "#dc2626")
        sig  = rec["state_snapshot"]["signals"]
        active = ", ".join(k for k, v in sig.items()
                           if not k.endswith("_confidence") and v != "none")
        ev   = sel.get("reason", {}).get("evidence", {})
        ev_parts = [f"{k}={v}" for k, v in ev.items()
                    if k not in ("trigger", "confidence") and v is not None]

        line = (
            f'<font color="#64748b">Iter {rec.get("iteration","?")}  </font>'
            f'<b>{rec["state_snapshot"]["model"]["model_name"]}</b>'
            f'<font color="#64748b">  →  </font>'
            f'<font color="#2563eb"><b>{sel["action_type"]}</b></font>'
            f'<font color="#64748b">  trigger=</font>{sel["reason"]["trigger"]}'
            f'<font color="#64748b">  conf=</font><font color="{cc}"><b>{conf:.2f}</b></font>'
        )
        if active:
            line += f'<font color="#64748b">  signals=</font><i>{active}</i>'
        if ev_parts:
            line += f'<font color="#64748b">  evidence=</font><i>{",  ".join(ev_parts)}</i>'
        items.append(Paragraph(line, body_s))
    return items


# ── Public entry point ────────────────────────────────────────────────────────

def build_pdf(
    run_id: str,
    dataset_name: str,
    records: list[dict],
    output_dir: Path,
    pdf_path: Path,
) -> None:
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2*cm,
    )
    st    = _ts()
    story = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("StratML", st["title"]))
    story.append(Paragraph("Experiment Report", st["sub"]))
    story.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=8))
    story.append(_build_header(run_id, dataset_name, len(records)))
    story.append(Spacer(1, 0.5*cm))

    # ── KPI cards ─────────────────────────────────────────────────────────────
    if records:
        best = max(records, key=lambda r: r["state_snapshot"]["metrics"].get("primary") or 0)
        story.append(_build_kpi(best, st["cell"]))
        story.append(Spacer(1, 0.4*cm))

    # ── Dataset profile ───────────────────────────────────────────────────────
    ds = (records[0]["state_snapshot"].get("dataset") or {}) if records else {}
    if ds:
        _section(story, "Dataset Profile", st)
        story.append(_build_dataset(ds))
        story.append(Spacer(1, 0.3*cm))

    # ── Full metrics ──────────────────────────────────────────────────────────
    if records:
        _section(story, "Metrics per Iteration", st)
        story.append(_build_metrics_table(records))
        story.append(Spacer(1, 0.3*cm))

    # ── Hyperparameters ───────────────────────────────────────────────────────
    if records:
        _section(story, "Hyperparameters Used", st)
        story.append(_build_hyperparams_table(records, st["mono"]))
        story.append(Spacer(1, 0.3*cm))

    # ── Agent deliberation ────────────────────────────────────────────────────
    if records:
        _section(story, "Agent Deliberation", st)
        story.append(_build_agent_table(records))
        story.append(Spacer(1, 0.3*cm))

    # ── Decision trace ────────────────────────────────────────────────────────
    _section(story, "Decision Trace", st)
    story.extend(_build_trace(records, st["body"]))

    doc.build(story)
