"""
reporting/report_generator.py
------------------------------
Public API for the reporting module.
Coordinates pdf_builder and comparison — no logic lives here.

Public functions (unchanged interface):
    generate_report(run_id, dataset_name, output_dir) -> Path
    generate_model_script(run_id, output_dir, records) -> Path
"""

from __future__ import annotations

import json
from pathlib import Path

from stratml.reporting.pdf_builder import build_pdf
from stratml.reporting.comparison import write_comparison, generate_model_script  # noqa: F401 — re-exported


def generate_report(run_id: str, dataset_name: str, output_dir: Path) -> Path:
    """Build PDF + comparison files from decision logs. Returns PDF path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir   = output_dir / "decision_logs"
    log_files = sorted(log_dir.glob(f"{run_id}_*.json")) if log_dir.exists() else []

    records = []
    for lf in log_files:
        try:
            records.append(json.loads(lf.read_text()))
        except Exception:
            continue

    write_comparison(records, output_dir)

    pdf_path = output_dir / "report.pdf"
    build_pdf(run_id, dataset_name, records, output_dir, pdf_path)
    return pdf_path
