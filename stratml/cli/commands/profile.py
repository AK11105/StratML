"""cli/commands/profile.py — `stratml profile-data` command."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

_console = Console(highlight=False)


def profile_data(args) -> None:
    from stratml.execution.data.loader import load_dataframe
    from stratml.execution.data.validator import build_dataset
    from stratml.execution.data.profiler import build_profile

    df, dataset_name = load_dataframe(args.dataset)
    dataset  = build_dataset(df, dataset_name, args.target)
    profile  = build_profile(dataset)

    out_dir  = Path("outputs") / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "data_profile.json"
    out_file.write_text(json.dumps(profile.model_dump(), indent=2))

    p = profile
    _console.print()
    _console.print(Panel.fit(
        f"[bold white]Dataset Profile[/bold white]  [dim]—[/dim]  [cyan]{p.dataset_name}[/cyan]",
        border_style="bright_blue", padding=(0, 2),
    ))

    meta = Table.grid(padding=(0, 3))
    meta.add_column(style="dim", width=18)
    meta.add_column()
    meta.add_row("Type",    f"[bold]{p.dataset_type}[/bold]  [dim]|[/dim]  Problem: [bold]{p.problem_type}[/bold]")
    meta.add_row("Shape",   f"[white]{p.rows:,} rows x {p.columns} columns[/white]")
    meta.add_row("Target",  f"[green]{p.target_column}[/green]")
    meta.add_row("Features", f"[white]{len(p.numerical_columns)} numerical[/white]  [dim]|[/dim]  [white]{len(p.categorical_columns)} categorical[/white]")
    meta.add_row("Missing", f"[{'red' if p.missing_value_ratio > 0.1 else 'yellow' if p.missing_value_ratio > 0 else 'green'}]{p.missing_value_ratio:.2%}[/]")
    if p.class_distribution:
        dist_str = "  ".join(f"[dim]{k}:[/dim] [white]{v}[/white]" for k, v in p.class_distribution.items())
        meta.add_row("Classes", dist_str)
    _console.print(meta)
    _console.print()

    t = Table(box=box.ROUNDED, border_style="bright_blue",
              header_style="bold white on bright_blue", show_lines=False)
    t.add_column("Feature",  style="cyan",  width=24)
    t.add_column("Type",     style="dim",   width=10)
    t.add_column("Unique",   justify="right", width=7)
    t.add_column("Missing",  justify="right", width=9)
    t.add_column("Dist",     width=12)
    for f in p.feature_summary:
        miss_style = "red" if f.missing_percentage > 10 else "yellow" if f.missing_percentage > 0 else "dim"
        t.add_row(
            f.name,
            f.dtype,
            str(f.unique_values),
            f"[{miss_style}]{f.missing_percentage:.1f}%[/{miss_style}]",
            f.distribution,
        )
    _console.print(t)
    _console.print()

    footer = Table.grid(padding=(0, 3))
    footer.add_column(style="dim", width=22)
    footer.add_column()
    footer.add_row("Recommended metrics", f"[white]{', '.join(p.recommended_metrics)}[/white]")
    footer.add_row("Saved to",            f"[dim]{out_file}[/dim]")
    _console.print(footer)
    _console.print()
