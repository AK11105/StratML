"""cli/commands/utils.py — `stratml init` and `stratml doctor` commands."""

from __future__ import annotations

import importlib
import sys
import yaml
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from stratml.cli.config import DEFAULT_CONFIG

_console = Console(highlight=False)


def init_config() -> None:
    with open("config.yaml", "w") as f:
        yaml.dump(DEFAULT_CONFIG, f, sort_keys=False)
    _console.print()
    _console.print(Panel.fit(
        "[bold green]Created config.yaml[/bold green]\n"
        "[dim]Edit [/dim][cyan]dataset.path[/cyan][dim] and [/dim][cyan]dataset.target_column[/cyan][dim] before running.[/dim]",
        border_style="green", padding=(0, 2),
    ))
    _console.print()


def doctor_check() -> None:
    packages = ["pandas", "numpy", "sklearn", "torch", "pydantic", "mlflow", "yaml"]
    _console.print()
    _console.print(Panel.fit(
        "[bold white]Environment Check[/bold white]",
        border_style="bright_blue", padding=(0, 2),
    ))

    t = Table(box=box.ROUNDED, border_style="bright_blue",
              header_style="bold white on bright_blue", show_lines=False)
    t.add_column("Status",  width=8)
    t.add_column("Package", style="cyan", width=20)
    t.add_column("Version", style="dim",  width=16)

    for pkg in packages:
        try:
            mod     = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "ok")
            t.add_row("[bold green]  ok[/bold green]", pkg, version)
        except ImportError:
            t.add_row("[bold red]MISSING[/bold red]", pkg, "[dim]—[/dim]")

    _console.print(t)
    _console.print()


def validate_config_cmd(args) -> None:
    from stratml.cli.config import load_yaml, validate_config
    config = load_yaml(args.config)
    try:
        validate_config(config)
        _console.print()
        _console.print(Panel.fit(
            f"[bold green]Config OK[/bold green]  [dim]—[/dim]  [cyan]{args.config}[/cyan]",
            border_style="green", padding=(0, 2),
        ))
        _console.print()
    except ValueError as e:
        _console.print()
        _console.print(Panel.fit(
            f"[bold red]Invalid config[/bold red]\n[white]{e}[/white]",
            border_style="red", padding=(0, 2),
        ))
        _console.print()
        sys.exit(1)
