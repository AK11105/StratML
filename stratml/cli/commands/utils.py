"""cli/commands/utils.py — `stratml init` and `stratml doctor` commands."""

from __future__ import annotations

import importlib
import sys
import yaml
from pathlib import Path

from stratml.cli.config import DEFAULT_CONFIG


def init_config() -> None:
    with open("config.yaml", "w") as f:
        yaml.dump(DEFAULT_CONFIG, f, sort_keys=False)
    print("\n  Created config.yaml — edit dataset.path and dataset.target_column before running.\n")


def doctor_check() -> None:
    packages = ["pandas", "numpy", "sklearn", "torch", "pydantic", "mlflow", "yaml"]
    sep = "-" * 44
    print(f"\n  Environment Check\n  {sep}")
    for pkg in packages:
        try:
            mod     = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "ok")
            print(f"  {'ok':<6} {pkg:<20} {version}")
        except ImportError:
            print(f"  {'MISSING':<6} {pkg}")
    print(f"  {sep}\n")


def validate_config_cmd(args) -> None:
    from stratml.cli.config import load_yaml, validate_config
    config = load_yaml(args.config)
    try:
        validate_config(config)
        print(f"\n  Config OK — {args.config}\n")
    except ValueError as e:
        print(f"\n  Invalid config: {e}\n")
        sys.exit(1)
