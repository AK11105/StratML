#!/usr/bin/env python
"""
stratml/cli/main.py
-------------------
Entry point — argument parsing and command dispatch only.
All command logic lives in cli/commands/.
All config logic lives in cli/config.py.
"""

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="stratml")
    sub    = parser.add_subparsers(dest="command", required=True)

    # ── run ───────────────────────────────────────────────────────────────────
    run = sub.add_parser("run", help="Run the AutoML pipeline")
    run.add_argument("config")
    run.add_argument("--path")
    run.add_argument("--mode", choices=["beginner", "intermediate", "expert"])
    run.add_argument("--max-iter", type=int)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--dl", action="store_true")
    run.add_argument("--architecture", choices=["MLP", "CNN1D", "RNN"])
    run.add_argument("--epochs", type=int)
    run.add_argument("--lr", type=float)
    run.add_argument("--batch-size", type=int)

    # ── validate-config ───────────────────────────────────────────────────────
    vc = sub.add_parser("validate-config", help="Validate a config file")
    vc.add_argument("config")

    # ── profile-data ──────────────────────────────────────────────────────────
    pd = sub.add_parser("profile-data", help="Profile a dataset")
    pd.add_argument("dataset")
    pd.add_argument("target")

    # ── init / doctor ─────────────────────────────────────────────────────────
    sub.add_parser("init",   help="Create a default config.yaml")
    sub.add_parser("doctor", help="Check environment dependencies")

    args = parser.parse_args()

    if args.command == "run":
        from stratml.cli.commands.run import run_pipeline
        run_pipeline(args)
    elif args.command == "validate-config":
        from stratml.cli.commands.utils import validate_config_cmd
        validate_config_cmd(args)
    elif args.command == "profile-data":
        from stratml.cli.commands.profile import profile_data
        profile_data(args)
    elif args.command == "init":
        from stratml.cli.commands.utils import init_config
        init_config()
    elif args.command == "doctor":
        from stratml.cli.commands.utils import doctor_check
        doctor_check()


if __name__ == "__main__":
    main()
