#!/usr/bin/env python

import argparse
import yaml 
import sys
from copy import deepcopy


# =========================
# DEFAULT CONFIG
# =========================
DEFAULT_CONFIG = {
    "mode": "beginner",
    "dataset": {
        "path": None,
        "target_column": None
    },
    "execution": {
        "max_iterations": 5,
        "timeout_per_run": 300,
        "random_seed": 42
    },
    "split": {
        "method": "stratified",
        "test_size": 0.2
    },
    "logging": {
        "enable_mlflow": False,
        "enable_tensorboard": False,
        "log_level": "info"
    },
    "constraints": {
        "max_memory": None,
        "max_cpu": None
    }
}


# =========================
# UTIL: DEEP MERGE
# =========================
def deep_merge(base, override):
    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


# =========================
# LOAD YAML
# =========================
def load_yaml(path):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        sys.exit(1)


# =========================
# APPLY CLI OVERRIDES
# =========================
def apply_cli_overrides(config, args):
    config = deepcopy(config)

    if getattr(args, "mode", None) is not None:
        config["mode"] = args.mode

    if getattr(args, "max_iter", None) is not None:
        config["execution"]["max_iterations"] = args.max_iter

    return config


# =========================
# VALIDATION
# =========================
def validate_config(config):
    if config["dataset"]["path"] is None:
        raise ValueError("dataset.path is required")

    if config["dataset"]["target_column"] is None:
        raise ValueError("dataset.target_column is required")

    if config["mode"] not in ["beginner", "intermediate", "expert"]:
        raise ValueError("Invalid mode")

    return True


# =========================
# MODE ENFORCEMENT
# =========================
def enforce_mode_rules(config):
    mode = config["mode"]

    if mode == "beginner":
        # wipe manual overrides
        config.pop("expert", None)
        config.pop("intermediate", None)

    elif mode == "intermediate":
        # restrict but keep allowed models
        pass

    elif mode == "expert":
        # trust user config
        pass

    return config


# =========================
# COMMAND HANDLERS
# =========================
def run_pipeline(args):
    yaml_config = load_yaml(args.config)

    config = deep_merge(DEFAULT_CONFIG, yaml_config)
    config = apply_cli_overrides(config, args)
    config = enforce_mode_rules(config)

    try:
        validate_config(config)
    except Exception as e:
        print(f"[INVALID CONFIG] {e}")
        sys.exit(1)

    if args.dry_run:
        print("=== DRY RUN CONFIG ===")
        print(yaml.dump(config, sort_keys=False))
        return

    # Placeholder for orchestrator
    print("Running AutoML pipeline...")
    print(yaml.dump(config, sort_keys=False))


def validate_config_cmd(args):
    config = load_yaml(args.config)

    try:
        validate_config(config)
        print("✅ Config is valid")
    except Exception as e:
        print(f"❌ Invalid config: {e}")
        sys.exit(1)


def profile_data(args):
    print(f"Profiling dataset: {args.input}")
    # Placeholder


def init_config():
    print("Generating default config.yaml...")
    with open("config.yaml", "w") as f:
        yaml.dump(DEFAULT_CONFIG, f, sort_keys=False)


def doctor_check():
    print("Running environment checks...")
    # Placeholder


# =========================
# CLI SETUP
# =========================
def main():
    parser = argparse.ArgumentParser(prog="stratml")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -----------------
    # RUN
    # -----------------
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=str, required=True)
    run_parser.add_argument("--mode", choices=["beginner", "intermediate", "expert"])
    run_parser.add_argument("--max-iter", type=int)
    run_parser.add_argument("--dry-run", action="store_true")

    # -----------------
    # VALIDATE CONFIG
    # -----------------
    validate_parser = subparsers.add_parser("validate-config")
    validate_parser.add_argument("config", type=str)

    # -----------------
    # PROFILE DATA
    # -----------------
    profile_parser = subparsers.add_parser("profile-data")
    profile_parser.add_argument("--input", type=str, required=True)

    # -----------------
    # INIT
    # -----------------
    subparsers.add_parser("init")

    # -----------------
    # DOCTOR
    # -----------------
    subparsers.add_parser("doctor")

    args = parser.parse_args()

    # =========================
    # ROUTING
    # =========================
    if args.command == "run":
        run_pipeline(args)

    elif args.command == "validate-config":
        validate_config_cmd(args)

    elif args.command == "profile-data":
        profile_data(args)

    elif args.command == "init":
        init_config()

    elif args.command == "doctor":
        doctor_check()


if __name__ == "__main__":
    main()