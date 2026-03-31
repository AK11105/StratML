#!/usr/bin/env python

import argparse
import yaml
import sys
from copy import deepcopy
from pathlib import Path

DEFAULT_CONFIG = {
    "mode": "beginner",
    "dataset": {"path": None, "target_column": None},
    "execution": {"max_iterations": 5, "timeout_per_run": 300, "random_seed": 42},
    "split": {"method": "stratified", "test_size": 0.2},
    "logging": {"enable_mlflow": False, "enable_tensorboard": False, "log_level": "info"},
    "constraints": {"max_memory": None, "max_cpu": None},
}


def deep_merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        sys.exit(1)


def apply_cli_overrides(config, args):
    config = deepcopy(config)
    if getattr(args, "mode", None) is not None:
        config["mode"] = args.mode
    if getattr(args, "max_iter", None) is not None:
        config["execution"]["max_iterations"] = args.max_iter
    if getattr(args, "path", None) is not None:
        config["dataset"]["path"] = args.path
    return config


def validate_config(config):
    if config["dataset"]["path"] is None:
        raise ValueError("dataset.path is required")
    if config["dataset"]["target_column"] is None:
        raise ValueError("dataset.target_column is required")
    if config["mode"] not in ["beginner", "intermediate", "expert"]:
        raise ValueError("Invalid mode")
    return True


def enforce_mode_rules(config):
    mode = config["mode"]
    if mode == "beginner":
        config.pop("expert", None)
        config.pop("intermediate", None)
    return config


def run_pipeline(args):
    yaml_config = load_yaml(args.config)
    config = deep_merge(DEFAULT_CONFIG, yaml_config)
    config = apply_cli_overrides(config, args)
    config = enforce_mode_rules(config)

    try:
        validate_config(config)
    except Exception as e:
        print(f"\n  [Invalid config] {e}\n")
        sys.exit(1)

    sep = "-" * 44
    d = config["dataset"]
    e = config["execution"]

    if args.dry_run:
        print()
        print("  Dry Run - Resolved Config")
        print(f"  {sep}")
        print(f"  Mode          : {config['mode']}")
        print(f"  Dataset       : {d['path']}")
        print(f"  Target        : {d['target_column']}")
        print(f"  Max iterations: {e['max_iterations']}")
        print(f"  Timeout/run   : {e['timeout_per_run']}s")
        print(f"  Random seed   : {e['random_seed']}")
        print(f"  MLflow        : {config['logging']['enable_mlflow']}")
        print(f"  TensorBoard   : {config['logging']['enable_tensorboard']}")
        print(f"  {sep}")
        print()
        return

    print()
    print("  AutoML Pipeline Starting")
    print(f"  {sep}")
    print(f"  Mode    : {config['mode']}")
    print(f"  Dataset : {d['path']}")
    print(f"  Target  : {d['target_column']}")
    print(f"  Budget  : {e['max_iterations']} iterations")
    print(f"  {sep}")
    print("  [Placeholder] Orchestrator not yet connected.")
    print()


def validate_config_cmd(args):
    config = load_yaml(args.config)
    try:
        validate_config(config)
        print(f"\n  Config OK — {args.config}\n")
    except Exception as e:
        print(f"\n  Invalid config: {e}\n")
        sys.exit(1)


def profile_data(args):
    import json

    # Ensure stratml package is importable (project root = 2 levels up from stratml/cli/)
    _root = str(Path(__file__).resolve().parents[2])
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from stratml.execution.data.loader import load_dataframe
    from stratml.execution.data.validator import build_dataset
    from stratml.execution.data.profiler import build_profile

    outputs_dir = Path(__file__).resolve().parents[3] / "outputs"

    df, dataset_name = load_dataframe(args.dataset)
    dataset = build_dataset(df, dataset_name, args.target)
    profile = build_profile(dataset)

    out_dir = outputs_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "data_profile.json"
    out_file.write_text(json.dumps(profile.model_dump(), indent=2))

    p = profile
    sep = "-" * 44

    print()
    print("  Dataset Profile")
    print(f"  {sep}")
    print(f"  Dataset       : {p.dataset_name}")
    print(f"  Type          : {p.dataset_type}  |  Problem: {p.problem_type}")
    print(f"  Shape         : {p.rows} rows x {p.columns} columns")
    print(f"  Target        : {p.target_column}")
    print(f"  {sep}")
    print(f"  Features      : {len(p.numerical_columns)} numerical, {len(p.categorical_columns)} categorical")
    print(f"  Missing ratio : {p.missing_value_ratio:.2%}")

    if p.class_distribution:
        dist_str = "  |  ".join(f"{k}: {v}" for k, v in p.class_distribution.items())
        print(f"  Classes       : {dist_str}")

    print(f"  {sep}")
    print(f"  Feature Summary")
    print(f"  {'Name':<24} {'Type':<10} {'Unique':>6}  {'Missing':>8}  {'Dist'}")
    print(f"  {'-'*24} {'-'*10} {'-'*6}  {'-'*8}  {'-'*10}")
    for f in p.feature_summary:
        print(f"  {f.name:<24} {f.dtype:<10} {f.unique_values:>6}  {f.missing_percentage:>7.1f}%  {f.distribution}")

    print(f"  {sep}")
    print(f"  Recommended metrics : {', '.join(p.recommended_metrics)}")
    print(f"  Saved to            : {out_file}")
    print()


def init_config():
    with open("config.yaml", "w") as f:
        yaml.dump(DEFAULT_CONFIG, f, sort_keys=False)
    print("\n  Created config.yaml with default settings.")
    print("  Edit dataset.path and dataset.target_column before running.\n")


def doctor_check():
    import importlib
    sep = "-" * 44
    packages = ["pandas", "numpy", "sklearn", "torch", "pydantic", "mlflow", "yaml"]
    print()
    print("  Environment Check")
    print(f"  {sep}")
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg if pkg != "sklearn" else "sklearn")
            version = getattr(mod, "__version__", "ok")
            print(f"  {'ok':<6} {pkg:<20} {version}")
        except ImportError:
            print(f"  {'MISSING':<6} {pkg}")
    print(f"  {sep}")
    print()


def main():
    parser = argparse.ArgumentParser(prog="stratml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("config", type=str)
    run_parser.add_argument("--path", type=str)
    run_parser.add_argument("--mode", choices=["beginner", "intermediate", "expert"])
    run_parser.add_argument("--max-iter", type=int)
    run_parser.add_argument("--dry-run", action="store_true")

    validate_parser = subparsers.add_parser("validate-config")
    validate_parser.add_argument("config", type=str)

    profile_parser = subparsers.add_parser("profile-data")
    profile_parser.add_argument("dataset", type=str)
    profile_parser.add_argument("target", type=str)

    subparsers.add_parser("init")
    subparsers.add_parser("doctor")

    args = parser.parse_args()

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
