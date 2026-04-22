"""cli/commands/profile.py — `stratml profile-data` command."""

from __future__ import annotations

import json
from pathlib import Path


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

    p   = profile
    sep = "-" * 44
    print(f"\n  Dataset Profile\n  {sep}")
    print(f"  Dataset       : {p.dataset_name}")
    print(f"  Type          : {p.dataset_type}  |  Problem: {p.problem_type}")
    print(f"  Shape         : {p.rows} rows x {p.columns} columns")
    print(f"  Target        : {p.target_column}\n  {sep}")
    print(f"  Features      : {len(p.numerical_columns)} numerical, {len(p.categorical_columns)} categorical")
    print(f"  Missing ratio : {p.missing_value_ratio:.2%}")
    if p.class_distribution:
        print(f"  Classes       : {'  |  '.join(f'{k}: {v}' for k, v in p.class_distribution.items())}")
    print(f"\n  {sep}")
    print(f"  {'Name':<24} {'Type':<10} {'Unique':>6}  {'Missing':>8}  Dist")
    print(f"  {'-'*24} {'-'*10} {'-'*6}  {'-'*8}  {'-'*10}")
    for f in p.feature_summary:
        print(f"  {f.name:<24} {f.dtype:<10} {f.unique_values:>6}  {f.missing_percentage:>7.1f}%  {f.distribution}")
    print(f"\n  {sep}")
    print(f"  Recommended metrics : {', '.join(p.recommended_metrics)}")
    print(f"  Saved to            : {out_file}\n")
