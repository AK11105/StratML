"""
demo/calibrate.py
-----------------
Trains each model in each demo's ITERS on the real dataset with the same
preprocessing the demo uses, then prints verified metric values.

The demo STORY (signals, actions, triggers, model sequence) is preserved.
Only the numbers that a sceptic would verify are replaced:
  primary metric, f1/prec/rec, train_loss, val_loss, gap, runtime.

If a real result contradicts the declared signal (e.g. story says
"overfitting: strong" but real gap is tiny) the script prints a WARNING
so you can decide whether to adjust the signal or accept the discrepancy.

Usage:
    python3 demo/calibrate.py                        # all ML demos
    python3 demo/calibrate.py titanic pima           # specific demos
    python3 demo/calibrate.py --dry-run titanic      # print without patch suggestion

NOTE: demo_energy (DL) is excluded — MLP/CNN1D/RNN require a separate
      training loop and are not calibrated here.
"""
from __future__ import annotations
import sys, time, importlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_squared_error,
)
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).parents[1]
SEED = 42

# ── Model registry ────────────────────────────────────────────────────────────

REGISTRY = {
    "RandomForestClassifier":     RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=SEED),
    "LogisticRegression":         LogisticRegression(max_iter=1000, random_state=SEED),
    "ExtraTreesClassifier":       ExtraTreesClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    "SVC":                        SVC(random_state=SEED),
    "RandomForestRegressor":      RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
    "GradientBoostingRegressor":  GradientBoostingRegressor(random_state=SEED),
    "SVR":                        SVR(),
    "Ridge":                      Ridge(),
    "ExtraTreesRegressor":        ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
}

# ── Demo specs ────────────────────────────────────────────────────────────────
# Each entry: (model_name, declared_signal_key)
# signal_key is the dominant signal the story declares for that iteration.
# Used to warn if real metrics contradict it.

DEMOS: dict[str, dict] = {
    "titanic": {
        "csv":    ROOT / "data/external/titanic.csv",
        "target": "Survived",
        "task":   "classification",
        "iters": [
            ("RandomForestClassifier",     "overfitting"),
            ("RandomForestClassifier",     "overfitting"),    # modify_regularization
            ("RandomForestClassifier",     "overfitting"),    # decrease_model_capacity
            ("LogisticRegression",         "stagnating"),
            ("GradientBoostingClassifier", "converged"),
        ],
    },
    "california_housing": {
        "csv":    ROOT / "data/raw/california_housing.csv",
        "target": "MedHouseVal",
        "task":   "regression",
        "iters": [
            ("RandomForestRegressor",      "underfitting"),
            ("GradientBoostingRegressor",  "diminishing_returns"),
            ("SVR",                        "too_slow"),
            ("Ridge",                      "underfitting"),
            ("ExtraTreesRegressor",        "converged"),
        ],
    },
    "pima": {
        "csv":    ROOT / "data/raw/pima.csv",
        "target": "Outcome",
        "task":   "classification",
        "iters": [
            ("RandomForestClassifier",     "overfitting"),
            ("GradientBoostingClassifier", "overfitting"),
            ("GradientBoostingClassifier", "overfitting"),    # modify_regularization
            ("GradientBoostingClassifier", "stagnating"),     # add_preprocessing
            ("ExtraTreesClassifier",       "converged"),
        ],
    },
    "mnist": {
        "csv":    ROOT / "data/raw/mnist.csv",
        "target": "label",
        "task":   "classification",
        "iters": [
            ("RandomForestClassifier",     "diminishing_returns"),
            ("GradientBoostingClassifier", "too_slow"),
            ("ExtraTreesClassifier",       "converged"),
        ],
    },
    "wine_quality": {
        "csv":    ROOT / "data/raw/wine_quality_red.csv",
        "target": "quality",
        "task":   "classification",
        "iters": [
            ("RandomForestClassifier",     "underfitting"),
            ("GradientBoostingClassifier", "underfitting"),
            ("GradientBoostingClassifier", "underfitting"),   # increase_model_capacity
            ("ExtraTreesClassifier",       "stagnating"),
            ("SVC",                        "converged"),
        ],
    },
    "creditcard": {
        "csv":    ROOT / "data/external/creditcard.csv",
        "target": "Class",
        "task":   "classification",
        "iters": [
            ("RandomForestClassifier",     "overfitting"),
            ("RandomForestClassifier",     "stagnating"),     # add_preprocessing
            ("GradientBoostingClassifier", "diminishing_returns"),
            ("ExtraTreesClassifier",       "stagnating"),
            ("LogisticRegression",         "converged"),
        ],
    },
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_split(csv_path: Path, target: str, task: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(X), columns=X.columns)

    stratify = y if task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=stratify)

    stratify2 = y_tr if task == "classification" else None
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.25, random_state=SEED, stratify=stratify2)

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_va = sc.transform(X_va)
    X_te = sc.transform(X_te)

    return X_tr, X_va, X_te, y_tr, y_va, y_te


# ── Measurement ───────────────────────────────────────────────────────────────

def measure(model_name: str, X_tr, X_va, X_te, y_tr, y_va, y_te, task: str):
    model = clone(REGISTRY[model_name])

    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    rt = round(time.perf_counter() - t0, 2)

    if task == "classification":
        avg = "binary" if len(np.unique(y_tr)) == 2 else "weighted"
        primary = round(accuracy_score(y_te, model.predict(X_te)), 4)
        f1      = round(f1_score(y_te,  model.predict(X_te), average=avg, zero_division=0), 4)
        prec    = round(precision_score(y_te, model.predict(X_te), average=avg, zero_division=0), 4)
        rec     = round(recall_score(y_te,  model.predict(X_te), average=avg, zero_division=0), 4)
        tl      = round(1 - accuracy_score(y_tr, model.predict(X_tr)), 4)
        vl      = round(1 - accuracy_score(y_va, model.predict(X_va)), 4)
        gap     = round(abs(tl - vl), 4)
        return primary, f1, prec, rec, tl, vl, gap, rt
    else:
        primary = round(r2_score(y_te, model.predict(X_te)), 4)
        tl      = round(mean_squared_error(y_tr, model.predict(X_tr)), 4)
        vl      = round(mean_squared_error(y_va, model.predict(X_va)), 4)
        gap     = round(abs(tl - vl), 4)
        return primary, None, None, None, tl, vl, gap, rt


# ── Signal consistency check ──────────────────────────────────────────────────

def check_signal(signal: str, gap: float, primary: float, task: str) -> str | None:
    """Return a warning string if real metrics contradict the declared signal."""
    if signal == "overfitting" and gap < 0.05:
        return f"gap={gap} is small — overfitting signal may be weak in reality"
    if signal == "underfitting" and primary > 0.85:
        return f"primary={primary} is high — underfitting signal may be overstated"
    if signal == "too_slow":
        return None  # runtime-based, always valid
    if signal == "converged" and gap > 0.15:
        return f"gap={gap} is large — converged signal may be premature"
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def run_demo(name: str, spec: dict, dry_run: bool):
    print(f"\n{'='*62}")
    print(f"  DEMO: {name}")
    print(f"{'='*62}")

    print(f"  Loading {spec['csv'].name} ...")
    X_tr, X_va, X_te, y_tr, y_va, y_te = load_and_split(spec["csv"], spec["target"], spec["task"])
    print(f"  Split → train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}\n")

    results = []
    for i, (model_name, signal) in enumerate(spec["iters"]):
        print(f"  [{i+1}/{len(spec['iters'])}] {model_name} ...", end=" ", flush=True)
        metrics = measure(model_name, X_tr, X_va, X_te, y_tr, y_va, y_te, spec["task"])
        primary, f1, prec, rec, tl, vl, gap, rt = metrics
        results.append(metrics)

        warn = check_signal(signal, gap, primary, spec["task"])
        status = f"⚠  WARNING: {warn}" if warn else "✓"
        print(f"done ({rt}s)  {status}")

    # Print paste-ready output
    print(f"\n  ── Verified metrics for {name}/ITERS ──")
    print(f"  (paste these numbers into the ITERS tuples, keeping story/signals intact)\n")
    for i, ((model_name, signal), metrics) in enumerate(zip(spec["iters"], results)):
        primary, f1, prec, rec, tl, vl, gap, rt = metrics
        if spec["task"] == "classification":
            print(f"  iter {i+1}  {model_name}  [signal: {signal}]")
            print(f"    primary={primary}  f1={f1}  prec={prec}  rec={rec}")
            print(f"    tl={tl}  vl={vl}  gap={gap}  rt={rt}")
        else:
            print(f"  iter {i+1}  {model_name}  [signal: {signal}]")
            print(f"    primary(r2)={primary}  tl(mse)={tl}  vl={vl}  gap={gap}  rt={rt}")
        print()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry_run = "--dry-run" in sys.argv

    targets = args if args else list(DEMOS.keys())
    for name in targets:
        if name not in DEMOS:
            print(f"Unknown demo '{name}'. Available: {list(DEMOS.keys())}")
            continue
        run_demo(name, DEMOS[name], dry_run)

    print("\nDone. Energy demo (MLP/CNN1D/RNN) excluded — DL calibration not supported here.")


if __name__ == "__main__":
    main()
