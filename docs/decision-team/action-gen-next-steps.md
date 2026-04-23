# Decision Engine — Remaining Work

## Current State

The full vertical pipeline is implemented and wired. LLM-backed agents (performance,
efficiency, stability, coordinator) and LLM-backed action generation are all done with
rule-based fallbacks. All dataset bugs are fixed. Unified decision dataset accumulates
across runs at `runs/decision_logs/decision_dataset.csv`.

Check progress toward value model activation:
```bash
python -c "
import pandas as pd
df = pd.read_csv('runs/decision_logs/decision_dataset.csv')
filled = df[df['observed_gain'].notna() & (df['observed_gain'] != '')].shape[0]
print(f'{filled}/50 rows filled — value model active: {filled >= 50}')
"
```

---

## Phase 4 — Activate Learning Stubs (data-gated)

**Threshold:** 50 rows with `observed_gain` filled in `runs/decision_logs/decision_dataset.csv`.

Stubs activate automatically — no code changes needed. Verify activation:

- `value_model.py` logs no "using stub" warning after 50 rows
- `calibration.py` fits on real `(predicted_gain, actual_gain)` pairs
- `uncertainty.py` ensemble variance is non-zero

One remaining fix needed in `uncertainty.py`: the ensemble re-encodes using only
`predicted_gain` as a proxy feature because `StateObject` is not available at that
stage. Fix by passing `state` through to `estimate()` and using `_encode_state_action()`
from `value_model.py` for proper feature encoding.

---

## Phase 5 — Preprocessing Adaptation

**File:** `stratml/decision/policy/action_selector.py`

`_DEFAULT_PREPROCESSING` is hardcoded. Replace with a function that reads `state.dataset`:
- `imbalance_strategy` from `state.dataset.imbalance_ratio`
- `missing_value_strategy` from `state.dataset.missing_ratio`
- `scaling` from model type (tree models don't need it)

---

## Phase 6 — Action Coverage: Execution Team Dependency

The decision engine can only generate actions the execution team has implemented.
Before adding any new action type to `_VALID_ACTION_TYPES` or the rule generator,
confirm the execution team has a handler in `build_experiment_config` and the pipeline.

### Current status

| Action | ML | DL | Notes |
|---|---|---|---|
| `switch_model` | ✅ | ✅ | |
| `modify_regularization` | ✅ | ✅ | |
| `decrease_model_capacity` | ✅ | ✅ | |
| `increase_model_capacity` | ✅ | ✅ | |
| `change_optimizer` | ❌ ignored | ✅ | ML path silently ignores this |
| `add_preprocessing` | ❌ | ❌ | Generated for imbalance; execution must implement |
| `terminate` | ✅ | ✅ | |

### Full action space (execution team backlog)

**Model selection**

| Action | Parameters | ML | DL |
|---|---|---|---|
| `switch_model` | `model_name` | ✅ | ✅ |
| `tune_hyperparameters` | `param_grid`, `method: grid/random/bayesian` | ❌ | ❌ |
| `ensemble` | `strategy: voting/stacking`, `model_names: list` | ❌ | ❌ |

**Capacity**

| Action | Parameters | ML | DL |
|---|---|---|---|
| `increase_model_capacity` | `scale` | ✅ | ✅ |
| `decrease_model_capacity` | `scale` | ✅ | ✅ |
| `add_layers` | `num_layers`, `units` | N/A | ❌ |
| `remove_layers` | `num_layers` | N/A | ❌ |
| `change_architecture` | `architecture: mlp/cnn/rnn` | N/A | ❌ |

**Regularization**

| Action | Parameters | ML | DL |
|---|---|---|---|
| `modify_regularization` | `direction: increase/decrease` | ✅ | ✅ |
| `add_dropout` | `rate` | N/A | ❌ |
| `add_batch_norm` | — | N/A | ❌ |
| `add_weight_decay` | `lambda` | N/A | ❌ |

**Optimization (DL-specific)**

| Action | Parameters | ML | DL |
|---|---|---|---|
| `change_optimizer` | `learning_rate_scale` | ❌ ignored | ✅ |
| `change_lr_schedule` | `schedule: cosine/step/plateau` | N/A | ❌ |
| `change_batch_size` | `batch_size` | N/A | ❌ |
| `add_gradient_clipping` | `max_norm` | N/A | ❌ |

**Preprocessing**

| Action | Parameters | ML | DL |
|---|---|---|---|
| `add_preprocessing` | `strategy: oversample/undersample/smote` | ❌ | ❌ |
| `add_feature_selection` | `method: variance/mutual_info/pca`, `n_features` | ❌ | ❌ |
| `change_scaling` | `method: standard/minmax/robust/none` | ❌ | ❌ |
| `add_augmentation` | `strategy` | N/A | ❌ |

**Control**

| Action | Parameters | ML | DL |
|---|---|---|---|
| `terminate` | — | ✅ | ✅ |

Execution team priority order: `change_optimizer` (ML) → `add_preprocessing` →
`add_feature_selection` → `tune_hyperparameters` → `ensemble` → DL-specific actions.

---

## Phase 7 — LangGraph: Intra-Decision Cycles

Not started. Warranted when any of these arise from real runs:

**Low-confidence deadlock** — coordinator confidence below threshold (`< 0.4`).
With LangGraph: conditional edge `coordinator → re_query_agent → coordinator`.

**Candidate rejection loop** — coordinator finds all candidates unsuitable.
With LangGraph: `coordinator → action_generator → coordinator`.

**Pre-commit validation** — verify calibrated gain justifies predicted cost before
committing. Conditional branch on efficiency agent re-evaluation.

Trigger: start only after Phase 4 is active and producing meaningful calibrated gains.

---

## Order of Work

1. ✅ All dataset bugs fixed
2. Phase 4 — verify stub activation after 50 rows; fix `uncertainty.py` state encoding
3. Phase 5 — adaptive preprocessing in `action_selector.py`
4. Phase 6 — coordinate with execution team on action handlers (see table above)
5. Phase 7 — LangGraph cycles (data-gated)
