# Decision Engine — Vertical Flow Plan

## Overview

Full pipeline from `StateObject` → `ActionDecision`, all files in correct positions.
Components that can't be fully implemented yet are **stubs** (valid signatures, neutral return values).
Stubs are swapped out during horizontal depth phase — no interface changes needed.

---

## File Locations & Status

### ACTION GENERATION
| File | Status | Notes |
|---|---|---|
| `stratml/decision/actions/action_generator.py` | **Full** | Rule-based, reads `state.signals` + `state.constraints`. Bootstrap-aware (`iteration == 0`). |
| `stratml/decision/learning/dataset_builder.py` | **Stub** | Appends `(state_features, action, gain)` row to CSV. No training yet. |

### PREDICTION AND CALIBRATION
| File | Status | Notes |
|---|---|---|
| `stratml/decision/learning/value_model.py` | **Stub** | Returns `predicted_gain=0.05`, `predicted_cost=0.5` per action. |
| `stratml/decision/learning/calibration.py` | **Stub** | Pass-through. Returns gain unchanged. |
| `stratml/decision/learning/uncertainty.py` | **Stub** | Returns `confidence=0.5`, `variance=0.0` per action. |

### DECISION COUNCIL
| File | Status | Notes |
|---|---|---|
| `stratml/decision/agents/performance_agent.py` | **Full** | Scores actions on accuracy/fitting signals. |
| `stratml/decision/agents/efficiency_agent.py` | **Full** | Scores actions on runtime/cost signals. |
| `stratml/decision/agents/stability_agent.py` | **Full** | Scores actions on stability/variance signals. |
| `stratml/decision/agents/coordinator_agent.py` | **Full** | Weighted aggregation → ranked action list. |

### FINAL DECISION
| File | Status | Notes |
|---|---|---|
| `stratml/decision/policy/action_selector.py` | **Full** | Picks top-ranked action → `ActionDecision`. |
| `stratml/decision/logging/decision_logger.py` | **Full** | Writes `DecisionRecord` to `runs/decision_logs/`. |
| `stratml/decision/validation/counterfactual.py` | **Stub** | Records action taken. Slot for future A/B comparison. |

---

## Full Vertical Flow

```
StateObject  (produced by Dev B — state_builder.py)
      ↓
actions/action_generator.py
    - iteration == 0  → bootstrap candidates (LogisticRegression, RandomForest)
    - iteration  > 0  → rule-based candidates from state.signals + state.constraints
    → list[CandidateAction]
      ↓
learning/dataset_builder.py
    - appends (state_features, action_type) row to decision_dataset.csv
    - fire-and-forget, does not block pipeline
      ↓
learning/value_model.py          [STUB → predicted_gain, predicted_cost per action]
      ↓
learning/calibration.py          [STUB → calibrated_gain per action]
      ↓
learning/uncertainty.py          [STUB → confidence, variance per action]
      ↓
agents/performance_agent.py  ─┐
agents/efficiency_agent.py   ─┼→ score per action (run independently)
agents/stability_agent.py    ─┘
      ↓
agents/coordinator_agent.py
    - weighted aggregation: w_perf=0.5, w_eff=0.25, w_stab=0.25
    - combines agent scores + calibrated_gain + uncertainty
    → ranked list[CandidateAction] with final_score
      ↓
policy/action_selector.py
    - picks top-ranked action
    - builds ActionDecision (fills reason, preprocessing, expected_gain, confidence)
    → ActionDecision
      ↓
logging/decision_logger.py
    - writes DecisionRecord to runs/decision_logs/{experiment_id}_{iteration}.json
      ↓
validation/counterfactual.py     [STUB → records action, slot for future comparison]
      ↓
ActionDecision  (returned to orchestrator / Team A)
```

---

## Bootstrap Path (iteration 0)

```
DataProfile
    → build_state_from_profile()   (Dev B)
    → StateObject  [iteration=0]
    → action_generator.py          detects iteration==0, returns fixed baseline set
    → rest of pipeline runs identically
    → ActionDecision
```

---

## Stub Contract

A stub is NOT a comment or `pass`. It must:
- Have the correct function signature
- Return a valid typed object
- Use neutral/default values that don't break downstream

When horizontal depth is added, only the function body changes. No interface changes.

---

## Build Order

1. `action_generator.py`
2. `learning/dataset_builder.py`
3. `learning/value_model.py`
4. `learning/calibration.py`
5. `learning/uncertainty.py`
6. `agents/performance_agent.py`
7. `agents/efficiency_agent.py`
8. `agents/stability_agent.py`
9. `agents/coordinator_agent.py`
10. `policy/action_selector.py`
11. `logging/decision_logger.py`
12. `validation/counterfactual.py`

---

## Progress Tracker

| # | File | Built | Notes |
|---|---|---|---|
| 1 | `decision/actions/action_generator.py` | ☐ | |
| 2 | `decision/learning/dataset_builder.py` | ☐ | |
| 3 | `decision/learning/value_model.py` | ☐ | |
| 4 | `decision/learning/calibration.py` | ☐ | |
| 5 | `decision/learning/uncertainty.py` | ☐ | |
| 6 | `decision/agents/performance_agent.py` | ☐ | |
| 7 | `decision/agents/efficiency_agent.py` | ☐ | |
| 8 | `decision/agents/stability_agent.py` | ☐ | |
| 9 | `decision/agents/coordinator_agent.py` | ☐ | |
| 10 | `decision/policy/action_selector.py` | ☐ | |
| 11 | `decision/logging/decision_logger.py` | ☐ | |
| 12 | `decision/validation/counterfactual.py` | ☐ | |
|---|---|---|---|
| 1 | `decision/actions/action_generator.py` | YES | |
| 2 | `decision/learning/dataset_builder.py` | YES | |
| 3 | `decision/learning/value_model.py` | YES | |
| 4 | `decision/learning/calibration.py` | YES | |
| 5 | `decision/learning/uncertainty.py` | YES | |
| 6 | `decision/agents/performance_agent.py` | YES | |
| 7 | `decision/agents/efficiency_agent.py` | YES | |
| 8 | `decision/agents/stability_agent.py` | YES | |
| 9 | `decision/agents/coordinator_agent.py` | YES | |
| 10 | `decision/policy/action_selector.py` | YES | |
| 11 | `decision/logging/decision_logger.py` | YES | |
| 12 | `decision/validation/counterfactual.py` | YES | |
