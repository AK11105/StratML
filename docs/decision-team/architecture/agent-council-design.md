# Agent Council — Design Reference

Documents the design of the multi-agent decision council: the three specialist agents,
the coordinator, and how they interact.

---

## Overview

The council sits between the uncertainty estimator and the action selector. It takes
`list[UncertaintyEstimate]` (one per candidate action) and produces `list[RankedAction]`
sorted by final score descending.

```
list[UncertaintyEstimate]
        │
        ├── PerformanceAgent  → dict[action_type → perf_score]
        ├── EfficiencyAgent   → dict[action_type → eff_score]
        └── StabilityAgent    → dict[action_type → stab_score]
                │
                ▼
        CoordinatorAgent
        → list[RankedAction]  (sorted descending by final_score)
                │
                ▼
        ActionSelector → ActionDecision
```

---

## Specialist Agents

Each agent has the same interface:

```python
def score(state: StateObject, estimates: list[UncertaintyEstimate]) -> dict[str, float]:
```

Returns a score per `action_type` in `[0.0, 1.0]`. Higher = better from that agent's
perspective.

### PerformanceAgent (`performance_agent.py`)

Concern: will this action improve the primary metric?

Rule fallback uses a lookup table keyed on fitting state:

| Fitting state | switch_model | increase_capacity | modify_reg | decrease_capacity | terminate |
|---|---|---|---|---|---|
| underfitting | 0.85 | 0.80 | 0.40 | 0.10 | 0.05 |
| overfitting | 0.60 | 0.10 | 0.85 | 0.75 | 0.20 |
| well_fitted | 0.30 | 0.20 | 0.20 | 0.20 | 0.90 |

Plateau signal adds a +0.15/+0.30 boost to `switch_model` and penalises
`modify_regularization` by −0.20.

LLM prompt context: fitting signals, trajectory slope, best_score, model name.

### EfficiencyAgent (`efficiency_agent.py`)

Concern: what is the compute cost of this action relative to remaining budget?

Rule fallback scores based on action cost proxy and remaining budget ratio. Actions that
consume more compute score lower when budget is tight.

LLM prompt context: runtime, remaining_budget, budget_exhausted, action cost proxies.

### StabilityAgent (`stability_agent.py`)

Concern: will this action destabilise training?

Rule fallback uses a base stability table:

| Action | Base stability |
|---|---|
| terminate | 0.95 |
| modify_regularization | 0.85 |
| decrease_model_capacity | 0.80 |
| change_optimizer | 0.70 |
| switch_model | 0.65 |
| increase_model_capacity | 0.35 |

Instability penalties applied when `diverging`, `unstable_training`, or `high_variance`
signals are present. Variance from the uncertainty estimator also penalises the score.

LLM prompt context: diverging, unstable_training, high_variance signals, generalization
gap, trajectory volatility.

---

## CoordinatorAgent (`coordinator_agent.py`)

Resolves disagreements between the three specialist agents and produces the final ranking.

### Rule fallback

Weighted sum with fixed weights:

```
final_score = 0.50 * perf + 0.25 * eff + 0.25 * stab
```

### LLM path

Sends the full state summary and all three agent scores per candidate to the LLM. The
coordinator is instructed to reason about which agent should dominate given the current
context (e.g., budget nearly exhausted → efficiency should dominate) and return a ranked
list with a one-sentence rationale per action.

The rationale string travels through `RankedAction.rationale` → `ActionDecision.reason.evidence["rationale"]`.

### RankedAction schema

```python
@dataclass
class RankedAction:
    action_type: str
    parameters: dict
    predicted_gain: float
    predicted_cost: float
    confidence: float
    agent_scores: AgentScore      # perf, eff, stab scores
    final_score: float
    rationale: str                # empty string on rule path
```

---

## Decision Source Tracing

`ActionSelector` sets `reason.source` based on whether the coordinator produced a
rationale:

- `"learned"` — LLM coordinator path was taken
- `"rule"` — rule fallback was used

This is logged in every `decision_<iteration>.json` file, making it easy to audit which
decisions were LLM-driven vs rule-driven across a run.

---

## LLM Configuration

All four agents use the same model: `llama-3.3-70b-versatile` via `langchain-groq`.
Temperature is 0 for all agents except the action generator (temperature=0.2 to allow
creative candidate proposals).

All LLM calls use structured output (`with_structured_output`) — no free-text parsing.
Every agent catches all exceptions and falls back to the rule path, logging a warning.
The system is fully functional without `GROQ_API_KEY`.
