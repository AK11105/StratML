# Decision Engine — Next Steps: Rule-Based → LLM-Backed Multi-Agent

## Context

The vertical flow is complete and all 46 tests pass. The pipeline shape is correct:

```
StateObject → action_generator → value_model → calibration → uncertainty
           → [performance_agent, efficiency_agent, stability_agent]
           → coordinator_agent → action_selector → ActionDecision
```

The problem: every component in this pipeline is deterministic. The "agents" are lookup
tables. The coordinator is a weighted sum. The value model returns constants. The
`reason.source` is hardcoded to `"rule"`. This is not a multi-agent system — it is a
scoring function with a multi-agent-shaped API.

The goal of this phase is to replace the rule-based internals with LLM-backed reasoning
while keeping every interface contract unchanged.

---

## What "Multi-Agent" Actually Means Here

Each agent must have its own LLM call with its own system prompt, its own perspective,
and its own structured output. The agents must be able to disagree. The coordinator must
resolve disagreement through reasoning, not arithmetic.

This is the only defensible definition of multi-agent in this context.

---

## LLM Provider

API-based (not local) due to latency constraints. LangChain's abstraction layer will be
used so the provider can be swapped. Target: OpenAI (`gpt-4o-mini` for speed/cost,
`gpt-4o` for quality). API key via environment variable `OPENAI_API_KEY`.

All LLM calls use structured output (Pydantic models via `.with_structured_output()`).
No free-text parsing.

---

## Phase 1 — LLM Backbone for Each Specialist Agent

**Files touched:** `agents/performance_agent.py`, `agents/efficiency_agent.py`,
`agents/stability_agent.py`

**What changes:** Each agent currently returns a hardcoded score from a lookup dict.
Replace each with a LangChain chain that:

1. Receives a structured prompt containing the relevant slice of `StateObject` and the
   candidate action list
2. Reasons from its specialist perspective
3. Returns a structured score dict (`dict[str, float]`) via `.with_structured_output()`

Each agent has a distinct system prompt that defines its perspective:

**Performance agent system prompt framing:**
> You are a performance analyst. Your only concern is whether a candidate action will
> improve the model's predictive accuracy given the current fitting state and trajectory.
> Score each action 0.0–1.0 on expected performance gain.

**Efficiency agent system prompt framing:**
> You are a resource analyst. Your only concern is compute cost and budget consumption.
> Score each action 0.0–1.0 on efficiency (higher = cheaper relative to remaining budget).

**Stability agent system prompt framing:**
> You are a training stability analyst. Your only concern is whether a candidate action
> risks destabilizing training given current variance, divergence, and loss signals.
> Score each action 0.0–1.0 on stability (higher = safer).

The function signatures stay identical — `score(state, estimates) -> dict[str, float]`.
The coordinator receives the same inputs as before. Nothing downstream changes.

**Fallback:** If the LLM call fails or returns malformed output, fall back to the
existing lookup-table logic. This makes the agents resilient without removing the
rule-based baseline.

---

## Phase 2 — LLM Coordinator with Explicit Deliberation

**File touched:** `agents/coordinator_agent.py`

**What changes:** The coordinator currently computes:
```
final_score = 0.5 * perf + 0.25 * eff + 0.25 * stab
```

This is not deliberation. Replace with a LangChain chain that:

1. Receives all three agent scores, the full candidate list, and a summary of the
   `StateObject` (trajectory, signals, resources, constraints)
2. Is explicitly told that the three agents have already scored the candidates from their
   own perspectives and may disagree
3. Reasons about which action to select given the current experiment context — it can
   override agent scores if the situation warrants it (e.g., budget is nearly exhausted
   so efficiency should dominate even if performance agent disagrees)
4. Returns a ranked list with a natural-language rationale per action

The rationale string gets stored in `DecisionReason.evidence["rationale"]`. This is the
trace that LangSmith will capture.

The `reason.source` field changes from `"rule"` to `"learned"` when the LLM path is
taken.

**Why this is Phase 2 and not Phase 1:** The coordinator's reasoning is only meaningful
if the agent scores it receives are themselves meaningful. If the agents are still
returning lookup-table values, the coordinator is reasoning over noise. Phase 1 must
come first.

---

## Phase 3 — LLM-Backed Candidate Generation

**File touched:** `actions/action_generator.py`

**What changes:** The generator currently uses an if-else tree on `state.signals`. It
can only propose actions it was explicitly programmed for and cannot compose them.

Replace with a LangChain chain that reads the full `StateObject` and proposes a
`list[CandidateAction]` via structured output. The LLM can propose:

- Actions the rules don't cover
- Composed parameter choices (e.g., `modify_regularization` with a specific alpha value
  derived from the current gap magnitude, not just `{"direction": "increase"}`)
- Context-sensitive model suggestions (e.g., recommending `GradientBoosting` specifically
  because the dataset has high imbalance ratio and the current model is `LogisticRegression`)

The rule-based generator becomes the fallback. The interface is unchanged:
`generate(state) -> list[CandidateAction]`.

**Why this is Phase 3:** Expanding the candidate space before the coordinator can reason
well (Phase 2) means bad candidates could get selected. The reasoning layer must be solid
before the input space grows.

---

## Phase 4 — Activate the Learning Stubs

**Files touched:** `learning/value_model.py`, `learning/calibration.py`,
`learning/uncertainty.py`

**What changes:** These are currently stubs returning constants. Once enough rows exist
in `decision_dataset.csv` (from real runs through Phases 1–3), activate them:

- `value_model.py`: Train `RandomForestRegressor` on `(state_features, action_type) →
  observed_gain`. The `observed_gain` column in the CSV is filled retroactively when the
  next `ExperimentResult` arrives.
- `calibration.py`: Fit `IsotonicRegression` on `(predicted_gain, actual_gain)` pairs.
- `uncertainty.py`: Replace the stub ensemble with 4–5 actual model predictions; compute
  variance across them as the confidence signal.

At this point the LLM agents in Phases 1–2 receive richer inputs — calibrated predicted
gains with confidence intervals — so their reasoning is grounded in empirical data from
actual runs, not just structural signals.

**Threshold to activate:** ~50 rows in `decision_dataset.csv` is enough to train a
meaningful value model. Below that, keep the stubs.

---

## File Change Summary

| File | Current State | Target State |
|---|---|---|
| `agents/performance_agent.py` | Lookup dict | LangChain chain + fallback |
| `agents/efficiency_agent.py` | Lookup dict | LangChain chain + fallback |
| `agents/stability_agent.py` | Lookup dict | LangChain chain + fallback |
| `agents/coordinator_agent.py` | Weighted sum | LangChain chain with deliberation |
| `actions/action_generator.py` | If-else on signals | LangChain chain + rule fallback |
| `learning/value_model.py` | Stub (0.05 constant) | Trained RandomForest |
| `learning/calibration.py` | Pass-through | Fitted IsotonicRegression |
| `learning/uncertainty.py` | Stub (0.5 constant) | Ensemble variance |
| `policy/action_selector.py` | Picks rank[0] | Unchanged — coordinator handles ranking |
| `core/schemas.py` | Frozen | Unchanged |

---

## Interface Contracts — Unchanged

Nothing in `core/schemas.py` changes. `StateObject` in, `ActionDecision` out. The
orchestrator integration (`engine.py`) is unaffected. Tests for the existing rule-based
behavior remain valid as fallback coverage.

---

## LangSmith Tracing

Every LLM call in Phases 1–3 is automatically traced by LangSmith if
`LANGCHAIN_TRACING_V2=true` is set. The coordinator's rationale string in
`DecisionReason.evidence["rationale"]` provides human-readable decision traces per
iteration. This is the observability story for the decision layer.

---

## Build Order

1. Phase 1: `performance_agent` → `efficiency_agent` → `stability_agent`
2. Phase 2: `coordinator_agent`
3. Phase 3: `action_generator`
4. Phase 4: activate stubs (data-gated, not time-gated)

Do not start Phase 2 until Phase 1 agents are returning meaningful LLM scores.
Do not start Phase 3 until Phase 2 coordinator is producing coherent rationales.
Phase 4 is triggered by data volume, not by completing Phase 3.

---

## Phase 5 — LangGraph: Intra-Decision Cycles

Phases 1–4 produce a linear pipeline. Every decision cycle is a single forward pass —
agents score, coordinator picks, done. LangGraph becomes the right tool when the
coordinator needs to loop back within a single decision cycle.

### Trigger condition

Phase 5 is warranted when any of these situations arise naturally from real runs:

**1. Low-confidence deadlock**
The coordinator receives three agent scores that are genuinely conflicting and its
confidence in the top-ranked action is below a threshold (e.g., `confidence < 0.4`).
Rather than forcing a pick, the coordinator should be able to send a targeted follow-up
question back to the disagreeing agent and re-score before committing.

With LangChain this requires manual orchestration code. With LangGraph it is a
conditional edge: `coordinator → re_query_agent → coordinator`.

**2. Candidate rejection loop**
The action generator proposes candidates, but the coordinator determines all of them are
unsuitable given the current state (e.g., budget too low for any `switch_model`, but
`terminate` is premature). It needs to send a constraint back to the generator and
request a revised candidate set. This is a cycle: `coordinator → action_generator →
coordinator`. LangGraph handles this natively; LangChain requires you to write the loop
yourself.

**3. Multi-step validation before commit**
Post Phase 4, the value model has real predictions. Before the coordinator commits to an
action, it may want to verify: "does the calibrated gain justify the predicted cost given
remaining budget?" This is a conditional branch — if yes, commit; if no, ask the
efficiency agent to re-evaluate with the calibrated gain as additional context. A
LangGraph conditional edge handles this cleanly.

### What changes architecturally

The `engine.py` entry point stays identical — `receive_result(result) -> ActionDecision`.
Internally, the linear chain:

```
action_generator → agents → coordinator → action_selector
```

becomes a LangGraph graph with the same nodes but with conditional edges between
coordinator and the agents/generator. The `StateObject` becomes the LangGraph state
dict (it already has the right shape for this — it's a Pydantic model with all relevant
fields).

`action_selector.py` and `decision_logger.py` remain pure Python — they sit outside the
graph as post-processing steps on the graph's final output.

### What does NOT change

- `core/schemas.py` — frozen, untouched
- `StateObject` contract — unchanged
- `ActionDecision` output — unchanged
- All existing tests — still valid

LangGraph is a drop-in replacement for the internal orchestration of the decision cycle.
Nothing outside `engine.py` and the agent files needs to know it happened.
