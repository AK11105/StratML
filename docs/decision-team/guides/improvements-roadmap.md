# Decision Team — Known Weaknesses & Proposed Improvements

Honest audit of the current decision team. Items are ordered by recommended fix sequence —
each item either unblocks the next or is a quick win that should be done first.

---

## 1. ReAct Agent Rebuilt on Every Call (Quick Fix)

### What is happening

In `signals.py`, `_build_agent()` is called inside `compute_signals()` on every single
iteration. It instantiates a new `ChatGroq` client and a new LangGraph ReAct agent every
time the signal pipeline runs — which is once per iteration.

### Why it matters

Over 20 iterations this adds unnecessary latency and redundant object construction. The
agent has no state between calls so there is no reason to rebuild it. It compounds with
the other LLM calls per iteration (action generator, coordinator, three specialist agents).

### Proposed solution

Make the agent a module-level singleton. Build it once on first use with a simple
`_AGENT = None` guard and reuse it for the lifetime of the process. One-liner fix.

---

## 2. Value Model Feature Encoding Is Lossy and Fragile

### What is happening

In `value_model.py`, `action_type` is encoded as:

```python
action_enc = abs(hash(action_type)) % 100
```

Hash collisions are possible across action types. More critically, the feature vector
drops all hyperparameter information — two `switch_model` candidates with different
`model_name` values look identical to the RandomForest. The model cannot learn that
`switch_model(RandomForest)` behaves differently from `switch_model(SVC)`.

### Why it matters

The value model's predictions feed directly into the uncertainty pipeline, which feeds
into all three specialist agents, which feed into the coordinator. If the value model
cannot distinguish between semantically different candidates, the entire scoring pipeline
is working on degraded input. Fixing coordinator weights (item 6) before fixing this
would be building on a weak foundation.

### Proposed solution

Replace the hash encoding with a stable integer lookup over a fixed vocabulary of action
types (no collision risk). Add a `model_name_enc` feature using the same vocabulary
pattern over `allowed_models`. Add a `complexity_hint_enc` feature from the existing
`StateModel.complexity_hint` field. These are all already present in `StateObject` —
it is purely an encoding change in `_encode_state_action()`.

---

## 3. Missing: Evaluator Agent (Post-Hoc Decision Auditor)

### Background

Planned addition motivated by:

> Du, Ahlawat, Liu, Wu — "A Framework for Assessing AI Agent Decisions and Outcomes
> in AutoML Pipelines" (2026). [lit review ref 2]

The paper's Evaluation Agent (EA) performs post-hoc audits of intermediate decisions.
Unlike the existing agents which score candidates before execution, the EA looks at what
actually happened after execution and asks whether the decision was good. The paper
reports F1 of 0.919 for fault detection and identified performance impacts ranging from
−4.9% to +8.3% from faulty decisions that went undetected.

### What it audits (four dimensions)

| Dimension | Question |
|---|---|
| Decision validity | Was the action type appropriate for the observed state signals? |
| Reasoning consistency | Did the stated trigger actually match the signals present? |
| Quality risk | Risks beyond primary metric (generalization gap, instability)? |
| Counterfactual impact | Was the actual gain better or worse than the expected gain? |

### Where it fits

Post-hoc observer — not in the decision loop. Runs inside `receive_result()` in
`engine.py` after the state is built, looking at the previous decision vs. the outcome
that just arrived.

```
receive_result(result)
    ├── backfill_last_gain()          ← already exists
    ├── build_state()                 ← already exists
    ├── _decide(state)                ← already exists
    └── evaluator_agent.audit(        ← new
            decision=self._last_decision,
            result=result,
            state=state
        )
```

Requires storing `self._last_decision` in the engine — one-line addition.

### Rule-based fallback

- Decision validity: does the action type belong to the expected set for the trigger?
  (e.g., trigger=underfitting → expected={switch_model, increase_model_capacity})
- Reasoning consistency: is the trigger's signal actually present in the state at
  decision time?
- Quality risk: generalization gap magnitude + instability signal presence
- Counterfactual impact: actual gain minus expected gain from the decision record

### LLM path

Sends full decision context (action, trigger, signals, actual outcome) to the coordinator
LLM and asks it to score all four dimensions with a brief rationale. Same
Groq/llama-3.3-70b-versatile pattern as the other agents. Falls back to rule-based on
failure.

### Output

Appends to `outputs/<run_id>/decision_logs/evaluation_log.jsonl`. Each entry is one
`EvaluationRecord` with all four dimension scores, a `fault_detected` boolean, and a
`notes` string.

### Why this is item 3

The evaluation log is the data source for item 6 (coordinator weight learning) and
directly addresses item 4 (counterfactual stub). It needs to exist before those can be
built properly.

---

## 4. Counterfactual Validator Is a Stub

### What is happening

`counterfactual.py` appends the selected action to a JSONL log and nothing else. The
docstring says "Full implementation: execute an alternative action in parallel and compare
outcomes." That has never been implemented. The log records experiment_id, iteration,
action_type, parameters, confidence — and nothing about the alternative not taken.

### Why it matters

Without counterfactual data you cannot measure whether the chosen action was better than
the runner-up. The value model trains on `decision_dataset.csv` which records predicted
gain and observed gain, but has no signal about whether a different action would have done
better. The evaluator agent (item 3) partially addresses this via the counterfactual
impact dimension, but the raw log should also carry the runner-up for completeness.

### Proposed solution

Two levels, in order of effort:

**Level 1 (no extra compute):** At decision time, record `ranked[1]` (the runner-up) and
its predicted gain alongside the chosen action in the counterfactual log. After the result
comes in, the evaluator agent computes the estimated gain difference. This is an
approximation but costs nothing and is a natural extension of what the evaluator agent
already does.

**Level 2 (proper, more compute):** On a configurable fraction of iterations (e.g., 20%),
execute both the chosen action and the runner-up in parallel and compare actual outcomes.
Requires the orchestrator to support branching — this is the full implementation the stub
was designed for.

Level 1 is the right starting point and can be done as part of the evaluator agent
integration.

---

## 5. ActionSelector Is Trivially Greedy

### What is happening

`action_selector.py` always picks `ranked[0]`:

```python
best = ranked[0]
```

No exploration policy of any kind.

### Why it matters

In early runs (fewer than 50 rows in the decision dataset) the value model returns neutral
stub predictions (0.05 for all candidates). In that regime the coordinator's rule-based
fallback dominates, and always picking the top rule-based recommendation means the system
never explores actions ranked lower by the rules but that might work better for a specific
dataset. The system can get stuck in a local optimum across iterations.

### Proposed solution

Add an epsilon-greedy layer in `action_selector.py`. When the value model is in low-data
regime (fewer than 50 rows), use epsilon = 0.2. Once the value model has sufficient data,
drop to epsilon = 0.05. Both values should be config-driven.

A more principled alternative is Thompson sampling over the existing
`UncertaintyEstimate.variance` field — sample from a distribution parameterised by
`(predicted_gain, variance)` per candidate and pick the argmax. This is a direct use of
infrastructure that already exists and requires no new config fields.

---

## 6. Hardcoded Coordinator Weights

### What is happening

In `coordinator_agent.py`, the rule-based fallback computes:

```python
final = 0.50 * perf + 0.25 * eff + 0.25 * stab
```

These weights never change regardless of how many runs have been completed or how often
each agent's recommendation turned out to be correct.

### Why it matters

A coordinator that cannot update its trust in each agent is not really coordinating — it
is doing a fixed weighted average. If the stability agent is systematically
over-penalising risky actions on small datasets, the system will never learn that. The LLM
path can override this in a single call but has no memory of past overrides either.

### Proposed solution

Use the evaluation log produced by item 3. After each iteration, compare each agent's
top recommendation against the action taken and the gain that resulted. Update per-agent
weights with an exponential moving average:

```
w_agent = (1 - alpha) * w_agent + alpha * (1 if agent_was_right else 0)
```

The coordinator reads these weights at the start of each run from the unified decision
dataset. No new infrastructure needed — the dataset and evaluation log already exist once
items 2 and 3 are done. This is a lightweight version of the hybrid reward mechanism in
I-MCTS (lit review [3]).

---

## 7. Signal Computation Is Stateless Across Iterations

### What is happening

`signals.py` computes each signal independently from the current state snapshot. It has
no memory of what signals were present in the previous iteration. The `StateActionContext`
carries `previous_action` and `previous_action_success` but there is no equivalent
`previous_signals` field.

### Why it matters

Signal transitions carry more information than point-in-time signals. "Was underfitting
last iteration, now well-fitted" is a much stronger termination signal than just
"well-fitted now." "Was well-fitted, now overfitting" after an increase_capacity action
is a clear signal that the action backfired. The action generator and coordinator never
see this transition context, so they cannot reason about whether the last action worked.

### Proposed solution

Add a `previous_signals: Optional[StateSignals]` field to `StateActionContext` (or as a
top-level field on `StateObject`). Populate it in `build_state()` by passing the previous
state's signals through from the engine. The action generator and coordinator can then
check transition pairs (e.g., `prev.underfitting != "none" and curr.well_fitted != "none"`)
to make stronger decisions. This is a schema addition + one extra argument to
`build_state()` — no new modules needed.

---

## 8. No Cross-Run Memory

### What is happening

`ExperimentHistory` is instantiated fresh for every run. The value model reads from a
unified CSV that accumulates across runs, but this is a flat file with no structure for
querying by dataset similarity. The coordinator and specialist agents have no access to
what worked on similar datasets in past runs.

### Why it matters

Every run starts from scratch in terms of decision policy. The value model improves over
time in aggregate but cannot do instance-based retrieval — it cannot answer "I have seen
a dataset with similar characteristics before, what worked then?" This is the gap that
meta-learning addresses. AutoML-Agent (lit review [1]) uses retrieval-augmented planning
for exactly this reason.

### Proposed solution

Add a `MetaMemory` module that indexes completed runs by dataset meta-features.
`DatasetMetaFeatures` already exists in `meta_features.py`. At the start of a new run,
retrieve the top-k most similar past runs by cosine similarity over the meta-feature
vector and inject their best-performing action sequences into the bootstrap candidates.

No vector database needed — cosine similarity over stored meta-feature vectors in numpy
is sufficient at this scale. The integration point is `receive_profile()` in `engine.py`,
before bootstrap candidates are generated.

---

## Summary

| # | Item | Severity | Effort | Dependency |
|---|---|---|---|---|
| 1 | ReAct agent singleton | Low | Trivial | None |
| 2 | Value model encoding | High | Low | None |
| 3 | Evaluator agent | High | Medium | None |
| 4 | Counterfactual stub | High | Low (L1) | Pairs with #3 |
| 5 | Greedy action selector | Medium | Low | None |
| 6 | Hardcoded coordinator weights | Medium | Low | Needs #3 |
| 7 | Stateless signal computation | Medium | Low | None |
| 8 | No cross-run memory | Medium | Medium | Needs #3 data |

Recommended implementation order: **1 → 2 → 3 → 4 → 5 → 7 → 6 → 8**

Items 1 and 2 are quick wins with no dependencies. Item 3 (evaluator agent) is the
central piece — its output unlocks 4, 6, and 8. Items 5 and 7 are independent and can
slot in anywhere after item 2.
