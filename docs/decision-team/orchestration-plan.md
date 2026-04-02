# Orchestration Integration Plan

## Status

Decision engine is built and tested (43/43 passing).
Orchestrator (`stratml/orchestration/orchestrator.py`) is built by the Execution team.

**Blocked on:** Dev B wiring `StateObject` construction into the orchestrator's
`send_profile` and `send_result` callbacks.

---

## What the Orchestrator Expects

Two callables injected at construction:

```python
SendProfileFn = Callable[[DataProfile], ActionDecision]   # iteration 0
SendResultFn  = Callable[[ExperimentResult], ActionDecision]  # iteration 1+
```

These are the two integration seams. Decision engine plugs in here.

---

## What Decision Engine Provides

A single entry point that wraps the full pipeline:

```
decision_engine.receive_profile(profile: DataProfile) -> ActionDecision
decision_engine.receive_result(result: ExperimentResult) -> ActionDecision
```

Each call internally runs:

```
StateObject
    → action_generator
    → dataset_builder (record)
    → value_model → calibration → uncertainty
    → performance_agent + efficiency_agent + stability_agent
    → coordinator_agent
    → action_selector
    → decision_logger (write to disk)
    → counterfactual (record)
    → ActionDecision
```

---

## File to Create (when Dev B is ready)

```
stratml/decision/engine.py
```

```python
class DecisionEngine:
    def receive_profile(self, profile: DataProfile) -> ActionDecision: ...
    def receive_result(self, result: ExperimentResult) -> ActionDecision: ...
```

This is the only file that needs to be created to complete integration.
All 12 pipeline components are already built and tested.

---

## Wiring into Orchestrator

```python
from stratml.decision.engine import DecisionEngine
from stratml.orchestration.orchestrator import ExecutionOrchestrator

engine = DecisionEngine()

orchestrator = ExecutionOrchestrator(
    send_profile=engine.receive_profile,
    send_result=engine.receive_result,
)

orchestrator.run("data/iris.csv", "species")
```

---

## Dependency Chain

```
ExperimentResult  ←  Execution team produces          [DONE]
StateObject       ←  Dev B builds from ExperimentResult  [PENDING]
ActionDecision    ←  Decision engine produces          [DONE]
engine.py         ←  Thin wrapper, create when Dev B ready  [PENDING]
orchestrator.py   ←  Already wired for callbacks       [DONE]
```

---

## Checklist (complete when Dev B updates orchestrator)

- [ ] Confirm `StateObject` schema unchanged (frozen contract)
- [ ] Create `stratml/decision/engine.py` with `DecisionEngine` class
- [ ] Pass `ExperimentHistory` instance across iterations (shared state)
- [ ] Pass `DataProfile` through to `build_state` for dataset fields
- [ ] Track `models_tried` and `remaining_budget` across iterations
- [ ] Wire `engine.receive_profile` + `engine.receive_result` into orchestrator
- [ ] Run integration test: `pytest tests/integration/`
