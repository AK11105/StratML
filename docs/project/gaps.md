# Known Gaps & Suggested Fixes

## 1. Demo intercept in `cli/main.py`

**Gap:** Lines 88–101 of `stratml/cli/main.py` check the dataset stem against a hardcoded map and route to `demo/` scripts, bypassing the real pipeline entirely for known datasets (titanic, pima, mnist, etc.).

**Fix:** Remove or comment out the `_DEMO_MAP` block. The real pipeline handles all these datasets correctly. Keep `demo/` scripts as standalone files for reference only.

---

## 2. `stratml/observability/` and `stratml/utils/` are empty

**Gap:** Both modules are `__init__.py` stubs. The README promises TensorBoard and LangSmith integration. TensorBoard log directories are created by the orchestrator but nothing is written to them. LangSmith is not wired anywhere.

**Fix (TensorBoard):** In `dl_pipeline.py`, add a `SummaryWriter` that logs train/val loss per epoch to the `tb_dir` already passed in from the orchestrator.

**Fix (LangSmith):** Add `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` to `.env` and `sample.env`. LangChain/LangGraph auto-traces when these are set — no code changes needed beyond documenting it.

---

## 3. DL pipeline never fires in practice

**Gap:** `build_experiment_config` only routes to DL for model names `MLP` or `PyTorchMLP`. The action generator's rule path never proposes these, and the LLM path only does so if explicitly prompted. On any tabular dataset demo, DL is unreachable.

**Fix:** Add `"MLP"` to `_DEFAULT_MODELS` and `_BOOTSTRAP_MODELS` in `action_generator.py`, or add a rule in `_rule_candidates` that proposes `MLP` when `num_features > 20` and no underfitting/overfitting signal is present.

---

## 4. Value model stuck in stub mode on fresh runs

**Gap:** `value_model.py` requires 50 rows of `observed_gain` in `runs/decision_logs/decision_dataset.csv` before the RandomForest activates. On any fresh environment the model returns a flat `predicted_gain=0.05` for all candidates, making the "learned decision advisor" claim hollow until data accumulates.

**Fix (short-term):** Lower `_MIN_ROWS` from 50 to 10, or seed `runs/decision_logs/decision_dataset.csv` with the existing output data from `outputs/*/decision_logs/decision_dataset.csv` rows.

**Fix (long-term):** Add a `stratml seed-memory` CLI command that consolidates all past run decision datasets into the unified path.

---

## 5. LangGraph deprecation warning

**Gap:** `stratml/decision/state/signals.py` imports `create_react_agent` from `langgraph.prebuilt`, which is deprecated since LangGraph v1.0 and will break in v2.0.

**Fix:** Change the import:
```python
# before
from langgraph.prebuilt import create_react_agent
# after
from langchain.agents import create_react_agent
```

---

## 6. `ActionDecision` test fixture passes `reason` as a string

**Gap:** Several test helpers pass `reason="test"` as a plain string to `ActionDecision`, which expects a `DecisionReason` object. Pydantic currently coerces this silently, but it will fail under strict mode and masks schema drift.

**Fix:** Update `_action()` helpers in `tests/integration/test_full_pipeline.py` and `tests/unit/test_config_builder.py` to pass a proper `DecisionReason`:
```python
from stratml.core.schemas import DecisionReason
reason=DecisionReason(trigger="test", evidence={}, source="rule")
```
