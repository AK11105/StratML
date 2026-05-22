# Known Gaps & Suggested Fixes

## ~~1. Demo intercept in `cli/main.py`~~ ✅ RESOLVED

**Resolution:** `_DEMO_MAP` block commented out in `cli/main.py`. All datasets now route through the real pipeline. `demo/` scripts kept as standalone references.

---

## ~~2. LangSmith tracing is silently inactive on every run~~ ✅ RESOLVED

**Resolution:** `load_dotenv` is already called at startup in both `cli/main.py` and `signals.py`, and `python-dotenv` is a declared dependency. A `.env` file exists at the project root. To activate LangSmith tracing, add the following keys to `.env`:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_PROJECT=stratml
```

No code changes needed — LangChain picks up the env vars automatically once set.

---

## ~~3. DL pipeline never fires in practice~~ ✅ BY DESIGN

**Resolution:** ML and DL are intentionally separate run modes. `--dl` (or `deep_learning.enabled: true` in config) sets `allowed_models = ["MLP"]` and passes DL hyperparams to the engine. Without `--dl`, the decision engine only sees ML models — this is correct. Auto-escalating to MLP on tabular data would be wrong by default and would complicate the decision logic. No change needed.

---

## 4. Value model stuck in stub mode on fresh runs ⏳ DEFERRED

**Gap (confirmed):** `value_model.py` requires 50 rows with `observed_gain` in `runs/decision_logs/decision_dataset.csv` before RandomForest activates. Currently 43 rows exist but only 18 have `observed_gain` filled — the last decision of every run never gets backfilled because backfill happens on the *next* result arriving, which never comes at run end.

**Planned fix:** Write a data collection script that runs the pipeline across multiple datasets and records real `(state, action, observed_gain)` tuples to seed the unified dataset with enough quality rows to activate the RF. This is a larger effort — deferred to end.

---

## ~~5. LangGraph deprecation warning~~ ✅ NOT A GAP

**Resolution:** LangGraph 1.0.10 is installed. `from langgraph.prebuilt import create_react_agent` imports cleanly with no warning. The suggested fix (`langchain.agents.create_react_agent`) would be incorrect — that is a different, older implementation. No change needed.

---

## ~~6. `ActionDecision` test fixture passes `reason` as a string~~ ✅ NOT A GAP

**Resolution:** `schemas.py` defines an explicit `BeforeValidator(_coerce_reason)` that intentionally coerces strings to `DecisionReason`. This is not silent Pydantic magic — it's a deliberate design choice and will work under strict mode. No change needed.
