# 🔵 FINAL TEAM STRUCTURE (EXPLICIT VERSION)

You will operate in **two modes over the semester**:

### Mode 1 (Foundation Phase)

You are Execution Anchor.

### Mode 2 (Stabilization Phase)

You step back from execution and move to Decision & Research layer.

If you don’t transition, you will burn out.

---

# 🧱 TEAM A — Execution Engine

### 👤 You — Execution Architect (Phase 1–2 Heavy)

You DO:

1. Write the base training loop (PyTorch)
2. Define DL template structure (MLP baseline only)
3. Define sklearn pipeline interface standard
4. Define ExperimentResult structure
5. Define metric naming conventions
6. Define early stopping logic
7. Define cost estimation logic
8. Write minimal working end-to-end flow

You DO NOT:

* Implement every model variant
* Write all preprocessing
* Tune DL endlessly
* Maintain TensorBoard wiring

Your job is to create a **clean skeleton**.

After skeleton is stable → you stop writing execution code.

---

### 👤 Member 2 — DL Implementation Owner

They DO:

1. Extend DL templates (add capacity variations)
2. Implement additional architectures (if needed)
3. Integrate TensorBoard logging
4. Improve training stability
5. Handle GPU/debug issues
6. Maintain MLflow model logging

They DO NOT:

* Redesign training loop
* Change metric schema
* Modify ExperimentResult without discussion

They build ON TOP of your baseline.

---

### 👤 Member 3 — ML & Pipeline Owner

They DO:

1. Dataset ingestion module
2. Schema detection
3. Preprocessing abstraction
4. sklearn models
5. Cross-validation
6. Metric computation

They DO NOT:

* Add new metric names randomly
* Modify output structure
* Touch DL training code

You review architecture once, then let them own it.

---

# 🟣 TEAM B — Decision Engine

### 👤 Member 1 — Controller & Orchestration Lead

They DO:

1. Implement Orchestrator
2. Implement Lifecycle Manager
3. Implement Budget Monitor enforcement
4. Implement LangGraph state machine
5. Implement Action schema
6. Integrate LangSmith tracing
7. Wire CLI to Orchestrator

They DO NOT:

* Define ML signals
* Write rule logic content
* Touch training loops

They implement control structure.

---

### 👤 You — Decision Logic & Research Lead (Phase 3+ Heavy)

You DO:

1. Define StateObject structure
2. Define signal extraction logic:

   * Underfitting detection
   * Overfitting detection
   * Slow convergence detection
   * Instability detection
   * Diminishing returns detection
3. Define Action space
4. Define Rule set
5. Define termination logic
6. Define evaluation metrics for decision quality
7. Write research comparison methodology

You DO NOT:

* Implement state machine transitions
* Write LangGraph plumbing
* Maintain CLI

You define logic.
Member 1 implements logic.

---

# 🔁 PHASE TRANSITION (CRITICAL)

### Phase 1–2:

You are 60% Execution, 40% Decision.

### Phase 3 onward:

You become 70% Decision, 30% Oversight.

If you don’t shift like this → timeline collapses.

---

# 📦 CONTRACT FREEZE (YOU MUST ENFORCE)

Before anyone builds anything, freeze:

## ExperimentResult

Must contain:

* primary_metric
* validation_metric
* loss_curve
* convergence_epoch
* runtime_seconds
* parameter_count (DL only)
* memory_estimate
* status

No one changes this mid-semester.

---

## Action Space (Keep Small)

* run_baseline
* switch_model
* increase_capacity
* decrease_capacity
* early_stop
* terminate

If someone suggests adding 10 more actions → reject.

---

# ⚠️ WHERE YOU WILL BECOME A BOTTLENECK

1. If only you understand DL training loop.
2. If only you understand signal extraction.
3. If you don’t document ExperimentResult properly.
4. If people depend on you to debug every issue.

So you must:

* Write documentation.
* Add comments.
* Create minimal examples.
* Enforce code reviews.

---

# 🧠 RISK MITIGATION PLAN

## Risk 1 — DL Instability

Solution:
You write stable baseline.
Member 2 only modifies within safe bounds.

---

## Risk 2 — Poor ML Signal Extraction

Solution:
You explicitly define signal formulas.
Not vague “looks like underfitting.”

Example:

Underfitting =
validation_loss decreasing + accuracy plateau + low training accuracy.

Write these formally.

---

## Risk 3 — Overengineering Controller

Solution:
Member 1 implements simple state machine.
No dynamic graph mutation.
No fancy reasoning.

---

# 🎯 WHAT YOU MUST NOT DO

* Do all DL coding.
* Fix everyone’s bugs.
* Hide architecture in your head.
* Add learned advisor this semester.
* Expand action space.

---

# 🧠 HARD TRUTH

Right now, you are:

* The only deep ML engineer.
* The intellectual center of the idea.

So your role is:

> Architect → Enable → Transition → Evaluate

Not:

> Code everything → Collapse → Burn out

---

# 🏁 FINAL STRUCTURE SUMMARY

Execution Team:

* You (foundation + DL baseline)
* Member 2 (DL extension + infra)
* Member 3 (ML pipelines)

Decision Team:

* Member 1 (controller & orchestration)
* You (rules + signals + research)

You bridge both early, then specialize.