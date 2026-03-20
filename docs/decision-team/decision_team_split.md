# 🧠 Decision Team Development Split (2 Developers)

This document defines how two developers should split work while building the **Decision Layer**.

---

# 🔥 Core Principle

Split by responsibility:

- **Dev A (Decision Engine)** → decides what action to take
- **Dev B (State Pipeline)** → builds understanding of experiment

They connect through:

```
StateObject
```

---

# 👤 Dev B — Input / State Pipeline

## Responsibilities

### 1. core/schemas.py
- ExperimentResult
- StateObject
- CandidateAction (basic structure)

---

### 2. decision/state_builder.py
Input:
```
ExperimentResult
```

Output:
```
StateObject
```

---

### 3. decision/state_history.py
- Maintain last 3 experiments
- Compute:
  - improvement_rate
  - accuracy_slope
  - runtime_trend

---

### 4. decision/meta_features.py
- Dataset features:
  - num_samples
  - num_features
  - class_entropy
  - missing ratio

---

### 5. decision/signals.py
Convert state → signals:

- underfitting
- overfitting
- convergence

---

## Final Output

```
StateObject = {
  raw_metrics,
  trajectory_features,
  dataset_features,
  signals
}
```

---

# 👤 Dev A — Decision Engine

## Responsibilities

### 1. decision/action_generator.py

Input:
```
StateObject
```

Output:
```
[CandidateAction, CandidateAction...]
```

---

### 2. decision/policy_selector.py (CORE)

Input:
- state
- candidate actions

Output:
```
ActionDecision
```

---

### 3. decision/learning/value_model.py

Train model:
```
state + action → gain
```

---

### 4. decision/learning/uncertainty.py
- Ensemble predictions
- Variance → confidence

---

### 5. decision/agents/
- performance_agent
- efficiency_agent
- stability_agent

---

### 6. decision/agents/coordinator_agent.py
- Combine agent scores

---

### 7. decision/learning/calibration.py
- Adjust predicted gains

---

# 🤝 Integration Point

```
StateObject  ← Dev B
      ↓
ActionDecision ← Dev A
```

---

# ⚠️ CRITICAL RULE

```
StateObject = CONTRACT
DO NOT MODIFY AFTER FREEZE
```

---

# 🚀 Day 1 Plan

## Dev B
- schemas.py
- state_builder.py

## Dev A
- action_generator.py (rule-based)
- policy_selector.py (rule-based)

---

# 🎯 Day 1 Goal

```
StateObject → ActionDecision (WORKING)
```

---

# 🧠 Development Phases

## Phase 1
- Working loop with rule-based decisions

## Phase 2
- Add:
  - state_history
  - signals

## Phase 3
- Add:
  - dataset_builder
  - value_model

## Phase 4
- Add:
  - agents
  - uncertainty
  - calibration

---

# 🧠 Final Mental Model

```
Dev B = builds understanding
Dev A = builds decision brain
```
