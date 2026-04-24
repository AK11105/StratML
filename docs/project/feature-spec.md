# Feature Specification

Design intent and capability definition for the Multi-Agent Controlled AutoML Framework.
This is a vision document — for current implementation status see the team docs.

---

## Core Identity

> A system that models ML experimentation as an explicit, controllable decision process
> rather than as blind optimization.

---

## I. Foundational Features

### Dataset Ingestion
- Accept datasets via CLI or config file
- Support tabular data (mandatory), extensible to text/vision
- Schema inference: numeric / categorical / datetime / text
- Decision engine requires data scale, feature types, and size to reason correctly

### Standardized Pipeline Interface
- Every pipeline component: `input → process → output(metrics + artifacts)`
- Applies to preprocessors, ML models, DL models
- Without this there is no comparability and no agent reasoning

### Classical ML Execution Engine
- Preprocessing, feature engineering, model training, evaluation, metric extraction
- Defines the baseline experimentation environment

### Structured Logging & Metrics
- Per experiment: pipeline config, metrics, compute usage, convergence behaviour, failure signals
- Logs are the agent's perception layer

### Experiment Lifecycle Manager
- Start, pause, resume, terminate, retry, failure handling
- Required for long DL runs

---

## II. Deep Learning Integration

### Template-Based DL Pipelines
- Tabular MLP (implemented), CNN1D, RNN
- Bounded hyperparameters for stable agent decisions

### DL Training Monitoring
- Loss curves, validation gap, convergence speed
- Agent decisions require training dynamics, not just final metrics

---

## III. Decision Abstraction (Core Intellectual Layer)

### State Representation Engine
- State includes: metrics history, model characteristics, pipeline complexity,
  compute/budget usage, experiment outcomes
- Agent reasons over state, not raw metrics

### Action Space
- switch_model, increase/decrease capacity, modify regularization, change optimizer,
  add preprocessing, terminate
- Defines the controllable decision universe

### Decision Traceability
- Every decision logs: state snapshot, rule/logic fired, chosen action, observed outcome
- Critical for interpretability and research validity

---

## IV. Rule-Based Agent (Baseline Intelligence)

### Rule Engine
- Rules for: underfitting, overfitting, instability, slow learning, budget exhaustion,
  diminishing returns
- Provides stable, deterministic baseline policy

### LangGraph State Machine
- Agent implemented as explicit states, transitions, termination nodes
- Formal control flow — no hidden logic

### Budget-Aware Controller
- Agent reasons over remaining budget, estimated experiment cost, marginal improvement

---

## V. Hybrid Intelligence (Learned + Rule)

### Learned Decision Advisors
- Predict: expected improvement, compute cost, convergence likelihood, instability risk
- Improves decision quality without losing rule-based control

### Advisory Decision Layer
- Learned models score and rank actions but never override rules unilaterally

### Rule–Advisor Arbitration
- Defines when the advisor is consulted, when ignored, and how conflicts resolve

---

## VI. User Extensibility

### Pluggable Pipeline Registry
- Users register preprocessors, ML models, DL models
- Each declares interface, constraints, knobs, cost profile

### Constraint Validator
- Rejects components that violate schema, budget assumptions, or interface contract
- Prevents silent system failure

---

## VII. Research & Analysis

### Decision Analytics
- Rule firing frequency, decision effectiveness, wasted experiments avoided, failure patterns

### Policy Comparison Framework
- Compare: static pipelines vs rule-based agent vs hybrid agent

### Experiment Efficiency Metrics
- Compute saved, convergence acceleration, exploration efficiency

### Reproducibility
- Exact run reconstruction via configuration snapshots
