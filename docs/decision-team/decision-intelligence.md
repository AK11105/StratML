# FINAL DECISION & INTELLIGENCE SYSTEM — IMPLEMENTATION PLAN (UPDATED)

The complete pipeline now becomes:

```
ExperimentResult
        ↓
State Builder
        ↓
Trajectory Feature Extractor
        ↓
Dataset Meta-Feature Injector
        ↓
Signal Extraction
        ↓
Candidate Action Generator
        ↓
Decision Dataset Builder
        ↓
Decision Value Model (ensemble)
        ↓
Calibration Layer
        ↓
Uncertainty Estimator
        ↓
Decision Council (Multi-Agent Deliberation)
        ↓
Coordinator Agent
        ↓
Action Selection Policy
        ↓
Decision Logger
        ↓
Counterfactual Validator
        ↓
Action
```

The **Decision Council** is the only major addition.

---

# PHASE 1 — Core Contracts

Create the interface between execution and decision layers.

File

```
core/schemas.py
```

Objects:

```
ExperimentResult
StateObject
CandidateAction
ActionDecision
DecisionRecord
ValidationRecord
AgentScore
```

ExperimentResult

```
metrics
runtime
model_name
iteration
dataset_metadata
```

StateObject

```
accuracy
validation_loss
improvement_rate
accuracy_slope
loss_slope
runtime_trend
dataset_meta_features
model_type
iteration
```

CandidateAction

```
action_type
parameters
```

ActionDecision

```
selected_action
expected_gain
expected_cost
confidence
agent_scores
```

AgentScore

```
performance_score
efficiency_score
stability_score
```

Freeze these interfaces.

---

# PHASE 2 — State Builder

File

```
decision/state_builder.py
```

Purpose

Convert ExperimentResult → StateObject.

Steps

1. Receive ExperimentResult
2. Retrieve experiment history
3. Extract base metrics

Features

```
accuracy
validation_loss
runtime
model_type
iteration
```

Compute improvement rate

```
improvement_rate = accuracy_t − accuracy_t-1
```

Return initial StateObject.

---

# PHASE 3 — Trajectory Feature Extraction

File

```
decision/state_history.py
```

Maintain experiment history buffer.

History window

```
last 3 experiments
```

Compute trajectory features

```
accuracy_slope = (accuracy_t − accuracy_t-2) / 2
loss_slope = (loss_t − loss_t-2) / 2
runtime_trend = runtime_t − runtime_t-1
model_switch_frequency
```

Inject into StateObject.

Now the state includes **experiment trajectory dynamics**.

---

# PHASE 4 — Dataset Meta-Feature Extraction

File

```
decision/meta_features.py
```

Compute dataset characteristics once.

Features

```
num_samples
num_features
feature_sample_ratio
class_entropy
missing_value_ratio
feature_variance_mean
```

Example

```
feature_sample_ratio = num_features / num_samples
```

Store and inject into StateObject.

Now the decision system is **dataset-aware**.

---

# PHASE 5 — Signal Extraction

File

```
decision/signals.py
```

Signals interpret the state.

Examples

Underfitting score

```
1 − accuracy
```

Overfitting score

```
train_accuracy − validation_accuracy
```

Convergence score

```
improvement_rate
```

Training stability

```
variance(loss_history)
```

Signals stored inside StateObject.

---

# PHASE 6 — Candidate Action Generator

File

```
decision/action_generator.py
```

Generate possible next actions.

Action space

```
switch_model
increase_model_capacity
decrease_model_capacity
modify_regularization
change_optimizer
terminate
```

Each action becomes CandidateAction.

Example output

```
[
 switch_model(RandomForest),
 increase_capacity,
 decrease_capacity,
 terminate
]
```

Actions depend on current state constraints.

---

# PHASE 7 — Decision Dataset Builder

File

```
decision/learning/dataset_builder.py
```

Create training dataset for the decision model.

Dataset structure

```
state_features
action_type
observed_gain
training_cost
```

Example row

```
accuracy=0.71
accuracy_slope=0.02
num_features=40
action=switch_model
observed_gain=0.08
training_cost=0.5
```

Save dataset

```
decision_data/decision_dataset.csv
```

Update dataset after each experiment.

---

# PHASE 8 — Decision Value Model

File

```
decision/learning/value_model.py
```

Purpose

Predict outcomes of candidate actions.

Model

```
RandomForestRegressor
```

Inputs

```
state_features + action_type
```

Outputs

```
predicted_accuracy_gain
predicted_training_cost
```

Training process

1. Load dataset
2. Train model
3. Save model

Location

```
models/decision_value_model.pkl
```

---

# PHASE 9 — Ensemble Model for Uncertainty

File

```
decision/learning/uncertainty.py
```

Train ensemble of decision models.

Example ensemble

```
DecisionModel_1
DecisionModel_2
DecisionModel_3
DecisionModel_4
```

For each candidate action compute predictions

```
gain_1
gain_2
gain_3
gain_4
```

Variance

```
variance(predictions)
```

Confidence

```
confidence = 1 / variance
```

Low variance → high confidence.

---

# PHASE 10 — Decision Calibration Layer

File

```
decision/learning/calibration.py
```

Purpose

Correct bias in predicted gains.

Dataset

```
predicted_gain
actual_gain
```

Calibration model

```
IsotonicRegression
```

Output

```
calibrated_gain
```

Used by the policy selector.

---

# PHASE 11 — Decision Council (Multi-Agent Deliberation)

NEW PHASE

Folder

```
decision/agents/
```

Agents

```
performance_agent.py
efficiency_agent.py
stability_agent.py
coordinator_agent.py
```

Each agent evaluates candidate actions from a different perspective.

---

## Performance Agent

Goal

```
maximize model performance
```

Inputs

```
predicted_accuracy_gain
dataset_meta_features
model_type
trajectory_features
```

Output

```
performance_score(action)
```

---

## Efficiency Agent

Goal

```
minimize compute cost
```

Inputs

```
predicted_training_cost
runtime_trend
dataset_size
model_complexity
```

Output

```
efficiency_score(action)
```

---

## Stability Agent

Goal

```
avoid unstable training behaviour
```

Inputs

```
loss_variance
convergence_speed
training_instability_signals
```

Output

```
stability_score(action)
```

---

# PHASE 12 — Coordinator Agent

File

```
decision/agents/coordinator_agent.py
```

Combine agent scores.

Example scoring table

| Action            | Performance | Efficiency | Stability |
| ----------------- | ----------- | ---------- | --------- |
| switch_model      | 0.85        | 0.60       | 0.90      |
| increase_capacity | 0.65        | 0.30       | 0.50      |
| terminate         | 0.10        | 0.95       | 0.95      |

Aggregate score

```
FinalScore =
w1 * performance
+ w2 * efficiency
+ w3 * stability
```

Example weights

```
w1 = 0.5
w2 = 0.25
w3 = 0.25
```

Coordinator returns aggregated score.

---

# PHASE 13 — Action Selection Policy

File

```
decision/policy_selector.py
```

For each candidate action

1. Predict gain
2. Predict cost
3. Compute uncertainty
4. Apply calibration
5. Get council scores

Compute final score

```
score = calibrated_gain + β * uncertainty + council_score
```

Select action with highest score.

Return ActionDecision.

---

# PHASE 14 — Decision Logger

File

```
decision/logger.py
```

Record decision details.

Log

```
state
candidate actions
predicted gains
uncertainty
agent_scores
selected action
confidence
```

Example log

```
accuracy=0.74
improvement=0.01

switch_model → gain=0.09
increase_capacity → gain=0.04
terminate → gain=0.00

performance_agent=0.85
efficiency_agent=0.60
stability_agent=0.90

selected_action=switch_model
confidence=0.88
```

Store logs

```
logs/decision_logs.json
```

---

# PHASE 15 — Counterfactual Validator

File

```
decision/validation/counterfactual.py
```

Procedure

1. Agent selects action A
2. System executes A
3. Randomly choose alternative action B
4. Execute B
5. Compare results

Example

```
accuracy(A)=0.91
accuracy(B)=0.87
```

Record

```
decision_correct=True
```

Store validation records.

---

# PHASE 16 — Decision Metrics

File

```
analysis/decision_metrics.py
```

Metrics

Decision accuracy

```
correct_decisions / validated_decisions
```

Experiment efficiency

```
performance_gain / number_of_experiments
```

Convergence speed

```
iterations_to_best_model
```

Experiment waste

```
non_improving_experiments / total_experiments
```

Agent disagreement metric

```
variance(agent_scores)
```

---

# FINAL DECISION LAYER STRUCTURE (UPDATED)

```
decision/

state_builder.py
state_history.py
meta_features.py
signals.py
action_generator.py
policy_selector.py
logger.py

agents/
    performance_agent.py
    efficiency_agent.py
    stability_agent.py
    coordinator_agent.py

learning/
    dataset_builder.py
    value_model.py
    uncertainty.py
    calibration.py

validation/
    counterfactual.py
```

---

# UPDATED FINAL SYSTEM BEHAVIOR

```
experiment executed
        ↓
state constructed
        ↓
trajectory + dataset features added
        ↓
candidate actions generated
        ↓
decision model predicts outcomes
        ↓
calibration adjusts predictions
        ↓
uncertainty estimated
        ↓
decision council evaluates actions
        ↓
coordinator aggregates agent scores
        ↓
policy selects best action
        ↓
decision logged
        ↓
counterfactual validation
        ↓
next experiment
```

---

# WHAT YOU CAN CLAIM IN PRESENTATION (UPDATED)

Your system now provides

```
learned experimentation strategy
dataset-aware experiment planning
trajectory-aware decision making
multi-agent decision council
decision confidence estimation
decision calibration
counterfactual decision validation
agent disagreement analysis
```

Which is significantly stronger than a **rule-based AutoML system** and clearly demonstrates a **multi-agent intelligent experimentation framework**.
