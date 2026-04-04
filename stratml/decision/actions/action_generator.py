"""
action_generator.py
-------------------
Decision/Actions — Candidate Action Generator.

Input:  StateObject
Output: list[CandidateAction]

Bootstrap (iteration 0): returns fixed baseline candidates.
Iteration 1+: derives candidates from state.signals + state.constraints.
"""

from __future__ import annotations

from stratml.core.schemas import CandidateAction, StateObject

# Models available when no constraints are set
_DEFAULT_MODELS = [
    "LogisticRegression",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "ExtraTreesClassifier",
    "SVC",
    "KNeighborsClassifier",
    "GaussianNB",
    "DecisionTreeClassifier",
]

_BOOTSTRAP_MODELS = [
    "LogisticRegression",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "ExtraTreesClassifier",
    "KNeighborsClassifier",
    "GaussianNB",
    "SVC",
    "DecisionTreeClassifier",
]


def generate(state: StateObject) -> list[CandidateAction]:
    if state.meta.iteration == 0:
        return _bootstrap_candidates(state)
    return _rule_candidates(state)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_candidates(state: StateObject) -> list[CandidateAction]:
    allowed = state.constraints.allowed_models or _BOOTSTRAP_MODELS
    return [
        CandidateAction(action_type="switch_model", parameters={"model_name": m})
        for m in allowed[:2]
    ]


# ---------------------------------------------------------------------------
# Rule-based (iteration 1+)
# ---------------------------------------------------------------------------

def _rule_candidates(state: StateObject) -> list[CandidateAction]:
    sig = state.signals
    candidates: list[CandidateAction] = []

    allowed = state.constraints.allowed_models or _DEFAULT_MODELS
    tried = set(state.search.models_tried)
    untried = [m for m in allowed if m not in tried]

    # Budget exhausted → only terminate
    if state.resources.budget_exhausted:
        return [CandidateAction(action_type="terminate", parameters={})]

    # Converged and well-fitted → terminate
    if sig.converged != "none" and sig.well_fitted != "none":
        candidates.append(CandidateAction(action_type="terminate", parameters={}))
        return candidates

    # Underfitting → try a more powerful model or increase capacity
    if sig.underfitting != "none":
        if untried:
            candidates.append(CandidateAction(
                action_type="switch_model",
                parameters={"model_name": untried[0]},
            ))
        candidates.append(CandidateAction(
            action_type="increase_model_capacity",
            parameters={"scale": 1.5},
        ))

    # Overfitting → regularize first; escalate to switch_model when plateau hits
    if sig.overfitting != "none":
        # If plateau is strong, switch model takes priority over regularization
        if sig.plateau_detected == "strong" and untried:
            candidates.append(CandidateAction(
                action_type="switch_model",
                parameters={"model_name": untried[0]},
            ))
        else:
            candidates.append(CandidateAction(
                action_type="modify_regularization",
                parameters={"direction": "increase"},
            ))
            candidates.append(CandidateAction(
                action_type="decrease_model_capacity",
                parameters={"scale": 0.75},
            ))
            if state.trajectory.steps_since_improvement >= 2 and untried:
                candidates.append(CandidateAction(
                    action_type="switch_model",
                    parameters={"model_name": untried[0]},
                ))

    # Stagnating / plateau → switch model
    if sig.stagnating != "none" or sig.plateau_detected != "none":
        if untried:
            candidates.append(CandidateAction(
                action_type="switch_model",
                parameters={"model_name": untried[0]},
            ))

    # Diverging → change optimizer / reduce lr
    if sig.diverging != "none":
        candidates.append(CandidateAction(
            action_type="change_optimizer",
            parameters={"learning_rate_scale": 0.1},
        ))

    # Diminishing returns → try next untried model or terminate
    if sig.diminishing_returns != "none":
        if untried:
            candidates.append(CandidateAction(
                action_type="switch_model",
                parameters={"model_name": untried[0]},
            ))
        else:
            candidates.append(CandidateAction(action_type="terminate", parameters={}))

    # Always include terminate as a fallback option
    if not any(c.action_type == "terminate" for c in candidates):
        candidates.append(CandidateAction(action_type="terminate", parameters={}))

    # Deduplicate by (action_type, parameters)
    seen: set[tuple] = set()
    unique: list[CandidateAction] = []
    for c in candidates:
        key = (c.action_type, str(sorted(c.parameters.items())))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique
