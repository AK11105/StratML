"""
action_generator.py
-------------------
Decision/Actions — Candidate Action Generator.

LLM path: LangChain chain reads full StateObject and proposes list[CandidateAction]
          via structured output. Can propose context-sensitive, composed actions.
Fallback: rule-based if-else tree (used when LLM fails or GROQ_API_KEY absent).

Bootstrap (iteration 0) always uses the rule-based path — no LLM needed there.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from pydantic import BaseModel

from stratml.core.schemas import CandidateAction, StateObject

log = logging.getLogger(__name__)

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

_VALID_ACTION_TYPES = {
    "switch_model",
    "increase_model_capacity",
    "decrease_model_capacity",
    "modify_regularization",
    "change_optimizer",
    "terminate",
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate(state: StateObject) -> list[CandidateAction]:
    """Return candidate actions for the current state."""
    if state.meta.iteration == 0:
        return _bootstrap_candidates(state)
    if os.getenv("GROQ_API_KEY"):
        result = _llm_candidates(state)
        if result is not None:
            return result
    return _rule_candidates(state)


# ---------------------------------------------------------------------------
# Bootstrap (iteration 0)
# ---------------------------------------------------------------------------

def _bootstrap_candidates(state: StateObject) -> list[CandidateAction]:
    allowed = state.constraints.allowed_models or _BOOTSTRAP_MODELS
    return [
        CandidateAction(action_type="switch_model", parameters={"model_name": m})
        for m in allowed[:2]
    ]


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an ML experimentation strategist. Given the current experiment state, "
    "propose a list of candidate next actions. You may suggest actions the rules don't "
    "cover, compose specific parameter values (e.g., a concrete alpha for regularization "
    "derived from the gap magnitude), or recommend a specific model based on dataset "
    "characteristics. Always include 'terminate' as one candidate. "
    "Valid action_types: switch_model, increase_model_capacity, decrease_model_capacity, "
    "modify_regularization, change_optimizer, terminate."
)


class _CandidateItem(BaseModel):
    action_type: str
    parameters: dict


class _CandidateList(BaseModel):
    candidates: list[_CandidateItem]


def _llm_candidates(state: StateObject) -> Optional[list[CandidateAction]]:
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage

        sig = state.signals
        traj = state.trajectory
        allowed = state.constraints.allowed_models or _DEFAULT_MODELS
        tried = state.search.models_tried
        untried = [m for m in allowed if m not in tried]

        human = (
            f"Iteration: {state.meta.iteration}\n"
            f"Model: {state.model.model_name} ({state.model.model_type})\n"
            f"Dataset: {state.dataset.num_samples} samples, {state.dataset.num_features} features, "
            f"imbalance_ratio={state.dataset.imbalance_ratio}\n"
            f"Fitting: underfitting={sig.underfitting}, overfitting={sig.overfitting}, "
            f"well_fitted={sig.well_fitted}, plateau={sig.plateau_detected}\n"
            f"Trajectory: trend={traj.trend}, slope={traj.slope:.4f}, "
            f"steps_since_improvement={traj.steps_since_improvement}, best_score={traj.best_score:.4f}\n"
            f"Resources: remaining_budget={state.resources.remaining_budget}, "
            f"budget_exhausted={state.resources.budget_exhausted}\n"
            f"Allowed models: {allowed}\n"
            f"Untried models: {untried}\n"
            f"Train/val gap: {state.generalization.gap:.4f}\n\n"
            "Propose 2-4 candidate actions as a JSON list. Each must have action_type and parameters."
        )

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2).with_structured_output(_CandidateList)
        output: _CandidateList = llm.invoke([SystemMessage(_SYSTEM_PROMPT), HumanMessage(human)])

        # Validate and filter
        candidates = [
            CandidateAction(action_type=item.action_type, parameters=item.parameters)
            for item in output.candidates
            if item.action_type in _VALID_ACTION_TYPES
        ]

        if not candidates:
            return None

        # Ensure terminate is always present
        if not any(c.action_type == "terminate" for c in candidates):
            candidates.append(CandidateAction(action_type="terminate", parameters={}))

        # Deduplicate
        seen: set[tuple] = set()
        unique: list[CandidateAction] = []
        for c in candidates:
            key = (c.action_type, str(sorted(c.parameters.items())))
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique
    except Exception as exc:
        log.warning("action_generator LLM failed (%s), using rule fallback", exc)
        return None


# ---------------------------------------------------------------------------
# Rule-based fallback (iteration 1+)
# ---------------------------------------------------------------------------

def _rule_candidates(state: StateObject) -> list[CandidateAction]:
    sig = state.signals
    candidates: list[CandidateAction] = []

    allowed = state.constraints.allowed_models or _DEFAULT_MODELS
    tried = set(state.search.models_tried)
    untried = [m for m in allowed if m not in tried]

    if state.resources.budget_exhausted:
        return [CandidateAction(action_type="terminate", parameters={})]

    if sig.converged != "none" and sig.well_fitted != "none":
        return [CandidateAction(action_type="terminate", parameters={})]

    if sig.underfitting != "none":
        if untried:
            candidates.append(CandidateAction(action_type="switch_model", parameters={"model_name": untried[0]}))
        candidates.append(CandidateAction(action_type="increase_model_capacity", parameters={"scale": 1.5}))

    if sig.overfitting != "none":
        if sig.plateau_detected == "strong" and untried:
            candidates.append(CandidateAction(action_type="switch_model", parameters={"model_name": untried[0]}))
        else:
            candidates.append(CandidateAction(action_type="modify_regularization", parameters={"direction": "increase"}))
            candidates.append(CandidateAction(action_type="decrease_model_capacity", parameters={"scale": 0.75}))
            if state.trajectory.steps_since_improvement >= 2 and untried:
                candidates.append(CandidateAction(action_type="switch_model", parameters={"model_name": untried[0]}))

    if sig.stagnating != "none" or sig.plateau_detected != "none":
        if untried:
            candidates.append(CandidateAction(action_type="switch_model", parameters={"model_name": untried[0]}))

    if sig.diverging != "none":
        candidates.append(CandidateAction(action_type="change_optimizer", parameters={"learning_rate_scale": 0.1}))

    if sig.diminishing_returns != "none":
        if untried:
            candidates.append(CandidateAction(action_type="switch_model", parameters={"model_name": untried[0]}))
        else:
            candidates.append(CandidateAction(action_type="terminate", parameters={}))

    if not any(c.action_type == "terminate" for c in candidates):
        candidates.append(CandidateAction(action_type="terminate", parameters={}))

    seen: set[tuple] = set()
    unique: list[CandidateAction] = []
    for c in candidates:
        key = (c.action_type, str(sorted(c.parameters.items())))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique
