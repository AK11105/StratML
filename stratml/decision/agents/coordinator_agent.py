"""
coordinator_agent.py
--------------------
Decision Council — Coordinator Agent.

Aggregates scores from performance, efficiency, and stability agents
into a single final_score per candidate action.

Weights: performance=0.50, efficiency=0.25, stability=0.25
"""

from __future__ import annotations

from dataclasses import dataclass

from stratml.core.schemas import AgentScore, StateObject
from stratml.decision.learning.uncertainty import UncertaintyEstimate

_W_PERF = 0.50
_W_EFF  = 0.25
_W_STAB = 0.25


@dataclass
class RankedAction:
    action_type: str
    parameters: dict
    predicted_gain: float
    predicted_cost: float
    confidence: float
    agent_scores: AgentScore
    final_score: float


def rank(
    state: StateObject,
    estimates: list[UncertaintyEstimate],
    perf_scores: dict[str, float],
    eff_scores: dict[str, float],
    stab_scores: dict[str, float],
) -> list[RankedAction]:
    """Return candidates sorted by final_score descending."""
    ranked: list[RankedAction] = []

    for e in estimates:
        p = perf_scores.get(e.action_type, 0.5)
        ef = eff_scores.get(e.action_type, 0.5)
        st = stab_scores.get(e.action_type, 0.5)

        final = round(_W_PERF * p + _W_EFF * ef + _W_STAB * st, 4)

        ranked.append(RankedAction(
            action_type=e.action_type,
            parameters=e.parameters,
            predicted_gain=e.predicted_gain,
            predicted_cost=e.predicted_cost,
            confidence=e.confidence,
            agent_scores=AgentScore(performance=p, efficiency=ef, stability=st),
            final_score=final,
        ))

    ranked.sort(key=lambda r: r.final_score, reverse=True)
    return ranked
