"""
engine.py
---------
Decision Engine — public entry point for the orchestrator.
All outputs (artifacts, logs, report, model) go to outputs/<run_id>/
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from stratml.execution.schemas import DataProfile, ExperimentResult
from stratml.core.schemas import ActionDecision, CandidateAction

from stratml.decision.state.state_builder import build_state_from_profile, build_state
from stratml.decision.state.state_history import ExperimentHistory
from stratml.decision.actions.action_generator import generate
from stratml.decision.learning.dataset_builder import record as record_dataset
from stratml.decision.learning.value_model import predict
from stratml.decision.learning.calibration import calibrate
from stratml.decision.learning.uncertainty import estimate
from stratml.decision.agents import performance_agent, efficiency_agent, stability_agent
from stratml.decision.agents.coordinator_agent import rank
from stratml.decision.policy.action_selector import select
from stratml.decision.logging import decision_logger
from stratml.decision.validation import counterfactual
from stratml.decision.learning import dataset_builder
from stratml.decision.validation.counterfactual import record as record_cf


class DecisionEngine:
    def __init__(
        self,
        primary_metric: str = "accuracy",
        optimization_goal: str = "maximize",
        allowed_models: Optional[list[str]] = None,
        max_iterations: int = 20,
        time_budget: Optional[float] = None,
        run_id: Optional[str] = None,
        dl_hyperparams: Optional[dict] = None,
    ) -> None:
        self.primary_metric    = primary_metric
        self.optimization_goal = optimization_goal
        self.allowed_models    = allowed_models
        self.max_iterations    = max_iterations
        self.time_budget       = time_budget
        self.dl_hyperparams    = dl_hyperparams or {}
        self.run_id            = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

        self._history          = ExperimentHistory()
        self._profile          = None
        self._models_tried: list[str] = []
        self._repeated_configs: int   = 0
        self._last_action: Optional[str] = None
        self._last_action_success: Optional[bool] = None

        # Redirect all outputs under outputs/<run_id>/
        self._out_dir = Path("outputs") / self.run_id
        self._out_dir.mkdir(parents=True, exist_ok=True)

        decision_logger._LOG_DIR      = self._out_dir / "decision_logs"
        counterfactual._CF_LOG        = self._out_dir / "decision_logs" / "counterfactual_log.jsonl"
        dataset_builder._DATASET_PATH = self._out_dir / "decision_logs" / "decision_dataset.csv"

    def receive_profile(self, profile: DataProfile) -> ActionDecision:
        self._profile = profile
        state = build_state_from_profile(
            profile,
            run_id=self.run_id,
            primary_metric=self.primary_metric,
            optimization_goal=self.optimization_goal,
            allowed_models=self.allowed_models,
            max_iterations=self.max_iterations,
            time_budget=self.time_budget,
        )
        return self._decide(state)

    def receive_result(self, result: ExperimentResult) -> ActionDecision:
        if result.model_name not in self._models_tried:
            self._models_tried.append(result.model_name)
        else:
            self._repeated_configs += 1

        remaining = max(0.0, self.max_iterations - result.iteration)
        state = build_state(
            result,
            history=self._history,
            profile=self._profile,
            primary_metric=self.primary_metric,
            optimization_goal=self.optimization_goal,
            allowed_models=self.allowed_models,
            max_iterations=self.max_iterations,
            time_budget=self.time_budget,
            previous_action=self._last_action,
            previous_action_success=self._last_action_success,
            models_tried=self._models_tried,
            repeated_configs=self._repeated_configs,
            remaining_budget=remaining,
        )
        return self._decide(state)

    def _decide(self, state) -> ActionDecision:
        candidates: list[CandidateAction] = generate(state)

        predictions = predict(state, candidates)
        calibrated  = calibrate(predictions)
        estimates   = estimate(calibrated)

        perf_scores = performance_agent.score(state, estimates)
        eff_scores  = efficiency_agent.score(state, estimates)
        stab_scores = stability_agent.score(state, estimates)

        ranked   = rank(state, estimates, perf_scores, eff_scores, stab_scores)
        decision = select(state, ranked)

        # Inject DL hyperparams when running in DL mode
        if self.dl_hyperparams and decision.action_type != "terminate":
            decision.parameters.update(self.dl_hyperparams)

        record_dataset(state, candidates[0])
        decision_logger.log(state, candidates, decision)
        record_cf(decision)

        self._last_action         = decision.action_type
        self._last_action_success = None
        return decision
