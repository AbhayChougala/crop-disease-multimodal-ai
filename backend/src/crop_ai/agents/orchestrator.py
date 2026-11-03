"""LLM-orchestrated agent team with guardrails.

Each turn the orchestrator LLM picks the next action from the actions whose preconditions are met.
Invalid choices, provider errors and exhausted budgets fall back to a deterministic policy, so a
consultation always completes and every step is recorded in a trace.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel

from crop_ai.agents import agents
from crop_ai.agents.compose import compose_reply
from crop_ai.agents.prompts import ORCHESTRATOR_PROMPT, orchestrator_user_prompt
from crop_ai.agents.safety import review
from crop_ai.agents.state import (
    STEP_LABELS,
    AgentState,
    AgentStep,
    OrchestratorDecision,
    TreatmentPlan,
)
from crop_ai.agents.tools import (
    SearchFn,
    WeatherFn,
    assess_risk,
    candidate_diseases,
    passages_of,
    retrieve_for_candidates,
)
from crop_ai.llm.base import ProviderUnavailable

MAX_QUESTION_ROUNDS = 2
ENDS_TURN = {"ask_farmer", "answer_question"}


class BudgetExceeded(RuntimeError):
    pass


class _BudgetedLLM:
    """Counts structured LLM calls and refuses once the per-turn budget is spent."""

    def __init__(self, provider: Any, state: AgentState, max_calls: int, language: str) -> None:
        self.provider = provider
        self.state = state
        self.max_calls = max_calls
        self.language = language

    async def __call__(self, system: str, user: str, schema: type[BaseModel], model=None):
        if self.state.llm_calls >= self.max_calls:
            raise BudgetExceeded("LLM call budget for this turn is used up.")
        self.state.llm_calls += 1
        user = f"{user}\n\n<PREFERRED_LANGUAGE>{self.language}</PREFERRED_LANGUAGE>"
        return await self.provider.complete_structured(system, user, schema, model=model)


@dataclass
class TurnResult:
    text: str
    source_ids: list[str]
    ended_with_questions: bool
    is_mock: bool
    status: str
    steps: list[AgentStep] = field(default_factory=list)


class AgentOrchestrator:
    def __init__(
        self,
        provider: Any,
        search: SearchFn,
        weather: WeatherFn | None = None,
        *,
        use_llm: bool = True,
        max_steps: int = 7,
        max_llm_calls: int = 10,
        orchestrator_model: str | None = None,
        confidence_threshold: float = 0.65,
    ) -> None:
        self.provider = provider
        self.search = search
        self.weather = weather
        self.use_llm = use_llm
        self.max_steps = max_steps
        self.max_llm_calls = max_llm_calls
        self.orchestrator_model = orchestrator_model or None
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------ policy
    def allowed_actions(self, state: AgentState, done: list[str], answered: bool) -> list[str]:
        allowed: list[str] = []
        reassessing = state.previous_evidence is not None
        if "retrieve_evidence" not in done:
            allowed.append("retrieve_evidence")
        if "diagnose" not in done and ("retrieve_evidence" in done or state.passages):
            allowed.append("diagnose")
        if (
            "diagnose" in done
            and state.diagnosis is not None
            and state.diagnosis.is_ambiguous
            and state.question_rounds < MAX_QUESTION_ROUNDS
            and "ask_farmer" not in done
        ):
            allowed.append("ask_farmer")
        if state.diagnosis is not None and "assess_risk" not in done:
            allowed.append("assess_risk")
        if state.diagnosis is not None and state.passages and "recommend_treatment" not in done:
            allowed.append("recommend_treatment")
        # Follow-up waits for treatment unless treatment was already tried (even if it failed),
        # is impossible (no evidence), or this turn is a photo reassessment.
        treatment_pending = "recommend_treatment" in allowed
        if (
            state.diagnosis is not None
            and "plan_followup" not in done
            and (reassessing or not treatment_pending)
        ):
            allowed.append("plan_followup")
        # Q&A is a turn of its own: only as the first action, never while answering questions
        # or reassessing a new photo (those turns must show the team's full result).
        if not done and not answered and not reassessing:
            allowed.append("answer_question")
        finished_work = "recommend_treatment" in done or "plan_followup" in done or not allowed
        if finished_work:
            allowed.append("finish")
        return allowed

    def fallback_action(
        self, state: AgentState, done: list[str], allowed: list[str], answered: bool, fresh: bool
    ) -> str:
        """Deterministic policy used when the orchestrator LLM is unavailable or invalid."""
        if not fresh and not answered and state.treatment is not None and not done:
            return "answer_question"
        order = ["retrieve_evidence", "diagnose"]
        if not answered:
            order.append("ask_farmer")
        order += ["assess_risk", "recommend_treatment", "plan_followup", "finish"]
        for action in order:
            if action in allowed:
                return action
        return "finish"

    async def _decide(
        self,
        llm: _BudgetedLLM,
        state: AgentState,
        message: str,
        done: list[str],
        allowed: list[str],
        answered: bool,
        fresh: bool,
    ) -> tuple[str, str, str]:
        """Return (action, decided_by, reason)."""
        if self.use_llm:
            try:
                decision = await llm(
                    ORCHESTRATOR_PROMPT,
                    orchestrator_user_prompt(state, message, done, allowed),
                    OrchestratorDecision,
                    model=self.orchestrator_model,
                )
                if decision.next_action in allowed:
                    return decision.next_action, "llm", decision.reason
                reason = f"LLM chose '{decision.next_action}', which was not allowed yet"
            except (ProviderUnavailable, BudgetExceeded) as exc:
                reason = f"orchestrator unavailable ({str(exc)[:80]})"
        else:
            reason = "demo mode: rule-based orchestration"
        return self.fallback_action(state, done, allowed, answered, fresh), "rule", reason

    # ------------------------------------------------------------------ actions
    async def _run_action(
        self, action: str, llm: _BudgetedLLM, state: AgentState, message: str, language: str
    ) -> tuple[str, str]:
        """Execute one action; return (status, summary)."""
        if action == "retrieve_evidence":
            count = retrieve_for_candidates(self.search, state, message)
            sources = {item["document_id"] for item in state.passages}
            if not count:
                return "skipped", "No fact sheets found (knowledge base empty or RAG disabled)."
            names = ", ".join(disease for _, disease in candidate_diseases(state))
            return "ok", f"{count} passages from {len(sources)} fact sheets for: {names}"

        if action == "diagnose":
            try:
                if not self.use_llm:
                    raise ProviderUnavailable("demo mode")
                state.diagnosis = await agents.run_diagnosis(llm, state, message)
                status = "ok"
            except (ProviderUnavailable, BudgetExceeded):
                state.diagnosis = agents.rule_based_diagnosis(state, self.confidence_threshold)
                status = "fallback"
            top = state.diagnosis.candidates[0] if state.diagnosis.candidates else None
            label = f"{top.disease} ({top.likelihood})" if top else "no candidate"
            ambiguity = "ambiguous" if state.diagnosis.is_ambiguous else "clear"
            return status, (
                f"Most likely {label}; {ambiguity}; "
                f"{state.diagnosis.agreement_with_model.replace('_', ' ')} with the image model"
            )

        if action == "ask_farmer":
            try:
                if not self.use_llm:
                    raise ProviderUnavailable("demo mode")
                questions = await agents.run_questions(llm, state, message)
                status = "ok"
            except (ProviderUnavailable, BudgetExceeded):
                questions, status = [], "fallback"
            if not questions:
                asked = {item["question"] for item in state.qa_history}
                questions = [q for q in agents.GENERIC_QUESTIONS if q not in asked][:3]
                status = "fallback"
            state.pending_questions = questions
            state.question_rounds += 1
            return status, f"Asked {len(questions)} question(s) to narrow the diagnosis"

        if action == "assess_risk":
            weather = None
            if state.location and self.weather:
                weather = await self.weather(
                    state.location["latitude"], state.location["longitude"]
                )
            state.risk = assess_risk(state, weather)
            source = "with weather" if weather else "without weather"
            return "ok", f"Spread risk {state.risk.level} (score {state.risk.score}, {source})"

        if action == "recommend_treatment":
            if not state.passages:
                return "skipped", "No fact sheets to ground a treatment plan"
            if not self.use_llm:
                state.treatment = TreatmentPlan(
                    summary="Demo mode: no treatment advice is generated without the language "
                    "model. Review the cited fact sheets with an expert.",
                    items=[],
                )
                return "skipped", "Demo mode: treatment agent needs the language model"
            try:
                state.treatment = await agents.run_treatment(llm, state, message)
            except (ProviderUnavailable, BudgetExceeded) as exc:
                return "error", f"Treatment agent unavailable: {str(exc)[:80]}"
            counts = {item.category for item in state.treatment.items}
            return "ok", f"{len(state.treatment.items)} actions ({', '.join(sorted(counts))})"

        if action == "plan_followup":
            try:
                if not self.use_llm:
                    raise ProviderUnavailable("demo mode")
                state.follow_up = await agents.run_followup(llm, state, message)
                status = "ok"
            except (ProviderUnavailable, BudgetExceeded):
                state.follow_up = agents.rule_based_followup(state)
                status = "fallback"
            note = " with photo comparison" if state.follow_up.reassessment else ""
            return status, f"Re-check in {state.follow_up.reinspect_in_days} days{note}"

        if action == "answer_question":
            # Search around the team's own conclusion when it has one, else the image model's.
            crop = state.evidence.get("predicted_crop")
            disease = state.evidence.get("predicted_disease")
            if state.diagnosis and state.diagnosis.candidates:
                crop = state.diagnosis.candidates[0].crop
                disease = state.diagnosis.candidates[0].disease
            found = self.search(message, crop, disease, 5)
            known = {item["citation_id"] for item in state.passages}
            state.passages += [asdict(item) for item in found if item.citation_id not in known]
            try:
                state.answer = await self.provider.generate(
                    state.evidence, message, found, None, language
                )
            except ProviderUnavailable as exc:
                state.answer = {
                    "answer": "I do not have enough trusted evidence to answer that safely.",
                    "source_ids": [],
                    "status": "insufficient_evidence",
                }
                return "error", f"Q&A agent unavailable: {str(exc)[:80]}"
            scope = (
                "off-topic, refused" if state.answer.get("status") == "out_of_scope" else "answered"
            )
            return "ok", f"Question {scope} using {len(found)} passages"

        return "skipped", f"Unknown action {action}"

    # ------------------------------------------------------------------ turn
    async def run_turn(
        self,
        state: AgentState,
        message: str,
        *,
        location: dict[str, float] | None = None,
        language: str = "en",
    ) -> TurnResult:
        fresh = state.turn == 0 or state.previous_evidence is not None
        state.turn += 1
        state.steps, state.warnings, state.answer, state.llm_calls = [], [], None, 0
        answered = bool(state.pending_questions)
        if answered:
            state.qa_history.append(
                {"question": " / ".join(state.pending_questions), "answer": message}
            )
            state.pending_questions = []
            fresh = True
        if location:
            state.location = location

        llm = _BudgetedLLM(self.provider, state, self.max_llm_calls, language)
        done: list[str] = []
        ended_with_questions = False
        while len(done) < self.max_steps:
            allowed = self.allowed_actions(state, done, answered)
            action, decided_by, reason = await self._decide(
                llm, state, message, done, allowed, answered, fresh
            )
            if action == "finish":
                state.steps.append(
                    AgentStep(
                        action="finish",
                        agent=STEP_LABELS["finish"],
                        status="ok",
                        summary="Composed the reply",
                        decided_by=decided_by,
                        reason=reason,
                    )
                )
                break
            started = time.perf_counter()
            status, summary = await self._run_action(action, llm, state, message, language)
            state.steps.append(
                AgentStep(
                    action=action,
                    agent=STEP_LABELS[action],
                    status=status,
                    summary=summary,
                    decided_by=decided_by,
                    reason=reason,
                    ms=round((time.perf_counter() - started) * 1000, 1),
                )
            )
            done.append(action)
            if action == "ask_farmer" and state.pending_questions:
                ended_with_questions = True
                break
            if action == "answer_question":
                break

        state.warnings += review(state)
        if "plan_followup" in done:
            state.previous_evidence = None
        text, cited = compose_reply(state, ended_with_questions=ended_with_questions)
        if state.answer and state.answer.get("status") == "out_of_scope":
            status = "out_of_scope"
        elif ended_with_questions:
            status = "needs_input"
        else:
            produced = cited or "plan_followup" in done
            status = "available" if produced else "insufficient_evidence"
        return TurnResult(
            text=text,
            source_ids=cited,
            ended_with_questions=ended_with_questions,
            is_mock=not self.use_llm,
            status=status,
            steps=list(state.steps),
        )


def passages_for_citations(state: AgentState):
    """Retrieved passages as dataclasses, for building citation cards."""
    return passages_of(state)
