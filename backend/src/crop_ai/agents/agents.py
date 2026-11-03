"""The LLM agents of the team, each with a rule-based fallback.

Every agent makes one structured call through `StructuredLLM`. When the LLM is unavailable (demo
mode, budget exhausted, provider error) the rule-based fallback runs instead, so a consultation
always completes; the orchestrator records which path was used.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from crop_ai.agents.prompts import (
    DIAGNOSIS_PROMPT,
    FOLLOWUP_PROMPT,
    QUESTIONS_PROMPT,
    TREATMENT_PROMPT,
    agent_user_prompt,
)
from crop_ai.agents.state import (
    AgentState,
    Candidate,
    DifferentialDiagnosis,
    FarmerQuestions,
    FollowUpPlan,
    TreatmentPlan,
)
from crop_ai.agents.tools import reinspect_days

SchemaT = TypeVar("SchemaT", bound=BaseModel)
AMBIGUITY_MARGIN = 0.2


class StructuredCaller(Protocol):
    async def __call__(
        self, system: str, user: str, schema: type[SchemaT], model: str | None = None
    ) -> SchemaT: ...


def _top_k(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in evidence.get("top_k") or [] if item.get("disease")]


def rule_based_diagnosis(state: AgentState, confidence_threshold: float) -> DifferentialDiagnosis:
    """Candidates straight from the image model's top-k; ambiguous on low confidence or margin."""
    ranked = _top_k(state.evidence)
    candidates = [
        Candidate(
            crop=str(item.get("crop") or state.evidence.get("predicted_crop") or "Unknown crop"),
            disease=str(item["disease"]),
            likelihood="high"
            if item.get("confidence", 0) >= 0.7
            else "medium"
            if item.get("confidence", 0) >= 0.4
            else "low",
        )
        for item in ranked[:3]
    ]
    if not candidates and state.evidence.get("predicted_disease"):
        candidates = [
            Candidate(
                crop=str(state.evidence.get("predicted_crop") or "Unknown crop"),
                disease=str(state.evidence["predicted_disease"]),
                likelihood="medium",
            )
        ]
    confidences = [float(item.get("confidence", 0)) for item in ranked]
    top = confidences[0] if confidences else float(state.evidence.get("confidence") or 0)
    margin = top - confidences[1] if len(confidences) > 1 else 1.0
    ambiguous = top < confidence_threshold or margin < AMBIGUITY_MARGIN
    name = f"{candidates[0].crop} - {candidates[0].disease}" if candidates else "unknown"
    return DifferentialDiagnosis(
        summary=f"The image model's top prediction is {name} ({top:.0%} confidence).",
        agreement_with_model="agrees" if candidates else "insufficient_evidence",
        candidates=candidates,
        is_ambiguous=ambiguous,
        distinguishing_signs=[],
    )


GENERIC_QUESTIONS = [
    "Are the spots mostly on the older, lower leaves?",
    "Do the spots have rings inside them or a yellow border around them?",
    "Is the problem spreading quickly to nearby plants?",
]


def rule_based_followup(state: AgentState) -> FollowUpPlan:
    reassessment = ""
    if state.previous_evidence:
        before = state.previous_evidence.get("severity_percentage")
        after = state.evidence.get("severity_percentage")
        same = state.previous_evidence.get("predicted_disease") == state.evidence.get(
            "predicted_disease"
        )
        if isinstance(before, int | float) and isinstance(after, int | float):
            trend = "less" if after < before else "more" if after > before else "the same"
            decision = "continue the plan" if after <= before and same else "review the plan"
            reassessment = (
                f"Visible lesion area went from {before:.0f}% to {after:.0f}% ({trend}; "
                f"experimental estimate), so {decision}."
            )
        else:
            reassessment = "Severity could not be compared; review both photos with an expert."
    return FollowUpPlan(
        reinspect_in_days=reinspect_days(state.risk),
        what_to_check=[
            "Whether spots have spread to new or upper leaves",
            "New symptoms on stems or fruit",
        ],
        escalate_if=[
            "More than a quarter of the leaves are affected",
            "Symptoms keep spreading after treatment",
        ],
        reassessment=reassessment,
    )


async def run_diagnosis(llm: StructuredCaller, state: AgentState, message: str):
    return await llm(DIAGNOSIS_PROMPT, agent_user_prompt(state, message), DifferentialDiagnosis)


async def run_questions(llm: StructuredCaller, state: AgentState, message: str) -> list[str]:
    asked = {item["question"] for item in state.qa_history}
    result = await llm(QUESTIONS_PROMPT, agent_user_prompt(state, message), FarmerQuestions)
    return [item for item in result.questions if item not in asked][:3]


async def run_treatment(llm: StructuredCaller, state: AgentState, message: str):
    return await llm(TREATMENT_PROMPT, agent_user_prompt(state, message), TreatmentPlan)


async def run_followup(llm: StructuredCaller, state: AgentState, message: str):
    plan = await llm(
        FOLLOWUP_PROMPT, agent_user_prompt(state, message, include_passages=False), FollowUpPlan
    )
    if not state.previous_evidence:
        plan.reassessment = ""
    return plan
