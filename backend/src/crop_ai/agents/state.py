"""Data models shared by the CropAI agent team.

Agent outputs are Pydantic models so LLM providers can return them as validated structured output.
`AgentState` is persisted per chat session, which lets the team continue across turns (answers to
clarifying questions, re-assessment with a new photo).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AgentAction = Literal[
    "retrieve_evidence",
    "diagnose",
    "ask_farmer",
    "assess_risk",
    "recommend_treatment",
    "plan_followup",
    "answer_question",
    "finish",
]

STEP_LABELS: dict[str, str] = {
    "retrieve_evidence": "Evidence agent",
    "diagnose": "Diagnosis agent",
    "ask_farmer": "Question agent",
    "assess_risk": "Risk assessment tool",
    "recommend_treatment": "Treatment agent",
    "plan_followup": "Follow-up agent",
    "answer_question": "Q&A agent",
    "finish": "Orchestrator",
}


class Candidate(BaseModel):
    crop: str
    disease: str
    likelihood: Literal["high", "medium", "low"]
    supporting_signs: list[str] = Field(
        default_factory=list, description="Up to 3 signs from evidence or farmer answers."
    )
    conflicting_signs: list[str] = Field(default_factory=list, description="Up to 2 signs.")


class DifferentialDiagnosis(BaseModel):
    summary: str = Field(description="1-2 sentences: most likely disease and why.")
    agreement_with_model: Literal[
        "agrees", "partially_agrees", "disagrees", "insufficient_evidence"
    ]
    candidates: list[Candidate] = Field(description="Most likely first, at most 3.")
    is_ambiguous: bool = Field(
        description="True when the top candidates cannot be separated without more information."
    )
    distinguishing_signs: list[str] = Field(
        default_factory=list, description="Up to 3 signs that would separate the candidates."
    )
    source_ids: list[str] = Field(default_factory=list)


class FarmerQuestions(BaseModel):
    questions: list[str] = Field(
        description=(
            "2-3 short yes/no or simple questions a farmer can answer by looking at the crop."
        )
    )


class TreatmentItem(BaseModel):
    action: str = Field(description="One short, actionable sentence.")
    category: Literal["immediate", "cultural", "organic", "chemical"]
    source_ids: list[str] = Field(default_factory=list)


class TreatmentPlan(BaseModel):
    summary: str = Field(description="One sentence.")
    items: list[TreatmentItem] = Field(description="At most 7 items, most urgent first.")
    safety_notes: list[str] = Field(default_factory=list, description="At most 2 notes.")


class FollowUpPlan(BaseModel):
    reinspect_in_days: int = Field(ge=1, le=30)
    what_to_check: list[str] = Field(description="At most 3 things to look for.")
    escalate_if: list[str] = Field(default_factory=list, description="At most 2 triggers.")
    reassessment: str = Field(
        default="",
        description="Only when a previous photo exists: continue or change the plan, and why.",
    )


class RiskAssessment(BaseModel):
    level: Literal["low", "moderate", "high", "unknown"]
    score: int = Field(ge=0, le=100)
    factors: list[str] = Field(default_factory=list)
    weather: dict[str, float] | None = None
    method: str = "experimental rule-based heuristic"


class OrchestratorDecision(BaseModel):
    next_action: AgentAction
    reason: str = Field(description="One short sentence.")


class AgentStep(BaseModel):
    action: str
    agent: str
    status: Literal["ok", "fallback", "skipped", "error"]
    summary: str
    decided_by: Literal["llm", "rule"] = "rule"
    reason: str = ""
    ms: float = 0.0


class AgentState(BaseModel):
    evidence: dict[str, Any] = Field(default_factory=dict)
    previous_evidence: dict[str, Any] | None = None
    location: dict[str, float] | None = None
    turn: int = 0
    question_rounds: int = 0
    pending_questions: list[str] = Field(default_factory=list)
    qa_history: list[dict[str, str]] = Field(default_factory=list)
    passages: list[dict[str, Any]] = Field(default_factory=list)
    diagnosis: DifferentialDiagnosis | None = None
    risk: RiskAssessment | None = None
    treatment: TreatmentPlan | None = None
    follow_up: FollowUpPlan | None = None
    answer: dict[str, Any] | None = None
    steps: list[AgentStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    llm_calls: int = 0
