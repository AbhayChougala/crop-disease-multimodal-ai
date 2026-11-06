from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.chat import AgentStepOut, ChatCitation, Location


class AgentRunRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100, description="Diagnosis session ID.")
    message: str = Field(default="", max_length=2000)
    preferred_language: Literal["en", "kn", "hi"] | None = None
    location: Location | None = None
    # False (default): a result already saved for this photo is returned without re-running.
    rerun: bool = False


class AgentMemory(BaseModel):
    """What the agent team already knows for this diagnosis, shown on the Agents page."""

    passages: int
    has_diagnosis: bool
    has_risk: bool
    has_treatment: bool
    has_follow_up: bool
    pending_questions: list[str] = Field(default_factory=list)
    answered_questions: int = 0


class AgentRunResponse(BaseModel):
    session_id: str
    agent: str
    steps: list[AgentStepOut]
    output: dict[str, Any] | None = Field(
        default=None, description="The requested agent's structured result."
    )
    sources: list[ChatCitation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    memory: AgentMemory
    is_mock: bool
    llm_calls: int
    total_ms: float
    # True when the saved result for this photo was returned and nothing was re-run.
    reused: bool = False


class ProjectPhotoMemory(BaseModel):
    diagnosis_id: str
    # None when the photo's diagnosis can no longer be found (archive deleted).
    memory: AgentMemory | None
