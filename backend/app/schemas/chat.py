from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    mode: Literal["general", "diagnosis"] = "general"
    diagnosis_id: str | None = Field(default=None, min_length=1, max_length=100)
    # General mode only: the chatbot answers with this project's photos and diagnoses as context.
    project_id: str | None = Field(default=None, min_length=1, max_length=100)
    preferred_language: Literal["en", "kn", "hi"] = "en"


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)
    preferred_language: Literal["en", "kn", "hi"] | None = None


class Location(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class SessionMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    preferred_language: Literal["en", "kn", "hi"] | None = None
    location: Location | None = None


class ReassessRequest(BaseModel):
    diagnosis_id: str = Field(min_length=1, max_length=100)
    location: Location | None = None


class AgentStepOut(BaseModel):
    action: str
    agent: str
    status: str
    summary: str
    decided_by: str
    reason: str = ""
    ms: float = 0.0


class FollowUpOut(BaseModel):
    reinspect_in_days: int
    due_date: str
    what_to_check: list[str] = Field(default_factory=list)
    escalate_if: list[str] = Field(default_factory=list)
    reassessment: str = ""


class ChatCitation(BaseModel):
    source_id: str
    chunk_id: str
    document_id: str
    title: str
    organization: str | None = None
    page: int | None = None
    section_name: str | None = None
    url: str | None = None
    excerpt: str
    score: float


class GenerationMetadata(BaseModel):
    provider: str
    model: str
    retriever: str
    retrieved_chunks: int
    context_characters: int
    retrieval_ms: float
    llm_ms: float
    total_ms: float


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    answer: str
    sources: list[ChatCitation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    follow_up_suggestions: list[str] = Field(default_factory=list)
    is_mock: bool
    insufficient_evidence: bool = False
    status: str = "available"
    preferred_language: str = "en"
    created_at: str
    metadata: GenerationMetadata
    agent_trace: list[AgentStepOut] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    follow_up: FollowUpOut | None = None
    risk_level: str | None = None


class ChatSessionResponse(BaseModel):
    session_id: str
    mode: Literal["general", "diagnosis"]
    diagnosis_id: str | None = None
    project_id: str | None = None
    provider: str
    model: str
    preferred_language: str
    created_at: str
    updated_at: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
