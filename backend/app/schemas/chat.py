from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    mode: Literal["general", "diagnosis"] = "general"
    diagnosis_id: str | None = Field(default=None, min_length=1, max_length=100)
    preferred_language: Literal["en", "kn", "hi"] = "en"


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)
    preferred_language: Literal["en", "kn", "hi"] | None = None


class SessionMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    preferred_language: Literal["en", "kn", "hi"] | None = None


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


class ChatSessionResponse(BaseModel):
    session_id: str
    mode: Literal["general", "diagnosis"]
    diagnosis_id: str | None = None
    provider: str
    model: str
    preferred_language: str
    created_at: str
    updated_at: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
