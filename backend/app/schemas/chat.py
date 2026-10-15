from __future__ import annotations

from pydantic import BaseModel, Field

from backend.app.schemas.diagnosis import SourceReference


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    warnings: list[str]
    is_mock: bool
    status: str = "available"
