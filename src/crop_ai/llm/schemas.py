from __future__ import annotations

from pydantic import BaseModel, Field


class GroundedAdvisory(BaseModel):
    summary: str
    symptoms: list[str] = Field(default_factory=list)
    possible_causes: list[str] = Field(default_factory=list)
    prevention: list[str] = Field(default_factory=list)
    management: list[str] = Field(default_factory=list)
    when_to_seek_expert_help: str
    source_ids: list[str] = Field(default_factory=list)
    uncertainty_note: str
