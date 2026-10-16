from __future__ import annotations

from pydantic import BaseModel, Field


class GroundedAnswer(BaseModel):
    answer: str
    answer_type: str = "grounded_knowledge"
    confidence_language: str
    source_ids: list[str] = Field(default_factory=list)
    unsupported_claims_detected: bool = False
    insufficient_evidence: bool = False
    follow_up_suggestions: list[str] = Field(default_factory=list)
    safety_notice: str


class DiagnosisNarrative(BaseModel):
    summary: str
    what_model_detected: str
    prediction_interpretation: str
    gradcam_explanation: str
    severity_explanation: str
    disease_information: str | None = None
    environmental_conditions: list[str] = Field(default_factory=list)
    prevention_information: list[str] = Field(default_factory=list)
    general_management_information: list[str] = Field(default_factory=list)
    uncertainty: str
    source_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    expert_verification_notice: str


class GroundedAdvisory(BaseModel):
    summary: str
    symptoms: list[str] = Field(default_factory=list)
    possible_causes: list[str] = Field(default_factory=list)
    prevention: list[str] = Field(default_factory=list)
    management: list[str] = Field(default_factory=list)
    when_to_seek_expert_help: str
    source_ids: list[str] = Field(default_factory=list)
    uncertainty_note: str
