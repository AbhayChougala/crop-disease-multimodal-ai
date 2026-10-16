from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from crop_ai.evidence.models import EvidencePrediction, EvidenceProvenance, PreprocessingEvidence


class TopPrediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class Prediction(BaseModel):
    crop: str | None
    disease: str | None
    class_id: str | None = None
    confidence: float = Field(ge=0, le=1)
    raw_confidence: float | None = Field(default=None, ge=0, le=1)
    calibration_status: str = "not_configured"
    architecture: str | None = None
    checkpoint_id: str | None = None
    inference_mode: str = "demo"
    abstained: bool = False
    abstention_reason: str | None = None
    top_k: list[TopPrediction]


class Severity(BaseModel):
    method: str
    affected_area_percent: float | None = Field(default=None, ge=0, le=100)
    category: str
    status: str
    is_experimental: bool
    warning: str | None = None


class Artifacts(BaseModel):
    original_image: str
    preprocessed_image: str
    leaf_mask: str | None = None
    lesion_mask: str | None = None
    leaf_isolated: str | None = None
    lesion_overlay: str | None = None
    gradcam: str | None = None


class Evidence(BaseModel):
    diagnosis_id: str
    timestamp: datetime
    image_identifier: str
    predicted_crop: str | None
    predicted_disease: str | None
    class_id: str | None = None
    top_k: list[EvidencePrediction]
    confidence: float
    calibration_status: str
    model_id: str
    model_version: str
    model_mode: Literal["DEMO", "REAL", "UNAVAILABLE"]
    preprocessing: PreprocessingEvidence
    gradcam_available: bool
    segmentation_status: Literal["valid", "invalid", "unavailable"]
    severity_status: Literal["unavailable", "experimental", "validated"]
    severity_category: str
    severity_percentage: float | None
    measured_visual_features: dict[str, Any]
    processing_warnings: list[str]
    uncertainty_flags: list[str]
    image_quality_warnings: list[str]
    provenance: EvidenceProvenance


class SourceReference(BaseModel):
    source_id: str
    title: str
    organization: str | None = None
    page: int | None = None
    url: str | None = None


class Advisory(BaseModel):
    summary: str
    symptoms: list[str] = Field(default_factory=list)
    possible_causes: list[str] = Field(default_factory=list)
    prevention: list[str] = Field(default_factory=list)
    management: list[str] = Field(default_factory=list)
    when_to_seek_expert_help: str
    sources: list[SourceReference] = Field(default_factory=list)
    is_mock: bool


class Timings(BaseModel):
    preprocessing: float
    classification: float
    gradcam: float
    severity: float
    retrieval: float
    llm: float
    total: float


class DiagnosisResponse(BaseModel):
    session_id: str
    is_mock: bool
    prediction: Prediction
    severity: Severity
    artifacts: Artifacts
    evidence: Evidence
    advisory: Advisory
    timings_ms: Timings
    runtime: dict[str, str | bool]
    warnings: list[str]


class PreprocessPreviewResponse(BaseModel):
    mode: str
    preprocessed_image: str
    leaf_mask: str | None
    lesion_mask: str | None
    leaf_isolated: str | None
    lesion_overlay: str | None
    segmentation_valid: bool
    foreground_percent: float
    warnings: list[str]
