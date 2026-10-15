from __future__ import annotations

from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class Prediction(BaseModel):
    crop: str
    disease: str
    confidence: float = Field(ge=0, le=1)
    top_k: list[TopPrediction]


class Severity(BaseModel):
    method: str
    affected_area_percent: float | None = Field(default=None, ge=0, le=100)
    category: str
    is_experimental: bool


class Artifacts(BaseModel):
    original_image: str
    preprocessed_image: str
    leaf_mask: str | None = None
    lesion_mask: str | None = None
    gradcam: str | None = None


class Evidence(BaseModel):
    crop: str
    disease: str
    confidence: float
    severity: str
    affected_area_percent: float | None
    segmentation_valid: bool
    model_id: str


class SourceReference(BaseModel):
    source_id: str
    title: str
    organization: str | None = None
    page: int | None = None
    url: str | None = None


class Advisory(BaseModel):
    summary: str
    symptoms: list[str] = []
    possible_causes: list[str] = []
    prevention: list[str] = []
    management: list[str] = []
    when_to_seek_expert_help: str
    sources: list[SourceReference] = []
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
    warnings: list[str]


class PreprocessPreviewResponse(BaseModel):
    mode: str
    preprocessed_image: str
    leaf_mask: str | None
    lesion_mask: str | None
    segmentation_valid: bool
    foreground_percent: float
    warnings: list[str]
