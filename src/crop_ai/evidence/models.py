from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class EvidencePrediction(BaseModel):
    rank: int = Field(ge=1)
    class_id: str | None = None
    crop: str
    disease: str
    confidence: float = Field(ge=0, le=1)


class PreprocessingEvidence(BaseModel):
    mode: str
    leaf_mask_valid: bool
    foreground_percent: float = Field(ge=0, le=100)


class EvidenceProvenance(BaseModel):
    classifier: str
    severity: str
    gradcam: str


class MultimodalEvidence(BaseModel):
    diagnosis_id: str
    timestamp: datetime
    image_identifier: str
    predicted_crop: str | None
    predicted_disease: str | None
    class_id: str | None = None
    top_k: list[EvidencePrediction] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    calibration_status: str
    model_id: str
    model_version: str
    model_mode: Literal["DEMO", "REAL", "UNAVAILABLE"]
    preprocessing: PreprocessingEvidence
    gradcam_available: bool
    segmentation_status: Literal["valid", "invalid", "unavailable"]
    severity_status: Literal["unavailable", "experimental", "validated"]
    severity_category: str
    severity_percentage: float | None = Field(default=None, ge=0, le=100)
    measured_visual_features: dict[str, float | int | bool | str] = Field(default_factory=dict)
    processing_warnings: list[str] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)
    image_quality_warnings: list[str] = Field(default_factory=list)
    provenance: EvidenceProvenance

    def llm_payload(self) -> dict[str, Any]:
        """Return structured fields only; never synthesize image descriptions."""
        return self.model_dump(mode="json")
