from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from crop_ai.cv.leaf_segmentation import LeafSegmentation
from crop_ai.cv.severity import SeverityResult
from crop_ai.evidence.models import (
    EvidencePrediction,
    EvidenceProvenance,
    MultimodalEvidence,
    PreprocessingEvidence,
)


def _confidence(prediction: Any) -> float:
    calibrated = getattr(prediction, "calibrated_confidence", None)
    return float(
        calibrated
        if calibrated is not None
        else getattr(prediction, "confidence", getattr(prediction, "raw_confidence", 0.0))
    )


def build_evidence(
    prediction: Any,
    severity: SeverityResult,
    leaf: LeafSegmentation,
    *,
    diagnosis_id: str,
    image_identifier: str,
    model_id: str,
    model_version: str,
    model_mode: str,
    preprocessing_mode: str,
    gradcam_available: bool,
    warnings: list[str] | None = None,
    confidence_threshold: float = 0.65,
) -> MultimodalEvidence:
    """Build evidence strictly from measured values emitted by the visual pipeline."""
    confidence = _confidence(prediction)
    uncertainty_flags: list[str] = []
    if confidence < confidence_threshold:
        uncertainty_flags.append("confidence_below_operational_threshold")
    if getattr(prediction, "abstained", False):
        uncertainty_flags.append("classifier_abstained")
    if model_mode != "REAL":
        uncertainty_flags.append("not_a_trained_model_diagnosis")

    ranked = [
        EvidencePrediction(
            rank=getattr(item, "rank", rank),
            class_id=getattr(item, "class_id", None),
            crop=item.crop,
            disease=item.disease,
            confidence=float(item.confidence),
        )
        for rank, item in enumerate(getattr(prediction, "top_k", ()), start=1)
    ]
    return MultimodalEvidence(
        diagnosis_id=diagnosis_id,
        timestamp=datetime.now(UTC),
        image_identifier=image_identifier,
        predicted_crop=prediction.crop,
        predicted_disease=prediction.disease,
        class_id=getattr(prediction, "class_id", None),
        top_k=ranked,
        confidence=confidence,
        calibration_status=getattr(prediction, "calibration_status", "not_configured"),
        model_id=model_id,
        model_version=getattr(prediction, "model_version", model_version),
        model_mode=model_mode,
        preprocessing=PreprocessingEvidence(
            mode=preprocessing_mode,
            leaf_mask_valid=leaf.is_valid,
            foreground_percent=leaf.foreground_percent,
        ),
        gradcam_available=gradcam_available,
        segmentation_status="valid" if leaf.is_valid else "invalid",
        severity_status=severity.status,
        severity_category=severity.category,
        severity_percentage=severity.affected_area_percent,
        measured_visual_features={"leaf_foreground_percent": leaf.foreground_percent},
        processing_warnings=warnings or [],
        uncertainty_flags=uncertainty_flags,
        image_quality_warnings=[leaf.warning] if leaf.warning else [],
        provenance=EvidenceProvenance(
            classifier=model_id,
            severity=severity.method,
            gradcam="class_specific_gradcam" if gradcam_available else "unavailable",
        ),
    )
