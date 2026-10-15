from __future__ import annotations

from typing import Any

from crop_ai.cv.leaf_segmentation import LeafSegmentation
from crop_ai.cv.severity import SeverityResult
from crop_ai.models.mock_classifier import MockPrediction


def build_evidence(
    prediction: MockPrediction,
    severity: SeverityResult,
    leaf: LeafSegmentation,
) -> dict[str, Any]:
    """Build evidence strictly from values emitted by the visual pipeline."""
    return {
        "crop": prediction.crop,
        "disease": prediction.disease,
        "confidence": prediction.confidence,
        "severity": severity.category,
        "affected_area_percent": severity.affected_area_percent,
        "segmentation_valid": leaf.is_valid,
        "model_id": "mock-classifier-v1",
    }
