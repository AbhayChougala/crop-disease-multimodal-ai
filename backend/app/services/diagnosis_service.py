from __future__ import annotations

import base64
import time
import uuid
from typing import Any

import numpy as np

from backend.app.core.config import Settings
from backend.app.schemas.diagnosis import DiagnosisResponse, PreprocessPreviewResponse
from crop_ai.cv.decode import decode_image, encode_jpeg, encode_png
from crop_ai.cv.preprocessing import preprocess_image
from crop_ai.cv.severity import calculate_severity
from crop_ai.evidence.builder import build_evidence
from crop_ai.llm.mock_provider import MockLLMProvider
from crop_ai.models.mock_classifier import MockClassifier


def _data_url(data: bytes, media_type: str) -> str:
    payload = base64.b64encode(data).decode("ascii")
    return f"data:{media_type};base64,{payload}"


def _mask_url(mask: np.ndarray) -> str:
    return _data_url(encode_png(mask), "image/png")


class DiagnosisService:
    def __init__(self, settings: Settings):
        if settings.model_mode != "mock":
            raise RuntimeError(
                "MODEL_MODE=real is unavailable until a validated model artifact loader "
                "is configured."
            )
        self.settings = settings
        self.classifier = MockClassifier()
        self.llm = MockLLMProvider()

    def preview(self, image_data: bytes, mode: str | None = None) -> PreprocessPreviewResponse:
        image = decode_image(image_data)
        selected_mode = mode or self.settings.cv_preprocessing_mode
        result = preprocess_image(image, selected_mode)  # type: ignore[arg-type]
        return PreprocessPreviewResponse(
            mode=selected_mode,
            preprocessed_image=_data_url(encode_jpeg(result.image), "image/jpeg"),
            leaf_mask=_mask_url(result.leaf.mask),
            lesion_mask=_mask_url(result.lesion_mask),
            segmentation_valid=result.leaf.is_valid,
            foreground_percent=result.leaf.foreground_percent,
            warnings=list(result.warnings),
        )

    async def diagnose(self, image_data: bytes) -> DiagnosisResponse:
        started = time.perf_counter()
        image = decode_image(image_data)

        stage = time.perf_counter()
        cv_result = preprocess_image(image, self.settings.cv_preprocessing_mode)
        preprocessing_ms = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        prediction = self.classifier.predict(cv_result.image)
        classification_ms = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        severity = calculate_severity(cv_result.leaf.mask, cv_result.lesion_mask)
        severity_ms = (time.perf_counter() - stage) * 1000
        evidence = build_evidence(prediction, severity, cv_result.leaf)

        stage = time.perf_counter()
        advisory = await self.llm.generate(evidence)
        llm_ms = (time.perf_counter() - stage) * 1000

        warnings = [
            "DEMO MODE: trained model not loaded; prediction and confidence are placeholders.",
            "Severity is an unvalidated experimental color-segmentation estimate.",
            *cv_result.warnings,
        ]
        if prediction.confidence < self.settings.confidence_threshold:
            warnings.append(
                "Low-confidence prediction. Provide another clear leaf image or seek "
                "expert verification."
            )

        payload: dict[str, Any] = {
            "session_id": str(uuid.uuid4()),
            "is_mock": True,
            "prediction": {
                "crop": prediction.crop,
                "disease": prediction.disease,
                "confidence": prediction.confidence,
                "top_k": [
                    {"label": item.label, "confidence": item.confidence}
                    for item in prediction.top_k
                ],
            },
            "severity": severity.__dict__,
            "artifacts": {
                "original_image": _data_url(encode_jpeg(image), "image/jpeg"),
                "preprocessed_image": _data_url(encode_jpeg(cv_result.image), "image/jpeg"),
                "leaf_mask": _mask_url(cv_result.leaf.mask),
                "lesion_mask": _mask_url(cv_result.lesion_mask),
                "gradcam": None,
            },
            "evidence": evidence,
            "advisory": advisory,
            "timings_ms": {
                "preprocessing": round(preprocessing_ms, 2),
                "classification": round(classification_ms, 2),
                "gradcam": 0.0,
                "severity": round(severity_ms, 2),
                "retrieval": 0.0,
                "llm": round(llm_ms, 2),
                "total": round((time.perf_counter() - started) * 1000, 2),
            },
            "warnings": warnings,
        }
        return DiagnosisResponse.model_validate(payload)
