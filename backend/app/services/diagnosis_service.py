from __future__ import annotations

import base64
import hashlib
import time
import uuid
from typing import Any

import numpy as np

from app.core.config import Settings
from app.schemas.diagnosis import DiagnosisResponse, PreprocessPreviewResponse
from app.services.diagnosis_archive import save_diagnosis
from app.services.knowledge_service import get_knowledge_service
from app.services.llm_service import get_llm_provider
from app.services.model_manager import ModelManager, get_model_manager
from app.services.session_store import session_store
from crop_ai.cv.decode import decode_image, encode_jpeg, encode_png
from crop_ai.cv.preprocessing import preprocess_image
from crop_ai.cv.severity import calculate_severity
from crop_ai.device import get_device_info
from crop_ai.evidence.builder import build_evidence
from crop_ai.llm.base import ProviderUnavailable


def _data_url(data: bytes, media_type: str) -> str:
    payload = base64.b64encode(data).decode("ascii")
    return f"data:{media_type};base64,{payload}"


def _mask_url(mask: np.ndarray) -> str:
    return _data_url(encode_png(mask), "image/png")


class DiagnosisService:
    def __init__(self, settings: Settings, model_manager: ModelManager | None = None):
        self.settings = settings
        self.model_manager = model_manager or get_model_manager()

    def preview(self, image_data: bytes, mode: str | None = None) -> PreprocessPreviewResponse:
        image = decode_image(image_data)
        selected_mode = mode or self.settings.cv_preprocessing_mode
        result = preprocess_image(image, selected_mode)  # type: ignore[arg-type]
        return PreprocessPreviewResponse(
            mode=selected_mode,
            preprocessed_image=_data_url(encode_jpeg(result.image), "image/jpeg"),
            leaf_mask=_mask_url(result.leaf.mask),
            lesion_mask=_mask_url(result.lesion_mask),
            leaf_isolated=_data_url(encode_jpeg(result.leaf_isolated), "image/jpeg"),
            lesion_overlay=_data_url(encode_jpeg(result.lesion_overlay), "image/jpeg"),
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
        prediction = self.model_manager.predict(cv_result.image)
        classification_ms = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        gradcam_result = self.model_manager.gradcam(cv_result.image)
        gradcam_ms = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        severity = calculate_severity(
            cv_result.leaf.mask,
            cv_result.lesion_mask,
            thresholds=(
                (self.settings.severity_low_max, "Operational low"),
                (self.settings.severity_moderate_max, "Operational moderate"),
                (self.settings.severity_high_max, "Operational high"),
                (101.0, "Operational very high"),
            ),
        )
        severity_ms = (time.perf_counter() - stage) * 1000
        is_demo = self.model_manager.status.state == "DEMO"
        disease = prediction.disease
        crop = prediction.crop
        confidence = getattr(
            prediction,
            "calibrated_confidence",
            None,
        ) or getattr(prediction, "confidence", getattr(prediction, "raw_confidence", 0.0))
        warnings = [
            "Severity is an unvalidated experimental color-segmentation estimate.",
            *cv_result.warnings,
        ]
        if is_demo:
            warnings.insert(
                0,
                "DEMO MODE: trained model not loaded; prediction and confidence are placeholders.",
            )
        if getattr(prediction, "abstained", False):
            warnings.append(str(prediction.abstention_reason))
        elif confidence < self.settings.confidence_threshold:
            warnings.append(
                "Low-confidence prediction. Provide another clear leaf image or seek "
                "expert verification."
            )

        session_id = str(uuid.uuid4())
        model_status = self.model_manager.status
        evidence_model = build_evidence(
            prediction,
            severity,
            cv_result.leaf,
            diagnosis_id=session_id,
            image_identifier=hashlib.sha256(image_data).hexdigest(),
            model_id=model_status.checkpoint_id or "mock-classifier-v1",
            model_version=getattr(prediction, "model_version", "demo-1"),
            model_mode=model_status.state,
            preprocessing_mode=self.settings.cv_preprocessing_mode,
            gradcam_available=gradcam_result is not None,
            warnings=warnings,
            confidence_threshold=self.settings.confidence_threshold,
        )
        evidence = evidence_model.llm_payload()

        stage = time.perf_counter()
        passages = []
        if self.settings.llm_provider != "mock":
            try:
                passages = get_knowledge_service().search(
                    f"{crop or ''} {disease or ''} diagnosis prevention uncertainty",
                    crop=crop,
                    disease=disease,
                )
            except RuntimeError as exc:
                warnings.append(str(exc))
        retrieval_ms = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        try:
            if self.settings.llm_provider != "mock" and not passages:
                raise ProviderUnavailable(
                    "Grounded guidance is unavailable because no relevant source was retrieved."
                )
            advisory = await get_llm_provider().generate(evidence, passages=passages)
            cited = set(advisory.get("source_ids", []))
            advisory["sources"] = [
                {
                    "source_id": item.citation_id,
                    "title": item.title,
                    "organization": item.organization,
                    "page": item.page,
                    "url": item.url or None,
                }
                for item in passages
                if item.citation_id in cited
            ]
        except ProviderUnavailable as exc:
            warnings.append(str(exc))
            advisory = {
                "summary": "Grounded language-model guidance is unavailable for this diagnosis.",
                "symptoms": [],
                "possible_causes": [],
                "prevention": [],
                "management": [],
                "when_to_seek_expert_help": (
                    "Seek a qualified agricultural expert before making treatment decisions."
                ),
                "sources": [],
                "is_mock": False,
            }
        llm_ms = (time.perf_counter() - stage) * 1000

        session_store.put(session_id, evidence)
        payload: dict[str, Any] = {
            "session_id": session_id,
            "is_mock": is_demo,
            "prediction": {
                "crop": crop,
                "disease": disease,
                "class_id": getattr(prediction, "class_id", None),
                "confidence": confidence,
                "raw_confidence": getattr(prediction, "raw_confidence", confidence),
                "calibration_status": getattr(prediction, "calibration_status", "not_configured"),
                "architecture": getattr(prediction, "architecture", "mock"),
                "checkpoint_id": getattr(prediction, "checkpoint_id", None),
                "inference_mode": getattr(prediction, "inference_mode", "demo"),
                "abstained": getattr(prediction, "abstained", False),
                "abstention_reason": getattr(prediction, "abstention_reason", None),
                "top_k": [
                    {
                        "label": getattr(item, "label", f"{item.crop} - {item.disease}"),
                        "confidence": item.confidence,
                    }
                    for item in prediction.top_k
                ],
            },
            "severity": severity.__dict__,
            "artifacts": {
                "original_image": _data_url(encode_jpeg(image), "image/jpeg"),
                "preprocessed_image": _data_url(encode_jpeg(cv_result.image), "image/jpeg"),
                "leaf_mask": _mask_url(cv_result.leaf.mask),
                "lesion_mask": _mask_url(cv_result.lesion_mask),
                "leaf_isolated": _data_url(encode_jpeg(cv_result.leaf_isolated), "image/jpeg"),
                "lesion_overlay": _data_url(encode_jpeg(cv_result.lesion_overlay), "image/jpeg"),
                "gradcam": (
                    _data_url(encode_jpeg(gradcam_result[0]), "image/jpeg")
                    if gradcam_result
                    else None
                ),
            },
            "evidence": evidence,
            "advisory": advisory,
            "timings_ms": {
                "preprocessing": round(preprocessing_ms, 2),
                "classification": round(classification_ms, 2),
                "gradcam": round(gradcam_ms, 2),
                "severity": round(severity_ms, 2),
                "retrieval": round(retrieval_ms, 2),
                "llm": round(llm_ms, 2),
                "total": round((time.perf_counter() - started) * 1000, 2),
            },
            "runtime": {
                "device": get_device_info().selected,
                "model_mode": model_status.state,
                "model_loaded": model_status.loaded,
            },
            "warnings": warnings,
        }
        response = DiagnosisResponse.model_validate(payload)
        save_diagnosis(session_id, response.model_dump(mode="json"))
        return response
