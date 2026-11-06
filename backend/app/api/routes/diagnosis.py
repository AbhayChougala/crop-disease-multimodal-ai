import logging
import sqlite3
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from app.core.config import Settings, get_settings
from app.schemas.diagnosis import DiagnosisResponse, PreprocessPreviewResponse
from app.services.diagnosis_archive import load_diagnosis
from app.services.diagnosis_service import DiagnosisService
from app.services.image_store import image_store, sha256_of
from app.services.model_manager import get_model_manager
from app.services.project_context import ensure_diagnosis_session
from crop_ai.cv.decode import ImageDecodeError

logger = logging.getLogger("crop-disease-api")
router = APIRouter(tags=["diagnosis"])
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


async def _read_upload(file: UploadFile, settings: Settings) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload a JPEG, PNG, or WebP image.",
        )
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds the {settings.max_upload_mb} MB upload limit.",
        )
    return data


@router.post("/preprocess/preview", response_model=PreprocessPreviewResponse)
async def preview(
    file: Annotated[UploadFile, File()],
    settings: Annotated[Settings, Depends(get_settings)],
    mode: Annotated[Literal["raw", "standard", "segmented"] | None, Query()] = None,
) -> PreprocessPreviewResponse:
    data = await _read_upload(file, settings)
    try:
        return DiagnosisService(settings).preview(data, mode)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _saved_diagnosis(data: bytes) -> DiagnosisResponse | None:
    """The stored diagnosis of an identical photo, if it was made by the model loaded now."""
    record = image_store.find_by_sha256(sha256_of(data))
    if record is None:
        return None
    try:
        payload = load_diagnosis(record.diagnosis_id)
    except (ValueError, OSError):
        return None
    if payload is None:
        return None
    runtime = get_model_manager().status
    same_model = payload.get("is_mock") == (runtime.state == "DEMO") and (
        payload["is_mock"]
        or payload.get("prediction", {}).get("checkpoint_id") == runtime.checkpoint_id
    )
    if not same_model:
        return None  # e.g. analysed in DEMO mode, or with another checkpoint: analyse again.
    # Chat sessions expire; recreate it so the chat and agents work on this diagnosis.
    ensure_diagnosis_session(record.diagnosis_id)
    return DiagnosisResponse.model_validate(
        {**payload, "from_saved": True, "saved_at": record.created_at}
    )


@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(
    file: Annotated[UploadFile, File()],
    settings: Annotated[Settings, Depends(get_settings)],
    project_id: Annotated[str | None, Form()] = None,
    note: Annotated[str, Form(max_length=300)] = "",
    reanalyse: Annotated[bool, Query()] = False,
) -> DiagnosisResponse:
    """Analyse a leaf photo. A photo analysed before returns its saved diagnosis instantly
    (unless `reanalyse=true`); `project_id` and `note` file it under a project with a label."""
    data = await _read_upload(file, settings)
    if project_id and image_store.project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found.")

    if settings.store_uploaded_images and not reanalyse:
        saved = _saved_diagnosis(data)
        if saved is not None:
            changes: dict[str, str] = {}
            if project_id:
                changes["project_id"] = project_id
            if note.strip():
                changes["note"] = note
            if changes:
                image_store.update(saved.session_id, **changes)
            if project_id:
                image_store.touch_project(project_id)
            return saved

    try:
        response = await DiagnosisService(settings).diagnose(data)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if settings.store_uploaded_images:
        try:
            image_store.save(
                response.session_id,
                data,
                filename=file.filename or "upload",
                content_type=file.content_type or "application/octet-stream",
                crop=response.prediction.crop,
                disease=response.prediction.disease,
                confidence=response.prediction.confidence,
                is_mock=response.is_mock,
                project_id=project_id,
                note=note,
            )
            if project_id:
                image_store.touch_project(project_id)
        except sqlite3.Error:
            # A storage failure must not lose the diagnosis the farmer is waiting for.
            logger.exception("could not store uploaded image %s", response.session_id)
    return response
