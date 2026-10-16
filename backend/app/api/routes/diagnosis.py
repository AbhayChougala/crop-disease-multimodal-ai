from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.core.config import Settings, get_settings
from app.schemas.diagnosis import DiagnosisResponse, PreprocessPreviewResponse
from app.services.diagnosis_service import DiagnosisService
from crop_ai.cv.decode import ImageDecodeError

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


@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(
    file: Annotated[UploadFile, File()],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DiagnosisResponse:
    data = await _read_upload(file, settings)
    try:
        return await DiagnosisService(settings).diagnose(data)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
