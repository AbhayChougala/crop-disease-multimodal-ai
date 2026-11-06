"""Leaf photos stored in the local database by POST /diagnose, and projects that group them."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from app.core.config import Settings, get_settings
from app.schemas.images import (
    DiagnosisCount,
    ImageUpdate,
    ProjectCreate,
    ProjectDetail,
    ProjectOut,
    ProjectUpdate,
    StoredImageList,
    StoredImageOut,
)
from app.services.image_store import Project, StoredImage, image_store

router = APIRouter(tags=["images"])


def _image(record: StoredImage) -> StoredImageOut:
    return StoredImageOut(**asdict(record), image_url=f"/api/v1/images/{record.diagnosis_id}/file")


def _label(record: StoredImage) -> str:
    if record.crop and record.disease:
        return f"{record.crop} – {record.disease}"
    return "No confident prediction"


def _project(project: Project) -> ProjectOut:
    return ProjectOut(**asdict(project))


def _found_project(project_id: str) -> Project:
    project = image_store.project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


# ---- photos -------------------------------------------------------------------------------


@router.get("/images", response_model=StoredImageList)
def list_images(
    settings: Annotated[Settings, Depends(get_settings)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> StoredImageList:
    return StoredImageList(
        enabled=settings.store_uploaded_images,
        total=image_store.count(),
        images=[_image(record) for record in image_store.list(limit)],
    )


@router.get("/images/{diagnosis_id}/file")
def image_file(diagnosis_id: str) -> Response:
    stored = image_store.get(diagnosis_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Image not found.")
    record, data = stored
    return Response(
        content=data,
        media_type=record.content_type,
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.patch("/images/{diagnosis_id}", response_model=StoredImageOut)
def update_image(diagnosis_id: str, request: ImageUpdate) -> StoredImageOut:
    if image_store.record(diagnosis_id) is None:
        raise HTTPException(status_code=404, detail="Image not found.")
    changes = request.model_dump(include=request.model_fields_set)
    if changes.get("project_id"):
        _found_project(changes["project_id"])
    record = image_store.update(diagnosis_id, **changes)
    assert record is not None
    if record.project_id:
        image_store.touch_project(record.project_id)
    return _image(record)


@router.delete("/images/{diagnosis_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_image(diagnosis_id: str) -> Response:
    if not image_store.delete(diagnosis_id):
        raise HTTPException(status_code=404, detail="Image not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---- projects -----------------------------------------------------------------------------


@router.get("/projects", response_model=list[ProjectOut])
def list_projects() -> list[ProjectOut]:
    return [_project(project) for project in image_store.projects()]


@router.post("/projects", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def create_project(request: ProjectCreate) -> ProjectOut:
    return _project(image_store.create_project(request.name, request.description))


@router.get("/projects/{project_id}", response_model=ProjectDetail)
def get_project(project_id: str) -> ProjectDetail:
    project = _found_project(project_id)
    records = image_store.list(limit=1000, project_id=project_id)
    counts = Counter(_label(record) for record in records)
    return ProjectDetail(
        **asdict(project),
        images=[_image(record) for record in records],
        summary=[DiagnosisCount(label=label, count=count) for label, count in counts.most_common()],
    )


@router.patch("/projects/{project_id}", response_model=ProjectOut)
def update_project(project_id: str, request: ProjectUpdate) -> ProjectOut:
    _found_project(project_id)
    project = image_store.update_project(
        project_id, name=request.name, description=request.description
    )
    assert project is not None
    return _project(project)


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: str) -> Response:
    """Deletes the project only; its photos and diagnoses stay in History."""
    if not image_store.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
