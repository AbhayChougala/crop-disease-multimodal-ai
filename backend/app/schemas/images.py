from pydantic import BaseModel, Field


class StoredImageOut(BaseModel):
    diagnosis_id: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    crop: str | None
    disease: str | None
    confidence: float | None
    is_mock: bool
    created_at: str
    project_id: str | None
    note: str
    image_url: str


class StoredImageList(BaseModel):
    enabled: bool
    total: int
    images: list[StoredImageOut]


class ImageUpdate(BaseModel):
    """Only the fields sent are changed; `project_id: null` removes the photo from its project."""

    project_id: str | None = None
    note: str | None = Field(default=None, max_length=300)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)


class ProjectOut(BaseModel):
    project_id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    image_count: int


class DiagnosisCount(BaseModel):
    label: str
    count: int


class ProjectDetail(ProjectOut):
    images: list[StoredImageOut]
    # How many photos in the project got each diagnosis, most common first.
    summary: list[DiagnosisCount]
