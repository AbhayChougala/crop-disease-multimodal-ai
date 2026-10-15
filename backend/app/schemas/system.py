from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class CapabilitiesResponse(BaseModel):
    device: str
    cuda_available: bool
    mps_available: bool
    torch_available: bool
    model_loaded: bool
    model_mode: str
    rag_available: bool
    llm_provider: str
    gradcam_available: bool


class FeedbackRequest(BaseModel):
    session_id: str
    verdict: str
    actual_disease: str | None = None
    comment: str | None = None


class FeedbackResponse(BaseModel):
    stored: bool
    feedback_id: str
