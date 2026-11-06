from fastapi import APIRouter, HTTPException, Response, status

from app.core.config import get_settings
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatSessionResponse,
    CreateSessionRequest,
    ReassessRequest,
    SessionMessageRequest,
)
from app.services.chat_service import chat_service
from app.services.image_store import image_store
from app.services.session_store import ChatSession, session_store

router = APIRouter(tags=["chat"])


def _serialize(session: ChatSession) -> ChatSessionResponse:
    return ChatSessionResponse(
        session_id=session.session_id,
        mode=session.mode,
        diagnosis_id=session.diagnosis_id,
        project_id=session.evidence.get("project_id") if session.mode == "general" else None,
        provider=session.provider,
        model=session.model,
        preferred_language=session.preferred_language,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=session.messages,
    )


def _require(session_id: str) -> ChatSession:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session was not found or expired.")
    return session


@router.post("/chat/sessions", response_model=ChatSessionResponse, status_code=201)
async def create_session(request: CreateSessionRequest) -> ChatSessionResponse:
    settings = get_settings()
    model = settings.llm_model or settings.openai_model or settings.local_llm_model
    if request.mode == "diagnosis":
        if not request.diagnosis_id:
            raise HTTPException(
                status_code=422, detail="diagnosis_id is required in diagnosis mode."
            )
        existing = session_store.get(request.diagnosis_id)
        if existing is None:
            raise HTTPException(
                status_code=404, detail="Diagnosis session was not found or expired."
            )
        return _serialize(existing)
    if request.project_id and image_store.project(request.project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return _serialize(
        session_store.create(
            mode="general",
            # Only the project id is stored; its photos are read fresh on every message.
            evidence={"project_id": request.project_id} if request.project_id else None,
            provider=settings.llm_provider,
            model=model,
            preferred_language=request.preferred_language,
        )
    )


@router.get("/chat/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_session(session_id: str) -> ChatSessionResponse:
    return _serialize(_require(session_id))


@router.get("/chat/sessions/{session_id}/messages")
async def get_messages(session_id: str) -> dict[str, object]:
    session = _require(session_id)
    return {"session_id": session_id, "messages": session.messages}


@router.post("/chat/sessions/{session_id}/messages", response_model=ChatResponse)
async def send_session_message(session_id: str, request: SessionMessageRequest) -> ChatResponse:
    location = request.location.model_dump() if request.location else None
    return await chat_service.answer(
        _require(session_id), request.message, request.preferred_language, location
    )


@router.post("/chat/sessions/{session_id}/reassess", response_model=ChatResponse)
async def reassess_session(session_id: str, request: ReassessRequest) -> ChatResponse:
    """Link a new photo's diagnosis to this chat and let the agents compare it with the old one."""
    session = _require(session_id)
    if session.mode != "diagnosis":
        raise HTTPException(status_code=409, detail="Session is not linked to a diagnosis.")
    if request.diagnosis_id == session_id:
        raise HTTPException(status_code=422, detail="Upload a new photo to reassess.")
    new_diagnosis = _require(request.diagnosis_id)
    if new_diagnosis.mode != "diagnosis" or not new_diagnosis.evidence:
        raise HTTPException(status_code=422, detail="The new diagnosis has no evidence.")
    location = request.location.model_dump() if request.location else None
    return await chat_service.reassess(session, new_diagnosis.evidence, None, location)


@router.delete("/chat/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str) -> Response:
    if not session_store.delete(session_id):
        raise HTTPException(status_code=404, detail="Chat session was not found or expired.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return await chat_service.answer(
        _require(request.session_id), request.message, request.preferred_language
    )


@router.post("/chat/general", response_model=ChatResponse, status_code=201)
async def general_chat(request: SessionMessageRequest) -> ChatResponse:
    settings = get_settings()
    session = session_store.create(
        mode="general",
        provider=settings.llm_provider,
        model=settings.llm_model or settings.openai_model or settings.local_llm_model,
        preferred_language=request.preferred_language or "en",
    )
    return await chat_service.answer(session, request.message, request.preferred_language)


@router.post("/chat/diagnosis/{diagnosis_id}", response_model=ChatResponse)
async def diagnosis_chat(diagnosis_id: str, request: SessionMessageRequest) -> ChatResponse:
    session = _require(diagnosis_id)
    if session.mode != "diagnosis":
        raise HTTPException(status_code=409, detail="Session is not linked to a diagnosis.")
    return await chat_service.answer(session, request.message, request.preferred_language)


@router.get("/chat/{session_id}/history")
async def chat_history(session_id: str) -> dict[str, object]:
    session = _require(session_id)
    return {"session_id": session_id, "messages": session.messages}


@router.get("/llm/status")
async def llm_status() -> dict[str, object]:
    return await chat_service.provider_status()
