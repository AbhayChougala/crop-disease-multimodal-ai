from fastapi import APIRouter, HTTPException

from backend.app.core.config import get_settings
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.schemas.diagnosis import SourceReference
from backend.app.services.knowledge_service import get_knowledge_service
from backend.app.services.llm_service import get_llm_provider
from backend.app.services.session_store import session_store
from crop_ai.llm.base import ProviderUnavailable

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session = session_store.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Diagnosis session was not found or expired.")
    session_store.append(request.session_id, "user", request.message)
    settings = get_settings()
    passages = []
    if settings.llm_provider != "mock":
        try:
            passages = get_knowledge_service().search(
                request.message,
                crop=session.evidence.get("predicted_crop"),
                disease=session.evidence.get("predicted_disease"),
            )
        except RuntimeError as exc:
            answer = "Grounded guidance is unavailable because the knowledge index is unavailable."
            session_store.append(request.session_id, "assistant", answer)
            return ChatResponse(
                answer=answer,
                warnings=[str(exc)],
                is_mock=False,
                status="unavailable",
            )
        if not passages:
            answer = "Grounded guidance is unavailable because no relevant source was retrieved."
            session_store.append(request.session_id, "assistant", answer)
            return ChatResponse(
                answer=answer,
                warnings=["No provider call was made without retrieved evidence."],
                is_mock=False,
                status="unavailable",
            )
    try:
        response = await get_llm_provider().generate(
            session.evidence,
            request.message,
            passages,
        )
    except ProviderUnavailable as exc:
        answer = "The configured language provider is unavailable."
        session_store.append(request.session_id, "assistant", answer)
        return ChatResponse(
            answer=answer,
            warnings=[str(exc)],
            is_mock=False,
            status="unavailable",
        )
    answer = response["summary"]
    session_store.append(request.session_id, "assistant", answer)
    cited_ids = set(response.get("source_ids", []))
    sources = [
        SourceReference(
            source_id=item.citation_id,
            title=item.title,
            organization=item.organization,
            page=item.page,
            url=item.url or None,
        )
        for item in passages
        if item.citation_id in cited_ids
    ]
    return ChatResponse(
        answer=answer,
        sources=sources,
        warnings=(
            [
                "The mock chat provider does not answer agricultural questions.",
                "No agricultural corpus was retrieved, so grounded advice is unavailable.",
            ]
            if response.get("is_mock")
            else []
        ),
        is_mock=bool(response.get("is_mock")),
        status=str(response.get("status", "available")),
    )


@router.get("/chat/{session_id}/history")
async def chat_history(session_id: str) -> dict[str, object]:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Diagnosis session was not found or expired.")
    return {"session_id": session_id, "messages": list(session.messages)}
