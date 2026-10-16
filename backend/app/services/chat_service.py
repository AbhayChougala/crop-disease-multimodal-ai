from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any

from app.core.config import get_settings
from app.schemas.chat import ChatCitation, ChatResponse, GenerationMetadata
from app.services.knowledge_service import get_knowledge_service
from app.services.llm_service import get_llm_provider
from app.services.session_store import ChatSession, session_store
from crop_ai.llm.base import ProviderCapabilities, ProviderUnavailable
from crop_ai.rag.contracts import RetrievedPassage

logger = logging.getLogger(__name__)


def _citation(passage: RetrievedPassage) -> ChatCitation:
    return ChatCitation(
        source_id=passage.citation_id,
        chunk_id=passage.chunk_id,
        document_id=passage.document_id,
        title=passage.title,
        organization=passage.organization or None,
        page=passage.page,
        section_name=passage.section_name or None,
        url=passage.url or None,
        excerpt=passage.text[:500],
        score=round(passage.score, 6),
    )


class ChatService:
    async def answer(
        self,
        session: ChatSession,
        message: str,
        preferred_language: str | None = None,
    ) -> ChatResponse:
        started = time.perf_counter()
        language = preferred_language or session.preferred_language
        session_store.append(session.session_id, "user", message)
        settings = get_settings()
        knowledge = get_knowledge_service()
        passages: list[RetrievedPassage] = []
        warnings: list[str] = []

        retrieval_started = time.perf_counter()
        try:
            passages = knowledge.search(
                message,
                crop=session.evidence.get("predicted_crop")
                if session.mode == "diagnosis"
                else None,
                disease=(
                    session.evidence.get("predicted_disease")
                    if session.mode == "diagnosis"
                    else None
                ),
            )
        except RuntimeError as exc:
            warnings.append(str(exc))
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        history = [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in session.messages
        ]
        evidence: dict[str, Any] = (
            session.evidence
            if session.mode == "diagnosis"
            else {
                "diagnosis_id": None,
                "model_mode": "UNAVAILABLE",
                "predicted_crop": None,
                "predicted_disease": None,
                "visual_evidence": "No image or diagnosis evidence is attached to this chat.",
                "uncertainty_flags": ["general_chat_without_visual_evidence"],
            }
        )

        try:
            provider = get_llm_provider()
            capabilities = provider.capabilities()
        except ProviderUnavailable as exc:
            provider = None
            capabilities = ProviderCapabilities(
                settings.llm_provider,
                settings.llm_model or settings.openai_model or settings.local_llm_model,
                False,
                False,
                None,
            )
            warnings.append(str(exc))
        llm_started = time.perf_counter()
        if provider is None:
            response = {
                "answer": "The configured language provider is unavailable.",
                "summary": "The configured language provider is unavailable.",
                "source_ids": [],
                "follow_up_suggestions": [],
                "insufficient_evidence": True,
                "is_mock": False,
                "status": "unavailable",
            }
        elif settings.llm_provider != "mock" and not passages:
            response: dict[str, Any] = {
                "answer": "I do not have sufficient retrieved evidence to answer that safely.",
                "summary": "I do not have sufficient retrieved evidence to answer that safely.",
                "source_ids": [],
                "follow_up_suggestions": [
                    "Try a more specific crop and symptom question.",
                    "Ask a qualified agricultural expert for local guidance.",
                ],
                "insufficient_evidence": True,
                "is_mock": False,
                "status": "insufficient_evidence",
            }
            warnings.append("No provider call was made because no relevant source was retrieved.")
        else:
            try:
                response = await provider.generate(evidence, message, passages, history, language)
            except ProviderUnavailable as exc:
                response = {
                    "answer": "The configured language provider is unavailable.",
                    "summary": "The configured language provider is unavailable.",
                    "source_ids": [],
                    "follow_up_suggestions": [],
                    "insufficient_evidence": True,
                    "is_mock": False,
                    "status": "unavailable",
                }
                warnings.append(str(exc))
                logger.warning("LLM provider failed: %s", exc)
        llm_ms = (time.perf_counter() - llm_started) * 1000

        allowed = {item.citation_id: item for item in passages}
        cited_ids = [item for item in response.get("source_ids", []) if item in allowed]
        citations = [_citation(allowed[item]) for item in dict.fromkeys(cited_ids)]
        if response.get("source_ids") and len(citations) != len(set(response["source_ids"])):
            warnings.append("One or more unsupported citation identifiers were removed.")

        answer = str(response.get("answer") or response.get("summary") or "")
        metadata = GenerationMetadata(
            provider=capabilities.provider,
            model=capabilities.model,
            retriever=settings.rag_retriever,
            retrieved_chunks=len(passages),
            context_characters=sum(len(item.text) for item in passages),
            retrieval_ms=round(retrieval_ms, 2),
            llm_ms=round(llm_ms, 2),
            total_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        assistant_entry = session_store.append(
            session.session_id,
            "assistant",
            answer,
            citations=[item.model_dump() for item in citations],
            metadata=metadata.model_dump(),
        )
        return ChatResponse(
            session_id=session.session_id,
            message_id=assistant_entry["message_id"],
            answer=answer,
            sources=citations,
            warnings=warnings,
            follow_up_suggestions=list(response.get("follow_up_suggestions", []))[:4],
            is_mock=bool(response.get("is_mock")),
            insufficient_evidence=bool(response.get("insufficient_evidence", not citations)),
            status=str(response.get("status", "available")),
            preferred_language=language,
            created_at=assistant_entry["created_at"],
            metadata=metadata,
        )

    async def provider_status(self) -> dict[str, Any]:
        settings = get_settings()
        try:
            provider = get_llm_provider()
        except ProviderUnavailable as exc:
            model = settings.llm_model or settings.openai_model or settings.local_llm_model
            return {
                "status": {
                    "provider": settings.llm_provider,
                    "model": model,
                    "state": "UNAVAILABLE",
                    "detail": str(exc),
                },
                "capabilities": {
                    "provider": settings.llm_provider,
                    "model": model,
                    "streaming": False,
                    "structured_output": False,
                    "context_limit": None,
                },
            }
        return {
            "status": asdict(await provider.health()),
            "capabilities": asdict(provider.capabilities()),
        }


chat_service = ChatService()
