import hashlib
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from backend.app.core.config import PROJECT_ROOT, get_settings
from backend.app.services.knowledge_service import get_knowledge_service

router = APIRouter(tags=["knowledge"])


@router.get("/rag/status")
def rag_status() -> dict[str, object]:
    return get_knowledge_service().status()


@router.get("/knowledge/sources")
def knowledge_sources() -> dict[str, object]:
    status = get_knowledge_service().status()
    return {
        "available": status["available"],
        "sources": status["sources"],
        "indexed_documents": status["indexed_documents"],
    }


@router.post("/rag/rebuild")
def rebuild_local_rag() -> dict[str, object]:
    """Rebuild only configured project-local indexes; no paths come from the request."""
    settings = get_settings()
    if settings.app_env.casefold() != "development":
        raise HTTPException(status_code=403, detail="Index rebuild is development-only.")
    try:
        from crop_ai.rag.embeddings import SentenceTransformerEmbeddingProvider
        from crop_ai.rag.ingest import build_chunks
        from crop_ai.rag.semantic import FaissStore
        from crop_ai.rag.store import TfidfStore

        manifest_path = PROJECT_ROOT / "knowledge/source_manifest.csv"
        chunks = build_chunks(
            manifest_path,
            PROJECT_ROOT / "knowledge/source_docs",
            PROJECT_ROOT / settings.rag_chunks_path,
        )
        if not chunks:
            raise ValueError("The knowledge manifest contains no approved documents.")
        if settings.rag_retriever in {"tfidf", "hybrid"}:
            TfidfStore.build(chunks).save(PROJECT_ROOT / settings.rag_index_path)
        if settings.rag_retriever in {"semantic", "hybrid"}:
            embedder = SentenceTransformerEmbeddingProvider(
                settings.embedding_model,
                allow_download=False,
            )
            semantic = FaissStore.build(
                chunks,
                embedder,
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            )
            semantic.save(
                PROJECT_ROOT / settings.rag_semantic_index_path,
                PROJECT_ROOT / settings.rag_semantic_metadata_path,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    get_knowledge_service.cache_clear()
    return {
        "status": "rebuilt",
        "retriever": settings.rag_retriever,
        "chunk_count": len(chunks),
        "note": "Semantic model downloads are never initiated by this endpoint.",
    }


@router.get("/knowledge/search")
def search_knowledge(
    query: Annotated[str, Query(min_length=2, max_length=500)],
    crop: Annotated[str | None, Query(max_length=100)] = None,
    disease: Annotated[str | None, Query(max_length=100)] = None,
) -> dict[str, object]:
    service = get_knowledge_service()
    try:
        passages = service.search(query, crop=crop, disease=disease)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "available" if passages else "no_relevant_source",
        "passages": [asdict(item) for item in passages],
    }
