from __future__ import annotations

from functools import lru_cache
from typing import Any

from backend.app.core.config import PROJECT_ROOT, get_settings
from crop_ai.rag.contracts import RetrievedPassage, RetrieverFilters


class KnowledgeService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = None
        self.error: str | None = None
        if self.settings.rag_enabled:
            try:
                from crop_ai.rag.factory import load_retriever

                self.store = load_retriever(
                    self.settings.rag_retriever,
                    lexical_index=PROJECT_ROOT / self.settings.rag_index_path,
                    semantic_index=PROJECT_ROOT / self.settings.rag_semantic_index_path,
                    semantic_metadata=PROJECT_ROOT / self.settings.rag_semantic_metadata_path,
                    embedding_model=self.settings.embedding_model,
                    lexical_weight=self.settings.hybrid_lexical_weight,
                    semantic_weight=self.settings.hybrid_semantic_weight,
                )
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self.store is not None

    def search(
        self,
        query: str,
        *,
        crop: str | None = None,
        disease: str | None = None,
        organization: str | None = None,
        source_type: str | None = None,
        document_id: str | None = None,
        top_k: int | None = None,
        strict_diagnosis_filters: bool | None = None,
    ) -> list[RetrievedPassage]:
        if not self.settings.rag_enabled:
            return []
        if self.store is None:
            raise RuntimeError(f"Knowledge index is unavailable: {self.error or 'not loaded'}")
        use_strict = (
            self.settings.rag_strict_diagnosis_filters
            if strict_diagnosis_filters is None
            else strict_diagnosis_filters
        )
        filters = RetrieverFilters(
            crop=crop if use_strict else None,
            disease=disease if use_strict else None,
            organization=organization,
            source_type=source_type,
            document_id=document_id,
        )
        hinted_query = " ".join(value for value in (crop, disease, query) if value)
        results = self.store.search(hinted_query, top_k or self.settings.rag_top_k, filters)
        return [item for item in results if item.score >= self.settings.rag_min_score]

    def status(self) -> dict[str, Any]:
        chunks = list(getattr(self.store, "chunks", [])) if self.store is not None else []
        metadata = dict(getattr(self.store, "metadata", {})) if self.store is not None else {}
        sources = {}
        for chunk in chunks:
            sources[chunk.document_id] = {
                "document_id": chunk.document_id,
                "title": chunk.title,
                "organization": chunk.organization,
                "crop": chunk.crop,
                "disease": chunk.disease,
                "source_type": chunk.source_type,
                "status": "indexed",
            }
        return {
            "enabled": self.settings.rag_enabled,
            "available": self.available,
            "retriever": self.settings.rag_retriever,
            "embedding_model": (
                self.settings.embedding_model
                if self.settings.rag_retriever in {"semantic", "hybrid"}
                else None
            ),
            "indexed_documents": len(sources),
            "indexed_chunks": len(chunks),
            "corpus_hash": metadata.get("corpus_hash"),
            "last_indexed_at": metadata.get("created_at"),
            "error": self.error,
            "sources": list(sources.values()),
        }


@lru_cache
def get_knowledge_service() -> KnowledgeService:
    return KnowledgeService()
