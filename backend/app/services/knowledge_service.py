from __future__ import annotations

from functools import lru_cache

from backend.app.core.config import get_settings
from crop_ai.rag.contracts import RetrievedPassage


class KnowledgeService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = None
        self.error: str | None = None
        if self.settings.rag_enabled:
            try:
                from crop_ai.rag.store import TfidfStore

                self.store = TfidfStore.load(self.settings.rag_index_path)
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
        top_k: int | None = None,
    ) -> list[RetrievedPassage]:
        if not self.settings.rag_enabled:
            return []
        if self.store is None:
            raise RuntimeError(f"Knowledge index is unavailable: {self.error or 'not loaded'}")
        return self.store.search(
            query,
            top_k=top_k or self.settings.rag_top_k,
            crop=crop,
            disease=disease,
        )


@lru_cache
def get_knowledge_service() -> KnowledgeService:
    return KnowledgeService()
