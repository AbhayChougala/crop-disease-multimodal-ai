from __future__ import annotations

from typing import Protocol

from crop_ai.rag.contracts import RetrievedPassage, RetrieverFilters, RetrieverMode


class Retriever(Protocol):
    mode: RetrieverMode

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrieverFilters | None = None,
    ) -> list[RetrievedPassage]: ...
