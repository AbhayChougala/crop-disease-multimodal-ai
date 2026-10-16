from __future__ import annotations

from dataclasses import replace

from crop_ai.rag.base import Retriever
from crop_ai.rag.contracts import RetrievedPassage, RetrieverFilters


class HybridRetriever:
    mode = "hybrid"

    def __init__(
        self,
        lexical: Retriever,
        semantic: Retriever,
        lexical_weight: float = 0.5,
        semantic_weight: float = 0.5,
    ) -> None:
        if lexical_weight < 0 or semantic_weight < 0 or lexical_weight + semantic_weight <= 0:
            raise ValueError("Hybrid weights must be non-negative with a positive total.")
        total = lexical_weight + semantic_weight
        self.lexical = lexical
        self.semantic = semantic
        self.lexical_weight = lexical_weight / total
        self.semantic_weight = semantic_weight / total

    @staticmethod
    def _normalized(results: list[RetrievedPassage]) -> dict[str, float]:
        if not results:
            return {}
        values = [item.score for item in results]
        low, high = min(values), max(values)
        if high - low < 1e-12:
            return {item.chunk_id: 1.0 for item in results}
        return {item.chunk_id: (item.score - low) / (high - low) for item in results}

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrieverFilters | None = None,
    ) -> list[RetrievedPassage]:
        size = max(top_k * 3, top_k)
        lexical = self.lexical.search(query, size, filters)
        semantic = self.semantic.search(query, size, filters)
        lexical_scores = self._normalized(lexical)
        semantic_scores = self._normalized(semantic)
        passages = {item.chunk_id: item for item in [*lexical, *semantic]}
        ranked = sorted(
            passages.values(),
            key=lambda item: (
                self.lexical_weight * lexical_scores.get(item.chunk_id, 0)
                + self.semantic_weight * semantic_scores.get(item.chunk_id, 0)
            ),
            reverse=True,
        )
        return [
            replace(
                item,
                score=round(
                    self.lexical_weight * lexical_scores.get(item.chunk_id, 0)
                    + self.semantic_weight * semantic_scores.get(item.chunk_id, 0),
                    6,
                ),
                retriever="hybrid",
            )
            for item in ranked[:top_k]
        ]
