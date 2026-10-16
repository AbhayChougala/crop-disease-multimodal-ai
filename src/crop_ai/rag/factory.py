from __future__ import annotations

from pathlib import Path

from crop_ai.rag.base import Retriever
from crop_ai.rag.embeddings import SentenceTransformerEmbeddingProvider
from crop_ai.rag.hybrid import HybridRetriever
from crop_ai.rag.semantic import FaissStore
from crop_ai.rag.store import TfidfStore


def load_retriever(
    mode: str,
    *,
    lexical_index: Path,
    semantic_index: Path,
    semantic_metadata: Path,
    embedding_model: str,
    lexical_weight: float = 0.5,
    semantic_weight: float = 0.5,
) -> Retriever:
    normalized = mode.casefold()
    if normalized == "tfidf":
        return TfidfStore.load(lexical_index)
    if normalized not in {"semantic", "hybrid"}:
        raise ValueError(f"Unsupported retriever mode: {mode}")
    embedder = SentenceTransformerEmbeddingProvider(embedding_model, allow_download=False)
    semantic = FaissStore.load(semantic_index, semantic_metadata, embedder)
    if normalized == "semantic":
        return semantic
    lexical = TfidfStore.load(lexical_index)
    return HybridRetriever(lexical, semantic, lexical_weight, semantic_weight)
