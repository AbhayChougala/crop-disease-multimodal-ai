from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from crop_ai.rag.contracts import (
    KnowledgeChunk,
    RetrievedPassage,
    RetrieverFilters,
    chunk_matches,
    citation_id_for,
)
from crop_ai.rag.embeddings import EmbeddingProvider


def corpus_hash(chunks: list[KnowledgeChunk]) -> str:
    encoded = json.dumps(
        [asdict(chunk) for chunk in chunks], sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class FaissStore:
    mode = "semantic"

    def __init__(self, chunks, index, embedder, metadata) -> None:
        self.chunks: list[KnowledgeChunk] = chunks
        self.index = index
        self.embedder: EmbeddingProvider = embedder
        self.metadata: dict[str, object] = metadata

    @classmethod
    def build(
        cls,
        chunks: list[KnowledgeChunk],
        embedder: EmbeddingProvider,
        manifest_hash: str = "",
    ) -> FaissStore:
        if not chunks:
            raise ValueError("Cannot build a semantic index without chunks.")
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("Install the 'semantic-rag' extra for FAISS retrieval.") from exc
        vectors = embedder.encode([chunk.text for chunk in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        metadata = {
            "format_version": 1,
            "embedding_model": embedder.model_id,
            "embedding_dimensions": int(vectors.shape[1]),
            "corpus_hash": corpus_hash(chunks),
            "manifest_hash": manifest_hash,
            "created_at": datetime.now(UTC).isoformat(),
            "document_count": len({chunk.document_id for chunk in chunks}),
            "chunk_count": len(chunks),
        }
        return cls(chunks, index, embedder, metadata)

    def save(self, index_path: Path, metadata_path: Path) -> None:
        import faiss

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        metadata_path.write_text(
            json.dumps(
                {**self.metadata, "chunks": [asdict(chunk) for chunk in self.chunks]}, indent=2
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        index_path: Path,
        metadata_path: Path,
        embedder: EmbeddingProvider,
    ) -> FaissStore:
        if not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("Semantic index or metadata file is missing.")
        import faiss

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        chunks = [KnowledgeChunk(**item) for item in payload.pop("chunks")]
        index = faiss.read_index(str(index_path))
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported semantic-index format.")
        if payload.get("embedding_model") != embedder.model_id:
            raise ValueError("Semantic index was built with a different embedding model.")
        if int(payload.get("embedding_dimensions", 0)) != index.d:
            raise ValueError("Semantic index dimension metadata is incompatible.")
        if int(payload.get("chunk_count", -1)) != index.ntotal or corpus_hash(
            chunks
        ) != payload.get("corpus_hash"):
            raise ValueError("Semantic index metadata or corpus hash is incompatible.")
        return cls(chunks, index, embedder, payload)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrieverFilters | None = None,
    ) -> list[RetrievedPassage]:
        if not query.strip():
            return []
        vector = self.embedder.encode([query])
        count = min(max(top_k * 5, top_k), len(self.chunks))
        scores, indices = self.index.search(np.asarray(vector, dtype=np.float32), count)
        results = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:
                continue
            chunk = self.chunks[int(index)]
            if not chunk_matches(chunk, filters):
                continue
            results.append(
                RetrievedPassage(
                    citation_id=citation_id_for(chunk),
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    title=chunk.title,
                    organization=chunk.organization,
                    page=chunk.page,
                    score=round(float(score), 6),
                    text=chunk.text,
                    url=chunk.url,
                    section_name=chunk.section_name,
                    retriever="semantic",
                )
            )
            if len(results) >= top_k:
                break
        return results
