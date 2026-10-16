from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from crop_ai.rag.contracts import (
    KnowledgeChunk,
    RetrievedPassage,
    RetrieverFilters,
    chunk_matches,
    citation_id_for,
)


class TfidfStore:
    mode = "tfidf"

    def __init__(self, chunks: list[KnowledgeChunk], vectorizer, matrix, metadata=None) -> None:
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.matrix = matrix
        if metadata is None:
            from crop_ai.rag.semantic import corpus_hash

            metadata = {
                "corpus_hash": corpus_hash(chunks),
                "created_at": datetime.now(UTC).isoformat(),
                "document_count": len({chunk.document_id for chunk in chunks}),
                "chunk_count": len(chunks),
            }
        self.metadata = metadata

    @classmethod
    def build(cls, chunks: list[KnowledgeChunk]) -> TfidfStore:
        if not chunks:
            raise ValueError("Cannot build a retrieval index without chunks.")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise RuntimeError("TF-IDF retrieval requires scikit-learn ML extras.") from exc
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
        matrix = vectorizer.fit_transform([item.text for item in chunks])
        return cls(chunks, vectorizer, matrix)

    def save(self, path: Path) -> None:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("Index persistence requires joblib from scikit-learn.") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "format_version": 1,
                "chunks": [asdict(item) for item in self.chunks],
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> TfidfStore:
        if not path.is_file():
            raise FileNotFoundError(f"Retrieval index does not exist: {path}")
        import joblib

        payload = joblib.load(path)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported retrieval-index format.")
        return cls(
            [KnowledgeChunk(**item) for item in payload["chunks"]],
            payload["vectorizer"],
            payload["matrix"],
            payload.get("metadata"),
        )

    def search(
        self,
        query: str,
        top_k: int = 4,
        filters: RetrieverFilters | None = None,
        crop: str | None = None,
        disease: str | None = None,
    ) -> list[RetrievedPassage]:
        from sklearn.metrics.pairwise import cosine_similarity

        if not query.strip():
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        candidates = []
        for index, score in enumerate(scores):
            chunk = self.chunks[index]
            selected_filters = filters or RetrieverFilters(crop=crop, disease=disease)
            if not chunk_matches(chunk, selected_filters):
                continue
            if score > 0:
                candidates.append((float(score), chunk))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedPassage(
                citation_id=citation_id_for(chunk),
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                title=chunk.title,
                organization=chunk.organization,
                page=chunk.page,
                score=round(score, 6),
                text=chunk.text,
                url=chunk.url,
                section_name=chunk.section_name,
                retriever="tfidf",
            )
            for score, chunk in candidates[:top_k]
        ]


def load_chunks(path: Path) -> list[KnowledgeChunk]:
    if not path.is_file():
        raise FileNotFoundError(f"Chunk file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [KnowledgeChunk(**item) for item in payload]
