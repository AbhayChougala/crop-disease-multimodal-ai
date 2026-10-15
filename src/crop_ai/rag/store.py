from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from crop_ai.rag.contracts import KnowledgeChunk, RetrievedPassage


class TfidfStore:
    def __init__(self, chunks: list[KnowledgeChunk], vectorizer, matrix) -> None:
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.matrix = matrix

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
        )

    def search(
        self,
        query: str,
        top_k: int = 4,
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
            if crop and chunk.crop and chunk.crop.lower() != crop.lower():
                continue
            if disease and chunk.disease and chunk.disease.lower() != disease.lower():
                continue
            if score > 0:
                candidates.append((float(score), chunk))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedPassage(
                citation_id=f"S{rank}",
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                title=chunk.title,
                organization=chunk.organization,
                page=chunk.page,
                score=round(score, 6),
                text=chunk.text,
                url=chunk.url,
            )
            for rank, (score, chunk) in enumerate(candidates[:top_k], start=1)
        ]


def load_chunks(path: Path) -> list[KnowledgeChunk]:
    if not path.is_file():
        raise FileNotFoundError(f"Chunk file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [KnowledgeChunk(**item) for item in payload]
