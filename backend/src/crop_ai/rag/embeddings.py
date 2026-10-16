from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    model_id: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_id: str, allow_download: bool = False) -> None:
        if not model_id:
            raise ValueError(
                "Set EMBEDDING_MODEL explicitly before using semantic retrieval; "
                "model downloads are not automatic."
            )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the 'semantic-rag' extra for dense retrieval.") from exc
        self.model_id = model_id
        try:
            self._model = SentenceTransformer(model_id, local_files_only=not allow_download)
        except OSError as exc:
            raise RuntimeError(
                "Embedding model is not available locally. Run the semantic index command with "
                "--allow-model-download after reviewing the configured model."
            ) from exc

    def encode(self, texts: list[str]) -> np.ndarray:
        values = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result = np.asarray(values, dtype=np.float32)
        if result.ndim != 2 or not np.isfinite(result).all():
            raise ValueError("Embedding provider returned an invalid matrix.")
        return result
