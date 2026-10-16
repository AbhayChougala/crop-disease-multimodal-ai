from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from crop_ai.rag.embeddings import SentenceTransformerEmbeddingProvider
from crop_ai.rag.semantic import FaissStore
from crop_ai.rag.store import TfidfStore, load_chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Build lexical and/or semantic local RAG indexes.")
    parser.add_argument("--mode", choices=["tfidf", "semantic", "all"], default="tfidf")
    parser.add_argument("--chunks", type=Path, default=Path("data/vectorstore/chunks.json"))
    parser.add_argument("--manifest", type=Path, default=Path("knowledge/source_manifest.csv"))
    parser.add_argument("--tfidf-index", type=Path, default=Path("data/vectorstore/tfidf.joblib"))
    parser.add_argument(
        "--semantic-index", type=Path, default=Path("data/vectorstore/semantic.faiss")
    )
    parser.add_argument(
        "--semantic-metadata",
        type=Path,
        default=Path("data/vectorstore/semantic.metadata.json"),
    )
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--allow-model-download", action="store_true")
    arguments = parser.parse_args()
    chunks = load_chunks(arguments.chunks)
    if arguments.mode in {"tfidf", "all"}:
        TfidfStore.build(chunks).save(arguments.tfidf_index)
        print(f"Wrote TF-IDF index to {arguments.tfidf_index}")
    if arguments.mode in {"semantic", "all"}:
        manifest_hash = (
            hashlib.sha256(arguments.manifest.read_bytes()).hexdigest()
            if arguments.manifest.is_file()
            else ""
        )
        embedder = SentenceTransformerEmbeddingProvider(
            arguments.embedding_model, allow_download=arguments.allow_model_download
        )
        store = FaissStore.build(chunks, embedder, manifest_hash)
        store.save(arguments.semantic_index, arguments.semantic_metadata)
        print(f"Wrote semantic index to {arguments.semantic_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
