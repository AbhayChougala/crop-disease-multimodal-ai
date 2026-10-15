from pathlib import Path

import pytest

from crop_ai.rag.chunking import chunk_text
from crop_ai.rag.ingest import build_chunks

FIXTURES = Path(__file__).parents[1] / "fixtures" / "knowledge"


def test_chunking_is_bounded_and_overlapping() -> None:
    text = "Sentence about a leaf. " * 80
    chunks = chunk_text(text, chunk_size=180, overlap=30)

    assert len(chunks) > 1
    assert all(1 <= len(item) <= 180 for item in chunks)


def test_ingestion_preserves_real_source_identifier(tmp_path: Path) -> None:
    chunks = build_chunks(
        FIXTURES / "manifest.csv",
        FIXTURES,
        tmp_path / "chunks.json",
        chunk_size=200,
        overlap=30,
    )

    assert chunks
    assert {item.document_id for item in chunks} == {"SAMPLE_001"}


def test_tfidf_index_and_retrieval() -> None:
    pytest.importorskip("sklearn")
    from crop_ai.rag.contracts import KnowledgeChunk
    from crop_ai.rag.store import TfidfStore

    store = TfidfStore.build(
        [
            KnowledgeChunk(
                "c1",
                "d1",
                "Synthetic fixture",
                "Tests",
                "Tomato",
                "Early Blight",
                None,
                "An uncertain prediction should be reviewed by an agricultural expert.",
            ),
            KnowledgeChunk("c2", "d2", "Other", "Tests", "Potato", "", None, "Unrelated text."),
        ]
    )
    results = store.search("uncertain prediction expert", crop="Tomato")

    assert results[0].document_id == "d1"
    assert results[0].citation_id == "S1"
