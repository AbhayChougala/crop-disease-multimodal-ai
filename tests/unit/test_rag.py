from pathlib import Path

import pytest

from crop_ai.rag.chunking import chunk_text
from crop_ai.rag.contracts import RetrievedPassage
from crop_ai.rag.evaluate import evaluate_retriever
from crop_ai.rag.hybrid import HybridRetriever
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
                "d1:0001",
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
    assert results[0].citation_id == "d1-C0001"


class _StubRetriever:
    def __init__(self, result: RetrievedPassage):
        self.result = result

    def search(self, query, top_k=5, filters=None):
        del query, top_k, filters
        return [self.result]


def test_hybrid_deduplicates_and_labels_passages() -> None:
    passage = RetrievedPassage("DOC-C0001", "DOC:0001", "DOC", "Title", "Org", 1, 0.8, "Text")
    hybrid = HybridRetriever(_StubRetriever(passage), _StubRetriever(passage), 0.5, 0.5)

    results = hybrid.search("leaf", top_k=3)

    assert len(results) == 1
    assert results[0].retriever == "hybrid"


def test_retrieval_metrics_are_explicitly_sample_only() -> None:
    passage = RetrievedPassage(
        "SAMPLE_001-C0001", "SAMPLE_001:0001", "SAMPLE_001", "Fixture", "Tests", None, 0.9, "Text"
    )
    report = evaluate_retriever(
        _StubRetriever(passage),
        [
            {
                "case_id": "Q1",
                "query": "test",
                "expected_document_ids": ["SAMPLE_001"],
                "is_sample": "true",
            }
        ],
        top_k=1,
        retriever_name="stub",
    )

    assert report["evaluation_status"] == "SAMPLE_FIXTURE_ONLY"
    assert report["hit_rate_at_k"] == 1
