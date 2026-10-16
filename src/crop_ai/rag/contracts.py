from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RetrieverMode = Literal["tfidf", "semantic", "hybrid"]


@dataclass(frozen=True)
class RetrieverFilters:
    crop: str | None = None
    disease: str | None = None
    organization: str | None = None
    source_type: str | None = None
    document_id: str | None = None


@dataclass(frozen=True)
class KnowledgeSource:
    document_id: str
    title: str
    organization: str
    url: str
    publication_date: str
    crop: str
    disease: str
    source_type: str
    license_or_usage_note: str
    local_file_path: str
    version_hash: str
    author: str = ""


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    document_id: str
    title: str
    organization: str
    crop: str
    disease: str
    page: int | None
    text: str
    url: str = ""
    publication_date: str = ""
    source_type: str = ""
    license_or_usage_note: str = ""
    version_hash: str = ""
    chunk_index: int = 0
    section_name: str = ""
    author: str = ""
    ingested_at: str = ""


@dataclass(frozen=True)
class RetrievedPassage:
    citation_id: str
    chunk_id: str
    document_id: str
    title: str
    organization: str
    page: int | None
    score: float
    text: str
    url: str = ""
    section_name: str = ""
    retriever: RetrieverMode = "tfidf"


def citation_id_for(chunk: KnowledgeChunk) -> str:
    suffix = chunk.chunk_id.split(":")[-1]
    return f"{chunk.document_id}-C{int(suffix):04d}" if suffix.isdigit() else chunk.chunk_id


def chunk_matches(chunk: KnowledgeChunk, filters: RetrieverFilters | None) -> bool:
    if filters is None:
        return True
    values = {
        "crop": chunk.crop,
        "disease": chunk.disease,
        "organization": chunk.organization,
        "source_type": chunk.source_type,
        "document_id": chunk.document_id,
    }
    for field, expected in filters.__dict__.items():
        actual = values[field]
        if expected and actual.casefold() != expected.casefold():
            return False
    return True
