from __future__ import annotations

from dataclasses import dataclass


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
