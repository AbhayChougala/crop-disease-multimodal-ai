from __future__ import annotations

import csv
from pathlib import Path
from urllib.parse import urlparse

from crop_ai.rag.contracts import KnowledgeSource

REQUIRED_COLUMNS = {
    "document_id",
    "title",
    "organization",
    "url",
    "publication_date",
    "crop",
    "disease",
    "source_type",
    "license_or_usage_note",
    "local_file_path",
    "version_hash",
}


def read_source_manifest(path: Path) -> list[KnowledgeSource]:
    if not path.is_file():
        raise FileNotFoundError(f"Knowledge manifest does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Knowledge manifest is missing columns: {sorted(missing)}")
        sources = [
            KnowledgeSource(
                **{field: row[field].strip() for field in REQUIRED_COLUMNS},
                author=(row.get("author") or "").strip(),
            )
            for row in reader
        ]
    identifiers = [source.document_id for source in sources]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Knowledge document IDs must be unique.")
    for source in sources:
        if not source.document_id or not source.title or not source.local_file_path:
            raise ValueError(
                "Every knowledge source requires document_id, title, and local_file_path."
            )
        if not source.license_or_usage_note:
            raise ValueError(f"Knowledge source {source.document_id} lacks usage/license notes.")
        if source.url and urlparse(source.url).scheme not in {"http", "https"}:
            raise ValueError(f"Knowledge source {source.document_id} URL must use HTTP or HTTPS.")
    return sources
