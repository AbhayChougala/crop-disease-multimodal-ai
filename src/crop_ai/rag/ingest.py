from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from html.parser import HTMLParser
from pathlib import Path

from crop_ai.rag.chunking import chunk_text
from crop_ai.rag.contracts import KnowledgeChunk, KnowledgeSource
from crop_ai.rag.manifest import read_source_manifest


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _read_pages(path: Path) -> list[tuple[int | None, str]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return [(None, path.read_text(encoding="utf-8"))]
    if suffix in {".html", ".htm"}:
        parser = _HTMLTextExtractor()
        parser.feed(path.read_text(encoding="utf-8"))
        return [(None, " ".join(parser.parts))]
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF ingestion requires pypdf: pip install pypdf") from exc
        return [
            (index, page.extract_text() or "")
            for index, page in enumerate(PdfReader(path).pages, 1)
        ]
    raise ValueError(f"Unsupported knowledge document type: {suffix}")


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ingest_source(
    source: KnowledgeSource, source_root: Path, chunk_size: int, overlap: int
) -> list[KnowledgeChunk]:
    path = (source_root / source.local_file_path).resolve()
    if not path.is_relative_to(source_root.resolve()):
        raise ValueError(f"Knowledge path escapes source root: {source.local_file_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Knowledge document does not exist: {path}")
    actual_hash = source_hash(path)
    if source.version_hash and source.version_hash != actual_hash:
        raise ValueError(f"Version hash mismatch for {source.document_id}.")
    chunks: list[KnowledgeChunk] = []
    sequence = 0
    for page, text in _read_pages(path):
        for value in chunk_text(text, chunk_size, overlap):
            sequence += 1
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{source.document_id}:{sequence:04d}",
                    document_id=source.document_id,
                    title=source.title,
                    organization=source.organization,
                    crop=source.crop,
                    disease=source.disease,
                    page=page,
                    text=value,
                    url=source.url,
                    publication_date=source.publication_date,
                    source_type=source.source_type,
                    license_or_usage_note=source.license_or_usage_note,
                    version_hash=source.version_hash or actual_hash,
                )
            )
    return chunks


def build_chunks(
    manifest: Path, source_root: Path, output: Path, chunk_size: int = 900, overlap: int = 120
) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    for source in read_source_manifest(manifest):
        chunks.extend(ingest_source(source, source_root, chunk_size, overlap))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([asdict(item) for item in chunks], indent=2), encoding="utf-8")
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest approved local knowledge documents.")
    parser.add_argument("--manifest", type=Path, default=Path("knowledge/source_manifest.csv"))
    parser.add_argument("--source-root", type=Path, default=Path("knowledge/source_docs"))
    parser.add_argument("--output", type=Path, default=Path("data/vectorstore/chunks.json"))
    arguments = parser.parse_args()
    try:
        chunks = build_chunks(arguments.manifest, arguments.source_root, arguments.output)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {len(chunks)} source-preserving chunks to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
