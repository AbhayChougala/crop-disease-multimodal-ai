from __future__ import annotations

import re


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 120,
    minimum_chunk_length: int = 100,
) -> list[str]:
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")
    if minimum_chunk_length < 1 or minimum_chunk_length > chunk_size:
        raise ValueError("minimum_chunk_length must be between one and chunk_size.")
    cleaned = clean_text(text)
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            boundary = max(cleaned.rfind("\n", start, end), cleaned.rfind(". ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = cleaned[start:end].strip()
        if len(chunk) >= minimum_chunk_length and chunk not in chunks:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks
