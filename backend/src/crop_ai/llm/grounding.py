from __future__ import annotations

from crop_ai.llm.schemas import GroundedAdvisory
from crop_ai.rag.contracts import RetrievedPassage


def validate_citations(
    advisory: GroundedAdvisory,
    passages: list[RetrievedPassage],
) -> GroundedAdvisory:
    allowed = {item.citation_id for item in passages}
    unknown = set(advisory.source_ids) - allowed
    if unknown:
        raise ValueError(
            f"Provider returned citation IDs that were not retrieved: {sorted(unknown)}"
        )
    if not passages and advisory.source_ids:
        raise ValueError("Provider cited sources even though retrieval returned no passages.")
    return advisory
