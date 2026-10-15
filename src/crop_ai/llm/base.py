from __future__ import annotations

from typing import Any, Protocol

from crop_ai.rag.contracts import RetrievedPassage


class ProviderUnavailable(RuntimeError):
    pass


class LLMProvider(Protocol):
    async def generate(
        self,
        evidence: dict[str, Any],
        question: str | None = None,
        passages: list[RetrievedPassage] | None = None,
    ) -> dict[str, Any]: ...
