from __future__ import annotations

from typing import Any, Protocol


class LLMProvider(Protocol):
    async def generate(
        self,
        evidence: dict[str, Any],
        question: str | None = None,
    ) -> dict[str, Any]: ...
