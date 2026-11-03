from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from crop_ai.rag.contracts import RetrievedPassage


class ProviderUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderCapabilities:
    provider: str
    model: str
    streaming: bool
    structured_output: bool
    context_limit: int | None = None


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    model: str
    state: str
    detail: str | None = None


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMProvider(Protocol):
    async def generate(
        self,
        evidence: dict[str, Any],
        question: str | None = None,
        passages: list[RetrievedPassage] | None = None,
        history: list[dict[str, str]] | None = None,
        preferred_language: str = "en",
    ) -> dict[str, Any]: ...

    async def stream(
        self,
        evidence: dict[str, Any],
        question: str,
        passages: list[RetrievedPassage] | None = None,
        history: list[dict[str, str]] | None = None,
        preferred_language: str = "en",
    ) -> AsyncIterator[str]: ...

    async def complete_structured(
        self, system: str, user: str, schema: type[SchemaT], model: str | None = None
    ) -> SchemaT:
        """One call returning a validated instance of `schema` (used by the agent team)."""
        ...

    async def health(self) -> ProviderStatus: ...

    def capabilities(self) -> ProviderCapabilities: ...
