from __future__ import annotations

from typing import Any

from crop_ai.llm.base import ProviderCapabilities, ProviderStatus, ProviderUnavailable
from crop_ai.llm.formatting import advisory_payload
from crop_ai.llm.grounding import validate_citations
from crop_ai.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from crop_ai.llm.schemas import GroundedAdvisory
from crop_ai.rag.contracts import RetrievedPassage


class OpenAIProvider:
    """Official async SDK provider using the Responses API and Pydantic structured output."""

    def __init__(
        self, api_key: str, model: str, timeout: float = 30.0, max_retries: int = 2
    ) -> None:
        if not api_key or not model:
            raise ProviderUnavailable("OpenAI provider requires an API key and configured model.")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ProviderUnavailable("Install the 'llm' project extra to use OpenAI.") from exc
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model

    async def generate(
        self,
        evidence: dict[str, Any],
        question: str | None = None,
        passages: list[RetrievedPassage] | None = None,
        history: list[dict[str, str]] | None = None,
        preferred_language: str = "en",
    ) -> dict[str, Any]:
        retrieved = passages or []
        if not retrieved:
            raise ProviderUnavailable(
                "Grounded generation is unavailable because retrieval was empty."
            )
        try:
            response = await self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_prompt(
                            evidence, retrieved, question, history, preferred_language
                        ),
                    },
                ],
                text_format=GroundedAdvisory,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ProviderUnavailable("The model returned no validated structured output.")
            advisory = validate_citations(parsed, retrieved)
            return {
                **advisory_payload(advisory),
                "usage": response.usage.model_dump() if response.usage else None,
            }
        except ProviderUnavailable:
            raise
        except Exception as exc:
            raise ProviderUnavailable(f"OpenAI generation failed: {type(exc).__name__}") from exc

    async def stream(
        self, evidence, question, passages=None, history=None, preferred_language="en"
    ):
        response = await self.generate(evidence, question, passages, history, preferred_language)
        yield response["answer"]

    async def health(self) -> ProviderStatus:
        return ProviderStatus(
            "openai",
            self.model,
            "CONFIGURED",
            "Credentials are configured; no billable health request was made.",
        )

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities("openai", self.model, False, True, None)
