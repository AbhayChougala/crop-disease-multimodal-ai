from __future__ import annotations

import json
from typing import Any

import httpx

from crop_ai.llm.base import ProviderCapabilities, ProviderStatus, ProviderUnavailable
from crop_ai.llm.formatting import advisory_payload
from crop_ai.llm.grounding import validate_citations
from crop_ai.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from crop_ai.llm.schemas import GroundedAdvisory
from crop_ai.rag.contracts import RetrievedPassage


class OllamaProvider:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0) -> None:
        if not base_url or not model:
            raise ProviderUnavailable("Ollama provider requires a base URL and model name.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

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
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{build_user_prompt(evidence, retrieved, question, history, preferred_language)}"
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "format": GroundedAdvisory.model_json_schema(),
                    },
                )
                response.raise_for_status()
            parsed = GroundedAdvisory.model_validate(json.loads(response.json()["response"]))
            advisory = validate_citations(parsed, retrieved)
            return {
                **advisory_payload(advisory),
            }
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise ProviderUnavailable(f"Ollama generation failed: {type(exc).__name__}") from exc

    async def stream(
        self, evidence, question, passages=None, history=None, preferred_language="en"
    ):
        response = await self.generate(evidence, question, passages, history, preferred_language)
        yield response["answer"]

    async def health(self) -> ProviderStatus:
        try:
            async with httpx.AsyncClient(timeout=min(self.timeout, 5.0)) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
            return ProviderStatus("ollama", self.model, "AVAILABLE")
        except httpx.HTTPError:
            return ProviderStatus("ollama", self.model, "UNAVAILABLE", "Ollama did not respond.")

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities("ollama", self.model, False, True, None)
