from __future__ import annotations

import asyncio
from typing import Any, Literal

from crop_ai.llm.base import ProviderCapabilities, ProviderStatus, ProviderUnavailable
from crop_ai.llm.formatting import advisory_payload
from crop_ai.llm.grounding import validate_citations
from crop_ai.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from crop_ai.llm.schemas import GroundedAdvisory
from crop_ai.rag.contracts import RetrievedPassage

GeminiBackend = Literal["auto", "developer", "vertex"]
# HTTP codes that mean "this key does not belong to this API", worth retrying on the other backend.
_AUTH_CODES = {400, 401, 403}
# Overloaded / rate-limited: retry once, then move on to the next model.
_TRANSIENT_CODES = {429, 500, 503, 504}
_RETRY_DELAY_SECONDS = 2.0


class GeminiProvider:
    """Google Gemini via the official google-genai SDK with Pydantic structured output.

    `backend` selects the Gemini Developer API (AI Studio keys), Vertex AI express mode (Google
    Cloud API keys), or `auto`, which tries the Developer API first and falls back to Vertex AI
    once if the key is rejected. When a model is retired (404), overloaded (503) or rate limited
    (429), the request moves on to `fallback_models` in order.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        backend: GeminiBackend = "auto",
        fallback_models: list[str] | None = None,
    ) -> None:
        if not api_key or not model:
            raise ProviderUnavailable("Gemini provider requires an API key and configured model.")
        try:
            from google import genai
        except ImportError as exc:
            raise ProviderUnavailable("Install the 'llm' project extra to use Gemini.") from exc
        self._genai = genai
        self.api_key = api_key
        self.model = model
        self.fallback_models = [item for item in (fallback_models or []) if item != model]
        self.timeout = timeout
        self.backend = backend
        self._clients: dict[str, Any] = {}

    def _client(self, backend: str) -> Any:
        if backend not in self._clients:
            http_options = self._genai.types.HttpOptions(timeout=int(self.timeout * 1000))
            self._clients[backend] = self._genai.Client(
                api_key=self.api_key,
                vertexai=backend == "vertex",
                http_options=http_options,
            )
        return self._clients[backend]

    def _backends(self) -> list[str]:
        return ["developer", "vertex"] if self.backend == "auto" else [self.backend]

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
        from google.genai import errors

        config = self._genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=GroundedAdvisory,
            temperature=0.2,
        )
        prompt = build_user_prompt(evidence, retrieved, question, history, preferred_language)
        response, model_used = await self._generate_with_fallback(prompt, config, errors)
        parsed = response.parsed
        if not isinstance(parsed, GroundedAdvisory):
            raise ProviderUnavailable("Gemini returned no validated structured output.")
        try:
            advisory = validate_citations(parsed, retrieved)
        except ValueError as exc:
            raise ProviderUnavailable(str(exc)) from exc
        usage = response.usage_metadata
        return {
            **advisory_payload(advisory),
            "model_used": model_used,
            "usage": usage.model_dump(exclude_none=True) if usage else None,
        }

    async def _generate_with_fallback(self, prompt: str, config: Any, errors: Any):
        failures: list[str] = []
        for model in [self.model, *self.fallback_models]:
            backends = self._backends()
            backend_index = 0
            attempt = 0
            while backend_index < len(backends):
                backend = backends[backend_index]
                try:
                    response = await self._client(backend).aio.models.generate_content(
                        model=model, contents=prompt, config=config
                    )
                except errors.APIError as exc:
                    code = exc.code
                    if code in _AUTH_CODES and backend_index + 1 < len(backends):
                        backend_index += 1  # key belongs to the other Gemini backend
                        continue
                    if code in _TRANSIENT_CODES and attempt == 0:
                        attempt += 1
                        await asyncio.sleep(_RETRY_DELAY_SECONDS)
                        continue
                    detail = (exc.message or "")[:160]
                    failures.append(f"{model}: HTTP {code} {exc.status}: {detail}")
                    if code in _TRANSIENT_CODES or code == 404:
                        break  # try the next model
                    raise ProviderUnavailable(
                        f"Gemini generation failed ({backend}): {failures[-1]}"
                    ) from exc
                except Exception as exc:
                    raise ProviderUnavailable(
                        f"Gemini generation failed: {type(exc).__name__}: {str(exc)[:300]}"
                    ) from exc
                # Remember the backend that accepted the key so later calls skip the failed one.
                self.backend = backend  # type: ignore[assignment]
                return response, model
        raise ProviderUnavailable(
            "Gemini generation failed for every model: " + " | ".join(failures)
        )

    async def stream(
        self, evidence, question, passages=None, history=None, preferred_language="en"
    ):
        response = await self.generate(evidence, question, passages, history, preferred_language)
        yield response["answer"]

    async def health(self) -> ProviderStatus:
        return ProviderStatus(
            "gemini",
            self.model,
            "CONFIGURED",
            f"Credentials are configured (backend: {self.backend}); no billable request was made.",
        )

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities("gemini", self.model, False, True, None)
