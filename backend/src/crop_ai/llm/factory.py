from __future__ import annotations

from crop_ai.llm.base import LLMProvider, ProviderUnavailable
from crop_ai.llm.gemini_provider import GeminiProvider
from crop_ai.llm.mock_provider import MockLLMProvider
from crop_ai.llm.ollama_provider import OllamaProvider
from crop_ai.llm.openai_provider import OpenAIProvider


def create_provider(
    provider: str,
    *,
    model: str = "",
    api_key: str = "",
    base_url: str = "",
    timeout: float = 30.0,
    gemini_backend: str = "auto",
    fallback_models: list[str] | None = None,
) -> LLMProvider:
    normalized = provider.strip().lower()
    if normalized == "mock":
        return MockLLMProvider()
    if normalized == "openai":
        return OpenAIProvider(api_key=api_key, model=model, timeout=timeout)
    if normalized == "gemini":
        return GeminiProvider(
            api_key=api_key,
            model=model,
            timeout=timeout,
            backend=gemini_backend,  # type: ignore[arg-type]
            fallback_models=fallback_models,
        )
    if normalized in {"ollama", "local"}:
        return OllamaProvider(base_url=base_url, model=model, timeout=timeout)
    raise ProviderUnavailable(f"Unsupported LLM provider: {provider}")
