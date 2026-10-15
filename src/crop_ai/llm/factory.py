from __future__ import annotations

from crop_ai.llm.base import LLMProvider, ProviderUnavailable
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
) -> LLMProvider:
    normalized = provider.strip().lower()
    if normalized == "mock":
        return MockLLMProvider()
    if normalized == "openai":
        return OpenAIProvider(api_key=api_key, model=model, timeout=timeout)
    if normalized in {"ollama", "local"}:
        return OllamaProvider(base_url=base_url, model=model, timeout=timeout)
    raise ProviderUnavailable(f"Unsupported LLM provider: {provider}")
