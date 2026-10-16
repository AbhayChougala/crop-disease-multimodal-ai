from __future__ import annotations

from functools import lru_cache

from backend.app.core.config import get_settings
from crop_ai.llm.factory import create_provider


@lru_cache
def get_llm_provider():
    settings = get_settings()
    model = settings.llm_model or settings.openai_model or settings.local_llm_model
    api_key = settings.llm_api_key or settings.openai_api_key
    base_url = (
        settings.llm_base_url
        or settings.local_llm_base_url
        or (settings.ollama_base_url if settings.llm_provider in {"ollama", "local"} else "")
    )
    return create_provider(
        settings.llm_provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=settings.llm_timeout,
    )
