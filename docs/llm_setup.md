# LLM setup

The default `LLM_PROVIDER=mock` is deterministic, offline, and visibly returns `DEMO / MOCK RESPONSE`.
It makes no agricultural reliability claim. Use it for CI and demonstrations without credentials.

```dotenv
LLM_PROVIDER=mock
LLM_MODEL=
LLM_TIMEOUT=30
```

Provider status is available at `GET /api/v1/llm/status`. `CONFIGURED` means credentials/settings are
present, not that a paid request was performed. `UNAVAILABLE` is returned safely for missing settings
or an unreachable local provider. Agricultural generation from OpenAI/Ollama requires RAG to be
enabled and a non-empty retrieval result.

Answers are validated against Pydantic schemas where supported. Provider exceptions are reduced to
safe type-level messages; credentials and raw environment values are never returned or logged.
