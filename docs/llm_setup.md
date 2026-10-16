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

## Gemini (Google)

Uses the official `google-genai` SDK (installed by the `llm` extra) with Pydantic structured output.
In `backend/.env`:

```dotenv
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.8-flash
LLM_API_KEY=<your key>
GEMINI_BACKEND=developer   # developer = AI Studio key, vertex = Vertex AI express key, auto = try both
# Tried in order when the main model is retired (404), overloaded (503) or rate limited (429)
LLM_FALLBACK_MODELS=gemini-3.7-flash,gemini-3.5-flash-lite,gemini-flash-latest
RAG_ENABLED=true
RAG_RETRIEVER=hybrid
```

Google retires model names regularly (`gemini-2.5-flash` is no longer offered to new keys). If a
model returns 404, pick one listed for your key in Google AI Studio. Each transient failure is
retried once before moving to the next fallback model.

Verify with one real request: `python scripts/check_llm.py` from `backend/`. Provider errors are
logged by the API (`LLM provider failed: ...`) and shown under Pipeline metadata on the Ask page.
Answers shown in chat are rendered from the structured advisory (summary, symptoms, what to do,
prevention, when to get expert help, sources) by `crop_ai/llm/formatting.py`.

## Scope and answer style

The base prompt (`src/crop_ai/llm/prompts.py`) has ROLE, SCOPE, SAFETY and STYLE sections. Only crop
and plant health questions (diseases, pests, symptoms, prevention, management, interpreting this
app's diagnosis, photo tips) are answered. For anything else, including greetings and attempts to
change the assistant's role, the model sets `in_scope=false` in `GroundedAdvisory`;
`advisory_payload` in `src/crop_ai/llm/formatting.py` then strips all advice and citations and
returns status `out_of_scope`, so the rule is enforced in code as well as in the prompt.
