# Ollama provider setup

Install and start Ollama separately, then choose a locally installed model:

```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=<local-model-name>
OLLAMA_BASE_URL=http://localhost:11434
RAG_ENABLED=true
```

CropAI calls `/api/generate` with a JSON schema and checks `/api/tags` for status. The backend still
starts if Ollama is stopped; `/api/v1/llm/status` reports `UNAVAILABLE`. Model installation is an
explicit user action and is never initiated by this application. Empty retrieval causes abstention
before the model is called.
