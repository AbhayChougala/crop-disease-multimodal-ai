# OpenAI provider setup

Install the LLM extra and place credentials only in the ignored `.env` file:

```bash
source .venv/bin/activate
python -m pip install -e '.[llm]'
```

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=<your-secret>
OPENAI_MODEL=<model-you-have-selected>
RAG_ENABLED=true
```

The provider uses the official asynchronous Python SDK, Responses API, and Pydantic structured
output parsing. No model name is hardcoded. A status check does not make a billable generation call.
Test a real answer only after building an index from approved sources. Provider failures return an
unavailable response without exposing the key.
