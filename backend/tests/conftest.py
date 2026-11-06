"""Isolate tests from the developer's backend/.env.

Environment variables take precedence over the .env file in pydantic-settings, so these force the
offline demo configuration: tests never call a paid LLM API or depend on local indexes.
"""

import os
import tempfile

os.environ.update(
    {
        "MODEL_MODE": "mock",
        "LLM_PROVIDER": "mock",
        "LLM_API_KEY": "",
        "OPENAI_API_KEY": "",
        "RAG_ENABLED": "false",
        "APP_ENV": "development",
        # Chats and uploaded photos go to a throwaway database, never the developer's own.
        "CHAT_DATABASE_PATH": os.path.join(tempfile.mkdtemp(prefix="crop-ai-"), "chat.sqlite3"),
    }
)
