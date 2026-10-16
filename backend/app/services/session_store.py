from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from app.core.config import PROJECT_ROOT, get_settings

ChatMode = Literal["general", "diagnosis"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ChatSession:
    session_id: str
    mode: ChatMode
    evidence: dict[str, Any] = field(default_factory=dict)
    diagnosis_id: str | None = None
    provider: str = "mock"
    model: str = ""
    preferred_language: str = "en"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    messages: list[dict[str, Any]] = field(default_factory=list)


class SessionStore:
    """SQLite-backed chat storage containing text and structured metadata only.

    Uploaded image data and provider credentials are deliberately never accepted by
    this store. Diagnosis image artifacts live in the separate diagnosis archive.
    """

    def __init__(self, database_path: Path | None = None, history_limit: int | None = None) -> None:
        settings = get_settings()
        configured = database_path or settings.chat_database_path
        self.database_path = configured if configured.is_absolute() else PROJECT_ROOT / configured
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_limit = history_limit or settings.chat_max_history_messages
        self._lock = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL CHECK (mode IN ('general', 'diagnosis')),
                    diagnosis_id TEXT,
                    evidence_json TEXT NOT NULL DEFAULT '{}',
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    preferred_language TEXT NOT NULL DEFAULT 'en',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_time
                    ON chat_messages(session_id, created_at);
                """
            )

    def create(
        self,
        *,
        mode: ChatMode = "general",
        diagnosis_id: str | None = None,
        evidence: dict[str, Any] | None = None,
        provider: str = "mock",
        model: str = "",
        preferred_language: str = "en",
        session_id: str | None = None,
    ) -> ChatSession:
        identifier = session_id or str(uuid.uuid4())
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO chat_sessions
                (session_id, mode, diagnosis_id, evidence_json, provider, model,
                 preferred_language, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    identifier,
                    mode,
                    diagnosis_id,
                    json.dumps(evidence or {}, separators=(",", ":")),
                    provider,
                    model,
                    preferred_language,
                    timestamp,
                    timestamp,
                ),
            )
        session = self.get(identifier)
        assert session is not None
        return session

    def put(self, session_id: str, evidence: dict[str, Any]) -> None:
        """Compatibility adapter used by the diagnosis pipeline."""
        settings = get_settings()
        model = settings.llm_model or settings.openai_model or settings.local_llm_model
        self.create(
            session_id=session_id,
            mode="diagnosis",
            diagnosis_id=session_id,
            evidence=evidence,
            provider=settings.llm_provider,
            model=model,
        )

    def get(self, session_id: str) -> ChatSession | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            messages = connection.execute(
                """SELECT message_id, role, content, citations_json, metadata_json, created_at
                FROM chat_messages WHERE session_id = ?
                ORDER BY created_at DESC LIMIT ?""",
                (session_id, self.history_limit),
            ).fetchall()
        return ChatSession(
            session_id=row["session_id"],
            mode=row["mode"],
            diagnosis_id=row["diagnosis_id"],
            evidence=json.loads(row["evidence_json"]),
            provider=row["provider"],
            model=row["model"],
            preferred_language=row["preferred_language"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            messages=[
                {
                    "message_id": item["message_id"],
                    "role": item["role"],
                    "content": item["content"],
                    "citations": json.loads(item["citations_json"]),
                    "metadata": json.loads(item["metadata_json"]),
                    "created_at": item["created_at"],
                }
                for item in reversed(messages)
            ],
        )

    def append(
        self,
        session_id: str,
        role: Literal["user", "assistant"],
        content: str,
        *,
        citations: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        message_id = str(uuid.uuid4())
        timestamp = _now()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO chat_messages
                (message_id, session_id, role, content, citations_json, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    message_id,
                    session_id,
                    role,
                    content,
                    json.dumps(citations or [], separators=(",", ":")),
                    json.dumps(metadata or {}, separators=(",", ":")),
                    timestamp,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(session_id)
            connection.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                (timestamp, session_id),
            )
        return {
            "message_id": message_id,
            "role": role,
            "content": content,
            "citations": citations or [],
            "metadata": metadata or {},
            "created_at": timestamp,
        }

    def delete(self, session_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM chat_sessions WHERE session_id = ?", (session_id,)
            )
            return cursor.rowcount > 0

    def clear_expired(self, retention_days: int | None = None) -> int:
        days = retention_days or get_settings().chat_retention_days
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM chat_sessions WHERE updated_at < ?", (cutoff,))
            return cursor.rowcount


session_store = SessionStore()
