from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any


@dataclass
class DiagnosisSession:
    session_id: str
    evidence: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    messages: deque[dict[str, str]] = field(default_factory=lambda: deque(maxlen=20))


class SessionStore:
    def __init__(self, maximum_sessions: int = 250) -> None:
        self.maximum_sessions = maximum_sessions
        self._sessions: dict[str, DiagnosisSession] = {}
        self._lock = Lock()

    def put(self, session_id: str, evidence: dict[str, Any]) -> None:
        with self._lock:
            if len(self._sessions) >= self.maximum_sessions:
                oldest = next(iter(self._sessions))
                del self._sessions[oldest]
            self._sessions[session_id] = DiagnosisSession(session_id, evidence)

    def get(self, session_id: str) -> DiagnosisSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def append(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.messages.append({"role": role, "content": content})


session_store = SessionStore()
