from pathlib import Path

from backend.app.services.session_store import SessionStore


def test_sqlite_chat_history_is_bounded_and_contains_no_evidence_blob(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "chat.sqlite3", history_limit=2)
    session = store.create(mode="general", provider="mock")
    store.append(session.session_id, "user", "one")
    store.append(session.session_id, "assistant", "two")
    store.append(session.session_id, "user", "three")

    loaded = store.get(session.session_id)

    assert loaded is not None
    assert [item["content"] for item in loaded.messages] == ["two", "three"]
    assert loaded.evidence == {}
