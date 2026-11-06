from __future__ import annotations

import hashlib
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from app.core.config import PROJECT_ROOT, get_settings


def _now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class StoredImage:
    diagnosis_id: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    crop: str | None
    disease: str | None
    confidence: float | None
    is_mock: bool
    created_at: str
    project_id: str | None = None
    note: str = ""


@dataclass
class Project:
    project_id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    image_count: int = 0


_COLUMNS = (
    "diagnosis_id, filename, content_type, size_bytes, sha256, crop, disease, confidence, "
    "is_mock, created_at, project_id, note"
)
_UNSET = object()


class ImageStore:
    """Uploaded leaf photos and projects, kept in the local SQLite database next to the chat tables.

    One row per analysed image (keyed by its diagnosis/session id) holding the original upload
    bytes, the prediction made for it, an optional free-text label ("Plant 3, row 2") and an
    optional project. The full diagnosis lives in the diagnosis archive under the same id, so a
    photo that was already analysed can be shown again without re-running the model.
    The database lives under artifacts/ and is git-ignored.
    """

    def __init__(self, database_path: Path | None = None) -> None:
        configured = database_path or get_settings().chat_database_path
        self.database_path = configured if configured.is_absolute() else PROJECT_ROOT / configured
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS diagnosis_images (
                    diagnosis_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    crop TEXT,
                    disease TEXT,
                    confidence REAL,
                    is_mock INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    image BLOB NOT NULL
                );
                """
            )
            info = connection.execute("PRAGMA table_info(diagnosis_images)")
            columns = {row["name"] for row in info}
            # Databases created before projects existed are migrated in place.
            if "project_id" not in columns:
                connection.execute("ALTER TABLE diagnosis_images ADD COLUMN project_id TEXT")
            if "note" not in columns:
                connection.execute(
                    "ALTER TABLE diagnosis_images ADD COLUMN note TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_images_sha ON diagnosis_images(sha256)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_images_project ON diagnosis_images(project_id)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    # ---- images -------------------------------------------------------------------------

    def save(
        self,
        diagnosis_id: str,
        data: bytes,
        *,
        filename: str,
        content_type: str,
        crop: str | None,
        disease: str | None,
        confidence: float | None,
        is_mock: bool,
        project_id: str | None = None,
        note: str = "",
    ) -> StoredImage:
        record = StoredImage(
            diagnosis_id=diagnosis_id,
            filename=Path(filename or "upload").name[:200],
            content_type=content_type,
            size_bytes=len(data),
            sha256=sha256_of(data),
            crop=crop,
            disease=disease,
            confidence=confidence,
            is_mock=is_mock,
            created_at=_now(),
            project_id=project_id,
            note=note.strip()[:300],
        )
        with self._lock, self._connect() as connection:
            # Re-analysing a saved photo keeps one row per image: the newest diagnosis wins, but
            # the original filename, label and project are kept unless new ones are given.
            previous = connection.execute(
                "SELECT filename, project_id, note FROM diagnosis_images WHERE sha256 = ? "
                "ORDER BY created_at LIMIT 1",
                (record.sha256,),
            ).fetchone()
            if previous is not None:
                record.filename = previous["filename"]
                record.project_id = record.project_id or previous["project_id"]
                record.note = record.note or previous["note"]
                connection.execute(
                    "DELETE FROM diagnosis_images WHERE sha256 = ?", (record.sha256,)
                )
            connection.execute(
                f"INSERT OR REPLACE INTO diagnosis_images ({_COLUMNS}, image) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*_values(record), sqlite3.Binary(data)),
            )
        return record

    def find_by_sha256(self, digest: str) -> StoredImage | None:
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {_COLUMNS} FROM diagnosis_images WHERE sha256 = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (digest,),
            ).fetchone()
        return _record(row) if row else None

    def update(
        self, diagnosis_id: str, *, project_id: object = _UNSET, note: object = _UNSET
    ) -> StoredImage | None:
        """Change an image's project (None removes it from its project) and/or its label."""
        with self._lock, self._connect() as connection:
            if project_id is not _UNSET:
                connection.execute(
                    "UPDATE diagnosis_images SET project_id = ? WHERE diagnosis_id = ?",
                    (project_id, diagnosis_id),
                )
            if note is not _UNSET:
                connection.execute(
                    "UPDATE diagnosis_images SET note = ? WHERE diagnosis_id = ?",
                    (str(note or "").strip()[:300], diagnosis_id),
                )
        return self.record(diagnosis_id)

    def list(self, limit: int = 100, project_id: str | None = None) -> list[StoredImage]:
        where, params = ("WHERE project_id = ?", [project_id]) if project_id else ("", [])
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {_COLUMNS} FROM diagnosis_images {where} "
                "ORDER BY created_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [_record(row) for row in rows]

    def count(self) -> int:
        with self._connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM diagnosis_images").fetchone()[0]

    def record(self, diagnosis_id: str) -> StoredImage | None:
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {_COLUMNS} FROM diagnosis_images WHERE diagnosis_id = ?",
                (diagnosis_id,),
            ).fetchone()
        return _record(row) if row else None

    def get(self, diagnosis_id: str) -> tuple[StoredImage, bytes] | None:
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {_COLUMNS}, image FROM diagnosis_images WHERE diagnosis_id = ?",
                (diagnosis_id,),
            ).fetchone()
        return (_record(row), bytes(row["image"])) if row else None

    def delete(self, diagnosis_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM diagnosis_images WHERE diagnosis_id = ?", (diagnosis_id,)
            )
        return cursor.rowcount > 0

    # ---- projects -----------------------------------------------------------------------

    def create_project(self, name: str, description: str = "") -> Project:
        timestamp = _now()
        project = Project(str(uuid.uuid4()), name.strip()[:120], description.strip()[:1000],
                          timestamp, timestamp)
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO projects (project_id, name, description, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (project.project_id, project.name, project.description, timestamp, timestamp),
            )
        return project

    def projects(self) -> list[Project]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT p.*, COUNT(i.diagnosis_id) AS image_count FROM projects p
                LEFT JOIN diagnosis_images i ON i.project_id = p.project_id
                GROUP BY p.project_id ORDER BY p.updated_at DESC"""
            ).fetchall()
        return [_project(row) for row in rows]

    def project(self, project_id: str) -> Project | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT p.*, COUNT(i.diagnosis_id) AS image_count FROM projects p
                LEFT JOIN diagnosis_images i ON i.project_id = p.project_id
                WHERE p.project_id = ? GROUP BY p.project_id""",
                (project_id,),
            ).fetchone()
        return _project(row) if row else None

    def update_project(
        self, project_id: str, *, name: str | None = None, description: str | None = None
    ) -> Project | None:
        with self._lock, self._connect() as connection:
            if name is not None:
                connection.execute(
                    "UPDATE projects SET name = ? WHERE project_id = ?",
                    (name.strip()[:120], project_id),
                )
            if description is not None:
                connection.execute(
                    "UPDATE projects SET description = ? WHERE project_id = ?",
                    (description.strip()[:1000], project_id),
                )
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE project_id = ?", (_now(), project_id)
            )
        return self.project(project_id)

    def touch_project(self, project_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE project_id = ?", (_now(), project_id)
            )

    def delete_project(self, project_id: str) -> bool:
        """Delete a project. Its photos and diagnoses are kept, just no longer grouped."""
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE diagnosis_images SET project_id = NULL WHERE project_id = ?", (project_id,)
            )
            cursor = connection.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
        return cursor.rowcount > 0


def _values(record: StoredImage) -> tuple:
    return (
        record.diagnosis_id,
        record.filename,
        record.content_type,
        record.size_bytes,
        record.sha256,
        record.crop,
        record.disease,
        record.confidence,
        int(record.is_mock),
        record.created_at,
        record.project_id,
        record.note,
    )


def _record(row: sqlite3.Row) -> StoredImage:
    return StoredImage(
        diagnosis_id=row["diagnosis_id"],
        filename=row["filename"],
        content_type=row["content_type"],
        size_bytes=row["size_bytes"],
        sha256=row["sha256"],
        crop=row["crop"],
        disease=row["disease"],
        confidence=row["confidence"],
        is_mock=bool(row["is_mock"]),
        created_at=row["created_at"],
        project_id=row["project_id"],
        note=row["note"] or "",
    )


def _project(row: sqlite3.Row) -> Project:
    return Project(
        project_id=row["project_id"],
        name=row["name"],
        description=row["description"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        image_count=row["image_count"],
    )


image_store = ImageStore()
