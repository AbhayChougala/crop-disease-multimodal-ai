from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.schemas.system import FeedbackRequest, FeedbackResponse

router = APIRouter(tags=["feedback"])
DB_PATH = Path("artifacts/runtime/feedback.sqlite3")
VALID_VERDICTS = {"correct", "incorrect", "unsure"}


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    verdict = request.verdict.lower()
    if verdict not in VALID_VERDICTS:
        raise HTTPException(
            status_code=422,
            detail="Verdict must be correct, incorrect, or unsure.",
        )
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    feedback_id = str(uuid.uuid4())
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                verdict TEXT NOT NULL,
                actual_disease TEXT,
                comment TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            "INSERT INTO feedback "
            "(id, session_id, verdict, actual_disease, comment) VALUES (?, ?, ?, ?, ?)",
            (
                feedback_id,
                request.session_id,
                verdict,
                request.actual_disease,
                request.comment,
            ),
        )
    return FeedbackResponse(stored=True, feedback_id=feedback_id)
