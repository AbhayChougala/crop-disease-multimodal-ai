from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT

ARCHIVE_ROOT = PROJECT_ROOT / "artifacts/runtime/diagnoses"
SAFE_ID = re.compile(r"^[0-9a-fA-F-]{16,64}$")


def _path_for(diagnosis_id: str) -> Path:
    if not SAFE_ID.fullmatch(diagnosis_id):
        raise ValueError("Invalid diagnosis identifier.")
    return ARCHIVE_ROOT / f"{diagnosis_id}.json"


def save_diagnosis(diagnosis_id: str, payload: dict[str, Any]) -> Path:
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    target = _path_for(diagnosis_id)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
    return target


def load_diagnosis(diagnosis_id: str) -> dict[str, Any] | None:
    target = _path_for(diagnosis_id)
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))
