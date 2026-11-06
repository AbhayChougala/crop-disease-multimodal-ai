"""Project-level context: stored diagnoses as chat evidence, and sessions for saved photos."""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.services.diagnosis_archive import load_diagnosis
from app.services.image_store import StoredImage, image_store
from app.services.session_store import ChatSession, session_store
from crop_ai.agents.state import AgentState

MAX_PROJECT_PHOTOS_IN_PROMPT = 40


def _archived(diagnosis_id: str) -> dict[str, Any] | None:
    try:
        return load_diagnosis(diagnosis_id)
    except (ValueError, OSError):
        return None


def ensure_diagnosis_session(diagnosis_id: str) -> ChatSession | None:
    """The chat session of a diagnosis, recreated from the archive if it expired.

    Saved photos can be weeks old while chat sessions are cleared after CHAT_RETENTION_DAYS; this
    lets the agents and the chat keep working on any photo that is still in the archive.
    """
    session = session_store.get(diagnosis_id)
    if session is not None:
        return session
    payload = _archived(diagnosis_id)
    if payload is None or not payload.get("evidence"):
        return None
    session_store.put(diagnosis_id, payload["evidence"])
    return session_store.get(diagnosis_id)


def agent_findings(diagnosis_id: str) -> dict[str, Any] | None:
    """What the agent team already saved for a photo (Doctor, risk, treatment, follow-up)."""
    raw = session_store.get_agent_state(diagnosis_id)
    if not raw:
        return None
    try:
        state = AgentState.model_validate(raw)
    except ValueError:
        return None
    findings: dict[str, Any] = {}
    if state.diagnosis is not None:
        top = state.diagnosis.candidates[0] if state.diagnosis.candidates else None
        findings["doctor"] = {
            "summary": state.diagnosis.summary,
            "most_likely": f"{top.crop} – {top.disease}" if top else None,
            "likelihood": top.likelihood if top else None,
            "agreement_with_image_model": state.diagnosis.agreement_with_model,
            "is_ambiguous": state.diagnosis.is_ambiguous,
        }
    if state.risk is not None:
        findings["spread_risk"] = {"level": state.risk.level, "score": state.risk.score}
    if state.treatment is not None:
        findings["treatment_summary"] = state.treatment.summary
        findings["treatment_actions"] = [item.action for item in state.treatment.items[:4]]
    if state.follow_up is not None:
        findings["reinspect_in_days"] = state.follow_up.reinspect_in_days
    if state.qa_history:
        findings["farmer_answers"] = state.qa_history[-2:]
    return findings or None


def _label(record: StoredImage) -> str:
    if record.crop and record.disease:
        return f"{record.crop} – {record.disease}"
    return "No confident prediction"


def project_evidence(project_id: str) -> dict[str, Any] | None:
    """Structured evidence for the chatbot: every photo in the project and its stored diagnosis.

    Built fresh on each message, so photos added to the project later are included. Labels are
    the farmer's own free text. Predictions are the image model's, not verified diagnoses.
    """
    project = image_store.project(project_id)
    if project is None:
        return None
    records = image_store.list(limit=1000, project_id=project_id)
    photos = []
    for record in records[:MAX_PROJECT_PHOTOS_IN_PROMPT]:
        severity = (_archived(record.diagnosis_id) or {}).get("severity") or {}
        photos.append(
            {
                "label": record.note or record.filename,
                "predicted_crop": record.crop,
                "predicted_disease": record.disease,
                "confidence": (
                    round(record.confidence, 3) if record.confidence is not None else None
                ),
                "severity_category": severity.get("category"),
                "affected_area_percent": severity.get("affected_area_percent"),
                "analysed_at": record.created_at[:10],
                "is_mock": record.is_mock,
                # Saved results of the agent team for this photo, if any agents were run.
                "agent_findings": agent_findings(record.diagnosis_id),
            }
        )
    counts = Counter(_label(record) for record in records)
    flags = ["project_summary_of_image_model_predictions", "severity_is_experimental"]
    if any(record.is_mock for record in records):
        flags.append("some_predictions_are_demo_placeholders")
    if len(records) > MAX_PROJECT_PHOTOS_IN_PROMPT:
        flags.append(f"only_newest_{MAX_PROJECT_PHOTOS_IN_PROMPT}_photos_listed")
    if not records:
        flags.append("project_has_no_photos_yet")
    return {
        "context_type": "project",
        "project_id": project.project_id,
        "project_name": project.name,
        "project_description": project.description,
        "photo_count": len(records),
        "diagnosis_summary": [{"label": label, "count": n} for label, n in counts.most_common()],
        "photos": photos,
        "visual_evidence": (
            f"Image-model predictions for {len(records)} leaf photo(s) in the farmer's project "
            f"'{project.name}'. Each photo was classified separately; answer about the project "
            "as a whole (which plants are affected, what to prioritise) and refer to plants by "
            "their label. Where agent_findings exist, they are the agent team's saved review of "
            "that photo (the Doctor's differential, spread risk, treatment, follow-up); prefer "
            "them over the raw image-model prediction."
        ),
        "uncertainty_flags": flags,
    }


def project_search_terms(evidence: dict[str, Any]) -> str:
    """Disease names in the project, appended to the retrieval query so fact sheets match."""
    labels = [
        item["label"]
        for item in evidence.get("diagnosis_summary", [])
        if item["label"] != "No confident prediction"
    ]
    return "; ".join(labels[:4])
