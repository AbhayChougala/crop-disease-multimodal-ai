"""Assemble the team's results into one plain-text chat reply (no extra LLM call)."""

from __future__ import annotations

from datetime import date, timedelta

from crop_ai.agents.state import AgentState

_CATEGORY_HEADINGS = (
    ("immediate", "What to do now"),
    ("cultural", "Field practices"),
    ("organic", "Organic options"),
    ("chemical", "Chemical options (follow the label and local regulations)"),
)


def _bullets(items: list[str]) -> str:
    return "\n".join(f"• {item}" for item in items if item.strip())


def compose_reply(state: AgentState, *, ended_with_questions: bool, today: date | None = None):
    """Return (text, cited source IDs) for the current turn."""
    sections: list[str] = []
    cited: list[str] = []

    if state.answer is not None:
        # Q&A turns: the advisory text is already complete (sources, notes); pass it through.
        text = str(state.answer.get("answer") or state.answer.get("summary") or "")
        if state.answer.get("status") == "out_of_scope":
            return text, []
        return text, list(dict.fromkeys(state.answer.get("source_ids") or []))

    if state.diagnosis:
        top = state.diagnosis.candidates[0] if state.diagnosis.candidates else None
        lines = [state.diagnosis.summary.strip()]
        if top:
            lines.insert(
                0, f"Most likely: {top.crop} - {top.disease} ({top.likelihood} likelihood)"
            )
        others = [f"{item.disease} ({item.likelihood})" for item in state.diagnosis.candidates[1:]]
        if others:
            lines.append("Also possible: " + ", ".join(others))
        sections.append("\n".join(lines))
        cited += state.diagnosis.source_ids

    if ended_with_questions and state.pending_questions:
        numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(state.pending_questions, 1))
        sections.append(f"To narrow this down, please tell me:\n{numbered}")
        return "\n\n".join(item for item in sections if item), list(dict.fromkeys(cited))

    if state.risk:
        factors = _bullets(state.risk.factors[:3])
        sections.append(
            f"Spread risk: {state.risk.level} ({state.risk.method})"
            + (f"\n{factors}" if factors else "")
        )

    if state.treatment:
        if state.treatment.summary.strip():
            sections.append(state.treatment.summary.strip())
        for category, heading in _CATEGORY_HEADINGS:
            items = [item for item in state.treatment.items if item.category == category]
            if items:
                sections.append(f"{heading}:\n" + _bullets([item.action for item in items]))
                for item in items:
                    cited += item.source_ids
        if state.treatment.safety_notes:
            sections.append("Safety:\n" + _bullets(state.treatment.safety_notes))

    if state.follow_up:
        due = (today or date.today()) + timedelta(days=state.follow_up.reinspect_in_days)
        lines = [f"Re-check in {state.follow_up.reinspect_in_days} days ({due:%d %b %Y})."]
        if state.follow_up.reassessment:
            lines.insert(0, state.follow_up.reassessment)
        if state.follow_up.what_to_check:
            lines.append("Look for:\n" + _bullets(state.follow_up.what_to_check))
        if state.follow_up.escalate_if:
            lines.append("Call an expert if:\n" + _bullets(state.follow_up.escalate_if))
        sections.append("Follow-up:\n" + "\n".join(lines))

    if state.evidence.get("model_mode") == "DEMO":
        sections.append(
            "Note: the image prediction is a DEMO placeholder until the classifier is trained."
        )
    cited = list(dict.fromkeys(cited))
    if cited:
        sections.append("Sources: " + ", ".join(cited))
    return "\n\n".join(item for item in sections if item), cited
