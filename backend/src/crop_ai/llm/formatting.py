from __future__ import annotations

from typing import Any

from crop_ai.llm.schemas import GroundedAdvisory

# Hard cap so a verbose model cannot flood the chat even if it ignores the style prompt.
MAX_ITEMS_PER_SECTION = 4


def format_advisory(advisory: GroundedAdvisory) -> str:
    """Render a structured advisory as readable plain text for the chat UI."""
    if not advisory.in_scope:
        return advisory.summary.strip()
    sections: list[str] = [advisory.summary.strip()]
    for heading, items in (
        ("Symptoms", advisory.symptoms),
        ("Possible causes", advisory.possible_causes),
        ("What to do now", advisory.management),
        ("Prevention", advisory.prevention),
    ):
        cleaned = [item.strip() for item in items if item.strip()][:MAX_ITEMS_PER_SECTION]
        if cleaned:
            sections.append(heading + ":\n" + "\n".join(f"• {item}" for item in cleaned))
    if advisory.when_to_seek_expert_help.strip():
        sections.append("When to get expert help:\n" + advisory.when_to_seek_expert_help.strip())
    if advisory.uncertainty_note.strip():
        sections.append("Note: " + advisory.uncertainty_note.strip())
    if advisory.source_ids:
        sections.append("Sources: " + ", ".join(advisory.source_ids))
    return "\n\n".join(sections)


def advisory_payload(advisory: GroundedAdvisory) -> dict[str, Any]:
    """Provider response fields shared by all real providers.

    Out-of-scope answers are enforced here, not only in the prompt: they carry no citations and
    are reported with status "out_of_scope".
    """
    payload = {**advisory.model_dump(), "answer": format_advisory(advisory), "is_mock": False}
    if not advisory.in_scope:
        payload.update(
            source_ids=[],
            symptoms=[],
            possible_causes=[],
            prevention=[],
            management=[],
            follow_up_suggestions=[],
            insufficient_evidence=True,
            status="out_of_scope",
        )
    else:
        payload["status"] = "available"
    return payload
