"""Deterministic safety review applied to agent output before it reaches the farmer."""

from __future__ import annotations

from crop_ai.agents.state import AgentState

MAX_TREATMENT_ITEMS = 7


def review(state: AgentState) -> list[str]:
    """Keep only citations that were actually retrieved; drop uncited chemical advice.

    Returns human-readable warnings describing what was removed.
    """
    allowed = {item["citation_id"] for item in state.passages}
    warnings: list[str] = []

    if state.diagnosis:
        invented = set(state.diagnosis.source_ids) - allowed
        if invented:
            warnings.append("Removed citation IDs the diagnosis agent did not retrieve.")
        state.diagnosis.source_ids = [
            item for item in state.diagnosis.source_ids if item in allowed
        ]
        state.diagnosis.candidates = state.diagnosis.candidates[:3]

    if state.treatment:
        kept = []
        for item in state.treatment.items:
            valid = [source for source in item.source_ids if source in allowed]
            if len(valid) != len(item.source_ids):
                warnings.append("Removed citation IDs the treatment agent did not retrieve.")
            if item.category == "chemical" and not valid:
                warnings.append(f"Dropped uncited chemical advice: {item.action[:60]}")
                continue
            item.source_ids = valid
            kept.append(item)
        state.treatment.items = kept[:MAX_TREATMENT_ITEMS]
        state.treatment.safety_notes = state.treatment.safety_notes[:2]

    if state.follow_up:
        state.follow_up.what_to_check = state.follow_up.what_to_check[:3]
        state.follow_up.escalate_if = state.follow_up.escalate_if[:2]
    return list(dict.fromkeys(warnings))
