from __future__ import annotations

from typing import Any


class MockLLMProvider:
    """Credential-free provider that avoids presenting agricultural claims as grounded facts."""

    async def generate(
        self,
        evidence: dict[str, Any],
        question: str | None = None,
    ) -> dict[str, Any]:
        disease = evidence.get("disease", "unknown condition")
        confidence = float(evidence.get("confidence", 0))
        uncertainty = (
            " The placeholder prediction is below the configured confidence threshold."
            if confidence < 0.65
            else ""
        )
        question_note = (
            " A real provider will answer the follow-up using retrieved sources."
            if question
            else ""
        )
        return {
            "summary": (
                f"Demo only: the mock classifier selected {disease}. This is not a trained-model "
                "diagnosis and no agricultural source corpus has been consulted."
                f"{uncertainty}{question_note}"
            ),
            "symptoms": [],
            "possible_causes": [],
            "prevention": [],
            "management": [],
            "when_to_seek_expert_help": (
                "Use a clear image and seek a qualified agricultural expert for confirmation, "
                "especially before treatment decisions."
            ),
            "sources": [],
            "is_mock": True,
        }
