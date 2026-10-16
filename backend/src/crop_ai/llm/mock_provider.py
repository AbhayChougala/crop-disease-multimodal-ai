from __future__ import annotations

from typing import Any

from crop_ai.llm.base import ProviderCapabilities, ProviderStatus
from crop_ai.rag.contracts import RetrievedPassage


class MockLLMProvider:
    """Credential-free provider that avoids presenting agricultural claims as grounded facts."""

    model = "deterministic-demo"

    async def generate(
        self,
        evidence: dict[str, Any],
        question: str | None = None,
        passages: list[RetrievedPassage] | None = None,
        history: list[dict[str, str]] | None = None,
        preferred_language: str = "en",
    ) -> dict[str, Any]:
        del history, preferred_language
        disease = evidence.get("predicted_disease") or evidence.get("disease", "unknown condition")
        confidence = float(evidence.get("confidence", 0))
        uncertainty = (
            " The placeholder prediction is below the configured confidence threshold."
            if confidence < 0.65
            else ""
        )
        retrieved = passages or []
        source_ids = [item.citation_id for item in retrieved]
        grounded_note = (
            f" {len(retrieved)} test or approved knowledge passage(s) were retrieved; "
            "source IDs are "
            "listed for interface testing, but this deterministic provider does not create new "
            "agricultural claims."
            if retrieved
            else " No verified source passage was available, so grounded guidance is unavailable."
        )
        question_note = f" The recorded question was: {question}" if question else ""
        answer = (
            f"DEMO / MOCK RESPONSE: the configured demo workflow references {disease}. "
            "This is not a real agricultural diagnosis."
            f"{uncertainty}{grounded_note}{question_note}"
        )
        return {
            "answer": answer,
            "summary": answer,
            "answer_type": "demo_grounded_interface" if retrieved else "insufficient_evidence",
            "confidence_language": "Demo output; no reliability claim is made.",
            "source_ids": source_ids,
            "unsupported_claims_detected": False,
            "insufficient_evidence": not retrieved,
            "follow_up_suggestions": ["Ask what evidence is missing.", "Review the source list."],
            "safety_notice": "Verify crop-health decisions with a qualified agricultural expert.",
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
            "status": "demo",
        }

    async def stream(
        self, evidence, question, passages=None, history=None, preferred_language="en"
    ):
        response = await self.generate(evidence, question, passages, history, preferred_language)
        yield response["answer"]

    async def health(self) -> ProviderStatus:
        return ProviderStatus("mock", self.model, "DEMO", "Deterministic offline provider")

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities("mock", self.model, True, True, None)
