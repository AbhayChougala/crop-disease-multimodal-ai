"""Send one real, grounded question through retrieval + the configured LLM provider.

Usage (from backend/, uses the settings in backend/.env):
    python scripts/check_llm.py
    python scripts/check_llm.py "Corn (Maize)" "Common Rust" "Is this serious?"

This makes one billable API call. It prints the provider's answer and citations, or the error.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.services.knowledge_service import get_knowledge_service  # noqa: E402
from app.services.llm_service import get_llm_provider  # noqa: E402
from crop_ai.llm.base import ProviderUnavailable  # noqa: E402


async def main() -> int:
    crop, disease, question = (sys.argv[1:4] + [None, None, None])[:3]
    crop = crop or "Tomato"
    disease = disease or "Early Blight"
    question = question or "What should I do to manage this disease?"
    settings = get_settings()
    print(f"Provider: {settings.llm_provider}  model: {settings.llm_model or '(unset)'}")
    print(f"RAG enabled: {settings.rag_enabled}  retriever: {settings.rag_retriever}")

    passages = get_knowledge_service().search(question, crop=crop, disease=disease)
    print(f"Retrieved {len(passages)} passages: {[item.citation_id for item in passages]}")
    if not passages:
        print("No passages retrieved; build the knowledge base and set RAG_ENABLED=true.")
        return 1

    evidence = {
        "model_mode": "CHECK_SCRIPT",
        "predicted_crop": crop,
        "predicted_disease": disease,
        "visual_evidence": "No image; connectivity check with a stated crop and disease.",
        "uncertainty_flags": ["no_image_connectivity_check"],
    }
    try:
        result = await get_llm_provider().generate(evidence, question, passages)
    except ProviderUnavailable as exc:
        print(f"\nFAILED: {exc}")
        if exc.__cause__ is not None:
            print(f"Cause: {exc.__cause__}")
        return 1
    backend = getattr(get_llm_provider(), "backend", None)
    print(f"\nOK{f' (backend: {backend})' if backend else ''}\n")
    print(result.get("answer") or result.get("summary"))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
