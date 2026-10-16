import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("google.genai")

from google.genai import errors

from crop_ai.llm.base import ProviderUnavailable
from crop_ai.llm.formatting import format_advisory
from crop_ai.llm.gemini_provider import GeminiProvider
from crop_ai.llm.schemas import GroundedAdvisory
from crop_ai.rag.contracts import RetrievedPassage

EVIDENCE = {"predicted_crop": "Tomato", "predicted_disease": "Early Blight"}


def _passage(citation_id: str = "DOC_001-C0001") -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=citation_id,
        document_id="DOC_001",
        title="Early blight fact sheet",
        organization="Test",
        page=None,
        text="Remove infected lower leaves and rotate crops.",
        score=0.8,
        citation_id=citation_id,
    )


def _advisory(source_ids: list[str]) -> GroundedAdvisory:
    return GroundedAdvisory(
        summary="This is likely early blight.",
        management=["Remove infected lower leaves."],
        prevention=["Rotate crops."],
        when_to_seek_expert_help="If it spreads quickly.",
        source_ids=source_ids,
        uncertainty_note="Confirm with an expert.",
    )


class _FakeModels:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    async def generate_content(self, *, model, **_kwargs):
        self.calls += 1
        outcome = self.outcome.get(model) if isinstance(self.outcome, dict) else self.outcome
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(parsed=outcome, usage_metadata=None)


def _provider(
    outcomes: dict[str, object], backend: str = "auto", fallback_models: list[str] | None = None
) -> GeminiProvider:
    provider = GeminiProvider(
        api_key="test-key", model="gemini-test", backend=backend, fallback_models=fallback_models
    )
    provider._clients = {
        name: SimpleNamespace(aio=SimpleNamespace(models=_FakeModels(outcome)))
        for name, outcome in outcomes.items()
    }
    return provider


def _client_error(code: int) -> errors.ClientError:
    return errors.ClientError(code, {"error": {"code": code, "status": "PERMISSION_DENIED"}})


def _server_error(code: int) -> errors.ServerError:
    return errors.ServerError(code, {"error": {"code": code, "status": "UNAVAILABLE"}})


def test_gemini_returns_formatted_grounded_answer() -> None:
    provider = _provider({"developer": _advisory(["DOC_001-C0001"])})
    result = asyncio.run(provider.generate(EVIDENCE, "What should I do?", [_passage()]))
    assert result["is_mock"] is False
    assert "What to do now:\n• Remove infected lower leaves." in result["answer"]
    assert result["source_ids"] == ["DOC_001-C0001"]


def test_gemini_auto_falls_back_to_vertex_when_key_rejected() -> None:
    provider = _provider(
        {"developer": _client_error(403), "vertex": _advisory(["DOC_001-C0001"])}
    )
    result = asyncio.run(provider.generate(EVIDENCE, "What should I do?", [_passage()]))
    assert result["status"] == "available"
    assert provider.backend == "vertex"


def test_gemini_rejects_invented_citations() -> None:
    provider = _provider({"developer": _advisory(["INVENTED-C9"])}, backend="developer")
    with pytest.raises(ProviderUnavailable):
        asyncio.run(provider.generate(EVIDENCE, "What should I do?", [_passage()]))


def test_gemini_never_called_without_retrieval() -> None:
    provider = _provider({"developer": _advisory([])}, backend="developer")
    with pytest.raises(ProviderUnavailable):
        asyncio.run(provider.generate(EVIDENCE, "What should I do?", []))
    assert provider._clients["developer"].aio.models.calls == 0


def test_format_advisory_omits_empty_sections() -> None:
    text = format_advisory(_advisory([]))
    assert "Symptoms" not in text
    assert text.startswith("This is likely early blight.")


def test_gemini_falls_back_to_next_model_when_overloaded(monkeypatch) -> None:
    monkeypatch.setattr("crop_ai.llm.gemini_provider._RETRY_DELAY_SECONDS", 0)
    outcomes = {"gemini-test": _server_error(503), "gemini-backup": _advisory(["DOC_001-C0001"])}
    provider = _provider(
        {"developer": outcomes}, backend="developer", fallback_models=["gemini-backup"]
    )
    result = asyncio.run(provider.generate(EVIDENCE, "What should I do?", [_passage()]))
    assert result["model_used"] == "gemini-backup"
    # one retry on the overloaded model, then one call to the fallback
    assert provider._clients["developer"].aio.models.calls == 3


def test_gemini_reports_every_failed_model(monkeypatch) -> None:
    monkeypatch.setattr("crop_ai.llm.gemini_provider._RETRY_DELAY_SECONDS", 0)
    outcomes = {"gemini-test": _client_error(404), "gemini-backup": _server_error(503)}
    provider = _provider(
        {"developer": outcomes}, backend="developer", fallback_models=["gemini-backup"]
    )
    with pytest.raises(ProviderUnavailable, match="gemini-test.*gemini-backup"):
        asyncio.run(provider.generate(EVIDENCE, "What should I do?", [_passage()]))


def test_out_of_scope_answer_has_no_citations_or_advice() -> None:
    off_topic = GroundedAdvisory(
        in_scope=False,
        summary="I can only help with crop health questions.",
        management=["Install CrewAI with pip."],
        when_to_seek_expert_help="",
        source_ids=["DOC_001-C0001"],
        uncertainty_note="",
    )
    provider = _provider({"developer": off_topic}, backend="developer")
    result = asyncio.run(provider.generate(EVIDENCE, "Write Python code for CrewAI", [_passage()]))
    assert result["status"] == "out_of_scope"
    assert result["answer"] == "I can only help with crop health questions."
    assert result["source_ids"] == [] and result["management"] == []
