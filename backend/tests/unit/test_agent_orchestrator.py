import asyncio

import pytest

from crop_ai.agents.orchestrator import AgentOrchestrator
from crop_ai.agents.state import (
    AgentState,
    Candidate,
    DifferentialDiagnosis,
    FarmerQuestions,
    FollowUpPlan,
    OrchestratorDecision,
    TreatmentItem,
    TreatmentPlan,
)
from crop_ai.llm.base import ProviderUnavailable
from crop_ai.rag.contracts import RetrievedPassage

EVIDENCE = {
    "model_mode": "REAL",
    "predicted_crop": "Tomato",
    "predicted_disease": "Early Blight",
    "confidence": 0.82,
    "top_k": [
        {"rank": 1, "crop": "Tomato", "disease": "Early Blight", "confidence": 0.82},
        {"rank": 2, "crop": "Tomato", "disease": "Septoria Leaf Spot", "confidence": 0.10},
    ],
    "severity_percentage": 12.0,
}


def _passage(citation_id: str) -> RetrievedPassage:
    return RetrievedPassage(
        citation_id=citation_id,
        chunk_id=citation_id,
        document_id=citation_id.split("-")[0],
        title="Fact sheet",
        organization="Test Extension",
        page=None,
        score=0.9,
        text="Remove infected leaves. Rotate crops. Mancozeb is labelled for early blight.",
    )


def _search(query, crop, disease, top_k):
    return [_passage("EB-C0001"), _passage("EB-C0002")][:top_k]


def _diagnosis(ambiguous: bool = False) -> DifferentialDiagnosis:
    return DifferentialDiagnosis(
        summary="Concentric rings on older leaves match early blight.",
        agreement_with_model="agrees",
        candidates=[Candidate(crop="Tomato", disease="Early Blight", likelihood="high")],
        is_ambiguous=ambiguous,
        source_ids=["EB-C0001", "INVENTED-C9"],
    )


TREATMENT = TreatmentPlan(
    summary="Act now to slow the spread.",
    items=[
        TreatmentItem(action="Remove infected lower leaves.", category="immediate"),
        TreatmentItem(
            action="Spray mancozeb as labelled.", category="chemical", source_ids=["EB-C0001"]
        ),
        TreatmentItem(action="Spray an unlisted product.", category="chemical"),
    ],
)
FOLLOW_UP = FollowUpPlan(reinspect_in_days=4, what_to_check=["New spots on upper leaves"])


class FakeProvider:
    """Scripted orchestrator decisions and agent outputs; exceptions are raised when scripted."""

    def __init__(self, decisions, outputs, answer=None):
        self.decisions = list(decisions)
        self.outputs = outputs
        self.answer = answer or {
            "answer": "Use clean seed.",
            "source_ids": [],
            "status": "available",
        }
        self.calls: list[str] = []

    async def complete_structured(self, system, user, schema, model=None):
        self.calls.append(schema.__name__)
        if schema is OrchestratorDecision:
            if not self.decisions:
                raise ProviderUnavailable("no scripted decision")
            return OrchestratorDecision(next_action=self.decisions.pop(0), reason="scripted")
        outcome = self.outputs[schema.__name__]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome.model_copy(deep=True)

    async def generate(self, evidence, question, passages, history, language):
        return dict(self.answer)


def _run(orchestrator, state, message="What should I do?", **kwargs):
    return asyncio.run(orchestrator.run_turn(state, message, **kwargs))


def _outputs(**overrides):
    outputs = {
        "DifferentialDiagnosis": _diagnosis(),
        "FarmerQuestions": FarmerQuestions(questions=["Do the spots have rings?"]),
        "TreatmentPlan": TREATMENT,
        "FollowUpPlan": FOLLOW_UP,
    }
    outputs.update(overrides)
    return outputs


def test_confident_case_runs_full_team_and_safety_review() -> None:
    decisions = [
        "retrieve_evidence",
        "diagnose",
        "assess_risk",
        "recommend_treatment",
        "plan_followup",
        "finish",
    ]
    provider = FakeProvider(decisions, _outputs())
    state = AgentState(evidence=EVIDENCE)
    result = _run(AgentOrchestrator(provider, _search), state)

    assert [step.action for step in result.steps] == decisions
    assert all(step.decided_by == "llm" for step in result.steps)
    assert result.status == "available" and not result.is_mock
    # uncited chemical advice and invented citations are removed by the safety reviewer
    assert [item.action for item in state.treatment.items if item.category == "chemical"] == [
        "Spray mancozeb as labelled."
    ]
    assert "INVENTED-C9" not in result.source_ids
    assert "What to do now:" in result.text and "Re-check in 4 days" in result.text


def test_ambiguous_case_asks_questions_then_rediagnoses_next_turn() -> None:
    provider = FakeProvider(
        ["retrieve_evidence", "diagnose", "ask_farmer"],
        _outputs(DifferentialDiagnosis=_diagnosis(True)),
    )
    orchestrator = AgentOrchestrator(provider, _search)
    state = AgentState(evidence=EVIDENCE)
    first = _run(orchestrator, state)
    assert first.ended_with_questions and first.status == "needs_input"
    assert "Do the spots have rings?" in first.text

    provider.decisions = ["retrieve_evidence", "diagnose", "recommend_treatment", "finish"]
    provider.outputs["DifferentialDiagnosis"] = _diagnosis(False)
    second = _run(orchestrator, state, "Yes, rings on the lower leaves")
    assert state.qa_history == [
        {"question": "Do the spots have rings?", "answer": "Yes, rings on the lower leaves"}
    ]
    assert not second.ended_with_questions
    assert "diagnose" in [step.action for step in second.steps]


def test_disallowed_choice_falls_back_to_rule_policy() -> None:
    provider = FakeProvider(["recommend_treatment"], _outputs())
    state = AgentState(evidence=EVIDENCE)
    result = _run(AgentOrchestrator(provider, _search), state)
    first = result.steps[0]
    assert first.action == "retrieve_evidence" and first.decided_by == "rule"
    assert "not allowed" in first.reason


def test_llm_call_budget_is_respected_and_turn_still_completes() -> None:
    decisions = [
        "retrieve_evidence",
        "diagnose",
        "assess_risk",
        "recommend_treatment",
        "plan_followup",
        "finish",
    ]
    provider = FakeProvider(decisions, _outputs())
    state = AgentState(evidence=EVIDENCE)
    result = _run(AgentOrchestrator(provider, _search, max_llm_calls=2), state)
    assert len(provider.calls) == 2
    assert any(step.decided_by == "rule" for step in result.steps)
    assert state.follow_up is not None  # rule-based follow-up still scheduled
    assert result.text


def test_provider_error_mid_turn_uses_rule_based_diagnosis() -> None:
    outputs = _outputs(DifferentialDiagnosis=ProviderUnavailable("HTTP 429"))
    provider = FakeProvider(["retrieve_evidence", "diagnose", "finish"], outputs)
    state = AgentState(evidence=EVIDENCE)
    result = _run(AgentOrchestrator(provider, _search), state)
    diagnose = next(step for step in result.steps if step.action == "diagnose")
    assert diagnose.status == "fallback"
    assert state.diagnosis.candidates[0].disease == "Early Blight"


def test_demo_mode_is_rule_based_and_marked_mock() -> None:
    provider = FakeProvider([], _outputs())
    state = AgentState(evidence={**EVIDENCE, "model_mode": "DEMO"})
    result = _run(AgentOrchestrator(provider, _search, use_llm=False), state)
    assert result.is_mock
    assert provider.calls == []
    assert {step.decided_by for step in result.steps} == {"rule"}
    assert "DEMO placeholder" in result.text


def test_off_topic_follow_up_returns_only_refusal() -> None:
    provider = FakeProvider(
        ["answer_question"],
        _outputs(),
        answer={
            "answer": "I can only help with crop health.",
            "source_ids": ["EB-C0001"],
            "status": "out_of_scope",
        },
    )
    state = AgentState(evidence=EVIDENCE, turn=1, treatment=TREATMENT, passages=[])
    result = _run(AgentOrchestrator(provider, _search), state, "Write me Python code")
    assert result.status == "out_of_scope"
    assert result.text == "I can only help with crop health."
    assert result.source_ids == []


@pytest.mark.parametrize("max_steps", [1, 3])
def test_step_budget_is_respected(max_steps) -> None:
    provider = FakeProvider([], _outputs())
    state = AgentState(evidence=EVIDENCE)
    result = _run(AgentOrchestrator(provider, _search, use_llm=False, max_steps=max_steps), state)
    assert len([step for step in result.steps if step.action != "finish"]) <= max_steps


def test_reassessment_cannot_end_in_qa_and_keeps_photo_comparison() -> None:
    provider = FakeProvider(
        ["answer_question", "retrieve_evidence", "diagnose", "plan_followup", "finish"],
        _outputs(
            FollowUpPlan=FollowUpPlan(
                reinspect_in_days=3,
                what_to_check=["Upper leaves"],
                reassessment="Worse; change plan.",
            )
        ),
    )
    state = AgentState(
        evidence=EVIDENCE, previous_evidence={**EVIDENCE, "severity_percentage": 4.0}, turn=1
    )
    result = _run(AgentOrchestrator(provider, _search), state, "Has it improved?")
    actions = [step.action for step in result.steps]
    assert "answer_question" not in actions
    assert "plan_followup" in actions and "Worse; change plan." in result.text
    assert state.previous_evidence is None  # comparison done, not repeated next turn


def test_single_treatment_agent_runs_missing_prerequisites_only() -> None:
    provider = FakeProvider([], _outputs())
    orchestrator = AgentOrchestrator(provider, _search)
    state = AgentState(evidence=EVIDENCE)
    steps = asyncio.run(orchestrator.run_agent(state, "recommend_treatment"))
    actions = [step.action for step in steps]
    assert actions == ["retrieve_evidence", "diagnose", "recommend_treatment"]
    assert [step.decided_by for step in steps] == ["rule", "rule", "user"]
    assert "OrchestratorDecision" not in provider.calls  # no orchestrator for single agents
    # safety review still applies: the uncited chemical item is dropped
    assert len([i for i in state.treatment.items if i.category == "chemical"]) == 1

    again = asyncio.run(orchestrator.run_agent(state, "assess_risk"))
    assert [step.action for step in again] == ["assess_risk"]


def test_single_agent_records_answers_and_rediagnoses() -> None:
    provider = FakeProvider([], _outputs(DifferentialDiagnosis=_diagnosis(True)))
    orchestrator = AgentOrchestrator(provider, _search)
    state = AgentState(evidence=EVIDENCE)
    asyncio.run(orchestrator.run_agent(state, "ask_farmer"))
    assert state.pending_questions == ["Do the spots have rings?"]

    steps = asyncio.run(orchestrator.run_agent(state, "recommend_treatment", "Yes, rings"))
    assert state.qa_history[-1]["answer"] == "Yes, rings"
    assert [step.action for step in steps] == ["diagnose", "recommend_treatment"]


def test_single_agent_reuses_saved_results_unless_rerun() -> None:
    provider = FakeProvider([], _outputs())
    orchestrator = AgentOrchestrator(provider, _search)
    state = AgentState(evidence=EVIDENCE)
    asyncio.run(orchestrator.run_agent(state, "plan_followup"))
    calls = len(provider.calls)

    saved = asyncio.run(orchestrator.run_agent(state, "plan_followup"))
    assert len(provider.calls) == calls  # nothing was re-run
    assert [step.action for step in saved] == ["plan_followup"]
    assert saved[0].summary.startswith("Saved result: re-check in")
    saved_doctor = asyncio.run(orchestrator.run_agent(state, "diagnose"))
    assert saved_doctor[0].summary.startswith("Saved result: most likely")
    assert len(provider.calls) == calls

    fresh = asyncio.run(orchestrator.run_agent(state, "diagnose", rerun=True))
    assert len(provider.calls) > calls
    assert not fresh[-1].summary.startswith("Saved result")
    # A new diagnosis makes the old follow-up plan stale.
    assert state.follow_up is None


def test_single_agent_rejects_finish() -> None:
    orchestrator = AgentOrchestrator(FakeProvider([], _outputs()), _search)
    with pytest.raises(ValueError):
        asyncio.run(orchestrator.run_agent(AgentState(evidence=EVIDENCE), "finish"))


def test_qa_turn_text_is_passed_through_unchanged() -> None:
    answer = {
        "answer": "Use copper.\n\nSources: EB-C0001",
        "source_ids": ["EB-C0001"],
        "status": "available",
    }
    provider = FakeProvider(["answer_question"], _outputs(), answer=answer)
    state = AgentState(evidence={**EVIDENCE, "model_mode": "DEMO"}, turn=1, treatment=TREATMENT)
    result = _run(AgentOrchestrator(provider, _search), state, "Organic options?")
    assert result.text == "Use copper.\n\nSources: EB-C0001"
    assert result.source_ids == ["EB-C0001"]
