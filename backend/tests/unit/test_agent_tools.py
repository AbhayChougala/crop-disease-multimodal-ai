import asyncio

import httpx

from crop_ai.agents import tools
from crop_ai.agents.agents import rule_based_diagnosis, rule_based_followup
from crop_ai.agents.safety import review
from crop_ai.agents.state import AgentState, TreatmentItem, TreatmentPlan


def test_risk_is_unknown_without_severity_or_weather() -> None:
    risk = tools.assess_risk(AgentState(evidence={}), None)
    assert risk.level == "unknown"


def test_humid_rainy_weather_and_lesions_give_high_risk() -> None:
    state = AgentState(evidence={"severity_percentage": 25.0})
    weather = {
        "mean_humidity_pct": 92,
        "humid_hours_90pct": 30,
        "rain_mm_3d": 12,
        "mean_temp_c": 24,
    }
    risk = tools.assess_risk(state, weather)
    assert risk.level == "high" and risk.score >= 60
    assert risk.weather == weather


def test_fetch_weather_summarises_open_meteo(monkeypatch) -> None:
    payload = {
        "hourly": {
            "relative_humidity_2m": [95, 95, 60, 60],
            "precipitation": [1.0, 2.0, 0.0, 0.0],
            "temperature_2m": [20, 22, 24, 26],
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["latitude"] == "12.9"
        return httpx.Response(200, json=payload)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        tools.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    tools._weather_cache.clear()
    weather = asyncio.run(tools.fetch_weather(12.9, 77.6))
    assert weather == {
        "mean_humidity_pct": 77.5,
        "humid_hours_90pct": 2.0,
        "rain_mm_3d": 3.0,
        "mean_temp_c": 23.0,
    }


def test_fetch_weather_returns_none_on_network_error(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        tools.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    tools._weather_cache.clear()
    assert asyncio.run(tools.fetch_weather(0, 0)) is None


def test_safety_review_drops_uncited_chemicals_and_invented_citations() -> None:
    state = AgentState(
        passages=[{"citation_id": "DOC-C0001"}],
        treatment=TreatmentPlan(
            summary="",
            items=[
                TreatmentItem(action="Prune", category="cultural", source_ids=["FAKE-C1"]),
                TreatmentItem(action="Spray X", category="chemical"),
                TreatmentItem(action="Spray Y", category="chemical", source_ids=["DOC-C0001"]),
            ],
        ),
    )
    warnings = review(state)
    assert [item.action for item in state.treatment.items] == ["Prune", "Spray Y"]
    assert state.treatment.items[0].source_ids == []
    assert any("uncited chemical" in item for item in warnings)


def test_rule_based_diagnosis_flags_small_margin_as_ambiguous() -> None:
    evidence = {
        "confidence": 0.5,
        "top_k": [
            {"crop": "Tomato", "disease": "Early Blight", "confidence": 0.5},
            {"crop": "Tomato", "disease": "Late Blight", "confidence": 0.4},
        ],
    }
    diagnosis = rule_based_diagnosis(AgentState(evidence=evidence), confidence_threshold=0.65)
    assert diagnosis.is_ambiguous
    assert [item.disease for item in diagnosis.candidates] == ["Early Blight", "Late Blight"]


def test_rule_based_followup_compares_photos() -> None:
    state = AgentState(
        evidence={"predicted_disease": "Early Blight", "severity_percentage": 8.0},
        previous_evidence={"predicted_disease": "Early Blight", "severity_percentage": 20.0},
    )
    plan = rule_based_followup(state)
    assert "20% to 8%" in plan.reassessment and "continue the plan" in plan.reassessment


def test_risk_message_distinguishes_missing_location_from_weather_outage() -> None:
    with_location = AgentState(evidence={"severity_percentage": 5}, location={"latitude": 1.0})
    without_location = AgentState(evidence={"severity_percentage": 5})
    assert "did not respond" in " ".join(tools.assess_risk(with_location, None).factors)
    assert "No location shared" in " ".join(tools.assess_risk(without_location, None).factors)
