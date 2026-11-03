"""Deterministic tools used by the agent team: evidence retrieval, weather and spread risk."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any

import httpx

from crop_ai.agents.state import AgentState, RiskAssessment
from crop_ai.rag.contracts import RetrievedPassage

# search(query, crop, disease, top_k) -> passages; the app passes KnowledgeService.search.
SearchFn = Callable[[str, str | None, str | None, int], list[RetrievedPassage]]
WeatherFn = Callable[[float, float], Awaitable[dict[str, float] | None]]

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_TIMEOUT_SECONDS = 8.0
WEATHER_CACHE_SECONDS = 3600
_weather_cache: dict[tuple[float, float], tuple[float, dict[str, float]]] = {}
MAX_PASSAGES = 9


def candidate_diseases(state: AgentState, limit: int = 3) -> list[tuple[str | None, str]]:
    """Diseases to research: the diagnosis agent's candidates, else the image model's top-k."""
    pairs: list[tuple[str | None, str]] = []
    if state.diagnosis:
        pairs = [(item.crop, item.disease) for item in state.diagnosis.candidates]
    if not pairs:
        for item in state.evidence.get("top_k") or []:
            if item.get("disease"):
                pairs.append((item.get("crop"), item["disease"]))
    if not pairs and state.evidence.get("predicted_disease"):
        pairs.append((state.evidence.get("predicted_crop"), state.evidence["predicted_disease"]))
    unique = list(dict.fromkeys(pairs))
    return unique[:limit]


def retrieve_for_candidates(search: SearchFn, state: AgentState, message: str) -> int:
    """Search per candidate disease, de-duplicate, and store passages on the state."""
    found: dict[str, RetrievedPassage] = {}
    pairs = candidate_diseases(state) or [(None, "")]
    per_candidate = max(2, MAX_PASSAGES // len(pairs))
    for crop, disease in pairs:
        query = f"{disease} symptoms management control {message}".strip()
        for passage in search(query, crop, disease or None, per_candidate):
            found.setdefault(passage.citation_id, passage)
    state.passages = [asdict(item) for item in list(found.values())[:MAX_PASSAGES]]
    return len(state.passages)


def passages_of(state: AgentState) -> list[RetrievedPassage]:
    return [RetrievedPassage(**item) for item in state.passages]


async def fetch_weather(latitude: float, longitude: float) -> dict[str, float] | None:
    """Next-3-day humidity, rain and temperature from Open-Meteo (free, no API key).

    Successful forecasts are cached per ~1 km cell for an hour, so later turns do not wait again.
    """
    key = (round(latitude, 2), round(longitude, 2))
    cached = _weather_cache.get(key)
    if cached and time.monotonic() - cached[0] < WEATHER_CACHE_SECONDS:
        return cached[1]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "relative_humidity_2m,precipitation,temperature_2m",
        "forecast_days": 3,
        "timezone": "auto",
    }
    try:
        async with httpx.AsyncClient(timeout=WEATHER_TIMEOUT_SECONDS) as client:
            response = await client.get(OPEN_METEO_URL, params=params)
            response.raise_for_status()
        hourly = response.json()["hourly"]
    except (httpx.HTTPError, KeyError, ValueError):
        return None
    humidity = [value for value in hourly.get("relative_humidity_2m", []) if value is not None]
    rain = [value for value in hourly.get("precipitation", []) if value is not None]
    temperature = [value for value in hourly.get("temperature_2m", []) if value is not None]
    if not humidity or not temperature:
        return None
    summary = {
        "mean_humidity_pct": round(sum(humidity) / len(humidity), 1),
        "humid_hours_90pct": float(sum(1 for value in humidity if value >= 90)),
        "rain_mm_3d": round(sum(rain), 1),
        "mean_temp_c": round(sum(temperature) / len(temperature), 1),
    }
    _weather_cache[key] = (time.monotonic(), summary)
    return summary


def assess_risk(state: AgentState, weather: dict[str, float] | None) -> RiskAssessment:
    """Transparent, experimental spread-risk score. Not validated against field data."""
    score = 0
    factors: list[str] = []
    severity = state.evidence.get("severity_percentage")
    if isinstance(severity, int | float):
        if severity >= 20:
            score += 35
            factors.append(f"High visible lesion area ({severity:.0f}%, experimental estimate)")
        elif severity >= 5:
            score += 20
            factors.append(f"Moderate visible lesion area ({severity:.0f}%, experimental)")
        elif severity > 0:
            score += 10
            factors.append(f"Small visible lesion area ({severity:.0f}%, experimental)")
    if state.diagnosis and state.diagnosis.candidates:
        top = state.diagnosis.candidates[0]
        if "healthy" in top.disease.lower():
            score = max(0, score - 20)
            factors.append("Most likely candidate is healthy")
        elif top.likelihood == "high":
            score += 10
            factors.append(f"Disease likely: {top.disease}")
    if weather:
        if weather.get("humid_hours_90pct", 0) >= 12:
            score += 25
            factors.append(
                f"{weather['humid_hours_90pct']:.0f} very humid hours forecast (favours fungi)"
            )
        if weather.get("rain_mm_3d", 0) >= 5:
            score += 15
            factors.append(
                f"{weather['rain_mm_3d']:.0f} mm rain forecast in 3 days (splash spread)"
            )
        if 18 <= weather.get("mean_temp_c", 0) <= 30:
            score += 10
            factors.append(f"Mild temperatures (~{weather['mean_temp_c']:.0f} °C)")
    elif state.location:
        factors.append("Weather service did not respond, so weather was not considered")
    else:
        factors.append("No location shared, so weather was not considered")
    if not isinstance(severity, int | float) and not weather:
        level = "unknown"
    elif score >= 60:
        level = "high"
    elif score >= 30:
        level = "moderate"
    else:
        level = "low"
    return RiskAssessment(level=level, score=min(score, 100), factors=factors, weather=weather)


def reinspect_days(risk: RiskAssessment | None) -> int:
    return {"high": 2, "moderate": 4, "low": 7}.get(risk.level if risk else "unknown", 5)


def summarize_weather(weather: dict[str, Any] | None) -> str:
    if not weather:
        return "no weather data"
    return (
        f"humidity {weather['mean_humidity_pct']:.0f}%, rain {weather['rain_mm_3d']:.0f} mm, "
        f"{weather['mean_temp_c']:.0f} °C"
    )
