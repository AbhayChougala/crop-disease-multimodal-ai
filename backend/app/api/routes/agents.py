"""Run a single agent from the agent team (the Agents page), instead of the whole consultation."""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Response, status

from app.schemas.agents import AgentMemory, AgentRunRequest, AgentRunResponse
from app.schemas.chat import AgentStepOut
from app.services.agent_service import agent_service
from app.services.chat_service import _citation
from app.services.session_store import ChatSession, session_store
from crop_ai.agents.state import AgentState, DirectAgent
from crop_ai.agents.tools import passages_of

router = APIRouter(tags=["agents"])


def _diagnosis_session(session_id: str) -> ChatSession:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Diagnosis was not found or expired.")
    if session.mode != "diagnosis" or not session.evidence:
        raise HTTPException(
            status_code=409, detail="Agents need a diagnosis: upload a leaf photo first."
        )
    return session


def _output(action: str, state: AgentState) -> dict[str, Any] | None:
    if action == "retrieve_evidence":
        return {
            "passages": [
                {
                    "citation_id": item["citation_id"],
                    "title": item["title"],
                    "organization": item.get("organization"),
                    "score": round(float(item["score"]), 3),
                    "excerpt": str(item.get("text", ""))[:300],
                }
                for item in state.passages
            ]
        }
    if action == "ask_farmer":
        return {"questions": state.pending_questions}
    if action == "plan_followup" and state.follow_up:
        due = date.today() + timedelta(days=state.follow_up.reinspect_in_days)
        return {**state.follow_up.model_dump(), "due_date": due.isoformat()}
    result = {
        "diagnose": state.diagnosis,
        "assess_risk": state.risk,
        "recommend_treatment": state.treatment,
        "answer_question": state.answer,
    }.get(action)
    return result.model_dump() if hasattr(result, "model_dump") else result


def _source_ids(value: Any) -> list[str]:
    """All `source_ids` mentioned anywhere in an agent's output, in order."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found += list(item) if key == "source_ids" else _source_ids(item)
    elif isinstance(value, list):
        for item in value:
            found += _source_ids(item)
    return list(dict.fromkeys(found))


def _memory(state: AgentState) -> AgentMemory:
    return AgentMemory(
        passages=len(state.passages),
        has_diagnosis=state.diagnosis is not None,
        has_risk=state.risk is not None,
        has_treatment=state.treatment is not None,
        has_follow_up=state.follow_up is not None,
        pending_questions=state.pending_questions,
        answered_questions=len(state.qa_history),
    )


@router.post("/agents/{agent}/run", response_model=AgentRunResponse)
async def run_agent(agent: DirectAgent, request: AgentRunRequest) -> AgentRunResponse:
    session = _diagnosis_session(request.session_id)
    if agent == "answer_question" and not request.message.strip():
        raise HTTPException(status_code=422, detail="Type a question for the Q&A agent.")
    started = time.perf_counter()
    steps, state, is_mock = await agent_service.run_agent(
        session,
        agent,
        request.message.strip(),
        location=request.location.model_dump() if request.location else None,
        language=request.preferred_language or session.preferred_language,
    )
    output = _output(agent, state)
    by_id = {item.citation_id: item for item in passages_of(state)}
    cited = [] if agent == "retrieve_evidence" else _source_ids(output)
    return AgentRunResponse(
        session_id=session.session_id,
        agent=agent,
        steps=[AgentStepOut(**step.model_dump()) for step in steps],
        output=output,
        sources=[_citation(by_id[item]) for item in cited if item in by_id],
        warnings=state.warnings,
        memory=_memory(state),
        is_mock=is_mock,
        llm_calls=state.llm_calls,
        total_ms=round((time.perf_counter() - started) * 1000, 1),
    )


@router.get("/agents/memory/{session_id}", response_model=AgentMemory)
async def agent_memory(session_id: str) -> AgentMemory:
    return _memory(agent_service.load_state(_diagnosis_session(session_id)))


@router.delete("/agents/memory/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def reset_agent_memory(session_id: str) -> Response:
    agent_service.reset(_diagnosis_session(session_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
