from __future__ import annotations

import logging
from typing import Any

from app.core.config import get_settings
from app.services.knowledge_service import get_knowledge_service
from app.services.llm_service import get_llm_provider
from app.services.session_store import ChatSession, session_store
from crop_ai.agents.orchestrator import AgentOrchestrator, TurnResult
from crop_ai.agents.state import AgentState, AgentStep
from crop_ai.agents.tools import fetch_weather
from crop_ai.llm.base import ProviderUnavailable
from crop_ai.llm.mock_provider import MockLLMProvider
from crop_ai.rag.contracts import RetrievedPassage

logger = logging.getLogger(__name__)

REASSESS_MESSAGE = (
    "I took a new photo of the same crop. "
    "Has it improved, and should I continue or change the plan?"
)


def _search(
    query: str, crop: str | None, disease: str | None, top_k: int
) -> list[RetrievedPassage]:
    try:
        return get_knowledge_service().search(query, crop=crop, disease=disease, top_k=top_k)
    except RuntimeError as exc:  # index missing or unreadable: agents degrade gracefully
        logger.warning("Agent retrieval unavailable: %s", exc)
        return []


class AgentService:
    def _orchestrator(self) -> tuple[AgentOrchestrator, Any]:
        settings = get_settings()
        try:
            provider = get_llm_provider()
        except ProviderUnavailable as exc:
            logger.warning("LLM provider unavailable, agents use rule-based fallbacks: %s", exc)
            provider = MockLLMProvider()
        use_llm = not isinstance(provider, MockLLMProvider)
        orchestrator = AgentOrchestrator(
            provider,
            _search,
            fetch_weather if settings.weather_enabled else None,
            use_llm=use_llm,
            max_steps=settings.agent_max_steps,
            max_llm_calls=settings.agent_max_llm_calls,
            orchestrator_model=settings.agent_orchestrator_model or None,
            confidence_threshold=settings.confidence_threshold,
        )
        return orchestrator, provider

    def load_state(self, session: ChatSession) -> AgentState:
        stored = session_store.get_agent_state(session.session_id)
        state = AgentState.model_validate(stored) if stored else AgentState()
        state.evidence = session.evidence
        return state

    async def run_turn(
        self,
        session: ChatSession,
        message: str,
        *,
        location: dict[str, float] | None = None,
        language: str = "en",
        state: AgentState | None = None,
    ) -> tuple[TurnResult, AgentState, Any]:
        orchestrator, provider = self._orchestrator()
        state = state or self.load_state(session)
        result = await orchestrator.run_turn(state, message, location=location, language=language)
        session_store.put_agent_state(session.session_id, state.model_dump(mode="json"))
        return result, state, provider

    async def run_agent(
        self,
        session: ChatSession,
        action: str,
        message: str = "",
        *,
        location: dict[str, float] | None = None,
        language: str = "en",
        rerun: bool = False,
    ) -> tuple[list[AgentStep], AgentState, bool]:
        """Run one agent from the Agents page; returns (steps, state, is_mock)."""
        orchestrator, _ = self._orchestrator()
        state = self.load_state(session)
        steps = await orchestrator.run_agent(
            state, action, message, location=location, language=language, rerun=rerun
        )
        session_store.put_agent_state(session.session_id, state.model_dump(mode="json"))
        return steps, state, not orchestrator.use_llm

    def reset(self, session: ChatSession) -> None:
        """Forget everything the agents worked out for this diagnosis."""
        session_store.put_agent_state(session.session_id, {})

    def prepare_reassessment(
        self, session: ChatSession, new_evidence: dict[str, Any]
    ) -> AgentState:
        """Keep the old evidence for comparison and start a fresh assessment of the new photo."""
        state = self.load_state(session)
        state.previous_evidence = dict(session.evidence)
        session_store.update_evidence(session.session_id, new_evidence)
        session.evidence = new_evidence
        state.evidence = new_evidence
        state.diagnosis = state.risk = state.follow_up = None
        state.pending_questions, state.passages = [], []
        return state


agent_service = AgentService()
