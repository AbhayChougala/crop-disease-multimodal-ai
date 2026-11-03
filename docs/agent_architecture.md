# Agent architecture: the CropAI "crop doctor" team

The diagnosis-aware chat is answered by a small team of agents coordinated by an **LLM
orchestrator**. General chat (no image) keeps the single-call RAG path.

```text
farmer message (+ diagnosis evidence, earlier answers, optional location)
        │
        ▼
 Orchestrator LLM ── picks ONE allowed action per step (max AGENT_MAX_STEPS) ──────────────┐
        │                                                                                   │
  retrieve_evidence   diagnose          ask_farmer        assess_risk       recommend_       plan_followup
  (tool: RAG per      (LLM: compare     (LLM: 2-3 Qs;     (tool: rules +    treatment        (LLM: re-check date,
  candidate disease)  ML top-k + facts) ends the turn)    Open-Meteo)       (LLM, cited)     photo comparison)
        │                                                                                   │
        └──────────────── results stored in AgentState (persisted per chat session) ───────┘
        ▼ finish                                         answer_question (LLM Q&A turn, scoped)
 Safety reviewer (rules): drop invented citations and uncited chemical advice
        ▼
 Composer (rules): one reply + agent trace + questions + risk + follow-up card
```

## Agents and tools

| Action | Kind | What it does | Code |
|---|---|---|---|
| `retrieve_evidence` | tool | Searches the knowledge base for each candidate disease (image model top-k, or the diagnosis agent's candidates) and de-duplicates passages | `tools.retrieve_for_candidates` |
| `diagnose` | LLM agent | Differential diagnosis: compares the image model's top-k with fact sheets and the farmer's answers, states agreement with the model and whether the case is ambiguous | `agents.run_diagnosis` |
| `ask_farmer` | LLM agent | 2-3 questions that separate the candidates; the farmer's next message is treated as the answers | `agents.run_questions` |
| `assess_risk` | tool | Experimental rule-based spread risk from lesion area, likelihood and a 3-day Open-Meteo forecast (humidity, rain, temperature) | `tools.assess_risk`, `tools.fetch_weather` |
| `recommend_treatment` | LLM agent | Immediate, cultural, organic and chemical actions; chemical items must cite a retrieved passage | `agents.run_treatment` |
| `plan_followup` | LLM agent | Re-inspection interval, what to check, when to escalate; after a new photo, compares old and new evidence and says continue or change the plan | `agents.run_followup` |
| `answer_question` | LLM agent | Specific follow-up questions (or refusing off-topic ones) through the scoped `GroundedAdvisory` path; a turn of its own | `provider.generate` |

All code lives in `backend/src/crop_ai/agents/`. The app wiring is `backend/app/services/agent_service.py`
and `chat_service.py`. The endpoints are the usual chat endpoints plus
`POST /api/v1/chat/sessions/{id}/reassess`.

## Guardrails

- **Preconditions:** the orchestrator only sees actions that make sense now. For example, treatment
  needs a diagnosis and evidence, questions need an ambiguous diagnosis (at most 2 rounds), and
  `finish` is offered only after treatment or follow-up.
- **Budgets:** `AGENT_MAX_STEPS` (default 7) actions and `AGENT_MAX_LLM_CALLS` (default 10)
  structured calls per turn. The orchestrator can run on a cheaper model via
  `AGENT_ORCHESTRATOR_MODEL`.
- **Deterministic fallback:** an invalid choice, a provider error, a 429/503 or an exhausted budget
  switches to a rule-based policy, and diagnosis and follow-up have rule-based versions. A turn
  therefore always completes, and the trace marks each step `llm`/`rule` and
  `ok`/`fallback`/`skipped`/`error`.
- **Safety review:** citations are intersected with retrieved IDs, uncited chemical advice is
  dropped, and lists are capped. The scope rule (`in_scope`) and DEMO labelling still apply.
- **Demo mode:** with the mock provider every step is rule-based, no treatment advice is generated,
  and responses are `is_mock=true`.

## UI

On the Ask page in diagnosis mode, and in the diagnose page's chat panel, each reply shows:
- clarifying questions (tap to answer)
- a risk badge and the re-check date
- a collapsible "How CropAI worked this out" trace

"Share location for weather" sends coordinates rounded to about 1 km. "Reassess with new photo"
runs a new diagnosis and links it to the chat.

## Agents page (run one agent at a time)

`/agents` in the frontend analyses a photo once and then lets you run any single agent, without
the orchestrator. Missing prerequisites run first and are marked "ran automatically" (for example,
Treatment on a fresh photo runs evidence → diagnosis → treatment). Results are stored in the same
per-diagnosis `AgentState` as the chat, so agents build on each other, and a message sent while the
Question agent's questions are pending is recorded as the answers and triggers a re-diagnosis.

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/agents/{agent}/run` | `{session_id, message?, location?}`; `agent` is any action except `finish`. Returns the steps, the agent's structured `output`, cited sources and the team memory |
| `GET /api/v1/agents/memory/{session_id}` | What the team already knows for this diagnosis |
| `DELETE /api/v1/agents/memory/{session_id}` | Forget it and start over |

Code: `AgentOrchestrator.run_agent` / `prerequisites` in `orchestrator.py`,
`app/api/routes/agents.py`, `frontend/src/pages/AgentsPage.tsx`, `components/AgentOutput.tsx`.

## Research use

Each assistant message stores the trace (`agent_trace`, `llm_calls`, timings) in the chat
database. That supports comparing the single-call RAG baseline with the agent team on the same
benchmark. Useful measures: citation validity, abstention or question rate on ambiguous cases,
latency and LLM calls per turn. Report only measured results. The risk score is an unvalidated
heuristic.

## Limits

- A full consultation makes about 5-10 LLM calls (about 15-30 s on the Gemini free tier). Free-tier
  rate limits can push steps to the rule-based fallback.
- The classifier is still a DEMO placeholder until a checkpoint is trained, so diagnoses compare
  against placeholder predictions.
- The weather service can be slow. Forecasts are cached for an hour per ~1 km cell.
