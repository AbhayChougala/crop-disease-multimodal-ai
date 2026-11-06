# Crop Disease Detection and Diagnosis Using Multimodal AI

Instructions for AI coding assistants (Claude, ChatGPT/Codex, Cursor, Copilot, Gemini) and humans.
This repository is an academic B.Tech AI/ML research project. **Read this whole file before changing
anything.** If you are a chat assistant without repository access, ask the user to paste the files
you need instead of guessing their contents.

## Core rules

- Never fabricate model metrics, experiment results, citations, predictions, or research findings.
- Any mock inference must be explicitly marked `is_mock=true` and the UI must visibly show DEMO MODE.
- Never commit secrets, `.env`, datasets, model checkpoints, vector stores, or private source documents.
- Keep all ML code device-agnostic: CUDA > MPS > CPU (use `crop_ai/device.py`).
- Never hard-code `.cuda()`.
- Computer-vision preprocessing used for inference must match the selected training configuration.
- Grad-CAM is explainability, not a disease segmentation ground-truth mask.
- Severity produced from heuristic lesion segmentation must be labelled experimental unless validated against annotated masks.
- LLM agricultural statements must be grounded in retrieved sources where possible.
- The system is decision support, not a replacement for an agricultural expert.
- Keep frontend, backend, training, evaluation, and research code reproducible.
- Run relevant tests, lint, and build checks after changes.
- Do not perform long model training or download large datasets unless explicitly requested.
- Prefer maintainable implementations suitable for a student research project over unnecessary complexity.
- Use current official provider documentation (OpenAI, Google Gemini) when changing LLM integration.

## Where things go

```text
crop-disease-multimodal-ai/
├── backend/            Python project root: pyproject.toml, .env, .venv. Run ALL Python here.
│   ├── app/            FastAPI web layer only (import name `app`, not `backend.app`)
│   │   ├── main.py         app, CORS, router registration
│   │   ├── core/config.py  every setting (read from backend/.env)
│   │   ├── api/routes/     HTTP endpoints
│   │   ├── schemas/        Pydantic request/response models
│   │   └── services/       orchestration that combines crop_ai pieces per request
│   ├── src/crop_ai/    Framework-independent library. Must NOT import FastAPI or `app`.
│   │   ├── cv/ models/ training/ segmentation/ explainability/ evidence/ data/ evaluation/
│   │   ├── rag/            ingest, chunking, TF-IDF / FAISS / hybrid retrieval
│   │   ├── llm/            providers (mock, openai, gemini, ollama), prompts.py, formatting.py
│   │   ├── agents/         multi-agent team: orchestrator, agents, tools, safety, compose
│   │   └── reports/ experiments/ device.py
│   ├── training/       thin CLI wrappers only; logic belongs in crop_ai
│   ├── scripts/        run_backend.sh, run_frontend.sh, check_environment.py, check_llm.py,
│   │                   fetch_knowledge_sources.py, build_knowledge_base.sh, smoke tests
│   ├── tests/          unit/, integration/ (API), ml/ (needs torch), fixtures/, conftest.py
│   ├── configs/        YAML for data, training, RAG, experiments
│   ├── knowledge/      source_manifest.csv (committed) + source_docs/ (git-ignored)
│   ├── data/ artifacts/ research/   generated content, git-ignored
├── frontend/           React + TypeScript + Vite. Run ALL npm commands here.
│   └── src/            pages/, components/, api/client.ts (all backend calls), types/api.ts
├── docs/               guides; docs/project_structure.md has the full per-folder table
├── AGENTS.md           this file
└── CLAUDE.md           imports this file for Claude Code
```

Placement rules:

| Task | Files to touch |
|---|---|
| New API endpoint | `app/api/routes/`, `app/schemas/`, `app/services/`, `frontend/src/api/client.ts`, `frontend/src/types/api.ts`, `tests/integration/` |
| New LLM provider | `src/crop_ai/llm/<name>_provider.py` (return `advisory_payload(...)`), `llm/factory.py`, `app/core/config.py`, `.env.example`, `tests/unit/` |
| Change chatbot behaviour or tone | `src/crop_ai/llm/prompts.py` (ROLE/SCOPE/SAFETY/STYLE), `llm/schemas.py`, `llm/formatting.py` |
| Change the agent team (diagnosis chat) | `src/crop_ai/agents/` (`orchestrator.py` preconditions/fallback, `agents.py`, `prompts.py`, `tools.py`), `tests/unit/test_agent_*.py`; see `docs/agent_architecture.md` |
| New setting | `app/core/config.py` **and** `backend/.env.example` |
| New knowledge source | add to `SOURCES` in `scripts/fetch_knowledge_sources.py`, then run it and `scripts/build_knowledge_base.sh` |
| New frontend page | `frontend/src/pages/`, route in `App.tsx`, link in `components/Navbar.tsx` |
| Model/training code | `src/crop_ai/models/` or `training/`, config in `configs/training/`, tests in `tests/ml/` |

Never put Python files at the repository root or in `frontend/`, never put frontend code in
`backend/`, and never put datasets, checkpoints, API keys or downloaded documents anywhere tracked
by git.

## Commands

```bash
# backend (from backend/), Python 3.12-3.14 only
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev,llm,reports]'        # + rag / semantic-rag / ml when needed
python -m pip install -e '.[ml]' && python model/download_model.py   # real classifier
python scripts/check_environment.py
scripts/run_backend.sh                                # uvicorn on :8000, reloads on app/ src/ only
ruff check . && pytest                                # must pass before finishing

# frontend (from frontend/), Node 20.19+ or 22.12+
npm ci && npm run dev                                 # :5173
npm run build                                         # must pass before finishing

# knowledge base (from backend/, needs '.[semantic-rag]')
python scripts/fetch_knowledge_sources.py             # downloads ~208 extension publications
scripts/build_knowledge_base.sh                       # builds data/vectorstore/
python scripts/check_llm.py                           # one real LLM call through RAG
```

## Current state

- Classifier: EfficientNet-B0 trained on PlantVillage (38 classes), downloaded by
  `python model/download_model.py` to `model/checkpoints/best_model.pth` (git-ignored, SHA-256
  checked), loaded through the legacy adapter in `models/checkpoints.py`. Needs `'.[ml]'` and
  `MODEL_MODE=real`; `MODEL_MODE=mock` gives DEMO placeholders. Its reported validation F1 comes
  from the external training run, not from this project: measure accuracy on held-out data before
  quoting it, and expect lower accuracy on field photos than on lab-style PlantVillage images.
- Diagnosis chat: an LLM orchestrator coordinates an agent team (evidence, diagnosis, clarifying
  questions, risk + Open-Meteo weather, treatment, follow-up/reassess) with preconditions, call/step
  budgets and a rule-based fallback; every reply carries an `agent_trace`. The `/agents` page runs
  any single agent directly (`POST /api/v1/agents/{agent}/run`). See `docs/agent_architecture.md`. Settings: `AGENTS_ENABLED`, `AGENT_MAX_STEPS`,
  `AGENT_MAX_LLM_CALLS`, `AGENT_ORCHESTRATOR_MODEL`, `WEATHER_ENABLED`.
- Photos and projects: `POST /diagnose` saves each upload (bytes, prediction, label, project) in the
  `diagnosis_images` table of the local SQLite database (`CHAT_DATABASE_PATH`, git-ignored) via
  `app/services/image_store.py`. Re-uploading an identical photo (same SHA-256) returns its archived
  diagnosis (`from_saved=true`) if the same model/checkpoint is loaded; `?reanalyse=true` forces a new
  run. The `projects` table groups photos (`/projects` pages); `/history` lists all photos.
  Setting: `STORE_UPLOADED_IMAGES`. Integration tests number each synthetic photo so they stay unique.
- Project context (`app/services/project_context.py`): a general chat created with `project_id` sends
  the project's photos, labels and stored diagnoses as evidence (rebuilt every message) and adds the
  project's diseases to the retrieval query. `/agents?project=<id>` shows a per-photo agents dashboard
  (`GET /agents/projects/{id}/memory`) with "run on all photos". Expired chat sessions are rebuilt
  from the diagnosis archive by `ensure_diagnosis_session`.
- Saved agent results: `run_agent` returns a result already saved for a photo (`reused=true`, no LLM
  call) unless `rerun=true`, a message/answer is sent, or a location is newly shared for risk.
  Re-running the Doctor clears that photo's risk, treatment and follow-up (they depended on it).
  Project chat evidence includes each photo's saved `agent_findings`.
- General chat: retrieve passages, then the LLM returns a structured `GroundedAdvisory`, citation IDs are
  intersected with retrieved IDs, and `format_advisory` renders the text. The LLM explains the
  classifier's result; it never classifies. Real providers are never called when retrieval is empty.
- Scope: the prompt only allows crop/plant-health topics; off-topic answers set `in_scope=false`
  and `advisory_payload` strips all advice and citations (status `out_of_scope`).
- LLM: Gemini via `google-genai` (`LLM_PROVIDER=gemini`, `LLM_MODEL=gemini-3.5-flash-lite`,
  `GEMINI_BACKEND=developer`, `LLM_FALLBACK_MODELS` for 404/429/503). OpenAI, Ollama and mock also
  exist. Google retires model names; check AI Studio if a model returns 404.
- RAG: ~207 extension publications (UC IPM, TNAU India, NC State, Ohio State, UF, WSU, ...) indexed
  into 1,292 chunks, hybrid TF-IDF + FAISS (all-MiniLM-L6-v2). Hybrid scores are min-max
  normalised, so `RAG_MIN_SCORE` cannot filter off-topic queries.

## Gotchas

- CORS origins come from comma-separated `FRONTEND_ORIGIN` in `backend/.env`; Vite port 5173 is fixed.
- `tests/conftest.py` forces mock/offline settings, so tests never read `.env` or call paid APIs.
- Run uvicorn with `--reload-dir app --reload-dir src`; plain `--reload` restarts on every data or
  index write under `backend/` and the UI hangs on "Starting session…".
- Provider `stream()` is not real token streaming; chat uses REST.
- Do not edit a user's `backend/.env` without asking; it holds their API key.

## Further docs

`docs/research_context.md` (full project detail for papers/reports), `docs/project_structure.md` (full folder table), `docs/architecture.md`, `docs/agent_architecture.md`, `docs/rag_guide.md`,
`docs/llm_setup.md`, `docs/research_protocol.md`, `docs/gpu_handoff.md`, `README.md` (setup).
